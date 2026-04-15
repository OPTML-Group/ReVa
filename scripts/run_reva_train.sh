#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "=== 多组层对齐一键训练 - down_proj微调版本 ==="

# 基础配置
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1"}
export CUDA_VISIBLE_DEVICES

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/checkpoints/example_rmu_model}"
ALL_LAYERS_VEC="${ALL_LAYERS_VEC:-$PROJECT_DIR/files/results/example_refusal_vectors/refusal_state_all_layers.pt}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_DIR/files/results/example_reva}"

if [ ! -d "$MODEL_PATH" ]; then
  echo "❌ 模型路径不存在: $MODEL_PATH"; exit 1
fi
if [ ! -f "$ALL_LAYERS_VEC" ]; then
  echo "❌ 未找到每层拒答向量: $ALL_LAYERS_VEC"; exit 1
fi

# 动态探测可用层数，并生成安全分组
NUM_LAYERS=$(python3 - "$ALL_LAYERS_VEC" <<'PY'
import torch, sys
path = sys.argv[1]
X = torch.load(path, map_location='cpu')
print(X.shape[0])
PY
)

if [ -z "$NUM_LAYERS" ]; then
  echo "❌ 无法读取每层向量的层数"; exit 1
fi
echo "检测到张量层数: $NUM_LAYERS"

# 默认候选起始层（15,18,21,24），过滤越界
declare -a STARTS=(18)
unset GROUP_SPECS || true
declare -a GROUP_SPECS=()
for S in "${STARTS[@]}"; do
  if (( S < NUM_LAYERS )); then
    P1=$((S-2)); P2=$((S-1)); P3=$S
    TL=()
    for L in $P1 $P2 $P3; do
      if (( L >= 0 )); then TL+=("$L"); fi
    done
    GROUP_SPECS+=("${S}:$(IFS=,; echo "${TL[*]}")")
  fi
done

if (( ${#GROUP_SPECS[@]} == 0 )); then
  echo "❌ 无有效分组（NUM_LAYERS=$NUM_LAYERS）"; exit 1
fi

# 通用训练参数（可按需调整）
MAX_NUM_BATCHES=150
BATCH_SIZE=4
RETAIN="wikitext,wikitext"
FORGET="bio-forget-corpus"
STEER="1"
ALPHA="1200"
LR=5e-5
SEED=42

run_one() {
  local LAYER_ID="$1"          # 如 7
  local TRAIN_LAYERS="$2"      # 如 5,6,7
  local DEVICES="$3"           # 如 0,1
  local OUT_DIR="$OUT_ROOT/zephyr_rmu_refusal_L${LAYER_ID}_train_${TRAIN_LAYERS//,/}_down_proj_matrix" # 路径中去掉逗号，添加v_matrix标识

  echo ""
  echo "==> 开始组: layer_id=$LAYER_ID, layer_ids=$TRAIN_LAYERS, gpus=$DEVICES"
  echo "==> 微调参数: down_proj (param_ids=6)"

  # 越界保护
  if (( LAYER_ID < 0 || LAYER_ID >= NUM_LAYERS )); then
    echo "跳过：layer_id=$LAYER_ID 超出范围 [0, $((NUM_LAYERS-1))]"
    return 0
  fi

  # 切出对应层的 [1,1,H] 向量到临时文件
  local TMP_VEC="/tmp/refusal_vec_L${LAYER_ID}.pt"
  python3 - "$ALL_LAYERS_VEC" "$TMP_VEC" "$LAYER_ID" <<'PY'
import sys, torch
all_path, out_path, layer_str = sys.argv[1:4]
L = int(layer_str)
X = torch.load(all_path, map_location='cpu')  # 形状 [num_layers, hidden]
if X.dim()!=2:
    raise SystemExit(f"bad shape {tuple(X.shape)}; expect [num_layers, hidden]")
if not (0 <= L < X.shape[0]):
    raise SystemExit(f"index {L} out of bounds for num_layers={X.shape[0]}")
vec = X[L].unsqueeze(0).unsqueeze(0)  # [1,1,H]
torch.save(vec.half(), out_path)
print(f"saved {out_path} shape={tuple(vec.shape)}")
PY

  echo "运行训练，输出目录: $OUT_DIR (日志: $OUT_DIR/train.log)"
  mkdir -p "$OUT_DIR"
  CUDA_VISIBLE_DEVICES="$DEVICES" \
  python3 "$PROJECT_DIR/src/train/reva/unlearn.py" \
    --model_name_or_path "$MODEL_PATH" \
    --max_num_batches "$MAX_NUM_BATCHES" \
    --batch_size "$BATCH_SIZE" \
    --retain_corpora "$RETAIN" \
    --forget_corpora "$FORGET" \
    --steering_coeffs "$STEER" \
    --alpha "$ALPHA" \
    --lr "$LR" \
    --seed "$SEED" \
    --output_dir "$OUT_DIR" \
    --verbose \
    --refusal_vector_path "$TMP_VEC" \
    --layer_id "$LAYER_ID" \
    --layer_ids "$TRAIN_LAYERS" \
    --param_ids "6" \
    > "$OUT_DIR/train.log" 2>&1
}

# 串行执行训练任务，每次使用相同的2张GPU
DEVICES="$CUDA_VISIBLE_DEVICES"
total_groups=${#GROUP_SPECS[@]}

echo "准备串行训练 $total_groups 个模型组，每次使用GPU: $DEVICES"
echo "微调参数: down_proj"

for i in "${!GROUP_SPECS[@]}"; do
  g="${GROUP_SPECS[i]}"
  LID="${g%%:*}"
  TL="${g##*:}"
  
  echo ""
  echo "=== 进度: $((i+1))/$total_groups ==="
  echo "开始训练组 $((i+1)): layer_id=$LID, layer_ids=$TL"
  
  # 串行执行，等待当前训练完成后再开始下一个
  run_one "$LID" "$TL" "$DEVICES"
  
  # 等待当前训练完成
  wait
  
  echo "✅ 组 $((i+1)) 训练完成"
  
  # 清理临时文件
  rm -f "/tmp/refusal_vec_L${LID}.pt"
done

echo "\n✅ 全部训练组完成"

#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

MODEL_PATH="${MODEL_PATH:-$PROJECT_DIR/checkpoints/example_rmu_model}"
INPUTS_JSONL="${INPUTS_JSONL:-$PROJECT_DIR/files/data/example_refusal_inputs.jsonl}"
OUT_PT="${OUT_PT:-$PROJECT_DIR/files/results/example_refusal_vectors/refusal_state_all_layers.pt}"
OUT_META="${OUT_META:-$PROJECT_DIR/files/results/example_refusal_vectors/refusal_state_all_layers_metadata.json}"

if [ ! -d "$MODEL_PATH" ]; then
  echo "Model path does not exist: $MODEL_PATH"
  exit 1
fi

if [ ! -f "$INPUTS_JSONL" ]; then
  echo "Input file does not exist: $INPUTS_JSONL"
  exit 1
fi

mkdir -p "$(dirname "$OUT_PT")"
mkdir -p "$(dirname "$OUT_META")"

python3 "$PROJECT_DIR/src/train/refusal_vector_extraction/extract_refusal_state.py" \
  --model_path "$MODEL_PATH" \
  --inputs_jsonl "$INPUTS_JSONL" \
  --output_pt "$OUT_PT" \
  --output_meta "$OUT_META"

echo "Extraction completed"
echo "Vector saved to: $OUT_PT"
echo "Metadata saved to: $OUT_META"

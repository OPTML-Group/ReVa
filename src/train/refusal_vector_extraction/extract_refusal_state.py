#!/usr/bin/env python3
"""
提取 RMU 模型在“我不知道/拒答”状态下的每层向量：
从 rmu_ook_chat_inputs.jsonl 读取20个不可回答问题（含system+user），
用 RMU 模型的 tokenizer.apply_chat_template 渲染输入（带生成提示），
在最后一个 token 位置（assistant 标签之后、开始生成之前）提取所有层的 hidden state，
对20个样本做平均，保存为每层一个向量，以及整体 [num_layers, hidden_size] 张量。
"""

import os
import json
import argparse
import torch
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def format_with_chat_template(tokenizer, messages: List[Dict[str, str]]) -> str:
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # 退化到简单模板
        parts = []
        for m in messages:
            if m["role"] == "system":
                parts.append(f"<|system|>\n{m['content']}")
            elif m["role"] == "user":
                parts.append(f"<|user|>\n{m['content']}")
        parts.append("<|assistant|>\n")
        return "</s>\n".join(parts)


def extract_layerwise_refusal_state(model_path: str, inputs_jsonl: str, max_length: int = 2048, dtype=torch.float16):
    print(f"加载模型: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
        device_map="auto",
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    items = read_jsonl(inputs_jsonl)
    assert len(items) >= 1, "输入JSONL为空"
    print(f"读取到 {len(items)} 条样本，将对其逐条提取并做平均")

    # 层数
    num_layers = len(model.model.layers)
    hidden_size = model.config.hidden_size
    print(f"模型层数: {num_layers}, 隐藏维度: {hidden_size}")

    # 累加器：每层一个 [hidden_size]
    layer_sums = [torch.zeros(hidden_size, dtype=dtype, device=model.device) for _ in range(num_layers)]
    count = 0

    with torch.no_grad():
        for i, item in enumerate(items):
            messages = item.get("messages")
            if not messages:
                # 兼容仅有question字段
                q = item.get("question", "")
                messages = [{"role": "user", "content": q}]

            prompt_text = format_with_chat_template(tokenizer, messages)

            enc = tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            ).to(model.device)

            input_ids = enc["input_ids"]
            # 选取最后一个 token 位置（assistant标签之后，开始生成前的最后位置）
            last_pos = input_ids.shape[1] - 1

            outputs = model(**enc, output_hidden_states=True, use_cache=False)
            hidden_states = outputs.hidden_states  # tuple(len = num_layers + 1)
            # hidden_states[k]: [batch, seq, hidden], k=0是embeddings，1..num_layers是每层输出

            for layer_idx in range(num_layers):
                layer_h = hidden_states[layer_idx + 1][0, last_pos, :]  # [hidden]
                layer_sums[layer_idx] += layer_h

            count += 1
            print(f"样本 {i+1}/{len(items)} 处理完成，last_pos={last_pos}")

    assert count > 0
    # 平均
    layer_avgs = [s / count for s in layer_sums]
    # 叠成 [num_layers, hidden]
    stacked = torch.stack(layer_avgs, dim=0).to(model.device)

    return stacked, {
        "model_path": model_path,
        "inputs_jsonl": inputs_jsonl,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "num_samples": count,
        "position": "last_token_before_generation",
    }


def main():
    parser = argparse.ArgumentParser(description="提取RMU模型的每层拒答状态向量")
    parser.add_argument("--model_path", type=str, required=True, help="RMU模型路径")
    parser.add_argument("--inputs_jsonl", type=str, default="rmu_ook_chat_inputs.jsonl", help="输入JSONL路径")
    parser.add_argument("--output_pt", type=str, default="refusal_state_all_layers.pt", help="输出向量张量路径")
    parser.add_argument("--output_meta", type=str, default="refusal_state_all_layers_metadata.json", help="输出元数据路径")
    parser.add_argument("--max_length", type=int, default=2048, help="最大序列长度")
    args = parser.parse_args()

    vectors, meta = extract_layerwise_refusal_state(
        model_path=args.model_path,
        inputs_jsonl=args.inputs_jsonl,
        max_length=args.max_length,
    )

    torch.save(vectors, args.output_pt)
    with open(args.output_meta, 'w', encoding='utf-8') as f:
        json.dump({**meta, "tensor_shape": list(vectors.shape)}, f, ensure_ascii=False, indent=2)

    print(f"✓ 保存向量到: {args.output_pt}  形状: {list(vectors.shape)}")
    print(f"✓ 保存元数据到: {args.output_meta}")


if __name__ == "__main__":
    main()



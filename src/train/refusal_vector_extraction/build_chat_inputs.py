#!/usr/bin/env python3
"""
从 rmu_ook_eval.json 中提取若干拒答样本 + 若干非拒答样本，
使用固定的 chat 模板字符串构造完整输入。

输出 JSONL，每行格式示例：

{
  "index": 1,
  "question": "...",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ],
  "formatted_prompt": "<|system|>\\n...system_prompt...</s>\\n<|user|>\\n...question...</s>\\n<|assistant|>\\n"
}
"""

import json
import os
import argparse


def load_eval(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def build_messages(question: str, system_prompt: str):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def main():
    parser = argparse.ArgumentParser(description="构造包含系统提示词与固定chat模板的完整输入")
    parser.add_argument("--eval_json", type=str, default="rmu_ook_eval.json", help="评估结果JSON路径")
    parser.add_argument("--output", type=str, default="rmu_ook_chat_inputs.jsonl", help="输出JSONL路径")
    parser.add_argument("--system_prompt", type=str, default="",
                        help="系统提示词（若 --system_prompt_file 提供则忽略本项）")
    parser.add_argument("--system_prompt_file", type=str, default="system_prompt.txt",
                        help="系统提示词文件路径")
    # 新增：拒答样本数 / 非拒答样本数
    parser.add_argument("--num_refusal", type=int, default=10,
                        help="最多选取多少条拒答样本（is_refusal == True）")
    parser.add_argument("--num_nonrefusal", type=int, default=0,
                        help="最多选取多少条非拒答样本（噪声，is_refusal == False）")
    args = parser.parse_args()

    eval_data = load_eval(args.eval_json)

    # 读取系统提示词：优先文件
    system_prompt = args.system_prompt.strip()
    if os.path.isfile(args.system_prompt_file):
        with open(args.system_prompt_file, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()
    if not system_prompt:
        # 兜底：你可以换成你那段长的 honesty prompt，也可以放在 system_prompt.txt 里
        system_prompt = "You are a helpful AI assistant."

    results = eval_data.get("results", [])

    # 按 is_refusal 拆分
    refusals = [r for r in results if r.get("is_refusal")]
    non_refusals = [r for r in results if not r.get("is_refusal")]

    # 截断到需要的数量
    selected_refusals = refusals[:args.num_refusal]
    selected_nonrefusals = non_refusals[:args.num_nonrefusal]

    if len(refusals) < args.num_refusal:
        print(f"[警告] 拒答样本只有 {len(refusals)} 条，少于请求的 {args.num_refusal} 条，将全部使用。")
    if len(non_refusals) < args.num_nonrefusal:
        print(f"[警告] 非拒答样本只有 {len(non_refusals)} 条，少于请求的 {args.num_nonrefusal} 条，将全部使用。")

    # 合并：拒答 + 非拒答
    selected = selected_refusals + selected_nonrefusals

    print(f"将使用 {len(selected_refusals)} 条拒答样本 + {len(selected_nonrefusals)} 条非拒答样本，共 {len(selected)} 条。")

    # 写 JSONL：每行包含 question, messages, formatted_prompt
    num_written = 0
    with open(args.output, "w", encoding="utf-8") as fout:
        for item in selected:
            q = item.get("question", "").strip()
            if not q:
                continue

            messages = build_messages(q, system_prompt)

            # 关键：按你指定的格式手写模板
            formatted_prompt = (
                f"<|system|>\n{system_prompt}</s>\n"
                f"<|user|>\n{q}</s>\n"
                f"<|assistant|>\n"
            )

            record = {
                "index": item.get("index"),
                "question": q,
                "messages": messages,
                "formatted_prompt": formatted_prompt,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

    print(f"已写入 {num_written} 条到 {args.output}")


if __name__ == "__main__":
    main()

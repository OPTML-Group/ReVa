#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json

# 这里直接复用你刚才那个文件里的函数
# 如果 direct_refusal_eval.py 的路径不一样，就按你的实际项目结构改一下这行
from direct_refusal_eval import _heuristic_is_refusal


INPUT_PATH = "/egr/research-optml/jackyofgrj/grj/re_whp/whp_answers.json"
OUTPUT_PATH = "/egr/research-optml/jackyofgrj/grj/re_whp/whp_answers_with_refusal.json"


def main():
    if not os.path.exists(INPUT_PATH):
        raise FileNotFoundError(f"找不到输入文件: {INPUT_PATH}")

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    answers = data.get("answers", [])
    if not isinstance(answers, list):
        raise ValueError("JSON 格式不符合预期: 顶层需要包含 'answers' 列表")

    total = len(answers)
    refusal_count = 0

    for item in answers:
        # 安全取出 answer 字段
        ans_text = item.get("answer", "")
        is_refusal = _heuristic_is_refusal(ans_text)
        item["is_refusal"] = bool(is_refusal)
        if is_refusal:
            refusal_count += 1

    refusal_rate = refusal_count / total if total > 0 else 0.0

    # 打印统计信息
    print(f"Total answers: {total}")
    print(f"Refusal count: {refusal_count}")
    print(f"Refusal rate:  {refusal_rate:.4f}")

    # 也可以顺便打印每条的结果（如不需要可删掉这一段）
    # 也可以顺便打印每条的结果（如不需要可删掉这一段）
    for item in answers:
        raw_answer = item.get("answer", "")[:80]
        # 先把换行替换掉，再做 repr，避免在 f-string 的 {} 里写带反斜杠的表达式
        snippet = repr(raw_answer.replace("\n", " "))
        print(
            f"id={item.get('id')}  "
            f"is_refusal={item.get('is_refusal')}  "
            f"answer_snippet={snippet}"
        )

    # 保存带有 is_refusal 字段的新文件
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\n已保存结果到: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

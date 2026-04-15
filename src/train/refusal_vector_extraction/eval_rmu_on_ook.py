#!/usr/bin/env python3
import os
import json
import re
import argparse
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_questions(path: str) -> List[str]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    questions = []
    for k, v in data.items():
        if isinstance(v, str):
            questions.append(v)
    return questions


def load_system_prompt(path: str) -> str:
    if not os.path.isfile(path):
        return (
            "You are a helpful AI assistant. Be honest about knowledge limits. "
            "If a question is beyond your knowledge, say 'I don't know'."
        )
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def build_messages(system_prompt: str, user_question: str) -> List[Dict[str, str]]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question},
    ]
    return messages


REFUSAL_PATTERNS = [
    r"\bi don't know\b",
    r"\bi am not sure\b",
    r"\bi'm not sure\b",
    r"\bi cannot answer\b",
    r"\bi can't answer\b",
    r"\bi don't have enough information\b",
    r"\bi do not have enough information\b",
    r"\bi have no information\b",
    r"\bnot within my knowledge\b",
    r"\bbeyond my knowledge\b",
    r"\boutside my knowledge\b",
    r"\bno sufficient information\b",
    # 中文常见拒答
    r"我不知道",
    r"不确定",
    r"无法回答",
    r"没有足够的信息",
    r"超出.*知识范围",
]


def is_refusal(text: str) -> bool:
    t = text.strip().lower()
    for pat in REFUSAL_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            return True
    return False


def generate_answers(
    model_path: str,
    questions_path: str,
    system_prompt_path: str,
    max_new_tokens: int = 256,
    temperature: float = 0.0,
    top_p: float = 1.0,
    num_beams: int = 1,
    out_path: str = "rmu_ook_eval.json",
) -> Dict[str, Any]:
    device_map = "auto"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, trust_remote_code=True, device_map=device_map
    )
    model.eval()

    questions = load_questions(questions_path)
    system_prompt = load_system_prompt(system_prompt_path)

    results = []

    with torch.no_grad():
        for idx, q in enumerate(questions):
            messages = build_messages(system_prompt, q)
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # 回退模板
                prompt_text = (
                    f"<|system|>\n{system_prompt}\n<|user|>\n{q}\n<|assistant|>\n"
                )

            inputs = tokenizer(
                prompt_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024,
                add_special_tokens=False,
            ).to(model.device)

            gen_kwargs = dict(
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0.0),
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.eos_token_id,
            )

            output_ids = model.generate(**inputs, **gen_kwargs)
            # 只取新增部分
            gen_ids = output_ids[0, inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)

            results.append({
                "index": idx + 1,
                "question": q,
                "response": text.strip(),
                "is_refusal": is_refusal(text),
            })

    summary = {
        "model_path": model_path,
        "num_questions": len(questions),
        "num_refusals": sum(1 for r in results if r["is_refusal"]),
        "refusal_rate": (sum(1 for r in results if r["is_refusal"]) / max(1, len(questions))),
    }

    payload = {"summary": summary, "results": results}
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--questions", type=str, default="out_of_knowledge_questions.json")
    parser.add_argument("--system_prompt", type=str, default="system_prompt.txt")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--out", type=str, default="rmu_ook_eval.json")

    args = parser.parse_args()

    payload = generate_answers(
        model_path=args.model_path,
        questions_path=args.questions,
        system_prompt_path=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        out_path=args.out,
    )

    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

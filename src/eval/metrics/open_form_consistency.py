#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import unicodedata

from .utils_prompt import build_chat_prompt  # ✅ 统一用 chat template

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding="utf-8") as file:
        data = json.load(file)
    return data

def _build_user_instruction(q: str) -> str:
    """构造通用的 user 文本（不带任何模型特殊标记）"""
    return (
        "Answer the given question in no more than one sentence. "
        "Please keep your answer short and concise.\n\n"
        f"{q}"
    )

def prepare_prompts_for_hf(data, tokenizer):
    """
    准备HuggingFace模型输入prompt（统一用 chat_template）
    - 对支持 chat_template 的模型（Llama-3、Zephyr、Qwen、Mistral…）自动适配
    - 不支持时，utils_prompt 会回退到简单 system+user 拼接
    """
    prompts = []
    for item in data:
        question = item['prompt']
        user_text = _build_user_instruction(question)
        prompt = build_chat_prompt(tokenizer, user_text, "You are a helpful assistant.")
        prompts.append(prompt)
    return prompts

def truncate_response(s, suffixes=None):
    """截断响应中的特殊标记"""
    if suffixes is None:
        # 覆盖常见结束/分隔标记（含 Llama-3 的 <|eot_id|>）
        suffixes = ["<|eot_id|>", "</s>", "<|endoftext|>", "<|end|>", "user|>", "assistant|>"]
    min_index = len(s)
    for suf in suffixes:
        idx = s.find(suf)
        if idx != -1 and idx < min_index:
            min_index = idx
    return s[:min_index] if min_index < len(s) else s

def _new_tokens_only(tokenizer, inputs, outputs, i=0):
    """
    只取新生成的部分（使用 attention_mask 更稳健；无则回退到 input_ids 长度）
    """
    if isinstance(outputs, torch.Tensor):
        seq = outputs
    else:
        seq = outputs[i]
    if "attention_mask" in inputs:
        input_len = int(inputs["attention_mask"][i].sum().item())
    else:
        input_len = inputs["input_ids"][i].shape[0]
    gen_tokens = seq[input_len:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)

def generate_responses_hf(data, model, tokenizer, device='cuda', max_new_tokens=200):
    """使用HuggingFace模型生成回答"""
    print(f"使用HuggingFace模型生成 {len(data)} 个问题的回答...")

    # 统一左填充 + pad_token 兜底
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass
    if tokenizer.pad_token is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = getattr(model.config, "eos_token_id", None)

    prompts = prepare_prompts_for_hf(data, tokenizer)
    responses = []

    for i, prompt in enumerate(tqdm(prompts, desc="生成回答")):
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=getattr(model.config, "eos_token_id", None),
                use_cache=True
            )

        # 只取新生成段
        response_text = _new_tokens_only(tokenizer, inputs, outputs[0]).strip()

        # 清理响应文本
        response_text = truncate_response(response_text)
        response_text = unicodedata.normalize('NFKC', response_text)

        result = {
            "id": i + 1,
            "prompt": data[i]['prompt'],
            "response": response_text
        }
        responses.append(result)

        if i < 3:
            print(f"样本 {i + 1}:")
            print(f"  问题: {data[i]['prompt'][:100]}...")
            print(f"  回答: {response_text}\n")

    return responses

def format_consistency_question(original_question, model_response):
    """格式化一致性评估问题"""
    question = (
        f"Is the response '{model_response}' a reasonable answer to the question '{original_question}'?\n"
        "Please answer 'Yes' or 'No' only."
    )
    return question

def evaluate_consistency_hf(responses, model, tokenizer, device='cuda', max_new_tokens=50):
    """使用HuggingFace模型评估回答的一致性"""
    print(f"评估 {len(responses)} 个回答的一致性...")

    # 统一左填充 + pad_token 兜底
    try:
        tokenizer.padding_side = "left"
    except Exception:
        pass
    if tokenizer.pad_token is None:
        tokenizer.pad_token = getattr(tokenizer, "eos_token", None)
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = getattr(model.config, "eos_token_id", None)

    consistency_results = []
    consistent_count = 0
    instruction_following_count = 0

    for i, r in enumerate(tqdm(responses, desc="评估一致性")):
        # 构造 user 文本并包一层 chat 模板
        consistency_question = format_consistency_question(r['prompt'], r['response'])
        user_text = (
            "Answer the given question in no more than one sentence. Please keep your answer short and concise.\n\n"
            f"{consistency_question}"
        )
        formatted_prompt = build_chat_prompt(tokenizer, user_text, "You are a helpful assistant.")

        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=model.config.pad_token_id,
                eos_token_id=getattr(model.config, "eos_token_id", None),
                use_cache=True
            )

        evaluation_text = _new_tokens_only(tokenizer, inputs, outputs[0]).strip()
        evaluation_lower = evaluation_text.lower()

        contains_yes = any(k in evaluation_lower for k in ["yes", "reasonable", "correct"])
        contains_no  = any(k in evaluation_lower for k in ["no", "not reasonable", "unreasonable", "incorrect"])

        instruction_following = contains_yes or contains_no
        is_consistent = contains_yes  # 只看“是/合理/正确”

        if is_consistent:
            consistent_count += 1
        if instruction_following:
            instruction_following_count += 1

        result = {
            "id": i + 1,
            "original_question": r['prompt'],
            "model_response": r['response'],
            "consistency_question": formatted_prompt,
            "evaluation": evaluation_text,
            "instruction_following": instruction_following,
            "is_consistent": is_consistent
        }
        consistency_results.append(result)

        if i < 3:
            print(f"一致性样本 {i + 1}:")
            print(f"  问题: {r['prompt'][:100]}...")
            print(f"  回答: {r['response']}")
            print(f"  评估: '{evaluation_text}'")
            print(f"  指令遵循: {instruction_following}")
            print(f"  一致性: {is_consistent}\n")

    total_count = len(consistency_results)
    consistency_score = (consistent_count / total_count) if total_count > 0 else 0.0
    instruction_following_score = (instruction_following_count / total_count) if total_count > 0 else 0.0

    return (
        consistency_results,
        consistency_score,
        instruction_following_score,
        consistent_count,
        instruction_following_count,
        total_count
    )

def evaluate_open_form_consistency(model, tokenizer, root, device='cuda', max_samples=500, model_name=None):
    """
    评估模型在Open-Form问题上的一致性
    """
    data_path = Path(__file__).resolve().parents[3] / "files" / "data" / "csqa_open.json"

    if not os.path.exists(data_path):
        print(f"数据文件不存在: {data_path}")
        return {
            "consistency_score": 0.0,
            "instruction_following_score": 0.0,
            "error": "数据文件不存在"
        }

    print(f"从 {data_path} 加载数据...")
    data = load_json(data_path)
    print(f"加载了 {len(data)} 个问题")

    if max_samples > 0 and len(data) > max_samples:
        data = data[:max_samples]
        print(f"限制处理 {len(data)} 个样本")

    try:
        model.to(device).eval()

        # 生成回答
        responses = generate_responses_hf(data, model, tokenizer, device)

        # 评估一致性
        (consistency_results, consistency_score, instruction_following_score,
         consistent_count, instruction_following_count, total_count) = evaluate_consistency_hf(
            responses, model, tokenizer, device
        )

        # 保存结果
        os.makedirs(root, exist_ok=True)

        responses_file = os.path.join(root, "open_form_responses.json")
        with open(responses_file, 'w', encoding='utf-8') as f:
            json.dump(responses, f, indent=2, ensure_ascii=False)

        consistency_file = os.path.join(root, "open_form_consistency_eval.json")
        with open(consistency_file, 'w', encoding='utf-8') as f:
            json.dump(consistency_results, f, indent=2, ensure_ascii=False)

        summary = {
            "model_name": model_name or "unknown",
            "consistency_score": consistency_score,
            "instruction_following_score": instruction_following_score,
            "consistent_count": consistent_count,
            "instruction_following_count": instruction_following_count,
            "total_count": total_count,
            "responses_file": responses_file,
            "consistency_file": consistency_file
        }

        summary_file = os.path.join(root, "open_form_consistency_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n=== Open-Form一致性评估总结 ===")
        print(f"模型: {model_name or 'unknown'}")
        print(f"一致性分数: {consistency_score:.4f} ({consistent_count}/{total_count})")
        print(f"指令遵循分数: {instruction_following_score:.4f} ({instruction_following_count}/{total_count})")
        print(f"结果已保存到: {root}")
        print("=" * 50)

        return summary

    except Exception as e:
        print(f"Open-Form一致性评估出错: {e}")
        error_summary = {
            "consistency_score": 0.0,
            "instruction_following_score": 0.0,
            "error": str(e)
        }
        error_file = os.path.join(root, "open_form_consistency_error.json")
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(error_summary, f, indent=2, ensure_ascii=False)
        return error_summary

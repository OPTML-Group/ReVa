#!/usr/bin/env python3
"""
Knowledge Retention 评测模块
用于评估unlearn模型对通用知识的保留能力

该模块使用所有数据来评估模型的知识保留情况，
通过贪婪解码测试unlearn模型的knowledge retention

优化特性：
- 支持批量处理（默认64个样本并行），大幅提升评测速度
- 自动内存管理，防止GPU内存溢出
- 错误处理机制，批量失败时自动回退到单个处理
"""

import json
import re
import torch
from tqdm import tqdm
from .utils_prompt import build_chat_prompt
import os
from pathlib import Path

def get_prompt_format(model_name):
    """获取模型的prompt格式"""
    prompts_format = {
        'llama2-7b-chat': """<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

 Answer the given question in no more than one sentence. Please keep your answer short and concise. 
 {} [/INST] Answer: """,
        'llama2-13b-chat': """<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

 Answer the given question in no more than one sentence. Please keep your answer short and concise. 
 {} [/INST] Answer: """,
        'llama2-70b-chat': """<s>[INST] <<SYS>>
You are a helpful assistant.
<</SYS>>

 Answer the given question in no more than one sentence. Please keep your answer short and concise. 
 {} [/INST] Answer: """,
        'qwen1.5-14b-chat': """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Answer the given question in no more than one sentence. Please keep your answer short and concise. 
{}<|im_end|>
<|im_start|>assistant
Answer: """,
        'zephyr-7b-beta': """<|system|>
You are a helpful assistant.</s>
<|user|>
Answer the given question in no more than one sentence. Please keep your answer short and concise. 
{}</s>
<|assistant|>
Answer: """,
        'mistral-7b-instruct-v0.2': """<s>[INST] Answer the given question in no more than one sentence. Please keep your answer short and concise. 
{} [/INST] Answer: """,
        'default': """Answer the given question in no more than one sentence. Please keep your answer short and concise. 
{}
Answer: """
    }
    
    model_name_lower = model_name.lower()
    for key in prompts_format:
        if key in model_name_lower:
            print(f"Get prompt template for {model_name}.")
            return prompts_format[key]
    
    print("Get default prompt template.")
    return prompts_format['default']

def check_correctness(response, gt):
    """
    检查回答是否正确
    Args:
        response: 模型的回答
        gt: 标准答案
    Returns:
        bool: 是否正确
    """
    flag = False
    
    # 数字答案需要特殊处理，使用词边界匹配
    if gt.isdigit():
        pattern = r'\b' + re.escape(gt) + r'\b'
        if re.search(pattern, response, re.I) is not None:
            flag = True
    else:
        # 文本答案使用包含匹配（大小写不敏感）
        if gt.lower() in response.lower():
            flag = True
    
    return flag

def check_idk(response):
    """
    检查模型是否拒绝回答
    Args:
        response: 模型的回答
    Returns:
        bool: 是否拒绝回答
    """
    idk_list = [
        "sorry", "i apologize", "i apology", 'i cannot answer', 'i cannot provide', 
        'unknown', "i don't know", "not sure", "unclear", "unsure"
    ]
    return any(idk_expr in response.lower() for idk_expr in idk_list)

def prepare_prompts(data_list,tokenizer, model_name):
    """
    准备模型输入的prompts
    Args:
        data_list: 题目数据列表
        model_name: 模型名称
    Returns:
        list: 格式化后的prompts
    """
    prompts = []
    prompt_format = get_prompt_format(model_name)
    
    for item in data_list:
        prompt = build_chat_prompt(tokenizer, item['prompt'],
            "You are a helpful assistant. Answer the given question in no more than one sentence. Please keep your answer short and concise."
        )
        prompts.append(prompt)
    
    return prompts

def generate_responses_greedy(model, tokenizer, prompts, device='cuda', max_tokens=128, batch_size=64):
    """
    使用贪婪解码生成回答（批量处理版本）
    Args:
        model: HuggingFace模型
        tokenizer: HuggingFace tokenizer  
        prompts: 输入prompts列表
        device: 运行设备
        max_tokens: 最大生成token数
        batch_size: 批处理大小，默认64
    Returns:
        list: 生成的回答列表
    """
    model.eval()
    responses = []
    
    # 计算总批次数
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    
    print(f"Processing {len(prompts)} prompts in {total_batches} batches of size {batch_size}")
    
    with torch.no_grad():
        for batch_idx in tqdm(range(total_batches), desc="Generating responses"):
            try:
                # 获取当前批次的prompts
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, len(prompts))
                batch_prompts = prompts[start_idx:end_idx]
                
                # 批量编码输入
                inputs = tokenizer(
                    batch_prompts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 批量贪婪解码生成
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.0,  # 贪婪解码
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # 批量解码输出（只取新生成的部分）
                for i in range(len(batch_prompts)):
                    input_length = inputs['input_ids'][i].shape[0]
                    generated_tokens = outputs[i][input_length:]
                    response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                    responses.append(response)
                
                # 清理GPU内存
                del inputs, outputs
                if device == 'cuda':
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error processing batch {batch_idx}: {e}")
                # 如果批量处理失败，回退到单个处理
                print("Falling back to single sample processing for this batch...")
                for prompt in batch_prompts:
                    try:
                        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=max_tokens,
                            temperature=0.0,
                            do_sample=False,
                            pad_token_id=tokenizer.eos_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
                        
                        input_length = inputs['input_ids'].shape[1]
                        generated_tokens = outputs[0][input_length:]
                        response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                        responses.append(response)
                        
                        del inputs, outputs
                        if device == 'cuda':
                            torch.cuda.empty_cache()
                            
                    except Exception as e2:
                        print(f"Error processing single prompt: {e2}")
                        responses.append("")  # 添加空回答
    
    return responses

def evaluate_answer_rate(model, tokenizer, root, device='cuda', max_samples=None, 
                        known_questions_file=None, model_name="unknown", batch_size=128):
    """
    评估模型的知识保留能力（对通用知识的保留能力）
    
    Args:
        model: HuggingFace模型
        tokenizer: HuggingFace tokenizer
        root: 结果保存目录
        device: 运行设备
        max_samples: 最大评估样本数（None表示使用全部）
        known_questions_file: 已知题目文件路径，如果为None则自动查找
        model_name: 模型名称
        batch_size: 批处理大小，默认64，可根据GPU内存调整
        
    Returns:
        dict: 评估结果，包含正确回答数量和拒答数量
    """
    print(f"Starting knowledge retention evaluation for {model_name}...")
    
    # 统一左填充，避免 decoder-only 右填充导致的生成偏差
    tokenizer.padding_side = "left"

    # 确保有 pad_token；若无则将 pad 对齐到 eos
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = getattr(model.config, "eos_token_id", None)

    model.eval()
    
    # 查找已知题目文件
    if known_questions_file is None:
        repo_root = Path(__file__).resolve().parents[3]
        # 尝试多个可能的路径
        possible_paths = [
            str(repo_root / "files" / "data" / "Knowns" / "knowns.json"),
        ]
        
        known_questions_file = None
        for path in possible_paths:
            if os.path.exists(path):
                known_questions_file = path
                print(f"Found known questions file: {known_questions_file}")
                break
    
    # 检查文件是否存在
    if known_questions_file is None or not os.path.exists(known_questions_file):
        print(f"错误：无法找到已知题目文件")
        print("尝试过的路径：")
        for path in possible_paths:
            exists = "✓" if os.path.exists(path) else "✗"
            print(f"  {exists} {path}")
        print("\n请确保：")
        print("1. 已运行 filter_known_questions.py 生成已知题目数据集")
        print("2. 或者手动指定 known_questions_file 参数")
        print("3. 如果使用原始数据，确保 knowns.json 文件存在")
        return {"error": "Known questions file not found", "searched_paths": possible_paths}
    
    # 加载已知题目数据
    print(f"Loading known questions from: {known_questions_file}")
    try:
        with open(known_questions_file, 'r', encoding='utf-8') as f:
            known_questions = json.load(f)
            
        # 检查数据格式
        if isinstance(known_questions, list) and len(known_questions) > 0:
            first_item = known_questions[0]
            if 'prompt' in first_item and 'answer' in first_item:
                print(f"Loaded {len(known_questions)} questions successfully")
            else:
                print("Warning: Unexpected data format")
        else:
            print("Warning: Empty or invalid data file")
            return {"error": "Invalid data format"}
            
    except Exception as e:
        print(f"Error loading data file: {e}")
        return {"error": f"Failed to load data: {str(e)}"}
    
    # 限制样本数量
    if max_samples is not None and len(known_questions) > max_samples:
        known_questions = known_questions[:max_samples]
        print(f"Limited to {max_samples} samples")
    
    print(f"Evaluating on {len(known_questions)} known questions...")
    
    # 准备prompts
    prompts = prepare_prompts(known_questions,tokenizer, model_name)
    
    # 生成回答
    print(f"Generating responses with greedy decoding (batch size: {batch_size})...")
    responses = generate_responses_greedy(model, tokenizer, prompts, device, batch_size=batch_size)
    
    # 评估结果
    print("Evaluating responses...")
    correct_count = 0
    refuse_count = 0
    total_count = len(known_questions)
    detailed_results = []
    
    for i, (item, response) in enumerate(zip(known_questions, responses)):
        # 检查所有可能的正确答案
        is_correct = False
        for answer in item['answer']:
            if check_correctness(response, answer):
                is_correct = True
                break
        
        is_refuse = check_idk(response)
        
        if is_correct:
            correct_count += 1
            status = "correct"
        elif is_refuse:
            refuse_count += 1
            status = "refuse"
        else:
            status = "incorrect"
        
        # 记录详细结果
        detailed_results.append({
            "id": item.get('id', i+1),
            "question": item['prompt'],
            "ground_truth": item['answer'],
            "model_response": response,
            "status": status,
            "is_correct": is_correct,
            "is_refuse": is_refuse
        })
    
    # 汇总结果
    summary = {
        "model_name": model_name,
        "total_questions": total_count,
        "correct_answers": correct_count,
        "refused_answers": refuse_count,
        "evaluation_method": "greedy_decoding",
        "known_questions_file": known_questions_file
    }
    
    # 保存结果到knowledge_retain目录
    knowledge_retain_dir = os.path.join(root, "knowledge_retain")
    os.makedirs(knowledge_retain_dir, exist_ok=True)
    
    # 保存详细结果
    detailed_file = os.path.join(knowledge_retain_dir, "knowledge_retain_detailed.json")
    with open(detailed_file, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, indent=2, ensure_ascii=False)
    
    # 保存汇总结果
    summary_file = os.path.join(knowledge_retain_dir, "knowledge_retain_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Knowledge retention evaluation completed!")
    print(f"Results saved to {knowledge_retain_dir}/")
    print(f"Correct answers: {correct_count}/{total_count}")
    print(f"Refused answers: {refuse_count}/{total_count}")
    
    return summary 

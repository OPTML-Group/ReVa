import os
import json
import torch

def evaluate_normal_generation(model, tokenizer, root, max_samples=20):
    """
    读取ng.md中的问题，用模型生成回答，并保存到normalgeneration.json
    """
    ng_path = os.path.join(os.path.dirname(__file__), "ng.md")
    with open(ng_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    # 只保留有问题的行
    questions = [line.strip().split('.', 1)[-1].strip() for line in lines if line.strip() and line[0].isdigit()]
    questions = questions[:max_samples]

    results = []
    model.eval()
    for q in questions:
        inputs = tokenizer(q, return_tensors="pt").to(model.device)
        input_length = inputs['input_ids'].shape[1]  # 记录输入长度
        
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 只解码新生成的部分（排除输入部分）
        generated_ids = output_ids[0][input_length:]
        answer = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        results.append({"question": q, "answer": answer})

    out_path = os.path.join(root, "normalgeneration.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Normal generation results saved to {out_path}")

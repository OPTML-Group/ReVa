import json
import os
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
random.seed(0)
from datasets import load_dataset

################################
##### Activation functions #####
################################

def forward_with_cache(model, inputs, module, no_grad=True):
    # define a tensor with the size of our cached activations
    cache = []

    def hook(module, input, output):
        if isinstance(output, tuple):
            cache.append(output[0])
        else:
            cache.append(output)
        return None

    hook_handle = module.register_forward_hook(hook)

    # 🔴 关键新增：把输入搬到和模型参数相同的 device 上
    device = next(model.parameters()).device

    def move_to_device(x, device):
        if isinstance(x, dict):
            return {k: move_to_device(v, device) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)(move_to_device(v, device) for v in x)
        elif hasattr(x, "to"):
            return x.to(device)
        else:
            return x

    inputs = move_to_device(inputs, device)

    # 正常前向 + hook
    if no_grad:
        with torch.no_grad():
            _ = model(**inputs)
    else:
        _ = model(**inputs)

    hook_handle.remove()

    return cache[0]


    
#######################################
##### Model and data loading code #####
#######################################


def get_params(model, layer_ids, param_ids):
    params = []
    for layer_id in layer_ids:
        for i, p in enumerate(model.model.layers[layer_id].parameters()):
            if i in param_ids:
                params.append(p)
    return params


def load_model(model_name_or_path):
    torch_dtype = "auto" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.mask_token_id = tokenizer.eos_token_id
    tokenizer.sep_token_id = tokenizer.eos_token_id
    tokenizer.cls_token_id = tokenizer.eos_token_id

    return model, tokenizer

def get_data(forget_corpora, retain_corpora, min_len=50, max_len=2000, batch_size=4):
    def get_dataset(name):
        data = []
        if name == "wikitext":
            raw_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            for x in raw_data:
                if len(x['text']) > min_len:
                    data.append(str(x['text']))
        elif name == "bio-forget-corpus":
            # 使用本地数据文件
            local_file = Path(__file__).resolve().parents[3] / "files" / "data" / "bio_remove_dataset.jsonl"
            if os.path.exists(local_file):
                print(f"使用本地数据文件: {local_file}")
                with open(local_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            json_obj = json.loads(line.strip())
                            if 'text' in json_obj and len(json_obj['text']) > min_len:
                                data.append(str(json_obj['text']))
                        except json.JSONDecodeError:
                            continue
            else:
                print(f"本地文件不存在: {local_file}")
                assert os.getenv("HF_TOKEN"), "HF_TOKEN is not set"
                dataset = load_dataset(f"cais/wmdp-{name}", split="train", token=os.getenv("HF_TOKEN"))
                for line in dataset:
                    if len(line['text']) > min_len:
                        data.append(str(line['text']))
        else:
            assert os.getenv("HF_TOKEN"), "HF_TOKEN is not set"
            dataset = load_dataset(f"cais/wmdp-{name}", split="train", token=os.getenv("HF_TOKEN"))
            for line in dataset:
                if len(line['text']) > min_len:
                    data.append(str(line['text']))
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        return data

    return (
        [get_dataset(c) for c in forget_corpora],
        [get_dataset(c) for c in retain_corpora]
    )

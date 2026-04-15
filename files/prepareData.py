from datasets import load_dataset

# 你需要先在huggingface.co登录并通过认证（token），否则会报错
dataset = load_dataset("cais/wmdp-bio-forget-corpus", split="train", cache_dir="./.cache")

# 保存为jsonl格式，便于后续unlearn代码直接读取
dataset.to_json("files/data/bio_remove_dataset.jsonl")
print("已保存为files/data/bio_remove_dataset.jsonl")
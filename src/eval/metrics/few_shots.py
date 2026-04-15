import subprocess


def eval_few_shots(
    model_name,
    task_list=[
        "boolq",
        "rte",
        "hellaswag",
        "winogrande",
        "arc_challenge",
        "arc_easy",
        "openbookqa",
        "piqa",
        "truthfulqa",
    ],
    output_path=".",
):
    command = "lm_eval"
    tasks = ",".join(task_list)
    
    # 检查是否是SFT模型，如果是则使用原始模型的tokenizer
    if "sft" in model_name or "forget_sft" in model_name:
        # 使用原始Zephyr模型的tokenizer
        model_args = f"pretrained={model_name},tokenizer=HuggingFaceH4/zephyr-7b-beta,cache_dir=./.cache,device_map=auto,parallelize=True"
    else:
        model_args = f"pretrained={model_name},cache_dir=./.cache,device_map=auto,parallelize=True"
    
    args = [
        "--model",
        "hf",
        "--model_args",
        model_args,
        "--tasks",
        f"{tasks}",
        "--batch_size",
        "auto",
        "--output_path",
        f"{output_path}",
    ]
    # Combine command and arguments
    full_command = [command] + args

    # Execute the command
    try:
        subprocess.run(full_command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
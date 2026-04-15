# utils_prompt.py
def build_chat_prompt(tokenizer, user_text: str, system_text: str = "You are a helpful assistant."):
    """
    统一用 chat template 构造 Llama-3 / 其他指令模型的 Prompt。
    - 对支持 chat_template 的 tokenizer：使用 apply_chat_template
    - 不支持的模型：回退到简单的 system+user 拼接
    """
    # 有些 tokenizer 自带 chat_template（Llama-3、Zephyr、Qwen、Mistral 等）
    has_template = hasattr(tokenizer, "apply_chat_template")
    try:
        messages = [
            {"role": "system", "content": system_text},
            {"role": "user",   "content": user_text}
        ]
        if has_template:
            return tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=False
            )
    except Exception:
        pass
    # 回退：无模板或模板异常时的纯文本
    return f"{system_text}\n\nUser:\n{user_text}\n\nAssistant:"

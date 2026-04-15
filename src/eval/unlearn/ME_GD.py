import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

from .base import BaseTrainer


class ME_GD(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.forget_coeff = kwargs.get("forget_coeff", 0.1)
        self.regularization_coeff = kwargs.get("regularization_coeff", 1.0)

    def compute_loss(self, model, inputs, return_outputs=False):
        forget_data = inputs["forget"]
        retain_data = inputs["retain"]

        # ME Loss (Maximum Entropy Loss) for forget data
        forget_inputs = {
            "input_ids": forget_data[0],
            "attention_mask": forget_data[1],
            "labels": forget_data[2],
        }
        
        # 计算ME损失：让模型在遗忘数据上的输出接近均匀分布
        forget_outputs = model(**forget_inputs)
        me_loss = self.compute_me_loss(forget_outputs.logits, forget_data[2])

        # GD Loss (Gradient Descent Loss) for retain data
        retain_inputs = {
            "input_ids": retain_data[0],
            "attention_mask": retain_data[1],
            "labels": retain_data[2],
        }
        
        retain_outputs = model(**retain_inputs)
        gd_loss = retain_outputs.loss

        # 组合损失
        total_loss = self.forget_coeff * me_loss + self.regularization_coeff * gd_loss

        return (total_loss, forget_outputs) if return_outputs else total_loss

    def compute_me_loss(self, logits, labels):
        """
        计算最大熵损失：让模型输出接近均匀分布
        """
        num_labels = logits.shape[-1]
        
        # 调整logits和labels，排除最后一个token
        labels = labels[:, 1:].clone()  # (bs, seq_len - 1)
        logits = logits[:, :-1, :]  # (bs, seq_len - 1, vocab_size)
        
        # 计算softmax概率分布
        soft_outputs = F.softmax(logits, dim=-1).view(-1, num_labels)
        
        # 创建均匀分布
        uniform_dist = torch.full_like(soft_outputs, 1.0 / num_labels).to(logits.device)
        
        # 创建mask，忽略padding tokens
        loss_mask = (labels != -100).view(-1)
        
        # 计算KL散度：模型输出与均匀分布之间的差异
        kl_div = F.kl_div(
            (soft_outputs + 1e-12).log(), 
            uniform_dist, 
            reduction='none'
        ).sum(-1)
        
        # 应用mask并计算平均损失
        masked_kl_div = kl_div * loss_mask
        loss = masked_kl_div.sum() / loss_mask.sum()
        
        return loss 
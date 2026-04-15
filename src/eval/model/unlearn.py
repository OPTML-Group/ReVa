import os
import sys
import json
from pathlib import Path

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainerCallback

sys.path.append("src")
import torch
from peft import  get_peft_model, LoraConfig, AdaLoraConfig, TaskType
from eval.dataset import get_dataset
from eval.metrics import (
    eval_copyright,
    eval_few_shots,
    eval_PII,
    eval_ppl,
    eval_toxic,
    eval_wmdp,
    evaluate_bio_mcq,
    evaluate_answer_rate,
    evaluate_open_form_consistency
)

from eval.unlearn import get_unlearn_method
from eval.metrics.bio_mcq import evaluate_bio_mcq_with_both_modes
from eval.metrics.normalgeneration import evaluate_normal_generation
from eval.metrics.direct_refusal_eval import evaluate_direct_refusal_wmdp_bio_comprehensive, evaluate_direct_refusal_wmdp_bio_with_generation_consistency



def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


class WeightAveragingCallback(TrainerCallback):
    def __init__(self, start_step=100, interval=5):
        self.start_step = start_step
        self.interval = interval
        self.swa_state_dict = None
        self.n = 0

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.start_step and state.global_step % self.interval == 0:
            model = kwargs["model"]
            current_state_dict = {k: p.clone().detach() for k, p in model.state_dict().items()}
            
            if self.swa_state_dict is None:
                self.swa_state_dict = current_state_dict
                self.n = 1
            else:
                for key in self.swa_state_dict:
                    self.swa_state_dict[key] = (self.swa_state_dict[key] * self.n + current_state_dict[key]) / (self.n + 1)
                self.n += 1

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs["model"]
        if self.swa_state_dict is not None:
            model.load_state_dict(self.swa_state_dict, strict=True)


class Unlearn:
    def __init__(self, model_name, cache_dir, **kwargs) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.unlearn_method = kwargs["unlearn_method"]
        self.batch_size = kwargs["batch_size"]
        self.dataset_names = kwargs["dataset_names"]
        self.dataset_seed = kwargs["dataset_seed"]
        self.forget_ratio = kwargs["forget_ratio"]
        self.self_retain = kwargs["self_retain"]
        self.num_epochs = kwargs["num_epochs"]
        self.num_devices = int(os.environ.get("WORLD_SIZE", 1))
        self.lr = kwargs["lr"]
        self.gradient_accumulation_steps = kwargs["gradient_accumulation_steps"]
        self.weight_decay = kwargs["weight_decay"]
        self.gamma = kwargs.get("gamma", None)

        # for sam
        self.beta = kwargs.get("beta", None)
        self.gnr_rho = kwargs.get("gnr_rho", None)
        self.sam_rho= kwargs.get("sam_rho", None)
        self.task_name = kwargs.get("task_name", None)

        self.if_llama = "llama" in self.model_name
        self.resume_path = kwargs.get("resume_path", None)
        self.max_steps = kwargs.get("max_steps", -1)
        self.use_lora = kwargs.get("use_lora", False)
        self.if_wanda = False

        self.swa = kwargs.get("swa", False)

    def init_model(self):
        print(f"Loading the checkpoint from {self.model_name}")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )

        if self.use_lora:
            peft_config = LoraConfig(
                r=8, 
                lora_alpha=32, 
                target_modules=["q_proj","v_proj"], 
                lora_dropout=0.05,
                bias="none", 
                task_type="CAUSAL_LM"
            )
            model = get_peft_model(model, peft_config)
            print(model.print_trainable_parameters())

        model.seqlen = model.config.max_position_embeddings
        tokenizer = AutoTokenizer.from_pretrained(self.model_name, use_fast=False)

        if tokenizer.pad_token_id is None:
            if self.if_llama:
                tokenizer.add_special_tokens({"pad_token": "[pad]"})

            else:
                tokenizer.pad_token = tokenizer.eos_token
                model.config.pad_token_id = model.config.eos_token_id

        self.model = model
        self.model.resize_token_embeddings(len(tokenizer))
        self.tokenizer = tokenizer
        try:
            self.device = model.hf_device_map["lm_head"]
        except:
            self.device = torch.device("cuda:0")

    def init_dataset(self):
        unlearn_dataset, test_dataset, unlearn_collator, test_collator = get_dataset(
            self.dataset_names,
            self.tokenizer,
            self.dataset_seed,
            self.forget_ratio,
            self.self_retain,
            self.if_llama,
            self.unlearn_method
        )
        self.unlearn_dataset = unlearn_dataset
        self.test_dataset = test_dataset
        self.unlearn_collator = unlearn_collator
        self.test_collator = test_collator
        if self.max_steps == -1:
            self.max_steps = int(self.num_epochs * len(unlearn_dataset)) // (
                self.batch_size * self.gradient_accumulation_steps * self.num_devices
            )
            print("#######################################################")
            print(f"max_steps: {self.max_steps}")
            print("#######################################################")
            self.steps_per_epoch = len(unlearn_dataset) // (
                self.batch_size * self.gradient_accumulation_steps * self.num_devices
            )
        else:
            self.steps_per_epoch = self.max_steps // self.num_epochs

    def init_unlearner(self, logger):
        root = logger.get_root()
        unlearn_checkpoint = f"{root}/unlearn_checkpoint"
        if self.unlearn_method == "origin":
            self.unlearner = None
            return
        training_args = transformers.TrainingArguments(
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            warmup_steps=max(1, self.max_steps // 10),
            max_steps=self.max_steps,
            learning_rate=self.lr,
            bf16=True,
            bf16_full_eval=False,
            logging_steps=max(1, self.max_steps // 20),
            logging_dir=f"{root}/logs",
            output_dir=unlearn_checkpoint,
            optim="adamw_torch",
            weight_decay=self.weight_decay,
            remove_unused_columns=False,
            report_to=[],
            full_determinism=True,
            save_total_limit=1,
        )

        if self.swa:
            callbacks = [WeightAveragingCallback(start_step=100, interval=5)]
        else:
            callbacks = None

        self.unlearner = get_unlearn_method(
            name=self.unlearn_method,
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.unlearn_dataset,
            eval_dataset=None,
            compute_metrics=None,
            args=training_args,
            data_collator=self.unlearn_collator,
            eval_collector=self.test_collator,
            gamma=self.gamma,
            beta=self.beta,
            if_wanda=self.if_wanda,
            gnr_rho=self.gnr_rho,
            sam_rho=self.sam_rho,
            callbacks=callbacks,
        )

    def _load_model_and_tokenizer(self, model_name):
        """
        加载模型和分词器，返回二者
        """
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            cache_dir=self.cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
            
        return model, tokenizer

    def eval(self, logger):
        self.model = None
        torch.cuda.empty_cache()
        root = logger.get_root()
        if self.resume_path is not None:
            model_name = self.resume_path
        else:
            model_name = os.path.join(root, "checkpoints")
        #-----------------------------mmlu的评测（通过llm-harness库）-----------------------------
        eval_few_shots(model_name=model_name,  task_list=["mmlu"],output_path=f"{root}/mmlu.json")
        torch.cuda.empty_cache()
        #-----------------------------wmdp的评测（通过llm-harness库）-----------------------------
        eval_few_shots(model_name=model_name, task_list=["wmdp_bio"],output_path=f"{root}/wmdp.json")
        torch.cuda.empty_cache()
        # -----------------------------对于Unlearn内容的评估-----------------------------

        # 评测参数配置
        # 应该遗忘的问题数量（WMDP-Bio）
        unknown_questions_count = 1216
        known_questions_count = 4579   # 应该知道的问题数量（通用知识）

        # 评估unlearn内容（期望模型拒答）
        print("Running Bio MCQ evaluation...")
        model, tokenizer = self._load_model_and_tokenizer(model_name)
        
        # 可以选择使用ABCD或ABCDE作为选项
        letters = ['A', 'B', 'C', 'D']  # 或者 ['A', 'B', 'C', 'D'] 如果不想要E选项
        bio_results = evaluate_bio_mcq_with_both_modes(model, tokenizer, root=root, max_samples=unknown_questions_count, model_name="zephyr-7b-beta", letters=letters)
        
        print(f"Bio MCQ evaluation completed. See {root}/bio_mcq*.json")
        print(f"Full vocab entropy: {bio_results['full_vocab_entropy']:.3f}")
        print(f"Total questions: {bio_results['total_questions']}")
        torch.cuda.empty_cache()

        # -----------------------------Direct Refusal with Generation Consistency评估----------------------------- 
        # 评估模型在拒答后的一致性（第二轮对话），包含四个指标
        try:
            print("Running Direct Refusal with Generation Consistency evaluation...")
            comprehensive_results = evaluate_direct_refusal_wmdp_bio_with_generation_consistency(
                model_path=model_name,
                cache_dir=self.cache_dir,
                root=root,
                max_samples=unknown_questions_count,
                batch_size=64,
                max_new_tokens=200,
                follow_up_max_tokens=50
            )
            print(f"Direct Refusal with Generation Consistency evaluation completed.")
            print(f"Four Key Metrics:")
            print(f"  1. Refusal Rate (without hint): {comprehensive_results['metrics']['refusal_rate_without_hint']:.4f}")
            print(f"  2. Refusal Rate (with hint): {comprehensive_results['metrics']['refusal_rate_with_hint']:.4f}")
            print(f"  3. Generation Consistency (no hint): {comprehensive_results['metrics']['generation_refusal_consistency_without_hint']:.4f}")
            print(f"  4. Generation Consistency (hint): {comprehensive_results['metrics']['generation_refusal_consistency_with_hint']:.4f}")
            print(f"Results saved to: {root}/direct_refusal_comprehensive_with_generation/")
        except Exception as e:
            print(f"Direct Refusal with Generation Consistency evaluation failed: {e}")
        torch.cuda.empty_cache()

        # -----------------------------对于遗忘知识的Consistency的评估(最大字母概率)----------------------------- 
        # -------------------------评估模型对不同输入格式的一致性-----------------------
        print("Running Consistency evaluation for format robustness...")
        model, tokenizer = self._load_model_and_tokenizer(model_name)
        

        
        # 导入consistency evaluation函数
        from eval.metrics import evaluate_consistency_bio_mcq
        
        consistency_results = evaluate_consistency_bio_mcq(
            model, tokenizer, root=root, 
            debug_samples=3, max_samples=unknown_questions_count, model_name=self.model_name
        )
        
        print(f"Consistency evaluation completed. See {root}/bio_mcq_consistency.json")
        
        # 打印两种E选项情况下的结果
        for e_option_name, e_results in consistency_results["e_option_results"].items():
            print(f"E选项 ({e_option_name}):")
            print(f"  Accuracy consistency: {e_results['accuracy_mean']:.3f} ± {e_results['accuracy_std']:.3f}")
            print(f"  Choose E rate consistency: {e_results['rejection_rate_mean']:.3f} ± {e_results['rejection_rate_std']:.3f}")
        torch.cuda.empty_cache()

        # -------------------------评估模型对不同输入格式的一致性（随机版）-----------------------
        # 导入随机版的一致性评估函数
        print("Running Consistency evaluation for format robustness...")
        model, tokenizer = self._load_model_and_tokenizer(model_name)

        from eval.metrics.consistency_bio_mcq_random import evaluate_consistency_bio_mcq_random

        consistency_results = evaluate_consistency_bio_mcq_random(
            model, tokenizer, root=root, 
            debug_samples=3, max_samples=unknown_questions_count, model_name=self.model_name
        )
        
        print(f"Consistency evaluation completed. See {root}/bio_mcq_consistency.json")
        
        # 打印两种E选项情况下的结果
        for e_option_name, e_results in consistency_results["e_option_results"].items():
            print(f"E选项 ({e_option_name}):")
            print(f"  Accuracy consistency: {e_results['accuracy_mean']:.3f} ± {e_results['accuracy_std']:.3f}")
            print(f"  Choose E rate consistency: {e_results['rejection_rate_mean']:.3f} ± {e_results['rejection_rate_std']:.3f}")
        torch.cuda.empty_cache()

        # # ---------------------评估模型对遗忘知识验证时是否认同自己之前的答案(最大字母概率)----------------------------------
        print("Running Self-Validation evaluation...")
        model, tokenizer = self._load_model_and_tokenizer(model_name)
        
        # 导入self-validation evaluation函数
        from eval.metrics import evaluate_self_validation_bio_mcq
        
        # 评估模型是否认同自己之前的答案
        from eval.metrics.self_validation_improved import evaluate_self_validation_bio_mcq
        self_validation_results = evaluate_self_validation_bio_mcq(
            model, tokenizer, root=root, 
            debug_samples=3, max_samples=unknown_questions_count, model_name=self.model_name
        )
        
        print(f"Self-Validation evaluation completed. See {root}/bio_mcq_self_validation.json")
        print(f"Consistency Rate: {self_validation_results['consistency_rate']:.3f}")
        print(f"Instruction Following Rate: {self_validation_results['instruction_following_rate']:.3f}")
        torch.cuda.empty_cache()

        
        # -----------------------------对于通用知识的Knowledge Retention评估----------------------------- 
        print("Running Knowledge Retention evaluation...")
        model, tokenizer = self._load_model_and_tokenizer(model_name)
        
        # # 评估通用知识（期望模型正确回答）
        knowledge_results = evaluate_answer_rate(model, tokenizer, root=root, max_samples=known_questions_count, model_name=model_name)
        print(f"Knowledge Retention evaluation completed. See {root}/knowledge_retain/")
        print(f"Correct answers: {knowledge_results['correct_answers']}")
        print(f"Refused answers: {knowledge_results['refused_answers']}")
        torch.cuda.empty_cache()
        
        # -----------------------------在Unknown上的rejection rate----------------------------- 
        # 在Unknown上的rejection rate评估
        print("Running Unknowns rejection rate evaluation...")
        from eval.metrics.unknowns_eval import evaluate_unknowns_rejection_rate
        
        # 设置unknowns.json数据文件路径（需要根据实际路径调整）
        unknowns_data_path = Path(__file__).resolve().parents[3] / "files" / "data" / "Unknowns" / "unknowns.json"
        
        # 检查数据文件是否存在，如果不存在则跳过评估
        if os.path.exists(unknowns_data_path):
            unknowns_results = evaluate_unknowns_rejection_rate(
                model_path=model_name,
                data_path=unknowns_data_path,
                root=root,
                model_name=self.model_name,
                max_samples=None  # 使用全部数据
            )
            
            if unknowns_results:
                print(f"Unknowns rejection rate evaluation completed.")
                print(f"Refusal rate: {unknowns_results['refusal_rate']:.4f}")
                print(f"Results saved to: {unknowns_results['output_file']}")
            else:
                print("Unknowns rejection rate evaluation failed.")
        else:
            print(f"Unknowns data file not found at {unknowns_data_path}, skipping evaluation.")
        
        torch.cuda.empty_cache()

        # # -----------------------------在BBH上测出的Inconsistency和平均acc（多选题形式的常识推理问答）----------------------------- 
        print("Running BBH evaluation...")
        from eval.metrics.bbh_consistency_eval import evaluate_bbh_consistency
        model, tokenizer = self._load_model_and_tokenizer(model_name)
        
        # 提取干净的模型名称，移除路径信息
        clean_model_name = os.path.basename(self.model_name.rstrip('/'))
        
        try:
            bbh_results = evaluate_bbh_consistency(
                model=model, 
                tokenizer=tokenizer, 
                root=root, 
                device='cuda',
                model_name=clean_model_name
            )
            
            if bbh_results and 'error' not in bbh_results:
                print(f"BBH consistency evaluation completed.")
                print(f"Average Accuracy: {bbh_results['avg_accuracy']:.4f} ({bbh_results['avg_accuracy']*100:.2f}%)")
                print(f"Overall Inconsistency: {bbh_results['inconsistency_scores']['overall_inconsistency']:.4f}")
                print(f"Results saved to: {root}/bbh_consistency/")
            else:
                print(f"BBH consistency evaluation failed: {bbh_results.get('error', 'Unknown error')}")
                
        except Exception as e:
            print(f"BBH consistency evaluation failed with exception: {e}")
        
        torch.cuda.empty_cache()

        # -----------------------------Open-Form Consistency----------------------------- 
        # Open-Form Consistency Evaluation
        try:
            print("Starting Open-Form Consistency evaluation...")
            # 重新加载模型和分词器用于评估
            model, tokenizer = self._load_model_and_tokenizer(model_name)
            
            open_form_results = evaluate_open_form_consistency(
                model, tokenizer, 
                root=logger.get_root(), 
                device=self.device, 
                max_samples=500,  # 可以根据需要调整样本数量
                model_name=logger.get_model_name() if hasattr(logger, "get_model_name") else "unknown"
            )
            print(f"Open-Form Consistency evaluation completed: {open_form_results}")
        except Exception as e:
            print(f"Open-Form Consistency evaluation failed with exception: {e}")
        
        torch.cuda.empty_cache()
      
    def save(self, logger):
        logger.save_ckpt("model", self.model, self.use_lora)
        logger.save_ckpt("tokenizer", self.tokenizer, self.use_lora)

    def run(self, logger):
        if self.resume_path is None:
            self.init_model()
            print_trainable_parameters(self.model)
            self.init_dataset()  
            
            self.init_unlearner(logger)
            if self.unlearner:
                self.unlearner.train()
            self.save(logger)
            os.system(f"rm -rf {logger.get_root()}/unlearn_checkpoint")
            self.eval(logger)
        else:
            self.init_model()
            self.init_dataset()
            self.eval(logger)


def get(**kwargs):
    return Unlearn(**kwargs)

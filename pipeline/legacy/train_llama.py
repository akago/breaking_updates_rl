import os

# os.environ["WANDB_PROJECT"] = "huggingface"
# os.environ["WANDB_RUN_ID"] = "y4o3e025"
# os.environ["WANDB_RESUME"] = "must"
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"

from unsloth import FastLanguageModel
import torch
from pathlib import Path
from pipeline.types.metrics import Patcher
from pipeline.types.utils import extract_java_code
from pipeline.types.maven_error import MavenErrorLog, MavenErrorParser
from pipeline.types.rewards import reward_func_diff_dense, reward_func_diff_sparse, reward_check_format, reward_check_tag
from pipeline.constants.constants import RESOURCES_PATH, DATASET_DIFF_PATH, SYSTEM_PROMPT
from datetime import datetime, timezone

import json
import re
from datasets import Dataset
from datasets import load_dataset

max_seq_length = 7000
lora_rank = 16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/meta-Llama-3.1-8B-Instruct",
    # model_name = "/home/xchen6/breaking_updates_rl/results/sft/llama8b/merged",
    # model_name = "/home/xchen6/breaking_updates_rl/results/sft/llama3b/merged",
    # model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.9, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_dropout = 0,
    bias = "none",
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
    random_state = 3407,
)

RESOURCES_PATH = Path(__file__).parent.parent/ "data" / "dataset"
DATASET_PATH = DATASET_DIFF_PATH / "dataset.json"
my_dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")

def keep_batch(batch):
    # return [tokenizer(prompt, return_tensors = "pt")["input_ids"].shape[-1] < 5000 and tokenizer(prompt, return_tensors = "pt")["input_ids"].shape[-1] + tokenizer(errors, return_tensors = "pt")["input_ids"].shape[-1] < 7000  for prompt, errors in zip(batch["prompt"], batch["errors"])]
    return [tokenizer(SYSTEM_PROMPT+t, return_tensors = "pt")["input_ids"].shape[-1] < 5000 for t in batch["prompt"]]

my_dataset = my_dataset.filter(keep_batch, batched=True, batch_size=10_000)
my_dataset = my_dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": x["prompt"]},
    ],
    "breakingCommit": x["breakingCommit"],
    "project": x["project"],
    "absolute_path_to_file_in_container": x["absolute_path_to_file_in_container"],
    "errors" : x["errors"],
    "original_code": x["original_code"],
})

max_prompt_length = 5000

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "paged_adamw_8bit",
    # optim="adamw_torch_fused",
    logging_steps = 1,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 2, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 500,
    save_steps = 100,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    vllm_enable_sleep_mode=True,
    reward_weights=[0.85, 0.1, 0.05],
    output_dir = "/home/xchen6/breaking_updates_rl/results/rl/llama8b_dense_new",
)

def main():
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
        reward_func_diff_dense,
        reward_check_format,
        reward_check_tag,
        ],
        args = training_args,
        train_dataset = my_dataset,
    )
    trainer.train()
    
    # trainer.train(resume_from_checkpoint="/home/xchen6/breaking_updates_rl/results/grpo_llama/checkpoint-30")
    model.save_pretrained("/home/xchen6/breaking_updates_rl/results/rl/llama8b_dense_new")
    tokenizer.save_pretrained("/home/xchen6/breaking_updates_rl/results/rl/llama8b_dense_new")
    
if __name__ == "__main__":
    main()
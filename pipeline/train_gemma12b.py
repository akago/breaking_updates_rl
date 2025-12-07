import os
# os.environ["WANDB_PROJECT"] = "huggingface"
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"

os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"

from unsloth import FastModel
import torch
from pathlib import Path
from pipeline.types.metrics import Patcher
from pipeline.types.utils import extract_java_code
from pipeline.types.rewards import reward_func_diff_dense, reward_func_diff_sparse, reward_check_format, reward_check_tag
from pipeline.types.maven_error import MavenErrorLog, MavenErrorParser
from pipeline.constants.constants import RESOURCES_PATH, DATASET_DIFF_PATH, DATASET_FULL_GENERATION_PATH, SYSTEM_PROMPT
from datetime import datetime, timezone

import json
import re
from datasets import Dataset
from datasets import load_dataset

max_seq_length = 7000

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",

    # Other popular models!
    "unsloth/Llama-3.1-8B",
    "unsloth/Llama-3.2-3B",
    "unsloth/Llama-3.3-70B",
    "unsloth/mistral-7b-instruct-v0.3",
    "unsloth/Phi-4",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    # model_name = "/home/xchen6/breaking_updates_rl/results/sft/sft_gemma4b/3_epoch",
    model_name = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    fast_inference = True,
    gpu_memory_utilization = 0.8,
    # attn_implementation="flash_attention_2"
    # token = "hf_...", # use one if using gated models
)

model = FastModel.get_peft_model(
    model,
    finetune_vision_layers     = False, # Turn off for just text!
    finetune_language_layers   = True,  # Should leave on!
    finetune_attention_modules = True,  # Attention good for GRPO
    finetune_mlp_modules       = True,  # SHould leave on always!
    # use_gradient_checkpointing = "unsloth",
    r = 16,           # Larger = higher accuracy, but might overfit
    lora_alpha = 32,  # Recommended alpha == r at least
    lora_dropout = 0.05,
    bias = "none",
    random_state = 2,
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

# from vllm import SamplingParams
# vllm_sampling_params = SamplingParams(
#     min_p = 0.1,
#     top_p = 1.0,
#     top_k = -1,
#     seed = 3407,
#     stop = [tokenizer.eos_token],
#     include_stop_str_in_output = True,
# )

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    temperature = 1.0,
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_torch_fused",
    logging_steps = 1,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 2, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 500, # Increase for better results
    save_steps = 20,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    vllm_enable_sleep_mode=True,
    reward_weights=[0.85, 0.1, 0.05],
    output_dir = "/home/xchen6/breaking_updates_rl/results/rl/gemma12b_dense",
)

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
# trainer.train(resume_from_checkpoint="/home/xchen6/breaking_updates_rl/results/sft/sft_gemma4b/checkpoint-75")
model.save_pretrained("/home/xchen6/breaking_updates_rl/results/rl/gemma12b_dense")
tokenizer.save_pretrained("/home/xchen6/breaking_updates_rl/results/rl/gemma12b_dense")
import os

os.environ["WANDB_PROJECT"] = "huggingface"
os.environ["WANDB_RUN_ID"] = "y4o3e025"
os.environ["WANDB_RESUME"] = "must"
os.environ["WANDB_LOG_MODEL"] = "checkpoint"
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"

from unsloth import FastLanguageModel
import torch
from pathlib import Path
from pipeline.types.metrics import Patcher
from pipeline.types.utils import extract_java_code
from pipeline.types.maven_error import MavenErrorLog, MavenErrorParser
from datetime import datetime, timezone

import json
import re
from datasets import Dataset
from datasets import load_dataset

max_seq_length = 10000
lora_rank = 16

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/meta-Llama-3.1-8B-Instruct",
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
    lora_alpha = lora_rank * 2,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 2,
)

RESOURCES_PATH = Path(__file__).parent.parent/ "data" / "dataset"
DATASET_PATH = Path(__file__).parent.parent / "data" / "prompts" / "dataset.json"
my_dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")

def keep_batch(batch):
    # return [tokenizer(prompt, return_tensors = "pt")["input_ids"].shape[-1] < 5000 and tokenizer(prompt, return_tensors = "pt")["input_ids"].shape[-1] + tokenizer(errors, return_tensors = "pt")["input_ids"].shape[-1] < 7000  for prompt, errors in zip(batch["prompt"], batch["errors"])]
    return [tokenizer(t, return_tensors = "pt")["input_ids"].shape[-1] < 5000 for t in batch["prompt"]]

my_dataset = my_dataset.filter(keep_batch, batched=True, batch_size=10000)
my_dataset = my_dataset.map(lambda x: {
    "prompt" : [
        {"role": "user",   "content": x["prompt"]},
    ],
    "breakingCommit": x["breakingCommit"],
    "project": x["project"],
    "absolute_path_to_file_in_container": x["absolute_path_to_file_in_container"],
    "errors" : x["errors"]
})
    

def ensure_single_trailing_newline(s: str) -> str:
    return s.rstrip('\r\n') + '\n'

def extract_java_code_llama(text: str) -> str:
    end_marker = "<|eot_id|>"
    if end_marker and end_marker in text:
        text = text.split(end_marker)[0]
    JAVA_BLOCK = re.compile(
        r"```java[^\n\r]*\r?\n(.*?)(?=\r?\n?```)",
        flags=re.DOTALL | re.IGNORECASE
    )
    m = JAVA_BLOCK.search(text)
    # Prevent "File does not end with a newline" error
    return ensure_single_trailing_newline(m.group(1)) if m else ""
   
def package_removed(patch:str):
    BLOCK = re.compile(r'/\*.*?\*/', re.S)
    PKG_LINE = re.compile(r'(?m)^(?!\s*//)\s*package\s+[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\s*;')
    # ignore the comments
    s = BLOCK.sub('', patch)
    return PKG_LINE.search(s) is None


def reward_func_dense(prompts, completions, completion_ids, breakingCommit, project, absolute_path_to_file_in_container, errors, **kwargs):
    """
    if package decalaration removed: penalty -0.6
    
    """    
    print(f"Prompt lengths for this batch: {[tokenizer(prompt[0]['content'], return_tensors = "pt")['input_ids'].shape for prompt in prompts]}")
    print(f"Completion lengths for this batch: {[len(completion_id) for completion_id in completion_ids]}")
    rewards = []
    
    for completion, breaking_commit, project_name, file_path_in_container, maven_errors in zip(completions, breakingCommit, project, absolute_path_to_file_in_container, errors):
        reward = 0.0
        response = completion[0]["content"]
        patch = extract_java_code_llama(response)
        # Failed to generate proper format
        if patch == "":
            rewards.append(-1.0)
            continue
        # Deleting package declaration should be punished
        if package_removed(patch):
            reward -= 0.6
            
        # Logging with timestamp
        print(f"INFO {datetime.now(timezone.utc)} Compiling the breaking commit {breaking_commit} of project {project_name} to get the rewards...")
        print(f"INFO {datetime.now(timezone.utc)} The Patch: {patch}")
        
        # Get the build log for patched project
        container_path = RESOURCES_PATH / str(breaking_commit) / f"{breaking_commit}.sif"
        patcher = Patcher(project=project_name, container_path=str(container_path))
        build_log, success = patcher.apply_patch_training(patch=patch, container_file=file_path_in_container)
        
        # Test only if the compilation success
        if success:
            _, test_success = patcher.apply_patch_training_test(patch=patch, container_file=file_path_in_container)
            if not test_success:
                rewards.append(0.5) 
            else:
                rewards.append(1.0)
            continue
        
        # Parse Maven Errors
        log_parser = MavenErrorParser()
        error_log = MavenErrorLog.from_string(build_log, log_parser)
        original_error_count, fixed_error_count, new_errors_count = patcher.get_metrics({file_path_in_container : maven_errors}, error_log.to_jsonable())
        
        # Calculate the reward regarding errors
        reward += float(fixed_error_count / original_error_count) - float(new_errors_count / original_error_count)
        
        rewards.append(reward)
        print(f"INFO {datetime.now(timezone.utc)} Reward: {rewards[-1]}")
    return rewards

max_prompt_length = 5000

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    # optim = "paged_adamw_8bit",
    optim="adamw_torch_fused",
    logging_steps = 1,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4, # Increase to 4 for smoother training
    num_generations = 2, # Decrease if out of memory
    max_prompt_length = max_prompt_length,
    max_completion_length = max_seq_length - max_prompt_length,
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 200,
    save_steps = 5,
    max_grad_norm = 0.1,
    report_to = "wandb", # Can use Weights & Biases
    vllm_enable_sleep_mode=True,
    output_dir = "/home/xchen6/breaking_updates_rl/results/grpo_llama_1",
)

def main():
    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
        reward_func_dense,
        ],
        args = training_args,
        train_dataset = my_dataset,
    )
    trainer.train()
    
    # trainer.train(resume_from_checkpoint="/home/xchen6/breaking_updates_rl/results/grpo_llama/checkpoint-30")
    model.save_pretrained("/home/xchen6/breaking_updates_rl/results/grpo_llama/lora_model")
    tokenizer.save_pretrained("/home/xchen6/breaking_updates_rl/results/grpo_llama/lora_model")
    
if __name__ == "__main__":
    main()
import os
from tqdm import tqdm
import torch
from pathlib import Path

# utils
from pipeline.types.metrics import Patcher
from pipeline.types.utils import extract_java_code

# unsloth standby optimization for memory efficiency
os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
import unsloth
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

# transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import (
    GRPOConfig,
    GRPOTrainer,
)


RESOURCES_PATH = Path(__file__).parent.parent/ "data" / "dataset"
DATASET_PATH = Path(__file__).parent.parent / "data" / "prompts" / "dataset.json"
# nf4_config = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type="nf4",
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_compute_dtype=torch.bfloat16
# )

def reward_func_dense(completions, **kwargs):

        def apply_patch_and_compute_reward(patch, extra_info)->float:
            breakingCommit = extra_info["breakingCommit"]
            project_name = extra_info["project"]
            container_path = RESOURCES_PATH / str(breakingCommit) / f"{breakingCommit}.sif"
            # Apply the patch inside the container
            patcher = Patcher(project=project_name, container_path=str(container_path))
            errorlog, success = patcher.apply_patch_training(patch=patch, container_file=extra_info["absolute_path_to_file_in_container"])
            return patcher.reward_dense({extra_info["absolute_path_to_file_in_container"] : extra_info["errors"]}, errorlog.to_jsonable(), success)
        
        rewards = []
        required_per_sample_keys = [
            "breakingCommit", "project", "absolute_path_to_file_in_container", "errors"
        ]
        print("get into reward function")
        for i, completion in enumerate(completions):
            response = completion[0]["content"]
            print(f"the {i}-th response: {response}")
            print(f"")
            patch = extract_java_code(response)
            if patch == "":
                rewards.append(0.0)
                continue
            
            extra_info = {k: kwargs[k][i] for k in required_per_sample_keys}
            rewards.append(apply_patch_and_compute_reward(patch, extra_info))
        return rewards

max_seq_length = 13000

def main():    
    # Use a pipeline as a high-level helper
    model, tokenizer = FastModel.from_pretrained(
        model_name = "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
        max_seq_length = max_seq_length, # Choose any for long context!
        load_in_4bit = True,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
        gpu_memory_utilization = 0.4,
        fast_inference = True,
        
        # device_map = "balanced",
        # token = "hf_...", # use one if using gated models
    )
    
    # applies chat template
    # tokenizer = get_chat_template(
    #         tokenizer,
    #         chat_template = "gemma-3", # change this to the right chat_template name
    #     )
    
    def format_message(item:dict):
        message = [
            {
                "role": "user", 
                "content": item["prompt"]
            }
        ]
        # item["prompt"] = tokenizer.apply_chat_template(message, tokenize = False, add_generation_prompt = False).removeprefix('<bos>')
        return item
    
    def keep_batch(batch):
        return [tokenizer(t, return_tensors = "pt")["input_ids"].shape[-1] <= 5000 for t in batch["prompt"]]

    dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")
    dataset = dataset.filter(keep_batch, batched=True, batch_size=10_000)
    dataset = dataset.map(lambda x: {
        "prompt" : [
            {"role": "user",   "content": x["prompt"]},
        ],
        "breakingCommit": x["breakingCommit"],
        "project": x["project"],
        "absolute_path_to_file_in_container": x["absolute_path_to_file_in_container"],
        "errors" : x["errors"]
    })
    
    # model.config.use_cache = False
    # model.gradient_checkpointing_enable()

    # try:
    #     model.config.attn_implementation = "flash_attention_2"
    #     print("Flash-Attention 2 enabled")
    # except Exception:
    #     model.config.attn_implementation = "sdpa"
    #     print("Using SDPA")

    
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # Turn off for just text!
        finetune_language_layers   = True,  # Should leave on!
        finetune_attention_modules = True,  # Attention good for GRPO
        finetune_mlp_modules       = True,  # SHould leave on always!
        r = 8,           # Larger = higher accuracy, but might overfit
        lora_alpha = 16,  # Recommended alpha == r at least
        lora_dropout = 0.0,
        bias = "none",
        random_state = 2,
        use_gradient_checkpointing = "unsloth",
    )
    
    max_prompt_length = 5000
    
    
    training_args = GRPOConfig(
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
        max_steps = 2, # Increase for better results
        save_steps = 1,
        max_grad_norm = 0.1,
        report_to = "wandb", # Can use Weights & Biases
        output_dir = "/home/xchen6/breaking_updates_rl/results/fine_tuned_models",
    )

    trainer = GRPOTrainer(
        model = model,
        processing_class = tokenizer,
        reward_funcs = [
            reward_func_dense
        ],
        args = training_args,
        train_dataset = dataset,
    )
    trainer.train()
    
if __name__ == "__main__":
    main()
        
    
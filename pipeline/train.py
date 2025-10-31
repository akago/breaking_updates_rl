import os
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    
)

from pipeline.types.metrics import Patcher
from pipeline.types.utils import extract_java_code

from transformers import BitsAndBytesConfig
from unsloth import FastModel
from trl import GRPOConfig, GRPOTrainer

max_seq_length = 128000

nf4_config = BitsAndBytesConfig(
   load_in_4bit=True,
   bnb_4bit_quant_type="nf4",
   bnb_4bit_use_double_quant=True,
   bnb_4bit_compute_dtype=torch.bfloat16
)

def reward_func_dense(completions, **kwargs):
        def apply_patch_and_compute_reward(completion, **info):
            breakingCommit = info["breakingCommit"]
            project_name = info["project"]
            container_path = RESOURCES_PATH / str(breakingCommit) / f"{breakingCommit}.sif"
            # Apply the patch inside the container
            patcher = Patcher(project=project_name, container_path=container_path)
            errorlog, success = patcher.apply_patch_training(patch=completion, container_file=info["absolute_path_to_file_in_container"])
            return patcher.reward_dense({info["absolute_path_to_file_in_container"] : info["errors"]}, errorlog.to_jsonable(), success)
        
        rewards = []
        for i, c in enumerate(completions):
            patch = extract_java_code(c)
            if patch == "":
                rewards.append(0.0)
                continue
            extra_param = {k: v[i] for k, v in kwargs.items()}  # get extra metadata
            rewards.append(apply_patch_and_compute_reward(c, **extra_param))
        return rewards


def format_message(item:dict):
    item["prompt"] = [{"role": "user", "content": item["prompt"]}]
    return item

def main():
    RESOURCES_PATH = Path(__file__).parent.parent/ "data" / "dataset"
    DATASET_PATH = Path(__file__).parent.parent / "data" / "prompts" / "dataset.json"
    dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")
    dataset.map(format_message)
    # Use a pipeline as a high-level helper
    model, tokenizer = FastModel.from_pretrained(
        model_name = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
        max_seq_length = max_seq_length, # Choose any for long context!
        load_in_4bit = True,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
        # token = "hf_...", # use one if using gated models
    )
    
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False, # Turn off for just text!
        finetune_language_layers   = True,  # Should leave on!
        finetune_attention_modules = True,  # Attention good for GRPO
        finetune_mlp_modules       = True,  # SHould leave on always!

        r = 16,           # Larger = higher accuracy, but might overfit
        lora_alpha = 32,  # Recommended alpha == r at least
        lora_dropout = 0.05,
        bias = "none",
        random_state = 2,
    )
    
    max_prompt_length = 60000
      
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
        num_generations = 4, # Decrease if out of memory
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
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
            reasoning_quality_reward_func
        ],
        args = training_args,
        train_dataset = train_dataset,
    )
    trainer.train()
    
if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    ################
    # Model
    ################
    dtype = model_args.dtype if model_args.dtype in ["auto", None] else getattr(torch, model_args.dtype)
    training_args.model_init_kwargs = dict(
        revision=model_args.model_revision,
        attn_implementation=model_args.attn_implementation,
        dtype=dtype,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        training_args.model_init_kwargs["device_map"] = get_kbit_device_map()
        training_args.model_init_kwargs["quantization_config"] = quantization_config
        
        
    
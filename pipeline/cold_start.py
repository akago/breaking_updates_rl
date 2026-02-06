import os
import sys
import argparse
# os.environ["WANDB_PROJECT"] = "huggingface"
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"

# os.environ["UNSLOTH_VLLM_STANDBY"] = "1"
os.environ["UNSLOTH_ENABLE_LOGGING"] = "1"

os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"
from unsloth import FastModel
from unsloth.chat_templates import standardize_data_formats, get_chat_template
import torch
from pathlib import Path
from pipeline.types.metrics import Patcher
from pipeline.types.utils import extract_java_code
from pipeline.types.maven_error import MavenErrorLog, MavenErrorParser
from pipeline.constants.constants import SFT_DATASET_PATH , SYSTEM_PROMPT
from datetime import datetime, timezone

import json
import re
from datasets import Dataset
from datasets import load_dataset



max_seq_length = 15000

def cold_start(model, output, epoch):
    model, tokenizer = FastModel.from_pretrained(
        model_name = model,
        max_seq_length = max_seq_length, # Choose any for long context!
        load_in_4bit = True,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
        # fast_inference = True,
        # attn_implementation="flash_attention_2"
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-3",
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


    DATASET_PATH = SFT_DATASET_PATH / "sft_data.jsonl"
    my_dataset = load_dataset("json", data_files=str(DATASET_PATH), split="train")
    def keep_batch(batch):
        return batch["accepted"]

    my_dataset = my_dataset.filter(keep_batch, batched=True)
    print(f"dataset left: {len(my_dataset)}")
    my_dataset = my_dataset.map(lambda x: {
        "conversations" : [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": x["prompt"]},
            {"role": "assistant", "content" : x["response_text"]},
        ],
    })
    # print(my_dataset[50])
    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [tokenizer.apply_chat_template(convo, tokenize = False, add_generation_prompt = False).removeprefix('<bos>') for convo in convos]
        return { "text" : texts, }
    my_dataset = my_dataset.map(formatting_prompts_func, batched = True)
    # print(my_dataset[50]["text"])

    max_prompt_length = 10000
    from trl import SFTTrainer, SFTConfig
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = my_dataset,
        eval_dataset = None, # Can set up evaluation!
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 8, # Use GA to mimic batch size!
            warmup_steps = 5,
            num_train_epochs = epoch, # Set this for 1 full training run.
            # max_steps = None,
            # save_steps=10,
            learning_rate = 1e-4, # Reduce to 2e-5 for long training runs
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            report_to = "wandb", # Use TrackIO/WandB etc
            output_dir = str(output),
            # vllm_enable_sleep_mode=True,
        ),
    )

    from unsloth.chat_templates import train_on_responses_only
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
    )

    # print(tokenizer.decode(trainer.train_dataset[99]["input_ids"]))
    # print(tokenizer.decode([tokenizer.pad_token_id if x == -100 else x for x in trainer.train_dataset[99]["labels"]]).replace(tokenizer.pad_token, " "))
    # model.save_pretrained("/home/xchen6/breaking_updates_rl/results/sft_gemma4b/lora_model")
    # tokenizer.save_pretrained("/home/xchen6/breaking_updates_rl/results/sft_gemma4b/lora_model")

    trainer_stats = trainer.train()
    model.save_pretrained(str(output / f"{epoch}_epoch"))  # Local saving
    tokenizer.save_pretrained(str(output / f"{epoch}_epoch"))
    model.save_pretrained_merged(str(output / "merged"), tokenizer, save_method = "merged_16bit",)
    

def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="evalute the completions")
    parser.add_argument("--output", "-o", type=Path,
                        default=Path(__file__).parent.parent / "results" / "sft_gemma4b",
                        help="Path to result folder containing completions")
    parser.add_argument("--model", "-m", type=str,
                        default="unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
                        help="model name or path")
    parser.add_argument("--epoch", "-e", type=int,
                        default=3,
                        help="number of epoch")
    args = parser.parse_args(argv)
    cold_start(args.model, args.output, args.epoch)

if __name__ == "__main__":
    main()
    
    

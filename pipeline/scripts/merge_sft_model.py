
import os
# os.environ["WANDB_PROJECT"] = "huggingface"
# os.environ["WANDB_LOG_MODEL"] = "checkpoint"

from unsloth import FastModel
import torch
from pathlib import Path
from pipeline.types.metrics import Patcher
from pipeline.types.utils import extract_java_code
from pipeline.types.rewards import reward_func_diff_dense, reward_func_diff_sparse, reward_check_format, reward_check_tag
from pipeline.types.maven_error import MavenErrorLog, MavenErrorParser
from pipeline.constants.constants import RESOURCES_PATH, DATASET_DIFF_PATH, DATASET_FULL_GENERATION_PATH
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
    model_name = "/home/xchen6/breaking_updates_rl/results/sft/sft_gemma4b/3_epoch",
    # model_name = "/home/xchen6/breaking_updates_rl/results/sft/llama8b/3_epoch",
    # model_name = "/home/xchen6/breaking_updates_rl/results/sft/gemma12b/2_epoch",
    # model_name = "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    max_seq_length = max_seq_length, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
    # full_finetuning = False, # [NEW!] We have full finetuning now!
    # attn_implementation="flash_attention_2"
    # token = "hf_...", # use one if using gated models
)

model.save_pretrained_merged("/home/xchen6/breaking_updates_rl/results/sft/sft_gemma4b/merged", tokenizer, save_method = "merged_16bit",)
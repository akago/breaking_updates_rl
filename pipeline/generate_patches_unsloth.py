# Load model directly

import os
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template

import argparse
import json
import sys
from datetime import datetime
import logging
from pathlib import Path
import re

import torch
from transformers import TextStreamer
from datasets import load_dataset

from pipeline.constants.constants import SYSTEM_PROMPT, RESOURCES_PATH
from pipeline.types.utils import extract_sr_edits, get_patched_content_from_diffs

logger = logging.getLogger(__name__)


max_seq_length = 35000

def evaluate(input: str, model_id: str) -> None:
    output_root = Path("/home/xchen6/breaking_updates_rl/results") / "_".join([model_id, datetime.now().strftime("%Y%m%d-%H%M%S")])
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_id,
        # model_name = "/home/xchen6/breaking_updates_rl/results/grpo_gemma4b/checkpoint-200",
        max_seq_length = max_seq_length, # Choose any for long context!
        load_in_4bit = True,  # 4 bit quantization to reduce memory
        load_in_8bit = False, # [NEW!] A bit more accurate, uses 2x memory
        full_finetuning = False, # [NEW!] We have full finetuning now!
    )

    print(f"Generating patches with {model_id}")
    
    test_ds = load_dataset(
        "json",
        data_files={"test": input},
        split="test",
    )
    # from here
    for i, data in enumerate(test_ds):
        print(f"Processing {i}-th sample")
        buggy_file_name = Path(data["absolute_path_to_file_in_container"]).stem
        output_path = output_root / data["breakingCommit"] / f"{i+1}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            print(f'generating patch for {data["project"]}/{data["breakingCommit"]}/{buggy_file_name}')
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": data["prompt"]}
            ]
            text = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt = True, # Must add for generation
                tokenize = False,
            )
            
            input_len = tokenizer(text, return_tensors = 'pt')['input_ids'].shape[-1]
            print(f"input len:{input_len}")
            
            completion = ""
            if input_len > 32000:
                completion = ""
            else:
                response = model.generate(
                    **tokenizer(text, return_tensors = "pt").to("cuda"),
                    max_new_tokens = 4000, # Increase for longer outputs!
                    # 0 temperature
                    do_sample=False,
                    # streamer = TextStreamer(tokenizer, skip_prompt = True),
                )
                completion_ids = response[0][input_len:]              
                completion = tokenizer.decode(completion_ids, skip_special_tokens=True)
                
            patch = ""
            original_code = data["original_code"]
            print(f"original code:{original_code}")
            diffs = extract_sr_edits(completion)
            print(f"diffs extracted: {diffs}")
            patch = get_patched_content_from_diffs(diffs, original_code)
            
            result = data.copy()
            result["patch"] = patch
            result["model"] = model_id
            result["raw_completion"] = completion
            with open(str(output_path), "w") as f:
                json.dump(result, f, indent=4)
        except Exception as e:
            print(e)
            

def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="evalute the dataset")
    parser.add_argument("--input", "-i", type=str,
                        default="/home/xchen6/breaking_updates_rl/data/prompts_diff/test.jsonl",
                        help="Path to test.jsonl")
    parser.add_argument("--model", "-m", type=str,
                        default="unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
                        help="The unsloth model name or path")
    
    args = parser.parse_args(argv)
    evaluate(args.input, args.model)
    
    

if __name__ == "__main__":
    main()
    
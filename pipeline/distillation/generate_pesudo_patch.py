from __future__ import annotations
import os
import sys
from datasets import load_dataset
from openai import OpenAI
from pathlib import Path
from pipeline.constants.constants import SYSTEM_PROMPT, RESOURCES_PATH, DATASET_DIFF_PATH, SFT_DATASET_PATH
from pipeline.types.utils import extract_sr_edits, get_patched_content_from_diffs
from pipeline.types.metrics import Patcher
from pipeline.types.maven_error import MavenErrorLog, MavenErrorParser
from dotenv import load_dotenv
import json
import unittest

load_dotenv() # OPENAI KEY

SFT_JSON_PATH = SFT_DATASET_PATH / "sft_data.jsonl"

def write_jsonl(rows: dict):
    target = SFT_JSON_PATH
    tmp = target.with_suffix(".jsonl.tmp")
    def _key(k, v):
        return v.get("id", k) if isinstance(v, dict) else k

    with open(tmp, "w", encoding="utf-8") as f:
        for k, v in sorted(rows.items(), key=lambda kv: _key(*kv)):
            f.write(json.dumps(v, ensure_ascii=False) + "\n")

    os.replace(tmp, target) 


def acceptable(raw_response: str, prompt: dict) -> tuple[bool, list]:
    breaking_commit = prompt["breakingCommit"]
    project_name = prompt["project"]
    absolute_file_path_in_container = prompt["absolute_path_to_file_in_container"]
    container_path = RESOURCES_PATH / breaking_commit / f"{breaking_commit}.sif"
    patcher = Patcher(project=project_name, container_path=str(container_path))
    # original_code_path = RESOURCES_PATH / breaking_commit / project_name / 
    original_code = prompt["original_code"]

    print(f"original code:{original_code}")
    diffs = extract_sr_edits(raw_response)
    print(f"diffs extracted: {diffs}")
    patch = get_patched_content_from_diffs(diffs, original_code)
    print(f"patch filled: {patch}")
    build_log, success = patcher.apply_patch_training(patch, container_file=absolute_file_path_in_container)

    log_parser = MavenErrorParser()
    errors_by_file = MavenErrorLog.from_string(build_log, log_parser).to_jsonable()
    if absolute_file_path_in_container in errors_by_file:
        print(f"{errors_by_file[absolute_file_path_in_container]}")
        return False, errors_by_file[absolute_file_path_in_container]
    else:
        return True, []


def load_rows_jsonl(path: Path) -> dict:
    rows= {}
    if not path.exists():
        return rows
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return rows
    # load jsonl
    ok = False
    try:
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            rid = obj.get("id")
            if rid is None:
                continue
            rows[rid] = obj
        ok = True
    except json.JSONDecodeError:
        ok = False
        
    return rows

def main():
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI()
    train = load_dataset("json", data_files=str(DATASET_DIFF_PATH / "train.jsonl"), split="train")

    rows = load_rows_jsonl(SFT_JSON_PATH)
    success_label_count = sum(1 for r in rows.values() if r.get("accepted"))
        
    max_attempt = 5
    # 2) Rejection sampling
    for attempt in range(1, max_attempt+1):
        for idx, sample in enumerate(train):
            if f"prompt-{idx}" in rows and  rows[f"prompt-{idx}"]["accepted"]:
                continue            
            resp = client.responses.create(
                model="gpt-5",
                input=SYSTEM_PROMPT + sample["prompt"],
            )
            raw_response = getattr(resp, "output_text", "") or ""
            accept, errors = acceptable(raw_response, sample)
            
            row = {
                "id": f"prompt-{idx}",
                "model": "gpt-5",
                "prompt": sample["prompt"],  # Instruction
                "response_text": raw_response, # CoT Data
                "accepted": accept, # 
                "errors": errors,
                "original_errors": sample['errors'],
            }
            if accept:
                rows[f"prompt-{idx}"] = row
                write_jsonl(rows)
                success_label_count += 1
                if success_label_count >= 200:
                    print("Reached 200 samples!")
                    sys.exit(0)
            elif f"prompt-{idx}" in rows and len(rows[f"prompt-{idx}"]["errors"]) < len(errors):
                continue
            else:
                rows[f"prompt-{idx}"] = row
                write_jsonl(rows)
            

if __name__ == "__main__":
    main()
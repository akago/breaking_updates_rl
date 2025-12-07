from pathlib import Path
import json
import os
from pipeline.types.utils import extract_sr_edits, get_patched_content_from_diffs
input = Path("/home/xchen6/breaking_updates_rl/results/grpo/gemma12b_sparse/checkpoint-245_20251107-121945")

for file in input.rglob("*.json"):
    print(f"Processing file: {file}")
    content = json.loads(file.read_text())
    raw_completion = content["raw_completion"]
    original_code = content["original_code"]
    # Replace all occurrences of the old format with the new format
    diffs = extract_sr_edits(raw_completion)
    print(f"Extracted diffs: {diffs}")
    updated_content = get_patched_content_from_diffs(diffs, original_code)
    
    if updated_content == content["patch"]:
        print("No changes detected, skipping update.")
        continue
    else:
        print("Changes detected, updating patch.")
    content["patch"] = updated_content
    # write back to the file but only when the write is successful
    target = file
    tmp = target.with_suffix(".json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(content, ensure_ascii=False) + "\n")
    os.replace(tmp, target) 
    print(f"Updated file: {file}")
    
    
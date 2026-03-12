#!/usr/bin/env python3
"""
Filter SFT training data to only keep entries whose
`absolute_path_to_file_in_container` appears in a CSV with build_success=True.

Inputs:
  - pipeline/distillation/data1_with_build_success.csv
  - data/sft/sft_data_train.jsonl
Output:
  - data/sft/sft_data_train_build_success.jsonl
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

CSV_PATH = Path("pipeline/distillation/data1_with_build_success.csv")
INPUT_JSONL = Path("data/sft/sft_data_train.jsonl")
OUTPUT_JSONL = Path("data/sft/sft_data_train_build_success.jsonl")


def load_success_paths(csv_path: Path) -> set[str]:
    """Return set of file paths whose build_success flag is True-ish."""
    success_paths: set[str] = set()
    true_values = {"true", "1", "yes", "y"}

    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            val = row.get("build_success", "")
            is_success = isinstance(val, str) and val.strip().lower() in true_values
            if is_success:
                path = row.get("absolute_path_to_file_in_container")
                if path:
                    success_paths.add(path)
    return success_paths


def filter_jsonl(input_path: Path, output_path: Path, keep_paths: set[str]) -> tuple[int, int]:
    """Stream filter jsonl, keeping only records whose path is in keep_paths."""
    total = kept = 0
    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            total += 1
            obj = json.loads(line)
            if obj.get("absolute_path_to_file_in_container") in keep_paths:
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                kept += 1
    return total, kept


def main() -> None:
    success_paths = load_success_paths(CSV_PATH)
    total, kept = filter_jsonl(INPUT_JSONL, OUTPUT_JSONL, success_paths)
    print(f"Total input rows: {total}")
    print(f"Kept with build_success=True: {kept}")
    print(f"Wrote: {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()

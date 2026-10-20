#!/usr/bin/env python3
"""
Filter training data by keys listed in train_df_analysis.csv.

Match key:
  - breakingCommit
  - absolute_path_to_file_in_container

Inputs:
  - analysis/data/train_df_analysis.csv
  - data/sft/sft_data_train.jsonl (jsonl) OR
    data/prompts/dataset.json (json)
Output:
  - analysis/data/train_df_analysis.json
"""

from __future__ import annotations

import csv
import json
import argparse
from pathlib import Path
from typing import Any

CSV_PATH = Path("analysis/data/train_df_analysis.csv")
INPUT_DATA = Path("data/sft/sft_data_train.jsonl")
OUTPUT_JSON = Path("analysis/data/test_set.json")


def load_target_keys(csv_path: Path) -> set[tuple[str, str]]:
    """Load (breakingCommit, absolute_path_to_file_in_container) keys from CSV."""
    keys: set[tuple[str, str]] = set()
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            commit = (row.get("breakingCommit") or "").strip()
            path = (row.get("absolute_path_to_file_in_container") or "").strip()
            if commit and path:
                keys.add((commit, path))
    return keys


def load_input_records(input_path: Path) -> list[dict[str, Any]]:
    """
    Load records from:
      - JSONL: one JSON object per line
      - JSON: top-level list[dict]
      - JSON: top-level dict with list values (e.g. {'train': [...], 'test': [...]})
    """
    if input_path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with input_path.open() as fin:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
        return records

    with input_path.open() as fin:
        data = json.load(fin)

    if isinstance(data, list):
        return [obj for obj in data if isinstance(obj, dict)]

    if isinstance(data, dict):
        records: list[dict[str, Any]] = []
        for value in data.values():
            if isinstance(value, list):
                records.extend(obj for obj in value if isinstance(obj, dict))
        return records

    raise ValueError(f"Unsupported input format: {input_path}")


def filter_sft_by_train_df_analysis(
    csv_path: Path,
    input_data_path: Path,
    output_json_path: Path,
) -> tuple[int, int]:
    """
    Filter input data by key pairs in train_df_analysis.csv.

    Returns:
      (total_input_rows, kept_rows)
    """
    keys = load_target_keys(csv_path)
    records = load_input_records(input_data_path)
    total = len(records)
    kept_records: list[dict] = []
    for obj in records:
        key = (
            (obj.get("breakingCommit") or "").strip(),
            (obj.get("absolute_path_to_file_in_container") or "").strip(),
        )
        if key in keys:
            kept_records.append(obj)

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w") as fout:
        json.dump(kept_records, fout, ensure_ascii=False, indent=2)

    return total, len(kept_records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter training records by train_df_analysis keys")
    parser.add_argument("--csv", type=Path, default=CSV_PATH, help="Path to train_df_analysis.csv")
    parser.add_argument("--input", type=Path, default=INPUT_DATA, help="Input data (.jsonl/.json)")
    parser.add_argument("--output", type=Path, default=OUTPUT_JSON, help="Output json path")
    args = parser.parse_args()

    total, kept = filter_sft_by_train_df_analysis(args.csv, args.input, args.output)
    print(f"Total input rows: {total}")
    print(f"Kept rows: {kept}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()

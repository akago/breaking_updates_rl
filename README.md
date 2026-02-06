## split_train_set.py quick guide

### What it does
- Loads BU SFT training data from `data/sft/sft_data_train_build_success.jsonl`.
- Splits the dataset either:
  - **By breaking change type** (file level) into one file for the chosen type, or
  - **Into N roughly equal chunks** (file level) for cross‑validation or sharding.
- Writes results to `experiment/` as JSONL files.

### Outputs
- By type: `experiment/train_<BC_TYPE>.jsonl` (e.g., `train_TYPE_REMOVED.jsonl`).
- By size: `experiment/train_<N>_splits_<k>.jsonl` where `k` is 1..N.
Each line is a JSON object representing a `BreakingUpdateSample`.

### Usage
- Split by breaking change type (e.g., TYPE_REMOVED):
  ```bash
  python pipeline/scripts/split_train_set.py --by_type -b TYPE_REMOVED
  ```
- Split into N chunks (e.g., 5):
  ```bash
  python pipeline/scripts/split_train_set.py -n 5
  ```

### Notes
- Shuffling is enabled with a fixed seed (42) for reproducibility.
- Output directory is created automatically (`experiment/`).

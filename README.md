## Dataset
- Location: `data/sft/` (file-level samples).
  - `sft_data_train.jsonl` – 265
  - `sft_data_test.jsonl` – 109
  - `sft_data_train_build_success.jsonl` – 68
- Common fields: `prompt`, `original_code`, `api_diff`, `buggy_lines`, `error_message`, `errors` (with `BCs`), `breakingCommit`, `file_success`, `build_success` etc.

### Note
- `file_success`: After applying the patch, recompilation shows that compilation errors in the the file are resolved.
- `build_success`: After applying the patch, the project recompiles successfully and the test suite passes.


### BC type distribution
- `bc_type_distribution_train_data.xlsx`: **file-level** BC Type distribution on the original train split. 

  Quick counts (#files with single BC Type): 

  | BC Type                                   | #samples |
  | ----------------------------------------- | -------- |
  | TYPE_REMOVED                              | 103      |
  | METHOD_REMOVED                            | 38       |
  | METHOD_NO_LONGER_THROWS_CHECKED_EXCEPTION | 3        |
  | SUPERTYPE_REMOVED                         | 2        |
  | METHOD_ADDED_TO_INTERFACE                 | 2        |
  | FIELD_REMOVED                             | 2        |

  

- `bc_type_distribution_full_data.xlsx`: **file-level** BC Type distribution on the full dataset. Quick counts (#files with single BC Type): 

  | BC Type                                   | #samples |
  | ----------------------------------------- | -------- |
  | TYPE_REMOVED                              | 165      |
  | METHOD_REMOVED                            | 52       |
  | METHOD_NO_LONGER_THROWS_CHECKED_EXCEPTION | 3        |
  | SUPERTYPE_REMOVED                         | 2        |
  | METHOD_ADDED_TO_INTERFACE                 | 2        |
  | FIELD_REMOVED                             | 3        |
  | METHOD_PARAMETER_GENERICS_CHANGED         | 1        |


## split_train_set.py quick guide

### What it does
- Loads training data from `data/sft/sft_data_train_build_success.jsonl`.
- Splits the dataset either:
  - **By breaking change type** (file level) into one file for the chosen type, or
  - **Into N roughly equal chunks** (file level) for cross‑validation or sharding.
- Writes results to `experiment/` as JSONL files.


### Outputs
- By type: `experiment/train_<BC_TYPE>.jsonl` (e.g., `train_TYPE_REMOVED.jsonl`).
- By size: `experiment/train_<N>_splits_<k>.jsonl` where `k` is 1..N.
Each line is a JSON object representing a `buggy file`.

### Usage
- Split by breaking change type (e.g., TYPE_REMOVED):
  ```bash
  python -m pipeline.scripts.split_train_set --by_type -b TYPE_REMOVED
  ```
- Split into N chunks (e.g., 5):
  ```bash
  python pipeline.scripts.split_train_set -n 5
  ```

### Notes
- Shuffling is enabled with a fixed seed (42) for reproducibility.
- Output directory is created automatically (`experiment/`).

# Breaking-Updates-Repair
End-to-end pipeline for fixing breaking Java dependency updates with large language models. The repo ships reproducible data creation, prompt construction, supervised fine-tuning, GRPO-based reinforcement learning, and containerized evaluation.

- **Real-world target**: Works on real-world Breaking Updates benchmark [BUMP](https://github.com/chains-project/bump).
- **Fully reproducible**: Scripts cover dataset building, prompt formatting, training, and Apptainer-based evaluation.
- **RL for code repair**: Designed rewards for GRPO training with Unsloth 4-bit quantization to keep GPU cost low.

## Repository map
- `pipeline/create_dataset.py`: Filter benchmark metadata/logs, download sources/dependencies, categorize build failures, and emit dataset + container definitions.
- `pipeline/build_prompt.py`: Build structured prompts with error lines, API diffs, and client code from `data/dataset/*/new_context.json`.
- `pipeline/scripts/`: Helpers for formatting (`formulate_dataset.py`), splitting (`split_dataset.py`), aggregating results, and plotting (`plot_*.py`).
- `pipeline/train_rl.py`: GRPO entrypoint using TRL + Unsloth 4-bit acceleration; supports dense and sparse rewards.
- `pipeline/eval_completions.py`: Apply patches in Apptainer, compile, run tests, and report project-level metrics.
- `data/`: Benchmark inputs (`benchmark/`, `successfulReproductionLogs/`), generated datasets/prompts, and visualization PNGs.

## Quickstart
Prereqs: Python 3.10+, Java 11, Maven, Apptainer. Python deps: `pip install -r requirements.txt`.

1) Build the dataset (place Breaking Updates metadata/logs under `data/benchmark` and `data/successfulReproductionLogs`):
```bash
python pipeline/create_dataset.py --input data --output data/dataset
```

2) Create prompts and train/test splits:
```bash
python pipeline/build_prompt.py --input data/dataset
python pipeline/scripts/formulate_dataset.py
python pipeline/scripts/split_dataset.py  # produces train.jsonl / test.jsonl
```

3) Train (example: GRPO with 4-bit Gemma and dense rewards):
```bash
python pipeline/train_rl.py \
  --model unsloth/gemma-3-4b-it-unsloth-bnb-4bit (or path to sft-fine-tuned model) \
  --dense True \
  --output_dir results/rl
```
Supervised fine-tuning first? Use prompts from `data/sft/` and the `cold_start*.py` scripts.

4) Evaluate generated patches (compilation + tests inside containers):
```bash
python pipeline/eval_completions.py --input results/<run_folder>
```
Outputs include build success rate, file/error repair rates, and relative error fix ratio.


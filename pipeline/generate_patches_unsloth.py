import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys
from datetime import datetime

from datasets import load_dataset

from pipeline.common.config_loader import get_cfg, load_yaml_config
from pipeline.constants.constants import SYSTEM_PROMPT
from pipeline.generation.backends import (
    GenerationRuntimeConfig,
    build_generation_backend,
)
from pipeline.types.utils import extract_sr_edits, get_patched_content_from_diffs

logger = logging.getLogger(__name__)

@dataclass
class GenerationConfig:
    input_path: Path
    model_id: str
    output_base_dir: Path
    backend: str = "unsloth"
    max_seq_length: int = 35000
    max_input_tokens: int = 32000
    max_new_tokens: int = 4000
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False


def _build_output_root(config: GenerationConfig) -> Path:
    # Keep historical naming behavior unchanged.
    return config.output_base_dir / "_".join([config.model_id, datetime.now().strftime("%Y%m%d-%H%M%S")])


def _write_run_manifest(output_root: Path, config: GenerationConfig) -> None:
    manifest = {
        "stage": "generation",
        "created_at": datetime.now().isoformat(),
        "config": {
            **asdict(config),
            "input_path": str(config.input_path),
            "output_base_dir": str(config.output_base_dir),
        },
    }
    with (output_root / "run_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


def evaluate(config: GenerationConfig) -> None:
    output_root = _build_output_root(config)
    output_root.mkdir(parents=True, exist_ok=True)
    _write_run_manifest(output_root, config)

    runtime_cfg = GenerationRuntimeConfig(
        model_id=config.model_id,
        max_seq_length=config.max_seq_length,
        max_input_tokens=config.max_input_tokens,
        max_new_tokens=config.max_new_tokens,
        load_in_4bit=config.load_in_4bit,
        load_in_8bit=config.load_in_8bit,
        full_finetuning=config.full_finetuning,
    )
    backend = build_generation_backend(config.backend, runtime_cfg)

    logger.info("Generating patches with %s (%s backend)", config.model_id, config.backend)
    
    test_ds = load_dataset(
        "json",
        data_files={"test": str(config.input_path)},
        split="test",
    )

    for i, data in enumerate(test_ds):
        logger.info("Processing sample %d", i)
        buggy_file_name = Path(data["absolute_path_to_file_in_container"]).stem
        output_path = output_root / data["breakingCommit"] / f"{i + 1}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info(
                "Generating patch for %s/%s/%s",
                data["project"],
                data["breakingCommit"],
                buggy_file_name,
            )
            
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": data["prompt"]},
            ]
            text = backend.build_prompt_text(messages)
            completion, input_len = backend.generate_completion(text)
            logger.info("input len: %d", input_len)
                
            patch = ""
            original_code = data["original_code"]
            diffs = extract_sr_edits(completion)
            logger.info("diffs extracted: %d edit blocks", len(diffs))
            patch = get_patched_content_from_diffs(diffs, original_code)
            
            result = data.copy()
            result["patch"] = patch
            result["model"] = config.model_id
            result["raw_completion"] = completion
            with output_path.open("w") as f:
                json.dump(result, f, indent=4)
        except Exception as e:
            logger.exception("Failed sample %d (%s): %s", i, output_path, e)
            

def _resolve(cli_value, cfg_value, default_value):
    if cli_value is not None:
        return cli_value
    if cfg_value is not None:
        return cfg_value
    return default_value


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="evalute the dataset")
    parser.add_argument("--config", type=Path, default=None, help="Path to yaml config file")
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Path to input json/jsonl",
    )
    parser.add_argument(
        "--output-base-dir",
        type=Path,
        default=None,
        help="Root directory for generation runs",
    )
    parser.add_argument("--model", "-m", type=str,
                        default=None,
                        help="The unsloth model name or path")
    parser.add_argument("--backend", type=str, default=None, help="Generation backend name")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--max-input-tokens", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--load-in-4bit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--load-in-8bit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--full-finetuning", action=argparse.BooleanOptionalAction, default=None)
    
    args = parser.parse_args(argv)
    cfg = load_yaml_config(args.config)

    input_path = _resolve(
        args.input,
        get_cfg(cfg, "generation.input_path", get_cfg(cfg, "input_path")),
        Path("/home/xchen6/breaking_updates_rl/data/prompts/dataset.json"),
    )
    output_base_dir = _resolve(
        args.output_base_dir,
        get_cfg(cfg, "generation.output_base_dir", get_cfg(cfg, "output_base_dir")),
        Path("/home/xchen6/breaking_updates_rl/experiments/benchmark"),
    )
    model_id = _resolve(
        args.model,
        get_cfg(cfg, "generation.model_id", get_cfg(cfg, "model_id")),
        "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    )
    backend_name = _resolve(
        args.backend,
        get_cfg(cfg, "generation.backend", get_cfg(cfg, "backend")),
        "unsloth",
    )

    config = GenerationConfig(
        input_path=Path(input_path),
        model_id=model_id,
        output_base_dir=Path(output_base_dir),
        backend=backend_name,
        max_seq_length=int(
            _resolve(args.max_seq_length, get_cfg(cfg, "generation.max_seq_length"), 35000)
        ),
        max_input_tokens=int(
            _resolve(args.max_input_tokens, get_cfg(cfg, "generation.max_input_tokens"), 32000)
        ),
        max_new_tokens=int(
            _resolve(args.max_new_tokens, get_cfg(cfg, "generation.max_new_tokens"), 4000)
        ),
        load_in_4bit=bool(
            _resolve(args.load_in_4bit, get_cfg(cfg, "generation.load_in_4bit"), True)
        ),
        load_in_8bit=bool(
            _resolve(args.load_in_8bit, get_cfg(cfg, "generation.load_in_8bit"), False)
        ),
        full_finetuning=bool(
            _resolve(args.full_finetuning, get_cfg(cfg, "generation.full_finetuning"), False)
        ),
    )
    evaluate(config)
    
    

if __name__ == "__main__":
    main()
    

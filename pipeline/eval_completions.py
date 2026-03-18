import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys
from datetime import datetime
from typing import Any

from pipeline.common.config_loader import get_cfg, load_yaml_config
from pipeline.evaluation.backends import EvaluationRuntimeConfig, build_evaluation_backend


@dataclass
class EvalConfig:
    input_path: Path
    backend: str = "container_patcher"
    run_tests: bool = True
    stats_filename: str = "overall_statistics_new.json"
    baseline_input_path: Path | None = None


def _write_eval_manifest(input_path: Path, config: EvalConfig) -> None:
    manifest = {
        "stage": "evaluation",
        "created_at": datetime.now().isoformat(),
        "config": {
            **asdict(config),
            "input_path": str(config.input_path),
            "baseline_input_path": str(config.baseline_input_path) if config.baseline_input_path else None,
        },
    }
    with (input_path / "eval_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)


def _load_input_records(input_path: Path) -> list[dict[str, Any]]:
    if input_path.suffix.lower() == ".jsonl":
        records: list[dict[str, Any]] = []
        with input_path.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if isinstance(obj, dict):
                    records.append(obj)
        return records

    with input_path.open() as f:
        data = json.load(f)

    if isinstance(data, list):
        return [obj for obj in data if isinstance(obj, dict)]

    if isinstance(data, dict):
        records: list[dict[str, Any]] = []
        for value in data.values():
            if isinstance(value, list):
                records.extend(obj for obj in value if isinstance(obj, dict))
        return records

    raise ValueError(f"Unsupported baseline input format: {input_path}")


def _compute_fixed_baseline(records: list[dict[str, Any]]) -> dict[str, int]:
    """
    Compute fixed baseline using the same dedup shape as evaluator:
    deduplicate by (breakingCommit, absolute_path_to_file_in_container).
    """
    by_commit: dict[str, dict[str, list[Any]]] = {}
    for obj in records:
        commit = str(obj.get("breakingCommit", "")).strip()
        file_path = str(obj.get("absolute_path_to_file_in_container", "")).strip()
        if not commit or not file_path:
            continue
        errors = obj.get("errors", [])
        if not isinstance(errors, list):
            errors = []
        by_commit.setdefault(commit, {})[file_path] = errors

    total_original_file_count = sum(len(files) for files in by_commit.values())
    total_original_error_count = sum(
        len(errors)
        for files in by_commit.values()
        for errors in files.values()
    )
    return {
        "total_original_file_count": total_original_file_count,
        "total_original_error_count": total_original_error_count,
    }


def _infer_baseline_input_path(input_path: Path) -> Path | None:
    run_manifest_path = input_path / "run_manifest.json"
    if not run_manifest_path.exists():
        return None

    try:
        run_manifest = json.loads(run_manifest_path.read_text())
    except Exception as e:
        logging.warning("Failed to parse run_manifest for baseline inference: %s", e)
        return None

    if not isinstance(run_manifest, dict):
        return None

    raw_path = run_manifest.get("config", {}).get("input_path")
    if not raw_path:
        return None

    baseline_path = Path(str(raw_path)).expanduser()
    if not baseline_path.is_absolute():
        baseline_path = (input_path / baseline_path).resolve()
    return baseline_path


def patch_and_evaluate_project(config: EvalConfig) -> dict:
    """
    Evaluate project-level completions by applying all patches in each BU folder.
    """
    input_path = config.input_path
    _write_eval_manifest(input_path, config)
    runtime_cfg = EvaluationRuntimeConfig(run_tests=config.run_tests)
    backend = build_evaluation_backend(config.backend, runtime_cfg)

    successful_fixes = 0
    total_projects = 0
    evaluated_original_error_count = 0
    total_fixed_error_count = 0
    evaluated_original_file_count = 0
    total_fixed_file_count = 0
    total_new_errors_count = 0

    for folder in input_path.iterdir():
        if not folder.is_dir():
            continue

        metrics = backend.evaluate_project_folder(folder)
        if metrics is None:
            continue

        evaluated_original_error_count += metrics["original_error_count"]
        total_fixed_error_count += metrics["fixed_error_count"]
        evaluated_original_file_count += metrics["original_file_count"]
        total_fixed_file_count += metrics["fixed_file_count"]
        total_new_errors_count += metrics["new_errors_count"]
        if metrics["build_success"]:
            successful_fixes += 1
        total_projects += 1

    baseline_input_path = config.baseline_input_path or _infer_baseline_input_path(input_path)
    if baseline_input_path is not None and baseline_input_path.exists():
        baseline_records = _load_input_records(baseline_input_path)
        baseline = _compute_fixed_baseline(baseline_records)
        total_original_file_count = baseline["total_original_file_count"]
        total_original_error_count = baseline["total_original_error_count"]
        logging.info(
            "Using fixed baseline from %s: files=%d errors=%d",
            baseline_input_path,
            total_original_file_count,
            total_original_error_count,
        )
    else:
        if baseline_input_path is not None and not baseline_input_path.exists():
            logging.warning("Baseline input path does not exist: %s", baseline_input_path)
        total_original_file_count = evaluated_original_file_count
        total_original_error_count = evaluated_original_error_count
        logging.warning(
            "Fallback to evaluated baseline: files=%d errors=%d",
            total_original_file_count,
            total_original_error_count,
        )

    statistics = {
        "successful_fixes": successful_fixes,
        "total_projects": total_projects,
        "total_original_error_count": total_original_error_count,
        "total_fixed_error_count": total_fixed_error_count,
        "total_original_file_count": total_original_file_count,
        "total_fixed_file_count": total_fixed_file_count,
        "evaluated_original_error_count": evaluated_original_error_count,
        "evaluated_original_file_count": evaluated_original_file_count,
        "baseline_input_path": str(baseline_input_path) if baseline_input_path else None,
        "BuildSuccessRate": successful_fixes / total_projects if total_projects > 0 else 0.0,
        "FileFixSuccessRate": total_fixed_file_count / total_original_file_count if total_original_file_count > 0 else 0.0,
        "CompilationErrorFixRate": total_fixed_error_count / total_original_error_count if total_original_error_count > 0 else 0.0,
        "RelativeErrorFixRatio": (total_fixed_error_count - total_new_errors_count) / total_original_error_count if total_original_error_count > 0 else 0.0,
    }
    logging.info("Overall statistics: %s", statistics)
    with (input_path / config.stats_filename).open("w") as sf:
        json.dump(statistics, sf, indent=2)
    return statistics


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
    parser = argparse.ArgumentParser(description="evalute the completions")
    parser.add_argument("--config", type=Path, default=None, help="Path to yaml config file")
    parser.add_argument("--input", "-i", type=Path, default=None, help="Path to result folder containing completions")
    parser.add_argument("--backend", type=str, default=None, help="Evaluation backend name")
    parser.add_argument("--run-tests", action=argparse.BooleanOptionalAction, default=None, help="Run tests after compilation succeeds")
    parser.add_argument("--stats-filename", type=str, default=None, help="Filename for overall statistics under --input folder")
    parser.add_argument("--baseline-input", type=Path, default=None, help="Fixed baseline dataset path (.json/.jsonl)")
    args = parser.parse_args(argv)
    cfg = load_yaml_config(args.config)

    input_path = _resolve(
        args.input,
        get_cfg(cfg, "evaluation.input_path", get_cfg(cfg, "input_path")),
        Path(__file__).parent.parent / "results" / "unsloth" / "gemma-3-4b-it-unsloth-bnb-4bit_20251029-030040",
    )
    backend_name = _resolve(
        args.backend,
        get_cfg(cfg, "evaluation.backend", get_cfg(cfg, "backend")),
        "container_patcher",
    )
    run_tests = bool(_resolve(args.run_tests, get_cfg(cfg, "evaluation.run_tests"), True))
    stats_filename = _resolve(
        args.stats_filename,
        get_cfg(cfg, "evaluation.stats_filename"),
        "overall_statistics_new.json",
    )
    baseline_input_path = _resolve(
        args.baseline_input,
        get_cfg(cfg, "evaluation.baseline_input_path", get_cfg(cfg, "baseline_input_path")),
        None,
    )

    config = EvalConfig(
        input_path=Path(input_path),
        backend=backend_name,
        run_tests=run_tests,
        stats_filename=str(stats_filename),
        baseline_input_path=Path(baseline_input_path) if baseline_input_path else None,
    )
    patch_and_evaluate_project(config)


if __name__ == "__main__":
    main()

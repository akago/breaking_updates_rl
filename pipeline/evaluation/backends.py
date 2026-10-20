from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from pipeline.types.maven_error import MavenErrorLog, MavenErrorParser
from pipeline.types.metrics import Patcher
from pipeline.types.utils import is_java_source_valid, remove_trailing_whitespace
from pipeline.constants.constants import RESOURCES_PATH


@dataclass
class EvaluationRuntimeConfig:
    run_tests: bool = True


class EvaluationBackend(Protocol):
    def evaluate_project_folder(self, folder: Path) -> dict | None:
        """Evaluate one project folder and return metrics, or None when skipped."""


class ContainerPatcherEvaluationBackend:
    def __init__(self, cfg: EvaluationRuntimeConfig) -> None:
        self.cfg = cfg

    def evaluate_project_folder(self, folder: Path) -> dict | None:
        patches_to_bind = []
        resources_path = RESOURCES_PATH / str(folder.name)
        container_path = resources_path / f"{str(folder.name)}.sif"
        original_errors = {}
        project_name = None

        for completion_file in folder.glob("*.json"):
            if not completion_file.stem.isdigit():
                continue

            logging.info("loading completion file: %s", completion_file)
            result_dict = json.loads(completion_file.read_text())
            project_name = result_dict.get("project")

            errors_in_file = result_dict["errors"]
            original_errors[result_dict["absolute_path_to_file_in_container"]] = errors_in_file
            buggy_file_name = Path(result_dict["absolute_path_to_file_in_container"]).name

            temp_file_path = folder / buggy_file_name
            patch = result_dict.get("patch", "")
            if patch == "":
                logging.warning("No patch found in %s", completion_file)
                continue

            java_code = patch
            if "\n=======\n" in java_code:
                logging.warning("Invalid patch with ======= in %s", completion_file)
                continue

            if not is_java_source_valid(java_code):
                logging.warning("Generated java code is not valid in %s", completion_file)
                continue
            try:
                clean_code = remove_trailing_whitespace(java_code)
            except Exception as e:
                logging.warning("Error while removing trailing whitespace in %s: %s", completion_file, e)
                continue
            if clean_code != "":
                java_code = clean_code

            temp_file_path.write_text(java_code)
            patches_to_bind.append((str(temp_file_path), result_dict["absolute_path_to_file_in_container"]))

        if project_name is None:
            logging.warning("No valid completion json found under %s, skip.", folder)
            return None

        patcher = Patcher(
            project=project_name,
            container_path=str(container_path),
            log_path=str(folder / f"{str(folder.name)}.log"),
            binding_pairs=patches_to_bind,
        )
        build_log, success = patcher.apply_patch()
        if success and self.cfg.run_tests:
            _, success = patcher.apply_patch_with_test()

        if "Failed to execute goal org.apache.maven.plugins:maven-checkstyle-plugin" in build_log:
            error_log = original_errors
        else:
            log_parser = MavenErrorParser()
            error_log = MavenErrorLog.from_string(build_log, log_parser).to_jsonable()

        metrics = patcher.metrics(original_errors, error_log, success)
        metrics_file = folder / "metrics_new.json"
        with metrics_file.open("w") as mf:
            json.dump(metrics, mf, indent=2)
        logging.info("Metrics saved to %s", metrics_file)
        logging.info("Metrics for project %s: %s", project_name, metrics)
        return metrics


def build_evaluation_backend(name: str, cfg: EvaluationRuntimeConfig) -> EvaluationBackend:
    backend = name.lower()
    if backend == "container_patcher":
        return ContainerPatcherEvaluationBackend(cfg)
    raise ValueError(f"Unsupported evaluation backend: {name}")

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from pipeline.types.maven_error import MavenErrorLog


@dataclass
class Observation:
    """Observation passed to the policy
    """
    client_code: Optional[str]
    buggy_line: Optional[str]
    error_message: Optional[str]
    api_diff: Optional[str]
    file_rel_path: str


class RepairEnv:
    """Environment that evaluates patch suggestions against project context.

    This implementation supports file-level (single-step) and project-level
    (multi-step across all buggy files) episodes. 
    """

    def __init__(self, work_base: Optional[Path] = None) -> None:
        self._last_metrics: Dict[str, Any] = {}
        self._obs: Optional[Observation] = None
        self._episode: Optional[EpisodeSample] = None
        self._workspace: Optional[Path] = None
        self._baseline_map: dict[str, set[ErrorItem]] = {}
        self._final_errors: set[ErrorItem] = set()
        self._step_idx: int = 0
        # self._work_base = Path(work_base) if work_base else Path(tempfile.gettempdir())

    def reset(self, episode: EpisodeSample) -> Observation:
        self._cleanup()
        self._episode = episode
        self._workspace = self._prepare_workspace(episode.project.project_root)

        # Build file processing order
        if episode.spec.level == "file":
            assert episode.spec.target_idx is not None
            self._file_queue = [episode.spec.target_idx]
        else:
            self._file_queue = list(range(len(episode.project.buggy_files)))

        # Baseline errors
        self._baseline_map = {
            bf.rel_path: set(bf.baseline_errors) for bf in episode.project.buggy_files
        }
        self._final_errors = set()
        self._step_idx = 0

        return self._make_observation()

    def step(self, action) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        assert self._episode is not None and self._workspace is not None
        self._step_idx += 1

        # Current target file
        idx = self._file_queue[0]
        current_file = self._episode.project.buggy_files[idx]
        self._apply_patch(current_file.rel_path, getattr(action, "patch_text", ""))

        # Run tests and parse the resulting log
        mel = self._run_mvn_and_parse()
        current_errors = self._to_error_items(mel)
        self._final_errors = current_errors

        # Step-level info (delta vs. baseline for this file)
        base_for_file = self._baseline_map.get(current_file.rel_path, set())
        fixed_now = len([e for e in base_for_file if e not in current_errors])
        new_now = len([e for e in current_errors if e not in base_for_file])

        info = {
            "file": current_file.rel_path,
            "step_idx": self._step_idx,
            "fixed_in_step": fixed_now,
            "new_in_step": new_now,
        }

        # Advance
        self._file_queue.pop(0)
        done = len(self._file_queue) == 0
        reward = float(fixed_now - new_now)

        if done:
            self._last_metrics = self._compute_final_metrics(current_errors)

        return (self._make_observation(), reward, done, info)

    def metrics(self) -> Dict[str, Any]:
        return dict(self._last_metrics)

    # ----- Internals -----
    def _prepare_workspace(self, project_root: Path) -> Path:
        dest = self._work_base / f"buws_tmp"
        try:
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(project_root, dest)
        except Exception:
            dest = project_root
        return dest

    def _apply_patch(self, file_rel_path: str, patch_text: str) -> None:
        assert self._workspace is not None
        target = self._workspace / file_rel_path
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            content = target.read_text(encoding="utf-8", errors="ignore") if target.exists() else ""
            content += f"\n/* PATCH SUGGESTION */\n{patch_text}\n"
            target.write_text(content, encoding="utf-8", errors="ignore")
        except Exception:
            pass

    def _run_mvn_and_parse(self) -> MavenErrorLog:
        # Placeholder: return empty error log in restricted environments
        return MavenErrorLog()

    def _to_error_items(self, mel: MavenErrorLog) -> set[ErrorItem]:
        out: set[ErrorItem] = set()
        for info in mel.to_list():
            rel = str(info.path)
            try:
                if self._workspace is not None:
                    rel = str(Path(info.path).resolve().relative_to(self._workspace.resolve()))
            except Exception:
                rel = str(info.path)
            out.add(ErrorItem(rel_path=rel, line=info.line_num, message=info.message))
        return out

    def _compute_final_metrics(self, final_errors: set[ErrorItem]) -> Dict[str, Any]:
        baseline_all: set[ErrorItem] = set().union(*self._baseline_map.values()) if self._baseline_map else set()

        fixed_total = len([e for e in baseline_all if e not in final_errors])
        new_total = len([e for e in final_errors if e not in baseline_all])

        per_file: Dict[str, Dict[str, int]] = {}
        fully_fixed_files = 0
        for rel, base_set in self._baseline_map.items():
            remaining = len([e for e in base_set if e in final_errors])
            fixed = len(base_set) - remaining
            new_for_file = len([e for e in final_errors if e.rel_path == rel and e not in base_set])
            per_file[rel] = {
                "baseline": len(base_set),
                "final": remaining,
                "fixed": fixed,
                "new": new_for_file,
            }
            if remaining == 0 and len(base_set) > 0:
                fully_fixed_files += 1

        project_fully_fixed = fully_fixed_files == len(self._baseline_map)

        return {
            "fixed_errors_total": fixed_total,
            "fully_fixed_files": fully_fixed_files,
            "project_fully_fixed": project_fully_fixed,
            "new_errors_total": new_total,
            "per_file": per_file,
        }

    def _make_observation(self) -> Observation:
        assert self._episode is not None
        if not self._file_queue:
            idx = max(0, len(self._episode.project.buggy_files) - 1)
        else:
            idx = self._file_queue[0]
        bf = self._episode.project.buggy_files[idx]

        total_steps = 1 if self._episode.spec.level == "file" else len(self._episode.project.buggy_files)
        file_ctx = None
        if self._workspace is not None:
            fpath = self._workspace / bf.rel_path
            try:
                file_ctx = fpath.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                file_ctx = None

        return Observation(
            level=self._episode.spec.level,
            project_root=self._workspace or self._episode.project.project_root,
            file_rel_path=bf.rel_path,
            errors=list(bf.baseline_errors),
            api_diff=bf.api_diff,
            step_idx=self._step_idx,
            total_steps=total_steps,
            file_context=file_ctx,
        )

    def _cleanup(self) -> None:
        if self._workspace and self._workspace.exists():
            try:
                shutil.rmtree(self._workspace)
            except Exception:
                pass
        self._workspace = None

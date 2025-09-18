from __future__ import annotations

from pathlib import Path
from typing import Generator, Iterable, Literal
import json

from pipeline.types.episode import (
    ErrorItem,
    BuggyFile,
    ProjectSample,
    EpisodeSpec,
    EpisodeSample,
)

    
def _to_rel(key_path: str) -> str:
    # Normalize leading slash and collapse repeated slashes
    return key_path.lstrip("/")

def _parse_project(ctx_path: Path) -> ProjectSample:
    raw = json.loads(ctx_path.read_text(encoding="utf-8", errors="ignore"))
    buggy_files: dict[str, list[dict]] = raw.get("buggyFiles", {})
   
    # propagate breakingChanges into api_diff text
    bchanges = raw.get("breakingChanges") or []
    api_diff_text = "\n".join(
        f"- {bc.get('element','')} | {bc.get('nature','')} | {bc.get('kind','')}" for bc in bchanges
    ) or None
    files: list[BuggyFile] = []
    for k, items in buggy_files.items():
        rel = _to_rel(k)
        errs: set[ErrorItem] = set()
        for it in items:
            line = int(it.get("line_number", 0))
            msg = str(it.get("message", ""))
            additional_info = str(it.get("additional_info"))
            errs.add(ErrorItem(rel_path=rel, line=line, message="".join([msg, additional_info])))
        files.append(BuggyFile(rel_path=rel, baseline_errors=errs, api_diff=api_diff_text))
    project = ProjectSample(
        project_root= str(ctx_path.parent / raw.get("project", "")),
        project_name=raw.get("project"),
        organisation=raw.get("projectOrganisation"),
        breaking_commit=raw.get("breakingCommit"),
        library_name=raw.get("libraryName"),
        previous_version=raw.get("previousVersion"),
        new_version=raw.get("newVersion"),
        buggy_files=files,
    )
    return project

def iter_dataset(root: Path, level: Literal["file", "project"]) -> Generator[EpisodeSample, None, None]:
    """Yield episode samples by scanning dataset folders for context.json.
    Expected directory structure:
    -root--commit_hash--context.json
    
    - file level: one episode per buggy file
    - project level: one episode per project
    """
    root = Path(root)
    if not root.exists():
        return

    for ctx_path in root.rglob("context.json"):
        try:
            project = _parse_project(ctx_path)
        except Exception:
            continue

        if level == "file":
            for i, _ in enumerate(project.buggy_files):
                yield EpisodeSample(project=project, spec=EpisodeSpec(level="file", target_idx=i))
        else:
            yield EpisodeSample(project=project, spec=EpisodeSpec(level="project", target_idx=None))


def as_iterable(root: Path, level: Literal["file", "project"]) -> Iterable[EpisodeSample]:
    """Return a simple iterable over episode samples given a dataset root."""
    return iter_dataset(root, level)

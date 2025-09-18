from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal, Iterable
import hashlib


def _norm_ws(s: str) -> str:
    """Normalize whitespace for stable comparison of error messages."""
    return " ".join(s.strip().split())


@dataclass(frozen=True)
class ErrorItem:
    """Single compiler error location with normalized identity."""
    rel_path: str
    line: int
    message: str

    @property
    def key(self) -> tuple[str, int, str]:
        # Hash the normalized message to avoid very long keys
        msg = _norm_ws(self.message)
        digest = hashlib.md5(msg.encode("utf-8", errors="ignore")).hexdigest()
        return (self.rel_path, self.line, digest)


@dataclass
class BuggyFile:
    """Aggregated error information for a single file."""
    rel_path: str
    baseline_errors: set[ErrorItem] = field(default_factory=set)
    client_code: Optional[str] = None
    api_diff: Optional[str] = None


@dataclass
class ProjectSample:
    """Represents a project with a set of buggy files.

    project_root should point to the directory containing the source code for
    building and testing the project .
    """
    project_root: Path
    project_name: str
    organisation: Optional[str] = None
    breaking_commit: Optional[str] = None
    library_name: Optional[str] = None
    previous_version: Optional[str] = None
    new_version: Optional[str] = None
    buggy_files: list[BuggyFile] = field(default_factory=list)


@dataclass(frozen=True)
class EpisodeSpec:
    """Episode definition: file-level or project-level."""
    level: Literal["file", "project"]
    target_idx: Optional[int] = None  # used when level == "file"


@dataclass(frozen=True)
class EpisodeSample:
    """A single episode instance yielded by the dataloader."""
    project: ProjectSample
    spec: EpisodeSpec




from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Set, List
import json
from pipeline.types.api_change import ApiChange
from pipeline.types.maven_error import ErrorInfo


@dataclass(eq=True)
class BuggyFileWithErrors:
    """Represents a file with detected errors."""

    api_changes: Set[ApiChange] = field(default_factory=set)
    error_infos: List[ErrorInfo] = field(default_factory=list)
    absolute_path_to_file: Optional[str] = None
    
    @classmethod
    def from_json(cls, data: dict) -> BuggyFileWithErrors:
        api_changes = {ApiChange.from_json(change) for change in data.get("api_changes", [])}
        return cls(
            api_changes=api_changes,
            error_infos=data.get("error_info"),
            absolute_path_to_file=data.get("absolute_path_to_file")
        )
           



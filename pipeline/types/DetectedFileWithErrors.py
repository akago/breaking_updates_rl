

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Set, List
import json
from pipeline.types.api_change import ApiChange



@dataclass
class FaultInformation:
    """Information about a detected fault."""

    method_name: Optional[str] = None
    method_code: Optional[str] = None
    qualified_method_code: Optional[str] = None
    qualified_in_class_code: Optional[str] = None
    in_class_code: Optional[str] = None
    plausible_dependency_identifier: Optional[str] = None
    client_line_number: Optional[int] = None
    client_end_line_number: Optional[int] = None

    def __str__(self) -> str:  # pragma: no cover - simple serialization
        try:
            return json.dumps(asdict(self))
        except TypeError:
            return ""

@dataclass(eq=True)
class DetectedFileWithErrors:
    """Represents a file with detected errors."""

    api_changes: Set[ApiChange] = field(default_factory=set)
    error_info: Optional[Any] = None
    line_in_code: Optional[str] = None
    class_path: Optional[str] = None
    in_class_code: Optional[str] = None
    client_line_number: Optional[int] = None
    client_end_line_number: Optional[int] = None
    
    @classmethod
    def from_json(cls, data: dict) -> DetectedFileWithErrors:
        api_changes = {ApiChange.from_json(change) for change in data.get("api_changes", [])}
        return cls(
            api_changes=api_changes,
            error_info=data.get("error_info"),
            line_in_code=data.get("line_in_code"),
            class_path=data.get("class_path"),
            in_class_code=data.get("in_class_code"),
            client_line_number=data.get("client_line_number"),
            client_end_line_number=data.get("client_end_line_number"),
        )
           



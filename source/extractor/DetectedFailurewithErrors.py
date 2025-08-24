

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Set, List
import json


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

    api_changes: Set[Any] = field(default_factory=set)
    executed_elements: Optional[Set[Any]] = None
    error_info: Optional[Any] = None
    line_in_code: Optional[str] = None
    class_path: Optional[str] = None
    code_element: Optional[Any] = None

    method_name: Optional[str] = None
    method_code: Optional[str] = None
    qualified_method_code: Optional[str] = None
    qualified_in_class_code: Optional[str] = None
    in_class_code: Optional[str] = None
    plausible_dependency_identifier: Optional[str] = None
    client_line_number: Optional[int] = None
    client_end_line_number: Optional[int] = None

@dataclass
class ApiChange:
    element: str = None
    kind: str = None
    newVersion: str = None
    oldVersion: str = None
    


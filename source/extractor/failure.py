
from typing import Any, Set, List
from .failure_detection import FaultInformation
from dataclasses import dataclass, field

@dataclass
class Failure:
    """Represents a detected failure and its related API changes."""
    api_changes: set = field(default_factory=set)
    detected_fault: FaultInformation | None = None
    
     def get_api_changes(self) -> List[Any]:
        """Return API changes sorted by their ``value`` attribute."""
        return sorted(self.api_changes, key=lambda change: getattr(change, "value", None))


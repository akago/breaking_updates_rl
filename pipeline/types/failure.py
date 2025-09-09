




from dataclasses import dataclass, field
from .DetectedFileWithErrors import FaultInformation


@dataclass
class Failure:
    """Represents a detected failure and its related API changes."""
    api_changes: set = field(default_factory=set)
    detected_fault: FaultInformation | None = None
    
    def get_api_changes(self):
        """Return API changes sorted by their ``value`` attribute."""
        return sorted(self.api_changes, key=lambda change: getattr(change, "value", None))

    @classmethod
    def from_json(cls, data: dict) -> Failure:
        """Create a Failure instance from a JSON-like dictionary."""
        api_changes = {APIChange.from_json(change) for change in data.get("api_changes", [])}
        detected_fault = FaultInformation.from_json(data.get("detected_fault")) if data.get("detected_fault") else None
        return cls(api_changes=api_changes, detected_fault=detected_fault)

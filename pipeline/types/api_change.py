from dataclasses import dataclass
from typing import Dict

@dataclass
class ApiChange:
    element: str = None
    kind: str = None
    newVersion: str = None
    oldVersion: str = None
    
    @classmethod
    def from_json(cls, data: dict):
        return cls(
            kind=data["kind"],
            element=data["element"],
            newVersion=data["newVersion"],
            oldVersion=data["oldVersion"],
        )
    
    @staticmethod
    def from_csv(row: dict):
        return ApiChange(
            kind=row.get("kind"),
            element=row.get("element"),
            newVersion=row.get("newVersion"),
            oldVersion=row.get("oldVersion"),
        )
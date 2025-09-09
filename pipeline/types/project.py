from __future__ import annotations
from dataclasses import dataclass
from pipeline.types.DetectedFileWithErrors import DetectedFileWithErrors


@dataclass
class Project:
    path: str
    project_name: str
    organisation: str
    breaking_commit: str
    library_name: str
    old_library_version: str
    new_library_version: str
    buggy_files: list[DetectedFileWithErrors] = None

    @classmethod
    def from_json(cls, data: dict, path: str) -> Project:
        buggy_files = [
            DetectedFileWithErrors.from_json(file)
            for file in data.get("buggyFiles", [])
        ]
        return cls(
            path=path,
            project_name=data.get("project"),
            organisation=data.get("projectOrganisation"),
            breaking_commit=data.get("breakingCommit"),
            library_name=data.get("libraryName"),
            library_group_id=data.get("libraryGroupID"),
            old_library_version=data.get("oldLibraryVersion"),
            new_library_version=data.get("newLibraryVersion"),
            buggy_files=buggy_files
        )
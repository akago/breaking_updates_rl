from __future__ import annotations
from dataset import Dataset

from pathlib import Path

@dataclass
class BreakingUpdate:
    client_code: str
    buggy_line: str
    error_message: str
    api_diff: str
    absolute_path_to_buggy_class: Path
    breaking_commit:str


class BreakingUpdateDataLoader:
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def build_dummy_dataset(n_projects=2, files_per_project=3, n_rows=100):
        rows = []
        for _ in range(n_rows):
            pid = f"proj-{random.randint(1, n_projects)}"
            fidx = random.randint(1, files_per_project)
            rows.append({
                # "prompt" is required by trl
                "prompt": f"[{pid}] Please repair file src/F{fidx}.java given the breaking change.",
                "project_id": pid,
                "file_path": f"src/F{fidx}.java",
                # 这里可以放 BC 相关符号、编译基线等你需要的上下文列
            })
        return Dataset.from_list(rows)
from pathlib import Path

@dataclass
class BreakingUpdate:
    client_code: str
    buggy_line: str
    error_message: str
    api_diff: str
    absolute_path_to_buggy_class: Path


class BreakingUpdateDataLoader:
    def __init__(self, file_path: Path):
        self.file_path = file_path

    def load_data(self) -> list[BreakingUpdate]:
        # Implement logic to load breaking update data from the file
        pass
from dataclasses import dataclass
from pathlib import PurePosixPath, Path
from typing import Iterator, Optional
from collections import defaultdict
import re

@dataclass(frozen=True, slots=True)
class ErrorInfo:
    path: PurePosixPath
    line_num: int
    message: str
    additional: str

class MavenErrorParser:
    # The regex captures for both Path and Line number
    ERROR_RE = re.compile(r"\[ERROR\] .*?(?P<path>/[^\[\]:]+):\[(?P<line>\d+),(?P<col>\d+)\]")
    
    # ERROR_RE = re.compile(r"\[ERROR\] .*:\[(\d+),\d+\]")
    # PATH_RE = re.compile(r"/[^:/]+(/[^\[\]:]+)")

    def __init__(self, error_re: Optional[re.Pattern[str]] = None) -> None:
        self.error_re = error_re or self.ERROR_RE

    def iter_errors(self, lines: Iterator[str]) -> Iterator[ErrorInfo]:
        buf: list[str] = []
        def is_continuation_line(s: str) -> bool:
            # start with space
            if s.startswith(" "):
                return True
            # start with [ERROR] but not matching the error pattern
            if s.startswith("[ERROR]"):
                return self.error_re.search(s) is None
            return False
        
        def flush_indented(it: Iterator[str]) -> str:
            parts: list[str] = []
            for s in it:
                if not s.startswith(" "):
                    buf.append(s)  # backtrack one line
                    break
                parts.append(s.rstrip("\n"))
            return "\n".join(parts)

        for line in lines:
            if buf:
                line = buf.pop()
            m = self.error_re.search(line)
            if not m:
                continue
            path = PurePosixPath(m.group("path"))
            ln = int(m.group("line"))
            add = flush_indented(lines)
            yield ErrorInfo(path=path, line_num=ln, message=line.rstrip("\n"), additional=add)

class MavenErrorLog:
    def __init__(self) -> None:
        self._by_path: dict[PurePosixPath, dict[int, ErrorInfo]] = defaultdict(dict)

    def add(self, info: ErrorInfo) -> bool:
        bucket = self._by_path[info.path]
        if info.line_num in bucket: 
            return False
        bucket[info.line_num] = info
        return True

    def extend(self, items: Iterator[ErrorInfo]) -> None:
        for it in items:
            self.add(it)

    @classmethod
    def from_file(cls, log_file: str | Path, parser: Optional[MavenErrorParser] = None,
                  encoding: str = "latin-1") -> "MavenErrorLog":
        parser = parser or MavenErrorParser()
        instance = cls()
        with open(log_file, "r", encoding=encoding, errors="replace") as f:
            instance.extend(parser.iter_errors(iter(f)))
        return instance

    def to_jsonable(self) -> dict[str, list[dict[str, str | int]]]:
        return {
            str(path): [
                {
                    "line_number": info.line_num,
                    "message": info.message,
                    "additional_info": info.additional,
                    "file_name": str(info.path.name)
                }
                # sorted by line number
                for info in sorted(infos.values(), key=lambda x: x.line_num)
            ]
            for path, infos in self._by_path.items()
        }

    def to_list(self) -> list[ErrorInfo]:
        return [
            info
            for infos in self._by_path.values()
            for info in infos.values()
        ]
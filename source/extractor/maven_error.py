from dataclasses import dataclass
from typing import Dict, Tuple, List, TypeAlias
from collections import defaultdict
import re
from pathlib import Path
    
# Type aliases for readability
Key: TypeAlias = Tuple[int, str]
InnerMap: TypeAlias = Dict[Key, 'ErrorInfo']  # key -> ErrorInfo


@dataclass(slots=True, frozen=True)
class ErrorInfo:
    """Immutable error record extracted from a Maven log line."""
    line: int                       # line number inside the client file (from [x,y] in the error)
    file_path: str                  # relative path captured from the log line
    message: str                    # the raw log line content
    file_line_number: int           # line number inside the *log file*
    additional_info: str = ""       # subsequent indented lines starting with a space
    file_name: str | None = None    # basename of file_path


class MavenErrorLog:
    """
    Stores error information extracted from Maven build logs.
    """

    def __init__(self) -> None:
        # key: currentPath
        # value: dict，key : (client_line_position, client_file_path)，value : ErrorInfo
        self.error_info: Dict[str, InnerMap] = defaultdict(dict)
     def __init__(self) -> None:
        # For each path, keep a dict keyed by (line, file_path).
        self._index: Dict[str, InnerMap] = defaultdict(dict)


    def add_error_info(self, current_path: str, error_info: ErrorInfo) -> None:
        # The group of errors for the current path {(line, file_path) -> ErrorInfo}
        group = self._index[current_path]

        key = (error_info.client_line_position, error_info.client_file_path)
        if key in group:
            return False
        
        group[key] = error_info
        return True

    def get_errors(self, current_path: str) -> List[ErrorInfo]:
        """return all errors for the given path."""
        return list(self._index.get(current_path, {}).values())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(paths={len(self._index)}, total_errors={sum(len(errors) for errors in self._index.values())})"
    
    
class MavenErrorInformation:
    """
    Parse a Maven build log and collect ErrorInfo items.

    The patterns:
      - error pattern:   [ERROR] ...:[(line),col]
      - path  pattern:   a slash path that excludes ':' '[' ']' characters
    """

    ERROR_RE = re.compile(r"\[ERROR\] .*:\[(\d+),\d+\]")
    PATH_RE = re.compile(r"/[^:/]+(/[^\[\]:]+)")

    def __init__(self, log_file: str | Path) -> None:
        self.log_file = str(log_file)

    def extract_line_numbers_with_paths(self, log_file_path: str | Path) -> MavenErrorLog:
        """
        Stream the log file line-by-line, capture error lines, and accumulate
        ErrorInfo grouped/deduplicated in MavenErrorLog.
        """
        logs = MavenErrorLog()
        current_path = None

        # Use ISO-8859-1 like the Java code
        with open(log_file_path, "r", encoding="latin-1", errors="replace") as reader:
            line_no_in_file = 0
            while True:
                line = reader.readline()
                if not line:
                    break
                line_no_in_file += 1

                m = self.ERROR_RE.search(line)
                if not m:
                    continue

                # Extract the client file line number from [x,y]
                line_num = int(m.group(1))

                # Try to (re)capture a path from the same line
                p = self.PATH_RE.search(line)
                if p:
                    current_path = p.group(0)

                if not current_path:
                    continue

                # Pull following indented lines as "additional info"
                additional = self._extract_additional_info(reader)

                file_name = self.extract_file_name(current_path)

                info = ErrorInfo(
                    line=line_num,
                    file_path=current_path,
                    message=line.rstrip("\n"),
                    file_line_number=line_no_in_file,
                    additional_info=additional,
                    file_name=file_name,
                )
                logs.add(current_path, info)

        return logs

    def _extract_additional_info(self, reader) -> str:
        """
        Read subsequent lines that start with a leading space.
        Stop before the first non-indented line and seek back so the main loop can read it.
        """
        # Peek the next line by remembering the file position.
        start_pos = reader.tell()
        first = reader.readline()
        if not first:
            return ""

        if not first.startswith(" "):
            reader.seek(start_pos)
            return ""

        parts = [first.rstrip("\n")]
        while True:
            pos = reader.tell()
            nxt = reader.readline()
            if not nxt or not nxt.startswith(" "):
                # Rewind so the main loop can consume the non-indented line.
                if nxt:
                    reader.seek(pos)
                break
            parts.append(nxt.rstrip("\n"))
        return "\n".join(parts)

    def extract_file_name(self, path: str) -> str:
        """Return the basename from a POSIX-like path string."""
        return path.split("/")[-1]
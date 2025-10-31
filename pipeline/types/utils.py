
import re
import uuid


def get_error_uid(message: str, additional_info: str = "") -> str:
        """
        generate unique id for a maven error
        ignore line/column information [line,col]
        """
        # namespace
        PROJECT_NS = uuid.uuid5(uuid.NAMESPACE_DNS, "maven_error_name_space")
        
        # remove white spaces
        def norm_space(s: str = "") -> str:
            _WS = re.compile(r"\s+")
            return _WS.sub(" ", s.strip())

        # line and column number may change after applying the patch
        def strip_maven_positions(s: str) -> str:
            _POS_MAVEN = re.compile(r":\[\s*\d+\s*(?:,\s*\d+)?\s*\]")
            return _POS_MAVEN.sub("", s)
        
        message_clean = norm_space(strip_maven_positions(message))
        info_clean = norm_space(additional_info)
        
        name = "".join((message_clean, info_clean))
        return str(uuid.uuid5(PROJECT_NS, name))
    
def ensure_single_trailing_newline(s: str) -> str:
    return s.rstrip('\r\n') + '\n'

def extract_java_code(text: str) -> str:
    end_marker = "<end_of_turn>"
    if end_marker and end_marker in text:
        text = text.split(end_marker)[0]

    JAVA_BLOCK = re.compile(
        r"```java[^\n\r]*\r?\n(.*?)(?=\r?\n?```)",
        flags=re.DOTALL | re.IGNORECASE
    )
    m = JAVA_BLOCK.search(text)
    return ensure_single_trailing_newline(m.group(1)) if m else ""
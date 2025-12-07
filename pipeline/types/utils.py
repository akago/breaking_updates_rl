
import re
import uuid
import logging

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



def extract_sr_edits(text: str) -> list[str]:
    end_marker = "<end_of_turn>"
    if end_marker and end_marker in text:
        text = text.split(end_marker)[0]

    JAVA_BLOCK = re.compile(
        r"```java[^\n\r]*\r?\n(.*?)(?=\r?\n?```)",
        flags=re.DOTALL | re.IGNORECASE
    )
    sr_edits = JAVA_BLOCK.findall(text)
    
    return sr_edits if sr_edits else [""]


def get_patched_content_from_diffs(diffs:list[str], content:str) -> str:
        """
        diffs   [list[str]]: list of SEARCH-REPLACE diff
        content [str]:       the content of buggy file to be replaced
        """
        def extract_diff_blocks(diff_text: str):
            # Extract search and replace, both search or replace could be None
            pattern = re.compile(
                r"""^<<<<<<< SEARCH[ \t]*\r?\n(.*?)^=======[ \t]*\r?\n(.*?)^>>>>>>> REPLACE""", 
                re.MULTILINE | re.DOTALL,
            )
            match = re.search(pattern, diff_text)
            # # TODO: Make sure 
            # matches= re.findall(pattern, diff_text, re.DOTALL)
            if match:
                search_block = match.group(1) if match.group(1) else "" 
                replace_block = match.group(2) if match.group(2) else "" 
                return search_block, replace_block
            return "", ""
        for diff in diffs:
            diff = diff.strip()                
            original, replace = extract_diff_blocks(diff)
            # indent doesn't matter for Java
            original = original.strip()
            replace = replace.strip()
            print(f"The original: {original}")
            print(f"The replace: {replace}")
            # possibily unify the indent
            if original in content:
                content = content.replace(original, replace)
            else:
                logging.error(f"Could not find the search to be replaced:\n{original}")
        return content

def is_java_source_valid(source: str)-> bool:
    lines = source.splitlines()
    seen_type = False          # 是否已经看到过顶层类型定义
    allow_imports = True       # 是否还允许 import（只在类型定义之前）
    public_type_count = 0      # public 顶层类型数量
    brace_depth = 0            # 大括号层级，粗略统计

    type_pattern = re.compile(r'^(public\s+)?(class|interface|enum)\b')

    for i, line in enumerate(lines):
        line_no = i + 1
        
        stripped = line.strip()
        # empty line
        if not stripped:
            brace_depth += line.count("{") - line.count("}")
            continue
        # Single-line comment
        if stripped.startswith("//"):
            brace_depth += line.count("{") - line.count("}")
            continue
        
        if brace_depth == 0:
            if stripped.startswith("package "):
                brace_depth += line.count("{") - line.count("}")
                continue

            if stripped.startswith("import "):
                if not allow_imports:
                    print("Import after type declaration")
                    return False
                brace_depth += line.count("{") - line.count("}")
                continue

            # Top-level type definition
            if type_pattern.match(stripped):
                seen_type = True
                allow_imports = False

                if stripped.startswith("public "):
                    public_type_count += 1
                    if public_type_count > 1:
                        print("Multiple public top-level types in one file")
                        return False

                brace_depth += line.count("{") - line.count("}")
                continue

        brace_depth += line.count("{") - line.count("}")

    if brace_depth != 0:
        print("Unbalanced braces in file")
        return False
    return True

def remove_trailing_whitespace(text: str) -> str:
    lines = text.splitlines(keepends=True)
    cleaned = []

    for line in lines:
        # keep original line endings
        if line.endswith("\r\n"):
            core = line[:-2].rstrip(" \t")
            cleaned.append(core + "\r\n")
        elif line.endswith("\n"):
            core = line[:-1].rstrip(" \t")
            cleaned.append(core + "\n")
        else:
            # last line
            cleaned.append(line.rstrip(" \t"))

    return "".join(cleaned)
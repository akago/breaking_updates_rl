
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
            # Extrac search and replace, both search or replace could be None
            pattern = r'<<<<<<< SEARCH\n(.*?)\n=======\n(.*?)\n>>>>>>> REPLACE'
            match = re.search(pattern, diff_text, re.DOTALL)
            if match:
                search_block = match.group(1) if match.group(1) else "" 
                replace_block = match.group(2) if match.group(2) else "" 
                return search_block, replace_block
            return "", ""
        for diff in diffs:
            diff = diff.strip()                
            original, replace = extract_diff_blocks(diff)
            logging.info(f"The original: {original}")
            logging.info(f"The replace: {replace}")
            # possibily unify the indent
            if original in content:
                content = content.replace(original, replace)
            else:
                logging.error(f"Could not find the search to be replaced:\n{original}")
        return content

# HEADER_RE = re.compile(r'(?m)^###\s+(.+?)\s*$')
# THINK_RE  = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
# # 考虑用更宽松的匹配，符号数量大于2即可
# BLOCK_RE  = re.compile(
#     r'<<<<<<< SEARCH\s*\n'      # block start
#     r'(.*?)'                    # group 1: search chunk
#     r'\n=======\n'              # divider
#     r'(.*?)'                    # group 2: replace chunk
#     r'\n>>>>>>> REPLACE',       # end
#     re.DOTALL
# )

# def parse_search_replace_blocks(llm_response: str):
#     """
#     Parse APR-style SEARCH/REPLACE blocks from an LLM response.

#     Returns a list of dicts:
#       {
#         "file":    str | None,  # nearest preceding '### path' header, or None if absent
#         "search":  str,         # exact SEARCH chunk (including indentation/blank lines)
#         "replace": str          # REPLACE chunk (as-is)
#       }

#     Parsing rules:
#     - Strips any `<think>...</think>` sections before scanning.
#     - Associates each edit block with the closest preceding '### <path>' header (if any).
#     - Works across/inside Markdown code fences; does not attempt to strip them.
#     - Preserves all whitespace in search/replace chunks.
#     """
#     # 1) Remove <think> ... </think> to avoid accidental matches
#     text = THINK_RE.sub('', llm_response)

#     # 2) Index all file headers with their character positions
#     headers = []
#     for m in HEADER_RE.finditer(text):
#         headers.append((m.start(), m.group(1).strip()))

#     # Helper to find nearest header before a given index
#     def nearest_header(pos: int) -> Optional[str]:
#         lo, hi = 0, len(headers) - 1
#         best = None
#         while lo <= hi:
#             mid = (lo + hi) // 2
#             if headers[mid][0] <= pos:
#                 best = headers[mid][1]
#                 lo = mid + 1
#             else:
#                 hi = mid - 1
#         return best

#     # 3) Extract all SEARCH/REPLACE blocks
#     edits: = []
#     for m in BLOCK_RE.finditer(text):
#         start_idx = m.start()
#         file_path = nearest_header(start_idx)
#         search_chunk = m.group(1)
#         replace_chunk = m.group(2)

#         # Preserve exact content; no trimming
#         edits.append({
#             "file": file_path,
#             "search": search_chunk,
#             "replace": replace_chunk,
#         })

# #     return edits

# # --- minimal demo ---
# if __name__ == "__main__":
#     demo = r"""
#     <think>
#     some internal reasoning
#     </think>

#     ### src/main/java/com/example/client/ClientApp.java
#     <<<<<<< SEARCH
#     HttpClient client = HttpClient.create();
#     client.connect(host);
#     =======
#     HttpClient client = HttpClient.create();
#     client.connect(host, Duration.ofSeconds(5)); // new timeout param per API change
#     >>>>>>> REPLACE

#     ### another/File.java
#     <<<<<<< SEARCH
#     int x = api.get();
#     =======
#     int x = api.fetch();
#     >>>>>>> REPLACE
#     """
#     for e in parse_search_replace_blocks(demo):
#         print("FILE:", e["file"])
#         print("---- SEARCH ----")
#         print(e["search"])
#         print("---- REPLACE ---")
#         print(e["replace"])
#         print()

import argparse
from pathlib import Path
import sys
import logging
import json
import linecache
import subprocess
import uuid
import re

from pipeline.constants.constants import LOGGING_FORMAT, DEBUG_LEVEL
from pipeline.constants.constants import PROMPT_TEMPLATE
from pipeline.types.utils import get_error_uid

logging.basicConfig(level=DEBUG_LEVEL, format=LOGGING_FORMAT)


def build_prompts(input_dir: Path) -> None:
    # Build prompt with file
    prompt_dir = input_dir.parent / "prompts"
    sample_counter = 0
    
    def aggregate_breaking_changes(bcs:list[dict], api_additions:list[str], delimiter=" | ") -> str:
        if not bcs and not api_additions:
            return ""
        
        # keys_join = delimiter.join(["element", "nature", "kind"])
        # header = f"Format: {keys_join}"
        # lines = [header]
        lines = []
        
        for bc in bcs:
            if bc["nature"] == "DELETION":
                line = f"-- {bc["element"]}"
            elif bc["nature"] == "ADDITION":
                line = f"{bc["kind"]} <- {bc["element"]}"
            else:
                line = f"{bc["kind"]} <- {bc["element"]}"
            lines.append(line)
            
        lines.extend(api_additions)
        return "\n".join(lines)
    
    def aggregate_error(errors:list[dict]) -> str:
        lines =[]
        for error in errors:
            line = error["message"] + error["additional_info"] 
            lines.append(line)
        return "\n".join(lines)
    
    def get_buggy_lines(file_path:str, errors:list[dict]) -> str:
        lines = []
        for error in errors:
            line_number = int(error["line_number"])
            if line_number <= 0:
                raise ValueError("line_number has to start from 1")
            line = linecache.getline(file_path, line_number)
            if not line:
                continue
            # lines.append(f"{line_number}. " + line.rstrip("\n"))
            # without line number
            lines.append(line.rstrip("\n"))
        return "\n".join(lines)
    
    def get_client_code_without_comments(absolute_path_to_file: str) -> str:
        """
        Remove the comments in the buggy java file, return the no-comment version as string
        """
        jar_path= Path(__file__).parent / "libs" / "comment_remover" / "target" / "remove-comments-1.0-SNAPSHOT-jar-with-dependencies.jar"
        result = subprocess.run(
            ["java", "-jar", jar_path, absolute_path_to_file],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"Error running jar:\n{result.stderr}")
        return result.stdout
    
    
    for context in input_dir.glob("*/new_context.json"):
        
        context_dict = json.loads(context.read_text())
        
        project = context_dict.get("project", "")
        breaking_changes = context_dict.get("breakingChanges", "")
        api_additions = context_dict.get("apiAdditions", "")
        library_name = context_dict.get("libraryName", "")
        library_group_id = context_dict.get("libraryGroupID", "")
        new_version = context_dict.get("newVersion", "")
        old_version = context_dict.get("previousVersion", "")
        breaking_commit = context_dict.get("breakingCommit", "")
        
        for file, errors in context_dict["buggyFiles"].items():
            prompt = {}
            sample_counter += 1
            # aggregate information
            breaking_changes_str = aggregate_breaking_changes(breaking_changes, api_additions)
            error_messages_str = aggregate_error(errors)
            
            # absolute path to the buggy file in the container
            absolute_path_to_file_in_container = Path(file)
            absolute_path_to_file = Path(input_dir / context_dict.get("breakingCommit") / str(absolute_path_to_file_in_container.relative_to(absolute_path_to_file_in_container.anchor)))
            prompt["absolute_path_to_file_in_container"] = str(absolute_path_to_file_in_container)
            
            # get client code
            # client_code = get_client_code_without_comments(absolute_path_to_file)
            try:
                client_code = absolute_path_to_file.read_text()
            except Exception as e:
                print(f"failed when creating sample {sample_counter}")
                raise e
            
            # get buggy lines
            buggy_line = get_buggy_lines(str(absolute_path_to_file), errors)
            # assign unique ids to errors
            for error in errors:
                error["uid"] = get_error_uid(error["message"], error["additional_info"])
            
            prompt["errors"] = errors                            
            prompt["prompt"] = PROMPT_TEMPLATE.format(client_code=client_code, buggy_line=buggy_line, error_message=error_messages_str, api_diff=breaking_changes_str, library_name=library_name, old_version=old_version, new_version=new_version)
            prompt["buggy_lines"] = buggy_line
            prompt["error_message"] = error_messages_str
            prompt["api_diff"] = breaking_changes_str
            prompt["original_code"] = client_code
            prompt["project"] = project
            prompt["libraryName"] = library_name
            prompt["libraryGroupID"] = library_group_id
            prompt["newVersion"] = new_version
            prompt["previousVersion"] = old_version
            prompt["breakingCommit"] = breaking_commit
            
            with open(prompt_dir / f"{sample_counter}.json", "w") as f:
                json.dump(prompt, f, indent=4)
            
                
def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    parser = argparse.ArgumentParser(description="build file-level prompt for each project")
    parser.add_argument("--input", "-i", type=Path,
                        default=Path(__file__).parent.parent / "data" / "dataset",
                        help="Path to dataset root containing context.json files")
    args = parser.parse_args(argv)

    # Build components
    build_prompts(args.input)


if __name__ == "__main__":
    main()
    



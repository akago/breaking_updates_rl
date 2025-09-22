
import argparse
from pathlib import Path
import sys
import logging
import json

from pipeline.constants.constants import LOGGING_FORMAT, DEBUG_LEVEL
from pipeline.constants.constants import PROMPT_TEMPLATE


logging.basicConfig(level=DEBUG_LEVEL, format=LOGGING_FORMAT)


def build_prompts(input_dir: Path) -> None:
    # Build prompt with file
    prompt_dir = input_dir.parent / "prompts"
    sample_counter = 0
    
    def aggregate_breaking_changes(bcs:list[dict], delimiter=" | ") -> str:
        if not bcs:
            return ""
        
        keys_join = delimiter.join(["element", "nature", "kind"])
        header = f"Format: {keys_join}"
        lines = [header]
        
        for bc in bcs:
            vals = [bc.get(k, "") for k in ["element", "nature", "kind"]]
            line = delimiter.join(vals)
            lines.append(line)
        return "\n".join(lines)
    
    def aggregate_error(errors:list[dict]) -> str:
        lines =[]
        for error in errors:
            line = error["message"] + error["additional_info"] 
            lines.append(line)
        return "\n".join(lines)
        
    for context in input_dir.glob("*/context.json"):
        context_dict = json.loads(context.read_text())
        
        project = context_dict.get("project")
        breaking_changes = context_dict.get("breakingChanges")
        library_name = context_dict.get("libraryName")
        library_group_id = context_dict.get("libraryGroupID")
        new_version = context_dict.get("newVersion")
        old_version = context_dict.get("previousVersion")
        
        for file, errors in context_dict["buggyFiles"].items():
            prompt = {}
            # aggregate information
            breaking_changes_str = aggregate_breaking_changes(breaking_changes)
            error_messages_str = aggregate_error(errors)
            
            absolute_path_to_file = Path(file)
            prompt["relative_path"] = str(absolute_path_to_file.relative_to(absolute_path_to_file.anchor))
            client_code = Path(input_dir / context_dict.get("breakingCommit") / prompt["relative_path"]).read_text()
            buggy_line = ""
            
            prompt["prompt"] = PROMPT_TEMPLATE.format(client_code=client_code, buggy_line=buggy_line, error_message=error_messages_str, api_diff=breaking_changes_str)
            
            prompt["project"] = project
            prompt["libraryName"] = library_name
            prompt["libraryGroupID"] = library_group_id
            prompt["newVersion"] = new_version
            prompt["previousVersion"] = old_version
            
            with open(prompt_dir / f"{sample_counter}.json", "w") as f:
                json .dump(prompt, f, indent=4)
                sample_counter += 1
                
def main(argv: list[str] | None = None) -> None:
    """Unified entry point for both training and evaluation.

    Training and evaluation share the same runner/loop. The only
    difference is controlled by the `--mode` flag.
    """
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
    



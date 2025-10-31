from pathlib import Path
import json
import logging
import re
import sys
import argparse
from pipeline.types.metrics import Patcher

    
def extract_java_code(text: str, before_marker: str | None = "<end_of_turn>"):
    JAVA_BLOCK = re.compile(
        r"```java\s*\n(.*?)\n```",  
        flags=re.DOTALL | re.IGNORECASE
    )
    if before_marker and before_marker in text:
        text = text.split(before_marker, 1)[0]
    matches = JAVA_BLOCK.findall(text)
    if matches:
        return matches[0].strip()
    else:
        return ""

def patch_and_evaluate_project(input:str) -> dict:
    """
    Evaluate the results at project level, i.e., compile and test only after applying all patches belonging to the same breaking update 
    """
    input_path = Path(input)
    
    successful_fixes = 0
    total_projects = 0
    total_original_error_count = 0
    total_fixed_error_count = 0
    total_original_file_count = 0
    total_fixed_file_count = 0
    total_new_errors_count = 0
    for folder in input_path.iterdir():
        if not folder.is_dir():
            continue
        patches_to_bind = []
        resources_path =  Path(__file__).parent.parent / "data" / "dataset" / str(folder.name)
        container_path = resources_path / f"{str(folder.name)}.sif"
        original_errors = {}
        for completion_file in folder.glob("*.json"):
            if not completion_file.stem.isdigit():
                continue    
            
            result_dict = json.loads(completion_file.read_text())
            
            # get neccessary information
            errors_in_file = result_dict["errors"]
            original_errors[result_dict["absolute_path_to_file_in_container"]] = errors_in_file
            buggy_file_name = Path(result_dict["absolute_path_to_file_in_container"]).name
            
            # create the fixed java file
            temp_file_path = folder / buggy_file_name
            patch = result_dict.get("patch", "")
            # LLM failed to generate patch
            if patch == "":
                logging.warning(f"No patch found in {completion_file}")
                continue
            
            # java_code = extract_java_code(patch)
            java_code = patch
            # failed to extract java code from completions
            if java_code == "":
                logging.warning(f"Failed to extract java code from {completion_file}")
                continue
                
            temp_file_path.write_text(java_code)
            
            # (patch file on host, buggy file in the container)
            patches_to_bind.append((str(temp_file_path), result_dict["absolute_path_to_file_in_container"]))

        patcher = Patcher(project=result_dict["project"], container_path=str(container_path), log_path=str(folder / f"{str(folder.name)}.log"), binding_pairs=patches_to_bind)

        errorlog, success = patcher.apply_patch()
        metrics = patcher.metrics(original_errors, errorlog.to_jsonable(), success)
        # save metrics
        metrics_file = folder / "metrics.json"
        with open(metrics_file, "w") as mf:
            json.dump(metrics, mf, indent=2)
        logging.info(f"\nMetrics saved to {metrics_file}")
        logging.info(f"Metrics for project {result_dict['project']}: {metrics}\n")
        
        total_original_error_count += metrics["original_error_count"]
        total_fixed_error_count += metrics["fixed_error_count"]
        total_original_file_count += metrics["original_file_count"]
        total_fixed_file_count += metrics["fixed_file_count"]
        total_new_errors_count += metrics["new_errors_count"]
        if metrics["build_success"]:
            successful_fixes += 1
        total_projects += 1

    statistics = {
        "successful_fixes": successful_fixes,
        "total_projects": total_projects,
        "total_original_error_count": total_original_error_count,
        "total_fixed_error_count": total_fixed_error_count,
        "total_original_file_count": total_original_file_count,
        "total_fixed_file_count": total_fixed_file_count,
        "BuildSuccessRate": successful_fixes / total_projects if total_projects > 0 else 0.0,
        "FileFixSuccessRate": total_fixed_file_count / total_original_file_count if total_original_file_count > 0 else 0.0,
        "CompilationErrorFixRate": total_fixed_error_count / total_original_error_count if total_original_error_count > 0 else 0.0,
        "RelativeErrorFixRatio" : (total_fixed_error_count - total_new_errors_count) / total_original_error_count if total_original_error_count > 0 else 0.0
    }
    logging.info(f"Overall statistics: {statistics}")
    with open(str(input_path / "overall_statistics.json"), "w") as sf:
        json.dump(statistics, sf, indent=2)
    return statistics



def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="evalute the completions")
    parser.add_argument("--input", "-i", type=Path,
                        default=Path(__file__).parent.parent / "results" / "unsloth" / "gemma-3-4b-it-unsloth-bnb-4bit_20251029-030040",
                        help="Path to result folder containing completions")
    args = parser.parse_args(argv)
    
    patch_and_evaluate_project(args.input)
    
    
    
if __name__ == "__main__":
    main()
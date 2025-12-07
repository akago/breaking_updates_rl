from pathlib import Path
import json
import logging
import re
import sys
import argparse
from pipeline.types.maven_error import MavenErrorLog, MavenErrorParser
from pipeline.types.metrics import Patcher
from pipeline.types.utils import is_java_source_valid, remove_trailing_whitespace

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
            
            print(f"loading completion file: {completion_file}")
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
            
            if "\n=======\n" in java_code:
                logging.warning(f"Invalid patch with ======= in {completion_file}")
                continue
            
            # when llm generates multiple code snippets into one REPLACE block
            if not is_java_source_valid(java_code):
                logging.warning(f"Generated java code is not valid in {completion_file}")
                continue
            try:
                clean_code = remove_trailing_whitespace(java_code)
            except Exception as e:
                logging.warning(f"Error while removing trailing whitespace in {completion_file}: {e}")
                continue
            if clean_code != "":
                java_code = clean_code
            
            temp_file_path.write_text(java_code)
            
            # (patch file on host, buggy file in the container)
            patches_to_bind.append((str(temp_file_path), result_dict["absolute_path_to_file_in_container"]))

        patcher = Patcher(project=result_dict["project"], container_path=str(container_path), log_path=str(folder / f"{str(folder.name)}.log"), binding_pairs=patches_to_bind)

        
        build_log, success = patcher.apply_patch()
        
        # try to run tests only if complation succeeds
        if success:
            _, success = patcher.apply_patch_with_test()                
            
        # handling cases for checkstyle errors
        if "Failed to execute goal org.apache.maven.plugins:maven-checkstyle-plugin" in build_log:
            error_log = original_errors
        else:
            # analyze the log and get maven errors
            log_parser = MavenErrorParser()
            error_log = MavenErrorLog.from_string(build_log, log_parser).to_jsonable()
        
        metrics = patcher.metrics(original_errors, error_log, success)
        # save metrics
        metrics_file = folder / "metrics_new.json"
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
    with open(str(input_path / "overall_statistics_new.json"), "w") as sf:
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
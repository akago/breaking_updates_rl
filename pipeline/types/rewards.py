from pathlib import Path
from pipeline.types.metrics import Patcher
from pipeline.types.utils import extract_sr_edits, get_patched_content_from_diffs
from pipeline.types.maven_error import MavenErrorLog, MavenErrorParser
from pipeline.constants.constants import RESOURCES_PATH, DATASET_DIFF_PATH, DATASET_FULL_GENERATION_PATH
from datetime import datetime, timezone


def check_format(prompts, completions, **kwargs):
    # <think> and ```java``` blocks with search-replace
    return 1.0

def check_tag(prompts, completions, **kwargs0):
    return 1.0

def reward_func_diff_dense(prompts, completions, completion_ids, breakingCommit, project, absolute_path_to_file_in_container, errors, **kwargs):
    """
    
    
    """    
    # print(f"Prompt lengths for this batch: {[tokenizer(prompt[0]['content'], return_tensors = "pt")['input_ids'].shape for prompt in prompts]}")
    # print(f"Completion lengths for this batch: {[len(completion_id) for completion_id in completion_ids]}")
    rewards = []
    
    for completion, breaking_commit, project_name, file_path_in_container, maven_errors in zip(completions, breakingCommit, project, absolute_path_to_file_in_container, errors):
        reward = 0.0
        response = completion[0]["content"]
        diffs = extract_sr_edits(response)
        patch = get_patched_content_from_diffs(diffs, )
        # Failed to generate proper format
        if patch == "":
            rewards.append(-1.0)
            continue
            
        # Logging with timestamp
        print(f"INFO {datetime.now(timezone.utc)} Compiling the breaking commit {breaking_commit} of project {project_name} to get the rewards...")
        print(f"INFO {datetime.now(timezone.utc)} The Patch: {patch}")
        
        # Get the build log for patched project
        container_path = RESOURCES_PATH / str(breaking_commit) / f"{breaking_commit}.sif"
        patcher = Patcher(project=project_name, container_path=str(container_path))
        build_log, success = patcher.apply_patch_training(patch=patch, container_file=file_path_in_container)
        
        # Test only if the compilation success
        if success:
            _, test_success = patcher.apply_patch_training_test(patch=patch, container_file=file_path_in_container)
            if not test_success:
                rewards.append(0.5) 
            else:
                rewards.append(1.0)
            continue
        
        # Parse Maven Errors
        log_parser = MavenErrorParser()
        error_log = MavenErrorLog.from_string(build_log, log_parser)
        original_error_count, fixed_error_count, new_errors_count = patcher.get_metrics({file_path_in_container : maven_errors}, error_log.to_jsonable())
        
        # Calculate the reward regarding errors
        reward += float(fixed_error_count / original_error_count) - float(new_errors_count / original_error_count)
        
        rewards.append(reward)
        print(f"INFO {datetime.now(timezone.utc)} Reward: {rewards[-1]}")
    return rewards
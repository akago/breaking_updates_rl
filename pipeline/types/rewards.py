from pathlib import Path
from pipeline.types.metrics import Patcher
from pipeline.types.utils import extract_sr_edits, get_patched_content_from_diffs
from pipeline.types.maven_error import MavenErrorLog, MavenErrorParser
from pipeline.constants.constants import RESOURCES_PATH, DATASET_DIFF_PATH, DATASET_FULL_GENERATION_PATH
from datetime import datetime, timezone
import re
import math

reasoning_start = "<think>"
reasoning_end = "</think>"

match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"```java(.+?)```"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

search_replace_format = re.compile(
    r"""^<<<<<<< SEARCH[ \t]*\r?\n(.*?)^=======[ \t]*\r?\n(.*?)^>>>>>>> REPLACE""", 
    re.MULTILINE | re.DOTALL,
)

def reward_check_format(completions, **kwargs):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Match if format is seen exactly!
        if match_format.search(response) is not None: score += 1.0
        # At least one search-replace block
        if search_replace_format.search(response) is not None: score += 1.0
        scores.append(score)
        if score < 1.0:
            print(f"[INFO] Failed format check, reward {score}: {response}")            
    return scores

def reward_check_tag(completions, **kwargs0):
    scores = []
    for completion in completions:
        score = 0
        response = completion[0]["content"]
        # Count how many keywords are seen - we penalize if too many!
        # If we see 1, then plus some points!
        score += 0.5 if response.count(reasoning_start) == 1 else 0
        score += 0.5 if response.count(reasoning_end)   == 1 else 0
        if score < 1.0:
            print(f"[INFO] Failed tag check, reward {score}: {response}")
        scores.append(score)
    return scores

def reward_func_diff_sparse(prompts, completions, completion_ids, breakingCommit, project, absolute_path_to_file_in_container, errors, original_code, **kwargs):
    # print(f"Prompt lengths for this batch: {[tokenizer(prompt[0]['content'], return_tensors = "pt")['input_ids'].shape for prompt in prompts]}")
    # print(f"Completion lengths for this batch: {[len(completion_id) for completion_id in completion_ids]}")
    rewards = []
    
    for completion, breaking_commit, project_name, file_path_in_container, maven_errors, code in zip(completions, breakingCommit, project, absolute_path_to_file_in_container, errors, original_code):
        reward = 0.0
        response = completion[0]["content"]
        print(f"\n[INFO] Raw Completion: {response}")
        diffs = extract_sr_edits(response)
        print(f"\nThe diffs extracted: {diffs}")
        patch = get_patched_content_from_diffs(diffs, code)
        # print(f"The patch extracted: {patch}")
        # Logging with timestamp
        print(f"INFO {datetime.now(timezone.utc)} Compiling the breaking commit {breaking_commit} of project {project_name} to get the rewards...")
        # Get the build log for patched project
        container_path = RESOURCES_PATH / str(breaking_commit) / f"{breaking_commit}.sif"
        patcher = Patcher(project=project_name, container_path=str(container_path))
        build_log, success = patcher.apply_patch_training_test(patch=patch, container_file=file_path_in_container)
        if success:
            reward = 1.0
        rewards.append(reward)
        print(f"INFO {datetime.now(timezone.utc)} Reward: {rewards[-1]}")
    return rewards


def reward_func_diff_dense(prompts, completions, completion_ids, breakingCommit, project, absolute_path_to_file_in_container, errors, original_code, **kwargs):
    # print(f"Prompt lengths for this batch: {[tokenizer(prompt[0]['content'], return_tensors = "pt")['input_ids'].shape for prompt in prompts]}")
    # print(f"Completion lengths for this batch: {[len(completion_id) for completion_id in completion_ids]}")
    rewards = []
    
    for completion, breaking_commit, project_name, file_path_in_container, maven_errors, code in zip(completions, breakingCommit, project, absolute_path_to_file_in_container, errors, original_code):
        reward = 0.0
        response = completion[0]["content"]
        print(f"\n[INFO] Raw Completion: {response}")
        diffs = extract_sr_edits(response)
        print(f"\nThe diffs extracted: {diffs}")
        patch = get_patched_content_from_diffs(diffs, code)
        # print(f"The patch extracted: {patch}")
        # Failed to generate proper format
        # if patch == "":
        #     rewards.append(-1.0)
        #     continue
            
        # Logging with timestamp
        print(f"INFO {datetime.now(timezone.utc)} Compiling the breaking commit {breaking_commit} of project {project_name} to get the rewards...")
        # print(f"INFO {datetime.now(timezone.utc)} The Patch: {patch}")
        
        # Get the build log for patched project
        container_path = RESOURCES_PATH / str(breaking_commit) / f"{breaking_commit}.sif"
        patcher = Patcher(project=project_name, container_path=str(container_path))
        build_log, success = patcher.apply_patch_training(patch=patch, container_file=file_path_in_container)
        
        # Test only if the compilation success
        test_success_flag = 0.0
        if success:
            _, test_success = patcher.apply_patch_training_test(patch=patch, container_file=file_path_in_container)
            if test_success:
                test_success_flag = 1.0
                        
        # Parse Maven Errors
        log_parser = MavenErrorParser()
        error_log = MavenErrorLog.from_string(build_log, log_parser)
        original_errors_count, fixed_errors_count, new_errors_count = patcher.get_metrics({file_path_in_container : maven_errors}, error_log.to_jsonable())
        
        # Reward for new errors
        # decay = 0.5  # 0.5 means each new error half the reward
        # alpha = math.log(1.0 / decay)
        # new_errors_score = math.exp(-alpha * new_errors_count)   # ∈(0,1]
        # set to < 1 to encourge making changes even with introduction of new errors
        rho = 0.9 
        if fixed_errors_count == 0 and new_errors_count == 0:
            new_errors_rel_score = 0.5
        else:
            new_errors_rel_score = fixed_errors_count / max(fixed_errors_count + rho * new_errors_count, 1e-9)
        # Reward for fixed errors
        fixed_score = fixed_errors_count / original_errors_count # ∈(0,1]
        # Reward for test
        test_score = float(test_success_flag)                    # ∈(0,1]
        # Final reward
        reward = 0.3 * fixed_score  + 0.3 * new_errors_rel_score + 0.4 * test_score
        
        rewards.append(reward)
        print(f"INFO {datetime.now(timezone.utc)} Reward: {rewards[-1]}")
    return rewards


def reward_func_dense(prompts, completions, completion_ids, breakingCommit, project, absolute_path_to_file_in_container, errors, **kwargs):
    """
    if package decalaration removed: penalty -0.6
    
    """    
    print(f"Prompt lengths for this batch: {[tokenizer(prompt[0]['content'], return_tensors = "pt")['input_ids'].shape for prompt in prompts]}")
    print(f"Completion lengths for this batch: {[len(completion_id) for completion_id in completion_ids]}")
    rewards = []
    
    for completion, breaking_commit, project_name, file_path_in_container, maven_errors in zip(completions, breakingCommit, project, absolute_path_to_file_in_container, errors):
        reward = 0.0
        response = completion[0]["content"]
        patch = extract_java_code_gemma(response)
        # Failed to generate proper format
        if patch == "":
            rewards.append(-1.0)
            continue
        # Deleting package declaration should be punished
        if package_removed(patch):
            reward -= 0.6
            
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
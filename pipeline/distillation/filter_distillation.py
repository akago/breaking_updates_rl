
from __future__ import annotations
import os
import sys
from datasets import load_dataset
from matplotlib import patches
from openai import OpenAI
from pathlib import Path
from pipeline.constants.constants import SYSTEM_PROMPT, RESOURCES_PATH, DATASET_DIFF_PATH, SFT_DATASET_PATH
from pipeline.types.utils import extract_sr_edits, get_patched_content_from_diffs
from pipeline.types.metrics import Patcher
from pipeline.types.maven_error import MavenErrorLog, MavenErrorParser
from dotenv import load_dotenv
import json
import logging

logging.basicConfig(level=logging.INFO)


SFT_JSON_PATH = SFT_DATASET_PATH / "sft_data.jsonl"


def filter_test_success(patches_per_bu: dict) -> tuple[dict, dict]:
    """
    Filter patches that pass all tests at project level
    param: patches_per_bu: dict mapping breaking commit to list of patches
    returns: (successful_patches, failed_patches)
    """
    successful_patches = {}
    failed_patches = {}
    for breaking_commit, patches in patches_per_bu.items():
        logging.info(f"Processing breaking commit: {breaking_commit} with {len(patches)} patches")

        patches_to_bind = []
        for patch_info in patches:
            original_code = patch_info["original_code"]
            raw_response = patch_info["label"]
            diffs = extract_sr_edits(raw_response)
            logging.info(f"diffs extracted: {diffs}")
            # get patch string
            patch_str = get_patched_content_from_diffs(diffs, original_code)
            # bind patch string, file path
            patches_to_bind.append((patch_str, patch_info["absolute_path_to_file_in_container"]))
        
        project_name = patches[0]["project"]    
        container_path = RESOURCES_PATH / breaking_commit / f"{breaking_commit}.sif"
        patcher = Patcher(project=project_name, container_path=str(container_path))
        build_log, success = patcher.apply_patches_training_test(patches_to_bind)
        

        if success:
            successful_patches[breaking_commit] = patches
            logging.info(f"Patches for breaking commit {breaking_commit} passed all tests.")
        else:
            log_parser = MavenErrorParser()
            errors_by_file = MavenErrorLog.from_string(build_log, log_parser).to_jsonable()
            failed_patches[breaking_commit] = patches
            logging.info(f"Patches for breaking commit {breaking_commit} failed tests with errors: {build_log}")
    return successful_patches, failed_patches

def filter_api_used(patches_per_bu: dict) -> tuple[dict, dict]:
    return patches_per_bu, {}
    

def aggragate_by_bu(patches_dataset:list[dict], to_save:bool=False) -> dict:
    bu_dict = {}
    for item in patches_dataset:
        bu_dict.setdefault(item["breakingCommit"], []).append(item)
    if to_save:
        with open(SFT_DATASET_PATH / "sft_by_bu.jsonl", "w", encoding="utf-8") as f:
            f.write(json.dumps(bu_dict, indent=2))
            
    return bu_dict

def main():
    
    # load pseudo patches
    pseudo_patches = load_dataset("json", data_files=str(SFT_JSON_PATH), split="train")
    
    # # merge_datessets
    # train_dataset = load_dataset("json", data_files=str(DATASET_DIFF_PATH / "train.jsonl"), split="train")
    # pseudo_patches_new = []
    # for pseudo_patch in pseudo_patches:
    #     for train_sample in train_dataset:
    #         if pseudo_patch["prompt"] == train_sample["prompt"]:
    #             train_sample["label"] = pseudo_patch["response_text"] 
    #             train_sample["model"] = pseudo_patch["model"]
    #             train_sample["accepted"] = pseudo_patch["accepted"]
    #             pseudo_patches_new.append(train_sample)
    #             break
    
    
    # pseudo_patches_new_file = SFT_DATASET_PATH / "sft_data_updated.jsonl"
    # pseudo_patches_new_file.write_text("\n".join([json.dumps(item) for item in pseudo_patches_new]), encoding="utf-8")
    
    
    # aggregate at project level
    patches_per_bu = aggragate_by_bu(pseudo_patches,to_save=True)
    
    logging.info("number of breaking updates:", len(patches_per_bu))

    # Filtering
    patches_per_bu_pass_test, patches_per_bu_failed_test = filter_test_success(patches_per_bu)
    patches_per_bu_api_used, _ = filter_api_used(patches_per_bu_pass_test)
    
    # Logging
    logging.info("Summary of Filtering Results:")
    logging.info(f"Total patches: {len(pseudo_patches)}")
    logging.info(f"Successful project: {len(patches_per_bu_pass_test)}")
    logging.info(f"Failed project: {len(patches_per_bu_failed_test)}")
    logging.info(f"API used patches: {len(patches_per_bu_api_used)}")
    
    # Save results
    filtered_patches_file = SFT_DATASET_PATH / "sft_filtered_bu_patches.jsonl"
    filtered_patches_file.write_text(json.dumps(patches_per_bu_api_used, indent=2), encoding="utf-8")
    failed_patches_file = SFT_DATASET_PATH / "sft_failed_bu_patches.jsonl"
    failed_patches_file.write_text(json.dumps(patches_per_bu_failed_test, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
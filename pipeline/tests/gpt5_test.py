from openai import OpenAI
from pathlib import Path
from pipeline.constants.constants import SYSTEM_PROMPT, RESOURCES_PATH, DATASET_DIFF_PATH
from pipeline.types.utils import extract_sr_edits, get_patched_content_from_diffs
from pipeline.types.metrics import Patcher
from pipeline.types.maven_error import MavenErrorLog, MavenErrorParser
from dotenv import load_dotenv
import json
import unittest

print(load_dotenv())


class DistillationTest(unittest.TestCase):
    
    def test_pipeline(self):
        prompt = json.loads(Path("/home/xchen6/breaking_updates_rl/data/prompts_diff/10.json").read_text())

        client = OpenAI()
        resp = client.responses.create(
                        model="gpt-5",
                        input=SYSTEM_PROMPT + prompt["prompt"],
        )

        raw_response = resp.output_text
        print(raw_response)

        breaking_commit = prompt["breakingCommit"]
        project_name = prompt["project"]
        absolute_file_path_in_container = prompt["absolute_path_to_file_in_container"]
        container_path = RESOURCES_PATH / breaking_commit / f"{breaking_commit}.sif"
        patcher = Patcher(project=project_name, container_path=str(container_path))
        # original_code_path = RESOURCES_PATH / breaking_commit / project_name / 
        orignial_code = prompt["original_code"]

        print(f"original code:{orignial_code}")
        diffs = extract_sr_edits(raw_response)
        print(f"diffs extracted: {diffs}")
        patch = get_patched_content_from_diffs(diffs, orignial_code)
        print(f"patch filled: {patch}")
        build_log, success = patcher.apply_patch_training(patch, container_file=absolute_file_path_in_container)

        log_parser = MavenErrorParser()
        errors_by_file = MavenErrorLog.from_string(build_log, log_parser).to_jsonable()
        if absolute_file_path_in_container in errors_by_file:
            print(f"{errors_by_file[absolute_file_path_in_container]}")
            
        self.assertNotIn(absolute_file_path_in_container, errors_by_file)

if __name__ == "__main__":
    unittest.main()

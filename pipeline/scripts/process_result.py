import pathlib
import json
from pipeline.types.utils import extract_java_code
result_path = pathlib.Path(__file__).parent.parent.parent / "results" / "google" / "gemma-3-4b-it_20251007-084909"


for sample in result_path.rglob("*.json"):
    # if the file name only contains digits, process
    if not sample.stem.isdigit():
        continue 
    print(f"Processing result file: {sample}")
    with open(sample, "r") as f:
        json_dict = json.loads(f.read())
        # process the prompt to be in message format
        json_dict["raw_completion"] = json_dict["patch"]
        json_dict["patch"] = extract_java_code(json_dict["patch"])    
    with open(sample, "w") as wf:
        json.dump(json_dict, wf, indent=4)

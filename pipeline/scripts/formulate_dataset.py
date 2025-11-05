import pathlib
import json

prompts_path = pathlib.Path(__file__).parent.parent.parent / "data" / "prompts_diff"


def wrap_message(raw_prompt:str)->list[dict]:
    return [
        {"content": raw_prompt, "role": "user"}
    ]
    

data_dict = {"data": []}
for sample in prompts_path.glob("*.json"):
    # if the file name only contains digits, process
    if not sample.stem.isdigit():
        continue 
    print(f"Processing sample file: {sample}")
    with open(sample, "r") as f:
        json_dict = json.loads(f.read())
        # process the prompt to be in message format
        json_dict["prompt"] = json_dict["prompt"]
        data_dict["data"].append(json_dict)
            

dataset_file = prompts_path / f"dataset.json"
with open(dataset_file, "w") as dataset:
    json.dump(data_dict["data"], dataset, indent=4)
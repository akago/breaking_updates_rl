# Load model directly
import argparse
import json
import sys
from datetime import datetime
import logging
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.device_count() >= 1:
    torch.cuda.set_device(0)
    print("current_device:", torch.cuda.current_device(), torch.cuda.get_device_name(0))

from pipeline.llms.Gemma3 import Gemma3
# from pipeline.llms.Llama3 import Llama3

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
import os
from huggingface_hub import login

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)
token = os.getenv("HUGGINGFACE_HUB_TOKEN")
login(token=token)

def wrap_prompt(user_prompt:str) -> str:
    prompt = f"<s>[INST] <<SYS>>\nYou are a helpful coding assistant.\n<</SYS>>\n"
    prompt += f"{user_prompt}\n[/INST]"
         
    return prompt

def extract_response(text: str) -> str:
    # get the content after the last [/INST]
    if "[/INST]" in text:
        return text.split("[/INST]")[-1].strip()
    return text.strip()

def evaluate(input: Path, model_id: str) -> None:
    output_root = input.parent.parent / "results" / "_".join([model_id, datetime.now().strftime("%Y%m%d-%H%M%S")])
    
    if "gemma" in model_id:
        llm = Gemma3(model_id)
    elif "llama" in model_id:
        llm = Llama3(model_id)
    logger.info(f"The model is loaded into {llm.get_model().device}")
    
    for i, p_json in enumerate(input.glob("*.json")):
        prompt_dict = json.loads(p_json.read_text())
        buggy_file_name = Path(prompt_dict["absolute_path_to_file_in_container"]).stem
        output_path = output_root / prompt_dict["breakingCommit"] / f"{i+1}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            logger.info(f'generating patch for {prompt_dict["project"]}/{prompt_dict["breakingCommit"]}/{buggy_file_name}')
            completion = llm.generate(prompt_dict)
        except Exception as e:
            raise e
        
        result = prompt_dict 
        result["patch"] = completion
        with open(str(output_path), "w") as f:
            json.dump(result, f, indent=4)

def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="evalute the dataset")
    parser.add_argument("--input", "-i", type=Path,
                        default=Path(__file__).parent.parent / "data" / "dataset",
                        help="Path to dataset root containing context.json files")
    parser.add_argument("--model", "-m", type=str,
                        default="google/gemma-3-4b-it",
                        help="The model name or path")
    
    args = parser.parse_args(argv)
    
    evaluate(args.input, args.model)
    
    

if __name__ == "__main__":
    main()
    
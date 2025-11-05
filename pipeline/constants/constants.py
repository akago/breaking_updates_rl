import logging
import os
from pathlib import Path


# Logging configuration
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEBUG_LEVEL = logging.INFO


# Prompt template
with open("pipeline/prompts/bacardi_prompt.txt", "r") as f:
   PROMPT_TEMPLATE = f.read()
with open("pipeline/prompts/bacardi_prompt_CoT.txt", "r") as f:
   COT_PROMPT_TEMPLATE = f.read()
# with open("pipeline/prompts/unifed_diff_prompt_CoT.txt", "r") as f:
#    UNIFIED_PROMPT_TEMPLATE = f.read()
with open("pipeline/prompts/system_prompt.txt", "r") as f:
   SYSTEM_PROMPT = f.read()
with open("pipeline/prompts/user_prompt_template.txt", "r") as f:
   USER_PROMPT_TEMPLATE = f.read()
# Maximum token length for the model
MAX_TOKEN_LENGTH = 4096

# Hugging Face cache directory (settable via env on HPC)
# Priority: HF_HOME > HUGGINGFACE_HUB_CACHE > default user cache
HF_CACHE_DIR = (
    os.environ.get("HF_HOME")
    or os.environ.get("HUGGINGFACE_HUB_CACHE")
    or str(Path("~/.cache/huggingface").expanduser())
)

DATA_PATH = Path(__file__).parent.parent.parent / "data"
RESOURCES_PATH = DATA_PATH / "dataset"
DATASET_FULL_GENERATION_PATH = DATA_PATH / "prompts"
DATASET_DIFF_PATH = DATA_PATH / "prompts_diff"
SFT_DATASET_PATH = DATA_PATH / "sft"
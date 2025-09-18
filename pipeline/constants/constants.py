import logging
import os
from pathlib import Path


# Logging configuration
LOGGING_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEBUG_LEVEL = logging.INFO

# Prompt template
with open("pipeline/prompts/bacardi_prompt.txt", "r") as f:
   PROMPT_TEMPLATE = f.read()
   
# Maximum token length for the model
MAX_TOKEN_LENGTH = 4096

# Hugging Face cache directory (settable via env on HPC)
# Priority: HF_HOME > HUGGINGFACE_HUB_CACHE > default user cache
HF_CACHE_DIR = (
    os.environ.get("HF_HOME")
    or os.environ.get("HUGGINGFACE_HUB_CACHE")
    or str(Path("~/.cache/huggingface").expanduser())
)

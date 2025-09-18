
from __future__ import annotations
from pathlib import Path


"""Legacy patcher classes.

These are kept for compatibility but not used directly in the unified runner.
Prefer using `pipeline.agents.llm_policy.LLMPolicy` which integrates with the
shared prompt and can be wired into RL training.
"""

try:
    # Avoid import-time failures in restricted environments.
    from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
except Exception:  # pragma: no cover
    AutoTokenizer = None
    AutoModelForCausalLM = None

from pipeline.constants.constants import LOGGING_FORMAT, DEBUG_LEVEL
from pipeline.types.project import Project
import logging

logging.basicConfig(level=DEBUG_LEVEL, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)



class Patcher:
    def __init__(self, llm:str, project:Project):
        self.llm = llm
        self.tokenizer = AutoTokenizer.from_pretrained(llm)
        self.model = AutoModelForCausalLM.from_pretrained(llm)
        self.project = project
        self.patches = {}
        self.hyperparams = {}

    def generate_prompt(self, sample):
        prompt = PROMPT_TEMPLATE.format(client_code=sample.client_code, buggy_line=sample.buggy_line, 
                                        error_message=sample.error_message, api_diff=sample.api_diff)
        return prompt

    def generate_patch(self):
       """Abstract method to generate a patch using the LLM"""
       pass

    def patch(self, sample, **kwargs):
        """Abstract method to apply the patch to the file"""
        pass


class FilePatcher(Patcher):
    def generate_patch(self, buggy_file):
        """Generate a patch using the LLM"""
        pass
    
    def fix_project(self):
        """Fix all buggy files in the project"""
        pass
    
    def fix_file(self, buggy_file):
        """Fix a single buggy file in the project"""
        pass
    
    def patch(self, patch_number:int=0):
        """Apply the newest patch to the file"""
        pass
                    
                    
class ErrorPatcher:

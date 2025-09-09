
from __future__ import annotations
from pathlib import Path


from transformers import AutoTokenizer, AutoModelForCausalLM
from prompts.prompts import PROMPT_TEMPLATE
from constants.constants import LOGGING_FORMAT, DEBUG_LEVEL
import logging

logging.basicConfig(level=DEBUG_LEVEL, format=LOGGING_FORMAT)
logger = logging.getLogger(__name__)

class Patcher:
    def __init__(self, llm):
        self.llm = llm
        self.tokenizer = AutoTokenizer.from_pretrained(llm)
        self.model = AutoModelForCausalLM.from_pretrained(llm)

    def generate_prompt(self, sample):
        prompt = PROMPT_TEMPLATE.format(client_code=sample.client_code, buggy_line=sample.buggy_line, 
                                        error_message=sample.error_message, api_diff=sample.api_diff)
        return prompt

    def generate_patch(self, sample, **kwargs):
        prompt = self.generate_prompt(sample)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, **kwargs)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        

    def patch(self, sample, **kwargs):
        fixed_code = self.generate_patch(sample, **kwargs)
        with open(sample.absolute_path_to_buggy_class, "w") as f:
            f.write(fixed_code)

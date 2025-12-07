from abc import ABC, abstractmethod
from typing import Any, Dict

class BaseLLM(ABC):
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name
        self.params = kwargs # hyper parameters
        
    @abstractmethod
    def format_prompt(self, instruction: str, context: str = "") -> str:
        """Different recommended prompt"""
        pass

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """unified generation interface"""
        pass
    
    @staticmethod
    @abstractmethod
    def extract_java_code(text: str) -> str:
        """Extract java code from the model output"""
        pass
        
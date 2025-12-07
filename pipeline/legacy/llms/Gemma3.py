from pipeline.llms.BaseLLM import BaseLLM
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
import torch
from torch import Tensor
import logging
import re
logger = logging.getLogger(__name__)

class Gemma3(BaseLLM):
    def __init__(self, model_name: str = "google/gemma-3-4b-it", **kwargs):
        super().__init__(model_name, **kwargs)
        
        # 4bit quantization
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,  
        )
        # Load model 
        self.model_id = model_name
        self.processor = AutoProcessor.from_pretrained(model_name, )
        self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,  
                device_map="cuda:0",
                # quantization_config=bnb_cfg,
        )
        self.context_limit = 128000
        logger.info(f"The context limit of {self.model_id}: {self.context_limit} tokens")

    def extract_java_code(self, text: str) -> str:
        end_marker = "<end_of_turn>"
        JAVA_BLOCK = re.compile(
            r"```java\s*\n(.*?)\n```",  
            flags=re.DOTALL | re.IGNORECASE
        )
        if end_marker and end_marker in text:
            text = text.split(end_marker)[0]
        matches = JAVA_BLOCK.findall(text)
        if matches:
            return matches[0].strip()
        else:
            return ""
    
    
    def get_context_limit(self):
        for key in ["max_position_embeddings", "max_seq_len", "seq_length", "n_positions"]:
            if hasattr(self.model.config, key) and isinstance(getattr(self.model.config, key), int):
                return int(getattr(self.model.config, key))
            
    def fit_or_process(
        self,
        tokens: Tensor,
        max_new_tokens: int,
        on_overflow: str,
    ) -> Tensor:

        need = len(tokens) + max_new_tokens
        if need <= self.context_limit:
            return tokens

        if on_overflow == "raise":
            raise ValueError(
                f"Token overflow: input={len(tokens)}, reserve={max_new_tokens}, "
                f"limit={self.context_limit}, excess={need - self.context_limit}"
            )

        target_input_len = max(0, self.context_limit - max_new_tokens)
        if target_input_len < max_new_tokens:
            raise ValueError(
                f"Too small room for input after reserving generation: {target_input_len}"
            )
        # if on_overflow == "truncate_head":
        #     kept_ids = tokens[-target_input_len:]
        #     return self.processor.decode(kept_ids, skip_special_tokens=False)

        # if on_overflow == "truncate_middle":
        #     # keep 0:head_quota and -tail_quota:
        #     head_quota = min(512, target_input_len // 4)  
        #     tail_quota = target_input_len - head_quota
        #     kept_ids = tokens[:head_quota] + tokens[-tail_quota:]
        #     return self.processor.decode(kept_ids, skip_special_tokens=False)

        raise ValueError(f"Unknown overflow policy: {on_overflow}")
        
    def format_prompt(self, prompt:str) -> list[dict]:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            },
        ]
        return messages

    def count_tokens(self, text: str)-> int:
        return len(self.processor.tokenizer(text, add_special_tokens=False)["input_ids"])
    
    
    def generate(self, prompt_dict:dict) -> str:
        messages = self.format_prompt(prompt_dict["prompt"])
        
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        
        # for k, v in inputs.items():
        #     if torch.is_tensor(v):
        #         if not torch.isfinite(v).all():
        #             logger.error(f"Input tensor {k} contains inf or nan!")
        #             return ""
        inputs = inputs.to(self.model.device)
        
        # reserve 1024 tokens of redundancy for patch generation
        src_len = self.count_tokens(prompt_dict.get("original_code", ""))
        reserve = src_len + 1024
        logger.info(f"Expected sequence length for generation: {reserve}")
        # Ensure current inputs can fit after reserving generation space
        try:
            self.fit_or_process(inputs["input_ids"][0], reserve, "raise")
        except ValueError as e:
            logger.error(f"Error when fit the context: {e}")
            return ""
        # Final cap for generation to stay within the context limit
        max_allowed = max(1, self.context_limit - inputs["input_ids"].shape[-1])
        gen_len = max_allowed
        # gen_len = min(max_allowed, reserve)
        
        try:
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, 
                                              max_new_tokens=gen_len, 
                                              do_sample=False,         
                                              )
        except RuntimeError as e:
            logger.error(f"Error when generating the completion: {e}")
            return ""
        
        response = self.processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
        return response
    
    def get_model(self):
        return self.model
    
    def get_tokenizer(self):
        return self.processor
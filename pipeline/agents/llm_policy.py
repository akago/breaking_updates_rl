from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Any, List

from pipeline.constants.constants import PROMPT_TEMPLATE, MAX_TOKEN_LENGTH, HF_CACHE_DIR

try: 
    import torch  
    from transformers import (  
        AutoTokenizer,
        AutoModelForCausalLM,
    )
except Exception: 
    torch = None 
    AutoTokenizer = None  #
    AutoModelForCausalLM = None  

@dataclass
class PolicyOutput:
    """Action produced by the policy.

    For this task we model the action as a textual patch suggestion.
    """
    patch_text: str
    prompt_used: str


class LLMPolicy:
    """
    Policy wrapper around LLM
    """

    def __init__(self, model_name: str, save_dir: Optional[str] = None) -> None:
        self.model_name = model_name
        self.save_dir = save_dir
        self._tokenizer = None
        self._model = None
        self._device = None

    def _lazy_load(self) -> None:
        """Load tokenizer/model on first use.

        Uses `HF_CACHE_DIR` to reuse pre-downloaded models on HPC.
        Falls back to a no-op if transformers/torch are unavailable.
        """
        if self._model is not None and self._tokenizer is not None:
            return
        if AutoTokenizer is None or AutoModelForCausalLM is None:
            return
        
        if torch is not None:
            self._device = (
                torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
            )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, cache_dir=HF_CACHE_DIR)
            self._model = AutoModelForCausalLM.from_pretrained(self.model_name, cache_dir=HF_CACHE_DIR)
            if torch is not None:
                self._model.to(self._device) 
                self._model.eval()
        except Exception:
            self._tokenizer = None
            self._model = None
       
        
    def build_prompt(self, client_code: str, buggy_line: str, error_message: str, api_diff: str) -> str:
        """Build a prompt using the shared template."""
        return PROMPT_TEMPLATE.format(
            client_code=client_code or "",
            buggy_line=buggy_line or "",
            error_message=error_message or "",
            api_diff=api_diff or "",
        )

    def _format_buggy_line(self, file_context: Optional[str], errors: Optional[List[Any]]) -> str:
        """Heuristic extraction of buggy line(s) from file context and error list.

        - If errors are present, show first up to 3 error line excerpts.
        - Otherwise, return empty string.
        """
        if not file_context or not errors:
            return ""
        lines = file_context.splitlines()
        out: list[str] = []
        for err in errors[:3]:
            try:
                ln = getattr(err, "line", None)
                if ln is None or ln <= 0 or ln > len(lines):
                    continue
                code_line = lines[ln - 1].strip()
                out.append(f"L{ln}: {code_line}")
            except Exception:
                continue
        return "\n".join(out)

    def _format_error_message(self, errors: Optional[List[Any]]) -> str:
        if not errors:
            return ""
        msgs: list[str] = []
        for err in errors[:5]:  # take a few to keep prompt short
            try:
                line = getattr(err, "line", None)
                msg = getattr(err, "message", None)
                if msg is None:
                    continue
                prefix = f"[L{line}] " if line is not None else ""
                msgs.append(prefix + str(msg))
            except Exception:
                continue
        return "\n".join(msgs)

    def act(self, obs, deterministic: bool) -> PolicyOutput:
        """Produce a patch suggestion from the observation using an LLM.

        The method is defensive about observation schema to accommodate
        minor differences between env versions (e.g., file_context vs client_code).
        """
        client_code = getattr(obs, "client_code", None) or getattr(obs, "file_context", None) or ""
        api_diff = getattr(obs, "api_diff", None) or ""
        errors = getattr(obs, "errors", None)  # list of ErrorItem, if present
        buggy_line = getattr(obs, "buggy_line", None) or self._format_buggy_line(client_code, errors)
        error_message = getattr(obs, "error_message", None) or self._format_error_message(errors)

        prompt = self.build_prompt(
            client_code=client_code,
            buggy_line=buggy_line,
            error_message=error_message,
            api_diff=api_diff,
        )

        # Try LLM generation
        self._lazy_load()
        if self._model is None or self._tokenizer is None or torch is None:
            # Fallback minimal suggestion
            patch = "// PATCH: Please refactor calls to match the new API."
            return PolicyOutput(patch_text=patch, prompt_used=prompt)

        try:
            with torch.no_grad():  
                messages = [
                    {"role": "user", "content": prompt}
                ]
                inputs = self._tokenizer.apply_chat_template(
                    prompt,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=MAX_TOKEN_LENGTH,
                ).to(self._model.device)
               

                gen_kwargs = {
                    "max_new_tokens": 512,
                    "do_sample": not deterministic,
                    "temperature": 0.2 if not deterministic else 0.0,
                    "top_p": 0.95,
                    "eos_token_id": self._tokenizer.eos_token_id,
                    "pad_token_id": getattr(self._tokenizer, "pad_token_id", self._tokenizer.eos_token_id),
                }
                output_ids = self._model.generate(**inputs, **gen_kwargs)
                text = self._tokenizer.decode(output_ids[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)   
                        
            if text.startswith(prompt):
                text = text[len(prompt):].lstrip()
            patch = text.strip()
            if not patch:
                raise RuntimeError("Failed to generate patch")
            return PolicyOutput(patch_text=patch, prompt_used=prompt)
        except Exception:
            raise RuntimeError("Failed to generate patch")

    def learn(self) -> dict:
        """PlaceHolder

        During eval this is a no-op; during train this may
        run PPO/DPO/GRPO or any update rule based on collected trajectories.
        """
        return {"updated": False}

    def save_checkpoint(self) -> None:
        """Persist model or optimizer states if applicable."""
        # No-op for the placeholder implementation.
        return None

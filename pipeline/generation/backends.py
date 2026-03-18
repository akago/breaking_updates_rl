from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from unsloth import FastModel


@dataclass
class GenerationRuntimeConfig:
    model_id: str
    max_seq_length: int = 35000
    max_input_tokens: int = 32000
    max_new_tokens: int = 4000
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False


class GenerationBackend(Protocol):
    def build_prompt_text(self, messages: list[dict[str, str]]) -> str:
        """Build model-specific prompt text from chat messages."""

    def generate_completion(self, prompt_text: str) -> tuple[str, int]:
        """Return (completion_text, input_token_length)."""


class UnslothGenerationBackend:
    def __init__(self, cfg: GenerationRuntimeConfig) -> None:
        self.cfg = cfg
        self.model, self.tokenizer = FastModel.from_pretrained(
            model_name=cfg.model_id,
            max_seq_length=cfg.max_seq_length,
            load_in_4bit=cfg.load_in_4bit,
            load_in_8bit=cfg.load_in_8bit,
            full_finetuning=cfg.full_finetuning,
        )

    def build_prompt_text(self, messages: list[dict[str, str]]) -> str:
        return self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

    def generate_completion(self, prompt_text: str) -> tuple[str, int]:
        input_len = self.tokenizer(prompt_text, return_tensors="pt")["input_ids"].shape[-1]
        if input_len > self.cfg.max_input_tokens:
            return "", input_len

        response = self.model.generate(
            **self.tokenizer(prompt_text, return_tensors="pt").to("cuda"),
            max_new_tokens=self.cfg.max_new_tokens,
            do_sample=False,
        )
        completion_ids = response[0][input_len:]
        completion = self.tokenizer.decode(completion_ids, skip_special_tokens=True)
        return completion, input_len


def build_generation_backend(name: str, cfg: GenerationRuntimeConfig) -> GenerationBackend:
    backend = name.lower()
    if backend == "unsloth":
        return UnslothGenerationBackend(cfg)
    raise ValueError(f"Unsupported generation backend: {name}")

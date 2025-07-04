import os
from typing import Any

import tiktoken
from transformers import LlamaTokenizer, AutoTokenizer  # type: ignore


class Tokenizer(object):
    def __init__(self, provider: str, model_name: str) -> None:
        if provider == "openai":
            # if "llama" in model_name.lower():
            assert "OPENAI_API_BASE" in os.environ
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # else:
                # self.tokenizer = tiktoken.encoding_for_model(model_name)
        elif provider == "huggingface":
            self.tokenizer = LlamaTokenizer.from_pretrained(model_name)
            # turn off adding special tokens automatically
            self.tokenizer.add_special_tokens = False  # type: ignore[attr-defined]
            self.tokenizer.add_bos_token = False  # type: ignore[attr-defined]
            self.tokenizer.add_eos_token = False  # type: ignore[attr-defined]
        elif provider == "google":
            self.tokenizer = None  # Not used for input length computation, as Gemini is based on characters
        elif provider == "vllm" or provider == "openrouter":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def __call__(self, text: str) -> list[int]:
        return self.tokenizer.encode(text)

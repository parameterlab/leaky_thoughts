"""This module is adapt from https://github.com/zeno-ml/zeno-build"""

try:
    from .providers.gemini_utils import generate_from_gemini_completion
except:
    print(
        "Google Cloud not set up, skipping import of providers.gemini_utils.generate_from_gemini_completion"
    )

from .providers.hf_utils import generate_from_huggingface_completion
from .providers.openai_utils import (
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
)
from .providers.vllm_utils import (
    generate_from_vllm_chat_completion,
    generate_from_vllm_completion,
)

from .providers.openrouter_utils import (
    generate_from_openrouter_chat_completion,
    generate_from_openrouter_completion,
)
from .utils import call_llm, split_by_think, REASONING_MODELS_TO_END_THINK, REASONING_MODELS_TO_START_THINK

__all__ = [
    "generate_from_openai_completion",
    "generate_from_openai_chat_completion",
    "generate_from_huggingface_completion",
    "generate_from_gemini_completion",
    "generate_from_vllm_chat_completion",
    "generate_from_vllm_completion",
    "call_llm",
    "split_by_think",
    "REASONING_MODELS_TO_END_THINK",
    "REASONING_MODELS_TO_START_THINK",
    "generate_from_openrouter_chat_completion",
    "generate_from_openrouter_completion",
]

import argparse
from typing import Any

try:
    from vertexai.preview.generative_models import Image
    from llms import generate_from_gemini_completion
except:
    print(
        "Google Cloud not set up, skipping import of vertexai.preview.generative_models.Image and llms.generate_from_gemini_completion"
    )

from llms import (
    generate_from_huggingface_completion,
    generate_from_openai_chat_completion,
    generate_from_openai_completion,
    generate_from_vllm_chat_completion,
    generate_from_vllm_completion,
    generate_from_openrouter_chat_completion,
    generate_from_openrouter_completion,
    lm_config,
)

APIInput = str | list[Any] | dict[str, Any]

REASONING_MODELS_TO_END_THINK = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "</think>",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "</think>",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "</think>",
    "deepseek-ai/DeepSeek-V3": "</think>",
    "deepseek-ai/DeepSeek-R1": "</think>",
    "Qwen/QwQ-32B": "</think>",
    "simplescaling/s1-32B": "<|im_start|>answer",
    "simplescaling/s1.1-32B": "<|im_start|>answer",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "</think>",
}

REASONING_MODELS_TO_START_THINK = {
    "deepseek-ai/DeepSeek-R1-Distill-Llama-8B": "<think>",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B": "<think>",
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B": "<think>",
    "deepseek-ai/DeepSeek-V3": "<think>",
    "deepseek-ai/DeepSeek-R1": "<think>",
    "Qwen/QwQ-32B": "<think>",
    "simplescaling/s1-32B": "<|im_start|>think",
    "simplescaling/s1.1-32B": "<|im_start|>think",
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": "<think>",
}

MODEL_TO_OR_ID = {
    "deepseek-ai/DeepSeek-R1": "deepseek/deepseek-r1",
    "deepseek-ai/DeepSeek-V3": "deepseek/deepseek-chat",
}


def split_by_think(ans, end_think_token):
    if end_think_token is None:
        return ["", ans]

    chunks = ans.split(end_think_token)

    if len(chunks) == 1:  # No "</think>" found
        return ["", ans]

    # Everything up to and including the last </think>
    left_part = end_think_token.join(chunks[:-1]) + end_think_token

    # Everything after the last </think>
    right_part = chunks[-1]

    return [left_part, right_part]


def call_llm(
    lm_config: lm_config.LMConfig,
    prompt: APIInput,
) -> str:
    response: str
    if lm_config.provider == "openai":
        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            response = generate_from_openai_chat_completion(
                messages=prompt,
                model=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                top_p=lm_config.gen_config["top_p"],
                context_length=lm_config.gen_config["context_length"],
                max_tokens=lm_config.gen_config["max_tokens"],
                stop_token=None,
            )
        elif lm_config.mode == "completion":
            assert isinstance(prompt, str)
            response = generate_from_openai_completion(
                prompt=prompt,
                engine=lm_config.model,
                temperature=lm_config.gen_config["temperature"],
                max_tokens=lm_config.gen_config["max_tokens"],
                top_p=lm_config.gen_config["top_p"],
                stop_token=lm_config.gen_config["stop_token"],
            )
        else:
            raise ValueError(f"OpenAI models do not support mode {lm_config.mode}")
    elif lm_config.provider == "huggingface":
        assert isinstance(prompt, str)
        response = generate_from_huggingface_completion(
            prompt=prompt,
            model_endpoint=lm_config.gen_config["model_endpoint"],
            temperature=lm_config.gen_config["temperature"],
            top_p=lm_config.gen_config["top_p"],
            stop_sequences=lm_config.gen_config["stop_sequences"],
            max_new_tokens=lm_config.gen_config["max_new_tokens"],
        )
    elif lm_config.provider == "google":
        assert isinstance(prompt, list)
        assert all([isinstance(p, str) or isinstance(p, Image) for p in prompt])
        response = generate_from_gemini_completion(
            prompt=prompt,
            engine=lm_config.model,
            temperature=lm_config.gen_config["temperature"],
            max_tokens=lm_config.gen_config["max_tokens"],
            top_p=lm_config.gen_config["top_p"],
        )
    elif lm_config.provider == "vllm":
        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            response = generate_from_vllm_chat_completion(
                messages=prompt,
                model=lm_config.model,
                context_length=None,
                max_tokens=lm_config.gen_config["max_tokens"],
                stop_token=None,
            )
        elif lm_config.mode == "completion":
            assert isinstance(prompt, str)
            response = generate_from_vllm_completion(
                prompt=prompt,
                engine=lm_config.model,
                max_tokens=lm_config.gen_config["max_tokens"],
                stop_token=lm_config.gen_config["stop_token"],
            )
    elif lm_config.provider == "openrouter":
        if lm_config.mode == "chat":
            assert isinstance(prompt, list)
            response = generate_from_openrouter_chat_completion(
                messages=prompt,
                model=MODEL_TO_OR_ID[lm_config.model],
                temperature=lm_config.gen_config["temperature"],
                max_tokens=lm_config.gen_config["max_tokens"],
                top_p=lm_config.gen_config["top_p"],
                context_length=lm_config.gen_config["context_length"],
                stop_token=lm_config.gen_config["stop_token"],
                openrouter_config=lm_config.gen_config["openrouter_config"],
            )
            if lm_config.model == "deepseek-ai/DeepSeek-R1":
                thinking = response.choices[0].message.model_extra["reasoning"]
                answer = response.choices[0].message.content
                response = f"{thinking}{lm_config.gen_config['end_think_token']}{answer}"
            else:
                response = response.choices[0].message.content
        elif lm_config.mode == "completion":
            assert isinstance(prompt, str)
            response = generate_from_openrouter_completion(
                prompt=prompt,
                engine=MODEL_TO_OR_ID[lm_config.model],
                temperature=lm_config.gen_config["temperature"],
                max_tokens=lm_config.gen_config["max_tokens"],
                top_p=lm_config.gen_config["top_p"],
                context_length=lm_config.gen_config["context_length"],
                stop_token=lm_config.gen_config["stop_token"],
                openrouter_config=lm_config.gen_config["openrouter_config"],
            )
    else:
        raise NotImplementedError(f"Provider {lm_config.provider} not implemented")

    return response

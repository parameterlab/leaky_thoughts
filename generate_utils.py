"""Utility functions for generating text with various models and strategies."""

import json
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Union

import requests
import torch
from dotenv import load_dotenv
from openai import OpenAI
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

from cp_eval_utils import (
    anonymize_reasonings_parallel,
    calculate_openrouter_cost,
    swap_reasonings_parallel,
)


@dataclass
class OutputObj:
    """Simple dataclass to mimic VLLM's output structure.

    Attributes
    ----------
    text : str
        The generated text output.
    """

    text: str


@dataclass
class RequestOutputObj:
    """Dataclass to mimic VLLM's RequestOutput structure.

    Attributes
    ----------
    outputs : List[OutputObj]
        A list of output objects, each containing generated text.
    prompt : Union[str, List[Dict]]
        The prompt used to generate the output.
    """

    outputs: List[OutputObj]
    prompt: Union[str, List[Dict]]


class UserDataLogitsProcessor:
    """A logits processor that blocks generation of user data tokens.

    This processor is used during the model's "thinking" phase to prevent it
    from leaking personally identifiable information (PII) or other sensitive
    user data that was part of the input prompt. It works by assigning a
    log-probability of -inf to token IDs corresponding to the user's data,
    effectively blocking them from being generated. The blocking is deactivated
    once an `end_think_token` is generated.

    Attributes
    ----------
    tokenizer : PreTrainedTokenizer
        The tokenizer used to encode text into token IDs.
    user_data : dict or list
        A nested structure containing user data to be blocked.
    end_think_token : str, optional
        The token that signals the end of the thinking phase. If None, blocking
        is always active.
    end_think_token_ids : list of int, optional
        The token IDs for the `end_think_token`.
    is_thinking_phase : bool
        A flag indicating whether the model is currently in the thinking phase.
    blocked_token_ids : set of int
        A set of token IDs that are blocked from being generated.
    """

    def __init__(self, tokenizer, user_data, end_think_token=None):
        """Initialize the UserDataLogitsProcessor.

        Parameters
        ----------
        tokenizer : PreTrainedTokenizer
            The tokenizer for encoding user data.
        user_data : dict or list
            The user data to block during generation.
        end_think_token : str, optional
            The string marking the end of the thinking phase. Default is None.
        """
        self.tokenizer = tokenizer
        self.user_data = user_data
        self.end_think_token = end_think_token
        self.end_think_token_ids = (
            None
            if end_think_token is None
            else tokenizer.encode(end_think_token, add_special_tokens=False)
        )
        self.is_thinking_phase = True

        # Pre-compute token IDs for all user data values
        self.blocked_token_ids = set()
        self.parsed_user_data = self._extract_values(user_data)

        # Get all values from the profile
        values = [
            str(v)
            for v in self.parsed_user_data
            if isinstance(v, (str, int, float, bool))
        ]

        values = [
            [v, " " + v, v.lower(), " " + v.lower(), v.upper(), " " + v.upper()]
            for v in values
        ]
        values = list(set([item for sublist in values for item in sublist]))
        token_ids = [self.tokenizer.encode(v, add_special_tokens=False) for v in values]
        token_ids = list(set([item for sublist in token_ids for item in sublist]))
        self.blocked_token_ids.update(token_ids)

    def _extract_values(self, data):
        """Recursively extract all values from nested dictionaries and lists.

        Parameters
        ----------
        data : dict or list
            The data structure to extract values from.

        Returns
        -------
        list
            A flat list of all values found in the data structure.
        """
        values = []
        if isinstance(data, dict):
            for value in data.values():
                values.extend(self._extract_values(value))
        elif isinstance(data, list):
            for item in data:
                values.extend(self._extract_values(item))
        else:
            values.append(data)
        return values

    def __call__(self, input_ids, logits):
        """Process logits to block user data tokens.

        This method is called at each generation step. It modifies the logits
        to prevent the generation of blocked tokens during the thinking phase.

        Parameters
        ----------
        input_ids : torch.Tensor
            The sequence of input IDs generated so far.
        logits : torch.Tensor
            The logits for the next token.

        Returns
        -------
        torch.Tensor
            The modified logits.
        """
        if (
            self.end_think_token_ids is not None
            and self.is_thinking_phase
            and len(input_ids) > 1
        ):
            last_tokens = input_ids[-len(self.end_think_token_ids) :]
            think_token_match = torch.equal(
                torch.tensor(last_tokens, device=logits.device),
                torch.tensor(self.end_think_token_ids, device=logits.device),
            )
            if think_token_match:
                self.is_thinking_phase = False
                return logits

        # Only block tokens during thinking phase
        if self.is_thinking_phase:
            for token_id in self.blocked_token_ids:
                logits[token_id] = float("-inf")

        return logits


def generate_with_openrouter(
    prompts, model_name, sampling_params, args, end_think_token=None, is_cot=False
):
    """Generate text using the OpenRouter API.

    This function sends prompts to the OpenRouter API for text generation,
    handling parallel requests, retries, and cost calculation. It's designed
    to work with models available through OpenRouter, such as DeepSeek-R1.

    Parameters
    ----------
    prompts : list of list of dict
        A list of prompts, where each prompt is a list of messages in chat format.
    model_name : str
        The name of the model to use on OpenRouter (e.g., 'deepseek/deepseek-chat').
    sampling_params : object
        An object containing sampling parameters like temperature, top_p, max_tokens.
    args : argparse.Namespace
        Command-line arguments, expected to contain `openrouter_settings`.
    end_think_token : str, optional
        The token that separates reasoning from the final answer. If provided,
        the two parts are concatenated. Default is None.
    is_cot : bool, optional
        Flag indicating if it is a Chain-of-Thought prompt. Default is False.

    Returns
    -------
    list of RequestOutputObj
        A list of output objects, each containing the generated text and original prompt.
    """
    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")

    # Load OpenRouter settings
    try:
        with open(args.openrouter_settings, "r") as f:
            openrouter_settings = json.load(f)
    except FileNotFoundError:
        print(
            f"Warning: OpenRouter settings file {args.openrouter_settings} not found. Using default settings."
        )
        openrouter_settings = {
            "provider": {
                "order": ["DeepInfra"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
            }
        }
    if (
        model_name == "deepseek/deepseek-chat"
    ):  # for some reason DeepInfra does not take tool outputs
        openrouter_settings["provider"].pop("order")
        openrouter_settings["provider"]["allow_fallbacks"] = True

    all_outputs = [None] * len(prompts)  # Initialize with correct size
    num_workers = min(50, len(prompts))  # Number of parallel workers
    generation_ids = []  # Store all generation IDs
    generation_id_to_prompt_idx = {}  # Map generation IDs to prompt indices

    print(
        f"Generating responses with OpenRouter API for {len(prompts)} prompts using {num_workers} workers..."
    )

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def make_api_request(params, prompt):
        """Make a single API request to OpenRouter with retries."""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/leaking_thoughts",
            "X-Title": "Leaking Thoughts",
        }

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json={**params, "messages": prompt},
        )
        response.raise_for_status()
        response_data = response.json()
        return response_data["choices"][0]["message"], response_data["id"]

    def process_single_prompt(prompt, prompt_idx, end_think_token=None, is_cot=False):
        """Process a single prompt to generate `n` samples."""
        batch_outputs = []
        for _ in range(sampling_params.n):
            # Set up generation parameters
            params = {
                "model": model_name,
                "max_tokens": sampling_params.max_tokens,
                "temperature": sampling_params.temperature,
            }

            if sampling_params.top_p is not None:
                params["top_p"] = sampling_params.top_p

            if hasattr(sampling_params, "stop") and sampling_params.stop:
                params["stop"] = sampling_params.stop

            # Add OpenRouter settings
            params.update(openrouter_settings)

            # Make API request with retry mechanism
            response_output, gen_id = make_api_request(params, prompt)
            if (
                end_think_token is not None
                and not is_cot
                and "reasoning" in response_output
                and "content" in response_output
                and response_output["reasoning"] is not None
                and response_output["content"] is not None
            ):
                output_text = (
                    response_output["reasoning"]
                    + end_think_token
                    + response_output["content"]
                )
            else:
                output_text = response_output["content"]
            generation_ids.append(gen_id)
            generation_id_to_prompt_idx[gen_id] = prompt_idx

            # Create object that mimics VLLM's output structure
            batch_outputs.append(OutputObj(output_text))

        return prompt_idx, batch_outputs

    # Process prompts in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all tasks
        future_to_idx = {
            executor.submit(
                process_single_prompt, prompt, i, end_think_token, is_cot
            ): i
            for i, prompt in enumerate(prompts)
        }

        # Create progress bar
        progress_bar = tqdm(total=len(prompts), desc="OpenRouter API calls")

        # Process results as they complete
        for future in as_completed(future_to_idx):
            idx, batch_outputs = future.result()
            # Create an object that mimics VLLM's RequestOutput structure and place it at the correct index
            all_outputs[idx] = RequestOutputObj(batch_outputs, prompts[idx])
            # Update progress bar
            progress_bar.update(1)

    print(f"Completed {len(all_outputs)} OpenRouter API calls")

    # Calculate and display total cost
    total_cost, provider_info = calculate_openrouter_cost(generation_ids, api_key)
    cost_console = Console()
    cost_panel = Panel(
        f"[bold white]Total OpenRouter API Cost:[/] [bold green]${total_cost:.2f}[/]",
        title="💰 Cost Summary",
        border_style="green",
    )
    cost_console.print()
    cost_console.print(cost_panel)
    cost_console.print()

    # Add provider info to outputs
    for gen_id, prompt_idx in generation_id_to_prompt_idx.items():
        if not hasattr(all_outputs[prompt_idx], "provider_info"):
            all_outputs[prompt_idx].provider_info = []
        all_outputs[prompt_idx].provider_info.append(provider_info[gen_id])

    return all_outputs


def get_provider_model_name(model_name, provider):
    """Get the correct model name format for the specified provider.

    Different providers (OpenRouter, DeepSeek API, local vLLM) may use
    different identifiers for the same model. This function canonicalizes
    the model name based on the specified provider.

    Parameters
    ----------
    model_name : str
        The generic model name (e.g., 'deepseek-ai/deepseek-r1').
    provider : str
        The provider name ('openrouter', or 'vllm').

    Returns
    -------
    str
        The provider-specific model name.

    Raises
    ------
    ValueError
        If a model is not supported by the specified provider.
    """
    # Handle DeepSeek model naming conventions per provider
    if model_name.lower() in [
        "deepseek-ai/deepseek-r1",
        "deepseek/deepseek-r1",
    ]:
        if provider == "openrouter":
            return "deepseek/deepseek-r1"

        elif provider == "vllm":
            raise ValueError(
                "Cannot use vLLM as provider, as models cannot be run locally. Please use 'openrouter' or 'deepseek' as provider."
            )
    elif model_name.lower() == "deepseek-ai/deepseek-v3":
        if provider == "openrouter":
            return "deepseek/deepseek-chat"

        elif provider == "vllm":
            raise ValueError(
                "Cannot use vLLM as provider, as models cannot be run locally. Please use 'openrouter' as provider."
            )
    elif model_name.lower() == "deepseek-ai/deepseek-v3-0324":
        if provider == "openrouter":
            return "deepseek/deepseek-chat-v3-0324"

        elif provider == "vllm":
            raise ValueError(
                "Cannot use vLLM as provider, as models cannot be run locally. Please use 'openrouter' or 'deepseek' as provider."
            )
    return model_name


def display_generation_config(console, sampling_params):
    """Display the generation configuration in a pretty table.

    Parameters
    ----------
    console : rich.console.Console
        The rich console object for printing.
    sampling_params : object
        An object containing the sampling parameters for generation.

    Returns
    -------
    dict
        A dictionary containing the generation configuration parameters.
    """
    # Save sampling parameters in a gen_conf dictionary
    gen_conf = {
        "temperature": sampling_params.temperature
        if hasattr(sampling_params, "temperature")
        else None,
        "top_p": sampling_params.top_p if hasattr(sampling_params, "top_p") else None,
        "top_k": sampling_params.top_k if hasattr(sampling_params, "top_k") else None,
        "repetition_penalty": sampling_params.repetition_penalty
        if hasattr(sampling_params, "repetition_penalty")
        else None,
        "max_tokens": sampling_params.max_tokens,
        "n": sampling_params.n,
        "seed": sampling_params.seed,
        "stop": sampling_params.stop if hasattr(sampling_params, "stop") else None,
        "skip_special_tokens": sampling_params.skip_special_tokens
        if hasattr(sampling_params, "skip_special_tokens")
        else None,
    }

    # Pretty print the generation configuration using rich
    gen_conf_table = Table(title="Generation Configuration", box=box.ROUNDED)
    gen_conf_table.add_column("Parameter", style="cyan")
    gen_conf_table.add_column("Value", style="green")

    for param, value in gen_conf.items():
        gen_conf_table.add_row(param, str(value))

    console.print()
    console.print(Panel(gen_conf_table, expand=False))
    console.print()

    return gen_conf


def generate_with_rana(
    llm,
    prompts,
    data,
    valid_indices,
    args,
    model_name,
    start_think_token,
    end_think_token,
    sampling_params=None,
):
    """Implement the Reason-Anonymize-Answer (RAnA) approach with a local model.

    This function orchestrates the RAnA pipeline:
    1. Generate an initial reasoning trace from the model, stopping at `end_think_token`.
    2. Anonymize the generated reasoning to remove PII.
    3. Feed the anonymized reasoning back into the model to generate the final answer.

    Parameters
    ----------
    llm : vllm.LLM
        The vLLM object to use for generation.
    prompts : list
        A list of prompts for the model.
    data : list of dict
        The dataset, where each item corresponds to a prompt and contains user profile data.
    valid_indices : list of int
        The indices of the prompts/data to be processed.
    args : argparse.Namespace
        Command-line arguments, used for prompt_type and other settings.
    model_name : str
        The name of the model being used.
    start_think_token : str
        The token to prepend to the reasoning/anonymized reasoning.
    end_think_token : str
        The token that signals the end of the reasoning phase.
    sampling_params : vllm.SamplingParams, optional
        The sampling parameters for generation.

    Returns
    -------
    list of RequestOutputObj
        A list of final outputs, each containing the combined anonymized reasoning and answer.
    """
    import time
    from copy import deepcopy

    print("Starting RAnA generation process")

    # Step 1: Generate reasoning (stop at end_think_token)
    reasoning_sampling_params = deepcopy(sampling_params)
    if end_think_token is not None:
        reasoning_sampling_params.stop = [end_think_token, " " + end_think_token]

    # Set max tokens to max_tokens - 500 for reasoning
    original_max_tokens = reasoning_sampling_params.max_tokens
    reasoning_sampling_params.max_tokens = max(original_max_tokens - 500, 1000)

    print(
        f"Step 1: Generating initial reasoning (max tokens: {reasoning_sampling_params.max_tokens})..."
    )
    reasoning_outputs = llm.chat(
        prompts,
        sampling_params=reasoning_sampling_params,
        chat_template=llm.get_tokenizer().chat_template,
        add_generation_prompt=False if "cot" in args.prompt_type else True,
        continue_final_message=True if "cot" in args.prompt_type else False,
    )

    # Step 2: Collect and prepare reasoning for anonymization
    reasoning_texts = []
    # Add end_think_token if needed and collect all reasoning texts
    for i in range(len(reasoning_outputs)):
        reasoning_text = reasoning_outputs[i].outputs[0].text
        if (
            end_think_token is not None
            and reasoning_text is not None
            and not reasoning_text.endswith(end_think_token)
        ):
            reasoning_text = reasoning_text + end_think_token
        reasoning_texts.append(reasoning_text)

    # Get a representative profile for anonymization
    # Using the first valid index's profile as a representative
    sample_profile = data[valid_indices[0]].get("profile", {})

    # Step 2: Anonymize all reasoning texts in parallel
    print("Step 2: Anonymizing reasoning in parallel...")
    anonymized_results = anonymize_reasonings_parallel(reasoning_texts, sample_profile)

    # Store anonymized reasoning and extracted PII in data
    anonymized_reasoning_list = []
    for i, idx in enumerate(valid_indices):
        reasoning_text = reasoning_texts[i]
        anonymized_text, extracted_pii = anonymized_results[i]

        # Store original and anonymized reasoning in data
        data[idx]["original_reasoning"] = reasoning_text

        # Store extracted PII data
        if "gpt_extractions" not in data[idx]:
            data[idx]["gpt_extractions"] = {}
        data[idx]["gpt_extractions"]["reasoning"] = extracted_pii

        # Add to anonymized list for next step
        anonymized_reasoning_list.append(anonymized_text)

    # Step 3: Create new prompts with anonymized reasoning
    print("Step 3: Generating answers based on anonymized reasoning...")
    answer_prompts = []

    for i, idx in enumerate(valid_indices):
        # Create new prompt with a single assistant message containing anonymized reasoning
        new_prompt = deepcopy(prompts[i])
        # Add anonymized reasoning as assistant message with Answer prompt
        if "reasoning" in args.prompt_type:
            new_prompt.append(
                {
                    "role": "assistant",
                    "content": start_think_token + "\n" + anonymized_reasoning_list[i],
                }
            )
        else:  # Cot
            new_prompt[1]["content"] += anonymized_reasoning_list[i]
        answer_prompts.append(new_prompt)

    # Adjust token limit for answer generation to 500
    answer_sampling_params = deepcopy(sampling_params)
    answer_sampling_params.max_tokens = 500

    print(f"Generating answers with max_tokens: {answer_sampling_params.max_tokens}")

    # Path to custom chat template
    # We need this for DeepSeek models, cause otherwise they og template will remove the reasoning
    custom_template_path = f"chat_templates/rana/{model_name.replace('/', '_')}.jinja"

    # Load custom chat template
    try:
        with open(custom_template_path, "r") as f:
            custom_template = f.read()
    except FileNotFoundError:
        print(f"Custom template not found for {model_name} at {custom_template_path}")
        print("Using default chat template")
        custom_template = None

    # Generate answers based on anonymized reasoning
    answer_outputs = llm.chat(
        answer_prompts,
        sampling_params=answer_sampling_params,
        chat_template=custom_template
        if custom_template is not None
        else llm.get_tokenizer().chat_template,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    # Step 4: Combine reasoning and answers
    print("Step 4: Combining reasoning and answers...")
    final_outputs = []

    for i, idx in enumerate(valid_indices):
        answer_text = answer_outputs[i].outputs[0].text
        combined_text = anonymized_reasoning_list[i] + answer_text

        # Create output object mimicking the regular output format
        output_obj = OutputObj(combined_text)
        request_output = RequestOutputObj([output_obj], prompts[i])
        final_outputs.append(request_output)

    return final_outputs


def generate_with_openrouter_rana(
    prompts,
    data,
    valid_indices,
    model_name,
    sampling_params,
    args,
    start_think_token,
    end_think_token,
):
    """Implement the Reason-Anonymize-Answer (RAnA) approach using the OpenRouter API.

    This function orchestrates the RAnA pipeline with OpenRouter as the backend:
    1. Generate reasoning in parallel for each prompt, stopping at `end_think_token`.
    2. Anonymize the generated reasoning traces to remove PII.
    3. Feed the anonymized reasoning back to the OpenRouter API to generate final answers.

    Parameters
    ----------
    prompts : list
        A list of prompts for the model.
    data : list of dict
        The dataset, containing user profile data for each prompt.
    valid_indices : list of int
        The indices of the prompts/data to be processed.
    model_name : str
        The name of the model to use on OpenRouter.
    sampling_params : object
        An object with sampling parameters (temperature, top_p, etc.).
    args : argparse.Namespace
        Command-line arguments, containing model path and prompt type.
    start_think_token : str
        The token to prepend to the reasoning.
    end_think_token : str
        The token to signal the end of the reasoning phase.

    Returns
    -------
    tuple
        - list of RequestOutputObj: The final generated outputs.
        - list of str: The generation IDs from OpenRouter.
        - dict: A mapping from generation IDs to prompt indices.
    """
    import time
    from copy import deepcopy

    from transformers import AutoTokenizer

    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")

    # Load OpenRouter settings
    try:
        with open(args.openrouter_settings, "r") as f:
            openrouter_settings = json.load(f)
    except FileNotFoundError:
        print(
            f"Warning: OpenRouter settings file {args.openrouter_settings} not found. Using default settings."
        )
        openrouter_settings = {
            "provider": {
                "order": ["DeepInfra"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
            }
        }
    if model_name == "deepseek/deepseek-chat":
        openrouter_settings["provider"].pop("order")
        openrouter_settings["provider"]["allow_fallbacks"] = True

    # Initialize variables to store generation results
    reasoning_texts = [None] * len(valid_indices)  # Initialize with correct size
    num_workers = min(50, len(valid_indices))  # Number of parallel workers
    generation_ids = []  # Store all generation IDs
    generation_id_to_prompt_idx = {}  # Map generation IDs to prompt indices

    print(
        f"Generating responses with OpenRouter API for {len(valid_indices)} prompts using {num_workers} workers in RAnA mode..."
    )

    # Load tokenizer for applying chat templates
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Path to custom chat template
    custom_template_path = f"chat_templates/rana/{args.model.replace('/', '_')}.jinja"

    # Load custom chat template
    try:
        with open(custom_template_path, "r") as f:
            custom_template = f.read()
            tokenizer.chat_template = custom_template
            print(f"Using custom chat template from {custom_template_path}")
    except FileNotFoundError:
        print(f"Custom template not found for {args.model} at {custom_template_path}")
        print("Using default chat template")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def make_api_request(params, prompt_text):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/leaking_thoughts",
            "X-Title": "Leaking Thoughts",
        }

        # Always use completions endpoint
        response = requests.post(
            url="https://openrouter.ai/api/v1/completions",
            headers=headers,
            json={**params, "prompt": prompt_text},
        )
        response.raise_for_status()
        gen_id = response.json()["id"]
        output = response.json()
        return output, gen_id

    # Step 1: Generate reasoning for each prompt in parallel
    # Max tokens for reasoning is max_tokens - 500
    reasoning_max_tokens = max(sampling_params.max_tokens - 500, 1000)
    print(
        f"Step 1: Generating reasoning in parallel (max tokens: {reasoning_max_tokens})..."
    )

    # Function to process a single reasoning prompt
    def process_reasoning_prompt(prompt_idx):
        idx = valid_indices[prompt_idx]
        prompt = prompts[idx]

        # Format the prompt using the chat template if it's a list (chat format)
        if isinstance(prompt, list):
            formatted_prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=False if "cot" in args.prompt_type else True,
                continue_final_message=True if "cot" in args.prompt_type else False,
            )
        else:
            # For non-chat prompts, use as-is
            formatted_prompt = prompt

        # Set up generation parameters for reasoning
        reasoning_params = {
            "model": model_name,
            "max_tokens": reasoning_max_tokens,
            "temperature": sampling_params.temperature,
        }

        if sampling_params.top_p is not None:
            reasoning_params["top_p"] = sampling_params.top_p

        # Add stop tokens to end at reasoning phase
        if end_think_token is not None:
            reasoning_params["stop"] = [end_think_token, " " + end_think_token]

        # Add OpenRouter settings
        reasoning_params.update(openrouter_settings)

        # Make API request for reasoning
        response_output, gen_id = make_api_request(reasoning_params, formatted_prompt)
        reasoning_key = "reasoning" if "reasoning" in args.prompt_type else "text"
        reasoning_text = response_output["choices"][0][reasoning_key]

        # Add end_think_token if needed
        if (
            end_think_token is not None
            and reasoning_text is not None
            and not reasoning_text.endswith(end_think_token)
        ):
            reasoning_text += end_think_token

        # Store generation ID mapping
        return prompt_idx, idx, reasoning_text, gen_id, formatted_prompt

    # Process reasoning prompts in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all reasoning tasks
        future_to_idx = {
            executor.submit(process_reasoning_prompt, i): i
            for i in range(len(valid_indices))
        }

        # Create progress bar
        progress_bar = tqdm(
            total=len(valid_indices), desc="Step 1: Reasoning generation"
        )

        # Process results as they complete
        for future in as_completed(future_to_idx):
            prompt_idx, data_idx, reasoning_text, gen_id, formatted_prompt = (
                future.result()
            )
            reasoning_texts[prompt_idx] = reasoning_text

            # Store generation ID information
            generation_ids.append(gen_id)
            generation_id_to_prompt_idx[gen_id] = data_idx

            # Update progress bar
            progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

    # Step 2: Anonymize all reasoning texts in parallel
    print("Step 2: Anonymizing reasoning in parallel...")

    # Get a representative profile for anonymization
    # Using the first valid index's profile as a representative
    sample_profile = data[valid_indices[0]].get("profile", {})

    # Anonymize all reasoning texts in parallel
    anonymized_results = anonymize_reasonings_parallel(reasoning_texts, sample_profile)

    # Store anonymized reasoning and extracted PII in data
    anonymized_reasoning_list = []
    for i, idx in enumerate(valid_indices):
        anonymized_text, extracted_pii = anonymized_results[i]

        # Store original reasoning in data
        data[idx]["original_reasoning"] = reasoning_texts[i]

        # Store extracted PII data
        if "gpt_extractions" not in data[idx]:
            data[idx]["gpt_extractions"] = {}
        data[idx]["gpt_extractions"]["reasoning"] = extracted_pii

        # Add to anonymized list for next step
        anonymized_reasoning_list.append(anonymized_text)

    # Step 3: Generate answers based on anonymized reasoning in parallel
    final_outputs = [None] * len(valid_indices)  # Initialize with correct size
    print("Step 3: Generating answers in parallel (max tokens: 500)...")

    # Max tokens for answer generation is fixed at 500
    answer_max_tokens = 500

    # Function to process a single answer generation
    def process_answer_prompt(prompt_idx):
        idx = valid_indices[prompt_idx]
        orig_prompt = prompts[idx]
        anonymized_reasoning = anonymized_reasoning_list[prompt_idx]

        # Prepare prompt for answer generation
        # Create a new chat prompt with anonymized reasoning as assistant message
        answer_messages = deepcopy(orig_prompt)
        if "cot" in args.prompt_type:
            answer_messages[1]["content"] += anonymized_reasoning
        else:
            answer_messages.append(
                {
                    "role": "assistant",
                    "content": start_think_token + "\n" + anonymized_reasoning,
                }
            )

        # Format with chat template
        formatted_answer_prompt = tokenizer.apply_chat_template(
            answer_messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )

        # Set up generation parameters for answer
        answer_params = {
            "model": model_name,
            "max_tokens": answer_max_tokens,
            "temperature": sampling_params.temperature,
        }

        if sampling_params.top_p is not None:
            answer_params["top_p"] = sampling_params.top_p

        # Add OpenRouter settings
        answer_params.update(openrouter_settings)

        # Generate answer
        response_output, gen_id = make_api_request(
            answer_params, formatted_answer_prompt
        )
        answer_text = response_output["choices"][0]["text"]

        # Combine reasoning and answer
        combined_text = anonymized_reasoning + answer_text

        # Create output object
        output_obj = OutputObj(combined_text)
        request_output = RequestOutputObj([output_obj], orig_prompt)

        return prompt_idx, idx, request_output, gen_id, formatted_answer_prompt

    # Process answer prompts in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all answer generation tasks
        future_to_idx = {
            executor.submit(process_answer_prompt, i): i
            for i in range(len(valid_indices))
        }

        # Create progress bar
        progress_bar = tqdm(total=len(valid_indices), desc="Step 3: Answer generation")

        # Process results as they complete
        for future in as_completed(future_to_idx):
            prompt_idx, data_idx, request_output, gen_id, formatted_answer_prompt = (
                future.result()
            )
            final_outputs[prompt_idx] = request_output

            # Store generation ID information
            generation_ids.append(gen_id)
            generation_id_to_prompt_idx[gen_id] = data_idx

            # Update progress bar
            progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

    print(f"Completed {len(final_outputs)} OpenRouter API calls with RAnA")
    return final_outputs, generation_ids, generation_id_to_prompt_idx


def generate_openrouter_hide_data(
    prompts, data, valid_indices, model_name, sampling_params, args, end_think_token
):
    """Generate text with OpenRouter, preventing PII leakage using logit biasing.

    This function implements the "hide_data" approach. It first generates a
    reasoning trace while using OpenRouter's `logit_bias` feature to prevent the
    model from generating tokens corresponding to user data. It then generates
    the final answer based on this "sanitized" reasoning.

    Parameters
    ----------
    prompts : list
        A list of prompts for the model.
    data : list of dict
        The dataset, containing user profile data for each prompt.
    valid_indices : list of int
        The indices of the prompts/data to be processed.
    model_name : str
        The name of the model to use on OpenRouter.
    sampling_params : object
        An object with sampling parameters (temperature, top_p, etc.).
    args : argparse.Namespace
        Command-line arguments, containing model path and other settings.
    end_think_token : str
        The token to signal the end of the reasoning phase.

    Returns
    -------
    tuple
        - list of RequestOutputObj: The final generated outputs.
        - list of str: The generation IDs from OpenRouter.
        - dict: A mapping from generation IDs to prompt indices.
    """
    import time
    from copy import deepcopy

    from transformers import AutoTokenizer

    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")

    # Load OpenRouter settings
    try:
        with open(args.openrouter_settings, "r") as f:
            openrouter_settings = json.load(f)
    except FileNotFoundError:
        print(
            f"Warning: OpenRouter settings file {args.openrouter_settings} not found. Using default settings."
        )
        openrouter_settings = {
            "provider": {
                "order": ["DeepInfra"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
            }
        }
    openrouter_settings["provider"].pop("order")
    openrouter_settings["provider"]["allow_fallbacks"] = True

    # Initialize variables to store generation results
    reasoning_texts = [None] * len(valid_indices)  # Initialize with correct size
    num_workers = min(50, len(valid_indices))  # Number of parallel workers
    generation_ids = []  # Store all generation IDs
    generation_id_to_prompt_idx = {}  # Map generation IDs to prompt indices

    print(
        f"Generating responses with OpenRouter API for {len(valid_indices)} prompts using {num_workers} workers with hide_data enabled..."
    )

    # Load tokenizer for applying chat templates and creating logit bias
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Path to custom chat template
    custom_template_path = f"chat_templates/rana/{args.model.replace('/', '_')}.jinja"

    # Load custom chat template
    try:
        with open(custom_template_path, "r") as f:
            custom_template = f.read()
            tokenizer.chat_template = custom_template
            print(f"Using custom chat template from {custom_template_path}")
    except FileNotFoundError:
        print(f"Custom template not found for {args.model} at {custom_template_path}")
        print("Using default chat template")

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def make_api_request(params, prompt_text):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/leaking_thoughts",
            "X-Title": "Leaking Thoughts",
        }

        # Always use completions endpoint
        response = requests.post(
            url="https://openrouter.ai/api/v1/completions",
            headers=headers,
            json={**params, "prompt": prompt_text},
        )
        response.raise_for_status()
        gen_id = response.json()["id"]
        output = response.json()
        return output, gen_id

    # Function to extract and tokenize user data for logit biasing
    def extract_user_data_tokens(profile):
        blocked_token_ids = set()

        # Recursively extract all values from the profile
        def extract_values(data):
            values = []
            if isinstance(data, dict):
                for value in data.values():
                    values.extend(extract_values(value))
            elif isinstance(data, list):
                for item in data:
                    values.extend(extract_values(item))
            else:
                values.append(data)
            return values

        # Get all values from the profile
        values = [
            str(v)
            for v in extract_values(profile)
            if isinstance(v, (str, int, float, bool))
        ]

        values = [
            [v, " " + v, v.lower(), " " + v.lower(), v.upper(), " " + v.upper()]
            for v in values
        ]
        values = list(set([item for sublist in values for item in sublist]))

        token_ids = [tokenizer.encode(v, add_special_tokens=False) for v in values]
        token_ids = list(set([item for sublist in token_ids for item in sublist]))
        blocked_token_ids.update(token_ids)

        return blocked_token_ids

    # Step 1: Generate reasoning for each prompt in parallel with hide_data
    # Max tokens for reasoning is max_tokens - 500
    reasoning_max_tokens = max(sampling_params.max_tokens - 500, 1000)
    print(
        f"Step 1: Generating reasoning with hide_data (max tokens: {reasoning_max_tokens})..."
    )

    # Function to process a single reasoning prompt
    def process_reasoning_prompt(prompt_idx):
        idx = valid_indices[prompt_idx]
        prompt = prompts[idx]
        profile = data[idx].get("profile", {})

        # Get token IDs to block from the user's profile
        blocked_token_ids = extract_user_data_tokens(profile)

        # Create logit_bias dictionary for OpenRouter API
        logit_bias = {token_id: -100 for token_id in blocked_token_ids}

        # Format the prompt using the chat template if it's a list (chat format)
        if isinstance(prompt, list):
            formatted_prompt = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=False if "cot" in args.prompt_type else True,
                continue_final_message=True if "cot" in args.prompt_type else False,
            )
        else:
            # For non-chat prompts, use as-is
            formatted_prompt = prompt

        # Set up generation parameters for reasoning
        reasoning_params = {
            "model": model_name,
            "max_tokens": reasoning_max_tokens,
            "temperature": sampling_params.temperature,
            "logit_bias": logit_bias,  # Add logit bias to block user data tokens
        }

        if sampling_params.top_p is not None:
            reasoning_params["top_p"] = sampling_params.top_p

        # Add stop tokens to end at reasoning phase
        if end_think_token is not None:
            reasoning_params["stop"] = [end_think_token, " " + end_think_token]

        # Add OpenRouter settings
        reasoning_params.update(openrouter_settings)

        # Make API request for reasoning
        response_output, gen_id = make_api_request(reasoning_params, formatted_prompt)
        reasoning_key = "reasoning" if "reasoning" in args.prompt_type else "text"
        reasoning_text = response_output["choices"][0][reasoning_key]

        # Add end_think_token if needed
        if (
            end_think_token is not None
            and reasoning_text is not None
            and not reasoning_text.endswith(end_think_token)
        ):
            reasoning_text += end_think_token

        # Store generation ID mapping
        return prompt_idx, idx, reasoning_text, gen_id, formatted_prompt

    print(process_reasoning_prompt(0))
    # Process reasoning prompts in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all reasoning tasks
        future_to_idx = {
            executor.submit(process_reasoning_prompt, i): i
            for i in range(len(valid_indices))
        }

        # Create progress bar
        progress_bar = tqdm(
            total=len(valid_indices), desc="Step 1: Reasoning generation with hide_data"
        )

        # Process results as they complete
        for future in as_completed(future_to_idx):
            prompt_idx, data_idx, reasoning_text, gen_id, formatted_prompt = (
                future.result()
            )
            reasoning_texts[prompt_idx] = reasoning_text

            # Store generation ID information
            generation_ids.append(gen_id)
            generation_id_to_prompt_idx[gen_id] = data_idx

            # Update progress bar
            progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

    # Step 2: Generate answers based on reasoning in parallel
    final_outputs = [None] * len(valid_indices)  # Initialize with correct size
    print("Step 2: Generating answers in parallel (max tokens: 500)...")

    # Max tokens for answer generation is fixed at 500
    answer_max_tokens = 500

    # Function to process a single answer generation
    def process_answer_prompt(prompt_idx):
        idx = valid_indices[prompt_idx]
        orig_prompt = prompts[idx]
        reasoning_text = reasoning_texts[prompt_idx]

        # Prepare prompt for answer generation
        # Create a new chat prompt with reasoning as assistant message
        answer_messages = deepcopy(orig_prompt)
        if "cot" in args.prompt_type:
            answer_messages[1]["content"] += reasoning_text
        else:
            answer_messages.append(
                {
                    "role": "assistant",
                    "content": reasoning_text,
                }
            )

        # Format with chat template
        formatted_answer_prompt = tokenizer.apply_chat_template(
            answer_messages,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )

        # Set up generation parameters for answer
        answer_params = {
            "model": model_name,
            "max_tokens": answer_max_tokens,
            "temperature": sampling_params.temperature,
        }

        if sampling_params.top_p is not None:
            answer_params["top_p"] = sampling_params.top_p

        # Add OpenRouter settings
        answer_params.update(openrouter_settings)

        # Generate answer
        response_output, gen_id = make_api_request(
            answer_params, formatted_answer_prompt
        )
        answer_text = response_output["choices"][0]["text"]

        # Combine reasoning and answer
        combined_text = reasoning_text + answer_text

        # Create output object
        output_obj = OutputObj(combined_text)
        request_output = RequestOutputObj([output_obj], orig_prompt)

        # Store the prompt with reasoning for debugging
        data[idx]["prompt_with_reasoning"] = formatted_answer_prompt

        return prompt_idx, idx, request_output, gen_id

    # Process answer prompts in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        # Submit all answer generation tasks
        future_to_idx = {
            executor.submit(process_answer_prompt, i): i
            for i in range(len(valid_indices))
        }

        # Create progress bar
        progress_bar = tqdm(total=len(valid_indices), desc="Step 2: Answer generation")

        # Process results as they complete
        for future in as_completed(future_to_idx):
            prompt_idx, data_idx, request_output, gen_id = future.result()
            final_outputs[prompt_idx] = request_output

            # Store generation ID information
            generation_ids.append(gen_id)
            generation_id_to_prompt_idx[gen_id] = data_idx

            # Update progress bar
            progress_bar.update(1)

        # Close progress bar
        progress_bar.close()

    print(f"Completed {len(final_outputs)} OpenRouter API calls with hide_data")
    return final_outputs, generation_ids, generation_id_to_prompt_idx


def generate_with_budget(
    llm, prompts, sampling_params, args, start_think_token, end_think_token
):
    """Generate text with a fixed token budget for the thinking phase.

    This function forces the model to "think" for a specific number of tokens
    (`args.budget_thinking`). It generates text iteratively until the budget is
    exhausted. If the model produces an `end_think_token` before the budget is
    used up, it is replaced with a filler phrase, and generation continues.

    Parameters
    ----------
    llm : vllm.LLM
        The vLLM object to use for generation.
    prompts : list
        A list of prompts for the model.
    sampling_params : vllm.SamplingParams
        The base sampling parameters for generation.
    args : argparse.Namespace
        Command-line arguments, must contain `budget_thinking`.
    start_think_token : str
        The token to prepend to the reasoning.
    end_think_token : str
        The token that signals the end of the reasoning phase.

    Returns
    -------
    list of RequestOutputObj
        A list of the final generated outputs.
    """

    # Load custom chat template
    custom_template_path = f"chat_templates/rana/{args.model.replace('/', '_')}.jinja"
    try:
        with open(custom_template_path, "r") as f:
            custom_template = f.read()
            print(f"Using custom chat template from {custom_template_path}")
    except FileNotFoundError:
        print(
            f"Custom template not found for {args.model} at {custom_template_path}, using default"
        )
        custom_template = llm.get_tokenizer().chat_template

    base_params = sampling_params.clone()
    ignore_strs = ["Oh wait", "Wait", "But wait,"]
    outputs = []

    prompts_with_reasoning = []
    for prompt in tqdm(prompts, desc="Processing prompts (reasoning)"):
        # Initialize the chat prompt messages
        full_prompt = deepcopy(prompt)
        full_prompt.append({"role": "assistant", "content": start_think_token + "\n"})

        remaining = args.budget_thinking
        while remaining > 0:
            think_params = base_params.clone()
            think_params.max_tokens = remaining
            think_params.min_tokens = 1
            think_params.stop = [end_think_token, f" {end_think_token}"]
            think_params.skip_special_tokens = False
            think_params.min_tokens = 1
            think_params.include_stop_str_in_output = True
            # Determine flags for this iteration

            think_outs = llm.chat(
                [full_prompt],
                sampling_params=think_params,
                chat_template=custom_template,
                add_generation_prompt=False,
                continue_final_message=True,
                use_tqdm=False,
            )
            # Reset first_loop after first iteration
            think_out = think_outs[0]
            text = think_out.outputs[0].text
            try:
                tokens_used = len(think_out.outputs[0].token_ids)
            except AttributeError:
                tokens_used = len(llm.get_tokenizer().encode(text))
            remaining -= tokens_used

            if text.endswith(end_think_token):
                if remaining > 0:
                    # Remove the end token and insert an ignore string
                    trimmed = text[: -len(end_think_token)] + random.choice(ignore_strs)
                    full_prompt[-1]["content"] += trimmed
                    continue
                else:
                    break
            else:
                # Append generated text to the last assistant message
                full_prompt[-1]["content"] += text
                continue

        # Append final thinking termination and prompt for answer
        full_prompt[-1]["content"] += (
            f" Okay, I think I have finished thinking.\n{end_think_token}\nAnswer: "
        )
        prompts_with_reasoning.append(full_prompt)

    # Generate the answer
    answer_params = base_params.clone()
    answer_params.max_tokens = 500
    answer_outs = llm.chat(
        prompts_with_reasoning,
        sampling_params=answer_params,
        chat_template=custom_template,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    for i, answer_out in enumerate(answer_outs):
        answer_outs[i].outputs[0].text = (
            prompts_with_reasoning[i][-1]["content"] + answer_outs[i].outputs[0].text
        )
        answer_outs[i].prompt = llm.get_tokenizer().apply_chat_template(
            prompts_with_reasoning[i],
            chat_template=custom_template,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        outputs.append(answer_outs[i])

    return outputs


def generate_with_swap(
    llm,
    prompts,
    data,
    valid_indices,
    args,
    model_name,
    start_think_token,
    end_think_token,
    sampling_params=None,
):
    """Implement the Reason-Swap-Answer (RSwA) approach with a local model.

    This function orchestrates the RSwA pipeline:
    1. Generate an initial reasoning trace from the model.
    2. Swap sensitive values in the reasoning with alternatives based on reference data.
    3. Feed the swapped reasoning back to the model to generate the final answer.

    Parameters
    ----------
    llm : vllm.LLM
        The vLLM object for generation.
    prompts : list
        The list of prompts.
    data : list of dict
        The dataset, containing original and alternative references for swapping.
    valid_indices : list of int
        Indices of the prompts/data to process.
    args : argparse.Namespace
        Command-line arguments.
    model_name : str
        The name of the model being used.
    start_think_token : str
        The token to prepend to the reasoning.
    end_think_token : str
        The token signaling the end of the reasoning phase.
    sampling_params : vllm.SamplingParams, optional
        Sampling parameters for generation.

    Returns
    -------
    list of RequestOutputObj
        A list of final outputs with swapped reasoning and answers.
    """
    from copy import deepcopy

    print("Starting RSwA generation process")

    # Step 1: Generate reasoning (stop at end_think_token)
    reasoning_sampling_params = deepcopy(sampling_params)
    if end_think_token is not None:
        reasoning_sampling_params.stop = [end_think_token, " " + end_think_token]

    # Path to custom chat template for swap flow
    custom_template_path = f"chat_templates/rana/{model_name.replace('/', '_')}.jinja"
    try:
        with open(custom_template_path, "r") as f:
            custom_template = f.read()
    except FileNotFoundError:
        custom_template = llm.get_tokenizer().chat_template

    # Set max tokens to max_tokens - 500 for reasoning
    original_max_tokens = reasoning_sampling_params.max_tokens
    reasoning_sampling_params.max_tokens = max(original_max_tokens - 500, 1000)

    print(
        f"Step 1: Generating initial reasoning (max tokens: {reasoning_sampling_params.max_tokens})..."
    )

    reasoning_outputs = llm.chat(
        prompts,
        sampling_params=reasoning_sampling_params,
        chat_template=custom_template,
        add_generation_prompt=True,
        continue_final_message=False,
    )

    # Step 2: Collect and prepare reasoning for swapping
    reasoning_texts = []
    for i in range(len(reasoning_outputs)):
        reasoning_text = reasoning_outputs[i].outputs[0].text
        if (
            end_think_token is not None
            and reasoning_text is not None
            and not reasoning_text.endswith(end_think_token)
        ):
            reasoning_text += end_think_token
        reasoning_texts.append(reasoning_text)

    # Step 2: Swap values in reasoning in parallel
    print("Step 2: Swapping reasoning values in parallel...")
    swapped_results = swap_reasonings_parallel(reasoning_texts, data, valid_indices)

    # Store swapped reasoning in data
    swapped_reasoning_list = []
    for i, idx in enumerate(valid_indices):
        reasoning_text = reasoning_texts[i]
        swapped_text, mapping = swapped_results[i]

        data[idx]["original_reasoning"] = reasoning_text
        data[idx]["swap_mapping"] = mapping

        swapped_reasoning_list.append(swapped_text)

    # Step 3: Create new prompts with swapped reasoning
    print("Step 3: Generating answers based on swapped reasoning...")
    answer_prompts = []

    for i, idx in enumerate(valid_indices):
        prefix = start_think_token + "\n" + swapped_reasoning_list[i]
        new_prompt = deepcopy(prompts[i])
        new_prompt.append(
            {
                "role": "assistant",
                "content": prefix,
            }
        )
        answer_prompts.append(new_prompt)

    answer_sampling_params = deepcopy(sampling_params)
    answer_sampling_params.max_tokens = 500

    print(f"Generating answers with max_tokens: {answer_sampling_params.max_tokens}")

    # Generate answers based on swapped reasoning
    answer_outputs = llm.chat(
        answer_prompts,
        sampling_params=answer_sampling_params,
        chat_template=custom_template,
        add_generation_prompt=False,
        continue_final_message=True,
    )

    # Step 4: Combine reasoning and answers
    print("Step 4: Combining reasoning and answers...")
    final_outputs = []
    for i, idx in enumerate(valid_indices):
        answer_text = answer_outputs[i].outputs[0].text
        combined_text = swapped_reasoning_list[i] + answer_text

        output_obj = OutputObj(combined_text)
        request_output = RequestOutputObj(
            [output_obj],
            llm.get_tokenizer().apply_chat_template(
                answer_prompts[i],
                tokenize=False,
                chat_template=custom_template,
                add_generation_prompt=False,
                continue_final_message=True,
            ),
        )
        final_outputs.append(request_output)

    return final_outputs


def generate_with_openrouter_swap(
    prompts,
    data,
    valid_indices,
    model_name,
    sampling_params,
    args,
    start_think_token,
    end_think_token,
):
    """Implement the Reason-Swap-Answer (RSwA) approach using the OpenRouter API.

    This function orchestrates the RSwA pipeline with OpenRouter as the backend:
    1. Generate reasoning in parallel for each prompt.
    2. Swap sensitive values in the reasoning with alternatives from reference data.
    3. Feed the swapped reasoning back to the OpenRouter API to generate final answers.

    Parameters
    ----------
    prompts : list
        The list of prompts.
    data : list of dict
        The dataset, containing references for swapping.
    valid_indices : list of int
        Indices of prompts/data to process.
    model_name : str
        The name of the model on OpenRouter.
    sampling_params : object
        An object with sampling parameters.
    args : argparse.Namespace
        Command-line arguments.
    start_think_token : str
        Token to prepend to the reasoning.
    end_think_token : str
        Token to signal the end of the reasoning phase.

    Returns
    -------
    tuple
        - list of RequestOutputObj: The final generated outputs.
        - list of str: The generation IDs from OpenRouter.
        - dict: A mapping from generation IDs to prompt indices.
    """
    import time
    from copy import deepcopy

    from transformers import AutoTokenizer

    # Load API key from .env file
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in .env file")

    # Load OpenRouter settings
    try:
        with open(args.openrouter_settings, "r") as f:
            openrouter_settings = json.load(f)
    except FileNotFoundError:
        print(
            f"Warning: OpenRouter settings file {args.openrouter_settings} not found. Using default settings."
        )
        openrouter_settings = {
            "provider": {
                "order": ["DeepInfra"],
                "allow_fallbacks": False,
                "require_parameters": True,
                "data_collection": "deny",
            }
        }
    if model_name == "deepseek/deepseek-chat":
        openrouter_settings["provider"].pop("order")
        openrouter_settings["provider"]["allow_fallbacks"] = True

    # Step 1: Generate reasoning for each prompt in parallel
    reasoning_texts = [None] * len(valid_indices)
    num_workers = min(50, len(valid_indices))
    generation_ids = []
    generation_id_to_prompt_idx = {}

    print(
        f"Generating responses with OpenRouter API for {len(valid_indices)} prompts using {num_workers} workers in RSwA mode..."
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Load custom chat template for swap flow
    custom_template_path = f"chat_templates/rana/{args.model.replace('/', '_')}.jinja"
    try:
        with open(custom_template_path, "r") as f:
            custom_template = f.read()
            tokenizer.chat_template = custom_template
            print(f"Using custom chat template from {custom_template_path}")
    except FileNotFoundError:
        print(f"Custom template not found for {args.model} at {custom_template_path}")
        print("Using default chat template")
        custom_template = tokenizer.chat_template

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def make_api_request(params, prompt_text):
        headers = {
            "Authorization": f"Bearer {api_key}",
            "HTTP-Referer": "https://github.com/leaking_thoughts",
            "X-Title": "Leaking Thoughts",
        }
        response = requests.post(
            url="https://openrouter.ai/api/v1/completions",
            headers=headers,
            json={**params, "prompt": prompt_text},
        )
        response.raise_for_status()
        data_json = response.json()
        return data_json, data_json.get("id")

    # Function to process a single reasoning prompt
    def process_reasoning_prompt(idx):
        i = valid_indices[idx]
        prompt = prompts[i]
        if isinstance(prompt, list):
            formatted = tokenizer.apply_chat_template(
                prompt,
                tokenize=False,
                add_generation_prompt=True,
                continue_final_message=False,
            )
        else:
            formatted = prompt
        params = {
            "model": model_name,
            "max_tokens": max(sampling_params.max_tokens - 500, 1000),
            "temperature": sampling_params.temperature,
        }
        if sampling_params.top_p is not None:
            params["top_p"] = sampling_params.top_p
        if end_think_token is not None:
            params["stop"] = [end_think_token, " " + end_think_token]
        params.update(openrouter_settings)
        output_json, gen_id = make_api_request(params, formatted)
        key = "reasoning" if "reasoning" in args.prompt_type else "text"
        text = output_json["choices"][0][key]
        if end_think_token and not text.endswith(end_think_token):
            text += end_think_token
        generation_ids.append(gen_id)
        generation_id_to_prompt_idx[gen_id] = i
        return idx, text

    # Generate reasonings in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_reasoning_prompt, idx): idx
            for idx in range(len(valid_indices))
        }
        for future in as_completed(futures):
            idx, text = future.result()
            reasoning_texts[idx] = text

    # Step 2: Swap reasoning values in parallel
    print("Step 2: Swapping reasoning values in parallel...")
    swapped_results = swap_reasonings_parallel(reasoning_texts, data, valid_indices)
    swapped_reasoning_list = []
    for i, idx in enumerate(valid_indices):
        swap_text, mapping = swapped_results[i]
        data[idx]["original_reasoning"] = reasoning_texts[i]
        data[idx]["swap_mapping"] = mapping
        swapped_reasoning_list.append(swap_text)

    # Step 3: Generate answers based on swapped reasoning
    final_outputs = [None] * len(valid_indices)
    num_workers_ans = min(50, len(valid_indices))

    def process_answer(idx):
        i = valid_indices[idx]
        orig_prompt = prompts[i]
        swap_text = swapped_reasoning_list[idx]
        reasoning_with_start = start_think_token + "\n" + swap_text
        if isinstance(orig_prompt, list):
            messages = deepcopy(orig_prompt)
            messages.append({"role": "assistant", "content": reasoning_with_start})
            formatted = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
                continue_final_message=True,
            )
        else:
            formatted = orig_prompt + reasoning_with_start
        params = {
            "model": model_name,
            "max_tokens": 500,
            "temperature": sampling_params.temperature,
        }
        if sampling_params.top_p is not None:
            params["top_p"] = sampling_params.top_p
        params.update(openrouter_settings)
        output_json, gen_id = make_api_request(params, formatted)
        answer = output_json["choices"][0].get("text")
        combined = reasoning_with_start + answer
        output_obj = OutputObj(combined)
        request_output = RequestOutputObj([output_obj], formatted)
        generation_ids.append(gen_id)
        generation_id_to_prompt_idx[gen_id] = i
        return idx, request_output

    with ThreadPoolExecutor(max_workers=num_workers_ans) as executor:
        futures = {
            executor.submit(process_answer, idx): idx
            for idx in range(len(valid_indices))
        }
        for future in as_completed(futures):
            idx, out = future.result()
            final_outputs[idx] = out

    print(f"Completed {len(final_outputs)} OpenRouter API calls with RSwA")
    return final_outputs, generation_ids, generation_id_to_prompt_idx

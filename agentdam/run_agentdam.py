# Copyright (c) Meta Platforms, Inc. and affiliates.
"""Script to run end-to-end evaluation on the benchmark.

Modified from https://github.com/web-arena-x/webarena/blob/main/run.py.
"""
import argparse
import glob
import json
import logging
import os
import random
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List

import openai
import requests
import torch
from PIL import Image
from transformers import AutoTokenizer
from tqdm import tqdm

from agent import (
    PromptAgent,
    construct_agent,
)
from agent.prompts import *
from browser_env import (
    Action,
    ActionTypes,
    ScriptBrowserEnv,
    StateInfo,
    Trajectory,
    create_stop_action,
)
from browser_env.actions import is_equivalent
from browser_env.auto_login import get_site_comb_from_filepath
from browser_env.helper_functions import (
    RenderHelper,
    get_action_description,
    action2str,
)
from evaluation_harness import evaluator_router, image_utils
from privacy_eval import PrivacyEvaluator
from llms import REASONING_MODELS_TO_END_THINK, REASONING_MODELS_TO_START_THINK

DATASET = os.environ["DATASET"]

LOG_FOLDER = "log_files"
Path(LOG_FOLDER).mkdir(parents=True, exist_ok=True)
LOG_FILE_NAME = f"{LOG_FOLDER}/log_{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{random.randint(0, 10000)}.log"

logger = logging.getLogger("logger")
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(LOG_FILE_NAME)
file_handler.setLevel(logging.DEBUG)
logger.addHandler(file_handler)

# Set the log format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


def insert_after_key(d, target_key, new_key, new_value):
    new_dict = {}
    for key, value in d.items():
        new_dict[key] = value
        if key == target_key:
            new_dict[new_key] = new_value
    return new_dict


def config() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run end-to-end evaluation on the agent privacy benchmark"
    )
    parser.add_argument("--render", action="store_true", help="Render the browser")

    parser.add_argument(
        "--slow_mo",
        type=int,
        default=0,
        help="Slow down the browser by the specified amount",
    )
    parser.add_argument(
        "--action_set_tag", default="id_accessibility_tree", help="Action type"
    )
    parser.add_argument(
        "--observation_type",
        choices=[
            "accessibility_tree",
            "accessibility_tree_with_captioner",
            "html",
            "image",
            "image_som",
        ],
        default="accessibility_tree",
        help="Observation type",
    )
    parser.add_argument(
        "--current_viewport_only",
        action="store_true",
        help="Only use the current viewport for the observation",
    )
    parser.add_argument("--viewport_width", type=int, default=1280)
    parser.add_argument("--viewport_height", type=int, default=2048)
    parser.add_argument("--save_trace_enabled", action="store_true")
    parser.add_argument("--sleep_after_execution", type=float, default=0.0)

    parser.add_argument("--max_steps", type=int, default=10)

    # privacy config
    parser.add_argument(
        "--privacy_test", action="store_true", help="enable privacy test"
    )
    parser.add_argument(
        "--privacy_config_path",
        type=str,
        default="./configs/p_cot_privacy_judge_3s.json",
    )

    parser.add_argument(
        "--privacy_config_path_reasoning",
        type=str,
        default="./configs/p_cot_privacy_judge_3s_reasoning.json",
    )

    parser.add_argument(
        "--openrouter_config",
        type=str,
        default="./openrouter_settings/no_order.json",
        help="Path to the openrouter config file",
    )

    # agent config
    parser.add_argument("--agent_type", type=str, default="prompt")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="./configs/p_cot_id_actree_3s.json",
    )
    parser.add_argument(
        "--parsing_failure_th",
        help="When consecutive parsing failures exceed this threshold, the agent will terminate early.",
        type=int,
        default=3,
    )
    parser.add_argument(
        "--repeating_action_failure_th",
        help="When consecutive repeated actions exceed this threshold, the agent will terminate early.",
        type=int,
        default=5,
    )

    parser.add_argument("--test_config_base_dir", type=str)

    parser.add_argument(
        "--eval_captioning_model_device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run eval captioning model on. By default, runs it on CPU.",
    )
    parser.add_argument(
        "--eval_captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl"],
        help="Captioning backbone for VQA-type evals.",
    )
    parser.add_argument(
        "--captioning_model",
        type=str,
        default="Salesforce/blip2-flan-t5-xl",
        choices=["Salesforce/blip2-flan-t5-xl", "llava-hf/llava-1.5-7b-hf"],
        help="Captioning backbone for accessibility tree alt text.",
    )

    # lm config
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--mode", type=str, default="chat")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--context_length", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=8000)
    parser.add_argument("--stop_token", type=str, default=None)
    parser.add_argument(
        "--max_retry",
        type=int,
        help="max retry times to perform generations when parsing fails",
        default=1,
    )
    parser.add_argument(
        "--max_obs_length",
        type=int,
        help="when not zero, will truncate the observation to this length before feeding to the model",
        default=5000,
    )

    # example config
    parser.add_argument("--test_start_idx", type=int, default=0)
    parser.add_argument("--test_end_idx", type=int, default=910)

    # logging related
    parser.add_argument("--result_dir", type=str, default="")
    args = parser.parse_args()

    # check the whether the action space is compatible with the observation space
    if args.action_set_tag == "id_accessibility_tree" and args.observation_type not in [
        "accessibility_tree",
        "accessibility_tree_with_captioner",
        "image_som",
    ]:
        raise ValueError(
            f"Action type {args.action_set_tag} is incompatible with the observation type {args.observation_type}"
        )

    return args


def early_stop(
    trajectory: Trajectory, max_steps: int, thresholds: dict[str, int]
) -> tuple[bool, str]:
    """Check whether need to stop early"""

    # reach the max step
    num_steps = (len(trajectory) - 1) / 2
    if num_steps >= max_steps:
        return True, f"Reach max steps {max_steps}"

    last_k_actions: list[Action]
    action_seq: list[Action]

    # Case: parsing failure for k times
    k = thresholds["parsing_failure"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    if len(last_k_actions) >= k:
        if all(
            [action["action_type"] == ActionTypes.NONE for action in last_k_actions]
        ):
            return True, f"Failed to parse actions for {k} times"

    # Case: same action for k times
    k = thresholds["repeating_action"]
    last_k_actions = trajectory[1::2][-k:]  # type: ignore[assignment]
    action_seq = trajectory[1::2]  # type: ignore[assignment]

    if len(action_seq) == 0:
        return False, ""

    last_action: Action = action_seq[-1]

    if last_action["action_type"] != ActionTypes.TYPE:
        if len(last_k_actions) >= k:
            if all([is_equivalent(action, last_action) for action in last_k_actions]):
                return True, f"Same action for {k} times"

    else:
        # check the action sequence
        if sum([is_equivalent(action, last_action) for action in action_seq]) >= k:
            return True, f"Same typing action for {k} times"

    return False, ""


def test(args: argparse.Namespace, config_file_list: list[str]) -> None:
    scores = []
    max_steps = args.max_steps

    # Initialize tokenizer for calculating token lengths
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Get end-of-thinking token for the model if it's a reasoning model
    is_thinking_model = False if "vanilla" in args.instruction_path else True
    end_think_token = "</think>" if "cot" in args.instruction_path else (
        REASONING_MODELS_TO_END_THINK.get(args.model) if args.model in REASONING_MODELS_TO_END_THINK else None
    )
    start_think_token = (
        "<think>"
        if "cot" in args.instruction_path
        else (
            REASONING_MODELS_TO_START_THINK.get(args.model)
            if args.model in REASONING_MODELS_TO_START_THINK
            else None
        )
    )
    
    job_id = os.environ.get("SLURM_JOB_ID")


    if job_id:
        print(f"Slurm Job ID: {job_id}")
    else:
        print("Not running inside a Slurm job.")

    
    args.job_id = job_id
    args.end_think_token = end_think_token
    args.start_think_token = start_think_token

    early_stop_thresholds = {
        "parsing_failure": args.parsing_failure_th,
        "repeating_action": args.repeating_action_failure_th,
    }

    if args.observation_type in [
        "accessibility_tree_with_captioner",
        "image_som",
    ]:
        device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        caption_image_fn = image_utils.get_captioning_fn(
            device, dtype, args.captioning_model
        )
    else:
        caption_image_fn = None

    # Load a (possibly different) captioning model for running VQA evals.
    if DATASET == "visualwebarena":
        if caption_image_fn and args.eval_captioning_model == args.captioning_model:
            eval_caption_image_fn = caption_image_fn
        else:
            eval_caption_image_fn = image_utils.get_captioning_fn(
                args.eval_captioning_model_device,
                torch.float16
                if (
                    torch.cuda.is_available()
                    and args.eval_captioning_model_device == "cuda"
                )
                else torch.float32,
                args.eval_captioning_model,
            )
    else:
        caption_image_fn = None
        eval_caption_image_fn = None

    agent = construct_agent(
        args,
        captioning_fn=caption_image_fn
        if args.observation_type == "accessibility_tree_with_captioner"
        else None,
    )  # NOTE: captioning_fn here is used for captioning input images.

    env = ScriptBrowserEnv(
        headless=not args.render,
        slow_mo=args.slow_mo,
        observation_type=args.observation_type,
        current_viewport_only=args.current_viewport_only,
        viewport_size={
            "width": args.viewport_width,
            "height": args.viewport_height,
        },
        save_trace_enabled=args.save_trace_enabled,
        sleep_after_execution=args.sleep_after_execution,
        # NOTE: captioning_fn here is used for LLM + captioning baselines.
        # This can be different from the captioning model used for evals.
        captioning_fn=caption_image_fn,
    )

    privacy_scores = []
    privacy_eval = (
        PrivacyEvaluator(Path(args.privacy_config_path)) if args.privacy_test else None
    )
    privacy_eval_reasoning = (
        PrivacyEvaluator(Path(args.privacy_config_path_reasoning))
        if args.privacy_test and is_thinking_model
        else None
    )
    start_time = datetime.now()

    # Create data structure to store detailed results
    results = {"args": vars(args), "summary": {}, "data": []}

    for config_file in tqdm(config_file_list, desc="Processing tasks"):
        try:
            render_helper = RenderHelper(
                config_file, args.result_dir, args.action_set_tag
            )

            # Load task.
            print("Config file", config_file)
            with open(config_file) as f:
                _c = json.load(f)
                intent = _c["intent"]
                task_id = _c["task_id"]
                if args.privacy_test:
                    try:
                        sensitive_data = _c["sensitive_data"]
                    except KeyError:
                        raise KeyError(
                            "The field sensitive_data must be in config json if privacy_test enabled"
                        )
                else:
                    print(
                        "[Warning!] Privacy Tests are disabled! Please rerun with --privacy_test."
                    )
                image_paths = _c.get("image", None)
                images = []

                # automatically login
                if _c["storage_state"]:
                    cookie_file_name = os.path.basename(_c["storage_state"])
                    comb = get_site_comb_from_filepath(cookie_file_name)
                    temp_dir = tempfile.mkdtemp()
                    # subprocess to renew the cookie
                    output_login = subprocess.run(
                        [
                            ".venv/bin/python",
                            "visualwebarena/browser_env/auto_login.py",
                            "--auth_folder",
                            temp_dir,
                            "--site_list",
                            *comb,
                        ]
                    )
                    print("***************************************************")
                    if output_login.returncode != 0:
                        logger.info(f"!!!!!Failed to login for {config_file}")
                    else:
                        logger.info(f"++++ Successfully logged in for {config_file}")
                    _c["storage_state"] = f"{temp_dir}/{cookie_file_name}"
                    assert os.path.exists(_c["storage_state"])
                    # update the config file
                    config_file = f"{temp_dir}/{os.path.basename(config_file)}"
                    with open(config_file, "w") as f:
                        json.dump(_c, f)

                # Load input images for the task, if any.
                if image_paths is not None:
                    if isinstance(image_paths, str):
                        image_paths = [image_paths]
                    for image_path in image_paths:
                        # Load image either from the web or from a local path.
                        if image_path.startswith("http"):
                            headers = {
                                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                            }
                            input_image = Image.open(
                                requests.get(
                                    image_path, stream=True, headers=headers
                                ).raw
                            )
                        else:
                            input_image = Image.open(image_path)

                        images.append(input_image)

            logger.info(f"[Config file]: {config_file}")
            logger.info(f"[Intent]: {intent}")

            agent.reset(config_file)
            trajectory: Trajectory = []
            obs, info = env.reset(options={"config_file": config_file})
            state_info: StateInfo = {"observation": obs, "info": info}
            trajectory.append(state_info)

            meta_data = {"action_history": ["None"]}
            privacy_score_i = 0
            privacy_score_i_reasoning = 0
            action_privacy_scores = []
            privacy_eval_txts = []  # Store all privacy evaluation texts
            llm_outputs = []
            prompts = []  # Store all prompts
            valid_actions = []  # Track valid actions (non-NONE actions)
            if is_thinking_model:
                reasoning_outputs = []
                answer_outputs = []
                action_privacy_scores_reasoning = []
                privacy_eval_txts_reasoning = []

            while True:
                early_stop_flag, stop_info = early_stop(
                    trajectory, max_steps, early_stop_thresholds
                )

                if early_stop_flag:
                    action = create_stop_action(f"Early stop: {stop_info}")
                else:
                    try:
                        action = agent.next_action(
                            trajectory,
                            intent,
                            images=images,
                            meta_data=meta_data,
                        )
                    except ValueError as e:
                        # get the error message
                        action = create_stop_action(f"ERROR: {str(e)}")

                trajectory.append(action)

                # Track if this is a valid action (not NONE)
                valid_actions.append(
                    1 if action["action_type"] != ActionTypes.NONE else 0
                )

                action_str = get_action_description(
                    action,
                    state_info["info"]["observation_metadata"],
                    action_set_tag=args.action_set_tag,
                    prompt_constructor=agent.prompt_constructor
                    if isinstance(agent, PromptAgent)
                    else None,
                )

                render_helper.render(
                    action, state_info, meta_data, args.render_screenshot
                )
                meta_data["action_history"].append(action_str)

                if action["action_type"] == ActionTypes.STOP:
                    logger.info(f">>> Action type: {action['action_type']} <<<")
                    logger.info(f">>> Action str: {action_str} <<<")
                    break
                
                logger.info(f">>> Sensitive data: {sensitive_data} <<<")
                logger.info(f">>> LLM output {action['raw_prediction']} <<<")
                if "reasoning" in action:
                    logger.info(f">>> LLM Reasoning: {action['reasoning']} <<<")
                if "answer" in action:
                    logger.info(f">>> LLM Answer: {action['answer']} <<<")
                logger.info(f">>> Action type: {action['action_type']} <<<")
                logger.info(f">>> Action str: {action_str} <<<")

                # Store LLM output for this action
                llm_outputs.append(action["raw_prediction"])

                # Store prompt if it exists
                if "prompt" in action:
                    prompts.append(action["prompt"])

                # Store reasoning and answer outputs if using a reasoning model
                if is_thinking_model:
                    if "reasoning" in action:
                        reasoning_outputs.append(action["reasoning"])
                    if "answer" in action:
                        answer_outputs.append(action["answer"])

                # perform privacy leakage test
                if args.privacy_test:
                    if action["action_type"] != ActionTypes.NONE:
                        privacy_eval_score, privacy_eval_txt = privacy_eval.test(
                            action_str=action_str, sensitive_data=sensitive_data
                        )
                        action_privacy_scores.append(privacy_eval_score)
                        privacy_eval_txts.append(
                            privacy_eval_txt
                        )  # Store the privacy evaluation text
                        if privacy_eval_score > 0:
                            privacy_score_i = privacy_eval_score
                            logger.info(f"Privacy Judge Reasoning: {privacy_eval_txt}")
                    else:
                        action_privacy_scores.append(0)
                        privacy_eval_txts.append("No action taken")

                    if is_thinking_model and "reasoning" in action:
                        privacy_eval_score_reasoning, privacy_eval_txt_reasoning = (
                            privacy_eval_reasoning.test(
                                action_str=action["reasoning"],
                                sensitive_data=sensitive_data,
                            )
                        )
                        action_privacy_scores_reasoning.append(
                            privacy_eval_score_reasoning
                        )
                        privacy_eval_txts_reasoning.append(
                            privacy_eval_txt_reasoning
                        )  # Store the privacy evaluation text
                        if privacy_eval_score_reasoning > 0:
                            privacy_score_i_reasoning = privacy_eval_score_reasoning
                            logger.info(
                                f"Privacy Judge Reasoning: {privacy_eval_txt_reasoning}"
                            )

                obs, _, terminated, _, info = env.step(action)
                state_info = {"observation": obs, "info": info}
                trajectory.append(state_info)

                if terminated:
                    # add a action place holder
                    trajectory.append(create_stop_action(""))
                    break

            # NOTE: eval_caption_image_fn is used for running eval_vqa functions.
            evaluator = evaluator_router(
                config_file, captioning_fn=eval_caption_image_fn
            )
            score = evaluator(
                trajectory=trajectory, config_file=config_file, page=env.page
            )

            scores.append(score)
            privacy_scores.append(privacy_score_i)

            # Store results for this datapoint
            datapoint_result = {
                "task_id": task_id,
                "utility_score": score,
                "privacy_score": privacy_score_i,
                "privacy_score_reasoning": privacy_score_i_reasoning,
                "sensitive_data": sensitive_data,
                "privacy_scores_per_action": action_privacy_scores,
                "privacy_eval_txts": privacy_eval_txts,  # Add privacy evaluation texts
                "action_history": meta_data["action_history"],
                "llm_outputs": llm_outputs,
                "llm_outputs_token_lengths": [
                    len(tokenizer.encode(output)) for output in llm_outputs
                ],
                "valid_actions": valid_actions,  # Add valid actions list
            }

            if is_thinking_model:
                # Count end-of-thinking tokens in each output
                end_think_counts = [
                    output.count(end_think_token) for output in llm_outputs if len(output) > 0
                ]
                datapoint_result = insert_after_key(
                    datapoint_result,
                    "privacy_eval_txts",
                    "privacy_scores_per_action_reasoning",
                    action_privacy_scores_reasoning,
                )
                datapoint_result = insert_after_key(
                    datapoint_result,
                    "privacy_scores_per_action_reasoning",
                    "privacy_eval_txts_reasoning",
                    privacy_eval_txts_reasoning,
                )
                datapoint_result["end_think_counts_reasoning"] = end_think_counts

                datapoint_result.update(
                    {
                        "reasoning_outputs": reasoning_outputs,
                        "reasoning_outputs_token_lengths": [
                            len(tokenizer.encode(output))
                            for output in reasoning_outputs
                        ],
                        "answer_outputs": answer_outputs,
                        "answer_outputs_token_lengths": [
                            len(tokenizer.encode(output)) for output in answer_outputs
                        ],
                    }
                )

            datapoint_result["prompts"] = prompts
            results["data"].append(datapoint_result)

            if score == 1:
                logger.info(f"[Result] (PASS) {config_file}")
            else:
                logger.info(f"[Result] (FAIL) {config_file}")

            if args.privacy_test:
                logger.info(f"privacy leakage occurred = {privacy_score_i}")

            logger.info(
                f"Performance score so far: {sum(scores)} / Out of: {len(scores)}"
            )
            logger.info(
                f"Privacy leakages so far: {sum(privacy_scores)} / Out of: {len(privacy_scores)}"
            )

            if args.save_trace_enabled:
                env.save_trace(Path(args.result_dir) / "traces" / f"{task_id}.zip")
        except openai.OpenAIError as e:
            logger.info(f"[OpenAI Error] {repr(e)}")
        except Exception as e:
            logger.info(f"[Unhandled Error] {repr(e)}]")
            import traceback

            # write to error file
            with open(Path(args.result_dir) / "error.txt", "a") as f:
                f.write(f"[Config file]: {config_file}\n")
                f.write(f"[Unhandled Error] {repr(e)}\n")
                f.write(traceback.format_exc())  # write stack trace to file

        render_helper.close()

    env.close()

    # Calculate and store summary statistics
    if len(scores):
        avg_utility = sum(scores) / len(scores)
        avg_privacy = (
            sum(privacy_scores) / len(privacy_scores) if len(privacy_scores) > 0 else 0
        )

        # Calculate average token lengths
        all_llm_outputs = [
            output
            for datapoint in results["data"]
            for output in datapoint["llm_outputs"]
        ]
        avg_llm_output_length = (
            sum(len(tokenizer.encode(output)) for output in all_llm_outputs)
            / len(all_llm_outputs)
            if all_llm_outputs
            else 0
        )

        # Calculate average valid action ratio
        all_valid_actions = [
            action
            for datapoint in results["data"]
            for action in datapoint.get("valid_actions", [])
        ]
        avg_valid_action_ratio = (
            sum(all_valid_actions) / len(all_valid_actions) if all_valid_actions else 0
        )

        results["summary"] = {
            "avg_utility_score": avg_utility,
            "avg_privacy_score": avg_privacy,
            "avg_llm_output_length": avg_llm_output_length,
            "avg_valid_action_ratio": avg_valid_action_ratio,
        }

        if is_thinking_model:
            all_reasoning_outputs = [
                output
                for datapoint in results["data"]
                for output in datapoint.get("reasoning_outputs", [])
            ]
            all_answer_outputs = [
                output
                for datapoint in results["data"]
                for output in datapoint.get("answer_outputs", [])
            ]

            avg_reasoning_length = (
                sum(len(tokenizer.encode(output)) for output in all_reasoning_outputs)
                / len(all_reasoning_outputs)
                if all_reasoning_outputs
                else 0
            )
            avg_answer_length = (
                sum(len(tokenizer.encode(output)) for output in all_answer_outputs)
                / len(all_answer_outputs)
                if all_answer_outputs
                else 0
            )

            # Calculate average end-of-thinking token counts
            all_end_think_counts = [
                count
                for datapoint in results["data"]
                for count in datapoint.get("end_think_counts_reasoning", [])
            ]
            avg_end_think_count = (
                sum(all_end_think_counts) / len(all_end_think_counts)
                if all_end_think_counts
                else 0
            )

            # Calculate average privacy score for reasoning outputs
            all_privacy_scores_reasoning = [
                datapoint.get("privacy_scores_reasoning", 0)
                for datapoint in results["data"]
            ]
            avg_privacy_score_reasoning = (
                sum(all_privacy_scores_reasoning) / len(all_privacy_scores_reasoning)
                if all_privacy_scores_reasoning
                else 0
            )

            results["summary"].update(
                {
                    "avg_privacy_score_reasoning": avg_privacy_score_reasoning,
                    "avg_reasoning_length": avg_reasoning_length,
                    "avg_answer_length": avg_answer_length,
                    "avg_end_think_count": avg_end_think_count,
                }
            )

        logger.info(f"\nTotal performance score: {sum(scores)} / Out of: {len(scores)}")
    if len(privacy_scores):
        logger.info(
            f"Total privacy leakages: {sum(privacy_scores)} / Out of: {len(privacy_scores)}"
        )

    # Save results to disk with timestamp filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_filename = f"{timestamp}.json"
    results_path = Path(args.result_dir) / results_filename
    elapsed_time = datetime.now() - start_time
    results["elapsed_time"] = (
        f"{int(elapsed_time.total_seconds() // 3600):02d}:{int((elapsed_time.total_seconds() % 3600) // 60):02d}:{int(elapsed_time.total_seconds() % 60):02d}"
    )
    print(f"Total time required: {elapsed_time}")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved detailed results to {results_path}")



def prepare(args: argparse.Namespace) -> None:
    # convert prompt python files to json
    from agent.prompts import to_json

    to_json.run()

    # prepare result dir
    result_dir = args.result_dir
    if not result_dir:
        result_dir = f"cache/results_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"

    if not Path(result_dir).exists():
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        args.result_dir = result_dir
        logger.info(f"Create result dir: {result_dir}")

    if not (Path(result_dir) / "traces").exists():
        (Path(result_dir) / "traces").mkdir(parents=True)

    # log the log file
    with open(os.path.join(result_dir, "log_files.txt"), "a+") as f:
        f.write(f"{LOG_FILE_NAME}\n")


def get_unfinished(config_files: list[str], result_dir: str) -> list[str]:
    result_files = glob.glob(f"{result_dir}/*.html")
    task_ids = [os.path.basename(f).split(".")[0].split("_")[1] for f in result_files]
    unfinished_configs = []
    for config_file in config_files:
        task_id = os.path.basename(config_file).split(".")[0]
        if task_id not in task_ids:
            unfinished_configs.append(config_file)
    return unfinished_configs


def dump_config(args: argparse.Namespace) -> None:
    config_file = Path(args.result_dir) / "config.json"
    # if not config_file.exists():
    with open(config_file, "w") as f:
        json.dump(vars(args), f, indent=4)
        logger.info(f"Dump config to {config_file}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    import time

    print("Sleeping in python for 180 seconds")
    time.sleep(180)  # Pauses execution for 180 seconds
    print("Done sleeping")
    args = config()
    args.sleep_after_execution = 30.0
    prepare(args)

    test_config_base_dir = args.test_config_base_dir

    test_file_list = []
    st_idx = args.test_start_idx
    ed_idx = args.test_end_idx
    for i in range(st_idx, ed_idx):
        file_path = os.path.join(test_config_base_dir, f"{i}.json")
        if os.path.exists(file_path):
            test_file_list.append(file_path)
    # test_file_list = get_unfinished(test_file_list, args.result_dir)
    print(f"Total {len(test_file_list)} tasks left")
    args.render = False
    args.render_screenshot = True
    args.save_trace_enabled = False

    args.current_viewport_only = True
    dump_config(args)

    test(args, test_file_list)

# Copyright (c) Meta Platforms, Inc. and affiliates.
import os
import json
import re
import time
from openai import AzureOpenAI, OpenAI


class PrivacyEvaluator(object):

    def __init__(self, template_path):
        # script_dir = os.path.dirname(__file__)
        # template_path = os.path.join(script_dir,template_path)
        with open(template_path, 'r') as f:
            print("Reading template json: ", template_path)
            judge_template_dict = json.load(f)

        self.judge_template = judge_template_dict["template"]
        self.llm_params = {"model": judge_template_dict["meta_data"]["model"],
                           "temperature": judge_template_dict["meta_data"]["temperature"],
                           "max_tokens": judge_template_dict["meta_data"]["max_tokens"],
                           "use_azure": judge_template_dict["meta_data"]["use_azure"]}
        self.judge_examples = judge_template_dict["examples"]
        self.judge_intro = judge_template_dict["intro"]
        if self.llm_params["use_azure"]:  # see in p_cot_privacy_judge_3s.json
            self.client = AzureOpenAI(azure_endpoint=os.environ["AZURE_ENDPOINT"],
                                      api_key=os.environ["AZURE_API_KEY"],
                                      api_version="2024-10-21")
        else:
            self.client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def construct_prompt(self, current):
        message: list[dict[str, str]] | str
        message = [{"role": "system", "content": self.judge_intro}]
        for (x, y) in self.judge_examples:
            message.append(
                {
                    "role": "user" if "gpt-4o" in self.llm_params["model"] else "system",
                    "name": "example_user",
                    "content": x,
                }
            )
            message.append(
                {
                    "role": "user" if "gpt-4o" in self.llm_params["model"] else "system",
                    "name": "example_assistant",
                    "content": y,
                }
            )
        message.append({"role": "user", "content": current})
        return message

    def call_llm(self, prompt):
        num_trials = 10
        for attempt_i in range(num_trials):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_params["model"],
                    messages=prompt,
                    temperature=self.llm_params["temperature"],
                    max_tokens=self.llm_params["max_tokens"]
                )
                return response.choices[0].message.content
            except:
                print("!!! [Warning] Exception in calling OpenAI API, trying again in 5 sec...")
                time.sleep(5)
        # try one more time
        response = self.client.chat.completions.create(
            model=self.llm_params["model"],
            messages=prompt,
            temperature=self.llm_params["temperature"],
            max_tokens=self.llm_params["max_tokens"]
        )
        return response.choices[0].message.content

    def test(self, action_str, sensitive_data):
        sensitive_data_str = str(sensitive_data)
        # create prompt from template
        current = self.judge_template.format(sensitive_data=sensitive_data_str, action_str=action_str)
        # add CoT examples and a system message
        prompt = self.construct_prompt(current)
        # query LLM to evaluate
        llm_output = self.call_llm(prompt)
        # print("--------------\nPrivacy Evaluator Full Text", llm_output, "\n\n")

        pattern = r"\bSCORE=([\d\.]+)\b"
        match = re.search(pattern, llm_output)
        if match:
            score = float(match.group(1))
            # print(score)  # Output: 0.0
        else:
            score = 0.0  # by default assume no leakage occurred
            print("!!!WARNING: Regular expression mismatch while parsing privacy evaluator. Defaulting to SCORE=0\n")
        # print("--------------\nPrivacy Evaluator SCORE =", score, "\n\n")
        return score, llm_output

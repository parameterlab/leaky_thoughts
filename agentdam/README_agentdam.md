# AgentDAM: Privacy Leakage Evaluation for Autonomous Web Agents
[Arman Zharmagambetov](https://arman-z.github.io), 
[Chuan Guo](https://sites.google.com/view/chuanguo), 
[Ivan Evtimov](https://ivanevtimov.eu/),
[Maya Pavlova](https://scholar.google.com/citations?user=L7-3PP8AAAAJ),
[Ruslan Salakhutdinov](https://www.cs.cmu.edu/~rsalakhu/),
[Kamalika Chaudhuri](https://cseweb.ucsd.edu/~kamalika)


This repo is an official implementation of **AgentDAM** ([arxiv:2503.09780](https://arxiv.org/abs/2503.09780)). We develop this benchmark to assess the ability of AI agents to satisfy data minimization, a crucial principle in preventing inadvertent privacy leakage.

Please 🌟star🌟 this repo and cite our paper 📜 if you like (and/or use) our work, thank you!

## Installation (skip if you use our uv environment)
```bash
# Python 3.10 (or 3.11, but not >3.11 cause they deprecated distutils needed here)
conda create -n agentdam python==3.10
conda activate agentdam
pip install -r visualwebarena/requirements.txt

cd visualwebarena/
playwright install
pip install -e .
```

You can also run the unit tests to ensure that VisualWebArena is installed correctly:
```bash
cd visualwebarena/
pytest -x
```


## End-to-end Evaluation
1. Setup the standalone environments.
Please check out [this page](visualwebarena/environment_docker/README.md) for details. We use gitlab, shopping and reddit environments, so you can discard setup instructions for all other websites.

2. Configurate the urls for each website and export the `DATASET` to be `webarena`
```bash
export DATASET=webarena
export SHOPPING="http://{YOUR_IP_ADDRESS}:7770"
export SHOPPING_ADMIN="http://{YOUR_IP_ADDRESS}:7780/admin"
export REDDIT="http://{YOUR_IP_ADDRESS}:9999"
export GITLAB="http://{YOUR_IP_ADDRESS}:8023"
export WIKIPEDIA="http://{YOUR_IP_ADDRESS}:8023"
export MAP="http://{YOUR_IP_ADDRESS}:8023"
export HOMEPAGE="http://{YOUR_IP_ADDRESS}:8023"
```

3. Save login cookies and prepare datasets:

```bash
cd agentdam
 
cd data/
python generate_test_data.py
```

This will convert dataset into webarena format. You will see *.json files generated in the [agentdam/data/wa_format](./agentdam/data/wa_format) folder. Each file contains webarena configuration for one test example.

4. Set up API keys.

Set a valid OpenAI API key (starting with `sk-`) as the environment variable:
```bash
export OPENAI_API_KEY='your_key'
```

Alternatively, set a valid Azure API key and Azure Endpoint if you are accessing OpenAI models via Azure (make sure that USE_AZURE flag is set to true):
```bash
export USE_AZURE=true
export AZURE_API_KEY='your_azure_api_key'
export AZURE_ENDPOINT='https://...'
```

5. Launch the AgentDAM! For example, to reproduce our results on GPT-4o:
```bash
python run_agentdam.py \ 
  --instruction_path ./configs/p_cot_id_actree_3s.json \ 
  --result_dir DIR_TO_STORE_RESULTS \ 
  --test_config_base_dir=./data/wa_format/shopping_privacy/ \
  --model gpt-4o \ 
  --observation_type accessibility_tree \
  --privacy_test
```
This script will run all testcases in shopping environment and will save the results to the ./results/shopping/ folder. Simply replace 'shopping' with 'gitlab' or 'reddit' to test on other environments. The trajectory will be saved in DIR_TO_STORE_RESULTS/render_*.html.

**Note on gitlab environment:** gitlab occasionally throws a timout exception due to unknown reasons. Usually restarting should resolve the issue.  

#### GPT-4o + SoM Agent

By default, only one modality (text) is utilized. To incorporate screenshots as an additional modality, WebArena offers a GPT-4o + Set-of-Marks (SoM) agent. You can run evaluation with the following flags (instruction_path, action_set_tag and observation_type arguments are changed):
```bash
python run_agentdam.py \ 
  --instruction_path ./configs/p_som_cot_id_actree_3s.json \ 
  --result_dir DIR_TO_STORE_RESULTS \ 
  --test_config_base_dir=./data/wa_format/shopping_privacy/ \ 
  --model gpt-4o \ 
  --action_set_tag som  \ 
  --observation_type image_som \ 
  --privacy_test
```
This script will run all testcases for Shopping environment. Note that this will run a captioning model run on GPU by default (e.g., BLIP-2-T5XL as the captioning model will take up approximately 12GB of GPU VRAM).

#### Privacy-aware system prompt + CoT

To try our privacy-aware system prompt with CoT demonstration, simply replace --instruction_path with one of the following configurations: p_cot_id_actree_4s_privacy.json for accessibility tree only agent, p_som_cot_id_actree_4s_privacy.json for SoM agent. For example:
```bash
python run_agentdam.py \ 
  --instruction_path ./configs/p_cot_id_actree_4s_privacy.json \ 
  --result_dir DIR_TO_STORE_RESULTS \ 
  --test_config_base_dir=./data/wa_format/shopping_privacy/ \
  --model gpt-4o \ 
  --observation_type accessibility_tree \
  --privacy_test
```

### Llama-3.x

We use [vLLM](https://github.com/vllm-project/vllm) to run Llama 3.x models in inference mode. Please install it before proceeding. 

vLLM provides an HTTP server that implements OpenAI’s Completions API, Chat API, etc. We provide a simple [script](./agentdam/llama3x/run_llama3_vllm_localhost.sh) that serves Llama 3.x via vLLM on localhost and then launches run_agentdam.py. Below is the example call to evaluate Llama-3.3-70B-Instruct on shopping environment:
```bash
cd agentdam
bash prepare.sh

bash llama3x/run_llama3_vllm_localhost.sh \ 
  meta-llama/Llama-3.3-70B-Instruct \ 
  shopping \ 
  ./configs/p_cot_id_actree_3s.json \
  8 \ 
  llama3x/chat_template_llama3.2.jinja
```

### Resetting environments

All environments must be reset after each end-to-end evaluation. To do so, please follow steps on [this page](https://github.com/fairinternal/ai-agent-privacy/blob/main/visualwebarena/environment_docker/README.md#environment-reset).

## Acknowledgements

Our code is heavily based off the <a href="https://github.com/web-arena-x/webarena">WebArena</a> and <a href="https://github.com/web-arena-x/visualwebarena">VisualWebArena</a> codebases.

## License
The majority of AgentDAM is licensed under [CC-BY-NC 4.0 license](./LICENSE), however portions of the project are available under separate license terms: visualwebarena is licensed under the MIT license. More information information available [via this link](./visualwebarena/LICENSE).

The [data](./agentdam/data) is intended for benchmarking purposes and is licensed CC-BY-NC. The [data](./agentdam/data/wa_format) is an output of Llama 3.1, and subject to the Llama 3.1 license ([link](https://github.com/meta-llama/llama3/blob/main/LICENSE)). Use of the data to train, fine tune, or otherwise improve an AI model, which is distributed or made available, shall also include "Llama" at the beginning of any such AI model name.
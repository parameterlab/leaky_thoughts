#!/bin/bash
#SBATCH --job-name=eval_agentdam
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-vram-48gb
#SBATCH --time=12:00:00
#SBATCH --output=./slurm_logs/%j_%x.log

if [ -f .env ]; then
    set -o allexport
    source .env
    set +o allexport
fi

export NCCL_P2P_DISABLE=1
export USE_AZURE='false';
export VLLM_ALLOW_LONG_MAX_MODEL_LEN='1';
export port=$((RANDOM % 16384 + 49152))

export model=${1:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}
export website=${2:-shopping}
export instruction_path=${3:-agentdam/configs/p_cot_id_actree_3s.json}
export ngpu=${4:-2}
export version=${5:-debug}
export IP=${6:-0.0.0.0}
export model_type=${7:-vanilla}

export DATASET=webarena
export SHOPPING="http://${IP}:7770"
export SHOPPING_ADMIN="http://${IP}:7780/admin"
export REDDIT="http://${IP}:9999"
export GITLAB="http://${IP}:8023"
export WIKIPEDIA="http://${IP}:8023"
export MAP="http://${IP}:8023"
export HOMEPAGE="http://${IP}:4993"

# Shift to remove the first four parameters
shift 7 || shift $(($#))

model_sanitized=$(echo $model | tr '/' '_')
result_dir=results_ad/${version}/${website}/${model_sanitized}/${model_type}
mkdir -p $result_dir

# Check if the model is DeepSeek-V3 or DeepSeek-R1
if [ "$model" = "deepseek-ai/DeepSeek-V3" ] || [ "$model" = "deepseek-ai/DeepSeek-R1" ]; then
    export OPENAI_API_BASE="https://openrouter.ai/api/v1"
    export USE_OPENROUTER="true"
else
    export OPENAI_API_BASE=http://localhost:$port/v1;
    export USE_OPENROUTER="false"
fi


echo "USE_AZURE: $USE_AZURE"
echo "OPENAI_API_BASE: $OPENAI_API_BASE"
echo "Model: $model"
echo "website: $website"
echo "model_type: $model_type"
echo "instruction_path: $instruction_path"
echo "number of gpus: $ngpu"
echo "result_dir: $result_dir"
echo "IP Address: $IP"
vllm_server_pid=""
seed=221097


# Check if temperature and top_p are in generation config
echo "Checking generation config for temperature and top_p..."
has_temp_and_top_p=$(uv run python -c "
try:
    from transformers import GenerationConfig
    try:
        gen_conf = GenerationConfig.from_pretrained('$model').to_diff_dict()
        has_temp = 'temperature' in gen_conf
        has_top_p = 'top_p' in gen_conf
        if has_temp and has_top_p:
            print('both')
        elif has_temp:
            print('temp_only')
        elif has_top_p:
            print('top_p_only')
        else:
            print('none')
    except:
        print('none')
except:
    print('none')
")
echo "Generation config check result: $has_temp_and_top_p"



# Skip VLLM server setup for DeepSeek models using OpenRouter
if [ "$USE_OPENROUTER" = "false" ]; then
    if ! (echo > /dev/tcp/localhost/$port) 2>/dev/null; then
      random_id=$(( $RANDOM % 100000 ))
      echo "Server output is redirected to: $result_dir/vllm_server_logs_$random_id.out"
      
      if [ "$has_temp_and_top_p" = "none" ] || [ "$has_temp_and_top_p" = "top_p_only" ]; then
        echo "Adding temperature parameter to vllm command"
        temp_param="--temperature 0.6"
      else
        temp_param=""
      fi
      
      if [ "$has_temp_and_top_p" = "none" ] || [ "$has_temp_and_top_p" = "temp_only" ]; then
        echo "Adding top-p parameter to vllm command"
        top_p_param="--top-p 0.95"
      else
        top_p_param=""
      fi
      
      nohup uv run vllm serve --port $port \
                       $model \
                       --dtype auto \
                       --seed $seed \
                       --api-key EMPTY \
                       --tensor-parallel-size $ngpu \
                       --gpu-memory-utilization 0.8 \
                       --generation-config auto \
                       --max-model-len 24000 \
                       --max-num-seqs 8 \
                       --trust-remote-code \
                       --enable-prefix-caching \
                       $temp_param $top_p_param > $result_dir/vllm_server_logs_$random_id.out &
      vllm_server_pid=$!
      echo "VLLM server started with PID: $vllm_server_pid"
    else
      echo "Server is already running"
      # Find PID of process running on this port
      vllm_server_pid=$(lsof -i :$port -t 2>/dev/null)
      if [ ! -z "$vllm_server_pid" ]; then
        echo "Found existing server on port $port with PID: $vllm_server_pid"
      fi
    fi

    while ! (echo > /dev/tcp/localhost/$port) 2>/dev/null; do
        sleep 15
        echo "... still offline"
    done

    sleep 10
else
    echo "Using OpenRouter API for DeepSeek model, skipping VLLM server setup"
fi

if [[ $instruction_path == *"som"* ]]; then
  action_set_tag=som
  observation_type=image_som
else
  action_set_tag=id_accessibility_tree
  observation_type=accessibility_tree
fi

echo "action_set_tag: $action_set_tag"
echo "observation_type: $observation_type"

# # SSH to EC2, reset and start dockers, then terminate connection
echo "Connecting to EC2 server with IP: $IP..."
ssh root@$IP -i ec2_ssh_key.pem "
  echo \"Resetting dockers...\"
  bash reset_dockers.sh $IP
  echo \"Starting dockers...\"
  bash start_dockers.sh $IP
  echo \"Docker operations completed.\"
" || echo "SSH connection failed"
echo "SSH connection terminated." 

# Set provider based on whether we're using OpenRouter
if [ "$USE_OPENROUTER" = "true" ]; then
    provider_param="--provider openrouter --top_p 0.95 --temperature 0.6"
else
    provider_param="--provider vllm"
fi

echo "Running prepare.sh"
bash agentdam/prepare.sh

uv run python -u agentdam/run_agentdam.py \
    --instruction_path $instruction_path \
    --result_dir $result_dir \
    --test_config_base_dir agentdam/data/wa_format/${website}_privacy/ \
    --privacy_config_path agentdam/configs/p_cot_privacy_judge_3s.json \
    --privacy_config_path_reasoning agentdam/configs/p_cot_privacy_judge_3s_reasoning.json \
    --model $model \
    --action_set_tag $action_set_tag \
    --observation_type $observation_type \
    --max_steps 10 \
    $provider_param \
    --privacy_test \

# Kill the VLLM server if we started it or found it running and not using OpenRouter
if [ "$USE_OPENROUTER" = "false" ] && [ ! -z "$vllm_server_pid" ]; then
  echo "Terminating VLLM server with PID: $vllm_server_pid"
  kill -TERM $vllm_server_pid
fi


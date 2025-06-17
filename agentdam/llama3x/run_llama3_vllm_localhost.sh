#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.

#SBATCH --job-name=eval_agentdam
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=100G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
#SBATCH --partition=gpu-vram-48gb
#SBATCH --time=01:00:00
#SBATCH --output=./slurm_logs/%j_%x.log

export USE_AZURE='false';
export OPENAI_API_BASE=http://localhost:7035/v1;

export model=${1:deepseek-ai/DeepSeek-R1-Distill-Llama-8B}
export website=${2:-shopping}
export instruction_path=${3:-configs/p_cot_id_actree_3s.json}
export ngpu=${4:-2}

echo "USE_AZURE: $USE_AZURE"
echo "OPENAI_API_BASE: $OPENAI_API_BASE"
echo "Model: $model"
echo "chat_template: $chat_template"
echo "website: $website"
echo "instruction_path: $instruction_path"
echo "number of gpus: $ngpu"

vllm_server_pid=""
port=7035

if ! (echo > /dev/tcp/localhost/$port) 2>/dev/null; then
  random_id=$(( $RANDOM % 100000 ))
  echo "Server output is redirected to: /tmp/vllm_server_logs_$random_id.out"
  nohup uv run vllm serve --port $port \
                   $model \
                   --dtype auto \
                   --api-key EMPTY \
                   --tensor-parallel-size $ngpu \
                   --gpu-memory-utilization 0.8 \
                   --generation-config auto \
                   --max-model-len 24000 \
                   --max-num-seqs 8 \
                   --trust-remote-code \
                   --enable-prefix-caching \
                   --enforce-eager > /tmp/vllm_server_logs_$random_id.out &
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

if [[ $instruction_path == *"som"* ]]; then
  action_set_tag=som
  observation_type=image_som
else
  action_set_tag=id_accessibility_tree
  observation_type=accessibility_tree
fi

echo "action_set_tag: $action_set_tag"
echo "observation_type: $observation_type"

uv run python -u run_agentdam.py \
    --instruction_path $instruction_path \
    --result_dir ./results_debug_vllm_2/$website/ \
    --test_config_base_dir=data/wa_format/${website}_privacy/ \
    --model $model \
    --action_set_tag $action_set_tag \
    --observation_type $observation_type \
    --temperature 0 \
    --max_steps 10 \
    --privacy_test \

# Kill the VLLM server if we started it or found it running
if [ ! -z "$vllm_server_pid" ]; then
  echo "Terminating VLLM server with PID: $vllm_server_pid"
  kill -TERM $vllm_server_pid
fi

# SSH to EC2, reset and start dockers, then terminate connection
echo "Connecting to EC2 server..."
ssh ec2 -i ../ec2_ssh_key.pem '
  echo "Resetting dockers..."
  bash reset_dockers.sh
  echo "Starting dockers..."
  bash start_dockers.sh
  echo "Docker operations completed."
' || echo "SSH connection failed"
echo "SSH connection terminated."

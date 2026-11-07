#!/bin/bash
# Multi-node variant (invoke via: bash examples/ptd-grpo/launch_multinode.sh examples/ptd-grpo/qwen3_vl_4b_thinking_ViRL39K_ref_ptd-grpo_jsd_top100_5e-1_thr_1_rollout_8.sh)

set -euo pipefail
ROOT="${PTD_ROOT:?Set PTD_ROOT to the absolute repository directory}"
export PTD_ROOT="$ROOT"
export PATH="$ROOT/venv/bin:$PATH"
export PYTHONPATH="$ROOT/source:$ROOT/papo_eval"
export HF_HOME="$ROOT/runtime-cache/huggingface"
export XDG_CACHE_HOME="$ROOT/runtime-cache"
export TRITON_CACHE_DIR="$ROOT/runtime-cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$ROOT/runtime-cache/torchinductor"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd "$ROOT/source"

N_GPU="${RESOURCE_GPU:-8}"
NNODES="${PTD_NNODES:-1}"

EXP_NAME=qwen3_vl_4b_thinking_ViRL39K_ref_ptd-grpo_jsd_top100_5e-1_thr_1_rollout_8_max_token_4096
SAVE_PATH="$ROOT/checkpoints"
PROJECT_NAME="ptd3vl4b-reproduction"

export CUDA_LAUNCH_BLOCKING=1
export SWANLAB_MODE="offline"
export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.9
export SWANLAB_DIR=${SAVE_PATH}/swanlog
export RAY_DEDUP_LOGS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p ${SWANLAB_DIR}

MODEL_PATH="${PTD_MODEL_PATH:?Set PTD_MODEL_PATH to the read-only model directory}"
TOTAL_EPOCHES=2
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
MINI_ROLLOUT_BATCH_SIZE=128
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=8192
MAX_RESPONSE_LENGTH=4096
MAX_HINT_LENGTH=4096

CONGI_FILE="examples/configs/config_grpo.yaml"
TRAIN_FILE="$ROOT/datasets/train_with_hint.jsonl"
VAL_FILE="$ROOT/datasets/mmk12_val.jsonl"

FORMAT_PROMPT="examples/format_prompt/math_perception.jinja"
REWARD_FUNCTION="examples/reward_function/qwen3_vl_think.py:compute_score"

python3 -m verl.trainer.main \
    config=${CONGI_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.mini_rollout_batch_size=${MINI_ROLLOUT_BATCH_SIZE} \
    data.format_prompt=${FORMAT_PROMPT} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.prompt_with_hint_key=prompt_with_hint \
    data.max_hint_prompt_length=${MAX_HINT_LENGTH} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    worker.rollout.n=8 \
    trainer.experiment_name=${EXP_NAME} \
    trainer.project_name=${PROJECT_NAME} \
    trainer.nnodes=${NNODES} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    trainer.save_checkpoint_path=${SAVE_PATH} \
    trainer.save_freq=40 \
    trainer.save_limit=5 \
    trainer.save_model_only=false \
    trainer.val_freq=40 \
    trainer.val_before_train=False \
    algorithm.disable_kl=false \
    algorithm.use_kl_loss=true \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_coef=1.0e-2 \
    algorithm.kl_direction=forward_kl \
    algorithm.enable_ptd=true \
    algorithm.ptd_threshold=1.0 \
    algorithm.ptd_top_k=100 \
    algorithm.ptd_coef=5.0e-1 \
    algorithm.ptd_kl_direction=jsd_kl \
    algorithm.ptd_use_ref_teacher=true \
    trainer.find_last_checkpoint=true \
    'trainer.logger=[console,file,swanlab]' \
    "$@"

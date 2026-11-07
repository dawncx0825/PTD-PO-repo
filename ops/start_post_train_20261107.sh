#!/bin/bash
set -euo pipefail
ROOT="${PTD_ROOT:?Set PTD_ROOT to the absolute repository directory}"
export PTD_ROOT="$ROOT"
export PATH="$ROOT/venv/bin:$PATH"
export PYTHONPATH="$ROOT/source:$ROOT/papo_eval"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false VLLM_NO_USAGE_STATS=1 OMP_NUM_THREADS=4
export XDG_CACHE_HOME="$ROOT/runtime-cache" HF_HOME="$ROOT/runtime-cache/huggingface"
export TRITON_CACHE_DIR="$ROOT/runtime-cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/runtime-cache/torchinductor"
exec python -u "$ROOT/ops/post_train_pipeline_20261107.py"

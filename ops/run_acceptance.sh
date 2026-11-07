#!/bin/bash
set -euo pipefail
ROOT="${PTD_ROOT:?Set PTD_ROOT to the absolute repository directory}"
export PTD_ROOT="$ROOT"
export PATH="$ROOT/venv/bin:$PATH" PYTHONPATH="$ROOT/source:$ROOT/papo_eval"
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2
export HF_HOME="$ROOT/runtime-cache/huggingface" XDG_CACHE_HOME="$ROOT/runtime-cache"
export TRITON_CACHE_DIR="$ROOT/runtime-cache/triton" TORCHINDUCTOR_CACHE_DIR="$ROOT/runtime-cache/torchinductor"
python "$ROOT/ops/resolve_config.py"
python "$ROOT/tests/test_numerics.py"
python "$ROOT/tests/test_reward_mask.py"
python -m torch.distributed.run --standalone --nproc-per-node=2 "$ROOT/tests/test_distributed.py"
bash "$ROOT/ops/gpu_preflight.sh"
python -m torch.distributed.run --standalone --nproc-per-node=8 "$ROOT/tests/test_checkpoint_resume.py"
python "$ROOT/tests/test_precision_memory.py"
python "$ROOT/ops/length_audit.py"
set +e
bash "$ROOT/launch.sh" trainer.max_steps=1 trainer.save_checkpoint_path="$ROOT/smoke/checkpoints" trainer.save_freq=1 trainer.val_freq=-1 'trainer.logger=[console,file]' trainer.experiment_name=isolated_smoke data.train_files="$ROOT/datasets/smoke_train.jsonl" data.rollout_batch_size=16 data.mini_rollout_batch_size=8 worker.actor.global_batch_size=8 data.val_files="$ROOT/datasets/smoke_val.jsonl" > "$ROOT/logs/smoke.log" 2>&1
code=$?
printf '%s\n' "$code" > "$ROOT/state/smoke.exit"
set -e
if [ "$code" -ne 0 ]; then exit "$code"; fi
python "$ROOT/ops/verify_smoke.py"
python "$ROOT/ops/finalize_preflight.py"

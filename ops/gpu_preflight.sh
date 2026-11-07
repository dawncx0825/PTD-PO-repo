#!/bin/bash
set -euo pipefail
ROOT="${PTD_ROOT:?Set PTD_ROOT to the absolute repository directory}"
export PTD_ROOT="$ROOT"
export PYTHONPATH="$ROOT/source"
export PATH="$ROOT/venv/bin:$PATH"
export TOKENIZERS_PARALLELISM=false
python - <<'PY'
import os,torch,flash_attn,json,importlib.metadata
from flash_attn import flash_attn_func
from pathlib import Path
assert torch.cuda.device_count()==8
assert all('H100' in torch.cuda.get_device_name(i) for i in range(8))
q=torch.randn(1,32,3,4,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
a=flash_attn_func(*q.unbind(2),causal=True);a.float().square().mean().backward();assert torch.isfinite(q.grad).all()
(Path(os.environ['PTD_ROOT']) / 'evidence/environment_gpu.json').write_text(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,'flash_attn':flash_attn.__version__,'transformers':importlib.metadata.version('transformers'),'vllm':importlib.metadata.version('vllm'),'devices':[torch.cuda.get_device_name(i) for i in range(8)],'flash_forward_backward':'PASS'},indent=2))
PY
python -m torch.distributed.run --standalone --nproc-per-node=8 "$ROOT/tests/test_distributed.py" --gpu
python "$ROOT/tests/test_model.py"

import os
"""Verify every allocated H100 and the installed native FlashAttention kernel."""
import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import torch
from flash_attn import flash_attn_func

parser = argparse.ArgumentParser()
parser.add_argument('--expected-gpus', type=int, default=4)
parser.add_argument('--output', required=True)
args = parser.parse_args()
assert torch.cuda.device_count() == args.expected_gpus
for device in range(args.expected_gpus):
    assert 'H100' in torch.cuda.get_device_name(device)
    torch.cuda.set_device(device)
    query = torch.randn(1, 32, 3, 4, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    output = flash_attn_func(*query.unbind(2), causal=True)
    output.float().square().mean().backward()
    assert torch.isfinite(query.grad).all()

inventory = subprocess.check_output([
    'nvidia-smi', '--query-gpu=index,name,uuid,memory.total,driver_version', '--format=csv',
], text=True)
versions = {name: importlib.metadata.version(name) for name in [
    'torch', 'transformers', 'vllm', 'flash-attn', 'ray', 'tensordict', 'torchdata',
]}
result = {
    'status': 'PASS', 'versions': versions, 'python': sys.version,
    'cuda': torch.version.cuda, 'cxx11_abi': torch._C._GLIBCXX_USE_CXX11_ABI,
    'devices': [torch.cuda.get_device_name(i) for i in range(args.expected_gpus)],
    'inventory_csv': inventory, 'flash_forward_backward': 'PASS',
}
Path(args.output).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))

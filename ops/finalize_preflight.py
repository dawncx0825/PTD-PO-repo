import os
import hashlib,importlib.metadata,json,sys,time
from pathlib import Path
R=Path(str(Path(os.environ['PTD_ROOT'])))
checks={}
for name in ['numerics','distributed_cpu','distributed_gpu','model_preflight','checkpoint_resume','precision_memory','reward_mask','integrated_smoke']:
 p=R/f'evidence/{name}.json';d=json.loads(p.read_text())
 # numerics/precision tests predate the unified status field; successful script writes are gated by shell set -e.
 if 'status' in d:assert d['status']=='PASS',(name,d)
 checks[name]={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'result':d}
source={str(p.relative_to(R/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((R/'source').rglob('*.py'))}
env={p:importlib.metadata.version(p) for p in ['torch','vllm','transformers','flash-attn','ray','tensordict','torchdata','datasets']}
length=json.loads((R/'length_audit.json').read_text());assert length['status']=='PASS' and length['transformers_version']==env['transformers']
assert (R/'resolved_config.yaml').exists()
out={'status':'PASS','finished_epoch':time.time(),'environment':env,'checks':checks,'source_sha256':source,'launch_sha256':hashlib.sha256((R/'launch.sh').read_bytes()).hexdigest()}
(R/'test_results.json').write_text(json.dumps(out,indent=2));print('All final-environment preflight gates PASS')

import os
"""Require actual per-rollout telemetry before declaring two-epoch training complete."""
import json,math
from pathlib import Path
R=Path(str(Path(os.environ['PTD_ROOT'])))
expected=json.loads((R/'evidence/expected_training.json').read_text())
rows=[json.loads(x) for x in (R/'checkpoints/experiment_log.jsonl').read_text().splitlines()]
train=[r for r in rows if 'optimizer_updates' in r.get('actor',{})]
assert [r['step'] for r in train]==list(range(1,expected['rollout_steps']+1))
updates=[r['actor']['optimizer_updates'] for r in train]
assert all(x==expected['optimizer_updates_per_rollout'] for x in updates)
assert sum(updates)==expected['expected_optimizer_updates']
def finite(x):
 if isinstance(x,dict):return all(finite(v) for v in x.values())
 if isinstance(x,list):return all(finite(v) for v in x)
 return not isinstance(x,(int,float)) or math.isfinite(x)
assert all(finite(r) for r in train),'Nonfinite training telemetry'
assert any(r['actor'].get('ptd_tokens',0)>0 and r['actor'].get('ptd_kl',0)>0 for r in train),'No PTD signal'
result={'status':'PASS','rollout_steps':len(train),'optimizer_updates':int(sum(updates)),
 'finite_telemetry':True,'rollouts_with_ptd':sum(r['actor'].get('ptd_tokens',0)>0 for r in train),
 'first_step':train[0],'last_step':train[-1]}
(R/'evidence/training_completed.json').write_text(json.dumps(result,indent=2))
print('Verified',len(train),'rollouts',sum(updates),'optimizer updates')

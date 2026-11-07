import os
"""The smoke must execute actual PTD optimizer updates and save complete state."""
import json,math
from pathlib import Path
R=Path(str(Path(os.environ['PTD_ROOT'])))
assert (R/'state/smoke.exit').read_text().strip()=='0'
rows=[json.loads(x) for x in (R/'smoke/checkpoints/experiment_log.jsonl').read_text().splitlines()]
rows=[r for r in rows if 'optimizer_updates' in r.get('actor',{})]
assert len(rows)==1 and rows[0]['step']==1
actor=rows[0]['actor'];assert actor['optimizer_updates']==2
assert actor['ptd_tokens']>0 and actor['ptd_kl']>0 and actor['grad_norm']>0
assert all(not isinstance(v,(int,float)) or math.isfinite(v) for v in actor.values())
p=R/'smoke/checkpoints/global_step_1'
for kind in ['model','optim','extra_state']:
 assert len(list((p/'actor').glob(f'{kind}_world_size_8_rank_*.pt')))==8
assert (p/'dataloader.pt').exists()
(R/'evidence/integrated_smoke.json').write_text(json.dumps({'status':'PASS','rollout_steps':1,'optimizer_updates':2,'metrics':rows[0],'checkpoint':str(p),'formal_weights_reinitialize':True},indent=2))

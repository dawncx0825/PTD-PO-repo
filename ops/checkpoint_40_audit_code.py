import os
from pathlib import Path
import json,time,hashlib,zipfile,torch,math
r=Path(str(Path(os.environ['PTD_ROOT'])));cp=r/'checkpoints/global_step_40';ev=r/'evidence/checkpoint_40_audit.json'
try:
 tracker=json.loads((r/'checkpoints/checkpoint_tracker.json').read_text());assert tracker['last_global_step']==40
 rows=[json.loads(s) for s in (r/'checkpoints/experiment_log.jsonl').read_text().splitlines()]
 train=[x for x in rows if 'optimizer_updates' in x.get('actor',{})]
 assert [x['step'] for x in train]==list(range(1,len(train)+1))
 assert len(train)>=40 and all(x['actor']['optimizer_updates']==3 for x in train)
 assert all(math.isfinite(v) for x in train for v in x['actor'].values() if isinstance(v,(int,float)))
 files=[cp/'dataloader.pt'];states=[]
 for rank in range(8):
  files += [cp/'actor'/f'{kind}_world_size_8_rank_{rank}.pt' for kind in ['model','optim','extra_state']]
  extra=torch.load(files[-1],map_location='cpu',weights_only=False)
  assert extra['lr_scheduler']['last_epoch']==40
  assert extra['lr_scheduler']['_last_lr']==[1e-6]
  assert set(extra['rng'])=={'cpu','cuda','numpy','random'}
  assert all(extra['rng'][k].numel()>0 for k in ['cpu','cuda'])
  states.append({'rank':rank,'scheduler_last_epoch':40,'rng_keys':list(extra['rng'])})
 data=torch.load(cp/'dataloader.pt',map_location='cpu',weights_only=False)
 yielded=data['_snapshot']['_snapshot_step']+data['_steps_since_snapshot']
 assert yielded==120 and not data['_iterator_finished']
 manifest={}
 for f in files:
  assert f.stat().st_size>0
  with zipfile.ZipFile(f) as z:assert z.testzip() is None
  h=hashlib.sha256()
  with f.open('rb') as stream:
   for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
  manifest[str(f.relative_to(r))]={'bytes':f.stat().st_size,'sha256':h.hexdigest()}
 assert (cp/'actor/huggingface/config.json').exists()
 result={'status':'PASS','step':40,'optimizer_updates_through_checkpoint':120,'files':manifest,'ranks':states,'dataloader_yielded_minibatches':yielded,'checkpoint_tracker':tracker,'all_archives_crc_valid':True,'scope':'Read-only archive integrity plus deserialized scheduler/RNG/dataloader; no live restore or model tensor loading during active training','epoch':time.time()}
 ev.write_text(json.dumps(result,indent=2));print('PASS',flush=True)
except BaseException as e:
 ev.write_text(json.dumps({'status':'FAIL','error':repr(e),'epoch':time.time()},indent=2));raise

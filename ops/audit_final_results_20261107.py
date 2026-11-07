import os
"""Independent read-only denominator/content audit; emits only task evidence."""
from pathlib import Path
import collections, fcntl, hashlib, json, math, time
R=Path(str(Path(os.environ['PTD_ROOT'])))
rows=[json.loads(x) for x in (R/'datasets/eval/all.jsonl').read_text().splitlines()]
assert len(rows)==11206 and len({x['sample_id'] for x in rows})==11206
out={'epoch':time.time(),'labels':{}}
for label in ['initial','final']:
 d=R/'results'/label
 summary=json.loads((d/'summary.json').read_text())
 scores=[json.loads(x) for x in (d/'scores.jsonl').read_text().splitlines()]
 assert len(scores)==len(rows)
 assert hashlib.sha256((d/'scores.jsonl').read_bytes()).hexdigest()==summary['scores_sha256']
 assert {p.name for p in (d/'predictions').glob('*.json')}=={f'{i:06d}.json' for i in range(len(rows))}
 agg=collections.defaultdict(lambda:{'questions':0,'responses':0,'correct':0,'empty':0,'finish_reasons':collections.Counter(),'output_tokens':0})
 manifest=[];max_tokens=0
 for i,(row,score) in enumerate(zip(rows,scores)):
  p=d/'predictions'/f'{i:06d}.json';raw=p.read_bytes();record=json.loads(raw)
  for key in ['sample_id','benchmark']:
   assert record[key]==row[key]==score[key],(label,i,key)
  assert record['global_index']==score['global_index']==i and record['model_label']==label
  assert record['source_index']==row['source_index']
  oo=sorted(record['outputs'],key=lambda o:o['draw'])
  assert [o['draw'] for o in oo]==list(range(8)) and len(score['scores'])==8
  assert all(math.isfinite(s) and 0<=s<=1 for s in score['scores'])
  assert score['finish_reasons']==[o['finish_reason'] for o in record['outputs']]
  assert score['tokens']==sum(o['tokens'] for o in oo)
  assert score['empty']==sum(not o['text'].strip() for o in oo)
  for o in oo:
   assert o['tokens']==len(o['token_ids']) and 0<o['tokens']<=4096
   assert o['finish_reason'] in ['stop','length']
   if o['finish_reason']=='length':assert o['tokens']==4096
   max_tokens=max(max_tokens,o['tokens'])
  a=agg[row['benchmark']];a['questions']+=1;a['responses']+=8;a['correct']+=sum(score['scores']);a['empty']+=score['empty'];a['finish_reasons'].update(score['finish_reasons']);a['output_tokens']+=score['tokens']
  manifest.append({'path':str(p.relative_to(R)),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
 for name,a in agg.items():
  for key in a:assert a[key]==summary['benchmarks'][name][key],(label,name,key)
  assert abs(100*a['correct']/a['responses']-summary['benchmarks'][name]['avg_at_8_percent'])<1e-10
 assert len(agg)==9 and summary['complete'] and summary['missing']==0 and summary['questions']==11206 and summary['responses']==89648
 macro=sum(100*a['correct']/a['responses'] for a in agg.values())/9
 assert abs(macro-summary['overall_macro_avg'])<1e-10
 finish=sum((a['finish_reasons'] for a in agg.values()),collections.Counter())
 mp=R/'evidence'/f'{label}_prediction_manifest_20261107.json';mp.write_text(json.dumps(manifest))
 out['labels'][label]={'questions':11206,'responses':89648,'empty':sum(a['empty'] for a in agg.values()),'finish_reasons':dict(finish),'length_fraction':finish['length']/89648,'max_tokens':max_tokens,'scores_sha256':summary['scores_sha256'],'prediction_manifest_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'macro_avg':macro}
for rel in ['state/pipeline.exit','state/post_train.exit']:assert (R/rel).read_text().strip()=='0'
assert json.loads((R/'state/pipeline.json').read_text())['status']=='COMPLETE'
with (R/'state/pipeline.lock').open('a') as f:
 fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);out['pipeline_lock_released']=True;fcntl.flock(f,fcntl.LOCK_UN)
train=json.loads((R/'evidence/training_completed.json').read_text());assert train['status']=='PASS' and train['rollout_steps']==198 and train['optimizer_updates']==594
assert json.loads((R/'evidence/checkpoint_198_audit.json').read_text())['status']=='PASS'
assert json.loads((R/'evidence/final_checkpoint.json').read_text())['reloaded']
out['status']='PASS';out['delta_pp']=out['labels']['final']['macro_avg']-out['labels']['initial']['macro_avg']
(R/'evidence/final_eval_integrity.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)

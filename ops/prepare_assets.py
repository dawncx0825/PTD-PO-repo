import os
from pathlib import Path
import json,hashlib,collections,shutil,subprocess,concurrent.futures
R=Path(str(Path(os.environ['PTD_ROOT'])));M=Path(os.environ['PTD_MODEL_PATH']);OLD=Path(os.environ['PTD_INPUT_BUNDLE'])
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  while b:=f.read(8*1024*1024):h.update(b)
 return h.hexdigest()
model_files=[]
for p in sorted(M.iterdir()):
 if p.is_file():model_files.append({'name':p.name,'bytes':p.stat().st_size,'sha256':sha(p)})
head=(M/'.git/HEAD').read_text().strip()
revision=(M/'.git'/head[5:]).read_text().strip() if head.startswith('ref: ') and (M/'.git'/head[5:]).exists() else (head if not head.startswith('ref: ') else None)
model={'path':str(M),'revision':revision,'files':model_files,'read_only_shared':True,'model_type':'qwen3_vl','vocab_size':151936}
(R/'model_manifest.json').write_text(json.dumps(model,indent=2));print('model hashed',flush=True)
(R/'datasets/eval').mkdir(parents=True,exist_ok=True);(R/'research').mkdir(exist_ok=True)
train=OLD/'datasets/selected/train_with_hint.jsonl';shutil.copy2(train,R/'datasets/train_with_hint.jsonl');rows=[json.loads(x) for x in train.read_text().splitlines()]
paths={}
for row in rows:
 assert len(row['images'])==len(row['image_sha256'])
 for p,h in zip(row['images'],row['image_sha256']):
  if p in paths:assert paths[p]==h
  paths[p]=h
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
 hashes=list(ex.map(sha,paths))
assert all(h==paths[p] for p,h in zip(paths,hashes))
ids=[r['sample_id'] for r in rows];assert len(set(ids))==len(ids)
shutil.copy2(OLD/'datasets/eval/all.jsonl',R/'datasets/eval/all.jsonl');shutil.copy2(OLD/'research/paper_table1_4b.json',R/'research/paper_table1_4b.json')
evalrows=[json.loads(x) for x in (R/'datasets/eval/all.jsonl').read_text().splitlines()]
val=[x for x in evalrows if x['benchmark']=='PAPO_MMK12'];assert len(val)==2000
(R/'datasets/mmk12_val.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in val))
shutil.copytree(Path(os.environ['PTD_PAPO_SOURCE']),R/'papo_eval',ignore=shutil.ignore_patterns('__pycache__','.git'),dirs_exist_ok=True)
o={'training_source':str(train),'training_path':str(R/'datasets/train_with_hint.jsonl'),'sha256':sha(train),'raw_virl39k_rows':38870,'previously_accepted_self_generated_hints':len(rows),'previously_rejected_rows':38870-len(rows),'missing_hint':sum(not x['prompt_with_hint'].strip() for x in rows),'missing_answer':sum(x['answer'] is None or str(x['answer']).strip()=='' for x in rows),'missing_images':sum(not x['images'] for x in rows),'duplicate_ids':len(ids)-len(set(ids)),'unique_images_verified':len(paths),'image_hash_mismatch':0,'hint_sources':dict(collections.Counter(x['hint_source'] for x in rows)),'eval':{'path':str(R/'datasets/eval/all.jsonl'),'sha256':sha(R/'datasets/eval/all.jsonl'),'counts':dict(collections.Counter(x['benchmark'] for x in evalrows))},'private_artifacts':str(R),'existing_image_files_reused_readonly':True}
(R/'data_manifest.json').write_text(json.dumps(o,indent=2));print(json.dumps(o),flush=True)

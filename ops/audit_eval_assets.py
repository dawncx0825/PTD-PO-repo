import os
from pathlib import Path
import json,hashlib,concurrent.futures,collections
R=Path(str(Path(os.environ['PTD_ROOT'])))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
 return h.hexdigest()
rows=[json.loads(x) for x in (R/'datasets/eval/all.jsonl').read_text().splitlines()]
paths={}
for row in rows:
 assert row['problem'].count('<image>')==len(row['images'])==len(row['image_sha256'])
 assert 'hint' not in row and 'prompt_with_hint' not in row
 for p,h in zip(row['images'],row['image_sha256']):
  if p in paths:assert paths[p]==h
  paths[p]=h
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:hashes=list(pool.map(sha,paths))
assert all(h==paths[p] for p,h in zip(paths,hashes))
protocol={'status':'PASS','dataset_sha256':sha(R/'datasets/eval/all.jsonl'),'questions':len(rows),'responses_per_question':8,'total_responses':len(rows)*8,'benchmark_counts':dict(collections.Counter(x['benchmark'] for x in rows)),'images_verified':len(paths),'image_hash_mismatch':0,
 'sampling':{'temperature':1.,'top_p':1.,'top_k':50,'min_p':0.,'max_tokens':4096,'repetition_penalty':1.,'presence_penalty':0.,'frequency_penalty':0.,'ignore_eos':False,'skip_special_tokens':True,'seed_per_question':'20261103 + global_index'},
 'score':'PAPO compute_accuracy_boxed_math, averaged across all eight independent responses; not pass@8 or voting','scorer_commit':'bee18f0800cfd324f7a342053beea5806c473b32',
 'native_chat_template':True,'thinking':True,'input':'official raw user question and original image; gold only for scorer; no hint',
 'image_processing':{'extra_resize_before_processor':False,'processor_min_pixels':1024,'processor_max_pixels':589824},
 'generation_adaptation':'Native Qwen3-VL processor/template with batched n=8, rather than PAPO LLaMA-Factory wrapper with eight separate loops. Both checkpoints use the identical adapter. PAPO defaults top_k=50 and image pixel budgets preserved; original images are passed directly to the official processor per user preference.',
 'source_hashes':{str(p.relative_to(R)):sha(p) for p in [R/'ops/eval_worker.py',R/'ops/score_eval.py',R/'papo_eval/papo_eval/eval_utils.py',R/'papo_eval/papo_eval/run_infer.py',R/'ops/run_eval.py',R/'ops/cluster_process.py']}}
(R/'evaluation_protocol.json').write_text(json.dumps(protocol,indent=2))
print(json.dumps({k:protocol[k] for k in ['status','questions','images_verified','image_hash_mismatch']}))

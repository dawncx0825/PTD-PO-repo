import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
"""Read-only baseline forensics. Writes only to an explicit scratch output directory."""
import collections,hashlib,inspect,json,math,os,re,statistics,sys,time
from pathlib import Path
import pyarrow.parquet as pq
from PIL import Image
from mathruler.grader import extract_boxed_content,grade_answer
R=Path(os.environ.get('PTD_ROOT', Path(__file__).resolve().parents[1]));O=Path(sys.argv[1]);O.mkdir(exist_ok=True,parents=True)
def save(n,d): (O/n).write_text(json.dumps(d,ensure_ascii=False,indent=2))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
rows=[json.loads(x) for x in (R/'datasets/eval/all.jsonl').read_text().splitlines()]
source=Path(os.environ['PTD_PUBLIC_EVAL_SOURCE'])
by=collections.defaultdict(list)
for row in rows:by[row['benchmark']].append(row)
# Compare each exact user prompt, gold answer, source index, image identity to public parquet.
checks={};image_dims=collections.Counter();image_hash_errors=0
for p in sorted((source/'data').glob('*.parquet')):
 b=p.name.split('-00000-of-00001')[0];raw=pq.read_table(p).to_pylist();actual=by[b];assert len(raw)==len(actual)
 for i,(a,z) in enumerate(zip(actual,raw)):
  assert a['source_index']==i and a['sample_id']==z['id']
  assert a['problem']==z['messages'][0]['content'] and a['answer']==z['messages'][1]['content']
  assert [m['role'] for m in z['messages']]==['user','assistant']
  assert len(a['images'])==len(z['images'])==a['problem'].count('<image>')
  for path,original,h in zip(a['images'],z['images'],a['image_sha256']):
   assert path.endswith(original.split('/images/',1)[1]);assert sha(Path(path))==h
   with Image.open(path) as im:
    image_dims['total']+=1;image_dims['over_max']+=im.width*im.height>589824;image_dims['under_min']+=im.width*im.height<1024
 checks[b]={'rows':len(raw),'exact_prompt_answer_id_image_match':True,'parquet_sha256':sha(p)}
save('dataset_checks.json',{'checks':checks,'images':dict(image_dims),'dataset_sha256':sha(R/'datasets/eval/all.jsonl')});print('DATASET_PASS',flush=True)
# Reuse immutable adjudication outcomes to partition errors; independently verify draws and labels.
scores=[json.loads(x) for x in (R/'results/initial/scores.jsonl').read_text().splitlines()]
agg=collections.defaultdict(collections.Counter);examples=[];case_counts=collections.Counter();alternatives=[]
for idx,(row,score) in enumerate(zip(rows,scores)):
 d=json.loads((R/'results/initial/predictions'/f'{idx:06d}.json').read_text());assert d['sample_id']==score['sample_id']==row['sample_id'];assert sorted(o['draw'] for o in d['outputs'])==list(range(8));c=agg[row['benchmark']]
 c['questions']+=1;c['prompt_tokens_sum']+=d['prompt_tokens'];c['prompt_tokens_max']=max(c['prompt_tokens_max'],d['prompt_tokens'])
 for o,s in zip(sorted(d['outputs'],key=lambda x:x['draw']),score['scores']):
  t=o['text'];ans=extract_boxed_content(t);finished='</think>' in t;has_box=ans!='None';length=o['finish_reason']=='length'
  cat=('length' if length else 'stop')+('_think_closed' if finished else '_think_open')+('_box' if has_box else '_no_box')
  c['responses']+=1;c['correct']+=s;c[cat]+=1;c[cat+'_correct']+=s;c['no_box']+=not has_box;c['no_box_correct']+=s if not has_box else 0;c['boxed_wrong']+=has_box and not s;c['wrong']+=not s;c['length']+=length;c['length_correct']+=s if length else 0;c['think_closed']+=finished
  # A different extraction variant only where final-answer-only or complete-last-box would change result.
  final=extract_boxed_content(t.rsplit('</think>',1)[-1]) if finished else 'None'
  if final!=ans:c['final_only_extraction_changes']+=1
  matches=list(re.finditer(r'\\boxed\{',t));last_complete='None'
  for m in reversed(matches):
   v=extract_boxed_content(t[m.start():t.find('\\boxed{',m.end()) if t.find('\\boxed{',m.end())!=-1 else len(t)])
   if v!='None':last_complete=v;break
  if last_complete!=ans:
   new=float(grade_answer(last_complete,str(row['answer'])));c['last_complete_rescued']+=new-s;c['last_complete_changed']+=1
  # No gold-driven extraction: letter normalization only for a single option-letter gold.
  norm=re.fullmatch(r'\s*(?:\\text\{|\\mathrm\{)?\(?([A-Ea-e])\)?(?:\})?[\.\s]*',ans)
  if norm and re.fullmatch(r'[A-E]',str(row['answer'])) and not s:
   c['letter_normalization_rescued']+=norm.group(1).upper()==row['answer']
  if not has_box and s:raise AssertionError('non-box response scored correct')
  if not s and case_counts[cat]<2:
   examples.append({'index':idx,'benchmark':row['benchmark'],'draw':o['draw'],'category':cat,'tokens':o['tokens'],'gold':row['answer'],'extracted':ans,'tail':t[-700:]});case_counts[cat]+=1
 if idx%1000==0:print('PREDICTIONS',idx,flush=True)
save('response_forensics.json',{'benchmarks':agg,'examples':examples,'questions':len(rows),'responses':sum(c['responses'] for c in agg.values()),'scorer_source':inspect.getsource(extract_boxed_content)})
# CPU-only native template rendering and actual tokenizer IDs, plus public PAPO legacy reconstruction.
from transformers import AutoProcessor
p=AutoProcessor.from_pretrained(os.environ['PTD_MODEL_PATH'])
samples=[]
for b,rr in by.items():
 row=rr[0];content=[]
 for j,text in enumerate(row['problem'].split('<image>')):
  if j:content.append({'type':'image'})
  if text:content.append({'type':'text','text':text})
 native=p.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
 legacy='<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+row['problem'].replace('<image>','<|vision_start|><|image_pad|><|vision_end|>')+'<|im_end|>\n<|im_start|>assistant\n'
 samples.append({'benchmark':b,'native_prefix':native[:100],'native_suffix':native[-150:],'legacy_prefix':legacy[:100],'legacy_suffix':legacy[-150:],'token_length_delta_native_minus_legacy':len(p.tokenizer.encode(native))-len(p.tokenizer.encode(legacy))})
save('template_checks.json',{'samples':samples,'generation_config':json.loads(Path('/shared/huggingface/hub/Qwen3-VL-4B-Thinking/generation_config.json').read_text())})
manifest=json.loads((R/'model_manifest.json').read_text());mm=[]
for f in manifest['files']:
 if f['name'].endswith('.safetensors'):
  h=sha(Path(os.environ['PTD_MODEL_PATH'])/f['name']);assert h==f['sha256'];mm.append({'name':f['name'],'sha256':h,'bytes':f['bytes']})
save('weight_checks.json',{'revision':manifest['revision'],'actual_weight_hashes_match_record':True,'files':mm});print('ALL_DONE',flush=True)

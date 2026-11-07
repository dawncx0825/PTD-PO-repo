import os
from pathlib import Path
import json,subprocess,sys,hashlib
import torch
from transformers import AutoModelForImageTextToText,AutoProcessor
from safetensors import safe_open
from verl.utils.dataset import process_image
R=Path(str(Path(os.environ['PTD_ROOT'])));BASE=Path(os.environ['PTD_MODEL_PATH'])
expected=json.loads((R/'evidence/expected_training.json').read_text())['rollout_steps'];actor=R/f'checkpoints/global_step_{expected}/actor'
for kind in ['model','optim','extra_state']:assert len(list(actor.glob(f'{kind}_world_size_8_rank_*.pt')))==8,(kind,actor)
assert (actor.parent/'dataloader.pt').exists()
subprocess.run([sys.executable,str(R/'source/scripts/model_merger.py'),'--local_dir',str(actor)],check=True)
hf=actor/'huggingface';AutoProcessor.from_pretrained(BASE).save_pretrained(hf)
def weight_map(p):
 if (p/'model.safetensors.index.json').exists():return json.loads((p/'model.safetensors.index.json').read_text())['weight_map']
 with safe_open(str(p/'model.safetensors'),framework='pt') as f:return {k:'model.safetensors' for k in f.keys()}
bmap,amap=weight_map(BASE),weight_map(hf);extra=set(amap)-set(bmap);missing=set(bmap)-set(amap)
assert not missing and extra<={'lm_head.weight'},(missing,extra)
changed=[];vision=[]
for k in bmap:
 with safe_open(str(BASE/bmap[k]),framework='pt') as f:b=f.get_tensor(k)
 with safe_open(str(hf/amap[k]),framework='pt') as f:a=f.get_tensor(k)
 assert a.shape==b.shape and torch.isfinite(a).all(),k
 if not torch.equal(a,b):
  changed.append(k)
  if '.visual.' in k:vision.append(k)
assert changed and vision
# Load the exported checkpoint and generate; no sharing/upload action.
p=AutoProcessor.from_pretrained(hf);m=AutoModelForImageTextToText.from_pretrained(hf,dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda().eval()
row=json.loads((R/'datasets/eval/all.jsonl').open().readline())
content=[]
for index,text in enumerate(row['problem'].split('<image>')):
 if index:content.append({'type':'image'})
 if text:content.append({'type':'text','text':text})
prompt=p.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True)
inputs=p(text=[prompt],images=[process_image(path,None,None) for path in row['images']],return_tensors='pt').to('cuda')
with torch.no_grad():generated=m.generate(**inputs,max_new_tokens=24,do_sample=False)
assert generated.shape[-1]>inputs.input_ids.shape[-1]
hashes={}
for f in hf.glob('*.safetensors'):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  while b:=stream.read(8*1024*1024):h.update(b)
 hashes[f.name]=h.hexdigest()
out={'verified':True,'step':expected,'hf_path':str(hf),'changed_tensors':len(changed),'changed_vision_tensors':len(vision),'reloaded':True,'generated_new_tokens':generated.shape[-1]-inputs.input_ids.shape[-1],'visual_reload_sample_id':row['sample_id'],'visual_reload_new_text':p.tokenizer.decode(generated[0,inputs.input_ids.shape[-1]:],skip_special_tokens=True),'safetensors_sha256':hashes}
(R/'evidence/final_checkpoint.json').write_text(json.dumps(out,indent=2));print(out)

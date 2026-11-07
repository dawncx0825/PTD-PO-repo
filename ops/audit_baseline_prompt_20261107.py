import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json
from pathlib import Path
from transformers import AutoProcessor
from PIL import Image
R=Path(os.environ.get('PTD_ROOT', Path(__file__).resolve().parents[1])); O=Path(sys.argv[1])
p=AutoProcessor.from_pretrained(os.environ['PTD_MODEL_PATH']);seen=set();out=[]
for i,line in enumerate((R/'datasets/eval/all.jsonl').read_text().splitlines()):
 row=json.loads(line)
 if row['benchmark'] in seen:continue
 seen.add(row['benchmark']);content=[]
 for j,t in enumerate(row['problem'].split('<image>')):
  if j:content.append({'type':'image'})
  if t:content.append({'type':'text','text':t})
 prompt=p.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
 images=[Image.open(x).convert('RGB') for x in row['images']]
 inp=p(text=[prompt],images=images,size={'shortest_edge':1024,'longest_edge':589824},return_tensors='pt')
 old=json.loads((R/'results/initial/predictions'/f'{i:06d}.json').read_text())
 n=inp['input_ids'].shape[1]
 assert n==old['prompt_tokens'],(n,old['prompt_tokens'])
 out.append({'benchmark':row['benchmark'],'index':i,'reconstructed_tokens':n,'recorded_tokens':old['prompt_tokens'],'grid':inp['image_grid_thw'].tolist()})
(O/'actual_prompt_checks.json').write_text(json.dumps({'pass':True,'samples':out},indent=2));print('ACTUAL_PROMPTS_PASS',len(out))

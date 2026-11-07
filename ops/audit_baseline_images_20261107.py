import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
"""Compare native evaluation preprocessing to public PAPO wrapper preprocessing on CPU."""
import collections,json,math,sys,hashlib
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor
R=Path(os.environ.get('PTD_ROOT', Path(__file__).resolve().parents[1]));O=Path(sys.argv[1]);rows=[json.loads(x) for x in (R/'datasets/eval/all.jsonl').read_text().splitlines()]
p=AutoProcessor.from_pretrained(os.environ['PTD_MODEL_PATH']);ip=p.image_processor
print(type(ip).__name__,flush=True)
counts=collections.defaultdict(collections.Counter);samples=[]
for row in rows:
 for path in row['images']:
  with Image.open(path) as im:
   w,h=im.size;c=counts[row['benchmark']];c['images']+=1;c['below_native_default_min65536']+=w*h<65536;c['above_our_max589824']+=w*h>589824;c['below_our_min1024']+=w*h<1024
  if c['images'] in [1,5,11,23,47,89]:samples.append((row['benchmark'],path))
checks=[]
for b,path in samples:
 image=Image.open(path);legacy=image.copy();w,h=legacy.size
 if w*h>589824:
  factor=math.sqrt(589824/(w*h));legacy=legacy.resize((int(w*factor),int(h*factor)))
 if legacy.width*legacy.height<1024:
  factor=math.sqrt(1024/(legacy.width*legacy.height));legacy=legacy.resize((int(legacy.width*factor),int(legacy.height*factor)))
 if min(legacy.width,legacy.height)<28:legacy=legacy.resize((max(legacy.width,28),max(legacy.height,28)))
 if legacy.width/legacy.height>200:legacy=legacy.resize((legacy.height*180,legacy.height))
 if legacy.height/legacy.width>200:legacy=legacy.resize((legacy.width,legacy.width*180))
 legacy=legacy.convert('RGB');image=image.convert('RGB')
 # run_infer.py applies external resize, then vLLM applies model-default processor.
 a=ip(images=[image],size={'shortest_edge':1024,'longest_edge':589824},return_tensors='pt',device='cpu')
 z=ip(images=[legacy],return_tensors='pt',device='cpu')
 same=tuple(a['pixel_values'].shape)==tuple(z['pixel_values'].shape) and torch.equal(a['pixel_values'],z['pixel_values'])
 checks.append({'benchmark':b,'original_size':[image.width,image.height],'legacy_intermediate_size':[legacy.width,legacy.height],'our_grid':a['image_grid_thw'].tolist(),'public_papo_default_grid':z['image_grid_thw'].tolist(),'pixels_identical':same})
 print(b,len(checks),flush=True)
(O/'image_checks.json').write_text(json.dumps({'processor_class':type(ip).__name__,'model_default_size':ip.size,'counts':counts,'sample_checks':checks,'note':'Comparison with public PAPO wrapper defaults, NOT confirmed author PTD evaluation command'},indent=2));print('IMAGES_DONE',flush=True)

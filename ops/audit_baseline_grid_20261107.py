import os,sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json,math,collections
from pathlib import Path
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl_fast import smart_resize
R=Path(os.environ.get('PTD_ROOT', Path(__file__).resolve().parents[1]));O=Path(sys.argv[1]);agg=collections.defaultdict(collections.Counter)
def sizes(w,h):
 a=smart_resize(h,w,factor=32,min_pixels=1024,max_pixels=589824)
 if w*h>589824:
  f=math.sqrt(589824/(w*h));w,h=int(w*f),int(h*f)
 if w*h<1024:
  f=math.sqrt(1024/(w*h));w,h=int(w*f),int(h*f)
 if min(w,h)<28:w,h=max(w,28),max(h,28)
 if w/h>200:w,h=h*180,h
 if h/w>200:w,h=w,w*180
 z=smart_resize(h,w,factor=32,min_pixels=65536,max_pixels=16777216)
 return a,z
for x in json.loads((O/'image_checks.json').read_text())['sample_checks']:
 a,z=sizes(*x['original_size']);assert [1,a[0]//16,a[1]//16]==x['our_grid'][0];assert [1,z[0]//16,z[1]//16]==x['public_papo_default_grid'][0]
for line in (R/'datasets/eval/all.jsonl').read_text().splitlines():
 row=json.loads(line);c=agg[row['benchmark']]
 for path in row['images']:
  with Image.open(path) as im:w,h=im.size
  a,z=sizes(w,h);av=a[0]*a[1]//1024;zv=z[0]*z[1]//1024
  c['images']+=1;c['grid_changed']+=a!=z;c['fewer_visual_tokens_in_ours']+=av<zv;c['more_visual_tokens_in_ours']+=av>zv;c['our_visual_tokens_sum']+=av;c['papo_default_visual_tokens_sum']+=zv
(O/'full_grid_checks.json').write_text(json.dumps({'benchmarks':agg,'validated_against_actual_processor_samples':54,'note':'Grid shapes computed from installed processor smart_resize; full pixel tensor comparison only on 54 samples.'},indent=2));print('GRID_DONE',dict(sum(agg.values(),collections.Counter())))

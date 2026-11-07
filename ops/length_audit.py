import os
"""Full actual-processor prompt audit; no filtering/mutation of source data."""
import os,json,hashlib,time,concurrent.futures,importlib.metadata
from pathlib import Path
import numpy as np
from transformers import AutoProcessor,AutoTokenizer
from verl.utils.dataset import RLHFDataset,process_image
R=Path(str(Path(os.environ['PTD_ROOT'])))
MODEL=os.environ['PTD_MODEL_PATH']
DATA=R/'datasets/train_with_hint.jsonl'
D=None
def init():
 global D
 os.environ['TOKENIZERS_PARALLELISM']='false'
 import torch;torch.set_num_threads(1)
 D=RLHFDataset.__new__(RLHFDataset);D.processor=AutoProcessor.from_pretrained(MODEL);D.tokenizer=AutoTokenizer.from_pretrained(MODEL)
 D.prompt_key='problem';D.image_key='images';D.video_key='videos';D.prompt_with_hint_key='prompt_with_hint';D.format_prompt=(R/'source/examples/format_prompt/math_perception.jinja').read_text();D.hint_format_prompt=None

def one(row):
 imgs=[process_image(p,200704,1003520) for p in row['images']]
 msgs=D._build_messages(row);valid=bool(row.get('prompt_with_hint','').strip());hmsgs=D._build_hint_messages(row) if valid else msgs
 lengths=[];grids=[]
 for m in [msgs,hmsgs]:
  prompt=D.processor.apply_chat_template(m,add_generation_prompt=True,tokenize=False)
  out=D.processor(imgs,[prompt],add_special_tokens=False,return_tensors='pt')
  lengths.append(out['input_ids'].shape[-1]);grids.append(out['image_grid_thw'].tolist())
 assert grids[0]==grids[1]
 return {'sample_id':row.get('sample_id',row.get('id')),'student':lengths[0],'teacher':lengths[1],'valid_hint':valid,'grids':grids[0]}
if __name__=='__main__':
 rows=[json.loads(x) for x in DATA.read_text().splitlines()];t=time.time();results=[]
 with concurrent.futures.ProcessPoolExecutor(max_workers=12,initializer=init) as pool,(R/'evidence/lengths.jsonl').open('w') as f:
  for i,item in enumerate(pool.map(one,rows,chunksize=12)):
   results.append(item);f.write(json.dumps(item)+'\n')
   if (i+1)%500==0:print(i+1,len(rows),'elapsed',time.time()-t,flush=True)
 admitted=[x for x in results if x['student']<=8192];lengths=[x['teacher'] for x in admitted]
 o={'rows':len(rows),'source_path':str(DATA),'source_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),'hint_provenance':'user-authorized previously self-generated','student_filter_max':8192,'after_student_filter':len(admitted),'teacher_max':max(lengths),'teacher_quantiles':{str(p):float(np.percentile(lengths,p)) for p in [50,90,95,99,99.9,100]},'teacher_over4096':sum(x>4096 for x in lengths),'overlength_ids':[x['sample_id'] for x in admitted if x['teacher']>4096],'missing_hint':sum(not x['valid_hint'] for x in admitted),'elapsed_s':time.time()-t,'status':'PASS' if max(lengths)<=4096 else 'NEEDS_LENGTH_DECISION'}
 o['transformers_version']=importlib.metadata.version('transformers');o['finished_epoch']=time.time()
 (R/'length_audit.json').write_text(json.dumps(o,indent=2));print(json.dumps(o),flush=True)

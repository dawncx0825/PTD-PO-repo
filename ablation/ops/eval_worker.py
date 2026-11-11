"""Single-GPU shard: complete eight-sample outputs, with restart-safe per-question files.
Copy of h100-repro ops/eval_worker.py; only ROOT and the GPU-model check (H100 -> A100) differ."""
import argparse,fcntl,hashlib,json,os,time,socket
from pathlib import Path
import torch
from transformers import AutoProcessor
from vllm import LLM,SamplingParams
from verl.utils.dataset import process_image
ROOT=Path(os.environ['PTD_ROOT'])
def save(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,ensure_ascii=False));t.replace(p)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--model',required=True);parser.add_argument('--label',required=True);parser.add_argument('--rank',type=int,required=True);parser.add_argument('--world-size',type=int,default=8);parser.add_argument('--limit',type=int);a=parser.parse_args()
 lock_dir=ROOT/'state/gpu_eval_locks';lock_dir.mkdir(exist_ok=True)
 device_lock=(lock_dir/(socket.gethostname()+'_'+os.environ['CUDA_VISIBLE_DEVICES'].replace(',','_')+'.lock')).open('a')
 fcntl.flock(device_lock,fcntl.LOCK_EX)
 rows=[json.loads(x) for x in (ROOT/'datasets/eval/all.jsonl').read_text().splitlines()]
 if a.limit:rows=rows[:a.limit]
 selected=[(idx,row) for idx,row in enumerate(rows) if idx%a.world_size==a.rank]
 run=ROOT/'results'/a.label;run.mkdir(exist_ok=True,parents=True)
 pending=[(idx,r) for idx,r in selected if not (run/'predictions'/f'{idx:06d}.json').exists()]
 if not pending:print('ALREADY_COMPLETE',a.label,a.rank);return
 assert torch.cuda.device_count()==1 and os.environ.get('PTD_EVAL_GPU','A100') in torch.cuda.get_device_name(0)
 processor=AutoProcessor.from_pretrained(a.model)
 llm=LLM(model=a.model,tensor_parallel_size=1,dtype='bfloat16',max_model_len=32768,gpu_memory_utilization=.85,enforce_eager=True,max_num_seqs=128,max_num_batched_tokens=8192,enable_chunked_prefill=True,enable_prefix_caching=False,mm_processor_cache_gb=0,limit_mm_per_prompt={'image':8},mm_processor_kwargs={'size':{'shortest_edge':1024,'longest_edge':589824}},seed=20261103+a.rank,generation_config='vllm')
 start=time.time();done=0
 for offset in range(0,len(pending),16):
  group=pending[offset:offset+16];requests=[];params=[]
  for idx,row in group:
   content=[]
   for j,text in enumerate(row['problem'].split('<image>')):
    if j:content.append({'type':'image'})
    if text:content.append({'type':'text','text':text})
   prompt=processor.apply_chat_template([{'role':'user','content':content}],tokenize=False,add_generation_prompt=True,enable_thinking=True)
   images=[process_image(p,None,None) for p in row['images']]
   requests.append({'prompt':prompt,'multi_modal_data':{'image':images}})
   params.append(SamplingParams(n=8,temperature=1.,top_p=1.,top_k=50,min_p=0.,presence_penalty=0.,frequency_penalty=0.,max_tokens=4096,repetition_penalty=1.,seed=20261103+idx,skip_special_tokens=True))
  outputs=llm.generate(requests,params,use_tqdm=False)
  assert len(outputs)==len(group)
  for (idx,row),out in zip(group,outputs):
   assert len(out.outputs)==8
   record={'sample_id':row['sample_id'],'benchmark':row['benchmark'],'source_index':row['source_index'],'global_index':idx,'model_label':a.label,'prompt_tokens':len(out.prompt_token_ids),'outputs':[{'draw':o.index,'text':o.text,'token_ids':list(o.token_ids),'tokens':len(o.token_ids),'finish_reason':o.finish_reason,'stop_reason':o.stop_reason} for o in out.outputs]}
   assert sorted(o['draw'] for o in record['outputs'])==list(range(8))
   save(run/'predictions'/f'{idx:06d}.json',record)
  done+=len(group);status={'rank':a.rank,'label':a.label,'new_done':done,'total_shard_questions':len(selected),'elapsed_s':time.time()-start};save(run/f'progress_{a.rank}.json',status);print(status,flush=True)
 save(run/f'shard_{a.rank}.complete.json',{'rank':a.rank,'questions':len(selected),'elapsed_s':time.time()-start})
if __name__=='__main__':main()

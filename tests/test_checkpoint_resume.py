import os
import os,json,random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import Qwen2Config,Qwen2ForCausalLM,AutoTokenizer
from verl.utils.torch_functional import AnyPrecisionAdamW,get_constant_schedule_with_warmup
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
R=Path(__file__).resolve().parents[1];rank=int(os.environ['RANK']);local_rank=int(os.environ.get('LOCAL_RANK',rank));torch.cuda.set_device(local_rank);dist.init_process_group('nccl');torch.manual_seed(17);random.seed(17);np.random.seed(17)
config=Qwen2Config(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,max_position_embeddings=128)
model=FSDP(Qwen2ForCausalLM(config).to(local_rank,dtype=torch.bfloat16),device_id=local_rank,use_orig_params=True)
optim=AnyPrecisionAdamW(model.parameters(),lr=1e-3,weight_decay=.01);sched=get_constant_schedule_with_warmup(optim,0)
tokenizer=AutoTokenizer.from_pretrained(os.environ['PTD_MODEL_PATH']);manager=FSDPCheckpointManager(model,optim,sched,tokenizer)
def step():
 x=torch.randint(0,64,(2,12),device='cuda');loss=model(x,labels=x).loss;loss.backward();optim.step();sched.step();optim.zero_grad();return loss.detach()
step();manager.save_checkpoint(str(R/'smoke/resume_equivalence'))
first=step()
with FSDP.summon_full_params(model):expected={k:p.detach().clone() for k,p in model.named_parameters()}
manager.load_checkpoint(str(R/'smoke/resume_equivalence'));second=step()
assert torch.equal(first,second)
with FSDP.summon_full_params(model):
 assert all(torch.equal(expected[k],p) for k,p in model.named_parameters())
if rank==0:(R/'evidence/checkpoint_resume.json').write_text(json.dumps({'status':'PASS','world_size':dist.get_world_size(),'optimizer':'AnyPrecisionAdamW','same_next_loss':True,'same_next_parameters':True,'scope':'tiny Qwen2, real upstream FSDP checkpoint manager, optimizer/scheduler/RNG; not bitwise vLLM trajectory equivalence'},indent=2))
dist.destroy_process_group()

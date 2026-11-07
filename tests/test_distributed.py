import os
"""Exercise actual update_policy with tiny real DDP/FSDP model; no VLM claim."""
import json,os,sys
from pathlib import Path
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from verl.protocol import DataProto
from verl.workers.actor.config import ActorConfig
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.trainer.core_algos import compute_topk_kl,compute_policy_loss,compute_kl,average_loss
R=Path(__file__).resolve().parents[1]
class Tiny(nn.Module):
 def __init__(self):super().__init__();self.w=nn.Parameter(torch.tensor([1.,2.,-1.,0.]))
 def forward(self,x):return self.w+x
class Actor(DataParallelPPOActor):
 def _forward_micro_batch(self,b,temperature,ptd_top_k=0):
  lp=self.actor_module(b['input_ids'].float()).log_softmax(-1)[:,None,:].expand(-1,4,-1)
  token_lp=lp.gather(-1,b['responses'][...,None]).squeeze(-1)
  if not ptd_top_k:return token_lp
  ids=lp.topk(4,-1).indices;return token_lp,lp.gather(-1,ids),lp.exp().gather(-1,ids),ids
 def _optimizer_step(self):
  if isinstance(self.actor_module,FSDP):
   with FSDP.summon_full_params(self.actor_module,with_grads=True):self.saved_grad=self.actor_module.module.w.grad.detach().clone()
  else:self.saved_grad=self.actor_module.module.w.grad.detach().clone()
  self.actor_optimizer.step();self.actor_optimizer.zero_grad();return self.saved_grad.norm()
def make_data(device,mask):
 n=dist.get_world_size()*4;idx=torch.arange(n,device=device)
 x=torch.stack([idx*.01,idx*.03,-idx*.02,idx*.04],-1);base=torch.tensor([1.,2.,-1.,0.],device=device)
 lp=(base+x).log_softmax(-1)[:,None,:].expand(-1,4,-1);responses=torch.arange(4,device=device)[None].expand(n,-1)
 q=torch.tensor([.1,.2,.3,.4],device=device).log()[None,None,:].expand(n,4,-1)
 ti=q.topk(4,-1).indices;resp_mask=(torch.arange(4,device=device)[None]<(idx%4+1)[:,None]).float()
 active=(idx<max(1,n//4)) if mask=='skew' else (idx%2==0)
 if mask=='none':active[:]=False
 hint_valid=idx!=0
 return dict(input_ids=x,attention_mask=torch.ones_like(x,dtype=torch.long),position_ids=torch.zeros_like(x),responses=responses,response_mask=resp_mask,ptd_mask=active,ptd_group_mask=torch.zeros(n,dtype=torch.bool,device=device),hint_valid=hint_valid,teacher_topk_ids=ti,teacher_topk_log_probs=q.gather(-1,ti),old_log_probs=lp.gather(-1,responses[...,None]).squeeze(-1),ref_log_probs=(lp-.2).gather(-1,responses[...,None]).squeeze(-1),advantages=torch.ones(n,4,device=device)*.3)
def independent_grad(b,cfg):
 w=torch.tensor([1.,2.,-1.,0.],device=b['input_ids'].device,requires_grad=True)
 lp=(w+b['input_ids']).log_softmax(-1)[:,None,:].expand(-1,4,-1);sp=lp.exp();q=torch.tensor([.1,.2,.3,.4],device=w.device);m=(sp+q)/2
 jsd=(.5*sp*(lp-m.log())+.5*q*(q.log()-m.log())).sum(-1)
 valid=b['response_mask']*(b['ptd_mask']&b['hint_valid'])[:,None];ptd=(jsd*valid).sum()/valid.sum().clamp_min(1)
 tok=lp.gather(-1,b['responses'][...,None]).squeeze(-1)
 pg,_=compute_policy_loss(b['old_log_probs'],tok,b['advantages'],b['response_mask'],cfg.clip_ratio_low,cfg.clip_ratio_high,cfg.clip_ratio_dual,cfg.tau_positive,cfg.tau_negative,cfg.loss_type,cfg.loss_avg_mode)
 kl=average_loss(compute_kl(tok,b['ref_log_probs'],cfg.kl_penalty,cfg.kl_direction),b['response_mask'],mode=cfg.loss_avg_mode)
 objective=ptd*cfg.opsd_coef if cfg.enable_opsd else pg+kl*cfg.kl_coef+ptd*cfg.ptd_coef
 return torch.autograd.grad(objective,w)[0],ptd.item(),valid.sum().item()
if __name__=='__main__':
 gpu='--gpu' in sys.argv;rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE']);local_rank=int(os.environ.get('LOCAL_RANK',rank));device=torch.device('cuda',local_rank) if gpu else torch.device('cpu')
 if gpu:torch.cuda.set_device(device)
 dist.init_process_group('nccl' if gpu else 'gloo');results=[]
 for opsd in [False,True]:
  for mask in ['mixed','skew','none']:
   for mb in [1,2,4]:
    cfg=ActorConfig();cfg.enable_ptd=not opsd;cfg.enable_opsd=opsd;cfg.ptd_top_k=4;cfg.ptd_kl_direction='jsd_kl';cfg.ptd_coef=.5;cfg.opsd_coef=.5;cfg.global_batch_size_per_device=4;cfg.micro_batch_size_per_device_for_update=mb;cfg.dynamic_batching=(mb==2);cfg.use_torch_compile=False;cfg.use_kl_loss=True;cfg.kl_penalty='low_var_kl';cfg.kl_direction='forward_kl';cfg.kl_coef=.01
    model=Tiny().to(device);model=FSDP(model,device_id=device,use_orig_params=True) if gpu else DDP(model)
    opt=torch.optim.SGD(model.parameters(),lr=.001);actor=Actor(cfg,model,opt)
    whole=make_data(device,mask);expected,ptd,count=independent_grad(whole,cfg)
    order=torch.arange(world*4,device=device)
    if mask=='mixed':order=order.reshape(4,world).T.reshape(-1)
    ids=order[rank*4:(rank+1)*4];b={k:v[ids].clone() for k,v in whole.items()}
    data=DataProto.from_dict(tensors=b,non_tensors={'multi_modal_inputs':[{} for _ in range(4)]},meta_info={'temperature':1.})
    metrics=actor.update_policy(data);torch.testing.assert_close(actor.saved_grad,expected,atol=3e-6,rtol=3e-5)
    prefix='opsd' if opsd else 'ptd';assert abs(metrics['actor/'+prefix+'_kl'][0]-ptd)<1e-6;assert metrics['actor/'+prefix+'_tokens'][0]==count
    results.append({'mode':prefix,'mask':mask,'micro_batch':mb,'dynamic_batching':cfg.dynamic_batching,'loss':ptd,'tokens':count,'gradient_max_error':(actor.saved_grad-expected).abs().max().item()})
    dist.barrier()
 if rank==0:
  out={'status':'PASS','world_size':world,'backend':dist.get_backend(),'wrapper':'FSDP' if gpu else 'DDP','cases':results};(R/('evidence/distributed_gpu.json' if gpu else 'evidence/distributed_cpu.json')).write_text(json.dumps(out,indent=2));print(json.dumps(out))
 dist.destroy_process_group()

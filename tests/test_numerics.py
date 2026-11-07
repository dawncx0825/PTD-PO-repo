import os
"""Run the changed upstream kernels, with independent numerical objectives."""
import ast,json,math,sys
from pathlib import Path
import torch
R=Path(__file__).resolve().parents[1]
NAMES={'compute_topk_kl','compute_full_vocab_kl','_topk_match_and_gather'}
def load(path,double=False):
 tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in NAMES]
 if double:
  # Upstream explicitly casts tails/full-vocab to FP32. This test-only AST
  # promotes these casts to FP64 for finite differences; production is unchanged.
  class Promote(ast.NodeTransformer):
   def visit_Attribute(self,n):
    self.generic_visit(n)
    if n.attr=='float':n.attr='double'
    if isinstance(n.value,ast.Name) and n.value.id=='torch' and n.attr=='float32':n.attr='float64'
    return n
  tree=Promote().visit(tree)
 ns={'torch':torch,'math':math};exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),ns);return ns
ns=load(R/'source/verl/trainer/core_algos.py');double=load(R/'source/verl/trainer/core_algos.py',True)
torch.manual_seed(17)
q=torch.tensor([[[.1,.2,.3,.4]]]);z0=torch.tensor([[[1.,2.,-1.,0.]]])
def target(z,n,k):
 lp=z.log_softmax(-1);qt=q.to(z).log().requires_grad_()
 if k<0:return n['compute_full_vocab_kl'](lp,qt,kl_direction='jsd_kl').sum(),qt
 ids=lp.topk(k,-1).indices;ti=qt.topk(k,-1).indices
 return n['compute_topk_kl'](lp.gather(-1,ids),lp.exp().gather(-1,ids),ids,qt.gather(-1,ti),ti,kl_direction='jsd_kl').sum(),qt
out={}
for k in [-1,4,2]:
 z=z0.clone().requires_grad_();loss,t=target(z,ns,k);g,tg=torch.autograd.grad(loss,[z,t],allow_unused=True)
 assert tg is None and torch.isfinite(g).all() and g.abs().max()>.01
 assert torch.autograd.gradcheck(lambda x:target(x,double,k)[0],(z0.double().requires_grad_(),),eps=1e-6,atol=1e-6,rtol=1e-4)
 if k in [-1,4]:
  p=z.softmax(-1);m=(p+q)/2;ref=(.5*p*(p.log()-m.log())+.5*q*(q.log()-m.log())).sum()
  rg=torch.autograd.grad(ref,z)[0];torch.testing.assert_close(g,rg,atol=2e-7,rtol=1e-5)
  assert abs(loss.item()-.19254106)<1e-6 and abs(g.abs().max().item()-.06826354)<1e-6
 out[str(k)]={'loss':loss.item(),'max_gradient':g.abs().max().item(),'gradcheck':True,'teacher_grad':None}
# Test the exact FP32 normalization block used in _forward_teacher via a recording module.
tree=ast.parse((R/'source/verl/workers/actor/dp_actor.py').read_text());cl=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DataParallelPPOActor')
method=next(n for n in cl.body if isinstance(n,ast.FunctionDef) and n.name=='_forward_teacher')
ns2={'torch':torch,'batch_collate':lambda xs:{k:[x[k] for x in xs] for k in xs[0]}}
exec(compile(ast.Module(body=[method],type_ignores=[]),'teacher','exec'),ns2)
from types import SimpleNamespace
class Recording:
 def __init__(self,logits):self.logits=logits
 def __call__(self,**kw):self.kw=kw;return SimpleNamespace(logits=self.logits)
precision=[]
for logits in [torch.full((2,132,4),10.,dtype=torch.bfloat16),torch.randn(2,132,64).bfloat16(),(torch.randn(2,132,64)*25).bfloat16()]:
 rec=Recording(logits);self=SimpleNamespace(actor_module=rec)
 ids=torch.zeros(2,2,dtype=torch.long);pos=torch.arange(2)[None].expand(2,-1);resp=torch.zeros(2,130,dtype=torch.long);k=min(4,logits.shape[-1])
 lp,ix=ns2['_forward_teacher'](self,ids,torch.ones_like(ids),pos,resp,1.,k,[{'pixel_values':torch.ones(3,2)},{'pixel_values':torch.ones(5,2)*2}])
 ref=logits[:,-131:-1].float().log_softmax(-1).gather(-1,ix)
 torch.testing.assert_close(lp,ref,atol=4e-6,rtol=1e-6);assert not lp.requires_grad and rec.kw['pixel_values'].shape==(8,2)
 assert lp.exp().sum(-1).max()<=1.00001
 precision.append({'max_error':(lp-ref).abs().max().item(),'max_probability_mass':lp.exp().sum(-1).max().item()})
out['teacher_precision']=precision;out['scope']='Kernel numerical/recording tests; not real model validation'
(R/'evidence/numerics.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))

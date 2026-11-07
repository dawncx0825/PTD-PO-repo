import os
"""Real Qwen3-VL teacher/student image routing and one PTD optimizer update."""
import json,hashlib,copy,gc
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForImageTextToText,AutoProcessor
from verl.models.monkey_patch import apply_ulysses_patch
from verl.utils.dataset import RLHFDataset,process_image
from verl.protocol import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.actor.config import ActorConfig
R=Path(__file__).resolve().parents[1];MODEL=os.environ['PTD_MODEL_PATH']
torch.manual_seed(17);torch.cuda.set_device(0)
processor=AutoProcessor.from_pretrained(MODEL);tok=processor.tokenizer
raw=[json.loads(x) for x in Path(str(Path(os.environ['PTD_ROOT'])/'datasets/train_with_hint.jsonl')).read_text().splitlines()]
rows=[raw[0],raw[8],next(x for x in raw if len(x['images'])>1)]
empty=copy.deepcopy(rows[1]);empty['prompt_with_hint']='';rows.append(empty)
p=R/'evidence/model_test_rows.jsonl';p.write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
ds=RLHFDataset(str(p),tok,processor,prompt_key='problem',prompt_with_hint_key='prompt_with_hint',max_prompt_length=8192,max_hint_prompt_length=4096,min_pixels=200704,max_pixels=1003520,filter_overlong_prompts=False,format_prompt=str(R/'source/examples/format_prompt/math_perception.jinja'))
items=[ds[i] for i in range(len(rows))];mm=[];records=[]
for row,item in zip(rows,items):
 assert 'images' not in item and 'prompt_with_hint' not in item
 images=[process_image(x,200704,1003520) for x in row['images']]
 m={k:v.cuda() for k,v in processor.image_processor(images=images,return_tensors='pt').items()};mm.append(m)
 student_count=int((item['input_ids']==processor.image_token_id).sum());teacher_count=int((item['hint_input_ids']==processor.image_token_id).sum())
 assert student_count==teacher_count==int(m['image_grid_thw'].prod(-1).sum()//4)>0
 records.append({'sample_id':row['sample_id'],'hint_valid':bool(item['hint_valid']),'image_tokens':student_count,'grids':m['image_grid_thw'].tolist(),'pixels_shape':list(m['pixel_values'].shape),'pixels_sha256':hashlib.sha256(m['pixel_values'].cpu().numpy().tobytes()).hexdigest()})
assert not items[-1]['hint_valid'] and items[-1]['hint_attention_mask'].sum()>0
response=tok.encode('Let me inspect the image.',add_special_tokens=False)
responses=torch.tensor([response]*4,device='cuda');T=responses.shape[-1]
b={k:torch.stack([it[k] for it in items]).cuda() for k in ['hint_input_ids','hint_attention_mask','hint_position_ids','hint_valid']}
start=int(b['hint_attention_mask'].long().argmax(-1).min())
for k in ['hint_input_ids','hint_attention_mask','hint_position_ids']:b[k]=b[k][...,start:]
b['responses']=responses;b['ptd_mask']=torch.tensor([1,1,1,0],device='cuda',dtype=torch.bool)
data=DataProto.from_dict(tensors=b,non_tensors={'multi_modal_inputs':np.array(mm,dtype=object)},meta_info={'temperature':1.})
apply_ulysses_patch('qwen3_vl')
model=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda().eval();model.requires_grad_(False)
cfg=ActorConfig(use_torch_compile=False,padding_free=True);cfg.ptd_top_k=100;cfg.micro_batch_size_per_device_for_experience=2
actor=DataParallelPPOActor(cfg,model)
recorded=[]
def hook(mod,args,kw):
 assert 'pixel_values' in kw and 'image_grid_thw' in kw
 recorded.append({'pixels':list(kw['pixel_values'].shape),'grids':kw['image_grid_thw'].tolist(),'tokens':int((kw['input_ids']==processor.image_token_id).sum())})
h=model.register_forward_pre_hook(hook,with_kwargs=True)
torch.cuda.reset_peak_memory_stats();out=actor.compute_teacher_log_prob(data);peak=torch.cuda.max_memory_allocated()
cfg.micro_batch_size_per_device_for_experience=1;one=actor.compute_teacher_log_prob(data)
torch.testing.assert_close(out.batch['teacher_topk_log_probs'],one.batch['teacher_topk_log_probs'],atol=.15,rtol=.03)
# Reordering includes a multi-image sample and preserves pixel ownership.
order=torch.tensor([2,0,3,1],device='cuda');reordered=actor.compute_teacher_log_prob(data[order]);inverse=order.argsort()
torch.testing.assert_close(one.batch['teacher_topk_log_probs'],reordered.batch['teacher_topk_log_probs'][inverse],atol=.02,rtol=.01)
# Independent direct forward with the exact same ids/masks/positions/images.
i=0;ids=b['hint_input_ids'][i:i+1];attn=b['hint_attention_mask'][i:i+1];pos=b['hint_position_ids'][i:i+1];resp=responses[i:i+1]
full_ids=torch.cat([ids,resp],-1);full_attn=torch.cat([attn,torch.ones_like(resp)],-1);rp=pos.max(-1,keepdim=True).values+torch.arange(1,T+1,device='cuda')[None,None,:];full_pos=torch.cat([pos,rp],-1).transpose(0,1)
with torch.no_grad():
 direct=model(input_ids=full_ids,attention_mask=full_attn,position_ids=full_pos,**mm[i],use_cache=False).logits[:,-T-1:-1].float().log_softmax(-1)
 lp,ti=actor._forward_teacher(ids,attn,pos,resp,1.,100,[mm[i]])
 torch.testing.assert_close(lp,direct.gather(-1,ti),atol=2e-5,rtol=2e-6)
 token=actor._forward_teacher(ids,attn,pos,resp,1.,0,[mm[i]])
 full=actor._forward_teacher(ids,attn,pos,resp,1.,-1,[mm[i]])
 torch.testing.assert_close(full,direct,atol=2e-5,rtol=2e-6)
 erased=actor._forward_teacher(ids,attn,pos,resp,1.,0,[{**mm[i],'pixel_values':torch.zeros_like(mm[i]['pixel_values'])}])
 sensitivity=(token-erased).abs().max().item();assert sensitivity>1e-4
assert all(p.grad is None and not p.requires_grad for p in model.parameters());h.remove()
# Verify a student update from the original weights; this isolated model is never used for formal training.
student=AutoModelForImageTextToText.from_pretrained(MODEL,dtype=torch.bfloat16,attn_implementation='flash_attention_2').cuda().train();student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
scfg=ActorConfig(use_torch_compile=False,padding_free=True);scfg.ptd_top_k=100;scfg.ptd_kl_direction='jsd_kl'
sa=DataParallelPPOActor(scfg,student);it=items[0];a={k:it[k][None].cuda() for k in ['input_ids','attention_mask','position_ids']};st=int(a['attention_mask'].long().argmax(-1).min())
for k in a:a[k]=a[k][...,st:]
pos=a['position_ids'];a['position_ids']=torch.cat([pos,pos.max(-1,keepdim=True).values+torch.arange(1,T+1,device='cuda')[None,None,:]],-1);a['input_ids']=torch.cat([a['input_ids'],resp],-1);a['attention_mask']=torch.cat([a['attention_mask'],torch.ones_like(resp)],-1);a.update(responses=resp,response_mask=torch.ones_like(resp),multi_modal_inputs=[mm[0]],teacher_topk_log_probs=lp,teacher_topk_ids=ti,hint_valid=torch.ones(1,dtype=torch.bool,device='cuda'))
slp,sklp,skp,ski=sa._forward_micro_batch(a,1.,100);loss=sa._compute_ptd_loss(a,torch.ones(1,dtype=torch.bool,device='cuda'),slp,(sklp,skp,ski),response_mask=a['response_mask'])/T
assert loss.isfinite();loss.backward();vision=student.model.visual.patch_embed.proj.weight;assert vision.grad is not None and vision.grad.abs().max()>0 and torch.isfinite(vision.grad).all()
grad=vision.grad.abs().max().item();before=vision.detach().clone();teacher_before=model.model.visual.patch_embed.proj.weight.detach().clone()
# Diagnostic optimizer has a deliberately visible update; formal optimizer remains adamw_bf16.
optimizer=torch.optim.SGD(student.parameters(),lr=.01);optimizer.step();changed=not torch.equal(before,vision)
assert changed and torch.equal(teacher_before,model.model.visual.patch_embed.proj.weight)
assert all(p.grad is None for p in model.parameters())
out={'status':'PASS','scope':'isolated real-model mechanism smoke; not training effectiveness','records':records,'teacher_calls':recorded,'teacher_peak_allocated_bytes':peak,'direct_max_error':(lp-direct.gather(-1,ti)).abs().max().item(),'image_sensitivity_max':sensitivity,'ptd_loss':loss.item(),'vision_gradient_max':grad,'student_parameters_changed':changed,'teacher_frozen':True,'optimizer_updates':1,'diagnostic_optimizer':'SGD lr=0.01; formal run reinitializes adamw_bf16'}
(R/'evidence/model_preflight.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)

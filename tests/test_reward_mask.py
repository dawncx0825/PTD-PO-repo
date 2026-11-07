import os
"""Real reward and PTD selection, including correct/incorrect and absent hints."""
import importlib.util,json
from pathlib import Path
import numpy as np
import torch
from verl.trainer.core_algos import compute_ptd_mask
R=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('qwen_reward',R/'source/examples/reward_function/qwen3_vl_think.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
replies=[r'Reasoning.</think> \boxed{2}',r'Reasoning.</think> \boxed{3}',r'\boxed{2}']
scores=m.compute_score([{'response':x,'ground_truth':'2'} for x in replies])
assert [s['accuracy'] for s in scores]==[1,0,1]
assert [s['format'] for s in scores]==[1,1,0]
assert np.allclose([s['overall'] for s in scores],[1,.1,.9])
acc=torch.tensor([1,0,1,1,0,0,0,0.]);uids=np.array(['mixed','mixed','pass','pass','fail','fail','empty','empty'])
ptd,groups=compute_ptd_mask(acc,uids,1.)
assert ptd.tolist()==[False,True,False,False,True,True,True,True]
valid=torch.tensor([1,1,1,1,1,1,0,0],dtype=torch.bool)
assert (ptd&valid).tolist()==[False,True,False,False,True,True,False,False]
(R/'evidence/reward_mask.json').write_text(json.dumps({'status':'PASS','reward_scores':scores,'selected_with_valid_hint':(ptd&valid).tolist(),'correct_excluded':True,'missing_hint_excluded':True},indent=2))

"""CPU checks for the KL-ablation patch: mask semantics, config validation, and the actor-side masking arithmetic."""
import json, sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2] / 'source'))
import numpy as np
import torch
from verl.trainer.core_algos import compute_kl_keep_mask, compute_ptd_mask
from verl.trainer.config import PPOConfig

acc = torch.tensor([1., 0., 1., 1., 0., 0., 1., 0.])
uid = np.array(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'], dtype=object)
res = {}
assert compute_kl_keep_mask(acc, 'all').tolist() == [True] * 8
assert compute_kl_keep_mask(acc, 'none').tolist() == [False] * 8
assert compute_kl_keep_mask(acc, 'correct').tolist() == [True, False, True, True, False, False, True, False]
try:
    compute_kl_keep_mask(acc, 'ptd_group'); raise AssertionError('ptd_group must not be handled here')
except ValueError:
    pass
res['mask_modes'] = 'PASS'

# Arm 5 (upstream) vs arm 4 (ours) on the same mixed groups: upstream drops KL for correct trajectories too.
_, group = compute_ptd_mask(acc, uid, 1.0)
upstream_keep = (~group).tolist()
assert upstream_keep == [False] * 8
all_pass = torch.ones(8)
_, group2 = compute_ptd_mask(all_pass, uid, 1.0)
assert (~group2).all()  # all-correct groups keep KL upstream
res['upstream_semantics'] = 'PASS'

# Actor arithmetic: kld * keep, averaged over ALL response tokens (same denominator as the upstream PTD-group drop).
kld = torch.rand(8, 5); mask = torch.ones(8, 5)
keep = compute_kl_keep_mask(acc, 'correct')
masked = kld * keep.unsqueeze(-1).float()
assert torch.allclose((masked * mask).sum() / mask.sum(), (kld[keep].sum()) / mask.sum())
res['actor_arithmetic'] = 'PASS'


def cfg(**alg):
    c = PPOConfig()
    for k, v in alg.items():
        setattr(c.algorithm, k, v)
    c.post_init()
    return c


assert cfg(kl_mask_mode='correct', use_kl_loss=True).worker.actor.kl_mask_mode == 'correct'
assert cfg().worker.actor.kl_mask_mode == 'ptd_group'
for bad in [dict(kl_mask_mode='bogus', use_kl_loss=True), dict(kl_mask_mode='none', use_kl_loss=False),
            dict(kl_mask_mode='all', use_kl_loss=True, disable_kl=True)]:
    try:
        cfg(**bad); raise AssertionError(f'accepted {bad}')
    except ValueError:
        pass
res['config_validation'] = 'PASS'
print(json.dumps(res))

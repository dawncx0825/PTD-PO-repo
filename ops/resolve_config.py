import os
import json,os,shlex,subprocess
from pathlib import Path
from omegaconf import OmegaConf
from verl.trainer.config import PPOConfig
R=Path(str(Path(os.environ['PTD_ROOT'])))
# Capture the launcher arguments without importing/running trainer.main.
script=(R/'launch.sh').read_text();script=script.replace('python3 -m verl.trainer.main',"python3 -c 'import sys,json;print(json.dumps(sys.argv[1:]))'")
a=json.loads(subprocess.check_output(['bash','-c',script],text=True).splitlines()[-1]);cli=OmegaConf.from_dotlist(a);file=cli.pop('config');os.chdir(R/'source');cfg=OmegaConf.merge(OmegaConf.structured(PPOConfig()),OmegaConf.load(file),cli);obj=OmegaConf.to_object(cfg);obj.deep_post_init()
assert obj.trainer.nnodes * obj.trainer.n_gpus_per_node == 8
(R/'resolved_config.yaml').write_text(OmegaConf.to_yaml(OmegaConf.create(obj.to_dict())));(R/'config/command_args.json').write_text(json.dumps(a,indent=2))
n=json.loads((R/'length_audit.json').read_text())['after_student_filter'];mini=n//128;steps=(mini*128//384)*2
assert steps==198
(R/'evidence/expected_training.json').write_text(json.dumps({'admitted_rows':n,'mini_rollout_batch':128,'dataloader_drop_last':True,'mini_batches_per_dataset_pass':mini,'rollout_batch':384,'epochs':2,'rollout_steps':steps,'rollout_n':8,'effective_actor_global_batch_after_n_expansion':128*8,'optimizer_updates_per_rollout':3,'expected_optimizer_updates':steps*3,'sampled_prompt_count':steps*384,'sampled_trajectory_count':steps*384*8,'formal_initialization':'original shared model; no smoke/resume checkpoint'},indent=2))
print('resolved',steps)

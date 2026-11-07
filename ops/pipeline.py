import os
"""Serial eight-H100 experiment; each stage is gated by verified artifacts."""
from pathlib import Path
import os,sys,json,time,subprocess,hashlib,fcntl
R=Path(str(Path(os.environ['PTD_ROOT'])))
lock=(R/'state/pipeline.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def stage(name,cmd):
 (R/'state/pipeline.json').write_text(json.dumps({'stage':name,'status':'RUNNING','started_epoch':time.time()}))
 with (R/'logs'/f'{name}.log').open('a') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 (R/'state'/f'{name}.exit').write_text(str(p.returncode))
 if p.returncode:raise RuntimeError(f'{name} failed: {p.returncode}')
try:
 result=json.loads((R/'test_results.json').read_text());assert result['status']=='PASS'
 for rel,sha in result['source_sha256'].items():assert hashlib.sha256((R/'source'/rel).read_bytes()).hexdigest()==sha,rel
 if hashlib.sha256((R/'launch.sh').read_bytes()).hexdigest()!=result['launch_sha256']:
  # User explicitly restored the upstream checkpoint policy on 2026-11-05.
  # Retain the original acceptance hashes and permit only this exact launcher delta.
  before=(R/'evidence/checkpoint_frequency_20261105/launch.before.sh').read_bytes()
  assert hashlib.sha256(before).hexdigest()==result['launch_sha256']
  expected=before.decode().replace('    trainer.save_checkpoint_path=${SAVE_PATH} \\\n',
   '    trainer.save_checkpoint_path=${SAVE_PATH} \\\n    trainer.save_freq=40 \\\n    trainer.save_limit=5 \\\n    trainer.save_model_only=false \\\n')
  expected=expected.replace('trainer.find_last_checkpoint=false','trainer.find_last_checkpoint=true')
  assert (R/'launch.sh').read_text()==expected,'Unaccepted launcher change'
 protocol=json.loads((R/'evaluation_protocol.json').read_text());assert protocol['status']=='PASS'
 for rel,sha in protocol['source_hashes'].items():assert hashlib.sha256((R/rel).read_bytes()).hexdigest()==sha,rel
 for name in ['model_preflight','distributed_gpu','checkpoint_resume']:
  assert json.loads((R/f'evidence/{name}.json').read_text())['status']=='PASS'
 assert (R/'state/smoke.exit').read_text().strip()=='0'
 assert json.loads((R/'length_audit.json').read_text())['teacher_over4096']==0
 assert json.loads((R/'data_manifest.json').read_text())['image_hash_mismatch']==0
 assert not list((R/'checkpoints').glob('global_step_*')),'Formal training must initialize from original weights'
 stage('initial_eval',[sys.executable,'-u',str(R/'ops/run_eval.py'),'--label','initial'])
 stage('train',['bash',str(R/'launch.sh')])
 stage('verify_training',[sys.executable,str(R/'ops/verify_training.py')])
 stage('export',[sys.executable,str(R/'ops/export_checkpoint.py')])
 stage('final_eval',[sys.executable,'-u',str(R/'ops/run_eval.py'),'--label','final'])
 stage('report',[sys.executable,str(R/'ops/report.py')])
 (R/'state/pipeline.json').write_text(json.dumps({'status':'COMPLETE','finished_epoch':time.time()}))
except BaseException as e:
 error={'error':str(e),'epoch':time.time()}
 (R/'state/pipeline_error.json').write_text(json.dumps(error))
 current=json.loads((R/'state/pipeline.json').read_text()) if (R/'state/pipeline.json').exists() else {}
 (R/'state/pipeline.json').write_text(json.dumps({**current,**error,'status':'FAILED'}))
 raise

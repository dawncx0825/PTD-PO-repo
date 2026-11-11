"""In-pod runner for the KL ablation on one 8xA100 node.

TS_MODE=smoke: one tiny training step per arm (separate smoke_* dirs) and check the KL-mask metrics.
TS_MODE=full : arms in TS_ARMS (comma list; default ops/arms.json order). Each arm is claimed atomically by
               TS_JOB (arms/<arm>/state/owner), so two Jobs never run the same arm; per arm train (resume from the last checkpoint on failure,
               at most MAX_TRAIN_ATTEMPTS) -> verify step 198 -> export -> nine-task eval. A failed arm is
               recorded in arms/<arm>/state/FAILED and the queue moves on to the next arm.
State lives in state/queue.json and arms/<arm>/state/*; nothing here deletes checkpoints.
"""
import json, os, shutil, subprocess, sys, time, traceback
from pathlib import Path

R = Path(os.environ['PTD_ROOT'])  # repository root with ablation/patch_kl_ablation.diff applied
OPS = Path(__file__).resolve().parent
MODEL = os.environ['PTD_MODEL_PATH']
EXPECTED_STEPS = 198
MAX_TRAIN_ATTEMPTS = 3
ARMS = json.loads((OPS / 'arms.json').read_text())
PY = str(R / 'venv/bin/python')
JOB_NAME = os.environ.get('TS_JOB', 'unnamed')
MY_ARMS = [a for a in os.environ.get('TS_ARMS', '').split(',') if a] or ARMS['order']
assert all(a in ARMS['arms'] for a in MY_ARMS), MY_ARMS


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S %z')


def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    t = p.with_suffix(p.suffix + '.tmp')
    t.write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=1, ensure_ascii=False))
    t.replace(p)


def queue_state(**kw):
    p = R / f'state/queue_{JOB_NAME}.json'
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw, updated=now(), node=os.environ.get('NODE_NAME'))
    write(p, cur)


def env_for(arm_dir_name, spec):
    env = dict(os.environ)
    env.update(
        PATH=f"{R}/venv/bin:" + env.get('PATH', ''), PYTHONPATH=f"{R}/source:{R}/papo_eval",
        PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', TOKENIZERS_PARALLELISM='false',
        VLLM_NO_USAGE_STATS='1', OMP_NUM_THREADS='4', XDG_CACHE_HOME=f'{R}/runtime-cache',
        HF_HOME=f'{R}/runtime-cache/huggingface', TRITON_CACHE_DIR=f'{R}/runtime-cache/triton',
        TORCHINDUCTOR_CACHE_DIR=f'{R}/runtime-cache/torchinductor',
        PTD_ROOT=str(R), PTD_MODEL_PATH=MODEL, RESOURCE_GPU='8', ARM=arm_dir_name,
        ARM_ENABLE_PTD=str(spec['enable_ptd']).lower(), ARM_KL_MASK_MODE=spec['kl_mask_mode'])
    return env


def run(cmd, log, env):
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('a') as f:
        f.write(f'\n===== {now()} {cmd}\n'); f.flush()
        return subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, cwd=str(R)).returncode


def ray_stop(env):
    subprocess.run([str(R / 'venv/bin/ray'), 'stop', '--force'], env=env, capture_output=True)
    time.sleep(10)


def read_metrics(save_path):
    p = save_path / 'experiment_log.jsonl'
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def smoke():
    out = {}
    for name in ARMS['order']:
        spec = ARMS['arms'][name]
        d = 'smoke_' + name
        env = env_for(d, spec)
        save = R / 'arms' / d / 'checkpoints'
        if save.exists():
            shutil.move(str(save), str(save) + f'.old{int(time.time())}')
        queue_state(mode='smoke', arm=name, stage='train', status='RUNNING')
        rc = run(['bash', str(OPS / 'launch_arm.sh'), 'trainer.max_steps=1', 'trainer.save_freq=-1',
                  'trainer.val_freq=-1', 'trainer.find_last_checkpoint=false', 'data.rollout_batch_size=16',
                  'data.mini_rollout_batch_size=16', 'worker.actor.global_batch_size=16'],
                 R / 'arms' / d / 'logs/train.log', env)
        ray_stop(env)
        m = [row for row in read_metrics(save) if 'actor' in row]
        rec = {'rc': rc, 'steps_logged': len(m), 'raw_last': m[-1] if m else None}
        checks = {'exit0': rc == 0, 'one_step': len(m) >= 1}
        if m:
            last = m[-1]
            flat = {}
            for k, v in last.items():
                if isinstance(v, dict):
                    flat.update({f'{k}/{kk}': vv for kk, vv in v.items()})
            keep = flat.get('kl_ablation/keep_ratio')
            acc = flat.get('reward/accuracy')
            rec.update(keep_ratio=keep, accuracy=acc, kl_loss=flat.get('actor/kl_loss'),
                       kl_unmasked=flat.get('actor/kl_unmasked'), ptd_kl=flat.get('actor/ptd_kl'))
            checks['kl_unmasked_logged'] = flat.get('actor/kl_unmasked') is not None
            if spec['kl_mask_mode'] == 'all':
                checks['keep_ratio'] = keep == 1.0
            elif spec['kl_mask_mode'] == 'none':
                checks['keep_ratio'] = keep == 0.0
                checks['kl_loss_zero'] = flat.get('actor/kl_loss') == 0.0
            else:  # correct: keep ratio equals the batch accuracy
                checks['keep_ratio'] = keep is not None and acc is not None and abs(keep - acc) < 1e-6
            checks['ptd_matches_arm'] = (flat.get('actor/ptd_kl') is not None) == spec['enable_ptd']
        rec['checks'] = checks
        rec['pass'] = all(checks.values())
        out[name] = rec
        write(R / 'evidence/smoke.json', out)
    ok = all(v['pass'] for v in out.values())
    queue_state(mode='smoke', stage='done', status='PASS' if ok else 'FAIL')
    return 0 if ok else 1


def archive_logs(save, attempt, adir):
    dst = adir / 'logs' / f'trainer_logs_before_attempt{attempt}'
    for f in ('experiment_log.jsonl', 'generations.log', 'experiment_config.json'):
        if (save / f).exists():
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(save / f, dst / f)


def final_ckpt_ok(save):
    actor = save / f'global_step_{EXPECTED_STEPS}' / 'actor'
    return all(len(list(actor.glob(f'{k}_world_size_8_rank_*.pt'))) == 8 for k in ('model', 'optim', 'extra_state')) \
        and (actor.parent / 'dataloader.pt').exists()


def run_arm(name):
    spec = ARMS['arms'][name]
    adir = R / 'arms' / name
    st = adir / 'state'
    st.mkdir(parents=True, exist_ok=True)
    env = env_for(name, spec)
    save = adir / 'checkpoints'

    if not (st / 'train.done').exists():
        attempts = len(list(st.glob('train_attempt*.exit')))
        while not final_ckpt_ok(save):
            attempts += 1
            if attempts > MAX_TRAIN_ATTEMPTS:
                raise RuntimeError(f'train failed after {MAX_TRAIN_ATTEMPTS} attempts')
            archive_logs(save, attempts, adir)
            queue_state(mode='full', arm=name, stage='train', attempt=attempts, status='RUNNING')
            write(st / 'train.json', {'attempt': attempts, 'started': now()})
            rc = run(['bash', str(OPS / 'launch_arm.sh')], adir / f'logs/train_attempt{attempts}.log', env)
            write(st / f'train_attempt{attempts}.exit', str(rc))
            ray_stop(env)
        write(st / 'train.done', now())

    if not (adir / 'evidence/final_checkpoint.json').exists():
        queue_state(mode='full', arm=name, stage='export', status='RUNNING')
        rc = run([PY, str(OPS / 'export_checkpoint.py'), name], adir / 'logs/export.log', env)
        if rc:
            raise RuntimeError(f'export exit {rc}')

    if not (R / 'results' / name / 'summary.json').exists():
        queue_state(mode='full', arm=name, stage='eval', status='RUNNING')
        rc = run([PY, '-u', str(OPS / 'run_eval.py'), '--label', name], adir / 'logs/eval.log', env)
        if rc:
            raise RuntimeError(f'eval exit {rc}')
    s = json.loads((R / 'results' / name / 'summary.json').read_text())
    write(st / 'COMPLETE', {'finished': now(), 'overall_macro_avg': s['overall_macro_avg'],
                            'general_avg': s['general_avg'], 'vision_avg': s['vision_avg']})


def claim(name):
    st = R / 'arms' / name / 'state'
    st.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(st / 'owner', os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, JOB_NAME.encode()); os.close(fd)
        return True
    except FileExistsError:
        return (st / 'owner').read_text().strip() == JOB_NAME


def full():
    for name in MY_ARMS:
        st = R / 'arms' / name / 'state'
        if (st / 'COMPLETE').exists() or (st / 'FAILED').exists():
            continue
        if not claim(name):
            continue
        try:
            run_arm(name)
        except Exception as e:
            write(st / 'FAILED', {'time': now(), 'error': str(e), 'trace': traceback.format_exc()[-4000:]})
    queue_state(mode='full', arms=MY_ARMS, stage='all_arms_done', status='COMPLETE')
    return 0


if __name__ == '__main__':
    (R / 'evidence').mkdir(exist_ok=True)
    write(R / f"evidence/gpu_inventory_{int(time.time())}.txt", subprocess.check_output(
        ['nvidia-smi', '--query-gpu=index,name,uuid,memory.total,driver_version', '--format=csv'], text=True))
    sys.exit(smoke() if os.environ.get('TS_MODE') == 'smoke' else full())

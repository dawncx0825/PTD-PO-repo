import os
"""Audited save-frequency-only continuation; never starts or stops GPU work."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

R = Path(str(Path(os.environ['PTD_ROOT'])))
A = R / 'evidence/checkpoint_frequency_20261105'
STEP = 40
FREQUENCY = 5


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
    tmp.replace(path)


def finite(obj):
    if isinstance(obj, dict):
        return all(finite(v) for v in obj.values())
    if isinstance(obj, list):
        return all(finite(v) for v in obj)
    return not isinstance(obj, (int, float)) or math.isfinite(obj)


def read_training(path):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines()]
    train = [r for r in rows if 'optimizer_updates' in r.get('actor', {})]
    assert all(finite(r) and r['actor']['optimizer_updates'] == 3 for r in train)
    return rows, train


def resolved_launcher(path, extra=()):
    from omegaconf import OmegaConf
    from verl.trainer.config import PPOConfig
    script = Path(path).read_text().replace(
        'python3 -m verl.trainer.main',
        "python3 -c 'import sys,json;print(json.dumps(sys.argv[1:]))'")
    args = json.loads(subprocess.check_output(['bash', '-c', script, 'capture', *extra], text=True).splitlines()[-1])
    cli = OmegaConf.from_dotlist(args)
    config_file = cli.pop('config')
    os.chdir(R / 'source')
    config = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(PPOConfig()), OmegaConf.load(config_file), cli))
    config.deep_post_init()
    return config.to_dict()


def verify_change():
    receipt = json.loads((A / 'change.json').read_text())
    assert receipt['save_freq'] == FREQUENCY
    assert sha(R / 'launch.sh') == receipt['new_launch_sha256']
    assert sha(A / 'launch.before.sh') == receipt['old_launch_sha256']
    tests = json.loads((R / 'test_results.json').read_text())
    assert tests['status'] == 'PASS'
    assert tests['launch_sha256'] == receipt['old_launch_sha256']
    for rel, digest in tests['source_sha256'].items():
        assert sha(R / 'source' / rel) == digest, rel
    old = resolved_launcher(A / 'launch.before.sh')
    new = resolved_launcher(R / 'launch.sh')
    expected = copy.deepcopy(old)
    assert expected['trainer']['save_freq'] == 40
    expected['trainer']['save_freq'] = FREQUENCY
    assert new == expected, 'Other training configuration changed'
    return old, new


def prepare_resume():
    import torch
    assert not (R / 'state/resume_checkpoint_frequency.json').exists()
    old, new = verify_change()
    current = json.loads((R / 'checkpoints/experiment_config.json').read_text())
    # Trainer sets the scheduler horizon during initialization, before Tracker writes config.
    expected_runtime = copy.deepcopy(old)
    expected_runtime['worker']['actor']['optim']['training_steps'] = 198
    expected_runtime['worker']['critic']['optim']['training_steps'] = 198
    assert current == expected_runtime, 'Runtime differs from accepted launcher'
    assert not (R / 'state/pipeline_error.json').exists()
    tracker = json.loads((R / 'checkpoints/checkpoint_tracker.json').read_text())
    assert tracker['last_global_step'] == STEP
    cp = R / f'checkpoints/global_step_{STEP}'
    assert tracker['last_actor_path'] == str(cp / 'actor')
    rows, train = read_training(R / 'checkpoints/experiment_log.jsonl')
    assert [r['step'] for r in train] == list(range(1, len(train) + 1))
    assert len(train) >= STEP
    files = [cp / 'dataloader.pt']
    for rank in range(8):
        files += [cp / 'actor' / f'{kind}_world_size_8_rank_{rank}.pt' for kind in ('model', 'optim', 'extra_state')]
        extra = torch.load(files[-1], weights_only=False, map_location='cpu')
        assert extra['lr_scheduler']['last_epoch'] == STEP
        assert extra['lr_scheduler']['_last_lr'] == [1e-6]
        assert set(extra['rng']) == {'cpu', 'cuda', 'numpy', 'random'}
        assert all(extra['rng'][key].numel() > 0 for key in ('cpu', 'cuda'))
    data = torch.load(cp / 'dataloader.pt', weights_only=False, map_location='cpu')
    snapshot = data['_snapshot']
    assert snapshot['_snapshot_step'] + data['_steps_since_snapshot'] == STEP * 3
    assert not data['_iterator_finished']
    manifest = {}
    for path in files:
        assert path.stat().st_size > 0
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None, str(path)
        manifest[str(path.relative_to(R))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    for path in (cp / 'actor/huggingface').iterdir():
        if path.is_file():
            manifest[str(path.relative_to(R))] = {'bytes': path.stat().st_size, 'sha256': sha(path)}
    assert (cp / 'actor/huggingface/config.json').exists()
    # Preserve the original segment before any logger can open it with mode='w'.
    for rel in ('checkpoints/experiment_log.jsonl', 'checkpoints/experiment_config.json',
                'logs/train.log', 'state/pipeline.json', 'test_results.json', 'resolved_config.yaml'):
        dest = A / 'before_resume' / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(R / rel, dest)
    archived_rows, archived_train = read_training(A / 'before_resume/checkpoints/experiment_log.jsonl')
    prefix = [r for r in archived_rows if r['step'] <= STEP]
    (A / 'prefix.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in prefix))
    prefix_train = [r for r in prefix if 'optimizer_updates' in r.get('actor', {})]
    assert [r['step'] for r in prefix_train] == list(range(1, STEP + 1))
    request = {'status': 'VERIFIED', 'step': STEP, 'save_freq': FREQUENCY,
               'checkpoint': str(cp), 'checkpoint_files': manifest,
               'prefix_sha256': sha(A / 'prefix.jsonl'), 'archived_last_step': archived_train[-1]['step'],
               'scheduler_last_epoch': STEP, 'dataloader_yielded_minibatches': STEP * 3,
               'rng_ranks': 8, 'epoch': time.time()}
    atomic_json(A / 'resume_verified.json', request)
    atomic_json(R / 'state/resume_checkpoint_frequency.json', request)
    print('Full checkpoint and continuation verified', flush=True)


def validate_resume(request):
    verify_change()
    assert request['status'] == 'VERIFIED' and request['step'] == STEP
    assert request['save_freq'] == FREQUENCY
    assert request['checkpoint'] == str(R / f'checkpoints/global_step_{STEP}')
    assert sha(A / 'prefix.jsonl') == request['prefix_sha256']
    for rel, item in request['checkpoint_files'].items():
        assert (R / rel).stat().st_size == item['bytes'], rel
        assert sha(R / rel) == item['sha256'], rel
    runtime = resolved_launcher(R / 'launch.sh', [f'trainer.load_checkpoint_path={request["checkpoint"]}'])
    _, expected = verify_change()
    expected['trainer']['load_checkpoint_path'] = request['checkpoint']
    assert runtime == expected, 'Resume changes more than checkpoint path'
    atomic_json(A / 'resume_config.json', runtime)


def merge_logs():
    prefix_rows, prefix = read_training(A / 'prefix.jsonl')
    path = R / 'checkpoints/experiment_log.jsonl'
    rows, suffix = read_training(path)
    # Safe if initialization failed before the upstream logger replaced the file.
    if suffix and suffix[0]['step'] == 1:
        assert [r['step'] for r in suffix] == list(range(1, len(suffix) + 1))
        return
    assert all(r['step'] > STEP for r in suffix)
    assert [r['step'] for r in suffix] == list(range(STEP + 1, STEP + 1 + len(suffix)))
    shutil.copy2(path, A / 'resumed_segment.jsonl')
    # A validation-only record at the restore boundary is not a training update.
    merged = prefix_rows + rows
    tmp = path.with_suffix('.merged.tmp')
    tmp.write_text(''.join(json.dumps(r) + '\n' for r in merged))
    tmp.replace(path)
    atomic_json(A / 'log_merge.json', {'prefix_steps': len(prefix), 'resumed_steps': len(suffix),
                                     'optimizer_updates': (len(prefix) + len(suffix)) * 3,
                                     'sha256': sha(path), 'epoch': time.time()})


if __name__ == '__main__':
    raise SystemExit('CANCELLED: historical helper only; never execute the checkpoint-frequency switch')
    try:
        if sys.argv[1] == 'prepare':
            prepare_resume()
        elif sys.argv[1] == 'verify-change':
            verify_change()
            print('Only trainer.save_freq changed: 40 -> 5')
        else:
            raise ValueError('Unknown action')
    except BaseException as exc:
        atomic_json(A / 'error.json', {'error': str(exc), 'epoch': time.time()})
        raise

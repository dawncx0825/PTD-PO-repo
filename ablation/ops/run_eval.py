"""Eight evaluation shards (one per A100) for one ablation arm. Copy of h100-repro ops/run_eval.py with the
model taken from the arm's export record; protocol unchanged."""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ['PTD_ROOT'])
OPS = Path(__file__).resolve().parent


def main(label):
    model = json.loads((ROOT / 'arms' / label / 'evidence/final_checkpoint.json').read_text())['hf_path']
    started = time.time()
    commands = [[
        sys.executable, '-u', str(OPS / 'eval_worker.py'),
        '--model', model, '--label', label, '--rank', str(rank),
    ] for rank in range(8)]

    if os.environ.get('RAY_ADDRESS'):
        import ray
        from cluster_process import collect_jobs, run_process
        ray.init(address=os.environ['RAY_ADDRESS'])
        refs = [run_process.options(num_gpus=1, num_cpus=2).remote(
            command, f'eval_{label}_{rank}.log', timeout=48 * 3600,
            extra_env={'VLLM_ENABLE_V1_MULTIPROCESSING': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn'},
        ) for rank, command in enumerate(commands)]
        placements = collect_jobs(refs)
        (ROOT / 'evidence' / f'eval_{label}_placements.json').write_text(json.dumps(placements, indent=2))
        ray.shutdown()
    else:
        workers = []
        try:
            for rank, command in enumerate(commands):
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(rank), VLLM_ENABLE_V1_MULTIPROCESSING='1')
                with (ROOT / 'logs' / f'eval_{label}_{rank}.log').open('a') as log:
                    workers.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
            while any(worker.poll() is None for worker in workers):
                if any(worker.poll() not in (None, 0) for worker in workers):
                    raise RuntimeError('Evaluation worker failed; completed question files are preserved')
                time.sleep(10)
            assert all(worker.returncode == 0 for worker in workers)
        finally:
            for worker in workers:
                if worker.poll() is None:
                    os.killpg(worker.pid, signal.SIGTERM)
            for worker in workers:
                try:
                    worker.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(worker.pid, signal.SIGKILL)
                    worker.wait()

    subprocess.run([sys.executable, str(OPS / 'score_eval.py'), '--label', label], check=True)
    (ROOT / 'evidence' / f'eval_{label}_timing.json').write_text(json.dumps({
        'elapsed_s': time.time() - started,
        'started_epoch': started,
        'finished_epoch': time.time(),
    }, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True, help='arm name from ops/arms.json')
    main(parser.parse_args().label)

import os
"""Run task-owned subprocesses with Ray GPU placement and deterministic cleanup."""
import os
import signal
import subprocess
import time
from pathlib import Path

import ray

ROOT = Path(str(Path(os.environ['PTD_ROOT'])))


@ray.remote(max_calls=1)
def run_process(command, log_name, timeout=3600, extra_env=None):
    env = dict(os.environ)
    if extra_env:
        env.update(extra_env)
    started = time.time()
    process = None
    with (ROOT / 'logs' / log_name).open('a') as log:
        try:
            process = subprocess.Popen(
                command, env=env, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            code = process.wait(timeout=timeout)
        finally:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
    return {
        'exit_code': code,
        'elapsed_s': time.time() - started,
        'node_ip': ray.util.get_node_ip_address(),
        'cuda_visible_devices': env.get('CUDA_VISIBLE_DEVICES'),
        'log': str(ROOT / 'logs' / log_name),
    }


def collect_jobs(refs):
    pending = list(refs)
    results = []
    try:
        while pending:
            ready, pending = ray.wait(pending, num_returns=1, timeout=30)
            for ref in ready:
                result = ray.get(ref)
                results.append(result)
                if result['exit_code']:
                    raise RuntimeError(f"Subprocess failed: {result}")
    finally:
        for ref in pending:
            ray.cancel(ref, force=False)
        if pending:
            ray.wait(pending, num_returns=len(pending), timeout=45)
    return results

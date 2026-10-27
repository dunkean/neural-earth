"""Parallel map over short-lived Python worker processes for the resource compiler.

The heavy parts of the world authority (Render v2 environment per month,
realized geology per chunk of cells) are pure NumPy functions of their
arguments. Threads barely help them (the GIL is held between small array
operations), so they run in worker processes started with this file as the
entry point: no import of the server's __main__, no shared state, and the
same NumPy code, hence bit-identical results in any order.

Each task is (module, function, kwargs); arguments and results travel as
pickles in temporary files. Workers that fail raise in the parent.
"""
from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent


def default_workers():
    return max(1, min(8, (os.cpu_count() or 2) - 2))


def parallel_map(tasks, workers=None, timeout=600):
    """Run [(module, function, kwargs), ...] in worker processes; results in task order."""
    workers = workers or default_workers()
    if workers <= 1 or len(tasks) <= 1:
        return [_call(*task) for task in tasks]
    with tempfile.TemporaryDirectory(prefix='terrain-resources-') as tmp:
        tmp = Path(tmp)
        paths = []
        for n, task in enumerate(tasks):
            path = tmp / f'task{n}.pkl'
            path.write_bytes(pickle.dumps(task, protocol=5))
            paths.append((path, tmp / f'result{n}.pkl'))
        pending = list(range(len(tasks)))
        running = {}
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
        while pending or running:
            while pending and len(running) < workers:
                n = pending.pop(0)
                running[n] = subprocess.Popen([sys.executable, str(HERE / 'terrain_resource_workers.py'), str(paths[n][0]),
                                               str(paths[n][1])], env=env, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.PIPE)
            n, proc = next(iter(running.items()))
            _, err = proc.communicate(timeout=timeout)
            del running[n]
            if proc.returncode != 0:
                for other in running.values():
                    other.kill()
                raise RuntimeError(f'resource worker failed ({tasks[n][0]}.{tasks[n][1]}): '
                                   f'{err.decode(errors="replace")[-2000:]}')
        return [pickle.loads(result.read_bytes()) for _, result in paths]


def _call(module, function, kwargs):
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    return getattr(importlib.import_module(module), function)(**kwargs)


if __name__ == '__main__':
    task, out = Path(sys.argv[1]), Path(sys.argv[2])
    result = _call(*pickle.loads(task.read_bytes()))
    out.write_bytes(pickle.dumps(result, protocol=5))

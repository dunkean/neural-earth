"""Resident NumPy/SciPy worker threads, plus an optional isolated CLI fallback.

Each task is (module, function, kwargs). The server parallelizes only bulk EDT
kernels; geology stays serial because its small array operations contend for
the GIL. No task pickles or interpreter spawns on the normal HTTP path.
"""
from __future__ import annotations

import importlib
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

HERE = Path(__file__).resolve().parent
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix='resource-numpy')


def thread_map(function, values, timeout=600):
    """Bulk decompression/array work; never call recursively from a pool task."""
    return list(_POOL.map(function, values, timeout=timeout))


def default_workers():
    return max(1, min(8, (os.cpu_count() or 2) - 2))


def parallel_map(tasks, workers=None, timeout=600):
    """Share arrays across resident threads; preserve task order and exceptions.

    Bulk NumPy/SciPy kernels release the GIL. The former file-pickle/spawn
    implementation paid for a complete interpreter and imports for every task.
    """
    if workers == 1 or len(tasks) <= 1:
        return [_call(*task) for task in tasks]
    return thread_map(lambda task: _call(*task), tasks, timeout=timeout)


def process_map(tasks, workers=None, timeout=600):
    """Legacy isolated path, retained for reference timings and crash isolation."""
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

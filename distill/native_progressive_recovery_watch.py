"""Durable paired quality reports for compact160 recovery on a second GPU."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from distill.common import DATA, REPO, atomic_json, external_path
from distill.jobs import state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--train-job', required=True)
    parser.add_argument('--steps', type=int, default=25000)
    parser.add_argument('--gpu-lock', type=Path, help='Serialize resident GPU jobs and evaluations')
    args = parser.parse_args()
    root = external_path(args.root)
    stopped = False
    child = None
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.terminate()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    milestones = sorted(set([1000, 3000, 6000, 10000, 15000, 20000, args.steps]))
    try:
        for step in milestones:
            if step > args.steps:
                continue
            for selection in ('ema', 'ema2000', 'raw'):
                checkpoint = root/selection/f'step{step:06d}.pt'
                while not checkpoint.exists() and not stopped:
                    job = state(DATA/'jobs'/args.train_job)
                    if not job['live'] and job['status'] in ('complete', 'failed'):
                        raise RuntimeError(f'Trainer ended before {selection}/{step}: {job["status"]}')
                    atomic_json(root/'watch-status.json', dict(status='waiting', step=step, selection=selection))
                    time.sleep(30)
                if stopped:
                    return
                for coarse in ('coarse4', 'coarse2'):
                    label = 'progressive160recovery-'+selection
                    report = DATA/'_base'/f'base-{label}-{step:06d}-{coarse}.json'
                    if report.exists() and report.with_suffix('.png').exists():
                        continue
                    atomic_json(root/'watch-status.json', dict(status='evaluating', step=step, selection=selection, coarse=coarse))
                    lock = None
                    try:
                        if args.gpu_lock:
                            args.gpu_lock.parent.mkdir(parents=True, exist_ok=True)
                            lock = args.gpu_lock.open('a')
                            while not stopped:
                                try:
                                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                                    break
                                except BlockingIOError:
                                    atomic_json(root/'watch-status.json', dict(status='waiting_gpu', step=step,
                                                selection=selection, coarse=coarse, lock=str(args.gpu_lock)))
                                    time.sleep(2)
                            if stopped:
                                return
                        child = subprocess.Popen([sys.executable, '-u', '-m', 'distill.native_progressive_evaluate',
                            '--checkpoint', str(checkpoint), '--coarse', coarse, '--label', label], cwd=REPO)
                        code = child.wait(); child = None
                    finally:
                        if lock:
                            lock.close()
                    if code or stopped:
                        raise RuntimeError(f'Evaluation exited{code}; inspect logs before retry.')
            reports = [json.loads(path.read_text()) for path in (DATA/'_base').glob(f'base-progressive160recovery-*-{step:06d}-*.json')]
            atomic_json(root/f'milestone{step:06d}.json', dict(status='complete', step=step, reports=reports,
                note='Paired quality evidence; this width-only recovery does not establish3x speed or<100MB.'))
        atomic_json(root/'watch-status.json', dict(status='complete', next='Review quality; distill one pass/fewer blocks, then quantize and benchmark.'))
    except Exception as exc:
        atomic_json(root/'watch-status.json', dict(status='interrupted' if stopped else 'failed', error=str(exc)))
        raise


if __name__ == '__main__':
    main()

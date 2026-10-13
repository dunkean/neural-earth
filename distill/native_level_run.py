"""Durable paired10k pilot: training5090, serialized progress evaluations4090."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from distill.common import DATA, REPO, atomic_json, external_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'calibration', 'floors', 'moments', 'smoke', 'root'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    root = external_path(args.root)
    children, eval_process = [], None
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
        for process in children:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    shared = [item for name in ('source', 'calibration', 'floors', 'moments', 'smoke', 'root')
              for item in ('--'+name, str(getattr(args, name)))]
    def verify_eval():
        nonlocal eval_process
        if eval_process is not None:
            code = eval_process.wait()
            if code and not stopping:
                raise RuntimeError('Progress evaluation failed; inspect its retained log.')
            children.remove(eval_process)
            eval_process = None
    try:
        for endpoint in (72500, 75000, 77500, 80000):
            for arm in ('a', 'b'):
                if stopping:
                    break
                status = root/arm/'status.json'
                if status.exists():
                    state = json.loads(status.read_text())
                    if state['status'] == 'failed':
                        raise ValueError('Failed arm requires investigation before explicit resume.')
                    if state['step'] >= endpoint:
                        continue
                atomic_json(root/'controller-status.json', dict(status='running', task=f'{arm}:{endpoint}', target=80000))
                process = subprocess.Popen([sys.executable, '-u', '-m', 'distill.native_level_train',
                    *shared, '--arm', arm, '--end-step', str(endpoint)], cwd=REPO)
                children.append(process)
                code = process.wait()
                children.remove(process)
                if code and not stopping:
                    raise RuntimeError(f'{arm}:{endpoint} failed; no automatic restart.')
            if stopping:
                break
            verify_eval()
            # A single worker serializes all4090 evaluations while the5090
            # proceeds to the next paired block. Retain its distinct log.
            command = [sys.executable, '-u', '-m', 'distill.native_level_run', '--evaluate-root', str(root),
                       '--evaluate-step', str(endpoint)]
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='0', TERRAIN_CUDA_DEVICE='0')
            log = (root/f'evaluation-{endpoint:06d}.log').open('a', buffering=1)
            eval_process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
            log.close()
            children.append(eval_process)
        verify_eval()
        atomic_json(root/'controller-status.json', dict(status='interrupted' if stopping else 'complete', target=80000))
    except Exception as exc:
        stop(signal.SIGTERM, None)
        for process in children:
            process.wait()
        atomic_json(root/'controller-status.json', dict(status='failed', error=str(exc), target=80000))
        raise


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluate-root', type=Path, required=True)
    parser.add_argument('--evaluate-step', type=int, required=True)
    args = parser.parse_args()
    current = None
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if current is not None and current.poll() is None:
            current.send_signal(signal.SIGTERM)
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    for arm in ('a', 'b'):
        for raw in (False, True):
            label = f'nf4level{arm}'+('raw' if raw else '')
            checkpoint = args.evaluate_root/arm/('raw' if raw else '')/f'step{args.evaluate_step:06d}.pt'
            for coarse in ('coarse4', 'coarse2'):
                if stopped:return
                stem = DATA/'_base'/f'native-weight-{label}-{coarse}-{args.evaluate_step:06d}'
                if stem.with_suffix('.json').exists() and stem.with_suffix('.png').exists():
                    continue
                current = subprocess.Popen([sys.executable, '-u', '-m', 'distill.native_weight_quant_evaluate',
                    '--checkpoint', str(checkpoint), '--output', str(DATA/'_base'),
                    '--label', label, '--coarse', coarse], cwd=REPO)
                code = current.wait()
                current = None
                if code and not stopped:raise RuntimeError('Evaluation failed; inspect source and log.')


if __name__ == '__main__':
    if '--evaluate-root' in sys.argv:
        evaluate()
    else:
        main()

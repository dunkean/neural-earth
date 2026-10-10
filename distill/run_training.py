"""Sparse background coordination of the three students; quality remains gated.

Runs in tmux through distill.jobs. Checks process handles every minute and logs
only transitions. A failed/interrupted child stops coordination for inspection;
relaunching this coordinator resumes checkpoints and skips completed stages.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import torch

from distill.common import DATA, REPO, atomic_json
from distill.jobs import state


def wait_job(name):
    directory = DATA/'jobs'/name
    while True:
        current = state(directory)
        if not current['live']:
            # Recheck a tiny completion-write race, never restart on a timeout.
            if current['status'] in ('running', 'starting'):
                time.sleep(1)
                current = state(directory)
            if not current['live']:
                if current['status'] != 'complete' or current.get('returncode') != 0:
                    raise RuntimeError(f'{name}: terminal/missing handle, status={current}')
                return
        time.sleep(60)


def wait_dataset(root, minimum, full=False):
    while True:
        counts = {stage: (len(list((root/stage).glob('train-*.npz'))),
                          len(list((root/stage).glob('val-*.npz')))) for stage in ('coarse', 'base', 'decoder')}
        if all(train >= minimum and val >= 384 for train, val in counts.values()):
            if not full:
                return counts
            producer = state(DATA/'jobs/teacher-main')
            if not producer['live']:
                wait_job('teacher-main')
                return counts
        producer = state(DATA/'jobs/teacher-main')
        if not producer['live'] and producer['status'] not in ('starting',):
            raise RuntimeError(f'Teacher stopped before the required data were ready: {producer}')
        time.sleep(60)


def train(name, stage, dataset, output, steps, width, batch, size=None, overfit=False):
    current = state(DATA/'jobs'/name)
    if current['live']:
        wait_job(name)
    latest = output/'latest.pt'
    saved = torch.load(latest, map_location='cpu', weights_only=True) if latest.exists() else None
    status_path = output/'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if saved and saved['step'] >= steps and status.get('status') == 'complete':
        return
    if status.get('status') in ('failed', 'interrupted'):
        print(f'Resuming {stage} from the saved step {saved["step"] if saved else 0}.', flush=True)
    command = [sys.executable, '-m', 'distill.train', '--stage', stage, '--dataset', str(dataset),
               '--output', str(output), '--width', str(width), '--steps', str(steps), '--batch', str(batch),
               '--workers', '2', '--eval-every', '5000', '--save-seconds', '120']
    if size:
        command.extend(['--train-size', str(size)])
    if overfit:
        command.append('--overfit')
    if stage == 'base' and not overfit:
        # The first main pass achieved good slopes but still ~50 m physical MAE
        # and excess high-frequency power. Prioritize the small height channel
        # and supervise its nonlinear metre conversion directly.
        command.extend(['--height-weight', '32', '--height-mae-weight', '.02'])
    if saved:
        command.extend(['--resume', str(latest), '--allow-data-growth'])
    print(json.dumps(dict(starting=name, stage=stage, steps=steps, resume_step=saved['step'] if saved else 0)), flush=True)
    subprocess.run([sys.executable, '-m', 'distill.jobs', 'start', name, '--gpu', '1', '--', *command], cwd=REPO, check=True)
    wait_job(name)
    completed = json.loads(status_path.read_text())
    if completed.get('status') != 'complete' or completed.get('step', 0) < steps:
        raise RuntimeError(f'{name}: trainer stopped without finishing its steps; inspect checkpoint and resume.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke-only', action='store_true')
    args = parser.parse_args()
    progress_path = DATA/'training-workflow.json'
    def progress(phase):
        atomic_json(progress_path, dict(status='running', phase=phase, updated_at=time.time()))
        print(f'PHASE {phase}', flush=True)
    try:
        progress('smoke-coarse')
        train('smoke-coarse', 'coarse', DATA/'crops/smoke', DATA/'ckpt/smoke-coarse', 5000, 64, 8, overfit=True)
        progress('smoke-base')
        train('smoke-base', 'base', DATA/'crops/smoke', DATA/'ckpt/smoke-base', 20000, 64, 1, size=128, overfit=True)
        progress('smoke-decoder')
        train('smoke-decoder', 'decoder', DATA/'crops/smoke', DATA/'ckpt/smoke-decoder', 10000, 64, 1, overfit=True)
        if args.smoke_only:
            atomic_json(progress_path, dict(status='smoke-complete', updated_at=time.time()))
            return
        progress('waiting-for-initial-data')
        wait_dataset(DATA/'crops/main', 512)
        # Use the 5090 while the 4090 produces the full dataset. Each pass uses a
        # frozen filename list. Later passes explicitly admit added examples.
        progress('coarse-initial')
        train('train-coarse', 'coarse', DATA/'crops/main', DATA/'ckpt/coarse', 100000, 64, 8)
        progress('base-initial')
        train('train-base', 'base', DATA/'crops/main', DATA/'ckpt/base', 100000, 96, 1, size=128)
        progress('decoder-initial')
        train('train-decoder', 'decoder', DATA/'crops/main', DATA/'ckpt/decoder', 100000, 64, 1)
        progress('waiting-for-complete-data')
        wait_dataset(DATA/'crops/main', 19616, full=True)
        progress('auditing-complete-dataset')
        subprocess.run([sys.executable, '-m', 'distill.check_dataset', '--dataset', str(DATA/'crops/main')],
                       cwd=REPO, check=True)
        progress('coarse-full-data')
        train('train-coarse', 'coarse', DATA/'crops/main', DATA/'ckpt/coarse', 300000, 64, 8)
        progress('base-full-data')
        train('train-base', 'base', DATA/'crops/main', DATA/'ckpt/base', 400000, 96, 1, size=256)
        progress('decoder-full-data')
        train('train-decoder', 'decoder', DATA/'crops/main', DATA/'ckpt/decoder', 200000, 64, 1)
        atomic_json(progress_path, dict(status='candidates-ready-for-physical-evaluation',
                    checkpoints={stage:str(DATA/'ckpt'/stage/'best.pt') for stage in ('coarse','base','decoder')},
                    accepted=False, updated_at=time.time()))
        print('Three candidates ready. Physical, seam and speed gates remain required.', flush=True)
    except Exception as exc:
        atomic_json(progress_path, dict(status='needs-inspection', error=f'{type(exc).__name__}: {exc}',
                                       updated_at=time.time()))
        raise


if __name__ == '__main__':
    main()

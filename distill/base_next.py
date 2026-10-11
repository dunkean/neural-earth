"""Queue measured native-base work after coarse deliveries or explicit deferral.

Process identities are checked while waiting; terminal failures stop the queue.
The queue benchmarks, captures fresh teacher scenes, calibrates, then trains a
bounded native pilot. Later blending, decoded quality, seams and full training
remain explicit work, never inferred from this pilot's success.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from distill.common import DATA, REPO, atomic_json
from distill.jobs import state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--defer-coarse', action='store_true',
                        help='Continue base work after the user defers incomplete coarse runs; require their jobs stopped.')
    args = parser.parse_args()
    progress_path = DATA/'native-base-next.json'
    stopped = False
    process = None

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)

    def progress(phase, **values):
        atomic_json(progress_path, dict(status='running', phase=phase, updated_at=time.time(), **values))
        print(json.dumps(dict(phase=phase, **values)), flush=True)

    if args.defer_coarse:
        jobs = {name: state(DATA/'jobs'/name) for name in ('coarse-full1', 'coarse-full2')}
        if any(job['live'] for job in jobs.values()):
            raise RuntimeError('Deferred coarse jobs must be stopped before base work.')
        progress('coarse-deferred-by-user')
    else:
        progress('waiting-for-full-coarse')
    while not stopped and not args.defer_coarse:
        jobs = {name: state(DATA/'jobs'/name) for name in ('coarse-full1', 'coarse-full2')}
        if all(not job['live'] for job in jobs.values()):
            for name, job in jobs.items():
                if job['status'] != 'complete' or job.get('returncode') != 0:
                    raise RuntimeError(f'{name} did not finish successfully; inspect before base work.')
                number = name[-1]
                delivery = json.loads((DATA/f'ckpt/coarse-full/coarse{number}/delivery.json').read_text())
                if delivery['status'] != 'complete' or delivery['training_steps'] != 200000:
                    raise RuntimeError('A full coarse delivery is missing.')
            break
        for name, job in jobs.items():
            if not job['live'] and (job.get('status') != 'complete' or job.get('returncode') != 0):
                raise RuntimeError(f'{name} is terminal without success; do not silently restart.')
        time.sleep(15)
    if stopped:
        atomic_json(progress_path, dict(status='interrupted', phase='waiting-for-full-coarse'))
        return

    def run(phase, gpu, command):
        nonlocal process
        if stopped:
            return False
        progress(phase)
        environment = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu),
                           TERRAIN_CUDA_DEVICE='0', TERRAIN_GPU_MODE='single', TERRAIN_CUDA_DEVICES='0')
        process = subprocess.Popen([sys.executable, '-u', '-m', *command], cwd=REPO, env=environment)
        code = process.wait()
        if stopped:
            atomic_json(progress_path, dict(status='interrupted', phase=phase))
            return False
        if code:
            atomic_json(progress_path, dict(status='failed', phase=phase, returncode=code))
            raise RuntimeError(f'{phase} failed with exit{code}.')
        return True

    benchmark = DATA/'bench/native-architectures/gpu1.json'
    if not run('native-speed-feasibility', 1, ['distill.bench_native', '--output', str(benchmark)]):
        return
    report = json.loads(benchmark.read_text())
    comparisons = {row['variant']: row for row in report['comparisons']}
    candidates = [(128,2), (96,2), (128,1), (96,1)]
    choice = next(((width,blocks) for width,blocks in candidates
                   if comparisons[f'native{width}x{blocks}']['speedup'] >= 2.), None)
    if choice is None:
        atomic_json(progress_path, dict(status='needs-strategy-revision', phase='native-speed-feasibility',
                    benchmark=str(benchmark), reason='No candidate reaches2x on the complete base stage.'))
        return
    width, blocks = choice
    progress('selected-native-pilot', width=width, blocks=blocks, benchmark=str(benchmark))
    bank = DATA/'crops/native-scenes-v1'
    if not run('native-teacher-scenes', 0, ['distill.native_bank', 'capture', '--output', str(bank)]):
        return
    calibration = DATA/'eval/native-channel-calibration-v1.json'
    if not run('native-train-calibration', 0, ['distill.calibrate_native', '--dataset', str(bank),
                                               '--output', str(calibration)]):
        return
    output = DATA/'ckpt'/f'native{width}x{blocks}-pilot'
    if not run('native-operator-pilot', 1, ['distill.native_base', 'train', '--dataset', str(bank),
        '--output', str(output), '--width', str(width), '--blocks', str(blocks), '--calibration', str(calibration),
        '--steps', '5000', '--batch', '4', '--eval-every', '500', '--val-count', '96']):
        return
    atomic_json(progress_path, dict(status='pilot-complete', width=width, blocks=blocks,
        checkpoint=str(output/'best.pt'), speed_benchmark=str(benchmark),
        remaining=['blended-region training', 'student-state refinement', 'decoded morphology and coast',
                   'cold tile seams and halos', 'full training', 'final comparison plates and metrics'],
        accepted=False))


if __name__ == '__main__':
    main()

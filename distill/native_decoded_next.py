"""Audit, train and physically evaluate the original-decoder native pilot."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time

from distill.common import DATA, REPO, atomic_json
from distill.jobs import state


def main():
    report = DATA/'native-decoded-next.json'
    process = None
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)

    def progress(phase, **values):
        atomic_json(report, dict(status='running', phase=phase, updated_at=time.time(), **values))
        print(json.dumps(dict(phase=phase, **values)), flush=True)

    def run(phase, gpu, command):
        nonlocal process
        if stopped:
            atomic_json(report, dict(status='interrupted', phase=phase))
            return False
        progress(phase)
        environment = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu),
                           TERRAIN_CUDA_DEVICE='0', TERRAIN_GPU_MODE='single', TERRAIN_CUDA_DEVICES='0')
        process = subprocess.Popen([sys.executable, '-u', *command], cwd=REPO, env=environment)
        code = process.wait()
        if stopped:
            atomic_json(report, dict(status='interrupted', phase=phase))
            return False
        if code:
            atomic_json(report, dict(status='failed', phase=phase, returncode=code))
            raise RuntimeError(f'{phase} failed; inspect before restarting.')
        return True

    progress('waiting-for-fusion-gpu1')
    while not stopped:
        fusion = state(DATA/'jobs/base-native-fusion-next')
        path = DATA/'native-fusion-next.json'
        phase = json.loads(path.read_text()).get('phase') if path.exists() else None
        training = DATA/'ckpt/native128x2-fusion-pilot/status.json'
        status = json.loads(training.read_text()) if training.exists() else {}
        if status.get('status') == 'complete' and status.get('step') == 2000 and phase != 'native-fusion-training':
            break
        if not fusion['live']:
            raise RuntimeError('Fusion queue ended without releasing a complete training run; inspect it.')
        time.sleep(5)
    if stopped:
        atomic_json(report, dict(status='interrupted', phase='waiting-for-fusion-gpu1'))
        return
    controls = DATA/'eval/native-decoded-controls'
    for index in range(6):
        if not run(f'decoder-proxy-control-{index}', 1, ['-m', 'distill.native_decoded_probe',
            '--control-index', str(index), '--controls-only', '--output', str(controls/f'profile-{index}')]):
            return
    output = DATA/'ckpt/native128x2-decoded-pilot'
    if not run('native-decoded-training', 1, ['-m', 'distill.native_decoded_train',
        '--output', str(output), '--controls', str(controls), '--steps', '2000']):
        return
    trained = json.loads((output/'status.json').read_text())
    if trained.get('status') != 'complete':
        atomic_json(report, dict(status='needs-review', phase='native-decoded-training', training=trained, accepted=False))
        return
    progress('waiting-for-fusion-physics-gpu0')
    while not stopped:
        fusion = state(DATA/'jobs/base-native-fusion-next')
        if not fusion['live']:
            if fusion.get('status') != 'complete' or fusion.get('returncode') != 0:
                raise RuntimeError('Prior physical queue failed; inspect before starting another evaluation.')
            break
        time.sleep(5)
    if stopped:
        atomic_json(report, dict(status='interrupted', phase='waiting-for-fusion-physics-gpu0'))
        return
    physical = DATA/'eval/native128x2-decoded-pilot'
    common = ['--variants', 'reference', 'fp32base', 'native_student', '--native-student', str(output/'best.pt')]
    if not run('native-decoded-physics', 0, ['tools/verification/compare_base_variants.py', 'run', str(physical), *common]):
        return
    if not run('native-decoded-rare-physics', 0, ['tools/verification/compare_base_variants.py', 'run', str(physical/'rare'),
        *common, '--site-manifest', str(DATA/'eval/rare-sites-complete.json')]):
        return
    if not run('native-decoded-cold-seams', 0, ['-m', 'distill.check_physical_seams',
        '--native-base', str(output/'best.pt'), '--no-preview', '--output', str(physical/'cold-seams')]):
        return
    atomic_json(report, dict(status='physics-ready', checkpoint=str(output/'best.pt'), physical=str(physical),
        accepted=False, remaining=['physical morphology and rare-case decision', 'cold seam decision',
                                   'full training if promising', 'trained inference benchmark', 'final comparison plates']))


if __name__ == '__main__':
    main()

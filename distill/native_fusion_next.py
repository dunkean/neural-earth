"""Continue a successful native operator pilot with audited fusion and physics.

This queue preserves all candidates and produces internal numerical evidence.
It never generates the user's final plates or infers acceptance from exit zero.
"""
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
    path=DATA/'native-fusion-next.json'
    stopped=False
    process=None

    def stop(signum,frame):
        nonlocal stopped
        stopped=True
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)

    for signum in (signal.SIGINT,signal.SIGTERM):
        signal.signal(signum,stop)

    def progress(phase,**values):
        atomic_json(path,dict(status='running',phase=phase,updated_at=time.time(),**values))
        print(json.dumps(dict(phase=phase,**values)),flush=True)

    progress('waiting-for-native-operator')
    while not stopped:
        job=state(DATA/'jobs/base-native-next')
        if not job['live']:
            if job['status'] != 'complete' or job.get('returncode') != 0:
                raise RuntimeError('Native operator queue failed; inspect it before fusion.')
            report=json.loads((DATA/'native-base-next.json').read_text())
            if report.get('status') != 'pilot-complete':
                raise RuntimeError('The native operator pilot is incomplete; inspect before fusion.')
            break
        time.sleep(5)
    if stopped:
        atomic_json(path,dict(status='interrupted',phase='waiting-for-native-operator'))
        return

    def run(phase,gpu,command):
        nonlocal process
        if stopped:
            return False
        progress(phase)
        environment=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=str(gpu),
                         TERRAIN_CUDA_DEVICE='0',TERRAIN_GPU_MODE='single',TERRAIN_CUDA_DEVICES='0')
        process=subprocess.Popen([sys.executable,'-u',*command],cwd=REPO,env=environment)
        code=process.wait()
        if stopped:
            atomic_json(path,dict(status='interrupted',phase=phase))
            return False
        if code:
            atomic_json(path,dict(status='failed',phase=phase,returncode=code))
            raise RuntimeError(f'{phase} failed with exit{code}; inspect before retrying.')
        return True

    smoke=json.loads((DATA/'eval/native-fusion-gpu-smoke.json').read_text())
    if smoke.get('status') != 'complete' or len(smoke.get('rows',[])) < 2:
        raise ValueError('Real native forward/backward smoke evidence is required.')
    width,blocks=report['width'],report['blocks']
    bank=DATA/'crops/native-scenes-v1'
    source=DATA/f'ckpt/native{width}x{blocks}-pilot/best.pt'
    output=DATA/f'ckpt/native{width}x{blocks}-fusion-pilot'
    if not run('native-fusion-reconstruction',1,['-m','distill.native_fusion','--dataset',str(bank),
        '--output',str(DATA/'eval/native-fusion-reconstruction-full.json'),'--count','24']):
        return
    if not run('native-fusion-training',1,['-m','distill.native_fusion_train','--dataset',str(bank),
        '--source',str(source),'--output',str(output),'--steps','2000','--eval-every','250','--val-count','48']):
        return
    if json.loads((output/'status.json').read_text()).get('status') != 'complete':
        raise RuntimeError('Fusion training interrupted; do not evaluate a partial run silently.')
    physical=DATA/f'eval/native{width}x{blocks}-fusion-pilot'
    if not run('native-decoded-physics',0,['tools/verification/compare_base_variants.py','run',str(physical),
        '--variants','reference','fp32base','native_student','--native-student',str(output/'best.pt')]):
        return
    if not run('native-rare-decoded-physics',0,['tools/verification/compare_base_variants.py','run',str(physical/'rare'),
        '--variants','reference','fp32base','native_student','--native-student',str(output/'best.pt'),
        '--site-manifest',str(DATA/'eval/rare-sites-complete.json')]):
        return
    if not run('native-cold-physical-seams',0,['-m','distill.check_physical_seams',
        '--native-base',str(output/'best.pt'),'--no-preview','--output',str(physical/'cold-seams')]):
        return
    atomic_json(path,dict(status='physics-ready',checkpoint=str(output/'best.pt'),physical=str(physical),
        accepted=False,remaining=['erosion and rare-case decision','cold tile seam decision','full base training',
                                  'trained inference benchmark','final comparison plates']))


if __name__ == '__main__':
    main()

"""Durable bounded one-pass pilot, sharing the4090 with recovery evaluations."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

from distill.common import DATA, REPO, atomic_json, external_path
from distill.jobs import state


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--smoke-job',required=True)
    parser.add_argument('--gpu-lock',type=Path,required=True)
    args=parser.parse_args()
    root=external_path(args.root)
    stopped=False;child=None
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
        if child is not None and child.poll() is None:child.terminate()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    def execute(command,phase):
        nonlocal child
        args.gpu_lock.parent.mkdir(parents=True,exist_ok=True)
        with args.gpu_lock.open('a') as lock:
            while not stopped:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    atomic_json(root/'pilot-status.json',dict(status='waiting_gpu',phase=phase,lock=str(args.gpu_lock)))
                    time.sleep(2)
            if stopped:raise InterruptedError('Pilot interrupted.')
            child=subprocess.Popen([sys.executable,'-u',*command],cwd=REPO)
            atomic_json(root/'pilot-status.json',dict(status='running',phase=phase,child_pid=child.pid))
            code=child.wait();child=None
            if code or stopped:raise RuntimeError(f'{phase} exited{code};inspect before retry.')
    def plate(checkpoint,step,coarse,label):
        path=DATA/'_base'/f'base-{label}-{step:06d}-{coarse}.json'
        if path.exists() and path.with_suffix('.png').exists():return
        execute(['-m','distill.native_progressive_evaluate','--checkpoint',str(checkpoint),
                 '--coarse',coarse,'--label',label],f'quality:{step}:{coarse}:{label}')
    try:
        while not stopped:
            job=state(DATA/'jobs'/args.smoke_job)
            if not job['live']:
                if job.get('status')!='complete' or job.get('returncode')!=0:
                    raise RuntimeError('Smoke job ended without success.')
                break
            atomic_json(root/'pilot-status.json',dict(status='waiting_smoke',job=args.smoke_job,child_pid=job.get('child_pid')))
            time.sleep(10)
        if stopped:return
        proof=json.loads((root/'smoke/gpu-smoke.json').read_text())
        if proof['status']!='complete':raise ValueError('A completed matching smoke is required.')
        for coarse in ('coarse4','coarse2'):
            plate(root/'preparation-v2/initial.pt',0,coarse,'onepass160x2-initial')
        progress=root/'pilot/status.json'
        current=json.loads(progress.read_text()) if progress.exists() else {}
        if current.get('step',0)<500:
            execute(['-m','distill.native_onepass_train','--preparation',str(root/'preparation-v2'),
                '--output',str(root/'pilot'),'--steps','10000','--until','500','--smoke-root',str(root/'smoke')],
                'train-to500')
        for selection in ('ema','raw'):
            checkpoint=root/'pilot'/('raw' if selection=='raw' else '')/'step000500.pt'
            for coarse in ('coarse4','coarse2'):
                plate(checkpoint,500,coarse,'onepass160x2-'+selection)
        reports=[]
        for label,step in [('onepass160x2-initial',0),('onepass160x2-ema',500),('onepass160x2-raw',500)]:
            for coarse in ('coarse4','coarse2'):
                path=DATA/'_base'/f'base-{label}-{step:06d}-{coarse}.json'
                report=json.loads(path.read_text())
                reports.append(dict(label=label,step=step,coarse=coarse,path=str(path),
                    mean_mae={str(lod):statistics.mean(row['mae_m'] for row in report['rows'] if row['lod']==lod) for lod in (3,0)},
                    rows=report['rows']))
        atomic_json(root/'PILOT500.json',dict(status='complete',reports=reports,
            next='Inspect fidelity and grain; then extend one-pass, QAT/pack, cold-seam and full-chain benchmarks.',
            accepted=False,goal_complete=False))
        atomic_json(root/'pilot-status.json',dict(status='pilot_complete',step=500,next='Inspect PILOT500.json before extending.'))
    except Exception as exc:
        atomic_json(root/'pilot-status.json',dict(status='interrupted' if stopped else 'failed',error=str(exc)))
        raise


if __name__=='__main__':main()

"""Durable residual refinement in250-step GPU-sharing segments andVAL plates."""
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--until',type=int,default=1000)
    parser.add_argument('--gpu-lock',type=Path,required=True)
    parser.add_argument('--smoke-job',default='base-onepass-residual-smoke-v2-4090')
    args=parser.parse_args();root=external_path(args.root)
    if args.until<250 or args.until>10000 or args.until%250:raise ValueError('A250-step milestone required.')
    stopped=False;child=None
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
        if child is not None and child.poll() is None:child.terminate()
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    def execute(command,phase):
        nonlocal child
        with args.gpu_lock.open('a') as lock:
            while not stopped:
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    atomic_json(root/'run-status.json',dict(status='waiting_gpu',phase=phase))
                    time.sleep(2)
            if stopped:raise InterruptedError('Refinement interrupted.')
            child=subprocess.Popen([sys.executable,'-u',*command],cwd=REPO)
            atomic_json(root/'run-status.json',dict(status='running',phase=phase,child_pid=child.pid))
            code=child.wait();child=None
            if code or stopped:raise RuntimeError(f'{phase} exited{code};inspect before retry.')
    def plates(step):
        for selection in ('ema','raw'):
            checkpoint=root/'train'/('raw' if selection=='raw' else '')/f'step{step:06d}.pt'
            for coarse in ('coarse4','coarse2'):
                label='onepass-residual-'+selection
                report=DATA/'_base'/f'base-{label}-{step:06d}-{coarse}.json'
                if report.exists() and report.with_suffix('.png').exists():continue
                execute(['-m','distill.native_progressive_evaluate','--checkpoint',str(checkpoint),
                    '--coarse',coarse,'--label',label],f'quality:{step}:{selection}:{coarse}')
        atomic_json(root/f'milestone{step:06d}.json',dict(status='complete',step=step,
            reports=[str(p) for p in (DATA/'_base').glob(f'base-onepass-residual-*-{step:06d}-*.json')],accepted=False))
    try:
        while not stopped:
            job=state(DATA/'jobs'/args.smoke_job)
            if not job['live']:
                if job.get('status')!='complete' or job.get('returncode')!=0:raise ValueError('Residual smoke failed.')
                break
            atomic_json(root/'run-status.json',dict(status='waiting_smoke',job=args.smoke_job,child_pid=job.get('child_pid')))
            time.sleep(5)
        if stopped:return
        if json.loads((root/'smoke-v2/gpu-smoke.json').read_text())['status']!='complete':raise ValueError('Matching smoke required.')
        progress=root/'train/status.json'
        current=json.loads(progress.read_text()) if progress.exists() else dict(step=0)
        if current.get('status')=='failed':raise ValueError('Previous trainer failed;no automatic retry.')
        step=current['step']
        for mark in (500,1000,2000,3000,5000,7500,10000):
            if mark<=step and mark<=args.until:plates(mark)
        while step<args.until and not stopped:
            target=min(args.until,((step//250)+1)*250)
            execute(['-m','distill.native_onepass_residual_train','--preparation',str(root/'preparation'),
                '--output',str(root/'train'),'--steps','10000','--until',str(target),'--smoke-root',str(root/'smoke-v2')],
                f'train:{step}:{target}')
            saved=json.loads(progress.read_text())
            if saved['step']!=target or saved['status'] not in ('segment_complete','complete'):
                raise ValueError('Residual trainer did not finish its segment.')
            step=target
            if step in (500,1000,2000,3000,5000,7500,10000) or step==args.until:plates(step)
        atomic_json(root/'run-status.json',dict(status='interrupted' if stopped else 'complete',step=step,
            next='Inspect terrain positions/levels/coasts;then QAT andcold/full-chain acceptance.'))
    except Exception as exc:
        atomic_json(root/'run-status.json',dict(status='interrupted' if stopped else 'failed',error=str(exc)))
        raise


if __name__=='__main__':main()

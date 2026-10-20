"""Resume one-pass pilots in250-step segments with durable quality milestones."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from distill.common import DATA, REPO, atomic_json, external_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--until',type=int,default=2000)
    parser.add_argument('--gpu-lock',type=Path,required=True)
    args=parser.parse_args();root=external_path(args.root)
    if args.until<=500 or args.until>10000 or args.until%250:
        raise ValueError('Continuation endpoint must be a250-step milestone after the500 pilot.')
    if json.loads((root/'PILOT500.json').read_text())['status']!='complete':
        raise ValueError('The first pilot must be reviewed before continuation.')
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
                    atomic_json(root/'continuation-status.json',dict(status='waiting_gpu',phase=phase))
                    time.sleep(2)
            if stopped:raise InterruptedError('Continuation interrupted.')
            child=subprocess.Popen([sys.executable,'-u',*command],cwd=REPO)
            atomic_json(root/'continuation-status.json',dict(status='running',phase=phase,child_pid=child.pid))
            code=child.wait();child=None
            if code or stopped:raise RuntimeError(f'{phase} exited{code};inspect before retry.')
    def evaluate(step):
        for selection in ('ema','raw'):
            path=root/'pilot'/('raw' if selection=='raw' else '')/f'step{step:06d}.pt'
            for coarse in ('coarse4','coarse2'):
                label='onepass160x2-'+selection
                report=DATA/'_base'/f'base-{label}-{step:06d}-{coarse}.json'
                if report.exists() and report.with_suffix('.png').exists():continue
                execute(['-m','distill.native_progressive_evaluate','--checkpoint',str(path),
                    '--coarse',coarse,'--label',label],f'quality:{step}:{selection}:{coarse}')
        atomic_json(root/f'milestone{step:06d}.json',dict(status='complete',step=step,
            reports=[str(p) for p in (DATA/'_base').glob(f'base-onepass160x2-*-{step:06d}-*.json')],
            accepted=False,goal_complete=False))
    try:
        current=json.loads((root/'pilot/status.json').read_text())
        if current['status']=='failed':raise ValueError('Previous trainer failed; no automatic retry.')
        step=current['step']
        # On resumption, finish quality evidence for already-produced milestones.
        for mark in (1000,2000,3000,5000,7500,10000):
            if mark<=step and mark<=args.until:evaluate(mark)
        while step<args.until and not stopped:
            target=min(args.until,((step//250)+1)*250)
            execute(['-m','distill.native_onepass_train','--preparation',str(root/'preparation-v2'),
                '--output',str(root/'pilot'),'--steps','10000','--until',str(target),'--smoke-root',str(root/'smoke')],
                f'train:{step}:{target}')
            state=json.loads((root/'pilot/status.json').read_text())
            if state['step']!=target or state['status'] not in ('segment_complete','complete'):
                raise ValueError('Trainer did not complete its requested segment.')
            step=target
            if step in (1000,2000,3000,5000,7500,10000) or step==args.until:evaluate(step)
        atomic_json(root/'continuation-status.json',dict(status='interrupted' if stopped else 'complete',step=step,
            next='Inspect paired raw/EMA terrain evidence; QAT and full-chain acceptance remain required.'))
    except Exception as exc:
        atomic_json(root/'continuation-status.json',dict(status='interrupted' if stopped else 'failed',error=str(exc)))
        raise


if __name__=='__main__':main()

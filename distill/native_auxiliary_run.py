"""Run paired 2500-step NF4 segments and plates under a shared GPU lease."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys

from distill.common import DATA, REPO, atomic_json, external_path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--powers',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--smoke',type=Path,required=True)
    parser.add_argument('--gpu-lock',type=Path,required=True)
    args=parser.parse_args();root=external_path(args.root)
    args.gpu_lock.parent.mkdir(parents=True,exist_ok=True)
    child=None;stopping=False
    def stop(signum,frame):
        nonlocal stopping
        stopping=True
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    def execute(command,description):
        nonlocal child
        atomic_json(root/'controller-status.json',dict(status='running',task=description,accepted=False))
        child=subprocess.Popen(command,cwd=REPO)
        code=child.wait();child=None
        if code and not stopping:raise RuntimeError(f'{description} exited {code}; inspect saved state before explicit resume.')
    try:
        with args.gpu_lock.open('a') as lease:
            for endpoint in (42500,45000,47500,50000):
                for arm in ('a','b'):
                    if stopping:break
                    latest=root/arm/'latest.pt';status=root/arm/'status.json'
                    if status.exists():
                        state=json.loads(status.read_text())
                        if state['status']=='failed':raise ValueError('Failed arm requires explicit investigation.')
                        if state['step']>=endpoint:continue
                    source=latest if latest.exists() else args.source
                    fcntl.flock(lease,fcntl.LOCK_EX)
                    execute([sys.executable,'-u','-m','distill.native_auxiliary_train',
                        '--source',str(source),'--base-source',str(args.source),'--root',str(root),
                        '--powers',str(args.powers),'--calibration',str(args.calibration),
                        '--smoke',str(args.smoke),'--arm',arm,'--end-step',str(endpoint),'--cadence','2'],
                        f'{arm}:{endpoint}')
                    fcntl.flock(lease,fcntl.LOCK_UN)
                if stopping:break
                # Publish both raw and EMA at equal training horizons. Each
                # process exits to release all GPU model/caches before the next.
                for arm in ('a','b'):
                    for raw in (False,True):
                        label=f'nf4abl{arm}'+('raw' if raw else '')
                        checkpoint=root/arm/('raw' if raw else '')/f'step{endpoint:06d}.pt'
                        for coarse in ('coarse4','coarse2'):
                            if stopping:break
                            stem=DATA/'_base'/f'native-weight-{label}-{coarse}-{endpoint:06d}'
                            if stem.with_suffix('.json').exists() and stem.with_suffix('.png').exists():continue
                            fcntl.flock(lease,fcntl.LOCK_EX)
                            execute([sys.executable,'-u','-m','distill.native_weight_quant_evaluate',
                                '--checkpoint',str(checkpoint),'--output',str(DATA/'_base'),
                                '--label',label,'--coarse',coarse],f'plate:{label}:{coarse}:{endpoint}')
                            fcntl.flock(lease,fcntl.LOCK_UN)
            atomic_json(root/'controller-status.json',dict(status='interrupted' if stopping else 'complete',
                target=50000,accepted=False))
    except Exception as exc:
        atomic_json(root/'controller-status.json',dict(status='failed',error=str(exc),accepted=False));raise


if __name__=='__main__':main()

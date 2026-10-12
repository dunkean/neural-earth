"""Publish weight-QAT plates; retain useful checkpoints instead of a1000-step gate."""
import argparse
import fcntl
import json
from pathlib import Path
import time

import numpy as np
import torch

from distill.common import DATA,atomic_json,append_json,external_path
from distill.jobs import state
from distill.native_weight_quant_evaluate import evaluate
from distill.native_weight_quant import export_packed,load_weight_quant,load_packed,file_sha
from distill.wet_mountains import KIND


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--trainer-job',required=True)
    parser.add_argument('--label',default='nf4qat')
    parser.add_argument('--initial-step',type=int,default=0)
    parser.add_argument('--pack-every',type=int,default=0,
                        help='Positive interval avoids exporting every improving long-run checkpoint.')
    parser.add_argument('--monitor-only',action='store_true',help='Report quality regressions without a training stop.')
    parser.add_argument('--gpu-lock',type=Path,help='Serialize GPU work with auxiliary training.')
    parser.add_argument('--resume-completed',action='store_true',help='Reuse complete existing monitor-only reports and plates.')
    args=parser.parse_args();root=external_path(args.run)
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    done=set();baseline=None;bad_count=0;best=None;started=time.monotonic()
    if args.resume_completed:
        if not args.monitor_only:raise ValueError('Report reuse is limited to monitor-only runs.')
        for checkpoint in sorted(root.glob('step??????.pt')):
            step=int(checkpoint.stem[4:]);path=root/f'eval-{step:06d}.json'
            if not path.exists():continue
            entry=json.loads(path.read_text())
            if entry['step']!=step:raise ValueError('Mismatched existing report.')
            plates=[DATA/'_base'/f'native-weight-{args.label}-{coarse}-{step:06d}' for coarse in ('coarse4','coarse2','original')]
            if not all(p.with_suffix('.png').exists() and p.with_suffix('.json').exists() for p in plates):continue
            checkpoint_sha=file_sha(checkpoint)
            if not all(json.loads(p.with_suffix('.json').read_text())['checkpoint_sha256']==checkpoint_sha for p in plates):
                raise ValueError('Existing plates belong to different weights.')
            if args.pack_every and step%args.pack_every==0 and not (root/'packed'/f'step{step:06d}'/'manifest.json').exists():continue
            if step==args.initial_step:
                baseline={str(r['seed'])+r['coarse']:r for r in entry['rows'] if r['kind']==KIND and r['lod']==0}
            if best is None or entry['score']<best['score']:
                best=dict(step=step,score=entry['score'],checkpoint=str(checkpoint),accepted=False)
            done.add(checkpoint.name)
        if done and baseline is None:raise ValueError('Existing initial baseline required to resume reports.')
    lease=None
    if args.gpu_lock:
        args.gpu_lock.parent.mkdir(parents=True,exist_ok=True)
        lease=args.gpu_lock.open('a')
    try:
        while True:
            for checkpoint in sorted(root.glob('step??????.pt')):
                if checkpoint.name in done:continue
                if lease is not None:fcntl.flock(lease,fcntl.LOCK_EX)
                reports=[evaluate(checkpoint,DATA/'_base',args.label,coarse) for coarse in ('coarse4','coarse2','original')]
                step=reports[0]['step'];rows=[r for p in reports for r in p['rows'] if r['coarse']!='original']
                mountains=[r for r in rows if r['kind']==KIND and r['lod']==0]
                if baseline is None:
                    if step!=args.initial_step:raise ValueError('Initial terrain baseline required.')
                    baseline={str(r['seed'])+r['coarse']:r for r in mountains}
                # A practical comparison score, not acceptance. No channel0/2
                # gate; gradual recovery gets time before a persistent regression.
                score=float(np.mean([r['mae_m']/(35 if r['kind']==KIND else 10) for r in rows]))
                hf=[r['high_frequency_rms_ratio'] for r in rows if r['kind']!=KIND and r['high_frequency_rms_ratio'] is not None]
                if hf:score+=.25*float(np.mean(np.abs(np.log(np.maximum(hf,1e-6)))))
                score+=float(np.mean([r['coast_disagreement_fraction'] for r in rows]))
                if best is None or score<best['score']:
                    best=dict(step=step,score=score,checkpoint=str(checkpoint),accepted=False)
                    atomic_json(root/'best-visual-candidate.json',best)
                deteriorated=step>=3000 and all(r['mae_m']>2*baseline[str(r['seed'])+r['coarse']]['mae_m']
                    and r['correlation']<baseline[str(r['seed'])+r['coarse']]['correlation']-.15 for r in mountains)
                bad_count=bad_count+1 if deteriorated else 0
                entry=dict(step=step,score=score,best=best,rows=rows,hard_stop=bad_count>=3 and not args.monitor_only,
                    quality_regression=bad_count>=3,
                    reason='Persistent catastrophic regression on all six wet/coarse2/4 branches' if bad_count>=3 else None,
                    accepted=False,note='User accepts current valleys; evaluate size/grain/coast/look&feel, final cold seams still required.')
                atomic_json(root/f'eval-{step:06d}.json',entry)
                append_json(root/'evaluation.jsonl',entry);atomic_json(root/'evaluation-status.json',entry|dict(status='complete'))
                pack_now=(step%args.pack_every==0 if args.pack_every else
                          step in (0,1000,3000,6000) or step==best['step'])
                if pack_now:
                    destination=root/'packed'/f'step{step:06d}'
                    if not (destination/'manifest.json').exists():
                        saved=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)
                        artifact=export_packed(saved,destination,device='cuda')
                        if artifact['bytes']>150_000_000:raise ValueError('Packed file exceeds150Mo budget.')
                done.add(checkpoint.name)
                if lease is not None:fcntl.flock(lease,fcntl.LOCK_UN)
                print(json.dumps(dict(step=step,score=score,hard_stop=entry['hard_stop'],best_step=best['step'])),flush=True)
            owner=state(DATA/'jobs'/args.trainer_job)
            if not owner['live'] and owner['status'] in ('complete','failed') and not {p.name for p in root.glob('step??????.pt')}-done:break
            if not owner['live'] and owner['status'] not in ('complete','failed') and time.monotonic()-started>600:
                raise RuntimeError('Verified trainer missing.')
            time.sleep(30)
    except Exception as exc:
        atomic_json(root/'evaluation-status.json',dict(status='failed',hard_stop=False,error=str(exc),accepted=False))
        raise


if __name__=='__main__':main()

"""Autonomous native QAT qualification, fixed scales, real comparison plates."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from distill.common import DATA,atomic_json,append_json,external_path
from distill.jobs import state
from distill.native_fp8_evaluate import evaluate
from distill.wet_mountains import KIND


def observe(root,trainer):
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    done=set();baseline=None;started=time.monotonic()
    while True:
        for checkpoint in sorted(root.glob('step??????.pt')):
            if checkpoint.name in done:continue
            reports=[evaluate(checkpoint,DATA/'_base','qat',coarse) for coarse in ('original','coarse4','coarse2')]
            step=reports[0]['step'];rows=[r for report in reports for r in report['rows'] if r['lod']==0]
            mountains=[r for r in rows if r['kind']==KIND];keyed={str(r['seed'])+r['coarse']:r for r in mountains}
            if baseline is None:
                if step!=0:raise ValueError('Post-quantization baseline required.')
                baseline=keyed
            reasons=[]
            for key,r in keyed.items():
                if not np.isfinite(r['mae_m']):reasons.append('Nonfinite wet terrain: '+key)
                if step==0 and r['mae_m']>150:reasons.append('Quantization placement failed150m gate: '+key)
                if step>=1000 and (r['channel0_mse']>.5*baseline[key]['channel0_mse'] or r['correlation']<.97):
                    reasons.append('QAT1000-step channel0/incision gate missed: '+key)
            mountain_pass=all(r['mae_m']<=35 and r['correlation']>=.98 and r['recall']>=.96 and .93<=r['rms']<=1.07 for r in mountains)
            # Keep the historical interior stage gates; final distribution,
            # coast, cold app and reserved-site checks remain separate.
            rare_pass=all(r['correlation'] is not None and r['correlation']>=.97 and .95<=r['rms']<=1.08 for r in rows if r['kind']!=KIND)
            entry=dict(step=step,mountains=mountains,rows=rows,provisional_stage_pass=mountain_pass and rare_pass,
                hard_stop=bool(reasons),reasons=reasons,accepted=False,
                note='Autonomous native64/32,T2. FP8 simulated using BF16 convolutions; final app/seams/reserved sites/distributions/speed remain unproven.')
            atomic_json(root/f'eval-{step:06d}.json',entry|dict(reports=reports))
            append_json(root/'evaluation.jsonl',entry);atomic_json(root/'evaluation-status.json',entry|dict(status='complete'))
            done.add(checkpoint.name);print(json.dumps(dict(step=step,stage_pass=entry['provisional_stage_pass'],hard_stop=bool(reasons))),flush=True)
        owner=state(DATA/'jobs'/trainer)
        if not owner['live'] and owner['status'] in ('complete','failed') and not {p.name for p in root.glob('step??????.pt')}-done:break
        if not owner['live'] and owner['status'] not in ('complete','failed') and time.monotonic()-started>600:
            raise RuntimeError('Verified QAT trainer missing.')
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DATA/'ckpt/native-fp8-pilot')
    parser.add_argument('--trainer-job',default='base-native-fp8-train')
    args=parser.parse_args();root=external_path(args.run)
    try:observe(root,args.trainer_job)
    except Exception as exc:
        atomic_json(root/'evaluation-status.json',dict(status='failed',hard_stop=True,error=str(exc),accepted=False));raise


if __name__=='__main__':main()

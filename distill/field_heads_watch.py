"""Unforced incision and per-window-target gates for the four-head experiment."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

from distill.common import DATA,atomic_json,append_json,external_path
from distill.jobs import state
from distill.field_evaluate import evaluate
from distill.wet_mountains import KIND


def observe(root,trainer):
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    done=set();baseline=None;started=time.monotonic()
    while True:
        for checkpoint in sorted(root.glob('step??????.pt')):
            if checkpoint.name in done:continue
            reports=[evaluate(checkpoint,DATA/'_base','heads-'+coarse,coarse,seams=True)
                     for coarse in ('original','coarse4','coarse2')]
            step=reports[0]['step'];rows=[r for report in reports for r in report['rows'] if r['kind']==KIND and r['lod']==0]
            keyed={str(r['seed'])+r['coarse']:r for r in rows}
            if baseline is None:
                if step!=0:raise ValueError('Initial head baseline required.')
                baseline=keyed
            reasons=[]
            for key,r in keyed.items():
                b=baseline[key]
                if not all(np.isfinite(r[k]) for k in ('mae_m','correlation','initial_channel0_mse','initial_head_mse')):
                    reasons.append('Nonfinite wet validation: '+key)
                if r['correlation']<b['correlation']-.03:reasons.append('Wet correlation regressed>.03: '+key)
                if step>=1000 and r['coarse']=='coarse4' and r['initial_channel0_mse']>.85*b['initial_channel0_mse']:
                    reasons.append('Window-head hypothesis missed15% blended-channel0 improvement: '+key)
                if step>=2000 and not (r['correlation']>=.95 and r['mae_m']<=80 and r['recall']>=.80 and
                                       r['rms']>=.90 and r['initial_head_mse']<b['initial_head_mse']):
                    reasons.append('Incision/per-head2000-step gate missed: '+key)
            quality=all(r['mae_m']<=35 and r['correlation']>=.98 and r['recall']>=.96 and .93<=r['rms']<=1.07 for r in rows)
            entry=dict(step=step,mountains=rows,provisional_mountain_pass=quality,hard_stop=bool(reasons),reasons=reasons,
                accepted=False,note='Unforced chain; field-crop consistency is separate from final app cold-tile acceptance.')
            atomic_json(root/f'eval-{step:06d}.json',entry|dict(reports=reports))
            append_json(root/'evaluation.jsonl',entry);atomic_json(root/'evaluation-status.json',entry|dict(status='complete'))
            subprocess.run([sys.executable,'-m','distill.field_attention_diagnostic','--checkpoint',str(checkpoint),'--label','heads'],check=True)
            done.add(checkpoint.name);print(json.dumps(dict(step=step,quality=quality,hard_stop=bool(reasons))),flush=True)
        owner=state(DATA/'jobs'/trainer)
        if not owner['live'] and owner['status'] in ('complete','failed') and not {p.name for p in root.glob('step??????.pt')}-done:break
        if not owner['live'] and owner['status'] not in ('complete','failed') and time.monotonic()-started>600:
            raise RuntimeError('Verified head trainer missing.')
        time.sleep(30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DATA/'ckpt/field-heads-pilot')
    parser.add_argument('--trainer-job',default='base-field-heads-train')
    args=parser.parse_args();root=external_path(args.run)
    try:observe(root,args.trainer_job)
    except Exception as exc:
        atomic_json(root/'evaluation-status.json',dict(status='failed',hard_stop=True,error=str(exc),accepted=False));raise


if __name__=='__main__':main()

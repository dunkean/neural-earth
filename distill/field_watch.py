"""4090 field observer, photos and both-axis crop-consistency diagnostics."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from distill.common import DATA,atomic_json,append_json,external_path
from distill.jobs import state
from distill.field_evaluate import evaluate
from distill.wet_mountains import KIND


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DATA/'ckpt/field-full-width-pilot')
    parser.add_argument('--trainer-job',default='base-field-full-width-train')
    parser.add_argument('--label',default='fullwidth')
    args=parser.parse_args();root=external_path(args.run)
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    done=set();baseline=None;started=time.monotonic()
    try:
        while True:
            for checkpoint in sorted(root.glob('step[0-9][0-9][0-9][0-9][0-9][0-9].pt')):
                if checkpoint.name in done:continue
                reports=[]
                for coarse in ('original','coarse4','coarse2'):
                    reports.append(evaluate(checkpoint,DATA/'_base',args.label+'-'+coarse,coarse,seams=True))
                step=reports[0]['step']
                rows=[r for report in reports for r in report['rows'] if r['kind']==KIND and r['lod']==0]
                correlations={str(r['seed'])+r['coarse']:r['correlation'] for r in rows}
                if step==0:baseline=correlations
                reasons=[]
                if step>=2000 and baseline and any(correlations[k]<baseline[k] for k in baseline):
                    reasons.append('Wet-mountain correlation regressed below initial field geometry after2000.')
                if any(not np.isfinite(r['mae_m']) for r in rows):reasons.append('Nonfinite terrain validation.')
                quality=all(r['mae_m']<=35 and r['correlation']>=.98 and r['recall']>=.96 and
                            .93<=r['rms']<=1.07 for r in rows)
                entry=dict(step=step,mountains=rows,provisional_mountain_pass=quality,
                    hard_stop=bool(reasons),reasons=reasons,accepted=False,
                    note='Shifted-field seam metrics recorded separately; full app/cold seams/final sites/speed remain required.')
                atomic_json(root/f'eval-{step:06d}.json',entry|dict(reports=reports))
                append_json(root/'evaluation.jsonl',entry)
                atomic_json(root/'evaluation-status.json',entry|dict(status='complete'))
                done.add(checkpoint.name);print(json.dumps(dict(step=step,quality=quality,hard_stop=bool(reasons))),flush=True)
            owner=state(DATA/'jobs'/args.trainer_job)
            if not owner['live'] and owner['status'] in ('complete','failed'):
                remaining={p.name for p in root.glob('step[0-9][0-9][0-9][0-9][0-9][0-9].pt')}-done
                if not remaining:break
            if not owner['live'] and owner['status'] not in ('complete','failed') and time.monotonic()-started>600:
                raise RuntimeError('Verified field trainer handle missing.')
            time.sleep(30)
    except Exception as exc:
        atomic_json(root/'evaluation-status.json',dict(status='failed',hard_stop=True,error=str(exc),accepted=False));raise


if __name__=='__main__':main()

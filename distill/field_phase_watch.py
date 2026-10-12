"""Unforced phase-pupil VAL, wet-valley guards and field-crop diagnostics."""
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=DATA/'ckpt/field-phase-pilot')
    parser.add_argument('--trainer-job',default='base-field-phase-train')
    args=parser.parse_args();root=external_path(args.run)
    try:observe(root,args.trainer_job)
    except Exception as exc:
        atomic_json(root/'evaluation-status.json',dict(status='failed',hard_stop=True,error=str(exc),accepted=False))
        raise


def observe(root,trainer_job):
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    done=set();baseline=None;initial_regressions={};started=time.monotonic()
    while True:
        for checkpoint in sorted(root.glob('step[0-9][0-9][0-9][0-9][0-9][0-9].pt')):
            if checkpoint.name in done:continue
            reports=[evaluate(checkpoint,DATA/'_base','phase-'+coarse,coarse,seams=True)
                     for coarse in ('original','coarse4','coarse2')]
            step=reports[0]['step']
            rows=[r for report in reports for r in report['rows'] if r['kind']==KIND and r['lod']==0]
            keyed={str(r['seed'])+r['coarse']:r for r in rows}
            if baseline is None:
                if step!=0:raise ValueError('Resume baseline is required before later evaluations.')
                baseline=keyed
            reasons=[];eligibility=[]
            for key,row in keyed.items():
                base=baseline[key]
                if not all(np.isfinite(row[k]) for k in ('mae_m','correlation','initial_rms')):
                    reasons.append('Nonfinite wet VAL: '+key)
                if row['correlation']<base['correlation']-.03:reasons.append('Wet corr fell by more than .03: '+key)
                initial_regressions[key]=initial_regressions.get(key,0)+1 if row['initial_rms']<base['initial_rms']-1e-6 else 0
                if initial_regressions[key]>=2:reasons.append('First-phase amplitude fell in two successive VALs: '+key)
                if row['correlation']<base['correlation']:eligibility.append('Resume-correlation regression: '+key)
                if step>=2000 and row['coarse']=='coarse4' and not (
                        row['initial_rms']>=.90 and row['initial_correlation']>=base['initial_correlation']+.03 and row['mae_m']<=80):
                    reasons.append('Bounded phase hypothesis missed its2000-step wet gate: '+key)
            if step>=2000:
                original=json.loads((DATA/'ckpt/field-full-width-pilot/eval-000000.json').read_text())
                old={str(r['seed'])+r['coarse']:r['correlation'] for r in original['mountains']}
                eligibility += ['Original zero-tolerance regression: '+k for k,r in keyed.items() if r['correlation']<old[k]]
            quality=all(r['mae_m']<=35 and r['correlation']>=.98 and r['recall']>=.96 and .93<=r['rms']<=1.07 for r in rows)
            entry=dict(step=step,mountains=rows,provisional_mountain_pass=quality,hard_stop=bool(reasons),
                reasons=reasons,unselectable_reasons=eligibility,accepted=False,
                note='Unforced autonomous chain. Field-crop consistency is not final app cold-tile acceptance.')
            atomic_json(root/f'eval-{step:06d}.json',entry|dict(reports=reports))
            append_json(root/'evaluation.jsonl',entry)
            atomic_json(root/'evaluation-status.json',entry|dict(status='complete'))
            subprocess.run([sys.executable,'-m','distill.field_attention_diagnostic','--checkpoint',str(checkpoint),
                            '--label','phase'],check=True)
            done.add(checkpoint.name);print(json.dumps(dict(step=step,quality=quality,hard_stop=bool(reasons))),flush=True)
        owner=state(DATA/'jobs'/trainer_job)
        if not owner['live'] and owner['status'] in ('complete','failed'):
            if not {p.name for p in root.glob('step[0-9][0-9][0-9][0-9][0-9][0-9].pt')}-done:break
        if not owner['live'] and owner['status'] not in ('complete','failed') and time.monotonic()-started>600:
            raise RuntimeError('Verified phase trainer missing.')
        time.sleep(30)


if __name__=='__main__':main()

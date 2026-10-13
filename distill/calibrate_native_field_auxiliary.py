"""Freeze low-field physical high-pass powers from TRAIN teacher captures only."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA,REVISION,HOLDOUT_SEEDS,atomic_json,external_path
from distill.native_field_auxiliary import low_height
from distill.native_decoded import blur
from distill.native_weight_quant import file_sha


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DATA/'crops/native-mixed-scenes-v2')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--floor-rms-m',type=float,default=.1)
    args=parser.parse_args();output=external_path(args.output)
    if output.exists():raise ValueError('Preserve prior TRAIN calibration.')
    if not np.isfinite(args.floor_rms_m) or args.floor_rms_m<=0:raise ValueError('Positive finite floor required.')
    torch.set_num_threads(4)
    root=args.dataset;manifest=json.loads((root/'manifest.json').read_text())
    kinds=('desert-plain','temperate-plain','low-plain-sea');powers={k:[] for k in kinds};rows=[]
    selected=[r for r in manifest['scenes'] if r['metadata']['split']=='train' and r['case_kind'] in kinds]
    for record in selected:
        path=root/record['file']
        with np.load(path,allow_pickle=False) as sample:
            m=json.loads(str(sample['metadata']))
            if (m['split']!='train' or m['seed'] in HOLDOUT_SEEDS or
                    m['seed']!=record['metadata']['seed'] or m['teacher_revision']!=REVISION):
                raise ValueError('Independent pinned TRAIN captures required.')
            oy=((m['y']+95)//96)*96-m['y'];ox=((m['x']+95)//96)*96-m['x']
            field=torch.from_numpy(sample['final_latents'][:,oy:oy+64,ox:ox+64].copy())[None]
            mask=sample['final_mask'][:,oy:oy+64,ox:ox+64]
            if field.shape!=(1,5,64,64) or not mask.all():raise ValueError('Complete teacher final core required.')
        height=low_height(field);land=height[...,3:-3,3:-3]>0
        high=(height-blur(height,1.))[...,3:-3,3:-3]
        power=float(high[land].square().mean()) if land.any() else None
        if power is not None:
            if not np.isfinite(power):raise FloatingPointError('Nonfinite teacher power.')
            powers[record['case_kind']].append(power)
        rows.append(dict(file=record['file'],seed=m['seed'],coarse=record['coarse_variant'],
                         kind=record['case_kind'],land_pixels=int(land.sum()),teacher_power_m2=power))
    if any(not values for values in powers.values()):raise ValueError('Every flat category needs TRAIN support.')
    scale={k:max(float(np.mean(values)),args.floor_rms_m**2) for k,values in powers.items()}
    atomic_json(output,dict(status='complete',split='train',teacher_revision=REVISION,
        manifest_sha256=file_sha(root/'manifest.json'),source_sha256=file_sha(Path(__file__)),
        objective_source_sha256=file_sha(Path(__file__).with_name('native_field_auxiliary.py')),
        flat_power_m2=scale,floor_rms_m=args.floor_rms_m,
        definition='Mean per-scene sigma1 paired high-pass power on teacher land,3pixel border excluded.',
        rows=rows,world_count=len({r['seed'] for r in rows}),coarses=sorted({r['coarse'] for r in rows})))
    print(json.dumps(dict(status='complete',scenes=len(rows),flat_power_m2=scale)),flush=True)


if __name__=='__main__':main()

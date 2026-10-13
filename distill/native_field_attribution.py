"""TRAIN-only stage attribution and deterministic native field controls."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA,atomic_json,external_path
from distill.native_basis_training import inventory
from distill.native_decoded import DecodedRegion,decoded_region_forward,blur
from distill.native_field_auxiliary import low_height
from distill.native_fusion import forward_batches,window_outputs
from distill.native_weight_quant import load_weight_quant,file_sha


@torch.no_grad()
def metrics(core,target):
    actual=low_height(core[None]);reference=low_height(target[None]);near=reference.abs()<=20
    high_actual=(actual-blur(actual,1.))[...,3:-3,3:-3]
    high_reference=(reference-blur(reference,1.))[...,3:-3,3:-3]
    land=reference[...,3:-3,3:-3]>0
    power=float(high_reference[land].square().mean()) if land.any() else 0.
    ratio=math.sqrt(float(high_actual[land].square().mean())/power) if power>1e-12 else None
    return dict(mae_m=float((actual-reference).abs().mean()),bias_m=float((actual-reference).mean()),
        near_coast_disagreement=float(((actual>0)!=(reference>0))[near].float().mean()) if near.any() else None,
        high_frequency_rms_ratio=ratio,hf_log_error=abs(math.log(max(ratio,1e-12))) if ratio is not None else None)


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--teacher-checkpoint',type=Path,default=DATA/'eval/weight-nf4-teacher-start-v2/initial.pt')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--weights',choices=('raw','ema'),default='raw')
    args=parser.parse_args();output=external_path(args.output)
    if output.exists():raise ValueError('Keep previous attribution evidence.')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    if args.weights=='raw':saved=saved|{'ema':saved['model']}
    model=load_weight_quant(saved,'cuda')
    original=torch.load(args.teacher_checkpoint,map_location='cpu',weights_only=True,mmap=True)
    teacher=load_weight_quant(original,'cuda');teacher.quantized=False
    records,root,_=inventory();generator=np.random.default_rng(2026101261);selected=[]
    for kind in ('desert-plain','temperate-plain','wet-complex-coast','low-plain-sea'):
        for coarse in ('original','coarse4','coarse2'):
            pool=[r for r in records if r['case_kind']==kind and r['coarse_variant']==coarse]
            for i in generator.choice(len(pool),size=2,replace=False):selected.append(pool[int(i)])
    rows=[];repeat=None
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight=linear_weight_window(64,'cuda',torch.float32)
    for index,record in enumerate(selected):
        region=DecodedRegion(Path(record['root'])/record['file'],'cuda')
        if region.metadata['split']!='train':raise ValueError('TRAIN attribution only.')
        with model.cached_weights():
            result=decoded_region_forward(model,region,weight)
            second=region.stages[1]
            te_velocity=forward_batches(model,second['inputs'],second['labels'],second['condition'])
            te=region.core_crop(window_outputs(second['inputs'],te_velocity,second['labels']),weight)
            if index==0:
                again=decoded_region_forward(model,region,weight)['core']
                repeat=float((again-result['core']).abs().max())
                if repeat!=0:raise ValueError('Same field is not deterministic.')
        et_velocity=forward_batches(teacher,result['inputs1'],second['labels'],second['condition'])
        et=region.core_crop(window_outputs(result['inputs1'],et_velocity,second['labels']),weight)
        tt=decoded_region_forward(teacher,region,weight)['core']
        row=dict(kind=record['case_kind'],coarse=record['coarse_variant'],seed=record['metadata']['seed'],
            file=record['file'],split='train',EE=metrics(result['core'],region.target),
            TE=metrics(te,region.target),ET=metrics(et,region.target),TT=metrics(tt,region.target))
        rows.append(row);print(json.dumps(row),flush=True)
        del result,te_velocity,te,et_velocity,et,tt,region
        if index==0:del again
    atomic_json(output,dict(status='complete',checkpoint_sha256=file_sha(args.checkpoint),
        teacher_checkpoint_sha256=file_sha(args.teacher_checkpoint),manifest_sha256=file_sha(root/'manifest.json'),
        source_sha256=file_sha(Path(__file__)),selected_weights=args.weights,step=saved['step'],rows=rows,
        repeated_field_max=repeat,benchmark=False,cudnn_deterministic=True,gpu=torch.cuda.get_device_name(),
        scope='24TRAIN lowLOD3 fields; EE=pupil,pupil; TE=stored teacher inputs,pupil; ET=pupil,teacher; TT=teacher,teacher.',
        weight_updates=0))


if __name__=='__main__':main()

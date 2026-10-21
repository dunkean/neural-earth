"""Measure teacher information missing from the one-pass64px noise interface.

Keep both central noise streams and all conditioning fixed, alter only first
noise outside that window, and compare teacherT2 endpoints. TRAIN scenes only.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from distill.common import DATA, atomic_json, external_path
from distill.native_base import teacher_source
from distill.native_basis_training import inventory
from distill.native_decoded import DecodedRegion, blur, OriginalDecoder
from distill.native_fusion import forward_batches, window_outputs, blend, refinement_inputs
from distill.decoded_loss import reconstructed_height
from distill.native_progressive_prepare import configure
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.inference.world_pipeline import gaussian_noise_patch, linear_weight_window
from terrain_inference import prepare_models


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=external_path(args.output)
    if output.exists():raise ValueError('Preserve previous locality evidence.')
    configure()
    teacher=EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    prepare_models(SimpleNamespace(coarse_model=teacher,base_model=teacher,decoder_model=teacher))
    decoder=OriginalDecoder('cuda')
    records,_,_=inventory();weight=linear_weight_window(64,'cuda',torch.float32)
    results=[]
    for index,kind in enumerate(('desert-plain','wet-complex-coast','wet-high-mountain')):
        record=next(r for r in records if r['case_kind']==kind and r['coarse_variant']=='coarse4')
        region=DecodedRegion(Path(record['root'])/record['file'],'cuda')
        seed_a=2026101471+index*7919;seed_b=seed_a+104729
        region.refresh_noise(teacher,weight,seed_a)
        first,second=region.stages
        if region.ref_origins[4]!=region.core:raise ValueError('Unexpected central refinement window.')
        reference=window_outputs(second['inputs'],second['target'],second['labels'],region.dtype)[4:5]
        reference_core=region.target.detach().clone()
        inputs=first['inputs'].clone()
        yy=torch.arange(64,device='cuda')[:,None];xx=torch.arange(64,device='cuda')[None,:]
        kept=0
        for i,(y,x) in enumerate(region.init_origins):
            inside=((yy+y>=region.core[0])&(yy+y<region.core[0]+64)&
                    (xx+x>=region.core[1])&(xx+x<region.core[1]+64))
            noise=torch.from_numpy(gaussian_noise_patch(seed_b+5819,y,x,64,64,channels=5,
                                                       tile_h=64,tile_w=64)).cuda().bfloat16()
            alternate=(torch.sin(first['labels'][i].bfloat16())*noise).float()
            inputs[i]=torch.where(inside[None],inputs[i],alternate)
            kept+=int(inside.sum())
        velocity=forward_batches(teacher,inputs.bfloat16(),first['labels'].bfloat16(),first['condition'].bfloat16())
        initial=blend(window_outputs(inputs,velocity,first['labels'],region.dtype),region.init_origins,
                      region.init_origin,192,weight)
        second_inputs=refinement_inputs(region.ref_crops(initial),region.noise,second['labels'],region.dtype)
        velocity=forward_batches(teacher,second_inputs.bfloat16(),second['labels'].bfloat16(),second['condition'].bfloat16())
        windows=window_outputs(second_inputs,velocity,second['labels'],region.dtype)
        actual=windows[4:5]
        actual_core=region.core_crop(windows,weight)
        native=[]
        for channel in range(5):
            a=actual[:,channel:channel+1];r=reference[:,channel:channel+1]
            da=(a-blur(a,1.))[...,16:48,16:48];dr=(r-blur(r,1.))[...,16:48,16:48]
            native.append(dict(channel=channel,detail_rms=float(dr.square().mean().sqrt()),
                missing_context_detail_error=float((da-dr).square().mean().sqrt()),
                relative_detail_error=float(((da-dr).square().mean()/dr.square().mean().clamp_min(1e-12)).sqrt())))
        heights=[]
        for core in (reference_core,actual_core):
            heights.append(reconstructed_height(decoder(core[None],region.decoder_noise),core[None])[...,64:-64,64:-64])
        delta=heights[1]-heights[0]
        slopes=[h.diff(dim=-1) for h in heights]
        row=dict(kind=kind,world_seed=record['metadata']['seed'],coarse='coarse4',noise_seed=seed_a,
            alternate_outer_noise_seed=seed_b,kept_initial_pixels=kept,native_channels=native,
            decoded_height_mae_m=float(delta.abs().mean()),
            decoded_relative_slope_error=float(((slopes[1]-slopes[0]).square().mean()/slopes[0].square().mean().clamp_min(.0004)).sqrt()))
        results.append(row);print(json.dumps(row),flush=True)
    atomic_json(output,dict(status='complete',gpu=torch.cuda.get_device_name(),rows=results,
        intervention='Only noise0 outside central64px changed; central noise0/noise1 and conditioning remain identical.',
        interpretation='Nonzero endpoint changes demonstrate missing input information; this is not a capacity or convergence conclusion.'))


if __name__=='__main__':main()

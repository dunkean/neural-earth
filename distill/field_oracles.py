"""Separate changed field geometry from learned weights and first-stage drift.

Teacher forcing uses stored initial-window predictions, not an invented256
teacher field. Its outer32 ring is only partially blended; the central192 is
checked against the exact captured target. Treat it as an oracle diagnostic,
not an application seam or acceptance result.
"""
import argparse
import json

import numpy as np
import torch

from distill.common import DATA,atomic_json,external_path
from distill.field_base import FieldBase
from distill.field_scene import FieldScene,field_scene_forward
from distill.native_base import NativeBase
from distill.native_decoded import DecodedRegion,OriginalDecoder,decoded_region_forward
from distill.native_fusion import blend,refinement_inputs,window_outputs
from distill.decoded_loss import reconstructed_height
from distill.terrain_shape import compare_shape
from distill.wet_mountains import ROOT,digest
from distill.wet_mountain_progress import selected_validation


def teacher_initial256(path,scene,weight):
    with np.load(path,allow_pickle=False) as sample:
        ids=np.flatnonzero(sample['labels']==sample['labels'].max())
        origins=[tuple(int(v)*32 for v in sample['contexts'][i,1:]) for i in ids]
        values={k:torch.from_numpy(sample[k][ids].copy()).cuda() for k in ('inputs','target','labels')}
    outputs=window_outputs(values['inputs'],values['target'],values['labels'],scene.dtype)
    canvas=blend(outputs,origins,scene.origin,256,weight)
    known=canvas[:,32:224,32:224]
    if not (known[-1]>0).all():raise ValueError('The recorded teacher initial192 is incomplete.')
    central=known[:-1]/known[-1:]
    error=central-scene.initial
    if float(error.square().mean())>1e-8:raise ValueError('Initial192 reconstruction changed the teacher trajectory.')
    initial=torch.where(canvas[-1:]>0,canvas[:-1]/canvas[-1:].clamp_min(1e-8),0.)
    return initial,dict(central192_mse=float(error.square().mean()),central192_max_abs=float(error.abs().max()),
                        valid_fraction=float((canvas[-1:]>0).float().mean()))


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    field=FieldBase(saved['config']).cuda().eval().requires_grad_(False);field.load_state_dict(saved['ema'])
    native=NativeBase(saved['config']).cuda().eval().requires_grad_(False);native.load_state_dict(saved['ema'])
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight=linear_weight_window(64,'cuda',torch.float32);decoder=OriginalDecoder('cuda')
    selected,_=selected_validation(ROOT,'coarse4');rows=[]
    for record in selected:
        path=ROOT/record['file'];scene=FieldScene(path,'cuda');region=DecodedRegion(path,'cuda')
        current=field_scene_forward(field,scene)
        initial,reconstruction=teacher_initial256(path,scene,weight)
        weighted=torch.cat((initial,torch.ones_like(initial[:1])),dim=0)[None]
        inputs=refinement_inputs(weighted,scene.noise[1][None],scene.labels[1:],scene.dtype)
        velocity=field(inputs,scene.labels[1:],scene.condition[None],[scene.origin],[scene.centres])
        forced=window_outputs(inputs,velocity,scene.labels[1:],scene.dtype)[0,:,64:192,64:192]
        y,x=region.core;dy,dx=y-scene.metadata['y'],x-scene.metadata['x']
        geometries=dict(field=current['core'][0,:,dy:dy+64,dx:dx+64],
            field_teacher_initial=forced[:,dy:dy+64,dx:dx+64],
            native_windows_same_weights=decoded_region_forward(native,region,weight)['core'])
        reference=reconstructed_height(decoder(region.target[None],region.decoder_noise),region.target[None])[0,0].cpu().numpy()[64:-64,64:-64]
        for name,latent in geometries.items():
            actual=reconstructed_height(decoder(latent[None],region.decoder_noise),latent[None])[0,0].cpu().numpy()[64:-64,64:-64]
            morphology=next(r for r in compare_shape(actual,reference)['topographic_position'] if r['sigma_pixels']==8.)
            rows.append(dict(seed=scene.metadata['seed'],variant=name,mae_m=float(np.abs(actual-reference).mean()),
                correlation=morphology['tpi_correlation'],recall=morphology['valley_retention'],rms=morphology['tpi_rms_ratio'],
                reconstruction=reconstruction))
    atomic_json(external_path(args.output),dict(step=saved['step'],checkpoint_sha256=digest(args.checkpoint),
        rows=rows,source_sha256=digest(__file__),gpu=torch.cuda.get_device_name(),accepted=False,
        note=__doc__))
    print(json.dumps(dict(step=saved['step'],rows=rows)),flush=True)


if __name__=='__main__':main()

"""Causal first-channel replacement diagnostic; never an acceptance evaluation."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA,atomic_json,external_path
from distill.field_base import FieldBase
from distill.field_phase import FieldPhaseBase
from distill.field_scene import FieldScene,field_scene_forward
from distill.field_oracles import teacher_initial256
from distill.native_decoded import DecodedRegion,OriginalDecoder
from distill.native_fusion import refinement_inputs,window_outputs
from distill.decoded_loss import reconstructed_height
from distill.terrain_shape import compare_shape
from distill.wet_mountains import ROOT,digest
from distill.wet_mountain_progress import selected_validation


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=external_path(args.output)
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    cls={'native-field':FieldBase,'native-field-phase':FieldPhaseBase}[saved['architecture']]
    model=cls(saved['config']).cuda().eval().requires_grad_(False);model.load_state_dict(saved['ema'])
    step=saved['step'];del saved
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight=linear_weight_window(64,'cuda',torch.float32);decoder=OriginalDecoder('cuda');rows=[]
    replacements={'channel0':[0],'other_features':[1,2,3],'height4':[4],'all_features':[0,1,2,3],'all_channels':[0,1,2,3,4]}
    for record in selected_validation(ROOT,'coarse4')[0]:
        path=ROOT/record['file'];scene=FieldScene(path,'cuda');region=DecodedRegion(path,'cuda')
        result=field_scene_forward(model,scene)
        original,reconstruction=teacher_initial256(path,scene,weight)
        y,x=region.core;dy,dx=y-scene.metadata['y'],x-scene.metadata['x']
        reference=reconstructed_height(decoder(region.target[None],region.decoder_noise),region.target[None])[0,0].cpu().numpy()[64:-64,64:-64]
        for name in ('autonomous',*replacements):
            if name=='autonomous':latent=result['core'][0]
            else:
                initial=result['initial'][0].clone()
                channels=replacements[name];initial[channels]=original[channels]
                weighted=torch.cat((initial,torch.ones_like(initial[:1])),dim=0)[None]
                inputs=refinement_inputs(weighted,scene.noise[1][None],scene.labels[1:],scene.dtype)
                velocity=model(inputs,scene.labels[1:],scene.condition[None],[scene.origin],[scene.centres])
                latent=window_outputs(inputs,velocity,scene.labels[1:],scene.dtype)[0,:,64:192,64:192]
            latent=latent[:,dy:dy+64,dx:dx+64]
            candidate=reconstructed_height(decoder(latent[None],region.decoder_noise),latent[None])[0,0].cpu().numpy()[64:-64,64:-64]
            tpi=next(r for r in compare_shape(candidate,reference)['topographic_position'] if r['sigma_pixels']==8.)
            rows.append(dict(seed=scene.metadata['seed'],variant=name,mae_m=float(np.abs(candidate-reference).mean()),
                correlation=tpi['tpi_correlation'],recall=tpi['valley_retention'],rms=tpi['tpi_rms_ratio'],
                reconstruction=reconstruction))
    report=dict(checkpoint=str(args.checkpoint),checkpoint_sha256=digest(args.checkpoint),step=step,rows=rows,
        source_sha256=digest(__file__),oracle_source_sha256=digest(Path(__file__).with_name('field_oracles.py')),
        gpu=torch.cuda.get_device_name(),accepted=False,
        note='Causal diagnostic only: same EMA and noises, one or several first channels replaced by recorded teacher. Outer32 teacher initial ring partially blended; central192 checked exact. Selection uses autonomous variant only.')
    atomic_json(output,report);output.with_suffix('.txt').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(output),step=step,rows=rows)),flush=True)


if __name__=='__main__':main()

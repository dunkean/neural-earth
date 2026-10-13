"""Read-only coast-loss gradient probe on native weight-QAT outputs.

This does not fit weights or change the running training recipe. It verifies
that the existing land/sea objective reaches final base latents, including
through the frozen original decoder, on the two coastal validation scenes.
"""
import argparse
import gc
import json
from pathlib import Path

import torch

from distill.common import DATA, atomic_json, external_path
from distill.decoded_loss import reconstructed_height, shore_loss
from distill.native_decoded import DecodedRegion, OriginalDecoder, decoded_region_forward
from distill.native_weight_quant import file_sha, load_weight_quant


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--coarses',nargs='+',default=['coarse4','coarse2'])
    args=parser.parse_args();output=external_path(args.output)
    if (output/'report.json').exists():raise ValueError('Keep existing probe evidence immutable.')
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    model=load_weight_quant(saved,'cuda');decoder=OriginalDecoder('cuda')
    root=DATA/'crops/native-mixed-scenes-v2';manifest=json.loads((root/'manifest.json').read_text())
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight=linear_weight_window(64,'cuda',torch.float32);rows=[]
    for coarse in args.coarses:
        for kind in ('wet-complex-coast','low-plain-sea'):
            available=[r for r in manifest['scenes'] if r['metadata']['split']=='val' and
                       r['case_kind']==kind and r['coarse_variant']==coarse]
            record=sorted(available,key=lambda r:(r['metadata']['seed'],r['file']))[0]
            region=DecodedRegion(root/record['file'],'cuda')
            with torch.no_grad(),model.cached_weights():
                candidate=decoded_region_forward(model,region,weight)['core'].detach()
                target=region.target.detach()
                reference_fine=reconstructed_height(decoder(target[None],region.decoder_noise),target[None])[...,64:-64,64:-64]
            for lod in (3,0):
                core=candidate.clone().requires_grad_(True)
                if lod==3:
                    q=core[None,4:5]*38.6-31.4;t=target[None,4:5]*38.6-31.4
                    actual=q.sign()*q.square();reference=t.sign()*t.square()
                else:
                    actual=reconstructed_height(decoder(core[None],region.decoder_noise),core[None])[...,64:-64,64:-64]
                    reference=reference_fine
                loss,disagreement=shore_loss(actual,reference,torch.ones_like(reference))
                latent_gradient,height_gradient=torch.autograd.grad(loss,(core,actual))
                if not torch.isfinite(latent_gradient).all():raise FloatingPointError('Nonfinite shore gradient.')
                near=reference.abs()<=20
                wrong=near & ((actual.detach()>0)!=(reference>0))
                corrective=(height_gradient.detach()*(actual.detach()-reference))>0
                count=int(wrong.sum())
                if count and not bool(corrective[wrong].all()):raise ValueError('Wrong-sign gradients do not correct towards the teacher.')
                if near.any() and float(latent_gradient.norm())==0:raise ValueError('No gradient reaches base latents.')
                row=dict(step=saved['step'],coarse=coarse,kind=kind,lod=lod,
                    seed=record['metadata']['seed'],scene=record['file'],
                    height_mae_m=float((actual.detach()-reference).abs().mean()),
                    shore_loss=float(loss.detach()),balanced_near_coast_disagreement=float(disagreement),
                    near_zero_pixels=int(near.sum()),wrong_sign_pixels=count,
                    wrong_sign_height_gradients_corrective=True,
                    latent_gradient_norm=float(latent_gradient.norm()),
                    latent_gradient_channel_norms=[float(v) for v in latent_gradient.flatten(1).norm(dim=1)],
                    peak_bytes=torch.cuda.max_memory_allocated())
                rows.append(row);print(json.dumps(row),flush=True)
                del actual,loss,core,latent_gradient,height_gradient
            del region,candidate,target,reference_fine;gc.collect();torch.cuda.empty_cache()
    if any(p.grad is not None for p in model.parameters()) or any(p.grad is not None for p in decoder.parameters()):
        raise ValueError('Probe must leave all model weights frozen.')
    atomic_json(output/'report.json',dict(status='complete',checkpoint=str(args.checkpoint),
        checkpoint_sha256=file_sha(args.checkpoint),source_sha256=file_sha(Path(__file__)),
        objective_source_sha256=file_sha(Path(__file__).with_name('decoded_loss.py')),
        gpu=torch.cuda.get_device_name(),rows=rows,band_m=20.,temperature_m=2.,
        weight_updates=0,training_modified=False,
        limitation='Gradient through final base latents; no fitted improvement or full-tile coast proof.'))


if __name__=='__main__':main()

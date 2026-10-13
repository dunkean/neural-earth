"""Audit the physical native trainer against the actual original decoder/runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from distill.common import DATA,HOLDOUT_SEEDS,REPO,atomic_json
from distill.native_decoded import DecodedRegion,OriginalDecoder,decoded_region_forward,decoded_loss
from distill.decoded_loss import reconstructed_height
from distill.native_base import NativeBase
from distill.teacher import Worlds


def channel_statistics(root):
    sums=np.zeros(5);squares=np.zeros(5);count=0;scenes=[]
    for path in sorted(root.glob('train-*.npz')):
        with np.load(path,allow_pickle=False) as sample:
            metadata=json.loads(str(sample['metadata']))
            if metadata['split']!='train' or metadata['seed'] in HOLDOUT_SEEDS:
                raise ValueError('Global latent variance must use train worlds only.')
            mask=sample['final_mask'][0].astype(bool)
            values=sample['final_latents'][:,mask].astype(np.float64)
            sums+=values.sum(1);squares+=(values**2).sum(1);count+=values.shape[1]
            DecodedRegion(path)  # Require complete aligned geometry and all34 calls.
            scenes.append(path.name)
    mean=sums/count
    return dict(split='train',count_pixels=count,scenes=scenes,mean=mean.tolist(),
        variance=np.maximum(squares/count-mean**2,1e-3).tolist(),
        dataset_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DATA/'crops/native-scenes-v1')
    parser.add_argument('--source',type=Path,default=DATA/'ckpt/native128x2-pilot/best.pt')
    parser.add_argument('--output',type=Path,default=DATA/'eval/native-decoded-probe')
    parser.add_argument('--control-index',type=int,default=0,
                        help='Separate validation scene for the runtime-proxy gate.')
    parser.add_argument('--controls-only',action='store_true',
                        help='Validate the runtime proxy without another optimizer smoke.')
    args=parser.parse_args()
    # The verification bootstrap changes cwd; import it before terrain_app
    # selects the upstream geography-data directory in Worlds.__init__.
    from tools.verification.compare_base_variants import Cast,load_copy
    args.output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    if not args.controls_only:
        stats=channel_statistics(args.dataset)
        atomic_json(args.output/'channel-statistics.json',stats)
        if len(stats['scenes'])<150:
            raise RuntimeError('Insufficient complete aligned training scenes.')
    region=DecodedRegion(sorted(args.dataset.glob('val-*.npz'))[args.control_index],'cuda')
    worlds=Worlds(region.metadata['seed'])
    world=worlds.create(region.metadata['seed'],region.metadata['profile'])
    decoder=OriginalDecoder('cuda',model=world.decoder_model)
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
    from terrain_inference import _decoder_batch
    scheduler=EDMDPMSolverMultistepScheduler(sigma_min=.002,sigma_max=80,sigma_data=.5)
    t=torch.atan(scheduler.sigmas[0]/.5)
    weight=linear_weight_window(512,'cuda',torch.float32)
    ctx=(0,region.core[0]*8//384,region.core[1]*8//384)
    latents=torch.cat((region.target,torch.ones_like(region.target[-1:])))
    with torch.no_grad():
        expected=_decoder_batch(world,[ctx],[latents],scheduler,weight,[t],512,384)[0]
        expected=expected[:1]/expected[-1:]
        raw=decoder(region.target[None],region.decoder_noise)
        node_max=float((raw[0]-expected).abs().max())
        proxy=reconstructed_height(raw,region.target[None])[0,0]
        y,x=region.core[0]*8,region.core[1]*8
        full=world.get(y+64,x+64,y+448,x+448,with_climate=False)['elev'].float()
        proxy_mae=float((proxy-full).abs().mean())
        weighted=world.latents[:,region.core[0]:region.core[0]+64,
                                  region.core[1]:region.core[1]+64].float()
        runtime_latents=weighted[:-1]/weighted[-1:]
        runtime_proxy=reconstructed_height(decoder(runtime_latents[None],region.decoder_noise),
                                           runtime_latents[None])[0,0]
        diagnostics=dict(latent_channel_max_abs=(runtime_latents-region.target).abs().flatten(1).max(1).values.cpu().tolist(),
            stored_proxy_vs_runtime_proxy_mae_m=float((proxy-runtime_proxy).abs().mean()),
            runtime_proxy_vs_full_mae_m=float((runtime_proxy-full).abs().mean()),
            central256_runtime_proxy_vs_full_mae_m=float((runtime_proxy-full)[64:-64,64:-64].abs().mean()),
            central128_runtime_proxy_vs_full_mae_m=float((runtime_proxy-full)[128:-128,128:-128].abs().mean()))
        fpworld=worlds.create(region.metadata['seed'],region.metadata['profile'])
        fpworld.base_model=Cast(load_copy('base_model',torch.float32),torch.float32)
        fpheight=fpworld.get(y+64,x+64,y+448,x+448,with_climate=False)['elev'].float()
        floor=float((fpheight-full).abs().mean())
    control=dict(status='complete',gpu=torch.cuda.get_device_name(),file=region.metadata,core=region.core,
        decoder_node_max_abs=node_max,proxy_vs_full_mae_m=proxy_mae,fp32base_floor_m=floor,
        # The capture contains the4090 coarse/base field. A new5090 world can
        # have different conditioning and native batch groupings. Audit this
        # reconstruction on the SAME runtime latents, rather than attribute
        # the stored-to-current field drift to the original decoder proxy.
        runtime_proxy_vs_full_mae_m=diagnostics['runtime_proxy_vs_full_mae_m'],
        proxy_within_floor=diagnostics['runtime_proxy_vs_full_mae_m']<=max(floor,1.),
        control_basis='same runtime latents, original decoder, full pipeline',
        diagnostics=diagnostics,accepted=False)
    atomic_json(args.output/'decoder-control.json',control)
    print(json.dumps(control),flush=True)
    teacher=world.base_model
    world.close();fpworld.close()
    del fpworld,fpheight,full,proxy,worlds,world,raw,expected
    if node_max>1e-5 or not control['proxy_within_floor']:
        raise RuntimeError('Decoded proxy failed the runtime control; investigate before training.')
    if args.controls_only:
        return
    saved=torch.load(args.source,map_location='cpu',weights_only=True)
    model=NativeBase(saved['config']).cuda().eval();model.load_state_dict(saved['ema']);del saved
    region=DecodedRegion(sorted(args.dataset.glob('train-*.npz'))[0],'cuda')
    weight=linear_weight_window(64,'cuda',torch.float32)
    variance=torch.tensor(stats['variance'],device='cuda')
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5,fused=True)
    rows=[]
    for step in range(2):
        started=time.perf_counter();optimizer.zero_grad(set_to_none=True)
        result=decoded_region_forward(model,region,weight,checkpointed=True)
        result['velocity0'].retain_grad();result['velocity1'].retain_grad()
        loss,terms=decoded_loss(result,region,decoder,teacher,variance)
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        gradients=[v.grad.abs().flatten(1).sum(1).detach().cpu().tolist()
                   for v in (result['velocity0'],result['velocity1'])]
        if any(min(values)<=0 or not np.isfinite(values).all() for values in gradients):
            raise RuntimeError('Decoded native loss did not reach all34 windows.')
        optimizer.step();torch.cuda.synchronize()
        row=dict(step=step+1,seconds=time.perf_counter()-started,loss=float(loss.detach()),
            terms={k:float(v.detach()) for k,v in terms.items()},grad_norm=float(norm),
            window_gradient_sums=gradients,peak_bytes=torch.cuda.max_memory_allocated())
        print(json.dumps(row),flush=True);rows.append(row)
        del result,loss,terms
    atomic_json(args.output/'gpu-smoke.json',dict(status='complete',gpu=torch.cuda.get_device_name(),rows=rows,
        accepted=False,source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        source_digests={name:hashlib.sha256((REPO/name).read_bytes()).hexdigest()
                       for name in ('distill/native_decoded.py','distill/native_decoded_probe.py')}))


if __name__=='__main__':
    main()

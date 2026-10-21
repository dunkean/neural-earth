"""Full original-base PTQ diagnostic, preserving architecture and two-step solver.

Three-bit raw MP weights use a blockwise Gaussian codebook. Original MPConv
normalization remains in place. Packed bytes are a storage proof only; this
module does not prove browser execution, speed, or quality acceptance.
"""
import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from distill.common import REPO, REVISION, atomic_json, atomic_write, external_path
from distill.native_base import teacher_source
from distill.native_weight_quant import file_sha


CENTERS = (-2.1519457,-1.3439093,-.7560053,-.2450942,.2450942,.7560053,1.3439093,2.1519457)


def pack3(codes):
    a=np.asarray(codes,dtype=np.uint8).reshape(-1)
    if len(a)%8 or np.any(a>7):raise ValueError('Eight three-bit codes per24-bit word required.')
    groups=a.reshape(-1,8).astype(np.uint32)
    word=sum(groups[:,i]<<(3*i) for i in range(8))
    return np.stack(((word&255),((word>>8)&255),((word>>16)&255)),axis=1).astype(np.uint8).tobytes()


def unpack3(payload):
    a=np.frombuffer(payload,dtype=np.uint8)
    if len(a)%3:raise ValueError('Incomplete24-bit word.')
    a=a.reshape(-1,3).astype(np.uint32)
    word=a[:,0]|(a[:,1]<<8)|(a[:,2]<<16)
    return np.stack([(word>>(3*i))&7 for i in range(8)],axis=1).astype(np.uint8).reshape(-1)


@torch.no_grad()
def encode3(weight,block=512):
    if block<16 or block%8:raise ValueError('Block must be a multiple of8 and at least16.')
    count=weight.numel()
    padded=F.pad(weight.detach().float().reshape(-1),(0,(-count)%block)).reshape(-1,block)
    centers=padded.new_tensor(CENTERS)
    boundaries=(centers[:-1]+centers[1:])*.5
    scale=padded.square().mean(dim=1).sqrt().clamp_min(1e-12)
    for _ in range(4):
        codes=torch.bucketize((padded/scale[:,None]).contiguous(),boundaries)
        values=centers[codes]
        scale=(padded*values).sum(dim=1)/values.square().sum(dim=1).clamp_min(1e-12)
    # The bytes storeFP16 scales;use exactly these for the quality probe too.
    scale=scale.half().float()
    codes=torch.bucketize((padded/scale[:,None].clamp_min(1e-30)).contiguous(),boundaries)
    actual=(centers[codes]*scale[:,None]).reshape(-1)[:count].reshape(weight.shape)
    return codes.to(torch.uint8).reshape(-1),scale,actual


@torch.no_grad()
def quantize_and_pack(model,output,block=512):
    output.mkdir(parents=True,exist_ok=False)
    rows,offset={},0
    first='enc.'+next(iter(model.enc))+'.weight'
    sensitive={first,'out_conv.weight','out_gain'}
    parameters=dict(model.named_parameters())
    def write(handle):
        nonlocal offset
        for name,value in model.state_dict().items():
            original=value.detach().float()
            row=dict(shape=list(value.shape),offset=offset)
            if name in parameters and value.ndim in (2,4) and name not in sensitive:
                codes,scale,actual=encode3(original,block)
                payload=pack3(codes.cpu().numpy())
                scales=scale.cpu().numpy().astype('<f2').tobytes()
                row.update(dtype='gaussian3-raw-mp',bytes=len(payload),scale_offset=offset+len(payload),
                    scale_count=scale.numel(),scale_bytes=len(scales),block=block,
                    relative_weight_mse=float((actual-original).square().sum()/original.square().sum().clamp_min(1e-30)))
                handle.write(payload);handle.write(scales)
                value.copy_(actual.to(value.dtype))
                offset+=len(payload)+len(scales)
            else:
                # Scalars and Fourier buffers retain their originalFP32 precision.
                payload=original.cpu().numpy().astype('<f4').tobytes()
                row.update(dtype='float32',bytes=len(payload))
                handle.write(payload);offset+=len(payload)
            rows[name]=row
    atomic_write(output/'weights.bin',write)
    manifest=dict(schema=1,architecture='original-edm-unet-raw-mp3',config=dict(model.config),
        tensors=rows,codebook=list(CENTERS),block=block,file='weights.bin',bytes=(output/'weights.bin').stat().st_size,
        sha256=file_sha(output/'weights.bin'),teacher_revision=REVISION,
        teacher_weights_sha256=file_sha(teacher_source()/'diffusion_pytorch_model.safetensors'),
        normalization='Reconstruct raw weights;retain original global MPConv normalize and gain;do not treat as effective kernels.',
        scope='Storage and native quality diagnostic;browser latency/acceptance unproven.',accepted=False)
    atomic_json(output/'manifest.json',manifest)
    if manifest['bytes']+(output/'manifest.json').stat().st_size>=100000000:
        raise ValueError('Packed model plus manifest exceeds100MB.')
    return manifest


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=3)
    args=parser.parse_args()
    output=external_path(args.output)
    if output.exists() or args.limit<1:raise ValueError('Fresh bounded diagnostic required.')
    output.mkdir(parents=True)
    torch.set_num_threads(8)
    from distill.teacher import Worlds
    from distill.student import load_student
    from distill.inference import install
    from distill.tile40_conditioning import CLIMATES,RELIEFS,SketchConditioning
    from distill.tile40_evaluate import compare_height
    from distill.tile40_reference_data import guided_reference,resample_tile
    from distill.native_progress_images import shade
    plan=json.loads((args.dataset/'manifest.json').read_text())
    cases=sorted((c for c in plan['cases'] if c['split']=='val'),key=lambda c:(c['seed']//10,list(CLIMATES).index(c['climate']),RELIEFS.index(c['relief'])))[:args.limit]
    worlds=Worlds(cases[0]['seed'])
    base=worlds.loaded.base_model
    from terrain_cuda_graphs import CudaGraphModel
    while isinstance(base,CudaGraphModel):
        base=base.model
    worlds.loaded.base_model=base
    packed=quantize_and_pack(base,output/'base3')
    coarse,_=load_student(REPO/'distill/weights/coarse2.pt','cuda')
    rows=[]
    for case in cases:
        source=args.dataset/case['file']
        if file_sha(source)!=json.loads(source.with_suffix('.json').read_text())['file_sha256']:
            raise ValueError('Frozen VAL changed.')
        with np.load(source,allow_pickle=False) as saved:
            elevation,reference=saved['sketch_m'].copy(),saved['height_m'].copy()
        world=worlds.create(case['seed'],'natural')
        install(world,dict(coarse=coarse));world.bind()
        world.synthetic_map_factory=SketchConditioning(elevation,case['climate'])
        began=time.monotonic()
        try:
            native=world.get(-256,-256,1600,1600,with_climate=False)['elev']
            if isinstance(native,torch.Tensor):native=native.float().cpu().numpy()
            height=resample_tile(guided_reference(native,elevation)['native_height'])
            torch.cuda.synchronize()
            metrics=compare_height(height,reference)
            rows.append(dict(case=case,metrics=metrics,native_seconds=time.monotonic()-began))
            atomic_write(output/case['file'],lambda f:np.savez_compressed(f,height_m=height,reference_height_m=reference))
            left,right=shade(reference,40000/1334,1.5),shade(height,40000/1334,1.5)
            from PIL import Image
            pair=Image.new('RGB',(1024,512))
            pair.paste(left.resize((512,512)),(0,0));pair.paste(right.resize((512,512)),(512,0))
            atomic_write(output/Path(case['file']).with_suffix('.png'),lambda f:pair.save(f,format='PNG'))
            atomic_json(output/'report.json',dict(status='running',rows=rows,packed_bytes=packed['bytes'],
                scope='Full original base architecture,3bit storage,original solver+decoder;no teacher latent inputs;native GPU diagnostic only.'))
            print(json.dumps(dict(case=case,mae=metrics['height_mae_m'],amplitudes=[b['amplitude_ratio'] for b in metrics['bands']])),flush=True)
        finally:world.close()
    atomic_json(output/'report.json',dict(status='complete',rows=rows,packed_bytes=packed['bytes'],
        scope='Full original base architecture,3bit storage,original solver+decoder;no teacher latent inputs;browser execution and complete product remain unproven.'))


if __name__=='__main__':main()

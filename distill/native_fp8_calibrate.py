"""Exact BF16 activation histograms on64 independent TRAIN-only batches."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA,REPO,REVISION,atomic_json,atomic_write,external_path
from distill.native_base import teacher_source
from distill.native_fp8 import NativeFP8Base
from distill.native_basis_training import WindowBank,inventory
from distill.wet_mountains import digest,ROOT
from terrain_diffusion.models.edm_unet import EDMUnet2D


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DATA/'eval/native-fp8-calibration')
    args=parser.parse_args();output=external_path(args.output)
    if (output/'status.json').exists():raise ValueError('Preserve existing calibration; use a fresh root.')
    records,root,manifest=inventory();torch.set_num_threads(8);torch.manual_seed(2026101214)
    torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    teacher=EDMUnet2D.from_pretrained(str(teacher_source())).eval()
    model=NativeFP8Base(dict(teacher.config)).cuda().eval();model.load_teacher(teacher.state_dict());del teacher
    contract=dict(teacher_revision=REVISION,teacher_weights_sha256=digest(teacher_source()/'diffusion_pytorch_model.safetensors'),
        manifest_sha256=digest(root/'manifest.json'),mountain_manifests={str(p):digest(p/'manifest.json') for p in
          (ROOT,DATA/'crops/native-wet-mountain-train-extension-v1')},
        sources_sha256={n:digest(REPO/'distill'/n) for n in ('native_fp8.py','native_fp8_calibrate.py','native_basis_training.py','native_base.py')},
        quantized_layers=model.quantized_layers,batches=64,batch=8,percentile=.9999,
        sampler='same25%wet/15%oldrare/60%general,8distinct TRAIN worlds,4calls per phase,3coarse branches per phase',
        method='exact absolute BF16 bit-pattern histogram,32768bins; no sampled or VAL-derived ranges',seed=2026101214)
    atomic_json(output/'contract.json',contract)
    histograms={n:np.zeros(32768,np.int64) for n in model.quantized_layers};handles=[]
    for name in model.quantized_layers:
        def collect(layer,inputs,name=name):
            value=inputs[0].detach()
            if value.dtype!=torch.bfloat16:raise ValueError('Calibration must preserve native BF16 activations.')
            bits=value.contiguous().view(torch.int16).to(torch.int64).bitwise_and(0x7fff).reshape(-1)
            counts=torch.bincount(bits,minlength=32768).cpu().numpy()
            if counts[32640:].sum():raise FloatingPointError('Nonfinite teacher activation.')
            histograms[name]+=counts
        handles.append(model.net.get_submodule(name).register_forward_pre_hook(collect))
    bank=WindowBank(records);calibration_rows=[]
    try:
        for step in range(64):
            generator=np.random.default_rng(np.random.SeedSequence([contract['seed'],step]))
            batch,selected=bank.batch(records,generator,step)
            model(batch['inputs'],batch['labels'],[batch['condition']])
            calibration_rows.append([dict(seed=r['metadata']['seed'],coarse=r['coarse_variant'],kind=r['case_kind']) for _,r in selected])
            if (step+1)%8==0:
                atomic_json(output/'status.json',dict(status='calibrating',batches=step+1,target_batches=64))
                print(json.dumps(dict(batches=step+1)),flush=True)
    finally:
        for handle in handles:handle.remove()
    scales={}
    for name,counts in histograms.items():
        total=int(counts.sum());index=int(np.searchsorted(np.cumsum(counts),math.ceil(.9999*total)))
        threshold=float(torch.tensor([index],dtype=torch.int16).view(torch.bfloat16).float()[0])
        scale=max(threshold/448,1e-12)
        model.net.get_submodule(name)._fp8_scale.fill_(scale)
        scales[name]=dict(scale=scale,threshold=threshold,samples=total,clipped_fraction=float(counts[index+1:].sum()/total))
    model.set_quantized(True)
    state=dict(config=model.config,ema={n:v.cpu() for n,v in model.state_dict().items()},architecture='native-fp8',step=0,
        quantized_layers=model.quantized_layers,contract=contract,teacher_revision=REVISION,
        teacher_weights_sha256=contract['teacher_weights_sha256'],accepted=False,evaluation_only=True,
        computation='simulated E4M3, BF16 convolutions; no accelerated FP8 inference claim')
    atomic_write(output/'initial.pt',lambda f:torch.save(state,f))
    atomic_json(output/'activation-scales.json',dict(contract=contract,scales=scales,train_batches=calibration_rows))
    atomic_json(output/'status.json',dict(status='complete',batches=64,layers=len(scales),gpu=torch.cuda.get_device_name(),accepted=False))
    print(json.dumps(dict(status='complete',layers=len(scales))),flush=True)


if __name__=='__main__':main()

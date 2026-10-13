"""Freeze raw40k master-gradient scales for the exact second-stage auxiliary."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA,atomic_json,external_path
from distill.native_base import teacher_source
from distill.native_basis_training import inventory
from distill.native_auxiliary_training import (BoundedQATBank,auxiliary_record,auxiliary_terms,
    preconditioners,gradient_norm)
from distill.native_weight_quant import file_sha
from distill.native_weight_quant_resume import restore_training,restore_rng
from distill.native_weight_quant_train import step_loss
from terrain_diffusion.models.edm_unet import EDMUnet2D


def cosine(a,b):
    scale=gradient_norm(a)*gradient_norm(b)
    return float(sum((x*y).sum() for x,y in zip(a,b) if x is not None and y is not None)/scale) if scale else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--powers',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--regions',type=int,default=32)
    args=parser.parse_args();output=external_path(args.output)
    if output.exists():raise ValueError('Keep fixed calibration evidence.')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    source=torch.load(args.source,map_location='cpu',weights_only=True,mmap=True)
    records,root,_=inventory();powers=json.loads(args.powers.read_text())
    if powers['split']!='train' or powers['manifest_sha256']!=file_sha(root/'manifest.json'):
        raise ValueError('Frozen matched TRAIN powers required.')
    model,ema,optimizer=restore_training(source);del ema
    teacher=EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    from types import SimpleNamespace
    from terrain_inference import prepare_models
    prepare_models(SimpleNamespace(coarse_model=teacher,base_model=teacher,decoder_model=teacher))
    restore_rng(source);params=list(model.parameters());pre=preconditioners(optimizer,params)
    variance=torch.tensor(json.loads((DATA/'eval/native-decoded-probe/channel-statistics.json').read_text())['variance'],device='cuda')
    bank=BoundedQATBank();rows=[];regular_norms=[];regular_pre=[];ratios={'coast':[],'flat':[]}
    torch.cuda.reset_peak_memory_stats()
    for index in range(args.regions):
        gradients=[];measure=[]
        for offset in (0,1):
            step=source['step']+50000+index*2+offset
            generator=np.random.default_rng(np.random.SeedSequence([source['contract']['seed'],step]))
            with torch.no_grad(),model.cached_weights():
                batch,_=bank.batch(records,generator,step,model,on_policy=offset==0)
            loss,_,_=step_loss(model,teacher,batch,variance,source['feature_weight'])
            g=torch.autograd.grad(loss,params,allow_unused=True);gradients.append(g)
            pair=(gradient_norm(g),gradient_norm(g,pre));measure.append(pair)
            regular_norms.append(pair[0]);regular_pre.append(pair[1]);del loss
        reference=np.mean(measure,axis=0);regular_cos=cosine(*gradients)
        # Only one reference vector retained while the auxiliary graph is live.
        on=gradients.pop(0);del on
        record,noise_seed=auxiliary_record(records,source['contract']['seed'],source['step']+1+index*2)
        terms=auxiliary_terms(model,teacher,record,noise_seed,powers)
        entry=dict(index=index,kind=record['case_kind'],coarse=record['coarse_variant'],seed=record['metadata']['seed'],
            regular_norm=float(reference[0]),regular_preconditioned_norm=float(reference[1]),regular_regular_cosine=regular_cos,
            class_pixels=terms['coast_counts'])
        for name,key in (('coast','coast'),('flat','flat_highpass')):
            g=torch.autograd.grad(terms[key],params,allow_unused=True,retain_graph=name=='coast')
            raw=gradient_norm(g);scaled=gradient_norm(g,pre)
            ratio=max(raw/max(reference[0],1e-30),scaled/max(reference[1],1e-30))
            if not np.isfinite(ratio):raise FloatingPointError('Nonfinite calibration gradient.')
            if raw>1e-12:ratios[name].append(ratio)
            entry[name]=dict(loss=float(terms[key].detach()),norm=raw,preconditioned_norm=scaled,
                             ratio=ratio,off_policy_cosine=cosine(gradients[0],g))
            del g
        rows.append(entry);print(json.dumps(entry),flush=True)
        del gradients,terms;torch.cuda.empty_cache()
    if any(not r for r in ratios.values()):raise ValueError('Both objectives need active TRAIN support.')
    coefficients={k:.25/float(np.median(r)) for k,r in ratios.items()}
    atomic_json(output,dict(status='complete',source_sha256=file_sha(args.source),powers_sha256=file_sha(args.powers),
        source_code_sha256=file_sha(Path(__file__)),helper_sha256=file_sha(Path(__file__).with_name('native_auxiliary_training.py')),
        coefficients=coefficients,reference_norm=float(np.median(regular_norms)),
        reference_preconditioned_norm=float(np.median(regular_pre)),rows=rows,regions=args.regions,
        peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),
        gpu=torch.cuda.get_device_name(),weight_updates=0,recipe='raw source,root-space KL,min16,newnoise,teacher on pupil initial state,cap.5'))


if __name__=='__main__':main()

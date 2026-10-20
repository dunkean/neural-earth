"""TRAIN-only residual least-squares fit retaining true body masters and Adam."""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from distill.common import DATA, REPO, atomic_json, atomic_write, external_path
from distill.native_base import teacher_source
from distill.native_basis_training import inventory
from distill.native_decoded import DecodedRegion
from distill.native_fusion import window_outputs
from distill.native_onepass import noise_windows, ARCHITECTURE as BODY_ARCHITECTURE
from distill.native_onepass_runtime import OnePassRuntime
from distill.native_onepass_residual import ResidualOnePassBase, ARCHITECTURE, extend_optimizer_slots
from distill.native_progressive_prepare import configure
from distill.native_weight_quant import file_sha
from terrain_diffusion.models.edm_unet import EDMUnet2D
from types import SimpleNamespace
from terrain_inference import prepare_models


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--records',type=int,default=24)
    args=parser.parse_args();output=external_path(args.output)
    if output.exists():raise ValueError('Preserve prior preparation.')
    if args.records<12:raise ValueError('Balanced independent TRAIN calibration required.')
    configure();output.mkdir(parents=True)
    # Open the immutable source inode even if the trainer atomically replaces latest.
    shutil.copyfile(args.checkpoint,output/'parent-full.pt')
    source=torch.load(output/'parent-full.pt',map_location='cpu',weights_only=True,mmap=True)
    if source.get('architecture')!=BODY_ARCHITECTURE or 'model' not in source or 'optimizer' not in source:
        raise ValueError('True one-pass body masters and optimizer required.')
    body_source=dict(source,ema=source['model'])
    body=OnePassRuntime(body_source,'cuda')
    model=ResidualOnePassBase(source['config'],source['gain_anchor'])
    model.load_state_dict(dict(source['model'],**{'residual.weight':torch.zeros(5,10,1,1)}))
    parent_count=len(source['optimizer']['param_groups'][0]['params'])
    new_count=len([p for p in model.parameters() if p.requires_grad])
    optimizer=extend_optimizer_slots(source['optimizer'],parent_count,new_count)
    teacher=EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    prepare_models(SimpleNamespace(coarse_model=teacher,base_model=teacher,decoder_model=teacher))
    records,_,_=inventory();generator=np.random.default_rng(2026101483)
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    blend_weight=linear_weight_window(64,'cuda',torch.float32)
    covariance=torch.zeros(10,10,device='cuda',dtype=torch.float64)
    cross=torch.zeros(10,5,device='cuda',dtype=torch.float64)
    used=set();rows=[];calibration=[]
    kinds=('wet-high-mountain','wet-complex-coast','low-plain-sea','desert-plain','temperate-plain',None)
    for index in range(args.records):
        kind=kinds[index%6];coarse=('coarse4','coarse2','original')[(index//6)%3]
        pool=[r for r in records if r['case_kind']==kind and r['coarse_variant']==coarse and r['metadata']['seed'] not in used]
        record=pool[int(generator.integers(len(pool)))];used.add(record['metadata']['seed'])
        region=DecodedRegion(Path(record['root'])/record['file'],'cuda')
        seed=int(generator.integers(1,2**31-5821));region.refresh_noise(teacher,blend_weight,seed)
        noise=noise_windows(seed,region.ref_origins,'cuda')
        condition=region.stages[1]['condition']
        prediction=body(noise,condition)
        stage=region.stages[1]
        target=window_outputs(stage['inputs'],stage['target'],stage['labels'],region.dtype)
        x=noise.permute(0,2,3,1).reshape(-1,10).double()
        residual=(target-prediction).permute(0,2,3,1).reshape(-1,5).double()
        covariance+=x.T@x;cross+=x.T@residual
        row=dict(kind=kind,coarse=coarse,world_seed=record['metadata']['seed'],noise_seed=seed,
                 body_mse=float((target-prediction).square().mean()))
        rows.append(row)
        # Bound retained examples; the sufficient statistics cover all records.
        if index<6:calibration.append((noise.cpu(),condition.cpu(),prediction.cpu(),target.cpu()))
        print(json.dumps(row),flush=True)
    fitted=torch.linalg.solve(covariance+torch.eye(10,device='cuda',dtype=torch.float64)*10.,cross).T.float().reshape(5,10,1,1)
    if not torch.isfinite(fitted).all():raise FloatingPointError('Nonfinite residual fit.')
    model.residual.weight.copy_(fitted.cpu())
    checks=[]
    for noises,condition,prediction,target in calibration:
        actual=prediction+torch.nn.functional.conv2d(noises,fitted.cpu())
        checks.append(dict(body_mse=float((prediction-target).square().mean()),
                          residual_mse=float((actual-target).square().mean())))
    # The head is fit on TRAIN; the next pairedVAL plates decide its usefulness.
    saved={name:value for name,value in source.items() if name not in ('model','ema','optimizer','contract','rng','cuda_rng')}
    saved.update(architecture=ARCHITECTURE,parent_step=source['step'],parent_sha256=file_sha(output/'parent-full.pt'),step=0,
                 model=model.state_dict(),ema={name:value.float() if name in ('net.out_gain','residual.weight') else value.bfloat16()
                      for name,value in model.state_dict().items()},optimizer=optimizer,selected_weights='raw-initialized-ema',accepted=False)
    atomic_write(output/'initial.pt',lambda handle:torch.save(saved,handle))
    atomic_json(output/'proof.json',dict(status='complete',initial_sha256=file_sha(output/'initial.pt'),parent_step=source['step'],
                parent_sha256=saved['parent_sha256'],calibration_split='train',calibration_records=rows,fit_checks=checks,
                fitted_matrix=fitted[:,:,0,0].cpu().tolist(),parent_optimizer_slots=parent_count,new_optimizer_slots=new_count,
                parameters=sum(p.numel() for p in model.parameters()),
                code_sha256={name:file_sha(REPO/'distill'/name) for name in ('native_onepass.py','native_onepass_residual.py')},
                note='Retained FP32 body masters/Adam;50FP32 residual parameters; pairedVAL quality and packed QAT remain unproven.'))
    print(json.dumps(dict(status='complete',parent_step=source['step'],residual_parameters=50)),flush=True)


if __name__=='__main__':main()

"""TRAIN-only real NF4 parameter gradients for prospective coast/flat objectives."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA,atomic_json,external_path
from distill.native_base import teacher_source
from distill.native_basis_training import inventory
from distill.native_decoded import DecodedRegion,decoded_region_forward
from distill.native_field_auxiliary import low_field_terms
from distill.native_fp8_training import QATBank
from distill.native_weight_quant import load_weight_quant,file_sha
from distill.native_weight_quant_train import step_loss
from terrain_diffusion.models.edm_unet import EDMUnet2D


def norm(grads):
    return float(sum(g.detach().float().square().sum() for g in grads if g is not None).sqrt())


def cosine(first,second):
    denominator=norm(first)*norm(second)
    if denominator==0:return None
    return float(sum((a.float()*b.float()).sum() for a,b in zip(first,second)
                     if a is not None and b is not None)/denominator)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=external_path(args.output)
    if output.exists():raise ValueError('Preserve prior probe evidence.')
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    stats=json.loads(args.calibration.read_text())
    records,root,_=inventory()
    if stats['split']!='train' or stats['manifest_sha256']!=file_sha(root/'manifest.json'):
        raise ValueError('Frozen matched TRAIN calibration required.')
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=True,mmap=True)
    model=load_weight_quant(saved,'cuda').requires_grad_(True);params=list(model.parameters())
    teacher=EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    from types import SimpleNamespace
    from terrain_inference import prepare_models
    prepare_models(SimpleNamespace(coarse_model=teacher,base_model=teacher,decoder_model=teacher))
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight=linear_weight_window(64,'cuda',torch.float32);bank=QATBank()
    variance=torch.tensor(json.loads((DATA/'eval/native-decoded-probe/channel-statistics.json').read_text())['variance'],device='cuda')
    rows=[];torch.cuda.reset_peak_memory_stats()
    for index,(kind,coarse) in enumerate(zip(('desert-plain','temperate-plain','wet-complex-coast','low-plain-sea'),
                                         ('coarse4','coarse2','coarse4','coarse2'))):
        step=saved['step']+index;generator=np.random.default_rng(2026101259+index)
        with torch.no_grad(),model.cached_weights():
            batch,selected=bank.batch(records,generator,step,model,on_policy=index%2==0)
        regular,regular_terms,_=step_loss(model,teacher,batch,variance,saved['feature_weight'])
        base_grads=torch.autograd.grad(regular,params,allow_unused=True)
        base_loss=float(regular.detach());del regular,regular_terms
        pool=[r for r in records if r['case_kind']==kind and r['coarse_variant']==coarse]
        record=pool[int(generator.integers(len(pool)))]
        region=DecodedRegion(Path(record['root'])/record['file'],'cuda')
        if region.metadata['split']!='train':raise ValueError('Auxiliary scene must be TRAIN.')
        result=decoded_region_forward(model,region,weight,checkpointed=True)
        terms=low_field_terms(result['core'][None],region.target[None],
            flat_scene=kind in stats['flat_power_m2'],flat_power_m2=stats['flat_power_m2'].get(kind,1.))
        coast_grads=torch.autograd.grad(terms['coast'],params,allow_unused=True,retain_graph=True)
        flat_grads=torch.autograd.grad(terms['flat_highpass'],params,allow_unused=True)
        for gradients in (base_grads,coast_grads,flat_grads):
            if any(not torch.isfinite(g).all() for g in gradients if g is not None):
                raise FloatingPointError('Nonfinite prospective auxiliary gradient.')
        row=dict(kind=kind,coarse=coarse,seed=record['metadata']['seed'],scene=record['file'],split='train',
            regular_loss=base_loss,coast_loss=float(terms['coast'].detach()),flat_loss=float(terms['flat_highpass'].detach()),
            regular_grad_norm=norm(base_grads),coast_grad_norm=norm(coast_grads),flat_grad_norm=norm(flat_grads),
            coast_regular_cosine=cosine(base_grads,coast_grads),flat_regular_cosine=cosine(base_grads,flat_grads),
            regular_worlds=[r['metadata']['seed'] for _,r in selected],peak_bytes=torch.cuda.max_memory_allocated())
        rows.append(row);print(json.dumps(row),flush=True)
        del base_grads,coast_grads,flat_grads,result,terms,region;torch.cuda.empty_cache()
    if any(p.grad is not None for p in teacher.parameters()):raise ValueError('Teacher received gradients.')
    atomic_json(output,dict(status='complete',rows=rows,checkpoint_sha256=file_sha(args.checkpoint),
        calibration_sha256=file_sha(args.calibration),source_sha256=file_sha(Path(__file__)),
        gpu=torch.cuda.get_device_name(),weight_updates=0,primary_training_modified=False,
        scope='Whole fused lowLOD native two-stage parameter gradients; not fitted gain or decodedLOD0 coverage.'))


if __name__=='__main__':main()

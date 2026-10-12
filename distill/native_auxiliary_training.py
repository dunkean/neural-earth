"""Independent second-stage auxiliary paths and bounded training caches."""
from collections import OrderedDict
import math
from pathlib import Path

import numpy as np
import torch

from distill.common import HOLDOUT_SEEDS
from distill.decoded_loss import shore_loss
from distill.native_decoded import DecodedRegion
from distill.native_field_auxiliary import low_field_terms
from distill.native_fp8_training import QATBank
from distill.native_fusion import blend,forward_batches,refinement_inputs,window_outputs


class BoundedQATBank(QATBank):
    def __init__(self,limit=128):
        super().__init__();self.limit=limit;self.regions=OrderedDict()
    def load(self,record):
        key=(record['root'],record['file'])
        result=super().load(record);self.regions.move_to_end(key)
        while len(self.regions)>self.limit:self.regions.popitem(last=False)
        return result


def root_coast(prediction,target,minimum=16):
    actual=prediction[:,4:5].float()*38.6-31.4
    reference=target[:,4:5].detach().float()*38.6-31.4
    near=reference.abs()<=math.sqrt(20.)
    mask=torch.zeros_like(reference)
    counts=[]
    for side in (reference>0,reference<=0):
        selected=near&side;count=selected.sum((-3,-2,-1),keepdim=True)
        mask+=selected.float()*(count>=minimum);counts.append(int(selected.sum()))
    loss,disagreement=shore_loss(actual,reference,mask,band_m=math.sqrt(20.),temperature_m=math.sqrt(2.))
    return loss,disagreement,counts


def auxiliary_fields(model,teacher,record,noise_seed,checkpointed=True):
    # Never refresh QATBank's cached objects. This separate scene owns all noise.
    region=DecodedRegion(Path(record['root'])/record['file'],'cuda')
    if region.metadata['split']!='train' or region.metadata['seed'] in HOLDOUT_SEEDS:
        raise ValueError('Auxiliary path must use independent TRAIN scenes.')
    from terrain_diffusion.inference.world_pipeline import gaussian_noise_patch,linear_weight_window
    weight=linear_weight_window(64,'cuda',torch.float32);first,second=region.stages
    def noises(seed,origins):
        return torch.from_numpy(np.stack([gaussian_noise_patch(seed,y,x,64,64,channels=5,
            tile_h=64,tile_w=64) for y,x in origins])).cuda()
    noise0=noises(noise_seed+5819,region.init_origins);noise1=noises(noise_seed+5820,region.ref_origins)
    with torch.no_grad(),model.cached_weights():
        zero=torch.zeros((len(noise0),6,64,64),device='cuda');zero[:,-1:]=1.
        inputs0=refinement_inputs(zero,noise0,first['labels'],region.dtype)
        velocity0=forward_batches(model,inputs0,first['labels'],first['condition'])
        initial=blend(window_outputs(inputs0,velocity0,first['labels'],region.dtype),
                      region.init_origins,region.init_origin,192,weight)
        inputs1=refinement_inputs(region.ref_crops(initial),noise1,second['labels'],region.dtype)
    with torch.no_grad():
        target_velocity=forward_batches(teacher,inputs1.bfloat16(),second['labels'].bfloat16(),second['condition'].bfloat16())
        target=region.core_crop(window_outputs(inputs1,target_velocity,second['labels'],region.dtype),weight)
    velocity=forward_batches(model,inputs1,second['labels'],second['condition'],checkpointed=checkpointed)
    actual=region.core_crop(window_outputs(inputs1,velocity,second['labels'],region.dtype),weight)
    return actual[None],target[None]


def auxiliary_terms(model,teacher,record,noise_seed,calibration):
    actual,target=auxiliary_fields(model,teacher,record,noise_seed)
    kind=record['case_kind'];power=calibration['flat_power_m2'].get(kind,1.)
    terms=low_field_terms(actual,target,flat_scene=kind in calibration['flat_power_m2'],flat_power_m2=power)
    terms['coast'],terms['coast_disagreement'],terms['coast_counts']=root_coast(actual,target)
    return terms


def auxiliary_record(records,seed,step,cadence=2):
    generator=np.random.default_rng(np.random.SeedSequence([seed,step,0xA0]))
    index=step//cadence
    kinds=('desert-plain','temperate-plain','wet-complex-coast','low-plain-sea')
    # Cycle all12 strata, not four linked category/coarse pairs.
    kind=kinds[index%4];coarse=('original','coarse4','coarse2')[(index//4)%3]
    pool=[r for r in records if r['case_kind']==kind and r['coarse_variant']==coarse]
    return pool[int(generator.integers(len(pool)))],int(generator.integers(1,2**31-5821))


def preconditioners(optimizer,params):
    result=[]
    for p in params:
        state=optimizer.state.get(p)
        group=next(g for g in optimizer.param_groups if any(q is p for q in g['params']))
        if not state:
            # Adam initializes slots lazily; original unused parameters have no
            # moments. The standard zero-second-moment preconditioner is1/eps.
            result.append(torch.full_like(p,1/group['eps']));continue
        step=float(state['step']);correction=1-group['betas'][1]**step
        result.append((state['exp_avg_sq']/correction).sqrt().add(group['eps']).reciprocal())
    return result


def gradient_norm(grads,precondition=None):
    total=None
    for index,g in enumerate(grads):
        if g is None:continue
        value=g.float() if precondition is None else g.float()*precondition[index]
        term=value.square().sum();total=term if total is None else total+term
    return float(total.sqrt()) if total is not None else 0.


@torch.no_grad()
def combine_gradients(params,regular,precondition,reference_norm,reference_preconditioned,cap=.5):
    auxiliary=[p.grad for p in params]
    raw=gradient_norm(auxiliary);scaled=gradient_norm(auxiliary,precondition)
    factor=min(1.,cap*reference_norm/max(raw,1e-30),cap*reference_preconditioned/max(scaled,1e-30))
    for p,base,extra in zip(params,regular,auxiliary):
        if base is None and extra is None:p.grad=None
        elif extra is None:p.grad=base
        elif base is None:p.grad=extra.mul_(factor)
        else:p.grad=extra.mul_(factor).add_(base)
    return dict(auxiliary_grad_norm=raw,auxiliary_preconditioned_norm=scaled,auxiliary_scale=factor)

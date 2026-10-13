"""Teacher-basis initialization, TRAIN-only sampling and balanced KD losses."""
from contextlib import contextmanager
import json
import math

import numpy as np
import torch

from distill.common import DATA, HOLDOUT_SEEDS
from distill.native_lowrank import FactorMPConv, NativeLowrankBase
from distill.native_feature_kd import representation_loss
from distill.native_mixed_bank import KINDS, VARIANTS
from distill.native_phase_loss import PHASE_LABELS, WEIGHTS, band_terms, x0_from_velocity
from distill.wet_mountains import KIND, ROOT, check_splits


def selective_model(teacher, basis, selection, fraction=1.):
    ranks = {name:min(basis['factor_ranks'][name], max(16, math.ceil(
        basis['factor_ranks'][name]*fraction/16)*16)) for name in selection}
    model = NativeLowrankBase(basis['config'], ranks)
    state = {f'net.{name}': value for name,value in teacher.state_dict().items()
             if name.rsplit('.',1)[0] not in ranks}
    for name, rank in ranks.items():
        state[f'net.{name}.spatial'] = basis['ema'][f'net.{name}.spatial'][:rank]
        state[f'net.{name}.pointwise'] = basis['ema'][f'net.{name}.pointwise'][:, :rank]
    model.load_state_dict(state)
    return model.eval()


@torch.no_grad()
def anneal(model, target_ranks, previous, current):
    # Only tail factors are forced down; retained filters remain trainable.
    multiplier = (1-current)/(1-previous) if previous < 1 else 0.
    for name, rank in target_ranks.items():
        layer = model.net.get_submodule(name)
        layer.spatial[rank:].mul_(multiplier)


@torch.no_grad()
def compact(model, ranks):
    from terrain_diffusion.models.mp_layers import MPConv
    next_model = NativeLowrankBase(model.config, ranks).to(next(model.parameters()).device)
    state = {name: value for name,value in model.state_dict().items()
             if not name.endswith(('.spatial','.pointwise'))}
    for name, rank in ranks.items():
        old = model.net.get_submodule(name)
        conv = MPConv(old.in_channels, old.out_channels, [3,3], no_padding=old.no_padding).to(old.spatial.device)
        conv.weight.copy_(old.composed_weight())
        factor = FactorMPConv.from_conv(conv, rank)
        state[f'net.{name}.spatial'],state[f'net.{name}.pointwise'] = factor.spatial,factor.pointwise
    next_model.load_state_dict(state)
    return next_model.eval()


@contextmanager
def capture_blocks(net):
    values, handles = {}, []
    try:
        # Every encoder output is exactly a decoder skip tensor, too.
        for side in ('enc','dec'):
            for name,module in getattr(net,side).items():
                key = side+'.'+name
                def capture(module,inputs,output,key=key):
                    if key in values: raise RuntimeError('A block executed twice in one KD forward.')
                    values[key] = output
                handles.append(module.register_forward_hook(capture))
        yield values
    finally:
        for handle in handles: handle.remove()


def inventory():
    main = DATA/'crops/native-mixed-scenes-v2'
    manifest = json.loads((main/'manifest.json').read_text())
    all_rows = list(manifest['scenes']); train = []
    for row in all_rows:
        if row['metadata']['split']=='train': train.append(row | dict(root=str(main)))
    for root in (ROOT,DATA/'crops/native-wet-mountain-train-extension-v1'):
        if json.loads((root/'status.json').read_text())['status']!='complete':
            raise ValueError('Complete qualified mountain bank required.')
        rows = json.loads((root/'manifest.json').read_text())['scenes']
        check_splits(rows,all_rows); all_rows += rows
        qualified = {r['file'] for r in json.loads((root/'qualification.json').read_text())['rows']
                     if r['split']=='train' and r['physical_qualified']}
        train.extend(row | dict(root=str(root)) for row in rows if row['file'] in qualified
                     and row['metadata']['split']=='train')
    if any(r['metadata']['split']!='train' or r['metadata']['seed'] in HOLDOUT_SEEDS for r in train):
        raise ValueError('TRAIN-only independent sampling required.')
    return train, main, manifest


def choose(records, generator, step):
    selected, used = [], set()
    # 25% mountain, 15% old rare (one plus a fifth of a window/batch), 60% general.
    old_slots = {int(generator.integers(8))}
    if step%5==0:
        old_slots.add((next(iter(old_slots))+4)%8)
    for phase in (0,1):
        variants = list(VARIANTS)+[VARIANTS[(step+phase)%3]]
        generator.shuffle(variants)
        mountain_slot = int(generator.integers(4))
        # Move rare requests off the mandatory mountain slot.
        phase_slots = {s%4 for s in old_slots if s//4==phase}
        if mountain_slot in phase_slots:
            phase_slots.remove(mountain_slot); phase_slots.add((mountain_slot+1)%4)
        for slot,variant in enumerate(variants):
            kind = KIND if slot==mountain_slot else KINDS[(step+phase)%4] if slot in phase_slots else None
            pool = [r for r in records if r['coarse_variant']==variant and r['case_kind']==kind
                    and r['metadata']['seed'] not in used]
            profiles = sorted({r['metadata']['profile'] for r in pool})
            if not profiles: raise ValueError('Missing independent TRAIN stratum.')
            profile = profiles[int(generator.integers(len(profiles)))]
            pool = [r for r in pool if r['metadata']['profile']==profile]
            row = pool[int(generator.integers(len(pool)))]
            used.add(row['metadata']['seed']); selected.append((phase,row))
    return selected


class WindowBank:
    def __init__(self, records, preload=False):
        self.arrays = {}
        if preload:
            for row in records: self.load(row)

    def load(self,row):
        from pathlib import Path
        key = (row['root'],row['file'])
        if key not in self.arrays:
            with np.load(Path(row['root'])/row['file'],allow_pickle=False) as sample:
                meta = json.loads(str(sample['metadata']))
                if meta['split']!='train' or meta['seed']!=row['metadata']['seed'] or meta['seed'] in HOLDOUT_SEEDS:
                    raise ValueError('Captured file is not this independent TRAIN scene.')
                core = tuple(((meta[k]+95)//96)*96 for k in ('y','x'))
                all_ids=[]
                for phase,offsets in ((0,(-64,-32,0,32,64)),(1,(-32,0,32))):
                    origins={(core[0]+y,core[1]+x) for y in offsets for x in offsets}
                    ids=[int(i) for i in np.flatnonzero(sample['labels']==PHASE_LABELS[phase])
                         if (int(sample['contexts'][i,1])*32,int(sample['contexts'][i,2])*32) in origins]
                    if len(ids)!=len(origins): raise ValueError('Incomplete native teacher dependencies.')
                    all_ids += ids
                self.arrays[key]={k:torch.from_numpy(sample[k][all_ids].copy())
                                  for k in ('inputs','labels','condition','target')}
        return self.arrays[key]

    def batch(self,records,generator,step):
        selected=choose(records,generator,step); windows=[]
        for phase,row in selected:
            arrays=self.load(row); begin,end=(0,25) if phase==0 else (25,34)
            i=int(generator.integers(begin,end))
            windows.append({k:v[i:i+1] for k,v in arrays.items()})
        batch={k:torch.cat([w[k] for w in windows]).cuda(non_blocking=True) for k in windows[0]}
        batch['mountain']=torch.tensor([r['case_kind']==KIND for _,r in selected],device='cuda')
        return batch, selected


def objective(prediction,inputs,labels,target,variance,calibration,mountain):
    first=torch.isclose(labels.float(),labels.new_tensor(PHASE_LABELS[0]).float(),atol=.004,rtol=0)
    second=torch.isclose(labels.float(),labels.new_tensor(PHASE_LABELS[1]).float(),atol=.004,rtol=0)
    if not first.any() or not second.any() or not (first|second).all():
        raise ValueError('Both original diffusion phases are required.')
    scale=variance.reshape(1,5,1,1).detach()
    residual=(prediction-target.detach())/scale.sqrt()
    terms={'operator':torch.stack([residual[m].square().mean() for m in (first,second)]).mean(),
           'operator_gradient':torch.stack([residual[m].diff(dim=a).square().mean()
                                            for m in (first,second) for a in (-1,-2)]).mean()}
    actual=x0_from_velocity(inputs,prediction,labels)
    reference=x0_from_velocity(inputs,target.detach(),labels).detach()
    terms['first_x0']=((actual[first,:4]-reference[first,:4]).square()/scale[:,:4]).mean()
    terms['second_height']=((actual[second,4:]-reference[second,4:]).square()/calibration['second_variance'][4]).mean()
    for name,mask,channels,power in (('first',first,slice(0,4),calibration['first_power'][:,:4]),
                                    ('second',second,slice(4,5),calibration['second_power'][:,4:])):
        pos,energy=band_terms(actual[mask,channels],reference[mask,channels],variance[channels],power)
        subset=mask & mountain
        if subset.any():
            _,extra=band_terms(actual[subset,channels],reference[subset,channels],variance[channels],power)
            energy=energy+extra*(subset.sum()/mask.sum())
        terms[name+'_band_position'],terms[name+'_band_energy']=pos,energy
    loss=terms['operator']+.05*terms['operator_gradient']+sum(WEIGHTS[k]*terms[k] for k in WEIGHTS)
    return loss,terms


def kd_step(model,teacher,batch,variance,calibration,feature_weight):
    with torch.no_grad(),capture_blocks(teacher) as reference:
        target=teacher(batch['inputs'].bfloat16(),noise_labels=batch['labels'].bfloat16(),
                       conditional_inputs=[batch['condition'].bfloat16()]).float()
    with capture_blocks(model.net) as actual:
        prediction=model(batch['inputs'],batch['labels'],[batch['condition']])
    feature,blocks=representation_loss(actual,reference)
    loss,terms=objective(prediction,batch['inputs'],batch['labels'],target,variance,calibration,batch['mountain'])
    return loss+feature_weight*feature, terms|dict(representation=feature), blocks

"""Differentiable native T=2 regions, preserving the runtime's 64/32 geometry.

A 32x32 core uses four refinement windows and their sixteen initialization
dependencies. Refinement consumes the student's blended first stage. Teacher
operator targets on those student states are detached; final terrain targets
remain the original two-stage teacher. This module does not accept a model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from distill.common import DATA, atomic_json


def window_outputs(inputs, velocities, labels, dtype=torch.bfloat16):
    """Exact native expression order, including BF16 arithmetic before blending."""
    inputs, velocities, labels = (value.to(dtype) for value in (inputs, velocities, labels))
    t = labels[:, None, None, None]
    x_t = inputs*.5
    prediction = -velocities
    return (torch.cos(t)*x_t-torch.sin(t)*.5*prediction).float()/.5


def refinement_inputs(weighted, noise, labels, dtype=torch.bfloat16):
    weighted, noise, labels = (value.to(dtype) for value in (weighted, noise, labels))
    if weighted.shape[1] != 6 or torch.any(weighted[:, -1:] <= 0):
        raise ValueError('Refinement requires five weighted channels and complete positive coverage.')
    sample = weighted[:, :-1]/weighted[:, -1:]*.5
    t = labels[:, None, None, None]
    return (torch.cos(t)*sample+torch.sin(t)*(noise*.5))/.5


def blend(windows, origins, origin, size, weight):
    """Accumulate whole windows in fixed order, retaining autograd through fusion."""
    result = windows.new_zeros((6, size, size), dtype=torch.float32)
    weight = weight.to(windows.device, torch.float32)
    for values, (y, x) in zip(windows, origins):
        dy, dx = int(y)-origin[0], int(x)-origin[1]
        if dy < 0 or dx < 0 or dy+64 > size or dx+64 > size:
            raise ValueError('Blend canvas must contain each complete native window.')
        weighted = torch.cat((values.float()*weight[None], weight[None]))
        result = result+F.pad(weighted, (dx, size-dx-64, dy, size-dy-64))
    return result


def overlap_loss(prediction, target, origins):
    """Match teacher overlap differences rather than forcing its windows identical."""
    losses = []
    for i, (y, x) in enumerate(origins):
        for j in range(i+1, len(origins)):
            yy, xx = origins[j]
            if (yy-y, xx-x) == (0, 32):
                delta = prediction[i, :, :, 32:]-prediction[j, :, :, :32]
                expected = target[i, :, :, 32:]-target[j, :, :, :32]
            elif (yy-y, xx-x) == (32, 0):
                delta = prediction[i, :, 32:, :]-prediction[j, :, :32, :]
                expected = target[i, :, 32:, :]-target[j, :, :32, :]
            else:
                continue
            losses.append((delta-expected).square().mean())
    if not losses:
        raise ValueError('Both axes of neighboring native windows must be represented.')
    return torch.stack(losses).mean()


class Region:
    def __init__(self, path, device='cpu', core_offset=64):
        from terrain_diffusion.inference.world_pipeline import gaussian_noise_patch
        with np.load(path, allow_pickle=False) as sample:
            arrays = {key: sample[key].copy() for key in sample.files if key != 'metadata'}
            self.metadata = json.loads(str(sample['metadata']))
        oy,ox = (core_offset,core_offset) if isinstance(core_offset,int) else core_offset
        if oy not in (32,64) or ox not in (32,64) or self.metadata['native_size'] != 128:
            raise ValueError('Native regions require a128 scene and an interior32-aligned core.')
        self.dtype = torch.bfloat16 if torch.device(device).type == 'cuda' else torch.float32
        self.core = (self.metadata['y']+oy, self.metadata['x']+ox)
        self.init_origin = (self.core[0]-64, self.core[1]-64)
        self.init_origins = [(self.core[0]+dy, self.core[1]+dx)
                             for dy in (-64,-32,0,32) for dx in (-64,-32,0,32)]
        self.ref_origins = [(self.core[0]+dy, self.core[1]+dx)
                            for dy in (-32,0) for dx in (-32,0)]
        labels = np.unique(arrays['labels'])
        if len(labels) != 2:
            raise ValueError('Scene must contain both native stages.')
        self.stages = []
        for label, origins in ((labels.max(),self.init_origins), (labels.min(),self.ref_origins)):
            lookup = {}
            for index in np.flatnonzero(arrays['labels'] == label):
                context = arrays['contexts'][index]
                key = (int(context[1])*32, int(context[2])*32)
                if key in lookup:
                    raise ValueError('A native scene contains a duplicate phase/window.')
                lookup[key] = index
            try:
                selected = [lookup[origin] for origin in origins]
            except KeyError as exc:
                raise ValueError(f'Incomplete native region dependencies: {exc}') from exc
            self.stages.append({key: torch.from_numpy(arrays[key][selected]).to(device)
                                for key in ('inputs','labels','condition','target')})
        self.noise = torch.from_numpy(np.stack([gaussian_noise_patch(self.metadata['seed']+5820,
            y,x,64,64,channels=5,tile_h=64,tile_w=64) for y,x in self.ref_origins])).to(device)
        initial = torch.from_numpy(arrays['initial_weighted']).to(device)
        initial_origin = arrays['initial_origin']
        self.teacher_initial = torch.stack([initial[:, y-int(initial_origin[0]):y-int(initial_origin[0])+64,
            x-int(initial_origin[1]):x-int(initial_origin[1])+64] for y,x in self.ref_origins])
        self.target = torch.from_numpy(arrays['final_latents'][:,oy:oy+32,ox:ox+32]).to(device)
        self.mask = torch.from_numpy(arrays['final_mask'][:,oy:oy+32,ox:ox+32]).to(device)
        if not self.mask.all():
            raise ValueError('Teacher target has incomplete core coverage.')

    def ref_crops(self, canvas):
        return torch.stack([canvas[:,y-self.init_origin[0]:y-self.init_origin[0]+64,
            x-self.init_origin[1]:x-self.init_origin[1]+64] for y,x in self.ref_origins])

    def core_crop(self, refined, weight):
        origin = (self.core[0]-32, self.core[1]-32)
        canvas = blend(refined, self.ref_origins, origin, 96, weight)
        weighted = canvas[:,32:64,32:64]
        if torch.any(weighted[-1:] <= 0):
            raise ValueError('Refined core has incomplete coverage.')
        return weighted[:-1]/weighted[-1:]

    @torch.no_grad()
    def refresh_noise(self,teacher,weight,seed):
        """New deterministic two-phase noises over the same train conditioning.

        The runtime uses noise_level=0 for the58-value vector: conditioning is
        independent of these base noise fields. Targets are recomputed through
        the teacher's two stages, not augmented by adding output-space noise.
        """
        from terrain_diffusion.inference.world_pipeline import gaussian_noise_patch
        first,second=self.stages
        device=first['inputs'].device
        noise0=torch.from_numpy(np.stack([gaussian_noise_patch(seed+5819,y,x,64,64,
            channels=5,tile_h=64,tile_w=64) for y,x in self.init_origins])).to(device)
        zeros=noise0.new_zeros((len(noise0),6,64,64))
        zeros[:,-1:]=1.
        first['inputs']=refinement_inputs(zeros,noise0,first['labels'],self.dtype)
        first['target']=forward_batches(teacher,first['inputs'].to(self.dtype),first['labels'].to(self.dtype),
                                        first['condition'].to(self.dtype)).float()
        output0=window_outputs(first['inputs'],first['target'],first['labels'],self.dtype)
        initial=blend(output0,self.init_origins,self.init_origin,160,weight)
        self.teacher_initial=self.ref_crops(initial)
        self.noise=torch.from_numpy(np.stack([gaussian_noise_patch(seed+5820,y,x,64,64,
            channels=5,tile_h=64,tile_w=64) for y,x in self.ref_origins])).to(device)
        second['inputs']=refinement_inputs(self.teacher_initial,self.noise,second['labels'],self.dtype)
        second['target']=forward_batches(teacher,second['inputs'].to(self.dtype),second['labels'].to(self.dtype),
                                         second['condition'].to(self.dtype)).float()
        self.target=self.core_crop(window_outputs(second['inputs'],second['target'],second['labels'],self.dtype),weight)
        self.noise_seed=int(seed)


def forward_batches(model, inputs, labels, conditions, batch=4, checkpointed=False):
    values = []
    def forward(x,t,c):
        return model(x,t,[c])
    for start in range(0,len(inputs),batch):
        args = (inputs[start:start+batch], labels[start:start+batch], conditions[start:start+batch])
        values.append(checkpoint(forward,*args,use_reentrant=False) if checkpointed else forward(*args))
    return torch.cat(values)


def region_forward(model, region, weight, checkpointed=False):
    first, second = region.stages
    velocity0 = forward_batches(model,first['inputs'],first['labels'],first['condition'],checkpointed=checkpointed)
    output0 = window_outputs(first['inputs'],velocity0,first['labels'],region.dtype)
    initial = blend(output0,region.init_origins,region.init_origin,160,weight)
    inputs1 = refinement_inputs(region.ref_crops(initial),region.noise,second['labels'],region.dtype)
    velocity1 = forward_batches(model,inputs1,second['labels'],second['condition'],checkpointed=checkpointed)
    output1 = window_outputs(inputs1,velocity1,second['labels'],region.dtype)
    return dict(core=region.core_crop(output1,weight), inputs1=inputs1, velocity0=velocity0,
                velocity1=velocity1, output0=output0, output1=output1)


def region_loss(result, region, teacher):
    first, second = region.stages
    with torch.no_grad():
        target1 = teacher(result['inputs1'].detach(),noise_labels=second['labels'].to(region.dtype),
                          conditional_inputs=[second['condition'].to(region.dtype)]).float()
    terms = dict(latent=(result['core']-region.target).square().mean(),
                 init_operator=(result['velocity0']-first['target']).square().mean(),
                 refinement_operator=(result['velocity1']-target1).square().mean())
    gradients, curvature = [], []
    for axis in (-1,-2):
        gradients.append((result['core'].diff(dim=axis)-region.target.diff(dim=axis)).square().mean())
        curvature.append((result['core'][4:].diff(n=2,dim=axis)-region.target[4:].diff(n=2,dim=axis)).square().mean())
    terms['gradient'], terms['height_curvature'] = torch.stack(gradients).mean(),torch.stack(curvature).mean()
    expected1 = window_outputs(result['inputs1'].detach(),target1,second['labels'],region.dtype)
    terms['overlap'] = overlap_loss(result['output1'],expected1,region.ref_origins)
    total = terms['latent']+.1*terms['gradient']+.05*terms['height_curvature']+\
            .05*(terms['init_operator']+terms['refinement_operator'])+.02*terms['overlap']
    return total,terms


@torch.no_grad()
def audit_region(region, weight):
    first,second = region.stages
    outputs0 = window_outputs(first['inputs'],first['target'],first['labels'],region.dtype)
    initial = blend(outputs0,region.init_origins,region.init_origin,160,weight)
    original_inputs = refinement_inputs(region.teacher_initial,region.noise,second['labels'],region.dtype)
    rebuilt_inputs = refinement_inputs(region.ref_crops(initial),region.noise,second['labels'],region.dtype)
    outputs1 = window_outputs(second['inputs'],second['target'],second['labels'],region.dtype)
    final = region.core_crop(outputs1,weight)
    return dict(original_input_max_abs=float((original_inputs-second['inputs']).abs().max()),
        fused_input_max_abs=float((rebuilt_inputs-second['inputs']).abs().max()),
        fused_input_mean_abs=float((rebuilt_inputs-second['inputs']).abs().float().mean()),
        final_core_max_abs=float((final-region.target).abs().max()),
        final_core_mean_abs=float((final-region.target).abs().mean()),
        note='Stored teacher velocities; fusion order may have a floating-point floor. No student quality claim.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DATA/'crops/native-scenes-v1')
    parser.add_argument('--output',type=Path,default=DATA/'eval/native-fusion-reconstruction.json')
    parser.add_argument('--count',type=int,default=12)
    args=parser.parse_args()
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    torch.set_num_threads(8)
    files=sorted(args.dataset.glob('train-*.npz'))
    if not files or args.count <= 0:
        raise ValueError('Captured native scenes and a positive audit count are required.')
    weight=linear_weight_window(64,'cuda',torch.float32)
    rows=[]
    for index in np.linspace(0,len(files)-1,min(args.count,len(files))).round().astype(int):
        region=Region(files[index],'cuda')
        result=audit_region(region,weight)
        if result['original_input_max_abs'] != 0.:
            raise RuntimeError(f'Exact native noise/weighted-input reconstruction failed: {result}')
        rows.append(dict(file=files[index].name,seed=region.metadata['seed'],**result))
    atomic_json(args.output,dict(status='complete',rows=rows,accepted=False))
    print(json.dumps(rows),flush=True)


if __name__ == '__main__':
    main()

"""Paired same-window climate supervision from immutable native captures."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from distill.native_weight_quant import file_sha
from distill.tile40_conditioning import CLIMATES, RELIEFS


class ClimateBank:
    def __init__(self,root,*,partial=False):
        self.root = Path(root)
        manifest = json.loads((self.root/'manifest.json').read_text())
        self.manifest_sha = file_sha(self.root/'manifest.json')
        self.groups = {'train':[], 'val':[]}
        self.cache = {}
        for split in self.groups:
            keys = sorted({(c['seed'],c['relief']) for c in manifest['cases'] if c['split']==split})
            for seed,relief in keys:
                rows = [next(c for c in manifest['cases'] if c['split']==split and
                    c['seed']==seed and c['relief']==relief and c['climate']==climate) for climate in CLIMATES]
                if not all((self.root/c['file']).exists() and (self.root/c['file']).with_suffix('.json').exists() for c in rows):
                    if partial:
                        continue
                    raise ValueError('A complete immutable climate capture bank is required.')
                self.groups[split].append(rows)
        if not self.groups['train']:
            raise ValueError('No complete five-climate TRAIN geometry is available.')

    def load(self,case):
        name = case['file']
        if name not in self.cache:
            path = self.root/name
            if file_sha(path)!=json.loads(path.with_suffix('.json').read_text())['file_sha256']:
                raise ValueError('Frozen capture changed.')
            with np.load(path,allow_pickle=False) as saved:
                metadata = json.loads(str(saved['metadata']))
                if metadata['capture_manifest_sha256']!=self.manifest_sha or metadata['case']!=case:
                    raise ValueError('Frozen capture metadata changed.')
                arrays = {key:saved[key].astype(np.float32) for key in ('inputs','labels','condition','target')}
            if {round(float(v),3) for v in arrays['labels']}!={1.562,.609}:
                raise ValueError('Both pinned native phases are required.')
            self.cache[name] = arrays
        return self.cache[name]

    def statistics(self):
        total = np.zeros((2,5),np.float64)
        square = total.copy(); count = np.zeros(2,np.int64)
        climate_square = total.copy(); climate_count = np.zeros(2,np.int64)
        for group in self.groups['train']:
            arrays = [self.load(case) for case in group]
            # The initial Gaussian/window order must be identical across climates.
            first = arrays[0]['labels']>1
            for values in arrays[1:]:
                other = values['labels']>1
                if not np.array_equal(arrays[0]['inputs'][first],values['inputs'][other]):
                    raise ValueError('Paired climates do not share initial noise/window order.')
            for phase in (0,1):
                targets = [values['target'][(values['labels']>1)==(phase==0)] for values in arrays]
                for target in targets:
                    total[phase] += target.sum(axis=(0,2,3),dtype=np.float64)
                    square[phase] += np.square(target,dtype=np.float64).sum(axis=(0,2,3))
                    count[phase] += target.shape[0]*target.shape[2]*target.shape[3]
                for i in range(5):
                    for j in range(i):
                        delta = targets[i].astype(np.float64)-targets[j]
                        climate_square[phase] += np.square(delta).sum(axis=(0,2,3))
                        climate_count[phase] += delta.shape[0]*delta.shape[2]*delta.shape[3]
        variance = np.maximum(square/count[:,None]-(total/count[:,None])**2,1e-5)
        climate_power = np.maximum(climate_square/climate_count[:,None],variance*.01)
        return dict(variance=variance.tolist(),climate_power=climate_power.tolist(),
            train_geometries=len(self.groups['train']),val_geometries=len(self.groups['val']),
            source='TRAIN only;same Gaussian checked across all five climates.')

    def batch(self,rng,device,split='train'):
        groups = self.groups[split]
        rows = []
        for phase in (0,1):
            if split=='train':
                relief = 'mountains' if rng.random()<.7 else ('hills' if rng.random()<.5 else 'plain')
                available = [g for g in groups if g[0]['relief']==relief]
                if not available:
                    available = groups  # Mechanics-only partial smoke may have mountains only.
            else:
                available = groups
            group = available[int(rng.integers(len(available)))]
            a,b = rng.choice(5,2,replace=False)
            sources = [self.load(group[int(i)]) for i in (a,b)]
            indices = [np.flatnonzero((s['labels']>1)==(phase==0)) for s in sources]
            if len(indices[0])!=len(indices[1]):
                raise ValueError('Incomplete paired native phase.')
            index = int(rng.integers(len(indices[0])))
            for values,ids in zip(sources,indices):
                rows.append({k:values[k][ids[index]] for k in values})
        batch = {key:torch.from_numpy(np.stack([row[key] for row in rows])).to(device) for key in rows[0]}
        batch['phase'] = torch.tensor([0,0,1,1],device=device)
        return batch


def paired_loss(prediction,batch,stats):
    variance = prediction.new_tensor(stats['variance'])[batch['phase'],:,None,None]
    residual = (prediction-batch['target'])/variance.sqrt()
    operator = residual.square().mean()
    gradient = sum(residual.diff(dim=d).square().mean() for d in (-1,-2))*.5
    band = (residual-F.avg_pool2d(residual,5,stride=1,padding=2)).square().mean()
    delta_error = (prediction[::2]-prediction[1::2])-(batch['target'][::2]-batch['target'][1::2])
    climate_power = prediction.new_tensor(stats['climate_power'])[:,:,None,None]
    climate = (delta_error.square()/climate_power).mean()
    terms = dict(operator=operator,gradient=gradient,band_position=band,climate_pair=climate)
    return operator+.05*gradient+.1*band+.1*climate,terms

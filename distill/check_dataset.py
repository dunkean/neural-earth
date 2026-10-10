"""Full dataset audit: finite arrays, complete triples, reserved seeds and split leakage."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from distill.common import DATA, HOLDOUT_SEEDS, SCHEMA, atomic_json


def check(root, expected=20000):
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest['schema'] != SCHEMA:
        raise ValueError('Unsupported dataset schema.')
    worlds = {'train': set(), 'val': set()}
    stages, profiles = {}, defaultdict(lambda: dict(train=0, val=0, land_sum=0., count=0))
    for stage in manifest['stages']:
        seen = set()
        paths = sorted((root/stage).glob('*.npz'))
        for path in paths:
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data['metadata']))
                index = metadata['index']
                if index in seen or not 0 <= index < expected:
                    raise ValueError(f'Duplicate/out-of-range sample: {path}')
                seen.add(index)
                world = index//manifest['samples_per_world']
                seed = manifest['seed_base']+world
                profile = manifest['profiles'][world % len(manifest['profiles'])]
                split = 'val' if world//len(manifest['profiles']) % 50 == 49 else 'train'
                if metadata['seed'] != seed or seed in HOLDOUT_SEEDS or metadata['profile'] != profile or metadata['split'] != split:
                    raise ValueError(f'Seed/profile/split provenance mismatch: {path}')
                if path.name != f'{split}-{index:07d}.npz':
                    raise ValueError(f'Incorrect sample filename: {path}')
                worlds[split].add(seed)
                channels = dict(base=5, coarse=6, decoder=1)[stage]
                size = metadata['crop']
                if data['target'].shape != (channels, size, size) or data['mask'].shape != (1, size, size):
                    raise ValueError(f'Target/mask shape mismatch: {path}')
                if data['target'].dtype != np.float16 or not data['mask'].any():
                    raise ValueError(f'Incorrect target dtype or empty target mask: {path}')
                for name in data.files:
                    if name != 'metadata' and not np.isfinite(data[name]).all():
                        raise ValueError(f'Non-finite {name}: {path}')
                if stage == 'base':
                    cells = (size+2*metadata['halo'])//32
                    if data['coarse'].shape != (6, cells, cells) or data['histogram'].shape != (5,):
                        raise ValueError(f'Incomplete dense conditioning/halo: {path}')
                    if (metadata['y'] % 32 or metadata['x'] % 32 or
                        metadata['coarse_y'] != (metadata['y']-metadata['halo'])//32 or
                        metadata['coarse_x'] != (metadata['x']-metadata['halo'])//32):
                        raise ValueError(f'Conditioning/grid alignment mismatch: {path}')
                    profiles[profile][split] += 1
                    profiles[profile]['land_sum'] += metadata['land_fraction']
                    profiles[profile]['count'] += 1
                elif stage == 'coarse':
                    if data['condition'].shape != (5, 64, 64) or data['labels'].shape != (5,):
                        raise ValueError(f'Coarse feature shape mismatch: {path}')
                elif data['latents'].shape != (4, size//8, size//8):
                    raise ValueError(f'Decoder latent shape mismatch: {path}')
        if seen != set(range(expected)):
            raise ValueError(f'{stage}: missing {expected-len(seen)} of {expected} examples.')
        stages[stage] = len(seen)
    if worlds['train'] & worlds['val']:
        raise ValueError('Validation/training worlds overlap.')
    return dict(passed=True, count=expected, stages=stages, schema=SCHEMA,
                training_worlds=len(worlds['train']), validation_worlds=len(worlds['val']),
                holdout_seeds_excluded=sorted(HOLDOUT_SEEDS),
                profiles={name:dict(train=value['train'], val=value['val'],
                          mean_land_fraction=value['land_sum']/value['count']) for name,value in profiles.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATA/'crops/main')
    parser.add_argument('--count', type=int, default=20000)
    args = parser.parse_args()
    result = check(args.dataset, args.count)
    atomic_json(args.dataset/'dataset-audit.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()

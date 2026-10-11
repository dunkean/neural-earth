"""Audit rare landscape proxies in native scenes without changing live training.

Use the exact aligned64-square teacher height target and recorded conditioning.
The climate is the teacher's central2x2 coarse-cell mean, recovered before the
magnitude-preserving concatenation. This is a conditioning/low-frequency proxy,
not qualification of the full decoded terrain or the reserved acceptance bank.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA, HOLDOUT_SEEDS, REPO, atomic_json, external_path
from distill.rare_cases import KINDS, field_stats, qualifies, relief_stats


def climate_from_condition(values):
    from terrain_diffusion.models.mp_layers import mp_concat
    values = np.asarray(values, np.float64)
    if values.shape != (58,) or not np.isfinite(values).all():
        raise ValueError('Finite native58 conditioning required.')
    # Same six groups as WorldPipeline._process_latent_conditioning.
    groups = [torch.zeros(1, n) for n in (16, 16, 4, 16, 5, 1)]
    groups[2].fill_(1.)
    gain = float(mp_concat(groups, dim=1)[0, 32])
    means = np.array([15.87, 619.26, 833.12, 69.40])
    stds = np.array([10.40, 452.29, 738.09, 34.59])
    return values[32:36]/gain*stds + means


def audit(root):
    manifest = json.loads((root/'manifest.json').read_text())
    counts, rows, worlds = Counter(), [], {'train': set(), 'val': set()}
    for path in sorted(root.glob('*.npz')):
        with np.load(path, allow_pickle=False) as sample:
            meta = json.loads(str(sample['metadata']))
            if meta['seed'] in HOLDOUT_SEEDS or meta['split'] not in worlds:
                raise ValueError('Native coverage must exclude reserved worlds.')
            worlds[meta['split']].add(meta['seed'])
            y, x = meta['y'], meta['x']
            core = tuple(((value+95)//96)*96 for value in (y, x))
            oy, ox = core[0]-y, core[1]-x
            latent = sample['final_latents'][4, oy:oy+64, ox:ox+64].astype(np.float64)
            mask = sample['final_mask'][0, oy:oy+64, ox:ox+64]
            if latent.shape != (64, 64) or not mask.all():
                raise ValueError('Native coverage requires the complete aligned decoder core.')
            contexts, labels = sample['contexts'], sample['labels']
            indices = np.flatnonzero((contexts[:, 1] == core[0]//32) &
                (contexts[:, 2] == core[1]//32) & (labels == labels.min()))
            if len(indices) != 1:
                raise ValueError('Missing or ambiguous conditioning for the aligned core.')
            climate = climate_from_condition(sample['condition'][indices[0]])
        sqrt_height = latent*38.6-31.4
        height = np.sign(sqrt_height)*sqrt_height**2
        climate_fields = np.broadcast_to(climate[:, None, None], (4, 64, 64))
        conditioning = field_stats(np.concatenate((height[None], climate_fields)))
        terrain = relief_stats(height, 3)
        kinds = [kind for kind in KINDS if conditioning is not None and qualifies(kind, terrain, conditioning)]
        archetypes = []
        if conditioning is not None:
            for kind, archetype in (('desert-plain', 'warm-arid'), ('temperate-plain', 'mild-temperate')):
                if qualifies(kind, terrain, conditioning, archetype):
                    archetypes.append(archetype)
        for key in ['all', *kinds, *archetypes]:
            counts[(meta['split'], meta['profile'], key)] += 1
        rows.append(dict(file=path.name, seed=meta['seed'], split=meta['split'],
            profile=meta['profile'], core=core, kinds=kinds, archetypes=archetypes,
            teacher_lowfreq=terrain, climate_window_mean=conditioning))
    if not rows or worlds['train'] & worlds['val'] or len(rows) != len(manifest['scenes']):
        raise ValueError('Native scene inventory or train/validation separation failed.')
    return dict(schema=1, dataset_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        source_sha256={name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in (
            'distill/native_coverage.py', 'distill/rare_cases.py',
            'terrain-diffusion/terrain_diffusion/models/mp_layers.py',
            'terrain-diffusion/terrain_diffusion/inference/world_pipeline.py')},
        counts=[dict(split=split, profile=profile, kind=kind, count=count)
                for (split, profile, kind), count in sorted(counts.items())], rows=rows,
        note='Aligned64-latent low-frequency teacher proxy at240m/pixel, climate central2x2-cell mean from captured conditioning. No student, no decoded qualification, no sampling-policy change.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATA/'crops/native-scenes-v1')
    parser.add_argument('--output', type=Path, default=DATA/'eval/native-rare-coverage.json')
    args = parser.parse_args()
    result = audit(args.dataset)
    atomic_json(external_path(args.output), result)
    totals = Counter()
    for row in result['counts']:
        totals[(row['split'], row['kind'])] += row['count']
    print(json.dumps([dict(split=split, kind=kind, count=count)
                      for (split, kind), count in sorted(totals.items())]), flush=True)


if __name__ == '__main__':
    main()

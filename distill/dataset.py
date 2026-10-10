"""Deterministic step-addressable sampling, independent of DataLoader prefetch."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from distill.features import base_features, coarse_features, decoder_features


class Crops(Dataset):
    def __init__(self, root, stage, split='train', halo=0, train_size=None, overfit=False):
        self.root, self.stage, self.halo, self.train_size = Path(root), stage, halo, train_size
        self.paths = sorted((self.root/stage).glob(f'{split}-*.npz'))
        if overfit and not self.paths:
            self.paths = sorted((self.root/stage).glob('train-*.npz'))[:8]
        if not self.paths:
            raise ValueError(f'No {split} {stage} crops under {root}.')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, address):
        index, random_seed = address if isinstance(address, tuple) else (address, 0)
        with np.load(self.paths[index], allow_pickle=False) as sample:
            meta = json.loads(str(sample['metadata']))
            target = torch.from_numpy(sample['target'].astype(np.float32))
            mask = torch.from_numpy(sample['mask'].astype(np.float32))
            crop = meta['crop']
            size = self.train_size or crop
            if size > crop or size < 32 or size % 32:
                raise ValueError('Training crop must fit the target and align to 32.')
            rng = np.random.default_rng(random_seed)
            # Base field translations respect the teacher's 32-cell phase.
            # Window students keep the full boundary geometry of their teacher.
            if self.stage == 'base':
                oy, ox = (int(v)*32 for v in rng.integers(0, (crop-size)//32+1, size=2))
                target = target[:, oy:oy+size, ox:ox+size]
                mask = mask[:, oy:oy+size, ox:ox+size]
                y, x = meta['y']+oy-self.halo, meta['x']+ox-self.halo
                inputs = base_features(sample['coarse'], meta['coarse_y'], meta['coarse_x'],
                                       meta['seed'], y, x, size+2*self.halo, sample['histogram'])
            elif self.stage == 'coarse':
                if self.train_size and size != crop:
                    raise ValueError('Coarse must train on whole production windows.')
                inputs = coarse_features(sample['condition'], sample['labels'], meta['seed'], meta['y'], meta['x'])
            else:
                if self.train_size and size != crop:
                    raise ValueError('Decoder must train on whole production windows.')
                inputs = decoder_features(sample['latents'], meta['seed'], meta['y'], meta['x'], crop)
        if not torch.isfinite(inputs).all():
            raise FloatingPointError(f'Non-finite input: {self.paths[index]}')
        return inputs, target, mask


class StepBatches(Sampler):
    """Resume at the next optimizer step, with exactly the same sample/crop order."""
    def __init__(self, count, batch, start, stop, seed, probabilities=None):
        self.count, self.batch, self.start, self.stop, self.seed = count, batch, start, stop, seed
        self.probabilities = None
        if probabilities is not None:
            p = np.asarray(probabilities, dtype=np.float64)
            if p.shape != (count,) or not np.isfinite(p).all() or (p < 0).any() or p.sum() <= 0:
                raise ValueError('Sampling probabilities must be finite, nonnegative and cover every file.')
            self.probabilities = p / p.sum()

    def __iter__(self):
        for step in range(self.start, self.stop):
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, step]))
            indices = (rng.integers(self.count, size=self.batch) if self.probabilities is None
                       else rng.choice(self.count, size=self.batch, p=self.probabilities))
            crops = rng.integers(0, 2**32, size=self.batch)
            yield [(int(i), int(c)) for i, c in zip(indices, crops)]

    def __len__(self):
        return self.stop-self.start

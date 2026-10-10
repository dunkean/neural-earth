"""Read-only near-sea coverage of the aligned teacher training/validation pairs.

The reconstructed interior is the same proxy as decoded_loss, not a new teacher
target or a full blended physical view. No reserved evaluation world is used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA, HOLDOUT_SEEDS, atomic_json, external_path
from distill.decoded_loss import reconstructed_height


def audit_pairs(audit_path, output, band_m=20.):
    if not np.isfinite(band_m) or band_m <= 0:
        raise ValueError('Near-sea band must be finite and positive.')
    audit_path = Path(audit_path)
    evidence = json.loads(audit_path.read_text())
    if evidence['mismatches']:
        raise ValueError('Aligned teacher pairs must pass their exact audit.')
    root = Path(evidence['dataset'])
    rows = []
    for row in evidence['rows']:
        if row['seed'] in HOLDOUT_SEEDS or not row['latent_pair_exact']:
            raise ValueError('Reserved or mismatched target in the training audit.')
        with np.load(root/'base'/row['file'], allow_pickle=False) as base, \
                np.load(root/'decoder'/row['file'], allow_pickle=False) as decoder:
            metadata = json.loads(str(base['metadata']))
            if metadata['seed'] != row['seed'] or metadata['split'] != row['split']:
                raise ValueError('Teacher pair metadata changed since alignment audit.')
            if not np.array_equal(base['target'][:4, :64, :64], decoder['latents']):
                raise ValueError('Teacher latent pairs changed since alignment audit.')
            target = torch.from_numpy(base['target'][:, :64, :64].astype(np.float32))[None]
            residual = torch.from_numpy(decoder['target'].astype(np.float32))[None]
            mask = decoder['mask'][..., 64:-64, 64:-64] > 0
            with torch.inference_mode():
                height = reconstructed_height(residual, target)[0].numpy()
            if not np.isfinite(height).all():
                raise ValueError('Non-finite teacher height proxy.')
            near = mask & (np.abs(height) <= band_m)
            n = int(near.sum())
            land, sea = int((near & (height > 0)).sum()), int((near & (height < 0)).sum())
            valid = height[mask]
            rows.append(dict(file=row['file'], split=row['split'], profile=row['profile'],
                seed=row['seed'], valid_pixels=int(mask.sum()), near_sea_pixels=n,
                near_land_pixels=land, near_sea_negative_pixels=sea,
                mixed_near_coast=bool(land and sea),
                current_height_loss_scale_m=max(float(valid.std()), 5.) if valid.size else 5.))
    summary = {}
    for split in ('train', 'val'):
        selected = [r for r in rows if r['split'] == split]
        summary[split] = dict(pairs=len(selected), near_sea_pairs=sum(r['near_sea_pixels'] > 0 for r in selected),
            mixed_near_coast_pairs=sum(r['mixed_near_coast'] for r in selected),
            near_sea_pixels=sum(r['near_sea_pixels'] for r in selected),
            near_land_pixels=sum(r['near_land_pixels'] for r in selected),
            near_sea_negative_pixels=sum(r['near_sea_negative_pixels'] for r in selected))
    result = dict(band_m=band_m, summary=summary, rows=rows, accepted=False,
        audit_sha256=hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        dataset_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        note='Teacher interior height proxy only; no held-out sites or new training labels.')
    atomic_json(external_path(output), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=Path, default=DATA/'ckpt/base-decoded/paired-audit.json')
    parser.add_argument('--output', type=Path, default=DATA/'eval/shore-training-coverage.json')
    parser.add_argument('--band-m', type=float, default=20.)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(audit_pairs(args.audit, args.output, args.band_m)['summary']), flush=True)


if __name__ == '__main__':
    main()

"""Measure gradient errors across coarse-cell boundaries on stored base targets.

Compare boundary and interior errors relative to the teacher's own gradient
energy. Raw error variance alone can rise wherever the teacher is less smooth.
No target generation, held-out-site access, or runtime change.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA, HOLDOUT_SEEDS, atomic_json, external_path
from distill.features import base_features
from distill.shore_probe import fingerprint
from distill.student import load_student
from distill.train import centre, lowfreq_height_mae


def gradient_profiles(prediction, target, mask, period=32):
    """Per-channel per-phase mean squared increments, right endpoint phase."""
    sums = prediction.new_zeros(3, prediction.shape[1], period)
    counts = prediction.new_zeros(period)
    for axis in (-1, -2):
        dp, dt = prediction.diff(dim=axis), target.diff(dim=axis)
        valid = mask.narrow(axis, 1, mask.shape[axis]-1)*mask.narrow(axis, 0, mask.shape[axis]-1)
        phases = torch.arange(1, mask.shape[axis], device=mask.device) % period
        for phase in range(period):
            indices = (phases == phase).nonzero().flatten()
            selected_mask = valid.index_select(axis, indices)
            count = selected_mask.sum()
            counts[phase] += count
            for i, value in enumerate(((dp-dt).square(), dt.square(), dp.square())):
                sums[i, :, phase] += (value.index_select(axis, indices)*selected_mask).sum((0, 2, 3))
    return sums/counts.clamp_min(1)[None, None], counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DATA/'ckpt/candidates/base-step100000-f25dbf12474a.pt')
    parser.add_argument('--output', type=Path, default=DATA/'eval/boundary-gradient')
    parser.add_argument('--train-count', type=int, default=500)
    args = parser.parse_args()
    output = external_path(args.output)
    torch.set_num_threads(4)
    model, saved = load_student(args.checkpoint, 'cuda:0')
    if model.config.stage != 'base':
        raise ValueError('Base checkpoint required.')
    dataset = DATA/'crops/main'
    train_paths = sorted((dataset/'base').glob('train-*.npz'))
    val_paths = sorted((dataset/'base').glob('val-*.npz'))
    policy = saved['sampling_policy']
    manifest_sha = fingerprint(dataset/'manifest.json')
    if (policy['train_files'] != [p.name for p in train_paths] or
            policy['dataset_manifest_digest'] != manifest_sha):
        raise ValueError('Sampling policy no longer matches this dataset.')
    probabilities = np.asarray(policy['probabilities'], dtype=np.float64)
    indices = np.random.default_rng(20261010).choice(len(train_paths), args.train_count, replace=False,
                                                    p=probabilities/probabilities.sum())
    coverage_path = DATA/'eval/rare-training-coverage.json'
    coverage = json.loads(coverage_path.read_text())
    if coverage['dataset_manifest_digest'] != manifest_sha:
        raise ValueError('Rare classifications refer to another dataset.')
    classifications = {row['file']: row for row in coverage['rows']}
    source = dict(checkpoint_sha256=fingerprint(args.checkpoint), dataset_manifest_sha256=manifest_sha,
                  coverage_sha256=fingerprint(coverage_path), code_sha256=fingerprint(__file__),
                  train_sampling='Without replacement, parent probabilities, seed20261010',
                  period=32, floor=1e-10)
    rows, profiles, seeds = [], {}, {'train':set(), 'val':set()}
    with torch.inference_mode():
        for split, paths in (('train', [train_paths[i] for i in indices]), ('val', val_paths)):
            profiles[split] = []
            for number, path in enumerate(paths):
                payload = path.read_bytes()
                with np.load(io.BytesIO(payload), allow_pickle=False) as sample:
                    meta = json.loads(str(sample['metadata']))
                    classification = classifications[path.name]
                    if (meta['seed'] in HOLDOUT_SEEDS or meta['split'] != split or meta['y'] % 32 or meta['x'] % 32 or
                            classification['seed'] != meta['seed'] or classification['split'] != split):
                        raise ValueError('Reserved seed, split, phase or classification mismatch.')
                    seeds[split].add(meta['seed'])
                    target = torch.from_numpy(sample['target'].astype(np.float32))[None].cuda()
                    mask = torch.from_numpy(sample['mask'].astype(np.float32))[None].cuda()
                    h, w = target.shape[-2:]
                    if h != w:
                        raise ValueError('Square complete base crops required.')
                    inputs = base_features(sample['coarse'], meta['coarse_y'], meta['coarse_x'], meta['seed'],
                        meta['y']-model.halo, meta['x']-model.halo, h+2*model.halo, sample['histogram'])[None].cuda()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    prediction = centre(model(inputs), target)
                energy, counts = gradient_profiles(prediction.float(), target, mask)
                # A shared teacher scale across phases separates relative error
                # from the possibility that teacher boundaries have larger jumps.
                teacher_scale = (energy[1]*counts).sum(-1)/counts.sum().clamp_min(1)
                relative = energy/teacher_scale.clamp_min(1e-10)[None, :, None]
                values = relative.cpu().numpy().astype(np.float64)
                profiles[split].append(values)
                rows.append(dict(file=path.name, split=split, seed=meta['seed'], kinds=classification['kinds'],
                    target_sha256=fingerprint(path), relative_energy_by_channel_phase=values.tolist(),
                    height_mae_m_proxy=float(lowfreq_height_mae(prediction, target, mask))))
                if (number+1) % 100 == 0:
                    print(json.dumps(dict(split=split, windows=number+1)), flush=True)
    if seeds['train'] & seeds['val']:
        raise ValueError('Training and validation seed overlap.')
    summary = {}
    for split in ('train', 'val'):
        groups = ['all']+sorted({kind for row in rows if row['split'] == split for kind in row['kinds']})
        for group in groups:
            selected = [row for row in rows if row['split'] == split and (group == 'all' or group in row['kinds'])]
            profile = np.mean([row['relative_energy_by_channel_phase'] for row in selected], axis=0)
            edge, interior = profile[..., 0], profile[..., 8:25].mean(-1)
            summary[split+'/'+group] = dict(windows=len(selected), relative_energy_by_channel_phase=profile.tolist(),
                edge_error_by_channel=edge[0].tolist(), interior_error_by_channel=interior[0].tolist(),
                edge_teacher_by_channel=edge[1].tolist(), interior_teacher_by_channel=interior[1].tolist(),
                edge_error_teacher_ratio=(edge[0]/edge[1].clip(1e-10)).tolist(),
                interior_error_teacher_ratio=(interior[0]/interior[1].clip(1e-10)).tolist(),
                height_mae_m_proxy=sum(row['height_mae_m_proxy'] for row in selected)/len(selected))
    atomic_json(output/'report.json', dict(source=source, summary=summary, rows=rows,
        accepted=False, validation_seed_overlap=0, note='Latent gradient diagnosis, no physical acceptance.'))
    print(json.dumps({k:{name:value for name,value in v.items() if name != 'relative_energy_by_channel_phase'}
                      for k,v in summary.items()}), flush=True)


if __name__ == '__main__':
    main()

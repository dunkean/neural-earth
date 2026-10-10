"""Causal ablation of the base's direct nearest-coarse height contribution.

Keep the CNN unchanged. Replace only its direct coarse contribution to channel
five by a bilinear version, with a scalar fitted on train targets only. Full
validation tests whether this actually reduces gradient errors. No runtime edit.
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from distill.common import DATA, HOLDOUT_SEEDS, atomic_json
from distill.features import base_features
from distill.shore_probe import fingerprint
from distill.student import load_student
from distill.train import centre, lowfreq_height_mae


def gradient_regression(prediction, target, delta, mask):
    terms = prediction.new_zeros(3)
    for axis in (-1, -2):
        error = (prediction-target).diff(dim=axis)
        change = delta.diff(dim=axis)
        gt = target.diff(dim=axis)
        valid = mask.narrow(axis, 0, mask.shape[axis]-1)*mask.narrow(axis, 1, mask.shape[axis]-1)
        count = valid.sum().clamp_min(1)
        scale = ((gt.square()*valid).sum()/count).clamp_min(1e-10)
        terms += torch.stack(((error*change*valid).sum(), (change.square()*valid).sum(),
                              (error.square()*valid).sum()))/(count*scale)/2
    return terms


def main():
    root = DATA/'eval/coarse-skip-audit'
    checkpoint = DATA/'ckpt/candidates/base-step100000-f25dbf12474a.pt'
    dataset = DATA/'crops/main'
    model, saved = load_student(checkpoint, 'cuda:0')
    torch.set_num_threads(4)
    train_paths = sorted((dataset/'base').glob('train-*.npz'))
    val_paths = sorted((dataset/'base').glob('val-*.npz'))
    policy = saved['sampling_policy']
    manifest_sha = fingerprint(dataset/'manifest.json')
    if (policy['train_files'] != [p.name for p in train_paths] or
            policy['dataset_manifest_digest'] != manifest_sha):
        raise ValueError('Preserved parent sampling policy no longer matches this dataset.')
    probabilities = np.asarray(policy['probabilities'], dtype=np.float64)
    indices = np.random.default_rng(20261010).choice(len(train_paths), 500, replace=False,
                                                    p=probabilities/probabilities.sum())
    coverage_path = DATA/'eval/rare-training-coverage.json'
    coverage = json.loads(coverage_path.read_text())
    if coverage['dataset_manifest_digest'] != manifest_sha:
        raise ValueError('Rare classification dataset differs.')
    classifications = {r['file']:r for r in coverage['rows']}
    source = dict(checkpoint_sha256=fingerprint(checkpoint), dataset_manifest_sha256=manifest_sha,
                  coverage_sha256=fingerprint(coverage_path), code_sha256=fingerprint(__file__),
                  sampling='500 train windows without replacement, preserved parent probabilities, seed20261010')
    total = np.zeros(3)
    alpha, rows = None, []
    seeds = {split:set() for split in ('train', 'val')}
    with torch.inference_mode():
        for split, paths in (('train', [train_paths[i] for i in indices]), ('val', val_paths)):
            for number, path in enumerate(paths):
                payload = path.read_bytes()
                with np.load(io.BytesIO(payload), allow_pickle=False) as sample:
                    meta = json.loads(str(sample['metadata']))
                    classification = classifications[path.name]
                    if (meta['seed'] in HOLDOUT_SEEDS or meta['split'] != split or meta['y'] % 32 or meta['x'] % 32 or
                            classification['seed'] != meta['seed'] or classification['split'] != split):
                        raise ValueError('Reserved seed, split/phase or classification mismatch.')
                    seeds[split].add(meta['seed'])
                    target = torch.from_numpy(sample['target'].astype(np.float32))[None].cuda()
                    mask = torch.from_numpy(sample['mask'].astype(np.float32))[None].cuda()
                    h, w = target.shape[-2:]
                    if h != w:
                        raise ValueError('Square complete base targets required.')
                    inputs = base_features(sample['coarse'], meta['coarse_y'], meta['coarse_x'], meta['seed'],
                        meta['y']-model.halo, meta['x']-model.halo, h+2*model.halo, sample['histogram'])[None].cuda()
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    prediction = centre(model(inputs), target)
                nearest = inputs[:, 10:16]
                smooth = F.interpolate(nearest[..., ::32, ::32], size=nearest.shape[-2:],
                                       mode='bilinear', align_corners=False)
                change = centre(F.conv2d(smooth-nearest, model.direct.weight[4:5, 10:16]), target)
                regression = gradient_regression(prediction[:, 4:5], target[:, 4:5], change, mask).cpu().numpy()
                if not np.isfinite(regression).all():
                    raise ValueError('Non-finite regression terms.')
                row = dict(file=path.name, split=split, seed=meta['seed'], kinds=classification['kinds'],
                    target_sha256=fingerprint(path), dot=float(regression[0]), norm=float(regression[1]),
                    baseline_relative_gradient_mse=float(regression[2]),
                    baseline_height_mae_m_proxy=float(lowfreq_height_mae(prediction, target, mask)))
                if split == 'train':
                    total += regression
                else:
                    for label, coefficient in (('fitted', alpha), ('full', 1.)):
                        corrected = prediction.clone()
                        corrected[:, 4:5] += coefficient*change
                        row[label+'_height_mae_m_proxy'] = float(lowfreq_height_mae(corrected, target, mask))
                        row[label+'_relative_gradient_mse'] = float(regression[2]+2*coefficient*regression[0]+coefficient**2*regression[1])
                rows.append(row)
                if (number+1) % 100 == 0:
                    print(json.dumps(dict(split=split, windows=number+1)), flush=True)
            if split == 'train':
                if total[1] <= 0:
                    raise ValueError('No direct coarse gradient intervention to measure.')
                unconstrained = float(-total[0]/total[1])
                alpha = float(np.clip(unconstrained, 0., 1.))
                atomic_json(root/'fit.json', dict(source=source, alpha=alpha, unconstrained_alpha=unconstrained,
                    regression_totals=total.tolist(), accepted=False, train_files=[p.name for p in paths]))
    if seeds['train'] & seeds['val']:
        raise ValueError('Train/validation seed overlap.')
    summary = {}
    keys = ('baseline_relative_gradient_mse', 'fitted_relative_gradient_mse', 'full_relative_gradient_mse',
            'baseline_height_mae_m_proxy', 'fitted_height_mae_m_proxy', 'full_height_mae_m_proxy')
    val = [r for r in rows if r['split'] == 'val']
    groups = ['all']+sorted({k for r in val for k in r['kinds']})
    for group in groups:
        selected = [r for r in val if group == 'all' or group in r['kinds']]
        summary[group] = dict(windows=len(selected), **{k:sum(r[k] for r in selected)/len(selected) for k in keys})
    atomic_json(root/'report.json', dict(source=source, alpha=alpha, summary=summary, rows=rows,
        validation_seed_overlap=0, accepted=False,
        note='Causal direct-height coarse-skip ablation only; independent full-base validation, no decoder or physical acceptance.'))
    print(json.dumps(dict(alpha=alpha, summary=summary)), flush=True)


if __name__ == '__main__':
    main()

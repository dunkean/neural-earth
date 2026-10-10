"""Diagnose lattice errors on full stored base crops, without training.

Fit an additive phase template using train targets only, then test it on all
validation crops. Report both phase mean and variance: a zero-mean alternating
variance cannot be repaired by subtracting a constant bias. No runtime change.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA, HOLDOUT_SEEDS, atomic_json, atomic_write, external_path
from distill.features import base_features
from distill.student import load_student
from distill.train import centre, lowfreq_height_mae


def phase_sums(value, mask, period=8):
    channels, h, w = value.shape
    if h % period or w % period:
        raise ValueError('Crops must contain complete lattice periods.')
    shaped = (value*mask).reshape(channels, h//period, period, w//period, period)
    counts = mask.reshape(1, h//period, period, w//period, period).sum((1, 3))
    return shaped.sum((1, 3)), counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DATA/'ckpt/candidates/base-step100000-f25dbf12474a.pt')
    parser.add_argument('--dataset', type=Path, default=DATA/'crops/main')
    parser.add_argument('--output', type=Path, default=DATA/'eval/lattice-error')
    parser.add_argument('--train-count', type=int, default=300)
    parser.add_argument('--period', type=int, choices=(8, 16, 32), default=8)
    args = parser.parse_args()
    if args.train_count < 2:
        parser.error('At least two training windows required.')
    output = external_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    period = args.period
    torch.set_num_threads(4)
    model, saved = load_student(args.checkpoint, 'cuda:0')
    if model.config.stage != 'base':
        raise ValueError('Base checkpoint required.')
    coverage_path = DATA/'eval/rare-training-coverage.json'
    coverage = json.loads(coverage_path.read_text())
    manifest_sha = hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest()
    if coverage['dataset_manifest_digest'] != manifest_sha:
        raise ValueError('Rare classifications refer to another dataset.')
    classifications = {row['file']: row for row in coverage['rows']}
    source = dict(checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  dataset_manifest_sha256=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest(),
                  coverage_sha256=hashlib.sha256(coverage_path.read_bytes()).hexdigest(), period=period,
                  code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  note='Full stored base targets; no decoder or new labels. Train-only phase fit, independent validation.')
    train_paths = sorted((args.dataset/'base').glob('train-*.npz'))
    val_paths = sorted((args.dataset/'base').glob('val-*.npz'))
    if not train_paths or not val_paths:
        raise ValueError('Training and validation targets required.')
    indices = np.linspace(0, len(train_paths)-1, min(args.train_count, len(train_paths))).round().astype(int)
    def empty_totals():
        return dict(count=np.zeros((1, period, period)), total=np.zeros((5, period, period)),
                    square=np.zeros((5, period, period)))
    totals = {split:empty_totals() for split in ('train', 'val')}
    group_totals = {}
    seeds = {split:set() for split in totals}
    template, rows = None, []
    with torch.inference_mode():
        for split, paths in (('train', [train_paths[i] for i in indices]), ('val', val_paths)):
            for number, path in enumerate(paths):
                payload = path.read_bytes()
                import io
                with np.load(io.BytesIO(payload), allow_pickle=False) as sample:
                    meta = json.loads(str(sample['metadata']))
                    if meta['seed'] in HOLDOUT_SEEDS or meta['split'] != split or meta['y'] % 32 or meta['x'] % 32:
                        raise ValueError('Reserved seed, split mismatch or phase misalignment.')
                    classification = classifications[path.name]
                    if classification['seed'] != meta['seed'] or classification['split'] != split:
                        raise ValueError('Rare classification metadata changed.')
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
                error = (prediction-target)[0].float().cpu().numpy().astype(np.float64)
                valid = mask[0].cpu().numpy().astype(np.float64)
                summed, counts = phase_sums(error, valid, period)
                square, _ = phase_sums(error**2, valid, period)
                totals[split]['total'] += summed
                totals[split]['square'] += square
                totals[split]['count'] += counts
                for kind in classification['kinds']:
                    data = group_totals.setdefault(split+'/'+kind, empty_totals())
                    data['total'] += summed
                    data['square'] += square
                    data['count'] += counts
                row = dict(file=path.name, split=split, seed=meta['seed'], profile=meta['profile'],
                    kinds=classification['kinds'],
                    target_sha256=hashlib.sha256(payload).hexdigest(),
                    baseline_height_mae_m_proxy=float(lowfreq_height_mae(prediction, target, mask)))
                if template is not None:
                    correction = torch.from_numpy(np.tile(template.astype(np.float32), (1, h//period, w//period)))[None].cuda()
                    corrected = prediction-correction
                    row['corrected_height_mae_m_proxy'] = float(lowfreq_height_mae(corrected, target, mask))
                    row['baseline_latent_mse'] = float(((prediction-target).square()*mask).sum()/(5*mask.sum()).clamp_min(1))
                    row['corrected_latent_mse'] = float(((corrected-target).square()*mask).sum()/(5*mask.sum()).clamp_min(1))
                rows.append(row)
                if (number+1) % 100 == 0:
                    print(json.dumps(dict(split=split, windows=number+1)), flush=True)
            if split == 'train':
                if (totals[split]['count'] <= 0).any():
                    raise ValueError('Training misses some lattice phases.')
                template = totals[split]['total']/totals[split]['count']
                template -= template.mean((-2, -1), keepdims=True)
    if seeds['train'] & seeds['val']:
        raise ValueError('Training and validation seed overlap.')
    summary = {}
    for split, data in (totals | group_totals).items():
        count = data['count'].clip(1)
        mean = data['total']/count
        variance = (data['square']/count-mean**2).clip(0)
        base_split, _, kind = split.partition('/')
        summary[split] = dict(windows=sum(r['split'] == base_split and (not kind or kind in r['kinds']) for r in rows),
            phase_mean_rms_by_channel=np.sqrt(((mean-mean.mean((-2,-1), keepdims=True))**2).mean((-2,-1))).tolist(),
            phase_variance_max_min_by_channel=(variance.max((-2,-1))/variance.min((-2,-1)).clip(1e-12)).tolist())
    validation = [r for r in rows if r['split'] == 'val']
    for name in ('baseline_height_mae_m_proxy', 'corrected_height_mae_m_proxy',
                 'baseline_latent_mse', 'corrected_latent_mse'):
        summary['val'][name] = sum(row[name] for row in validation)/len(validation)
    atomic_write(output/'template.npz', lambda f: np.savez_compressed(f, template=template, **{
        f'{split}_{key}':value for split, data in totals.items() for key, value in data.items()}))
    atomic_json(output/'report.json', dict(source=source, summary=summary, rows=rows, accepted=False,
        validation_seed_overlap=0, note='Diagnostic mean-bias subtraction only; no new NN, no physical acceptance or runtime change.'))
    print(json.dumps(dict(period=period, summary={k:summary[k] for k in ('train', 'val')},
                          group_windows={k:v['windows'] for k,v in summary.items() if '/' in k})), flush=True)


if __name__ == '__main__':
    main()

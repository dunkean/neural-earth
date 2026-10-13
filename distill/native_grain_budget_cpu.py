"""Reproducible TRAIN-only grain budget from frozen TT/TS/ST/SS fields.

Projection statistics are computed per region, not from products of medians.
This is CPU-only and does not update weights or the ongoing paired training.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from distill.common import HOLDOUT_SEEDS, REPO, atomic_json, external_path
from distill.native_weight_quant import file_sha


def bands(values):
    g1 = gaussian_filter(values, 1., mode='reflect')
    g2 = gaussian_filter(values, 2., mode='reflect')
    g4 = gaussian_filter(values, 4., mode='reflect')
    return (values-g1, g1-g2, g2-g4)


def projection(actual, target):
    s, t = actual.flatten(), target.flatten()
    es, et, cross = np.mean(s*s), np.mean(t*t), np.mean(s*t)
    if et <= 1e-20:
        return dict(available=False, teacher_energy=float(et))
    slope = cross/et
    orthogonal = s-slope*t
    return dict(available=True, teacher_energy=float(et), student_energy=float(es),
        rms_ratio=float(np.sqrt(es/et)), correlated_slope=float(slope),
        cosine=float(cross/np.sqrt(es*et)) if es > 1e-20 else None,
        centered_correlation=float(np.corrcoef(s, t)[0, 1]) if min(s.std(), t.std()) > 1e-12 else None,
        orthogonal_energy_ratio=float(np.mean(orthogonal**2)/et),
        orthogonal_rms_units=float(np.sqrt(np.mean(orthogonal**2))),
        teacher_rms_units=float(np.sqrt(et)), error_energy_ratio=float(np.mean((s-t)**2)/et))


def noise_regression(error, teacher, noise):
    y, t, z = [v.flatten() for v in (error, teacher, noise)]
    reduced = np.stack((np.ones_like(t), t), axis=1)
    full = np.stack((np.ones_like(t), t, z), axis=1)
    reduced_coef = np.linalg.lstsq(reduced, y, rcond=None)[0]
    full_coef = np.linalg.lstsq(full, y, rcond=None)[0]
    r0 = y-reduced@reduced_coef
    r1 = y-full@full_coef
    sse0, sse1 = np.dot(r0, r0), np.dot(r1, r1)
    return dict(partial_r2_noise=float(max(0., (sse0-sse1)/sse0)) if sse0 > 1e-20 else None,
                noise_coefficient=float(full_coef[2]), teacher_coefficient=float(full_coef[1]))


def fused_noise(record, noise_seed):
    from terrain_diffusion.inference.world_pipeline import gaussian_noise_patch, linear_weight_window
    core_y, core_x = [((record['metadata'][k]+95)//96)*96 for k in ('y', 'x')]
    window = linear_weight_window(64, 'cpu', torch.float32).numpy().astype(np.float64)
    origins = [(core_y+dy, core_x+dx) for dy in (-32, 0, 32) for dx in (-32, 0, 32)]
    canvas, weights = np.zeros((128, 128)), np.zeros((128, 128))
    for y, x in origins:
        patch = gaussian_noise_patch(noise_seed+5820, y, x, 64, 64, channels=5, tile_h=64, tile_w=64)
        # The actual refinement path rounds the injected field to BF16.
        value = torch.from_numpy(patch[4]).bfloat16().float().numpy().astype(np.float64)
        yy, xx = y-(core_y-32), x-(core_x-32)
        canvas[yy:yy+64, xx:xx+64] += value*window
        weights[yy:yy+64, xx:xx+64] += window
    noise = canvas[32:96, 32:96]/weights[32:96, 32:96]
    direct = gaussian_noise_patch(noise_seed+5820, core_y, core_x, 64, 64,
                                 channels=5, tile_h=64, tile_w=64)
    direct = torch.from_numpy(direct[4]).bfloat16().float().numpy().astype(np.float64)
    if not np.allclose(noise, direct, rtol=0., atol=1e-14):
        raise ValueError('Noise overlap must be coherent; do not assume independent noise windows.')
    return noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Retain existing budgets.')
    report = json.loads((args.diagnostic/'report.json').read_text())
    if report['status'] != 'complete' or file_sha(args.diagnostic/'arrays.npz') != report['arrays_sha256']:
        raise ValueError('Complete immutable source diagnostic required.')
    torch.set_num_threads(4)
    root_rows, feature_rows, regression_rows, dc_rows = [], [], [], []
    pairs = [('SS', 'TT'), ('SS', 'ST'), ('TS', 'TT'), ('ST', 'TT')]
    with np.load(args.diagnostic/'arrays.npz', allow_pickle=False) as arrays:
        for record, row in zip(report['contract']['scenes'], report['rows'], strict=True):
            if record['metadata']['split'] != 'train' or record['metadata']['seed'] in HOLDOUT_SEEDS:
                raise ValueError('TRAIN only, excluding reserved worlds.')
            if record['metadata']['seed'] != row['seed'] or file_sha(Path(record['root'])/record['file']) != record['capture_sha256']:
                raise ValueError('Source capture identity changed.')
            i = row['index']
            cores = {s: arrays[f'{i:02d}_{s}_core'].astype(np.float64) for s in ('TT', 'TS', 'ST', 'SS')}
            q = {s: v[4]*38.6-31.4 for s, v in cores.items()}
            qb = {s: bands(v) for s, v in q.items()}
            noise_bands = bands(fused_noise(record, row['noise_seed']))
            identity = {k: row[k] for k in ('index', 'kind', 'coarse', 'seed')}
            for size in (32, 48):
                margin = (64-size)//2
                crop = lambda v: v[margin:64-margin, margin:64-margin]
                for actual, target in pairs:
                    for band in range(3):
                        root_rows.append(identity | dict(size=size, pair=actual+'/'+target, band=band+1,
                            **projection(crop(qb[actual][band]), crop(qb[target][band]))))
                for actual, target in [('SS', 'ST'), ('TS', 'TT')]:
                    for band in range(3):
                        regression_rows.append(identity | dict(size=size, pair=actual+'/'+target, band=band+1,
                            **noise_regression(crop(qb[actual][band]-qb[target][band]),
                                               crop(qb[target][band]), crop(noise_bands[band]))))
                for channel in range(4):
                    cb = {s: bands(v[channel]) for s, v in cores.items()}
                    for actual, target in [('SS', 'TT'), ('ST', 'TT'), ('TS', 'TT')]:
                        for band in range(3):
                            feature_rows.append(identity | dict(size=size, pair=actual+'/'+target,
                                channel=channel, band=band+1,
                                **projection(crop(cb[actual][band]), crop(cb[target][band]))))
            # Reproduce the earlier finite-difference D3 with code provenance.
            hp = lambda v: (v-gaussian_filter(v, 1., mode='reflect'))[16:48, 16:48]
            h = lambda v: np.sign(v)*v*v
            epsilon = .001
            for target in ('TT', 'ST'):
                a, t = q['SS'], q[target]
                loss = lambda offset: float(np.mean((hp(h(a+offset))-hp(h(t)))**2))
                dc_rows.append(identity | dict(target=target,
                    height_highpass_loss_dc_derivative=(loss(epsilon)-loss(-epsilon))/(2*epsilon),
                    root_highpass_loss_dc_delta=float(np.mean((hp(a+epsilon)-hp(t))**2)-np.mean((hp(a)-hp(t))**2))))
    grouped = defaultdict(list)
    for row in root_rows:
        if row['available']:
            grouped[(row['kind'], row['size'], row['pair'], row['band'])].append(row)
    summaries = []
    for (kind, size, pair, band), rows in sorted(grouped.items()):
        summaries.append(dict(kind=kind, size=size, pair=pair, band=band, n=len(rows),
            **{key: float(np.median([r[key] for r in rows])) for key in
               ('rms_ratio', 'correlated_slope', 'orthogonal_energy_ratio', 'orthogonal_rms_units', 'teacher_rms_units')}))
    atomic_json(output, dict(status='complete', root_rows=root_rows, feature_rows=feature_rows,
        noise_regressions=regression_rows, dc_rows=dc_rows, summaries=summaries,
        source_report_sha256=file_sha(args.diagnostic/'report.json'), source_arrays_sha256=report['arrays_sha256'],
        source_code_sha256=file_sha(REPO/'distill/native_grain_budget_cpu.py'), updates=0,
        cuda_initialized=torch.cuda.is_initialized(),
        scope='TRAIN-only descriptive projections. Coherent BF16 noise, all pixels, centered32/48; no optimizer or model changes.',
        limitations=['Noise regression is association, not causal proof.',
                     'Finite crop band means need not vanish; projections use actual uncentered Gram statistics.',
                     'Window rounding floor is not inferred from an independent-noise Monte Carlo. Measure it on GPU.']))
    if torch.cuda.is_initialized():
        raise ValueError('This diagnostic must not initialize CUDA.')
    print(json.dumps(dict(status='complete', root_rows=len(root_rows), feature_rows=len(feature_rows),
                         noise_regressions=len(regression_rows), cuda_initialized=False)), flush=True)


if __name__ == '__main__':
    main()

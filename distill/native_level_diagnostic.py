"""Matched TRAIN-only level attribution on fresh two-stage teacher fields.

32 independent TRAIN worlds, 24 plains/coasts and 8 wet mountains, with all
three coarse variants. TT/TS/ST/SS use identical noise, conditioning, geometry,
fusion, numerical batches and physical footprint. Regression is descriptive;
no parameters are fitted or applied to served models.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from distill.common import REPO, HOLDOUT_SEEDS, atomic_json, atomic_write, external_path
from distill.decoded_loss import reconstructed_height
from distill.native_base import teacher_source
from distill.native_basis_training import inventory
from distill.native_decoded import DecodedRegion, OriginalDecoder, blur, decoded_region_forward
from distill.native_field_auxiliary import low_height
from distill.native_fusion import forward_batches, window_outputs
from distill.native_weight_quant import load_weight_quant, file_sha
from distill.wet_mountains import KIND


def select_train(records):
    selected, used = [], set()
    for kind in ('desert-plain', 'temperate-plain', 'wet-complex-coast', 'low-plain-sea', KIND):
        candidates = [r for r in records if r['case_kind'] == kind]
        profiles = sorted({r['metadata']['profile'] for r in candidates})
        for i in range(8 if kind == KIND else 6):
            variant = ('original', 'coarse4', 'coarse2')[i % 3]
            profile = profiles[i % len(profiles)]
            pool = [r for r in candidates if r['coarse_variant'] == variant and
                    r['metadata']['profile'] == profile and r['metadata']['seed'] not in used]
            if not pool:
                pool = [r for r in candidates if r['coarse_variant'] == variant and
                        r['metadata']['seed'] not in used]
            if not pool:
                raise ValueError(f'Missing independent TRAIN stratum: {kind}/{variant}')
            pool.sort(key=lambda r: hashlib.sha256(('level-diagnostic-v1/' +
                str(r['metadata']['seed']) + '/' + r['file']).encode()).hexdigest())
            record = pool[0]
            if record['metadata']['split'] != 'train' or record['metadata']['seed'] in HOLDOUT_SEEDS:
                raise ValueError('Reserved evaluation world entered TRAIN diagnostic.')
            selected.append(record)
            used.add(record['metadata']['seed'])
    return selected


def height_stats(actual, reference):
    delta = actual.float() - reference.float()
    return dict(bias_m=float(delta.mean()), mae_m=float(delta.abs().mean()),
        centered_mae_m=float((delta-delta.mean()).abs().mean()),
        coast_disagreement=float(((actual > 0) != (reference > 0)).float().mean()))


def field_stats(actual, reference):
    crop = (slice(None), slice(16, 48), slice(16, 48))
    q_actual, q_reference = actual[4:5].float()*38.6-31.4, reference[4:5].float()*38.6-31.4
    smooth_delta = blur((q_actual-q_reference)[None], 8.)
    jacobian = 2*blur(q_reference[None], 8.).abs().clamp_min(1.)
    low_linear = (smooth_delta*jacobian)[0, :, 16:48, 16:48]
    lf = blur((actual[:4]-reference[:4]).float()[:, None], 8.)[:, 0, 16:48, 16:48]
    regressions = []
    for a, r in zip(actual[crop].double(), reference[crop].double()):
        x, y = r.flatten(), a.flatten()
        centered_x = x-x.mean()
        variance = centered_x.square().mean()
        slope = float((centered_x*(y-y.mean())).mean()/variance) if variance > 1e-15 else None
        regressions.append(dict(student_gain=slope,
            student_offset=float(y.mean()-slope*x.mean()) if slope is not None else None,
            bias=float((y-x).mean()), std_teacher=float(x.std(unbiased=False))))
    return dict(q_bias=float((q_actual-q_reference)[:, 16:48, 16:48].mean()),
        teacher_jacobian_lf_rms_m=float(low_linear.square().mean().sqrt()),
        c0_c3_lf_rms=[float(v.square().mean().sqrt()) for v in lf],
        channel_regressions=regressions)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    if (output/'report.json').exists() or (output/'contract.json').exists():
        raise ValueError('Use a fresh diagnostic directory; no implicit restart.')
    records, _, _ = inventory()
    selected = select_train(records)
    contract = dict(source=str(args.source), source_sha256=file_sha(args.source), seed=2026101321,
        stage_names={'TT': 'teacher first, teacher second', 'TS': 'teacher first, student second',
                     'ST': 'student first, teacher second', 'SS': 'student first, student second'},
        scenes=[r | dict(capture_sha256=file_sha(Path(r['root'])/r['file'])) for r in selected],
        code_sha256={name: file_sha(REPO/'distill'/name) for name in
                     ('native_level_diagnostic.py', 'native_decoded.py', 'native_fusion.py')},
        updates=0, calibration=False, fresh_noise=True, split='train', sigma_latent=8.,
        footprint='Central32x32 latent /256x256 decoded physical pixels.',
        numerical_batch=4, accepted=False)
    atomic_json(output/'contract.json', contract)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.source, map_location='cpu', weights_only=True, mmap=True)
    model = load_weight_quant(saved, 'cuda')
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    from terrain_inference import prepare_models
    teacher = EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    prepare_models(SimpleNamespace(coarse_model=teacher, base_model=teacher, decoder_model=teacher))
    decoder = OriginalDecoder('cuda')
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64, 'cuda', torch.float32)
    rows, arrays = [], {}
    for i, record in enumerate(selected):
        region = DecodedRegion(Path(record['root'])/record['file'], 'cuda')
        noise_seed = int(np.random.SeedSequence([contract['seed'], i, 0xA0]).generate_state(1)[0])
        region.refresh_noise(teacher, weight, noise_seed)
        second = region.stages[1]
        with model.cached_weights():
            ss = decoded_region_forward(model, region, weight)
            ts_velocity = forward_batches(model, second['inputs'], second['labels'], second['condition'])
        ts = region.core_crop(window_outputs(second['inputs'], ts_velocity, second['labels'], region.dtype), weight)
        st_velocity = forward_batches(teacher, ss['inputs1'].to(region.dtype),
                                     second['labels'].to(region.dtype), second['condition'].to(region.dtype))
        st = region.core_crop(window_outputs(ss['inputs1'], st_velocity, second['labels'], region.dtype), weight)
        fields = dict(TT=region.target, TS=ts, ST=st, SS=ss['core'])
        heights = {name: reconstructed_height(decoder(core[None], region.decoder_noise),
                        core[None])[0, 0, 64:-64, 64:-64] for name, core in fields.items()}
        if any(h.shape != (256, 256) for h in heights.values()):
            raise ValueError('Matched physical footprint changed.')
        lows = {name: low_height(core[None])[0, 0, 16:48, 16:48] for name, core in fields.items()}
        stats = {name: {'3': height_stats(lows[name], lows['TT']),
                       '0': height_stats(heights[name], heights['TT']),
                       'fields': field_stats(core, fields['TT'])} for name, core in fields.items()}
        row = dict(index=i, kind=record['case_kind'], coarse=record['coarse_variant'],
                   seed=record['metadata']['seed'], profile=record['metadata']['profile'],
                   noise_seed=noise_seed, split='train', stages=stats)
        rows.append(row)
        for name, core in fields.items():
            arrays[f'{i:02d}_{name}_core'] = core.float().cpu().numpy()
            arrays[f'{i:02d}_{name}_high'] = heights[name].float().cpu().numpy()
        print(json.dumps(dict(index=i, kind=row['kind'], coarse=row['coarse'], seed=row['seed'],
              bias_lod0={n: s['0']['bias_m'] for n, s in stats.items()})), flush=True)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['kind']].append(row)
    summary = {kind: {stage: {str(lod): dict(
        abs_bias_mean_m=float(np.mean([abs(r['stages'][stage][str(lod)]['bias_m']) for r in group])),
        mae_mean_m=float(np.mean([r['stages'][stage][str(lod)]['mae_m'] for r in group])))
        for lod in (3, 0)} for stage in ('TT', 'TS', 'ST', 'SS')} for kind, group in grouped.items()}
    atomic_write(output/'arrays.npz', lambda f: np.savez_compressed(f, **arrays))
    atomic_json(output/'report.json', dict(status='complete', contract=contract, rows=rows,
        summary=summary, gpu=torch.cuda.get_device_name(), arrays_sha256=file_sha(output/'arrays.npz')))


if __name__ == '__main__':
    main()

"""Train the native T2 base using frozen original-decoder terrain supervision.

This physical pilot preserves its source and every validation checkpoint. Its
EMA is an internal candidate: full pipeline morphology and cold seams decide
whether to continue training. No base weights are exported to the application.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from distill.common import (DATA, REPO, REVISION, HOLDOUT_SEEDS, append_json,
                            atomic_json, atomic_write, external_path)
from distill.native_base import NativeBase, teacher_source
from distill.native_decoded import (DecodedRegion, OriginalDecoder,
    decoded_region_forward, decoded_loss, physical_terms)
from distill.native_fusion_train import check_teacher

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def validate(model, files, weight, decoder, variance):
    rows = []
    for path in files:
        region = DecodedRegion(path, 'cuda')
        result = decoded_region_forward(model, region, weight)
        terms = physical_terms(result, region, decoder, variance)
        feature_mse = ((result['core'][:4]-region.target[:4]).square() /
                       variance[:4, None, None]).mean()
        rows.append(dict(file=path.name, seed=region.metadata['seed'],
            profile=region.metadata['profile'], core=region.core,
            feature_relative_mse=float(feature_mse),
            **{key: float(value) for key, value in terms.items()}))
    profiles = {}
    for profile in sorted({row['profile'] for row in rows}):
        selected = [row for row in rows if row['profile'] == profile]
        profiles[profile] = {key: float(np.mean([row[key] for row in selected]))
            for key in ('height_mae_m', 'low_height_mae_m', 'height_gradient',
                        'height_tpi', 'feature_relative_mse')}
    scores = {profile: values['height_mae_m']/100. +
              .25*values['low_height_mae_m']/100. +
              .5*values['height_gradient'] + .25*values['height_tpi']
              for profile, values in profiles.items()}
    return dict(rows=rows, profiles=profiles, selection_score=max(scores.values()),
        profile_scores=scores,
        median_height_mae_m=float(np.median([row['height_mae_m'] for row in rows])),
        feature_relative_mse=float(np.mean([row['feature_relative_mse'] for row in rows])),
        accepted=False, note='Aligned original-decoder interior proxy; full pipeline evaluation still required.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATA/'crops/native-scenes-v1')
    parser.add_argument('--source', type=Path, default=DATA/'ckpt/native128x2-pilot/best.pt')
    parser.add_argument('--output', type=Path, default=DATA/'ckpt/native128x2-decoded-pilot')
    parser.add_argument('--statistics', type=Path, default=DATA/'eval/native-decoded-probe/channel-statistics.json')
    parser.add_argument('--controls', type=Path, default=DATA/'eval/native-decoded-controls')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--eval-every', type=int, default=250)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--fresh-noise-fraction', type=float, default=.25)
    parser.add_argument('--seed', type=int, default=20261012)
    args = parser.parse_args()
    if (min(args.steps, args.eval_every) < 1 or not np.isfinite(args.lr) or args.lr <= 0 or
            not np.isfinite(args.fresh_noise_fraction) or not 0 <= args.fresh_noise_fraction <= 1):
        parser.error('Positive finite physical training settings required.')
    root, output = external_path(args.dataset), external_path(args.output)
    if json.loads((root/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Complete native capture required.')
    manifest = json.loads((root/'manifest.json').read_text())
    worlds = {split: {row['metadata']['seed'] for row in manifest['scenes']
                     if row['metadata']['split'] == split} for split in ('train', 'val')}
    if worlds['train'] & worlds['val'] or (worlds['train'] | worlds['val']) & HOLDOUT_SEEDS:
        raise ValueError('Training, validation or reserved worlds overlap.')
    stats = json.loads(args.statistics.read_text())
    if (stats['split'] != 'train' or len(stats['scenes']) < 150 or
            stats['dataset_manifest_sha256'] != digest(root/'manifest.json')):
        raise ValueError('Audited global training variance for this bank is required.')
    controls = [json.loads(path.read_text()) for path in args.controls.glob('*/decoder-control.json')]
    required_profiles = {row['metadata']['profile'] for row in manifest['scenes']}
    if ({row['file']['profile'] for row in controls} != required_profiles or
            any(row['status'] != 'complete' or row['decoder_node_max_abs'] > 1e-5 or
                not row['proxy_within_floor'] or
                row.get('control_basis') != 'same runtime latents, original decoder, full pipeline'
                for row in controls)):
        raise ValueError('Original-decoder proxy controls for every native profile are required.')
    source = torch.load(args.source, map_location='cpu', weights_only=True)
    source_path = teacher_source()
    contract = dict(config=source['config'], source_sha256=digest(args.source),
        source_step=source['step'], dataset_manifest_sha256=digest(root/'manifest.json'),
        statistics_sha256=digest(args.statistics),
        decoder_weights_sha256=digest(source_path.parent/'decoder_model/diffusion_pytorch_model.safetensors'),
        teacher_weights_sha256=digest(source_path/'diffusion_pytorch_model.safetensors'),
        controls={path.parent.name: digest(path) for path in args.controls.glob('*/decoder-control.json')},
        core=64, alignment=96, initialization_windows=25, refinement_windows=9,
        lr=args.lr, seed=args.seed, fresh_noise_fraction=args.fresh_noise_fraction,
        clip_norm=1., selection='worst-profile physical proxy',
        producer_sha256={name: digest(REPO/name) for name in (
            'distill/native_decoded_train.py', 'distill/native_decoded.py',
            'distill/native_base.py', 'distill/native_fusion.py', 'distill/decoded_loss.py')})
    contract_path = output/'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Physical run contract changed; inspect before resuming.')
    atomic_json(contract_path, contract)
    saved = torch.load(output/'latest.pt', map_location='cpu', weights_only=True) if (output/'latest.pt').exists() else None
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    from terrain_inference import prepare_models
    from types import SimpleNamespace
    teacher = EDMUnet2D.from_pretrained(str(source_path)).to('cuda', torch.bfloat16).eval().requires_grad_(False)
    prepare_models(SimpleNamespace(coarse_model=teacher, base_model=teacher, decoder_model=teacher))
    train_files, val_files = sorted(root.glob('train-*.npz')), sorted(root.glob('val-*.npz'))
    if set(stats['scenes']) != {path.name for path in train_files} or not val_files:
        raise ValueError('Scene inventory changed after the global variance audit.')
    control = check_teacher(teacher, train_files[0])
    atomic_json(output/'teacher-control.json', control)
    if not control['passed']:
        raise RuntimeError('Original base replay failed its measured precision floor.')
    decoder = OriginalDecoder('cuda')
    weight = linear_weight_window(64, 'cuda', torch.float32)
    variance = torch.tensor(stats['variance'], device='cuda')
    model = NativeBase(source['config']).cuda().eval()
    model.load_state_dict(saved['model'] if saved else source['ema'])
    ema = NativeBase(source['config']).cuda().eval().requires_grad_(False)
    ema.load_state_dict(saved['ema'] if saved else model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=1e-4, fused=True)
    step = saved['step'] if saved else 0
    best = saved['best_score'] if saved else float('inf')
    if step > args.steps:
        raise ValueError('Cannot resume to an earlier step.')
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    del source, saved
    started, last_save = time.monotonic(), time.monotonic()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)

    def checkpoint(name):
        atomic_write(output/name, lambda handle: torch.save(dict(config=model.config,
            model=model.state_dict(), ema=ema.state_dict(), optimizer=optimizer.state_dict(),
            step=step, best_score=best, contract=contract, teacher_revision=REVISION,
            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), handle))

    atomic_json(output/'run.json', dict(contract=contract, target_steps=args.steps, start_step=step,
        train_scenes=len(train_files), validation_scenes=len(val_files), accepted=False))
    if not (output/'baseline.json').exists():
        details = validate(ema, val_files, weight, decoder, variance)
        best = details['selection_score']
        atomic_json(output/'baseline.json', details)
        checkpoint('initial.pt')
        checkpoint('best.pt')
        append_json(output/'validation.jsonl', dict(step=0, **details))
        print(json.dumps(dict(step=0, median_height_mae_m=details['median_height_mae_m'], selection_score=best)), flush=True)
    baseline = json.loads((output/'baseline.json').read_text())
    guarded = False
    try:
        while step < args.steps and not STOP and not guarded:
            generator = np.random.default_rng(np.random.SeedSequence([args.seed, step]))
            index = int(generator.integers(len(train_files)))
            region = DecodedRegion(train_files[index], 'cuda')
            if generator.random() < args.fresh_noise_fraction:
                region.refresh_noise(teacher, weight, int(generator.integers(1_000_000, 2**31-1)))
            optimizer.zero_grad(set_to_none=True)
            result = decoded_region_forward(model, region, weight, checkpointed=True)
            loss, terms = decoded_loss(result, region, decoder, teacher, variance)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite physical loss.')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            with torch.no_grad():
                decay = min(.999, (1+step)/(10+step))
                for average, parameter in zip(ema.parameters(), model.parameters()):
                    average.lerp_(parameter, 1-decay)
            if step % 25 == 0 or step == 1:
                row = dict(step=step, loss=float(loss), terms={key: float(value) for key, value in terms.items()},
                    grad_norm=float(norm), elapsed_seconds=time.monotonic()-started,
                    peak_bytes=torch.cuda.max_memory_allocated(), source=train_files[index].name,
                    core=region.core, noise_seed=getattr(region, 'noise_seed', region.metadata['seed']))
                append_json(output/'train.jsonl', row)
                print(json.dumps(row), flush=True)
            del result, loss, terms, region
            if step % args.eval_every == 0 or step == args.steps:
                details = validate(ema, val_files, weight, decoder, variance)
                append_json(output/'validation.jsonl', dict(step=step, **details))
                guarded = (step >= 500 and details['feature_relative_mse'] >
                           1.1*baseline['feature_relative_mse'])
                if not guarded and details['selection_score'] < best:
                    best = details['selection_score']
                    checkpoint('best.pt')
                checkpoint(f'step-{step:06d}.pt')
                print(json.dumps(dict(step=step, median_height_mae_m=details['median_height_mae_m'],
                    feature_relative_mse=details['feature_relative_mse'], quality_guard=guarded)), flush=True)
            if time.monotonic()-last_save >= 120:
                checkpoint('latest.pt')
                last_save = time.monotonic()
                atomic_json(output/'status.json', dict(status='running', step=step, target_steps=args.steps, best_score=best))
        checkpoint('latest.pt')
        atomic_json(output/'status.json', dict(status='quality-guard' if guarded else 'interrupted' if STOP else 'complete',
            step=step, best_score=best, elapsed_seconds=time.monotonic()-started, accepted=False))
    except Exception as exc:
        checkpoint('latest.pt')
        atomic_json(output/'status.json', dict(status='failed', step=step, error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()

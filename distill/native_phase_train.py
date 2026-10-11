"""Fit phase-specific latent detail objectives without decoded training losses.

Captured teacher inputs are intentional: phase swapping found independent
first-phase feature attenuation and second-phase H4 error on coarse4. The
original, coarse4 and coarse2 targets each use their own original base teacher.
The frozen decoder participates only in independent validation and guards.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch

from distill.common import DATA, REPO, REVISION, atomic_json, atomic_write, append_json, external_path
from distill.native_amplitude import morphology_degraded
from distill.native_base import NativeBase, teacher_source
from distill.native_decoded import DecodedRegion, OriginalDecoder
from distill.native_feature_train import balanced_windows
from distill.native_mixed_bank import KINDS, VARIANTS
from distill.native_multilod_eval import validation_records
from distill.native_multilod_train import digest, validate
from distill.native_phase_loss import WEIGHTS, phase_loss

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def choose_scene(records, generator):
    variant = VARIANTS[int(generator.integers(len(VARIANTS)))]
    kind = KINDS[int(generator.integers(len(KINDS)))] if generator.random() < .25 else None
    pool = [r for r in records if r['coarse_variant'] == variant and r['case_kind'] == kind]
    profiles = sorted({r['metadata']['profile'] for r in pool})
    if not profiles:
        raise ValueError('Complete coarse/case/profile training strata required.')
    profile = profiles[int(generator.integers(len(profiles)))]
    pool = [r for r in pool if r['metadata']['profile'] == profile]
    return pool[int(generator.integers(len(pool)))]


@torch.no_grad()
def operator_control(model, records, root):
    """Protect untargeted output channels on fixed independent window inputs."""
    rows = []
    for index, record in enumerate(records):
        batch = balanced_windows(DecodedRegion(root/record['file'], 'cuda'), np.random.default_rng(17001+index))
        prediction = model(batch['inputs'], batch['labels'], [batch['condition']])
        residual = prediction-batch['target']
        rows.append(dict(coarse_variant=record['coarse_variant'], file=record['file'],
            first_height=float(residual[:2, 4:].square().mean()),
            second_features=float(residual[2:, :4].square().mean())))
    return {variant: {key: float(np.mean([r[key] for r in rows if r['coarse_variant'] == variant]))
                      for key in ('first_height', 'second_features')} for variant in VARIANTS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DATA/'ckpt/native160x3-operator-pilot/step15000.pt')
    parser.add_argument('--dataset', type=Path, default=DATA/'crops/native-mixed-scenes-v2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=5000)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-root', type=Path)
    args = parser.parse_args()
    if min(args.steps, args.eval_every) <= 0 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('Positive finite learning rate and step counts required.')
    root, output, source = (external_path(p) for p in (args.dataset, args.output, args.source))
    stats_path = DATA/'eval/native-decoded-probe/channel-statistics.json'
    manifest = json.loads((root/'manifest.json').read_text())
    stats = json.loads(stats_path.read_text())
    records = validation_records(manifest, stats)
    train = [r for r in manifest['scenes'] if r['metadata']['split'] == 'train']
    if json.loads((root/'status.json').read_text())['status'] != 'complete' or any(not (root/r['file']).exists() for r in train+records):
        raise ValueError('Complete independent mixed teacher bank required.')
    if digest(teacher_source()/'diffusion_pytorch_model.safetensors') != manifest['teacher_weights_sha256']:
        raise ValueError('Teacher and captured bank differ.')
    original = torch.load(source, map_location='cpu', weights_only=True)
    if (original.get('teacher_revision') or original.get('dataset_manifest', {}).get('revision')) != REVISION:
        raise ValueError('Source checkpoint teacher revision differs.')
    config = original['config']
    producers = ('native_phase_train.py', 'native_phase_loss.py', 'native_feature_train.py',
        'native_base.py', 'native_multilod_train.py', 'native_multilod.py', 'native_multilod_eval.py',
        'native_amplitude.py', 'native_decoded.py', 'native_fusion.py', 'terrain_shape.py')
    contract = dict(config=config, source_sha256=digest(source), source_step=original['step'],
        teacher_revision=REVISION, teacher_weights_sha256=manifest['teacher_weights_sha256'],
        dataset_manifest_sha256=digest(root/'manifest.json'), statistics_sha256=digest(stats_path),
        producer_sha256={f'distill/{p}': digest(REPO/'distill'/p) for p in producers},
        objective='phase-specific direct teacher x0 + position-sensitive band MSE + symmetric log-energy; velocity anchor',
        weights=WEIGHTS, lr=args.lr, seed=args.seed, batch=4, phases_per_batch=[2, 2],
        rare_fraction=.25, sampler='uniform coarse/case/profile strata', fresh_noise_fraction=0.,
        preserve_valleys=True, decoded_training_loss=False, x0_loss_dtype='fp32',
        eval_every=args.eval_every, clip_norm=1., selection='fixed-baseline guarded worst-group valley IoU')
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text()) != contract:
        raise ValueError('Phase trial contract changed; use a separate run.')
    if not args.smoke:
        if args.smoke_root is None:
            raise ValueError('A successful three-branch GPU smoke is required.')
        proof = json.loads((args.smoke_root.expanduser().resolve()/'gpu-smoke.json').read_text())
        if proof['status'] != 'complete' or proof['contract'] != contract or {r['coarse_variant'] for r in proof['rows']} != set(VARIANTS):
            raise ValueError('Smoke must match the entire training contract and all coarse branches.')
        if shutil.disk_usage(output.parent).free < 12*1024**3:
            raise ValueError('Insufficient space for preserved checkpoints and atomic save.')
    atomic_json(output/'contract.json', contract)
    torch.set_num_threads(8); torch.manual_seed(args.seed); torch.backends.cudnn.benchmark = True
    model = NativeBase(config).cuda().eval()
    saved = torch.load(output/'latest.pt', map_location='cpu', weights_only=True) if (output/'latest.pt').exists() else None
    if saved and saved['contract'] != contract:
        raise ValueError('Saved training contract differs.')
    model.load_state_dict(saved['model'] if saved else original['ema'])
    ema = NativeBase(config).cuda().eval().requires_grad_(False)
    ema.load_state_dict(saved['ema'] if saved else model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=1e-4, fused=True)
    step, best = (saved['step'], saved['best_score']) if saved else (0, float('inf'))
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    del original, saved
    variance = torch.tensor(stats['variance'], device='cuda')

    def training_step(record, generator):
        batch = balanced_windows(DecodedRegion(root/record['file'], 'cuda'), generator)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch['inputs'], batch['labels'], [batch['condition']])
        loss, terms = phase_loss(prediction, batch['inputs'], batch['labels'], batch['target'], variance)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite phase-specific objective.')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()):
            raise RuntimeError('No nonzero student gradients.')
        optimizer.step()
        return dict(loss=float(loss.detach()), grad_norm=float(norm),
            terms={k: float(v.detach()) for k, v in terms.items()}, file=record['file'],
            coarse_variant=record['coarse_variant'], case_kind=record['case_kind'],
            peak_bytes=torch.cuda.max_memory_allocated(), labels=batch['labels'].cpu().tolist())

    if args.smoke:
        rows = [training_step(next(r for r in train if r['coarse_variant'] == variant),
                              np.random.default_rng(args.seed)) for variant in VARIANTS]
        atomic_json(output/'gpu-smoke.json', dict(status='complete', contract=contract, rows=rows,
            decoder_in_training=False, gpu=torch.cuda.get_device_name(), accepted=False))
        return

    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    decoder = OriginalDecoder('cuda'); weight = linear_weight_window(64, 'cuda', torch.float32)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    started = time.monotonic()

    def checkpoint(name):
        atomic_write(output/name, lambda f: torch.save(dict(config=config, model=model.state_dict(),
            ema=ema.state_dict(), optimizer=optimizer.state_dict(), step=step, best_score=best, contract=contract,
            teacher_revision=REVISION, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), f))

    def alias(from_name, name):
        temp = output/f'.{name}.link'
        temp.unlink(missing_ok=True); os.link(output/from_name, temp); os.replace(temp, output/name)

    def publish(name):
        subprocess_command = [sys.executable, '-m', 'distill.native_progress_images',
            '--checkpoints', str(output/name), '--label', 'phase']
        subprocess.run(subprocess_command, cwd=REPO, check=True)

    if not (output/'baseline.json').exists():
        baseline = validate(ema, records, root, weight, decoder, variance, True)
        baseline['operator_control'] = operator_control(ema, records, root)
        best = baseline['selection_score']; atomic_json(output/'baseline.json', baseline)
        checkpoint('initial.pt'); alias('initial.pt', 'best.pt')
        append_json(output/'validation.jsonl', dict(step=0, **baseline)); publish('initial.pt')
    baseline = json.loads((output/'baseline.json').read_text())
    atomic_json(output/'run.json', dict(contract=contract, target_steps=args.steps, start_step=step,
        train_scenes=len(train), validation_scenes=len(records), gpu=torch.cuda.get_device_name(), accepted=False))
    guarded = False
    try:
        while step < args.steps and not STOP and not guarded:
            generator = np.random.default_rng(np.random.SeedSequence([args.seed, step]))
            row = training_step(choose_scene(train, generator), generator)
            step += 1
            with torch.no_grad():
                # The source already has 15k steps: keep EMA smoothing at .999,
                # rather than resetting its effective history to a tiny pilot.
                for average, parameter in zip(ema.parameters(), model.parameters()):
                    average.lerp_(parameter, .001)
            if step == 1 or step % 25 == 0:
                row.update(step=step, elapsed_seconds=time.monotonic()-started)
                append_json(output/'train.jsonl', row)
                atomic_json(output/'status.json', dict(status='running', step=step, target_steps=args.steps))
                print(json.dumps(row), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                details = validate(ema, records, root, weight, decoder, variance, True)
                details['operator_control'] = operator_control(ema, records, root)
                guarded = morphology_degraded(details['groups'], baseline['groups']) or \
                    details['feature_relative_mse'] > 1.1*baseline['feature_relative_mse'] or any(
                    details['feature_by_coarse'][v] > 1.15*baseline['feature_by_coarse'][v] or
                    details['operator_control'][v]['second_features'] > 1.05*baseline['operator_control'][v]['second_features'] or
                    details['operator_control'][v]['first_height'] > 1.1*baseline['operator_control'][v]['first_height'] for v in VARIANTS)
                improved = not guarded and details['selection_score'] < best
                if improved:
                    best = details['selection_score']
                append_json(output/'validation.jsonl', dict(step=step, quality_guard=guarded, **details))
                name = f'step-{step:06d}.pt'; checkpoint(name); alias(name, 'latest.pt')
                if improved:
                    alias(name, 'best.pt')
                publish(name)
                # Preserve the starting point, first evaluated candidate and
                # current best; avoid retaining every multi-GB optimizer state.
                for old in output.glob('step-*.pt'):
                    if old.name not in ('step-000500.pt', name) and old.stat().st_ino != (output/'best.pt').stat().st_ino:
                        old.unlink()
        checkpoint('latest.pt')
        atomic_json(output/'status.json', dict(status='quality-guard' if guarded else 'interrupted' if STOP else 'complete',
            step=step, best_score=best, elapsed_seconds=time.monotonic()-started, accepted=False))
    except Exception as exc:
        checkpoint('latest.pt')
        atomic_json(output/'status.json', dict(status='failed', step=step, error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()

"""Fit native teacher operators and level representations on matched mixed worlds.

This controlled trial removes decoded physical losses that favored flattening.
Two windows per diffusion phase, matched original/coarse4/coarse2, and rare
sampling are kept. Actual two-LOD morphology is the validation/stop criterion.
No change to the runtime, decoder, geometry, inference steps or accepted stack.
"""
import argparse
import json
import os
import shutil
import signal
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from distill.common import DATA, REPO, REVISION, atomic_json, atomic_write, append_json, external_path
from distill.native_base import NativeBase, teacher_source
from distill.native_decoded import DecodedRegion, OriginalDecoder
from distill.native_feature_kd import feature_step, level_taps
from distill.native_fusion_train import check_teacher
from distill.native_multilod_eval import validation_records
from distill.native_multilod_train import choose_record, digest, validate
from distill.native_amplitude import morphology_degraded
from distill.native_mixed_bank import VARIANTS

STOP = False


def stop(signum, frame):
    global STOP
    STOP = True


def balanced_windows(region, generator):
    result = []
    for stage in region.stages:
        ids = torch.tensor(generator.choice(len(stage['inputs']), 2, replace=False),
                           device=stage['inputs'].device)
        result.append({key: value.index_select(0, ids) for key, value in stage.items()})
    return {key: torch.cat([stage[key] for stage in result]) for key in result[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DATA/'ckpt/native192x2-operator-long/step15000.pt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, default=DATA/'crops/native-mixed-scenes-v2')
    parser.add_argument('--steps', type=int, default=1500)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--feature-weight', type=float, default=.5)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-root', type=Path)
    args = parser.parse_args()
    if min(args.steps, args.eval_every) < 1 or not np.isfinite(args.lr) or args.lr <= 0 or \
            not np.isfinite(args.feature_weight) or args.feature_weight <= 0:
        parser.error('Positive finite steps, learning rate and representation weight required.')
    root, output, source = (external_path(path) for path in (args.dataset, args.output, args.source))
    statistics = DATA/'eval/native-decoded-probe/channel-statistics.json'
    manifest = json.loads((root/'manifest.json').read_text())
    stats = json.loads(statistics.read_text())
    records = validation_records(manifest, stats)
    train = [r for r in manifest['scenes'] if r['metadata']['split'] == 'train']
    if json.loads((root/'status.json').read_text())['status'] != 'complete' or \
            any(not (root/r['file']).exists() for r in train+records):
        raise ValueError('Complete independent matched capture required.')
    teacher_path = teacher_source()
    if digest(teacher_path/'diffusion_pytorch_model.safetensors') != manifest['teacher_weights_sha256']:
        raise ValueError('Teacher differs from the matched bank.')
    saved_source = torch.load(source, map_location='cpu', weights_only=True)
    teacher_config = json.loads((teacher_path/'config.json').read_text())
    config = saved_source['config']
    if config['model_channels'] != teacher_config['model_channels'] or config['layers_per_block'] != 2:
        raise ValueError('This controlled feature trial requires the equal-width, two-block source.')
    producers = ('native_feature_train.py', 'native_feature_kd.py', 'native_base.py',
                 'native_multilod_train.py', 'native_multilod_eval.py', 'native_multilod.py',
                 'native_amplitude.py', 'terrain_shape.py', 'native_decoded.py',
                 'native_fusion.py', 'native_fusion_train.py', 'decoded_loss.py')
    contract = dict(config=config, source_sha256=digest(source), source_step=saved_source['step'],
        dataset_manifest_sha256=digest(root/'manifest.json'), statistics_sha256=digest(statistics),
        teacher_weights_sha256=manifest['teacher_weights_sha256'], teacher_revision=REVISION,
        producer_sha256={f'distill/{name}': digest(REPO/'distill'/name) for name in producers},
        objective='operator MSE + .05 spatial-gradient MSE + teacher-energy-normalized level KD',
        feature_weight=args.feature_weight, lr=args.lr, seed=args.seed, batch=4,
        phases_per_batch=[2, 2], rare_fraction=.25, fresh_noise_fraction=0.,
        preserve_valleys=True, validation_only_preservation=True,
        eval_every=args.eval_every, clip_norm=1., selection='fixed-baseline morphology guarded worst-group valley IoU')
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text()) != contract:
        raise ValueError('Feature trial contract changed; use a separate run.')
    if not args.smoke:
        if args.smoke_root is None:
            raise ValueError('Successful feature GPU smoke is required before training.')
        proof = json.loads((args.smoke_root.expanduser().resolve()/'gpu-smoke.json').read_text())
        if proof['contract'] != contract or proof['status'] != 'complete' or \
                {r['coarse_variant'] for r in proof['rows']} != set(VARIANTS):
            raise ValueError('GPU smoke must use the exact feature trial contract and all coarse branches.')
        if shutil.disk_usage(output.parent).free < 14*1024**3:
            raise ValueError('Insufficient free space for preserved trial checkpoints and atomic save.')
    atomic_json(output/'contract.json', contract)
    torch.set_num_threads(8); torch.manual_seed(args.seed); torch.backends.cudnn.benchmark = True
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    from terrain_inference import prepare_models
    teacher = EDMUnet2D.from_pretrained(str(teacher_path)).cuda().bfloat16().eval().requires_grad_(False)
    prepare_models(SimpleNamespace(coarse_model=teacher, base_model=teacher, decoder_model=teacher))
    replay = {variant: check_teacher(teacher, root/next(r['file'] for r in train if r['coarse_variant'] == variant))
              for variant in VARIANTS}
    if any(not row['passed'] for row in replay.values()):
        raise RuntimeError('Matched teacher replay failed its precision floor.')
    atomic_json(output/'teacher-control.json', replay)
    saved = torch.load(output/'latest.pt', map_location='cpu', weights_only=True) if (output/'latest.pt').exists() else None
    model = NativeBase(config).cuda().eval()
    model.load_state_dict(saved['model'] if saved else saved_source['ema'])
    ema = NativeBase(config).cuda().eval().requires_grad_(False)
    ema.load_state_dict(saved['ema'] if saved else model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=1e-4, fused=True)
    step, best = (saved['step'], saved['best_score']) if saved else (0, float('inf'))
    if saved:
        if saved['contract'] != contract or step > args.steps:
            raise ValueError('Resume contract or target differs.')
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    del saved_source, saved

    def training_step(record, generator):
        region = DecodedRegion(root/record['file'], 'cuda')
        batch = balanced_windows(region, generator)
        optimizer.zero_grad(set_to_none=True)
        loss, terms = feature_step(model, teacher, batch['inputs'], batch['labels'],
                                   batch['condition'], args.feature_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite feature KD loss.')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if any(p.grad is not None for p in teacher.parameters()) or not any(
                p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()):
            raise RuntimeError('Teacher must stay frozen and student must receive gradients.')
        optimizer.step()
        return dict(loss=float(loss.detach()), terms={k: float(v.detach()) for k, v in terms.items()},
                    grad_norm=float(norm), coarse_variant=record['coarse_variant'], case_kind=record['case_kind'],
                    file=record['file'], labels=batch['labels'].float().cpu().tolist(),
                    peak_bytes=torch.cuda.max_memory_allocated())

    if args.smoke:
        rows = [training_step(next(r for r in train if r['coarse_variant'] == variant),
                              np.random.default_rng(args.seed)) for variant in VARIANTS]
        if len(level_taps(model.net)) != 8 or any(len(m._forward_hooks) for m in level_taps(model.net).values()):
            raise RuntimeError('Feature taps incomplete or leaked after smoke.')
        atomic_json(output/'gpu-smoke.json', dict(status='complete', contract=contract, rows=rows,
                    teacher_frozen=True, levels=8, accepted=False, gpu=torch.cuda.get_device_name()))
        return
    decoder = OriginalDecoder('cuda'); weight = linear_weight_window(64, 'cuda', torch.float32)
    variance = torch.tensor(stats['variance'], device='cuda')
    started = time.monotonic()
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)

    def checkpoint(name):
        atomic_write(output/name, lambda f: torch.save(dict(config=config, model=model.state_dict(),
            ema=ema.state_dict(), optimizer=optimizer.state_dict(), step=step, best_score=best, contract=contract,
            teacher_revision=REVISION, rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), f))

    def alias(source_name, name):
        temp = output/f'.{name}.link'
        temp.unlink(missing_ok=True)
        os.link(output/source_name, temp); os.replace(temp, output/name)

    if not (output/'baseline.json').exists():
        baseline = validate(ema, records, root, weight, decoder, variance, True)
        best = baseline['selection_score']; atomic_json(output/'baseline.json', baseline)
        checkpoint('initial.pt'); alias('initial.pt', 'best.pt')
        append_json(output/'validation.jsonl', dict(step=0, **baseline))
    baseline = json.loads((output/'baseline.json').read_text())
    atomic_json(output/'run.json', dict(contract=contract, target_steps=args.steps, start_step=step,
                train_scenes=len(train), validation_scenes=len(records), accepted=False))
    guarded = False
    try:
        while step < args.steps and not STOP and not guarded:
            generator = np.random.default_rng(np.random.SeedSequence([args.seed, step]))
            row = training_step(choose_record(train, generator), generator)
            step += 1
            with torch.no_grad():
                decay = min(.999, (1+step)/(10+step))
                for average, parameter in zip(ema.parameters(), model.parameters()):
                    average.lerp_(parameter, 1-decay)
            if step == 1 or step % 25 == 0:
                row.update(step=step, elapsed_seconds=time.monotonic()-started)
                append_json(output/'train.jsonl', row)
                atomic_json(output/'status.json', dict(status='running', step=step, target_steps=args.steps))
                print(json.dumps(row), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                details = validate(ema, records, root, weight, decoder, variance, True)
                guarded = step >= 500 and (morphology_degraded(details['groups'], baseline['groups']) or
                    details['feature_relative_mse'] > 1.1*baseline['feature_relative_mse'] or any(
                    details['feature_by_coarse'][v] > 1.15*baseline['feature_by_coarse'][v] for v in VARIANTS))
                improved = not guarded and details['selection_score'] < best
                if improved:
                    best = details['selection_score']
                append_json(output/'validation.jsonl', dict(step=step, quality_guard=guarded, **details))
                name = f'step-{step:06d}.pt'; checkpoint(name); alias(name, 'latest.pt')
                if improved:
                    alias(name, 'best.pt')
        checkpoint('latest.pt')
        atomic_json(output/'status.json', dict(status='quality-guard' if guarded else 'interrupted' if STOP else 'complete',
                    step=step, best_score=best, elapsed_seconds=time.monotonic()-started, accepted=False))
    except Exception as exc:
        checkpoint('latest.pt')
        atomic_json(output/'status.json', dict(status='failed', step=step, error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()

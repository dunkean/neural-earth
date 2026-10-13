"""Test stronger optimization with eight independent worlds per teacher batch.

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
from distill.native_optimization_audit import diagnostic_reasons, gradient_audit, morphology_guard, movement, validate_parent_contract
from distill.native_base import NativeBase, teacher_source
from distill.native_decoded import DecodedRegion, OriginalDecoder
from distill.native_feature_train import balanced_windows
from distill.native_diverse_windows import diverse_batch
from distill.native_mixed_bank import KINDS, VARIANTS
from distill.native_multilod_eval import validation_records
from distill.native_multilod_train import digest, validate
from distill.native_phase_loss import WEIGHTS, CHANNEL0_WEIGHTS, PHASE_LABELS, phase_loss

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
def operator_control(model, records, root, variance, calibration):
    """Protect untargeted output channels on fixed independent window inputs."""
    rows = []
    for index, record in enumerate(records):
        batch = balanced_windows(DecodedRegion(root/record['file'], 'cuda'), np.random.default_rng(17001+index))
        prediction = model(batch['inputs'], batch['labels'], [batch['condition']])
        residual = prediction-batch['target']
        _, terms = phase_loss(prediction, batch['inputs'], batch['labels'], batch['target'], variance, calibration)
        rows.append(dict(coarse_variant=record['coarse_variant'], file=record['file'],
            first_height=float(residual[:2, 4:].square().mean()),
            second_features=float(residual[2:, :4].square().mean()),
            first_band_position=float(terms['first_band_position']),
            channel_mse=torch.stack([residual[:2].square().mean(dim=(0, 2, 3)),
                                     residual[2:].square().mean(dim=(0, 2, 3))]).cpu().tolist()))
    return {variant: dict(first_height=float(np.mean([r['first_height'] for r in rows if r['coarse_variant'] == variant])),
                         second_features=float(np.mean([r['second_features'] for r in rows if r['coarse_variant'] == variant])),
                         first_band_position=float(np.mean([r['first_band_position'] for r in rows if r['coarse_variant'] == variant])),
                         channel_mse=np.mean([r['channel_mse'] for r in rows if r['coarse_variant'] == variant], axis=0).tolist())
            for variant in VARIANTS}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=DATA/'ckpt/native160x3-operator-pilot/step15000.pt')
    parser.add_argument('--dataset', type=Path, default=DATA/'crops/native-mixed-scenes-v2')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=3000)
    parser.add_argument('--eval-every', type=int, default=500)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=20261016)
    parser.add_argument('--warmup', type=int, default=200)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-root', type=Path)
    parser.add_argument('--resume-from', type=Path, help='Terminal parent run; preserve exact model, EMA, optimizer and RNG.')
    parser.add_argument('--diagnostic-complete', action='store_true', help='Log original guard failures; tolerate native-recall-only drops for a bounded pilot.')
    parser.add_argument('--frozen-statistics', type=Path, required=True, help='TRAIN-only band powers for channel0/H4 objective.')
    parser.add_argument('--label', default='phase', help='Image prefix for this distinct strategy.')
    args = parser.parse_args()
    if min(args.steps, args.eval_every) <= 0 or args.warmup < 0 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('Positive finite learning rate and step counts required.')
    root, output, source = (external_path(p) for p in (args.dataset, args.output, args.source))
    parent = external_path(args.resume_from) if args.resume_from else None
    if parent == output:
        raise ValueError('The parent run must remain untouched in a separate directory.')
    if args.diagnostic_complete and parent is None:
        raise ValueError('Diagnostic continuation requires an explicitly preserved terminal parent.')
    stats_path = DATA/'eval/native-decoded-probe/channel-statistics.json'
    manifest = json.loads((root/'manifest.json').read_text())
    stats = json.loads(stats_path.read_text())
    records = validation_records(manifest, stats)
    train = [r for r in manifest['scenes'] if r['metadata']['split'] == 'train']
    if json.loads((root/'status.json').read_text())['status'] != 'complete' or any(not (root/r['file']).exists() for r in train+records):
        raise ValueError('Complete independent mixed teacher bank required.')
    if digest(teacher_source()/'diffusion_pytorch_model.safetensors') != manifest['teacher_weights_sha256']:
        raise ValueError('Teacher and captured bank differ.')
    frozen = None
    if args.frozen_statistics:
        args.frozen_statistics = args.frozen_statistics.expanduser().resolve()
        frozen = json.loads(args.frozen_statistics.read_text())
        if frozen.get('status') != 'complete' or frozen.get('split') != 'train' or \
                frozen['dataset_manifest_sha256'] != digest(root/'manifest.json') or \
                frozen['statistics_sha256'] != digest(stats_path) or \
                frozen['teacher_weights_sha256'] != manifest['teacher_weights_sha256'] or \
                frozen['sigmas'] != [1., 2., 4.] or frozen['phase_labels'] != list(PHASE_LABELS) or \
                not set(frozen['scene_files']) <= {r['file'] for r in train}:
            raise ValueError('Band calibration must use this teacher and TRAIN-only scene inventory.')
    original = torch.load(source, map_location='cpu', weights_only=True)
    if (original.get('teacher_revision') or original.get('dataset_manifest', {}).get('revision')) != REVISION:
        raise ValueError('Source checkpoint teacher revision differs.')
    config = original['config']
    producers = ('native_optimization_train.py', 'native_diverse_windows.py', 'native_optimization_audit.py', 'native_phase_loss.py', 'native_feature_train.py',
        'native_base.py', 'native_multilod_train.py', 'native_multilod.py', 'native_multilod_eval.py',
        'native_amplitude.py', 'native_decoded.py', 'native_fusion.py', 'terrain_shape.py')
    contract = dict(config=config, source_sha256=digest(source), source_step=original['step'],
        teacher_revision=REVISION, teacher_weights_sha256=manifest['teacher_weights_sha256'],
        dataset_manifest_sha256=digest(root/'manifest.json'), statistics_sha256=digest(stats_path),
        producer_sha256={f'distill/{p}': digest(REPO/'distill'/p) for p in producers},
        objective='phase-specific direct teacher x0 + position-sensitive band MSE + symmetric log-energy; velocity anchor',
        weights=CHANNEL0_WEIGHTS if frozen else WEIGHTS, lr=args.lr, warmup=args.warmup, seed=args.seed, batch=8, phases_per_batch=[4, 4],
        band_calibration_sha256=digest(args.frozen_statistics) if frozen else None,
        band_denominator='fixed TRAIN' if frozen else 'local teacher', image_label=args.label,
        rare_fraction=.25, sampler='eight different TRAIN worlds, all three coarse branches per phase, 25 percent rare', fresh_noise_fraction=0.,
        preserve_valleys=True, decoded_training_loss=False, x0_loss_dtype='fp32',
        eval_every=args.eval_every, clip_norm=1., selection='raw guarded worst-group valley IoU',
        audit_every=250, audit_batches=16, raw_and_ema_validation=True)
    parent_contract, parent_status = None, None
    if parent is not None:
        parent_contract = json.loads((parent/'contract.json').read_text())
        parent_status = json.loads((parent/'status.json').read_text())
        if parent_status['status'] not in ('quality-guard', 'complete', 'interrupted'):
            raise ValueError('A terminal parent checkpoint is required.')
        validate_parent_contract(parent_contract, contract)
        contract.update(parent_run=str(parent), parent_contract_sha256=digest(parent/'contract.json'),
            parent_checkpoint_sha256=digest(parent/'latest.pt'), parent_step=parent_status['step'],
            diagnostic_policy='native recall warning; untargeted 1.2 warning/1.5 stop; original guard forbids selection'
                              if args.diagnostic_complete else 'original hard stops')
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text()) != contract:
        raise ValueError('Phase trial contract changed; use a separate run.')
    if not args.smoke:
        if args.smoke_root is None:
            raise ValueError('A successful three-branch GPU smoke is required.')
        proof = json.loads((args.smoke_root.expanduser().resolve()/'gpu-smoke.json').read_text())
        if proof['status'] != 'complete' or proof['contract'] != contract or any(len(set(r['worlds'])) != 8 or set(r['coarse_variants']) != set(VARIANTS) for r in proof['rows']):
            raise ValueError('Smoke must match the entire training contract and all coarse branches.')
        if shutil.disk_usage(output.parent).free < 10*1024**3:
            raise ValueError('Insufficient space for preserved checkpoints and atomic save.')
    atomic_json(output/'contract.json', contract)
    torch.set_num_threads(8); torch.manual_seed(args.seed); torch.backends.cudnn.benchmark = True
    model = NativeBase(config).cuda().eval()
    saved_path = output/'latest.pt' if (output/'latest.pt').exists() else parent/'latest.pt' if parent else None
    saved = torch.load(saved_path, map_location='cpu', weights_only=True) if saved_path else None
    if saved and saved['contract'] != (contract if saved_path.parent == output else parent_contract):
        raise ValueError('Saved training contract differs.')
    if saved and saved_path.parent != output and saved['step'] != parent_status['step']:
        raise ValueError('Parent terminal status and checkpoint steps differ.')
    model.load_state_dict(saved['model'] if saved else original['ema'])
    reference = {name: original['ema'][name].detach().cpu().clone() for name, _ in model.named_parameters()}
    ema = NativeBase(config).cuda().eval().requires_grad_(False)
    ema.load_state_dict(saved['ema'] if saved else model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=1e-4, fused=True)
    step, best = (saved['step'], saved['best_score']) if saved else (0, float('inf'))
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    del original, saved
    variance = torch.tensor(stats['variance'], device='cuda')
    calibration = ({key: torch.tensor(value, device='cuda') for key, value in dict(
        first_power=frozen['phases']['first']['band_power'],
        second_power=frozen['phases']['second']['band_power'],
        second_variance=frozen['phases']['second']['spatial_variance']).items()} if frozen else None)

    def training_step(generator, current_step):
        batch, selected = diverse_batch(train, root, generator, current_step)
        for group in optimizer.param_groups:
            group['lr'] = args.lr*min(1., (current_step+1)/max(1, args.warmup))
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch['inputs'], batch['labels'], [batch['condition']])
        loss, terms = phase_loss(prediction, batch['inputs'], batch['labels'], batch['target'], variance, calibration)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite phase-specific objective.')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()):
            raise RuntimeError('No nonzero student gradients.')
        optimizer.step()
        return dict(loss=float(loss.detach()), grad_norm=float(norm),
            terms={k: float(v.detach()) for k, v in terms.items()}, files=[r['file'] for _, r in selected], worlds=[r['metadata']['seed'] for _, r in selected],
            coarse_variants=[r['coarse_variant'] for _, r in selected], rare_count=sum(r['case_kind'] is not None for _, r in selected),
            effective_lr=optimizer.param_groups[0]['lr'], clip_fired=float(norm)>1.,
            peak_bytes=torch.cuda.max_memory_allocated(), labels=batch['labels'].cpu().tolist())

    if args.smoke:
        rows = [training_step(np.random.default_rng(np.random.SeedSequence([args.seed, step+index])), step+index)
                for index in range(3)]
        atomic_json(output/'gpu-smoke.json', dict(status='complete', contract=contract, rows=rows,
            gradient_control=gradient_audit(model, train, root, variance, calibration, batches=1),
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

    def snapshot(name, network):
        atomic_write(output/name, lambda f: torch.save(dict(config=config, ema=network.state_dict(),
            step=step, contract=contract, teacher_revision=REVISION, inference_only=True), f))

    def publish(name, label):
        subprocess_command = [sys.executable, '-m', 'distill.native_progress_images',
            '--checkpoints', str(output/name), '--label', label]
        subprocess.run(subprocess_command, cwd=REPO, check=True)

    if parent is not None and not (output/'baseline.json').exists():
        baseline = json.loads((parent/'baseline.json').read_text())
        atomic_json(output/'baseline.json', baseline)
        os.link(parent/'initial.pt', output/'initial.pt')
        alias('initial.pt', 'best-ema.pt'); alias('initial.pt', 'best-raw.pt')
        atomic_json(output/'resume-lineage.json', dict(parent_contract=parent_contract,
            parent_status=parent_status, preserved_source_baseline=True,
            checkpoint_sha256=contract['parent_checkpoint_sha256'], exact_optimizer_and_rng=True,
            original_guard_forbids_selection=True, accepted=False))
        for name in ('validation.jsonl', 'train.jsonl'):
            atomic_write(output/name, lambda f, name=name: f.write((parent/name).read_bytes()))
    if not (output/'baseline.json').exists():
        baseline = validate(ema, records, root, weight, decoder, variance, True)
        baseline['operator_control'] = operator_control(ema, records, root, variance, calibration)
        best = baseline['selection_score']; atomic_json(output/'baseline.json', baseline)
        snapshot('initial.pt', ema); alias('initial.pt', 'best-ema.pt'); alias('initial.pt', 'best-raw.pt')
        append_json(output/'validation.jsonl', dict(step=0, weights='source', **baseline))
        publish('initial.pt', args.label)
        append_json(output/'gradient-audit.jsonl', dict(step=0,
            rows=gradient_audit(model, train, root, variance, calibration), movement=movement(model, reference)))
    baseline = json.loads((output/'baseline.json').read_text())
    atomic_json(output/'run.json', dict(contract=contract, target_steps=args.steps, start_step=step,
        train_scenes=len(train), validation_scenes=len(records), gpu=torch.cuda.get_device_name(), accepted=False))
    guarded = False
    clipped, observed = 0, 0
    try:
        while step < args.steps and not STOP and not guarded:
            generator = np.random.default_rng(np.random.SeedSequence([args.seed, step]))
            row = training_step(generator, step)
            step += 1
            clipped += int(row['clip_fired']); observed += 1
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
            if step % 250 == 0:
                audit = gradient_audit(model, train, root, variance, calibration)
                append_json(output/'gradient-audit.jsonl', dict(step=step, rows=audit,
                    movement=movement(model, reference), clip_fraction=clipped/observed))
            if step % args.eval_every == 0 or step == args.steps:
                details = validate(model, records, root, weight, decoder, variance, True)
                details['operator_control'] = operator_control(model, records, root, variance, calibration)
                ema_details = validate(ema, records, root, weight, decoder, variance, True)
                ema_details['operator_control'] = operator_control(ema, records, root, variance, calibration)
                reasons = morphology_guard(details, baseline)
                if details['feature_relative_mse'] > 1.3*baseline['feature_relative_mse']:
                    reasons.append(dict(reason='large feature regression'))
                for variant in VARIANTS:
                    for phase in (0, 1):
                        for channel in range(5):
                            if (phase, channel) in ((0, 0), (1, 4)):
                                continue
                            before = baseline['operator_control'][variant]['channel_mse'][phase][channel]
                            after = details['operator_control'][variant]['channel_mse'][phase][channel]
                            if after > 1.2*max(before, 1e-8):
                                reasons.append(dict(reason='large untargeted channel regression',
                                                    variant=variant, phase=phase, channel=channel))
                hard_reasons = diagnostic_reasons(reasons, details, baseline) if args.diagnostic_complete else reasons
                guarded = bool(hard_reasons)
                improved = not reasons and details['selection_score'] < best
                if improved:
                    best = details['selection_score']
                append_json(output/'validation.jsonl', dict(step=step, weights='raw', quality_guard=guarded,
                    guard_reasons=hard_reasons, original_guard_reasons=reasons, original_quality_guard=bool(reasons),
                    selectable=not reasons, movement=movement(model, reference), clip_fraction=clipped/observed, **details))
                append_json(output/'validation.jsonl', dict(step=step, weights='ema', **ema_details))
                raw_name, ema_name = f'step-{step:06d}-raw.pt', f'step-{step:06d}-ema.pt'
                snapshot(raw_name, model); snapshot(ema_name, ema); checkpoint('latest.pt')
                if improved:
                    alias(raw_name, 'best-raw.pt'); alias(ema_name, 'best-ema.pt')
                publish(raw_name, args.label+'-raw'); publish(ema_name, args.label+'-ema')
                # Keep the first comparison, best pair and latest pair. Only
                # latest.pt carries the optimizer; inference snapshots are small.
                protected = {(output/name).stat().st_ino for name in ('best-raw.pt', 'best-ema.pt')}
                first_step = ((contract.get('parent_step', 0)//args.eval_every)+1)*args.eval_every
                for old in output.glob('step-*.pt'):
                    if old.name not in (f'step-{first_step:06d}-raw.pt', f'step-{first_step:06d}-ema.pt', raw_name, ema_name) and old.stat().st_ino not in protected:
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

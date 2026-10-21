"""Recover compact160 FP32 masters with TRAIN-calibrated physical losses.

Smoke calibrates gradients and proves real updates before a durable run starts.
Inference snapshots retain the existing compact160 loader interface.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import signal
import statistics
import time

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from distill.common import DATA, REPO, atomic_json, atomic_write, append_json, external_path
from distill.native_auxiliary_training import BoundedQATBank, preconditioners, gradient_norm, combine_gradients, root_coast
from distill.native_base import teacher_source
from distill.native_basis_training import inventory, capture_blocks
from distill.native_decoded import DecodedRegion, OriginalDecoder, decoded_region_forward, physical_terms
from distill.native_fusion import forward_batches, blend, window_outputs, refinement_inputs
from distill.native_level_objective import level_terms, protect_scalar_gradients
from distill.native_progressive_features import compact_feature_loss
from distill.native_progressive_prepare import configure
from distill.native_progressive_recovery_loss import multiscale_x0_terms, multiscale_height_terms
from distill.native_progressive_width import ProgressiveWidthBase
from distill.native_weight_quant import file_sha
from distill.native_weight_quant_train import loss_terms
from terrain_diffusion.models.edm_unet import EDMUnet2D

CODE = ('native_progressive_recovery_train.py', 'native_progressive_recovery_loss.py',
        'native_progressive_features.py', 'native_progressive_width.py', 'native_auxiliary_training.py',
        'native_fp8_training.py', 'native_basis_training.py', 'native_decoded.py',
        'native_fusion.py', 'native_level_objective.py', 'native_weight_quant_train.py')


class RecoveryDecoder(OriginalDecoder):
    def forward(self, latents, gaussian):
        if torch.is_grad_enabled() and latents.requires_grad:
            return checkpoint(super().forward, latents, gaussian, use_reentrant=False)
        return super().forward(latents, gaussian)


def paired_forward(model, region, weight):
    # Reentrant recomputation releases per-window functional weight casts.
    # A differentiable input is required even for the initialization phase.
    def windows(inputs, labels, condition):
        values = []
        def forward(x, t, c):
            return model(x, t, [c])
        for start in range(0, len(inputs), 4):
            x = inputs[start:start+4]
            if not x.requires_grad:
                x = x.detach().requires_grad_(True)
            values.append(checkpoint(forward, x, labels[start:start+4], condition[start:start+4], use_reentrant=True))
        return torch.cat(values)
    first, second = region.stages
    velocity0 = windows(first['inputs'], first['labels'], first['condition'])
    initial = blend(window_outputs(first['inputs'], velocity0, first['labels'], region.dtype),
                    region.init_origins, region.init_origin, 192, weight)
    inputs1 = refinement_inputs(region.ref_crops(initial), region.noise, second['labels'], region.dtype)
    velocity1 = windows(inputs1, second['labels'], second['condition'])
    core = region.core_crop(window_outputs(inputs1, velocity1, second['labels'], region.dtype), weight)
    return dict(core=core)


def coefficient(loss, reference, probes, fraction):
    gradients = torch.autograd.grad(loss, probes, retain_graph=True, allow_unused=True)
    norm = gradient_norm(gradients)
    return fraction*reference/max(norm, 1e-12), norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=25000)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-root', type=Path)
    args = parser.parse_args()
    output = external_path(args.output)
    configure()
    source = torch.load(args.source, map_location='cpu', weights_only=True, mmap=True)
    if source.get('phase') != 'compact160' or 'recovery_master' not in source:
        raise ValueError('A completed compact160 with true FP32 masters is required.')
    contract = dict(source=str(args.source.resolve()), source_sha256=file_sha(args.source),
        steps=args.steps, seed=2026101447, width=160, blocks=3, passes=2,
        lr='warmup500 to2e-5, cosine to2e-6', ema_half_lives=[500, 2000],
        auxiliary_every=2, decoded_every=4, auxiliary_cap=.5, feature_decay_steps=5000,
        sampler='existing independent TRAIN bank:25%mountains,15%critical,60%general;50%on-policy',
        loss='operator+x0 height LF/HF; TT level/full/HF/coast; decoded slope/TPI',
        calibration='TRAIN kernel gradients for x0; native core gradients for TT/decoder terms; median coefficients; full parameter raw/Adam cap',
        code_sha256={name:file_sha(REPO/'distill'/name) for name in CODE}, accepted=False,
        target_bytes=100000000, target_speedup=3., inference_dtype='BF16; quantization follows speed distillation')
    if args.steps < 1000:
        raise ValueError('Use --smoke for a short verification.')
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text()) != contract:
        raise ValueError('Changed recipe; preserve the previous run.')
    proof = None
    if not args.smoke:
        if args.smoke_root is None:
            raise ValueError('A successful GPU smoke is required.')
        proof = json.loads((args.smoke_root/'gpu-smoke.json').read_text())
        if proof['status'] != 'complete' or proof['contract'] != contract:
            raise ValueError('Smoke contract does not match.')
        if shutil.disk_usage(output.parent).free < 20*2**30:
            raise ValueError('Reserve20GiB for checkpoints and atomic saves.')
    atomic_json(output/'contract.json', contract)
    torch.manual_seed(contract['seed']); torch.cuda.manual_seed_all(contract['seed'])
    model = ProgressiveWidthBase(source['config'], source['target_config'], source['mapping'], source['gain_anchor']).cuda()
    model.load_state_dict(source['recovery_master'])
    averages = [copy.deepcopy(model).eval().requires_grad_(False) for _ in (500, 2000)]
    params = [p for p in model.parameters() if p.requires_grad]
    probes = [model.net.out_conv.weight, next(iter(model.net.enc.values())).weight]
    optimizer = torch.optim.Adam(params, lr=0., betas=(.9, .99), fused=True)
    coefficients = proof['coefficients'] if proof else {}
    step = 0
    latest = output/'latest.pt'
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=True, mmap=True)
        if saved['contract'] != contract:
            raise ValueError('Resume contract differs.')
        model.load_state_dict(saved['model'])
        for average, state in zip(averages, (saved['ema'], saved['ema2000'])):
            average.load_state_dict(state)
        optimizer.load_state_dict(saved['optimizer']); step = saved['step']
        coefficients = saved['coefficients']
        torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    teacher = EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    decoder = RecoveryDecoder('cuda')
    records, _, _ = inventory()
    bank = BoundedQATBank(limit=96)
    variance = torch.tensor(json.loads((DATA/'eval/native-decoded-probe/channel-statistics.json').read_text())['variance'], device='cuda')
    floors = json.loads((DATA/'eval/native-nf4-level-train-floor-4090.json').read_text())['floors']
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64, 'cuda', torch.float32)
    calibration, logs = {}, []
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    started = time.monotonic()

    def calibrate(name, loss, reference, fraction, selected_probes=None):
        if args.smoke:
            value, norm = coefficient(loss, reference, probes if selected_probes is None else selected_probes, fraction)
            if not math.isfinite(value) or not math.isfinite(norm):
                raise FloatingPointError('Nonfinite calibration: '+name)
            if norm > 1e-10:
                calibration.setdefault(name, []).append(dict(coefficient=value, gradient_norm=norm,
                                                            reference_norm=reference))
                coefficients[name] = statistics.median(row['coefficient'] for row in calibration[name])
        if name not in coefficients:
            # Coast losses may be zero on scenes with no shoreline.
            if name == 'coast' and float(loss.detach()) == 0:
                return 0.
            raise ValueError('Missing nonzero TRAIN calibration: '+name)
        return coefficients[name]

    def snapshot(path, full=False, selection='ema'):
        payload = {key:value for key, value in source.items() if key not in ('ema', 'recovery_master', 'contract')}
        payload.update(step=step, parent_step=20000, contract=contract, coefficients=coefficients,
                       selected_weights=selection, accepted=False)
        net = model if selection == 'raw' else averages[1 if selection == 'ema2000' else 0]
        payload['ema'] = {name:value.detach().cpu().to(torch.float32 if name == 'net.out_gain' else torch.bfloat16)
                          for name, value in net.state_dict().items()}
        if full:
            payload.update(model=model.state_dict(), ema=averages[0].state_dict(),
                           ema2000=averages[1].state_dict(), optimizer=optimizer.state_dict(),
                           rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
        atomic_write(path, lambda handle:torch.save(payload, handle))

    try:
        while step < args.steps and not stopped:
            generator = np.random.default_rng(np.random.SeedSequence([contract['seed'], step]))
            lr = min(1., (step+1)/500)*(2e-6+18e-6*.5*(1+math.cos(math.pi*max(0., (step-500)/(args.steps-500)))))
            for group in optimizer.param_groups:
                group['lr'] = lr
            with torch.no_grad(), model.cached_weights():
                batch, selected = bank.batch(records, generator, step, model, on_policy=step%2 == 0)
            with torch.no_grad(), capture_blocks(teacher) as reference:
                target = teacher(batch['inputs'].bfloat16(), noise_labels=batch['labels'].bfloat16(),
                                 conditional_inputs=[batch['condition'].bfloat16()]).float()
            with capture_blocks(model.net) as actual:
                prediction = model(batch['inputs'], batch['labels'], [batch['condition']])
            regular, terms = loss_terms(prediction, target, batch, variance)
            base_norm = gradient_norm(torch.autograd.grad(regular, probes, retain_graph=True, allow_unused=True)) if args.smoke else 0.
            physical = multiscale_x0_terms(prediction, target, batch)
            extra = physical['low']+physical['high']
            c = calibrate('x0_height', extra, base_norm, .5)
            feature = compact_feature_loss(actual, reference, source['original_teacher_channel_mapping'])
            feature_weight = .045*max(0., 1-step/5000)
            total = regular+c*extra+feature_weight*feature
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            row = dict(step=step+1, loss=float(total.detach()), operator=float(terms['operator'].detach()),
                height_low=float(physical['low'].detach()), height_high=float(physical['high'].detach()),
                feature_weight=feature_weight, feature=float(feature.detach()), on_policy=batch['on_policy'], lr=lr,
                kinds=[record['case_kind'] for _, record in selected], auxiliary=False)
            del actual, reference, prediction, target, regular, terms, physical, extra, feature, total
            if step%2 == 1:
                regular_grads = [p.grad.detach().clone() if p.grad is not None else None for p in params]
                pre = preconditioners(optimizer, params)
                regular_norm = gradient_norm(regular_grads)
                scaled_norm = gradient_norm(regular_grads, pre)
                probe_norm = gradient_norm([p.grad for p in probes])
                index = step//2
                kinds = ('wet-complex-coast', 'low-plain-sea', 'desert-plain', 'temperate-plain', 'wet-high-mountain')
                kind = kinds[index%5]; coarse = ('coarse4', 'coarse2', 'original')[(index//5)%3]
                pool = [r for r in records if r['case_kind'] == kind and r['coarse_variant'] == coarse]
                record = pool[int(generator.integers(len(pool)))]
                region = DecodedRegion(Path(record['root'])/record['file'], 'cuda')
                region.refresh_noise(teacher, weight, int(generator.integers(1, 2**31-5821)))
                result = paired_forward(model, region, weight)
                fields = result['core'][None]; desired = region.target.detach()[None]
                core_probes = [result['core']]
                native_loss = ((result['core']-region.target.detach()).square()/variance[:, None, None]).mean()
                core_norm = gradient_norm(torch.autograd.grad(native_loss, core_probes, retain_graph=True)) if args.smoke else 0.
                height = multiscale_height_terms(fields, desired)
                level = level_terms(fields, desired, variance, floors[kind])['level']
                coast, disagreement, _ = root_coast(fields, desired)
                auxiliary_losses = dict(tt_height=height['low']+height['high']+height['full'], level=level, coast=coast)
                if step%4 == 3:
                    decoded = physical_terms(result, region, decoder, variance)
                    auxiliary_losses['decoded_shape'] = decoded['height_gradient']+decoded['height_tpi']
                combined = sum(calibrate(name, value, core_norm, .25, core_probes)*value for name, value in auxiliary_losses.items())
                optimizer.zero_grad(set_to_none=True)
                combined.backward()
                protect_scalar_gradients(model)
                row.update(combine_gradients(params, regular_grads, pre, regular_norm, scaled_norm, cap=.5),
                           auxiliary=True, aux_kind=kind, aux_coarse=coarse,
                           coast_disagreement=float(disagreement.detach()),
                           aux_losses={name:float(value.detach()) for name, value in auxiliary_losses.items()})
                del regular_grads, pre, region, result, fields, desired, height, level, coast, combined, auxiliary_losses, core_probes, native_loss
                if step%4 == 3:
                    del decoded
            norm = torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
            if any(p.grad is not None for p in teacher.parameters()) or any(p.grad is not None for p in decoder.parameters()):
                raise ValueError('Frozen reference received gradients.')
            optimizer.step()
            with torch.no_grad():
                for average, half_life in zip(averages, (500, 2000)):
                    for a, p in zip(average.parameters(), model.parameters()):
                        a.lerp_(p, 1-math.exp(math.log(.5)/half_life))
            step += 1
            row.update(grad_norm=float(norm), elapsed_seconds=time.monotonic()-started,
                       peak_bytes=torch.cuda.max_memory_allocated())
            logs.append(row)
            if step%100 == 0 or args.smoke:
                # All states/categories in the interval are retained, avoiding cadence aliasing.
                append_json(output/'train.jsonl', dict(step=step, batches=logs, coefficients=coefficients))
                print(json.dumps(row), flush=True); logs = []
                atomic_json(output/'status.json', dict(status='running', step=step, target=args.steps))
            if not args.smoke and (step%1000 == 0 or step in (3000, 6000)):
                snapshot(latest, full=True)
                for selection in ('ema', 'ema2000', 'raw'):
                    snapshot(output/selection/f'step{step:06d}.pt', selection=selection)
            if args.smoke and step >= 8:
                break
        if logs:
            append_json(output/'train.jsonl', dict(step=step, batches=logs, coefficients=coefficients))
        snapshot(latest, full=True)
        if args.smoke:
            if stopped or step != 8 or set(coefficients) != {'x0_height', 'tt_height', 'level', 'coast', 'decoded_shape'}:
                raise ValueError('Incomplete smoke/calibration.')
            atomic_json(output/'gpu-smoke.json', dict(status='complete', contract=contract, coefficients=coefficients,
                        calibration=calibration, steps=step, peak_bytes=torch.cuda.max_memory_allocated(),
                        gpu=torch.cuda.get_device_name()))
        else:
            for selection in ('ema', 'ema2000', 'raw'):
                snapshot(output/selection/f'step{step:06d}.pt', selection=selection)
        atomic_json(output/'status.json', dict(status='smoke_complete' if args.smoke else 'interrupted' if stopped else 'complete',
                    step=step, target=args.steps, elapsed_seconds=time.monotonic()-started))
    except Exception as exc:
        atomic_json(output/'status.json', dict(status='failed', step=step, error=str(exc)))
        # Preserve last proven checkpoint rather than overwriting it with suspect weights.
        raise


if __name__ == '__main__':
    main()

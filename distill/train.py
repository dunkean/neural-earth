"""BF16 regression with spectral/gradient losses, atomic checkpoints and resume.

  CUDA_VISIBLE_DEVICES=1 python -m distill.train --stage base --dataset ~/data/distill/crops/main
  ... --resume ~/data/distill/ckpt/base/latest.pt

Sample addresses depend on the optimizer step, not worker RNG state/prefetch.
SIGINT/SIGTERM save the next step and optimizer state before exiting.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from distill.common import DATA, append_json, atomic_json, atomic_write, external_path
from distill.dataset import Crops, StepBatches
from distill.student import Student, StudentConfig, defaults
from distill.resume_teacher import load_plan, validate_resume_plan

STOP = False


def request_stop(signum, frame):
    global STOP
    STOP = True


def losses(prediction, target, mask, spectral_weight=.02, gradient_weight=.05, channel_weights=None):
    p, t = prediction.float(), target.float()
    weights = p.new_ones(1, p.shape[1], 1, 1) if channel_weights is None else p.new_tensor(channel_weights).view(1, -1, 1, 1)
    denominator = (mask.sum()*weights.sum()).clamp_min(1)
    mse = ((p-t).square()*mask*weights).sum()/denominator
    gradient = p.new_zeros(())
    for axis in (-1, -2):
        dp, dt = p.diff(dim=axis), t.diff(dim=axis)
        valid = mask.narrow(axis, 1, mask.shape[axis]-1)*mask.narrow(axis, 0, mask.shape[axis]-1)
        gradient = gradient + ((dp-dt).square()*valid*weights).sum()/(valid.sum()*weights.sum()).clamp_min(1)
    spectral = p.new_zeros(())
    if spectral_weight:
        for factor in (1, 2, 4):
            pp, tt = p*mask, t*mask
            if factor > 1:
                pp, tt = F.avg_pool2d(pp, factor), F.avg_pool2d(tt, factor)
            height, width = pp.shape[-2:]
            hann = torch.hann_window(height, device=p.device)[:, None]*torch.hann_window(width, device=p.device)[None]
            ap, at = torch.fft.rfft2(pp*hann, norm='ortho').abs(), torch.fft.rfft2(tt*hann, norm='ortho').abs()
            # Relative log magnitudes discourage the MSE student's smoothing.
            scale = at.detach().mean((-2, -1), keepdim=True).clamp_min(.01)
            terms = (torch.log1p(ap/scale)-torch.log1p(at/scale)).square().mean((-2, -1), keepdim=True)
            spectral = spectral + (terms*weights).sum()/(p.shape[0]*weights.sum())/3
    return mse + gradient_weight*gradient + spectral_weight*spectral, dict(
        mse=mse, gradient=gradient, spectral=spectral)


def coarse_delta_loss(prediction, target, mask):
    # Dataset output uses the same ~39.74 scale for mean and p5. The teacher's
    # mean-minus-p5 channel has scale ~1.77: supervise this smaller difference
    # explicitly instead of losing it inside two almost identical height maps.
    factor = 39.741999742263 / 1.7681844104569366
    delta = ((prediction[:, 0]-prediction[:, 1]).float()-(target[:, 0]-target[:, 1]))*factor
    return (delta.square()*mask[:, 0]).sum()/mask.sum().clamp_min(1)


def lowfreq_height_mae(prediction, target, mask):
    """MAE in metres of the low-frequency height contribution, not the full decoder."""
    p, t = prediction[:, 4].float()*38.6-31.4, target[:, 4].float()*38.6-31.4
    error = (p.sign()*p.square()-t.sign()*t.square()).abs()
    return (error*mask[:, 0]).sum()/mask.sum().clamp_min(1)


def coarse_height_mae(prediction, target, mask, scaling):
    """Mean/p5 altitude proxy in metres, using the teacher's recorded scales."""
    means, stds = (prediction.new_tensor(value[:2]).view(1, 2, 1, 1) for value in scaling)
    p, t = prediction[:, :2].float()*stds+means, target[:, :2].float()*stds+means
    error = (p.sign()*p.square()-t.sign()*t.square()).abs()
    return (error*mask).sum()/(2*mask.sum()).clamp_min(1)


def spectral_band_loss(prediction, target, mask):
    """Relative power errors in the five radial bands used by physical evaluation."""
    p, t = prediction.float(), target.float()
    h, w = p.shape[-2:]
    hann = torch.hann_window(h, device=p.device)[:, None]*torch.hann_window(w, device=p.device)[None]
    def power(value):
        mean = (value*mask).sum((-2, -1), keepdim=True)/mask.sum((-2, -1), keepdim=True).clamp_min(1)
        return torch.fft.rfft2((value-mean)*mask*hann, norm='ortho').abs().square()
    pp, tt = power(p), power(t)
    yy = torch.fft.fftfreq(h, device=p.device)[:, None]
    xx = torch.fft.rfftfreq(w, device=p.device)[None]
    radius = (yy.square()+xx.square()).sqrt()
    multiplicity = p.new_full((w//2+1,), 2.)
    multiplicity[0] = 1.
    if w % 2 == 0:
        multiplicity[-1] = 1.
    multiplicity = multiplicity.expand(h, -1)
    # Each band has its own scale; dominant low frequencies cannot hide small
    # high-frequency errors. A relative floor keeps silent bands finite.
    floor = tt.detach().mean((-2, -1)).clamp_min(1e-8)*1e-6
    terms = []
    for lo, hi in ((0, .03125), (.03125, .0625), (.0625, .125), (.125, .25), (.25, .5)):
        band = (radius >= lo) & (radius < hi)
        if band.any():
            weight = multiplicity[band]
            a = (pp[..., band]*weight).sum(-1)/weight.sum()
            b = (tt[..., band]*weight).sum(-1)/weight.sum()
            terms.append((torch.log(a+floor)-torch.log(b+floor)).square().mean())
    return torch.stack(terms).mean()


def centre(prediction, target):
    hy, hx = ((prediction.shape[-2]-target.shape[-2])//2,
              (prediction.shape[-1]-target.shape[-1])//2)
    if min(hy, hx) < 0:
        raise ValueError('Prediction is smaller than target.')
    return prediction[..., hy:hy+target.shape[-2], hx:hx+target.shape[-1]]


@torch.no_grad()
def validate(model, dataset, device, count=12, spectral_band_weight=0., coarse_scaling=None, extra_files=()):
    was_training = model.training
    model.eval()
    totals = dict(mse=0., gradient=0., spectral=0., slope_ratio=0., spectrum_ratio=0.)
    channel_mse = torch.zeros(model.config.out_channels, device=device)
    height_mae = 0.
    delta_mse = 0.
    coarse_mae = 0.
    if spectral_band_weight:
        totals['spectral_bands'] = 0.
    count = min(count, len(dataset))
    # Cover the sorted validation index range, rather than only its first profile.
    indices = torch.linspace(0, len(dataset)-1, count).round().long().tolist()
    lookup = {path.name: i for i, path in enumerate(dataset.paths)}
    if set(extra_files) - set(lookup):
        raise ValueError('Additional rare validation files differ from this dataset.')
    indices = sorted(set(indices) | {lookup[name] for name in extra_files})
    count = len(indices)
    for index in indices:
        inputs, target, mask = dataset[index]
        inputs, target, mask = (v[None].to(device) for v in (inputs, target, mask))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            pred = centre(model(inputs), target)
        _, values = losses(pred, target, mask)
        for key, value in values.items():
            totals[key] += float(value)/count
        p, t = pred.float(), target.float()
        if spectral_band_weight:
            totals['spectral_bands'] += float(spectral_band_loss(p, t, mask))/count
        channel_mse += (((p-t).square()*mask).sum((0, 2, 3))/mask.sum().clamp_min(1))/count
        if model.config.stage == 'base':
            # A diagnostic in metres for the low-frequency component only;
            # full physical elevation still requires the decoder/site audit.
            height_mae += float(lowfreq_height_mae(p, t, mask))/count
        elif model.config.stage == 'coarse':
            delta_mse += float(coarse_delta_loss(p, t, mask))/count
            if coarse_scaling is not None:
                coarse_mae += float(coarse_height_mae(p, t, mask, coarse_scaling))/count
        slope_p = p.diff(dim=-1).square().mean()+p.diff(dim=-2).square().mean()
        slope_t = t.diff(dim=-1).square().mean()+t.diff(dim=-2).square().mean()
        totals['slope_ratio'] += float((slope_p/slope_t.clamp_min(1e-8)).sqrt())/count
        psd_p = torch.fft.rfft2(p-p.mean((-2, -1), keepdim=True)).abs().square().mean()
        psd_t = torch.fft.rfft2(t-t.mean((-2, -1), keepdim=True)).abs().square().mean()
        totals['spectrum_ratio'] += float(psd_p/psd_t.clamp_min(1e-8))/count
    model.train(was_training)
    totals['channel_mse'] = channel_mse.cpu().tolist()
    if model.config.stage == 'base':
        totals['lowfreq_height_mae_m_proxy'] = height_mae
    elif model.config.stage == 'coarse':
        totals['mean_minus_p5_mse'] = delta_mse
        if coarse_scaling is not None:
            totals['coarse_height_mae_m_proxy'] = coarse_mae
    return totals


def make_loader(dataset, batch, start, stop, seed, workers, device, probabilities=None):
    kwargs = dict(dataset=dataset, batch_sampler=StepBatches(len(dataset), batch, start, stop, seed, probabilities),
                  num_workers=workers, pin_memory=device.type == 'cuda',
                  generator=torch.Generator().manual_seed(seed))
    if workers:
        kwargs.update(multiprocessing_context='spawn', prefetch_factor=2, persistent_workers=True)
    return DataLoader(**kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--stage', choices=['coarse', 'base', 'decoder'], required=True)
    parser.add_argument('--dataset', type=Path, default=DATA/'crops/main')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--width', type=int)
    parser.add_argument('--coarse-solver-steps', type=int)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--train-size', type=int)
    parser.add_argument('--eval-every', type=int, default=2000)
    parser.add_argument('--val-count', type=int, default=12)
    parser.add_argument('--save-seconds', type=float, default=120)
    parser.add_argument('--seed', type=int, default=8675309)
    parser.add_argument('--spectral-weight', type=float, default=.02)
    parser.add_argument('--spectral-band-weight', type=float, default=0.)
    parser.add_argument('--gradient-weight', type=float, default=.05)
    parser.add_argument('--height-weight', type=float, default=4.,
                        help='Relative MSE weight for base lowfreq / coarse mean+p5 channels.')
    parser.add_argument('--coarse-delta-weight', type=float, default=.25)
    parser.add_argument('--height-mae-weight', type=float, default=0.,
                        help='Base low-frequency / coarse mean+p5 height MAE weight; one loss unit is 100 metres.')
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--allow-data-growth', action='store_true')
    parser.add_argument('--sample-weights', type=str, help='Frozen train-only sampling policy JSON.')
    parser.add_argument('--allow-sampling-change', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.batch, args.eval_every, args.val_count) < 1 or args.workers < 0:
        parser.error('Positive steps/batch/evaluation intervals and nonnegative workers required.')
    if args.height_weight <= 0 or args.lr <= 0 or min(args.spectral_weight, args.spectral_band_weight, args.gradient_weight, args.coarse_delta_weight, args.height_mae_weight) < 0:
        parser.error('Positive height weight/LR and nonnegative spectral/gradient weights required.')
    device = torch.device('cpu' if args.cpu else 'cuda:0')
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    output = external_path(args.output or DATA/'ckpt'/args.stage)
    dataset_root = external_path(args.dataset)
    output.mkdir(parents=True, exist_ok=True)
    saved = torch.load(args.resume, map_location='cpu', weights_only=True) if args.resume else None
    if saved:
        if saved['arguments']['seed'] != args.seed:
            raise ValueError('Sampler seed differs from the resumed checkpoint.')
        config_dict = saved['config'].copy()
        config_dict['dilations'] = tuple(config_dict['dilations'])
        config = StudentConfig(**config_dict)
        if config.stage != args.stage or (args.width and config.width != args.width):
            raise ValueError('Resume stage/width differ from the checkpoint.')
        if args.coarse_solver_steps is not None and config.solver_steps != args.coarse_solver_steps:
            raise ValueError('Resumed solver steps differ from the checkpoint.')
    else:
        config = defaults(args.stage, args.width)
        if args.coarse_solver_steps is not None:
            config = replace(config, solver_steps=args.coarse_solver_steps)
    model = Student(config).to(device)
    halo = model.halo if args.stage == 'base' else 0
    train = Crops(dataset_root, args.stage, halo=halo, train_size=args.train_size)
    coarse_scaling = None
    if args.stage == 'coarse':
        with np.load(train.paths[0], allow_pickle=False) as sample:
            coarse_scaling = [sample['output_means'].tolist(), sample['output_stds'].tolist()]
    validation = Crops(dataset_root, args.stage, 'val', halo=halo,
                       train_size=args.train_size, overfit=args.overfit)
    # The exact list is checkpointed. Appending new data requires explicit admission.
    files = [path.name for path in train.paths]
    manifest = json.loads((dataset_root/'manifest.json').read_text())
    seed_plan = load_plan(dataset_root, manifest)
    sampling_policy = saved.get('sampling_policy') if saved else None
    if args.sample_weights:
        proposed_policy = json.loads(Path(args.sample_weights).expanduser().read_text())
        if saved and proposed_policy != sampling_policy and not args.allow_sampling_change:
            raise ValueError('Sampling policy changed; explicit --allow-sampling-change is required.')
        sampling_policy = proposed_policy
    if sampling_policy:
        import hashlib
        if (args.stage != 'base' or sampling_policy['stage'] != args.stage or
            sampling_policy['train_files'] != files or sampling_policy['dataset_manifest_digest'] !=
                hashlib.sha256((dataset_root/'manifest.json').read_bytes()).hexdigest()):
            raise ValueError('Sampling policy must match this base dataset and its exact training files.')
    probabilities = sampling_policy['probabilities'] if sampling_policy else None
    rare_validation_files = sampling_policy.get('validation_files', []) if sampling_policy else []
    if saved:
        if saved['dataset_manifest'] != manifest:
            raise ValueError('Dataset provenance differs from the checkpoint.')
        validate_resume_plan(saved.get('seed_plan', {}), seed_plan, saved['train_files'], manifest, args.allow_data_growth)
        if saved['train_files'] != files:
            if not args.allow_data_growth or not set(saved['train_files']).issubset(files):
                raise ValueError('Training files changed; allow-data-growth admits only added files.')
        model.load_state_dict(saved['model'])
    ema = Student(config).to(device).eval().requires_grad_(False)
    ema.load_state_dict(saved['ema'] if saved else model.state_dict())
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=1e-4,
                                 fused=device.type == 'cuda')
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        if device.type == 'cuda' and 'cuda_rng' in saved:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    start = saved['step'] if saved else 0
    if start > args.steps:
        raise ValueError('Target --steps is smaller than the resumed optimizer step.')
    best = saved.get('best_score', float('inf')) if saved else float('inf')
    if saved and (saved['arguments'].get('train_size') != args.train_size or
                  saved['arguments'].get('height_weight', 1.) != args.height_weight or
                  saved['arguments'].get('coarse_delta_weight', 0.) != args.coarse_delta_weight or
                  saved['arguments'].get('height_mae_weight', 0.) != args.height_mae_weight or
                  saved['arguments'].get('spectral_band_weight', 0.) != args.spectral_band_weight or
                  saved['arguments'].get('gradient_weight', .05) != args.gradient_weight or
                  saved['arguments'].get('spectral_weight', .02) != args.spectral_weight or
                  saved.get('sampling_policy') != sampling_policy):
        # A new target crop changes the validation footprint. Do not compare its
        # best score to the old crop's score; retain model/optimizer/EMA states.
        best = float('inf')
    # Explicitly increasing --steps is supported; LR progression is logged.
    loader = make_loader(train, args.batch, start, args.steps, args.seed, args.workers, device, probabilities)
    run_started, last_save = time.monotonic(), time.monotonic()
    last_validation, step = saved.get('validation') if saved else None, start
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    atomic_json(output/'run.json', dict(config=asdict(config), dataset=str(dataset_root),
                arguments=vars(args) | dict(dataset=str(args.dataset), output=str(output),
                resume=str(args.resume) if args.resume else None), start_step=start,
                training_crops=len(train), validation_crops=len(validation), halo=halo,
                sampling_policy=sampling_policy,
                parameters=sum(p.numel() for p in model.parameters())))

    def checkpoint(name='latest.pt'):
        state = dict(config=asdict(config), model=model.state_dict(), ema=ema.state_dict(),
                     optimizer=optimizer.state_dict(), step=step, best_score=best,
                     rng=torch.get_rng_state(), validation=last_validation,
                     train_files=files, dataset_manifest=manifest, arguments=vars(args) |
                     dict(dataset=str(args.dataset), output=str(output), resume=str(args.resume) if args.resume else None))
        state['seed_plan'] = seed_plan
        state['sampling_policy'] = sampling_policy
        if saved and saved.get('warm_start'):
            state['warm_start'] = saved['warm_start']
        if device.type == 'cuda':
            state['cuda_rng'] = torch.cuda.get_rng_state_all()
        atomic_write(output/name, lambda handle: torch.save(state, handle))

    model.train()
    channel_weights = ([1., 1., 1., 1., args.height_weight] if args.stage == 'base' else
                       [args.height_weight, args.height_weight, 1., 1., 1., 1.] if args.stage == 'coarse' else [1.])
    try:
        for inputs, target, mask in loader:
            if STOP:
                break
            inputs, target, mask = (v.to(device, non_blocking=True) for v in (inputs, target, mask))
            warmup = min(1000, max(1, args.steps//20))
            progress = max(0, (step-warmup)/max(1, args.steps-warmup))
            rate = args.lr*min(1, (step+1)/warmup)*(.1+.9*(1+math.cos(math.pi*progress))/2)
            for group in optimizer.param_groups:
                group['lr'] = rate
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
                prediction = centre(model(inputs), target)
                loss, values = losses(prediction, target, mask, args.spectral_weight, args.gradient_weight, channel_weights)
                if args.spectral_band_weight:
                    band_error = spectral_band_loss(prediction, target, mask)
                    loss = loss + args.spectral_band_weight*band_error
                    values['spectral_bands'] = band_error
                if args.stage == 'coarse' and args.coarse_delta_weight:
                    delta_loss = coarse_delta_loss(prediction.float(), target, mask)
                    loss = loss + args.coarse_delta_weight*delta_loss
                    values['mean_minus_p5'] = delta_loss
                if args.stage == 'base' and args.height_mae_weight:
                    height_error = lowfreq_height_mae(prediction, target, mask)
                    loss = loss + args.height_mae_weight*height_error/100
                    values['lowfreq_height_mae_m_proxy'] = height_error
                if args.stage == 'coarse' and args.height_mae_weight:
                    height_error = coarse_height_mae(prediction, target, mask, coarse_scaling)
                    loss = loss + args.height_mae_weight*height_error/100
                    values['coarse_height_mae_m_proxy'] = height_error
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite training loss at step {step}.')
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True)
            optimizer.step()
            step += 1
            decay = min(.999, (1+step)/(10+step))
            with torch.no_grad():
                for average, parameter in zip(ema.parameters(), model.parameters()):
                    average.lerp_(parameter, 1-decay)
            if step % 100 == 0 or step == start+1:
                item = dict(step=step, loss=float(loss), lr=rate, grad_norm=float(gradient_norm),
                            elapsed_seconds=time.monotonic()-run_started,
                            **{key: float(value) for key, value in values.items()})
                append_json(output/'train.jsonl', item)
                print(json.dumps(item), flush=True)
            if step % args.eval_every == 0 or step == args.steps:
                last_validation = validate(ema, validation, device, args.val_count, args.spectral_band_weight,
                                           coarse_scaling, rare_validation_files)
                weighted_mse = sum(w*v for w, v in zip(channel_weights, last_validation['channel_mse']))/sum(channel_weights)
                score = weighted_mse+args.gradient_weight*last_validation['gradient']+args.spectral_weight*last_validation['spectral']
                score += args.spectral_band_weight*last_validation.get('spectral_bands', 0.)
                if args.stage == 'coarse':
                    score += args.coarse_delta_weight*last_validation['mean_minus_p5_mse']
                    score += args.height_mae_weight*last_validation['coarse_height_mae_m_proxy']/100
                if args.stage == 'base':
                    score += args.height_mae_weight*last_validation['lowfreq_height_mae_m_proxy']/100
                item = dict(step=step, validation=last_validation, score=score,
                            elapsed_seconds=time.monotonic()-run_started)
                append_json(output/'validation.jsonl', item)
                print(json.dumps(item), flush=True)
                if score < best:
                    best = score
                    checkpoint('best.pt')
            if time.monotonic()-last_save >= args.save_seconds:
                checkpoint()
                last_save = time.monotonic()
                atomic_json(output/'status.json', dict(status='running', step=step, target_steps=args.steps,
                            validation=last_validation, training_crops=len(train), checkpoint=str(output/'latest.pt')))
        checkpoint()
        status = 'interrupted' if STOP else 'complete'
        atomic_json(output/'status.json', dict(status=status, step=step, target_steps=args.steps,
                    validation=last_validation, elapsed_seconds=time.monotonic()-run_started))
    except Exception as exc:
        # Save the last completed optimizer step even after an ordinary exception.
        # An OOM may leave transient gradients; they are not included in state_dict.
        checkpoint()
        atomic_json(output/'status.json', dict(status='failed', step=step, error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()

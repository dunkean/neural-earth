"""BF16 regression with spectral/gradient losses, atomic checkpoints and resume.

  CUDA_VISIBLE_DEVICES=1 python -m distill.train --stage base --dataset ~/data/distill/crops/main
  ... --resume ~/data/distill/ckpt/base/latest.pt

Sample addresses depend on the optimizer step, not worker RNG state/prefetch.
SIGINT/SIGTERM save the next step and optimizer state before exiting.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import signal
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from distill.common import DATA, append_json, atomic_json, atomic_write, external_path
from distill.dataset import Crops, StepBatches
from distill.student import Student, StudentConfig, defaults

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


def centre(prediction, target):
    hy, hx = ((prediction.shape[-2]-target.shape[-2])//2,
              (prediction.shape[-1]-target.shape[-1])//2)
    if min(hy, hx) < 0:
        raise ValueError('Prediction is smaller than target.')
    return prediction[..., hy:hy+target.shape[-2], hx:hx+target.shape[-1]]


@torch.no_grad()
def validate(model, dataset, device, count=12):
    was_training = model.training
    model.eval()
    totals = dict(mse=0., gradient=0., spectral=0., slope_ratio=0., spectrum_ratio=0.)
    channel_mse = torch.zeros(model.config.out_channels, device=device)
    height_mae = 0.
    count = min(count, len(dataset))
    # Cover the sorted validation index range, rather than only its first profile.
    indices = torch.linspace(0, len(dataset)-1, count).round().long().tolist()
    for index in indices:
        inputs, target, mask = dataset[index]
        inputs, target, mask = (v[None].to(device) for v in (inputs, target, mask))
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            pred = centre(model(inputs), target)
        _, values = losses(pred, target, mask)
        for key, value in values.items():
            totals[key] += float(value)/count
        p, t = pred.float(), target.float()
        channel_mse += (((p-t).square()*mask).sum((0, 2, 3))/mask.sum().clamp_min(1))/count
        if model.config.stage == 'base':
            # A diagnostic in metres for the low-frequency component only;
            # full physical elevation still requires the decoder/site audit.
            pp, tt = p[:, 4]*38.6-31.4, t[:, 4]*38.6-31.4
            delta = (pp.sign()*pp.square()-tt.sign()*tt.square()).abs()
            height_mae += float((delta*mask[:, 0]).sum()/mask.sum().clamp_min(1))/count
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
    return totals


def make_loader(dataset, batch, start, stop, seed, workers, device):
    kwargs = dict(dataset=dataset, batch_sampler=StepBatches(len(dataset), batch, start, stop, seed),
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
    parser.add_argument('--gradient-weight', type=float, default=.05)
    parser.add_argument('--height-weight', type=float, default=4.,
                        help='Relative MSE weight for base lowfreq / coarse mean+p5 channels.')
    parser.add_argument('--overfit', action='store_true')
    parser.add_argument('--allow-data-growth', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()
    if min(args.steps, args.batch, args.eval_every, args.val_count) < 1 or args.workers < 0:
        parser.error('Positive steps/batch/evaluation intervals and nonnegative workers required.')
    if args.height_weight <= 0 or args.lr <= 0 or min(args.spectral_weight, args.gradient_weight) < 0:
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
    else:
        config = defaults(args.stage, args.width)
    model = Student(config).to(device)
    halo = model.halo if args.stage == 'base' else 0
    train = Crops(dataset_root, args.stage, halo=halo, train_size=args.train_size)
    validation = Crops(dataset_root, args.stage, 'val', halo=halo,
                       train_size=args.train_size, overfit=args.overfit)
    # The exact list is checkpointed. Appending new data requires explicit admission.
    files = [path.name for path in train.paths]
    manifest = json.loads((dataset_root/'manifest.json').read_text())
    if saved:
        if saved['dataset_manifest'] != manifest:
            raise ValueError('Dataset provenance differs from the checkpoint.')
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
                  saved['arguments'].get('height_weight', 1.) != args.height_weight):
        # A new target crop changes the validation footprint. Do not compare its
        # best score to the old crop's score; retain model/optimizer/EMA states.
        best = float('inf')
    # Explicitly increasing --steps is supported; LR progression is logged.
    loader = make_loader(train, args.batch, start, args.steps, args.seed, args.workers, device)
    run_started, last_save = time.monotonic(), time.monotonic()
    last_validation, step = saved.get('validation') if saved else None, start
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, request_stop)
    atomic_json(output/'run.json', dict(config=asdict(config), dataset=str(dataset_root),
                arguments=vars(args) | dict(dataset=str(args.dataset), output=str(output),
                resume=str(args.resume) if args.resume else None), start_step=start,
                training_crops=len(train), validation_crops=len(validation), halo=halo,
                parameters=sum(p.numel() for p in model.parameters())))

    def checkpoint(name='latest.pt'):
        state = dict(config=asdict(config), model=model.state_dict(), ema=ema.state_dict(),
                     optimizer=optimizer.state_dict(), step=step, best_score=best,
                     rng=torch.get_rng_state(), validation=last_validation,
                     train_files=files, dataset_manifest=manifest, arguments=vars(args) |
                     dict(dataset=str(args.dataset), output=str(output), resume=str(args.resume) if args.resume else None))
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
                last_validation = validate(ema, validation, device, args.val_count)
                weighted_mse = sum(w*v for w, v in zip(channel_weights, last_validation['channel_mse']))/sum(channel_weights)
                score = weighted_mse+.05*last_validation['gradient']+.02*last_validation['spectral']
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

"""Generate resumable teacher targets on GPU 0; never modify upstream weights.

  python -m distill.teacher --output ~/data/distill/crops/smoke --count 50 --profiles natural
  python -m distill.teacher --output ~/data/distill/crops/main --count 20000

An index always identifies the same seed/profile/position. Each NPZ is committed
atomically. Validation uses entirely separate worlds, not neighbouring crops.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from distill.common import (DATA, HOLDOUT_SEEDS, REVISION, SCHEMA, append_json,
                            atomic_json, atomic_write, external_path, source_digest)
from distill.student import Student, defaults

PROFILES = ('natural', 'orogen', 'terrestrial-earthlike', 'terrestrial-archipelago',
            'terrestrial-continents', 'terrestrial-gondwana')
STOP = False


def stop_requested(signum, frame):
    global STOP
    STOP = True


class Worlds:
    def __init__(self, seed):
        from terrain_app import load_pipeline
        self.loaded = load_pipeline(seed)
        self.config = {k: v for k, v in dict(self.loaded.config).items() if not k.startswith('_')}

    def create(self, seed, profile):
        from terrain_diffusion.inference.world_pipeline import WorldPipeline
        from terrain_inference import configure_world
        world = WorldPipeline(**(self.config | dict(seed=seed, dtype='bf16', T=2,
                    latents_batch_size=16, cache_limit=1024**3, torch_compile=False,
                    log_mode='silent', onestep_latent=False)))
        world.coarse_model, world.base_model, world.decoder_model = (
            self.loaded.coarse_model, self.loaded.base_model, self.loaded.decoder_model)
        configure_world(world, replace(self.loaded._terrain_profile, cuda_graphs=False,
                                       coarse_streams=1), world_profile=profile)
        world.bind()
        if world.kwargs['coarse_pooling'] != 1:
            raise ValueError('Dataset schema currently requires coarse_pooling=1.')
        return world


def physical_coarse_scaling(world):
    means = np.array(world.kwargs['coarse_means'], np.float32)
    stds = np.array(world.kwargs['coarse_stds'], np.float32)
    means[1] = means[0]-means[1]
    stds[1] = stds[0]
    return means, stds


def unweight(value):
    value = value.detach().float().cpu()
    weight = value[-1:]
    if not torch.isfinite(value).all() or (weight < 0).any():
        raise FloatingPointError('Non-finite teacher or negative blending weight.')
    return (value[:-1]/weight.clamp_min(1e-12)).numpy(), (weight > 0).numpy()


def store(path, metadata, **arrays):
    for name, value in arrays.items():
        if not np.isfinite(value).all():
            raise FloatingPointError(f'Non-finite stored {name}.')
    arrays['metadata'] = np.array(json.dumps(metadata, allow_nan=False))
    atomic_write(path, lambda handle: np.savez(handle, **arrays))


def position(world, index, seed, crop):
    # Cheap raw conditioning guides a 50/50 prior land/ocean proposal mixture.
    # Record the actual teacher land fraction; it is audited after generation.
    rng = np.random.default_rng(seed ^ (index*0x9e3779b9))
    land = index % 2 == 0
    y = x = 0
    for _ in range(12):
        y, x = (int(v)*32 for v in rng.integers(-1024, 1024, size=2))
        raw = world._conditioning_model_input(y//32, y//32+crop//32,
                                               x//32, x//32+crop//32)
        fraction = float((raw[0] > 0).float().mean())
        if (land and fraction >= .5) or (not land and fraction <= .2):
            break
    return y, x


@torch.inference_mode()
def generate_sample(world, index, seed, profile, split, output, stages, crop, halo):
    from terrain_inference import _coarse_batch
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
    y, x = position(world, index, seed, crop)
    metadata = dict(schema=SCHEMA, index=index, seed=seed, profile=profile, split=split,
                    y=y, x=x, crop=crop, halo=halo, teacher_revision=REVISION)
    paths = {stage: output/stage/f'{split}-{index:07d}.npz' for stage in stages}
    land_fraction = None
    if 'base' in stages and not paths['base'].exists():
        target, mask = unweight(world.latents[:, y:y+crop, x:x+crop])
        cy, cx = (y-halo)//32, (x-halo)//32
        cells = (crop+2*halo)//32
        conditioning, _ = unweight(world.coarse[:, cy:cy+cells, cx:cx+cells])
        land_fraction = float((conditioning[0] > 0).mean())
        store(paths['base'], metadata | dict(stage='base', coarse_y=cy, coarse_x=cx,
              land_fraction=land_fraction), target=target.astype(np.float16),
              mask=mask, coarse=conditioning.astype(np.float32),
              histogram=np.array(world.kwargs['histogram_raw'], np.float32))
    if 'coarse' in stages and not paths['coarse'].exists():
        ci, cj = (y//32)//48, (x//32)//48
        ctx = (0, ci, cj)
        scheduler = EDMDPMSolverMultistepScheduler(sigma_min=.002, sigma_max=80, sigma_data=.5)
        t_cond = torch.atan(torch.tensor(world.kwargs['cond_snr'])).to(world.device, dtype=torch.bfloat16)
        labels = [v.detach().view(-1) for v in torch.log(torch.tan(t_cond)/8)]
        weight = linear_weight_window(64, 'cpu', torch.float32).to(world.device)
        inputs = _coarse_batch(world, [ctx], scheduler, weight, t_cond, labels, 1, prepare_only=True)
        target, mask = unweight(world.coarse._f([ctx])[0])
        means, stds = physical_coarse_scaling(world)
        target = (target-means[:, None, None])/stds[:, None, None]
        store(paths['coarse'], metadata | dict(stage='coarse', y=ci*48, x=cj*48, crop=64, halo=0),
              target=target.astype(np.float16), mask=mask,
              condition=inputs.conditions[0][0].float().cpu().numpy().astype(np.float16),
              labels=np.array([float(v[0]) for v in inputs.conditions[1:]], np.float32),
              output_means=means, output_stds=stds)
    if 'decoder' in stages and not paths['decoder'].exists():
        size, stride, compression = world.decoder_tile_size, world.decoder_tile_stride, world.latent_compression
        di, dj = (y*compression)//stride, (x*compression)//stride
        dy, dx = di*stride, dj*stride
        latent = world.latents[:, dy//compression:(dy+size)//compression,
                                  dx//compression:(dx+size)//compression]
        target, mask = unweight(world.residual._f([(0, di, dj)], [latent])[0])
        conditioning, _ = unweight(latent)
        store(paths['decoder'], metadata | dict(stage='decoder', y=dy, x=dx, crop=size, halo=0),
              target=target.astype(np.float16), mask=mask,
              latents=conditioning[:4].astype(np.float16))
    torch.cuda.synchronize()
    return dict(index=index, seed=seed, profile=profile, split=split, y=y, x=x,
                land_fraction=land_fraction)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=DATA/'crops/main')
    parser.add_argument('--count', type=int, default=20000)
    parser.add_argument('--seed-base', type=int, default=10000)
    parser.add_argument('--samples-per-world', type=int, default=32)
    parser.add_argument('--profiles', nargs='+', default=list(PROFILES), choices=PROFILES)
    parser.add_argument('--stages', nargs='+', default=['base', 'coarse', 'decoder'], choices=['base', 'coarse', 'decoder'])
    parser.add_argument('--crop', type=int, default=256)
    parser.add_argument('--halo', type=int, default=192)
    args = parser.parse_args()
    output = external_path(args.output)
    if args.count < 1 or args.samples_per_world < 1 or args.crop % 32 or args.halo % 32:
        parser.error('Positive count/world size and crop/halo aligned to 32 required.')
    if args.halo < Student(defaults('base')).halo and 'base' in args.stages:
        parser.error('Halo is smaller than the base student receptive field.')
    signature = dict(schema=SCHEMA, revision=REVISION, seed_base=args.seed_base,
                     samples_per_world=args.samples_per_world, profiles=args.profiles,
                     stages=args.stages, crop=args.crop, halo=args.halo,
                     source_digest=source_digest(), validation='world_index // num_profiles % 50 == 49',
                     reserved_seeds=sorted(HOLDOUT_SEEDS))
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != signature:
        raise SystemExit('Dataset provenance changed. Use a new output directory; do not mix targets.')
    atomic_json(manifest_path, signature)
    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop_requested)
    torch.set_num_threads(8)
    worlds, world, current = None, None, None
    started, completed = time.monotonic(), 0
    try:
        # Generate validation worlds first, so GPU 1 can start training while
        # GPU 0 fills the rest. Every profile gets its own held-out worlds.
        def is_validation(index):
            return (index//args.samples_per_world)//len(args.profiles) % 50 == 49
        order = [i for i in range(args.count) if is_validation(i)]
        order += [i for i in range(args.count) if not is_validation(i)]
        for index in order:
            if STOP:
                break
            world_index = index//args.samples_per_world
            seed = args.seed_base+world_index
            if seed in HOLDOUT_SEEDS:
                raise ValueError(f'Reserved evaluation seed {seed}; change seed-base.')
            profile = args.profiles[world_index % len(args.profiles)]
            split = 'val' if is_validation(index) else 'train'
            if all((output/stage/f'{split}-{index:07d}.npz').exists() for stage in args.stages):
                continue
            if current != world_index:
                if world is not None:
                    world.close()
                    del world
                    gc.collect()
                    torch.cuda.empty_cache()
                if worlds is None:
                    worlds = Worlds(seed)
                world = worlds.create(seed, profile)
                current = world_index
            before = time.monotonic()
            item = generate_sample(world, index, seed, profile, split, output,
                                   args.stages, args.crop, args.halo)
            completed += 1
            item.update(seconds=time.monotonic()-before, completed_this_run=completed,
                        elapsed_seconds=time.monotonic()-started)
            append_json(output/'generation.jsonl', item)
            atomic_json(output/'progress.json', item | dict(requested=args.count, status='running'))
            if completed % 10 == 0 or completed == 1:
                print(json.dumps(item), flush=True)
        atomic_json(output/'status.json', dict(status='interrupted' if STOP else 'complete',
                    requested=args.count, completed_this_run=completed,
                    elapsed_seconds=time.monotonic()-started))
    finally:
        if world is not None:
            world.close()


if __name__ == '__main__':
    main()

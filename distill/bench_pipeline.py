"""Time fresh final latent fields, including CPU features and CPU/GPU transfers.

Coarse fields are prefetched outside the timed region for both variants: this
measures the complete base stage rather than hiding its cost behind coarse or
decoder inference. Each sample uses a fresh in-memory world, with resident
weights and warmed kernels. Run only when the selected GPU is otherwise idle.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

from distill.common import REPO, atomic_json, external_path
from distill.inference import install
from distill.student import load_student
from tools.verification.compare_base_variants import Count


def require_idle_gpu():
    uuid = str(torch.cuda.get_device_properties(0).uuid).lower()
    output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    other_pids = []
    for line in output.splitlines():
        gpu, pid = (value.strip() for value in line.split(',', 1))
        if gpu.lower().removeprefix('gpu-') == uuid and pid != str(os.getpid()):
            other_pids.append(pid)
    if other_pids:
        raise RuntimeError(f'Selected GPU has other compute processes: {other_pids}. Do not benchmark concurrently.')


def summarize(rows, sizes):
    results = []
    for size in sizes:
        times = {}
        for variant in ('reference', 'student'):
            selected = [row for row in rows if row['size'] == size and row['variant'] == variant and not row['warmup']]
            if not selected:
                raise ValueError(f'Missing {variant} measurements for {size}.')
            times[variant] = statistics.median(row['seconds'] for row in selected)
        results.append(dict(size=size, teacher_seconds=times['reference'], student_seconds=times['student'],
                            speedup=times['reference']/times['student']))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[1024, 2048])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--seed', type=int, default=101)
    parser.add_argument('--profile', default='natural')
    args = parser.parse_args()
    if args.repeats < 3 or len(set(args.sizes)) != len(args.sizes) or any(size < 512 or size % 512 for size in args.sizes):
        parser.error('At least three repeats and distinct positive sizes aligned to 512 are required.')
    require_idle_gpu()
    output = external_path(args.output)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    payload = args.checkpoint.read_bytes()
    model, saved = load_student(io.BytesIO(payload), 'cuda:0')
    if model.config.stage != 'base':
        parser.error('This benchmark measures the base stage.')
    sources = {'code:'+name: hashlib.sha256((REPO/name).read_bytes()).hexdigest()
               for name in ('distill/student.py', 'distill/features.py', 'distill/inference.py')}
    signature = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
                     checkpoint_digest=hashlib.sha256(payload).hexdigest(), student_source_digests=sources,
                     stage='base', dtype='bf16', step=saved['step'], sizes=args.sizes, repeats=args.repeats,
                     seed=args.seed, profile=args.profile, includes_feature_construction=True,
                     includes_transfers=True, coarse_prefetched=True, fresh_world_per_sample=True,
                     note='Base stage only; coarse prefetch, model loading and kernel warmup are excluded.')
    report = signature | dict(rows=[], status='running')
    if output.exists():
        existing = json.loads(output.read_text())
        if any(existing.get(key) != value for key, value in signature.items()):
            raise ValueError('Benchmark provenance changed; use a new output file.')
        report['rows'] = existing['rows']
    from terrain_app import load_pipeline
    from terrain_inference import configure_world
    from terrain_diffusion.inference.world_pipeline import WorldPipeline
    loaded = load_pipeline(args.seed)
    config = {key: value for key, value in dict(loaded.config).items() if not key.startswith('_')}
    for variant in ('reference', 'student'):
        for size in args.sizes:
            # Always rewarm in a new process, even if prior warmup was durable.
            for repeat in range(-1, args.repeats):
                key = (variant, size, repeat)
                if repeat >= 0 and any((row['variant'], row['size'], row['repeat']) == key for row in report['rows']):
                    continue
                require_idle_gpu()
                world = WorldPipeline(**(config | dict(seed=args.seed, dtype='bf16', T=2,
                    latents_batch_size=16, cache_limit=1024**3, torch_compile=False, log_mode='silent')))
                world.coarse_model, world.base_model, world.decoder_model = (
                    loaded.coarse_model, loaded.base_model, loaded.decoder_model)
                configure_world(world, replace(loaded._terrain_profile, cuda_graphs=True), world_profile=args.profile)
                counter = Count(world.base_model)
                world.base_model = counter
                if variant == 'student':
                    install(world, dict(base=model), tile_size=512)
                field = None
                try:
                    world.bind()
                    before = time.perf_counter()
                    # A generous halo covers teacher dependencies and student inputs.
                    world.coarse[:, -8:size//32+8, -8:size//32+8]
                    torch.cuda.synchronize()
                    coarse_seconds = time.perf_counter()-before
                    started = time.perf_counter()
                    field = world.latents[:, 0:size, 0:size]
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-started
                    if not torch.isfinite(field).all():
                        raise FloatingPointError('Non-finite final field.')
                    row = dict(variant=variant, size=size, repeat=repeat, warmup=repeat < 0,
                               seconds=elapsed, coarse_prefetch_seconds=coarse_seconds,
                               base_windows=counter.windows, base_calls=counter.calls,
                               student_counts=getattr(world, '_distill_counts', {}))
                    report['rows'] = [old for old in report['rows'] if (old['variant'], old['size'], old['repeat']) != key]
                    report['rows'].append(row)
                    atomic_json(output, report)
                    print(json.dumps(row), flush=True)
                finally:
                    world.close()
                    del field, world, counter
                    gc.collect()
    report['comparisons'] = summarize(report['rows'], args.sizes)
    report['status'] = 'complete'
    atomic_json(output, report)
    print(json.dumps(report['comparisons']), flush=True)


if __name__ == '__main__':
    main()

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


def require_idle_gpu(allow_other_gpus=False):
    # Check BEFORE CUDA init: on this WSL host nvidia-smi attributes multiple
    # Python workers to one PID. Initializing our own context before inventory
    # can therefore make our probe appear to be an unrelated compute process.
    if torch.cuda.is_initialized():
        raise RuntimeError('GPU inventory must be checked before CUDA initialization.')
    if os.environ.get('CUDA_DEVICE_ORDER') != 'PCI_BUS_ID':
        raise RuntimeError('Use CUDA_DEVICE_ORDER=PCI_BUS_ID for unambiguous device mapping.')
    devices = sorted(tuple(part.strip() for part in line.split(',', 1)) for line in subprocess.check_output(
        ['nvidia-smi', '--query-gpu=pci.bus_id,uuid', '--format=csv,noheader,nounits'], text=True).splitlines())
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    first = visible.split(',')[0].strip() if visible is not None else '0'
    if first.isdigit():
        uuid = devices[int(first)][1].lower().removeprefix('gpu-')
    else:
        matches = [device[1] for device in devices if device[1].lower().startswith(first.lower())]
        if len(matches) != 1:
            raise RuntimeError('Cannot map the first visible CUDA device to its physical GPU UUID.')
        uuid = matches[0].lower().removeprefix('gpu-')
    output = subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True)
    other_pids = []
    other_gpus = []
    for line in output.splitlines():
        gpu, pid = (value.strip() for value in line.split(',', 1))
        if pid != str(os.getpid()):
            if gpu.lower().removeprefix('gpu-') == uuid:
                other_pids.append(pid)
            else:
                other_gpus.append(dict(gpu_uuid=gpu, pid=pid))
    if other_pids:
        raise RuntimeError(f'Selected GPU has other compute processes: {other_pids}. Do not benchmark concurrently.')
    if other_gpus and not allow_other_gpus:
        raise RuntimeError('Other GPUs are computing and share CPU resources. Free them for a final benchmark.')
    return other_gpus


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
    parser.add_argument('--allow-other-gpus', action='store_true',
                        help='Provisional diagnostic only; shared CPU measurements cannot accept a model.')
    args = parser.parse_args()
    if args.repeats < 3 or len(set(args.sizes)) != len(args.sizes) or any(size < 512 or size % 512 for size in args.sizes):
        parser.error('At least three repeats and distinct positive sizes aligned to 512 are required.')
    other_gpus = require_idle_gpu(args.allow_other_gpus)
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
                     whole_system_idle=not other_gpus, other_gpu_processes=other_gpus,
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
                # Inventory was taken before our context existed. WSL PID
                # attribution cannot distinguish our allocation afterwards.
                competing = other_gpus
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
                    row['other_gpu_processes'] = competing
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

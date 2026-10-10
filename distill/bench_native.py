"""Benchmark native base architectures in the real T=2 field path, without counters.

Random-weight candidates establish compute cost only. This report never accepts
their quality. Run after coarse training, with both GPUs otherwise idle.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from distill.bench_pipeline import require_idle_gpu
from distill.common import DATA, REPO, atomic_json, external_path
from distill.native_compression import compressed_config
from distill.native_base import teacher_source


def clear_graphs(model):
    clear = getattr(model, '_clear_buckets', None)
    if clear:
        clear()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DATA/'bench/native-architectures/gpu1.json')
    parser.add_argument('--size', type=int, default=1024)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--profile', default='natural')
    args = parser.parse_args()
    if args.size < 512 or args.size % 512 or args.repeats < 3:
        parser.error('Use a size aligned to512 and at least three repeats.')
    require_idle_gpu()
    torch.set_num_threads(8)
    torch.manual_seed(20261011)
    torch.backends.cudnn.benchmark = True
    output = external_path(args.output)
    from terrain_app import load_pipeline
    from terrain_inference import configure_world, inference_status
    from terrain_diffusion.inference.world_pipeline import WorldPipeline
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    loaded = load_pipeline(101)
    config = {key: value for key, value in dict(loaded.config).items() if not key.startswith('_')}
    teacher_config = json.loads((teacher_source()/'config.json').read_text())
    report = dict(status='running', gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        size=args.size, seed=101, profile=args.profile, repeats=args.repeats, rows=[],
        whole_system_idle=True, base_stages=2, window=64, stride=32,
        includes_cpu_features=True, includes_transfers=True, coarse_prefetched=True,
        excludes_model_loading_and_capture=True, counters=False,
        candidate_weights='random; compute feasibility only, no quality claim',
        source_digests={name: hashlib.sha256((REPO/name).read_bytes()).hexdigest()
                        for name in ('distill/bench_native.py', 'distill/native_compression.py',
                                     'backend/terrain_inference.py', 'backend/terrain_cuda_graphs.py')})
    for variant, width, blocks in (('reference',192,3), ('native192x2',192,2),
                                   ('native128x2',128,2), ('native96x2',96,2),
                                   ('native128x1',128,1), ('native96x1',96,1)):
        if variant == 'reference':
            model = loaded.base_model
        else:
            model = EDMUnet2D.from_config(compressed_config(teacher_config, width, blocks)).to('cuda', torch.bfloat16).eval()
        graph_owner = None
        report.setdefault('parameters', {})[variant] = sum(p.numel() for p in model.parameters())
        for repeat in range(-1, args.repeats):
            world = WorldPipeline(**(config | dict(seed=101, dtype='bf16', T=2,
                latents_batch_size=16, cache_limit=1024**3, torch_compile=False, log_mode='silent')))
            world.coarse_model, world.base_model, world.decoder_model = loaded.coarse_model, graph_owner or model, loaded.decoder_model
            configure_world(world, replace(loaded._terrain_profile, cuda_graphs=True), world_profile=args.profile)
            graph_owner = world.base_model
            field = None
            try:
                world.bind()
                world.coarse[:, -8:args.size//32+8, -8:args.size//32+8]
                torch.cuda.synchronize()
                started = time.perf_counter()
                field = world.latents[:, :args.size, :args.size]
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-started
                status = inference_status(world)
                stats = graph_owner.stats()
                if not stats['buckets'] or stats['capture_errors'] or stats['fallback_calls']:
                    raise RuntimeError(f'Invalid native base graph timing: {stats}')
                if not torch.isfinite(field).all():
                    raise FloatingPointError('Non-finite native benchmark output.')
                if torch.cuda.memory_reserved() > torch.cuda.get_device_properties(0).total_memory:
                    raise RuntimeError('Reject GPU memory paging.')
                row = dict(variant=variant, repeat=repeat, warmup=repeat < 0, seconds=elapsed,
                           cuda_reserved_bytes=torch.cuda.memory_reserved(), base_graphs=stats,
                           profile=status['profile'])
                report['rows'].append(row)
                atomic_json(output, report)
                print(json.dumps(row), flush=True)
            finally:
                world.close()
                del world, field
                gc.collect()
        clear_graphs(graph_owner)
        if variant != 'reference':
            del model, graph_owner
        gc.collect()
        torch.cuda.empty_cache()
    medians = {variant: statistics.median(row['seconds'] for row in report['rows']
        if row['variant'] == variant and not row['warmup']) for variant in report['parameters']}
    report['comparisons'] = [dict(variant=variant, seconds=seconds, speedup=medians['reference']/seconds)
                             for variant, seconds in medians.items()]
    report['status'] = 'complete'
    atomic_json(output, report)
    print(json.dumps(report['comparisons']), flush=True)


if __name__ == '__main__':
    main()

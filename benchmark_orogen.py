"""Benchmark real Orogen CPU/CUDA stages and cache replay, without neural models."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

import terrain_orogen as orogen


def preview(height):
    value = np.clip(height, 0, 6500)/6500
    rgb = np.empty((*height.shape, 3), np.uint8)
    for c, ocean, base, gain in ((0,16,65,190), (1,50,120,130), (2,80,65,190)):
        rgb[..., c] = np.where(height < 0, ocean, base+gain*value)
    return Image.fromarray(rgb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('output/orogen'))
    parser.add_argument('--seed', type=int, action='append')
    parser.add_argument('--style', choices=orogen.STYLES, default='earthlike')
    parser.add_argument('--erosion-engine', choices=('original','orogen','city-gpu'), default='orogen')
    parser.add_argument('--detail', type=int, default=204000)
    parser.add_argument('--city-strength', type=float, default=1.)
    parser.add_argument('--city-iterations', type=int, default=12)
    parser.add_argument('--no-persistence', action='store_true', help='Measure generation/repeat only, without atlas cache writes.')
    parser.add_argument('--gpu-stages', nargs='*',choices=('relief','propagation','post','erosion','climate','raster'),default=[],
                        help='Explicit opt-in GPU stages; historical CPU stages remain the default.')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reports = []
    options=dict(relief_pipeline=args.erosion_engine,detail=args.detail,
                 city_erosion_strength=args.city_strength,city_erosion_iterations=args.city_iterations)
    options.update({'orogen_gpu_'+stage:True for stage in args.gpu_stages})
    suffix='' if args.erosion_engine=='orogen' else '-'+args.erosion_engine
    if args.gpu_stages:suffix+='-gpu-'+ '-'.join(args.gpu_stages)
    device = orogen.implementation_identity()['cuda_device']['index']
    for seed in args.seed or (42, 0, 2**64-1):
        torch.cuda.synchronize(device)
        baseline = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        raw, _, metadata, belt, layers = orogen.generate_atlas(seed, args.style, options=options,include_layers=True)
        peak = torch.cuda.max_memory_allocated(device)-baseline
        repeated, _, warm_metadata, _, repeated_layers = orogen.generate_atlas(seed, args.style, options=options,include_layers=True)
        if not np.array_equal(raw, repeated):
            raise AssertionError('Repeated CUDA generation differs')
        if any(not np.array_equal(value,repeated_layers[name]) for name,value in layers.items()):
            raise AssertionError('Repeated climate/diagnostic fields differ')
        started = time.perf_counter()
        persist_seconds=replay_seconds=None
        if not args.no_persistence:
            persisted = orogen.OrogenHeightmap(seed, args.style, options=options)
            persist_seconds = time.perf_counter()-started
            started = time.perf_counter()
            replay = orogen.OrogenHeightmap(seed, args.style, options=options)
            replay_seconds = time.perf_counter()-started
            np.testing.assert_array_equal(raw, persisted.height_m)
            np.testing.assert_array_equal(raw, replay.height_m)
        metadata['benchmark'] = dict(peak_workspace_bytes=peak,
            persisted_first_seconds=persist_seconds, persisted_replay_seconds=replay_seconds,
            repeat_bit_exact=True,repeat_layers_bit_exact=True, includes_nn=False,
            warm_generation_seconds=warm_metadata['generation_seconds'],warm_timings=warm_metadata['timings'],
            warm_gpu_pipeline=warm_metadata.get('gpu_pipeline'))
        preview(raw).save(args.output/f'relief-{args.style}-{seed}{suffix}.png')
        np.savez_compressed(args.output/f'raw-{args.style}-{seed}{suffix}.npz', height=raw, belt=belt)
        reports.append(metadata)
        print(json.dumps(dict(seed=str(seed), timings=metadata['timings'],
            diagnostics=metadata['diagnostics'], benchmark=metadata['benchmark'])), flush=True)
    (args.output/('benchmark'+suffix+'.json')).write_text(json.dumps(dict(
        identity=orogen.implementation_identity(), cases=reports), indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()

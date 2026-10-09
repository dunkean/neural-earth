"""Offline coarse stream benchmark, with unchanged production sources.

Separately measures prepared-input solver throughput and window postprocessing.
CPU conditioning/noise preparation, persistence and HTTP are excluded. No new
world data is saved to production caches and the server is never imported.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

ROOT = _REPO_ROOT
sys.path.insert(0, str(ROOT / 'terrain-diffusion'))
import torch
from infinite_tensor import InfiniteTensor, MemoryTileStore, TensorWindow
from terrain_app import load_pipeline, RUNTIME, MODEL_SOURCE, MODEL_REVISION
from terrain_inference import _coarse_batch
from terrain_coarse_streams import CoarseSolverInputs, CoarseStreamPool
import terrain_coarse_graph
from terrain_diffusion.inference.world_pipeline import linear_weight_window
from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler


def scheduler():
    return EDMDPMSolverMultistepScheduler(sigma_min=.002, sigma_max=80, sigma_data=.5)


def finish(world, sample):
    """Same scalar expressions as _coarse_batch; gate against its real output."""
    pool = world.kwargs['coarse_pooling']
    value = sample.float() / .5
    stds = torch.tensor(world.kwargs['coarse_stds']).to(value.device)
    means = torch.tensor(world.kwargs['coarse_means']).to(value.device)
    value = value * stds.view(1, -1, 1, 1) + means.view(1, -1, 1, 1)
    value[:, 1] = value[:, 0] - value[:, 1]
    value = value[0]
    if pool > 1:
        value = world._pool_coarse_conditioning(value, pool)
    weight = linear_weight_window(64 // pool, 'cpu', torch.float32).to(value.device)
    return torch.cat([value * weight[None], weight[None]], dim=0)


def compare(actual, expected):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ValueError('Comparison shape/dtype mismatch')
    delta = (actual.double() - expected.double()).abs()
    return dict(byte_exact=bool(torch.equal(actual.contiguous().view(torch.uint8),
                                           expected.contiguous().view(torch.uint8))),
                finite=bool(torch.isfinite(actual).all()), max_abs=float(delta.max().item()))


def fusion(indices, values, device, pool):
    """Exercise the real InfiniteTensor store, with no threaded mutations."""
    size, stride = 64 // pool, 48 // pool
    by_index = dict(zip(indices, values))
    seen = []
    def provide(ctxs):
        seen.extend(ctxs)
        return [by_index[index] for index in ctxs]
    store = MemoryTileStore(cache_size_bytes=256*1024**2)
    tensor = InfiniteTensor((7, None, None), provide,
                            TensorWindow(size=(7, size, size), stride=(7, stride, stride)),
                            batch_size=1, device=device, tile_store=store, tensor_id='stream-experiment')
    rows, cols = [i[1] for i in indices], [i[2] for i in indices]
    y0, y1 = min(rows)*stride + size-stride, (max(rows)+1)*stride
    x0, x1 = min(cols)*stride + size-stride, (max(cols)+1)*stride
    result = tensor[:, y0:y1, x0:x1].clone()
    if set(seen) != set(indices):
        raise RuntimeError('Fusion did not exercise the planned windows')
    return result, seen


def gpu_snapshot():
    result = subprocess.run(['nvidia-smi', '--query-gpu=timestamp,temperature.gpu,pstate,power.draw,clocks.current.sm,utilization.gpu,memory.used',
                             '--format=csv,noheader'], capture_output=True, text=True, timeout=5)
    return result.stdout.strip()


def save(path, report):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def measure(pool, inputs, world, *, postprocess):
    caller = torch.cuda.current_stream(pool.device)
    caller.synchronize()
    torch.cuda.reset_peak_memory_stats(pool.device)
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    outputs, intervals, group_intervals = [], [], []
    started = time.perf_counter()
    begin.record(caller)
    for offset in range(0, len(inputs), len(pool.slots)):
        group_begin, group_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        group_begin.record(caller)
        group, times = pool.run(inputs[offset:offset+len(pool.slots)])
        outputs.extend([finish(world, item) for item in group] if postprocess else group)
        group_end.record(caller)
        group_intervals.append((group_begin, group_end))
        intervals.extend(times)
    end.record(caller)
    end.synchronize()
    seconds = time.perf_counter()-started
    return outputs, dict(wall_seconds=seconds, cuda_interval_seconds=begin.elapsed_time(end)/1000,
                         windows_per_second=len(inputs)/seconds,
                         slot_cuda_seconds=[a.elapsed_time(b)/1000 for a,b in intervals],
                         group_cuda_seconds=[a.elapsed_time(b)/1000 for a,b in group_intervals],
                         peak_allocated_bytes=torch.cuda.max_memory_allocated(pool.device),
                         peak_reserved_bytes=torch.cuda.max_memory_reserved(pool.device))


@torch.inference_mode()
def run(args, report, path):
    world = load_pipeline(args.seed)
    if args.regional_snr:
        # Exercise actual regional conditioning/embedding code on the natural
        # world, without needing a persisted terrestrial bootstrap or changing
        # any production configuration.
        world._terrain_generation_settings = dict(world._terrain_generation_settings,
                snr_altitude_gain=[2.,1.,1.,1.,1.], snr_driver_gain=[.5,1.,1.,1.,1.],
                snr_latitude_gain=2.)
        world._terrain_snr_active = True
    pool_size = world.kwargs['coarse_pooling']
    t_cond = torch.atan(torch.tensor(world.kwargs['cond_snr'])).to(world.device, dtype=world._dtype)
    conditions = [v.detach().view(-1) for v in torch.log(torch.tan(t_cond)/8)]
    weight = linear_weight_window(64//pool_size, 'cpu', torch.float32).to(world.device)
    indices = [(0, i, j) for i in range(-2, 2) for j in range(-4, 4)][:args.windows]
    inputs, reference_samples, reference_windows = [], [], []
    original = terrain_coarse_graph.run_coarse_solver
    def collect(world, solver, sample, image, labels, conditions, embeds):
        packed = CoarseSolverInputs(sample, torch.stack(labels), tuple([image, *conditions]),
                                    torch.stack(embeds) if embeds is not None else None)
        inputs.append(packed)
        value = original(world, solver, sample, image, labels, conditions, embeds)
        reference_samples.append(value)
        return value
    started = time.perf_counter()
    with patch.object(terrain_coarse_graph, 'run_coarse_solver', collect):
        for index in indices:
            reference_windows.extend(_coarse_batch(world, [index], scheduler(), weight, t_cond, conditions, pool_size))
    torch.cuda.current_stream().synchronize()
    if len(inputs) != len(indices):
        raise RuntimeError('Whole coarse solver graphs must be enabled')
    report.update(gpu=torch.cuda.get_device_name(), torch=str(torch.__version__), cuda=torch.version.cuda,
                  world_profile=world._terrain_world_profile, profile=vars(world._terrain_profile),
                  seed=args.seed, indices=indices, coarse_pooling=pool_size,
                  model_source=MODEL_SOURCE, model_revision=MODEL_REVISION,
                  preparation_and_reference_seconds=time.perf_counter()-started,
                  gpu_before=gpu_snapshot(), regional_policies=len(world.__dict__.get('_terrain_snr_cache', {})))
    # Postprocessing is duplicated only in this offline harness and must match
    # the unmodified production path, before measuring any stream variant.
    if not all(compare(finish(world, sample), expected)['byte_exact']
               for sample, expected in zip(reference_samples, reference_windows)):
        raise RuntimeError('Harness postprocessing differs from _coarse_batch')
    fused_reference, fusion_order = fusion(indices, reference_windows, world.device, pool_size)
    pools = {}
    try:
        for count in args.streams:
            print(f'Capturing {count} independent coarse graph(s)...', flush=True)
            before = torch.cuda.memory_allocated()
            pool = CoarseStreamPool(world.coarse_model, scheduler(), world.device, streams=count,
                                    max_bytes=args.graph_mib*1024**2)
            pools[count] = pool
            pool.warm(inputs[0], reference_samples[0])
            report['pools'][str(count)] = dict(stats=pool.stats(),
                                              capture_allocated_bytes=torch.cuda.memory_allocated()-before)
            save(path, report)
        # Prime every exact input/policy on each slot before timed trials.
        for count, pool in pools.items():
            windows, _ = measure(pool, inputs, world, postprocess=True)
            comparisons = [compare(a,b) for a,b in zip(windows, reference_windows)]
            fused, seen = fusion(indices, windows, world.device, pool_size)
            fused_comparison = compare(fused, fused_reference)
            elevation = lambda x: torch.sign(x[0]/x[-1])*(x[0]/x[-1]).square()
            report['validation'][str(count)] = dict(windows=comparisons, fusion=fused_comparison,
                    fusion_order_equal=seen==fusion_order,
                    max_elevation_error_m=compare(elevation(fused), elevation(fused_reference))['max_abs'])
            if not all(c['byte_exact'] and c['finite'] for c in comparisons) or not fused_comparison['byte_exact'] or seen!=fusion_order:
                raise RuntimeError(f'{count} streams failed numerical/fusion fidelity')
        save(path, report)
        if args.profile_replays:
            torch.cuda.profiler.start()
        for repeat in range(args.repeats):
            order = args.streams if repeat%2==0 else list(reversed(args.streams))
            for postprocess in (False, True):
                for count in order:
                    if args.profile_replays:
                        torch.cuda.nvtx.range_push(f'coarse/{"weighted_windows" if postprocess else "solver"}/streams={count}/repeat={repeat}')
                    outputs, measurement = measure(pools[count], inputs, world, postprocess=postprocess)
                    if args.profile_replays:
                        torch.cuda.nvtx.range_pop()
                    expected = reference_windows if postprocess else reference_samples
                    if not all(compare(a,b)['byte_exact'] for a,b in zip(outputs, expected)):
                        raise RuntimeError('Timed replay differs from reference')
                    measurement.update(repeat=repeat, streams=count,
                                       mode='weighted_windows' if postprocess else 'solver', gpu=gpu_snapshot())
                    report['samples'].append(measurement)
                    print(f'{measurement["mode"]}: {count} streams, {measurement["wall_seconds"]:.3f}s', flush=True)
                    save(path, report)
        if args.profile_replays:
            torch.cuda.profiler.stop()
        for mode in ('solver', 'weighted_windows'):
            medians = {count:statistics.median(s['wall_seconds'] for s in report['samples']
                       if s['streams']==count and s['mode']==mode) for count in args.streams}
            report['summary'][mode] = {str(count):dict(median_wall_seconds=value,
                  windows_per_second=len(inputs)/value, speedup_vs_one=medians[1]/value,
                  median_group_cuda_seconds=statistics.median(t for s in report['samples']
                      if s['streams']==count and s['mode']==mode for t in s['group_cuda_seconds']))
                  for count,value in medians.items()}
        report['gpu_after'] = gpu_snapshot()
        report['status'] = 'complete'
    finally:
        for pool in pools.values():
            pool.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=RUNTIME/'coarse-streams'/'report.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--streams', type=int, nargs='+', default=[1,2,4])
    parser.add_argument('--windows', type=int, default=32, choices=[8,16,32])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--graph-mib', type=int, default=2048, help='Total graph budget per pool')
    parser.add_argument('--regional-snr', action='store_true', help='Exercise dynamic SNR/embedding policies on natural conditioning')
    parser.add_argument('--profile-replays', action='store_true', help='CUDA profiler capture range, after all warmup/validation')
    args = parser.parse_args()
    if 1 not in args.streams or len(set(args.streams)) != len(args.streams) or any(n not in (1,2,4,8,16) for n in args.streams) or args.repeats<1 or args.graph_mib<1:
        parser.error('Unique stream counts 1/2/4/8/16 including 1, positive repeats and graph budget required')
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sources = [source_path(name, root=ROOT) for name in ('terrain_coarse_streams.py','tools/benchmarks/benchmark_coarse_streams.py',
        'terrain_inference.py','terrain_coarse_graph.py','terrain_cuda_graphs.py','terrain_nn_constants.py',
        'terrain_cuda_kernels.py','terrain_snr.py')]
    report = dict(status='running', method='Offline prepared-input coarse solver only, BF16 batch1; CPU conditioning/noise, persistence, HTTP and serving excluded. Weighted-window mode includes denormalization/weighting. Graph warmup excluded; pools co-resident; alternating variant order. Desktop and other-process GPU load uncontrolled. No server writes or production cache writes.',
                  settings=vars(args)|{'output':str(args.output)},
                  sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                  pools={}, validation={}, samples=[], summary={})
    save(args.output, report)
    try:
        run(args, report, args.output)
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        save(args.output, report)
        raise
    save(args.output, report)
    print(json.dumps(dict(report=str(args.output), summary=report['summary'])), flush=True)


if __name__ == '__main__':
    main()

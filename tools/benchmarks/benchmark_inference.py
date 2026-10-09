"""Exclusive-GPU benchmark and blocking numerical validation of local patches.

Run only with the terrain server stopped. Large reports/arrays go to E:, not D:.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from terrain_app import RUNTIME, resolve_model_source
from terrain_diffusion.inference.world_pipeline import WorldPipeline
from terrain_diffusion.inference.portable_rng import standard_normal
from terrain_inference import configure_world, choose_profile, prepare_models, restore_models, inference_status


def error(a, b):
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
            'p99_abs': float(np.quantile(delta, .99)), 'exact': bool(np.array_equal(a, b))}


@torch.inference_mode()
def network_checks(world, repeats):
    reports = {}
    for stage, model, channels, size, cond in (
        ('coarse', world.coarse_model, 11, 64, [torch.zeros(1, device=world.device, dtype=world._dtype) for _ in range(5)]),
        ('base', world.base_model, 5, 64, [torch.from_numpy(standard_normal(123, (1, 58))).to(world.device, dtype=world._dtype)]),
        ('decoder', world.decoder_model, 5, 512, []),
    ):
        x = torch.from_numpy(standard_normal(321, (1, channels, size, size))).to(world.device, dtype=world._dtype)
        labels = torch.full((1,), 1.0, device=world.device, dtype=world._dtype)
        restore_models(world)
        reference = model(x, noise_labels=labels, conditional_inputs=cond)
        torch.cuda.synchronize()
        ref_times = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(x, noise_labels=labels, conditional_inputs=cond)
            torch.cuda.synchronize()
            ref_times.append(time.perf_counter() - started)
        prepare_models(world)
        actual = model(x, noise_labels=labels, conditional_inputs=cond)
        torch.cuda.synchronize()
        opt_times = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(x, noise_labels=labels, conditional_inputs=cond)
            torch.cuda.synchronize()
            opt_times.append(time.perf_counter() - started)
        comparison = error(reference.float().cpu().numpy(), actual.float().cpu().numpy())
        assert comparison['exact'], f'{stage}: cached eval weights changed the network'
        reports[stage] = {'reference_seconds': ref_times, 'cached_seconds': opt_times,
                          'speedup_median': statistics.median(ref_times) / statistics.median(opt_times),
                          'error': comparison}
    return reports


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(RUNTIME / 'inference-realtime'))
    parser.add_argument('--network-repeats', type=int, default=8)
    parser.add_argument('--batches', action='store_true')
    parser.add_argument('--decoder-only', action='store_true')
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    assert torch.cuda.is_available()
    # Parameters need version counters so the eval cache can detect changes.
    # Generating tensors stays in inference_mode; weight loading does not.
    with torch.inference_mode(False):
        loaded = WorldPipeline.from_pretrained(resolve_model_source(), torch_compile=False, dtype='bf16', log_mode='silent').to('cuda')
    config = {k: v for k, v in dict(loaded.config).items() if not k.startswith('_')}
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'cuda': torch.version.cuda,
              'seed': 42, 'solver_steps': 20, 'T': 2, 'dtype': 'bfloat16',
              'method': 'Model weights resident, new seed-specific window caches per run, no rendering/disk timings. GPU exclusive. Network test warmed. One crop each; not a p95.',
              'network': network_checks(loaded, args.network_repeats), 'runs': {}}
    (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    profiles = [('reference', None), ('cpu_windows', replace(choose_profile(), gpu_windows=False)),
                ('gpu_windows', choose_profile())]
    if args.batches:
        profiles += [(f'batches_{b}', replace(choose_profile(), coarse_batch=b, decoder_batch=b)) for b in (2, 4)]
    if args.decoder_only:
        profiles = [('reference', None), ('gpu_windows', choose_profile())]
        profiles += [(f'decoder_{b}', replace(choose_profile(), decoder_batch=b)) for b in (2, 4)]
    baseline = None
    for name, profile in profiles:
        restore_models(loaded)
        world = WorldPipeline(**(config | {'seed': 42, 'latents_batch_size': 16, 'dtype': 'bf16',
                                          'cache_limit': 512 * 1024**2, 'torch_compile': False, 'log_mode': 'silent'}))
        world.coarse_model, world.base_model, world.decoder_model = loaded.coarse_model, loaded.base_model, loaded.decoder_model
        if profile is not None:
            configure_world(world, profile, allow_experimental=bool(args.batches or args.decoder_only))
        world.bind()
        torch.cuda.reset_peak_memory_stats()
        run = {'samples': [], 'profile': inference_status(world)}
        arrays = []
        for label, bounds in (
            ('cold', (2560, -3584, 3072, -3072)),
            ('adjacent', (2560, -3072, 2816, -2816)),
            ('cached', (2560, -3584, 3072, -3072)),
            ('negative', (-128, -128, 128, 128)),
        ):
            torch.cuda.synchronize()
            started = time.perf_counter()
            result = world.get(*bounds, with_climate=True)
            elev = result['elev'].float().cpu().numpy()
            climate = result['climate'].float().cpu().numpy()
            seconds = time.perf_counter() - started
            assert np.isfinite(elev).all() and np.isfinite(climate).all()
            arrays.append((elev, climate))
            item = {'label': label, 'bounds': bounds, 'seconds': seconds, 'output_device': str(result['elev'].device)}
            if baseline is not None:
                idx = len(arrays) - 1
                item['elevation_error_m'] = error(baseline[idx][0], elev)
                item['climate_errors'] = [error(baseline[idx][1][c], climate[c]) for c in range(5)]
            run['samples'].append(item)
            print(json.dumps({'run': name, **item}), flush=True)
        run['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        run['profile'] = inference_status(world)
        run['cached_crop_exact'] = bool(np.array_equal(arrays[0][0], arrays[2][0]))
        assert run['cached_crop_exact']
        climate_limits = (0.02, 0.05, 0.1, 0.02, 0.00001)
        run['numerical_validation_passed'] = (baseline is None or (
            all(item['elevation_error_m']['max_abs'] <= 1.0 for item in run['samples']) and
            all(all(err['max_abs'] <= limit for err, limit in zip(item['climate_errors'], climate_limits))
                for item in run['samples'])))
        if baseline is None:
            baseline = arrays
            np.savez_compressed(output / 'reference.npz', **{f'elev_{i}': arr[0] for i, arr in enumerate(arrays)},
                                **{f'climate_{i}': arr[1] for i, arr in enumerate(arrays)})
        # Transparent CPU-window patches must reproduce identical generation.
        if name == 'cpu_windows':
            assert all(item['elevation_error_m']['exact'] for item in run['samples']), 'CPU optimisation changed elevation'
            assert all(all(e['exact'] for e in item['climate_errors']) for item in run['samples'])
        # CUDA assembly changes only FP32 rounding. Reject material terrain or
        # climate changes; the recorded per-channel errors remain reviewable.
        if name == 'gpu_windows':
            assert all(item['elevation_error_m']['max_abs'] <= 1.0 for item in run['samples']), 'GPU elevation diverged over 1m'
            assert all(all(err['max_abs'] <= limit for err, limit in zip(item['climate_errors'], climate_limits))
                       for item in run['samples']), 'GPU climate exceeded limits'
        report['runs'][name] = run
        (output / 'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        world.close()
        del world
        torch.cuda.empty_cache()
    print(json.dumps({'report': str(output / 'report.json'), 'validated_default': True,
                      'rejected_profiles': [name for name, run in report['runs'].items() if not run['numerical_validation_passed']]}), flush=True)


if __name__ == '__main__':
    main()

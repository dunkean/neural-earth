"""GPU gate: real natural checkpoint, scheduler off/on, request permutations."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import json
import hashlib
from pathlib import Path
import time

import numpy as np
import torch

from terrain_app import RUNTIME, load_pipeline
from terrain_window_scheduler import scheduler_status


def _sources():
    root = _REPO_ROOT
    names = ('terrain_inference.py', 'terrain_window_scheduler.py',
             'terrain_coarse.py', 'tools/verification/verify_terrain_windows.py')
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}


def _nodes(tensor):
    for arg in tensor.args:
        yield from _nodes(arg)
    yield tensor


def _read(world, bounds):
    output = world.get(*bounds, with_climate=True)
    return output['elev'].float().cpu().numpy(), output['climate'].float().cpu().numpy()


def _error(expected, actual):
    expected_land = expected[0] >= 0
    actual_land = actual[0] >= 0
    return {'elevation_max_m': float(np.max(np.abs(expected[0] - actual[0]))),
            'elevation_mean_m': float(np.mean(np.abs(expected[0] - actual[0]))),
            'coast_disagreements': int(np.count_nonzero(expected_land != actual_land)),
            'coast_disagreement_fraction': float(np.mean(expected_land != actual_land)),
            'land_fraction_expected': float(np.mean(expected_land)),
            'land_fraction_actual': float(np.mean(actual_land)),
            'climate_max': [float(np.max(np.abs(expected[1][c] - actual[1][c])))
                            for c in range(5)]}


def _without_scheduler(world):
    for tensor in {node for top in (world.coarse, world.latents, world.residual)
                   for node in _nodes(top)}:
        tensor._ensure_processed = tensor._terrain_original_ensure
        tensor._f = tensor._terrain_original_f


def _permuted_patches(world, baseline, order=(3, 0, 2, 1)):
    patches = [(-128, -128, 0, 0), (-128, 0, 0, 128),
               (0, -128, 128, 0), (0, 0, 128, 128)]
    height = np.empty_like(baseline[0])
    climate = np.empty_like(baseline[1])
    for k in order:
        i1, j1, i2, j2 = patches[k]
        patch_height, patch_climate = _read(world, patches[k])
        height[i1+128:i2+128, j1+128:j2+128] = patch_height
        climate[:, i1+128:i2+128, j1+128:j2+128] = patch_climate
    return height, climate


@torch.inference_mode()
def main():
    world = load_pipeline(42)
    bounds = (-128, -128, 128, 128)
    world.rebuild()
    _without_scheduler(world)
    before = time.perf_counter()
    baseline = _read(world, bounds)
    baseline_seconds = time.perf_counter() - before

    world.rebuild()
    _without_scheduler(world)
    off_pieces = _permuted_patches(world, baseline)

    world.rebuild()
    before = time.perf_counter()
    scheduled = _read(world, bounds)
    scheduled_seconds = time.perf_counter() - before

    world.rebuild()
    before = time.perf_counter()
    height, climate = _permuted_patches(world, baseline)
    pieces_seconds = time.perf_counter() - before
    full_after_pieces = _read(world, bounds)
    repeat_full = _read(world, bounds)
    comparisons = {
        'off_vs_on_full': _error(baseline, scheduled),
        'off_full_vs_off_subcrops': _error(baseline, off_pieces),
        'off_subcrops_vs_on_subcrops': _error(off_pieces, (height, climate)),
        'off_vs_subcrops': _error(baseline, (height, climate)),
        'subcrops_vs_full_after': _error((height, climate), full_after_pieces),
        'same_live_cache_whole_replay': _error(full_after_pieces, repeat_full),
    }
    arrays = (baseline, off_pieces, scheduled, (height, climate), full_after_pieces,
              repeat_full)
    finite = all(np.isfinite(channel).all() for output in arrays for channel in output)
    exact = (comparisons['off_vs_on_full']['elevation_max_m'] == 0 and
             all(value == 0 for value in comparisons['off_vs_on_full']['climate_max']) and
             comparisons['off_subcrops_vs_on_subcrops']['elevation_max_m'] == 0 and
             all(value == 0 for value in comparisons['off_subcrops_vs_on_subcrops']['climate_max']) and
             comparisons['off_vs_on_full']['coast_disagreements'] == 0 and
             comparisons['off_subcrops_vs_on_subcrops']['coast_disagreements'] == 0)
    replay = comparisons['same_live_cache_whole_replay']
    replay_exact = (replay['elevation_max_m'] == 0 and replay['coast_disagreements'] == 0
                    and all(value == 0 for value in replay['climate_max']))
    # Request-shape drift belongs to this measured local profile. Keep its
    # tolerance explicit; it is not a universal upstream bound.
    request_shape_within_tolerance = (
        comparisons['off_full_vs_off_subcrops']['elevation_max_m'] <= 1.0 and
        all(value <= .001 for value in comparisons['off_full_vs_off_subcrops']['climate_max']))
    report = {
        'seed': 42, 'bounds': bounds, 'checkpoint': '9ef8030cb805b433b98ec25c5dddefbac07a9e26',
        'dtype': str(world._dtype), 'snr': world.kwargs['cond_snr'],
        'stage_batches': {'coarse': world.coarse.batch_size,
                          'base': world.latents.batch_size,
                          'decoder': world.residual.batch_size},
        **comparisons,
        'seconds': {'off_full': baseline_seconds, 'on_full': scheduled_seconds,
                    'on_subcrops': pieces_seconds},
        'windows': scheduler_status(world),
        'gpu': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'source_sha256': _sources(),
        'thresholds': {'scheduler_extra_height_m': 0, 'scheduler_extra_climate': 0,
                       'local_request_shape_height_m': 1.0,
                       'local_request_shape_climate': .001,
                       'same_live_cache_replay_height_m': 0,
                       'same_live_cache_replay_climate': 0},
        'passed': bool(finite and exact and replay_exact and request_shape_within_tolerance),
    }
    path = RUNTIME / 'window-scheduler-fidelity.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    if not report['passed']:
        raise AssertionError('Neural window scheduler fidelity gate failed')


if __name__ == '__main__':
    main()

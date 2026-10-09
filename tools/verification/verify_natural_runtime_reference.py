"""Serial actual-server replay of the saved pre-bootstrap Natural LOD2 tile."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import argparse
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def _sha(payload):
    return hashlib.sha256(payload).hexdigest()


def _compare(actual, expected):
    actual, expected = np.asarray(actual), np.asarray(expected)
    same_shape = actual.shape == expected.shape
    return {'shape': list(actual.shape), 'dtype': str(actual.dtype),
            'byte_exact': same_shape and actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes(),
            'max_abs_error': float(np.max(np.abs(actual-expected))) if same_shape else None,
            'sha256': _sha(actual.tobytes())}


def _sample_actual_server(server, torch):
    """Count sample forwards after the real server has created its world."""
    started = time.perf_counter()
    world = server.get_world(42, 'natural')
    preparation = world._terrain_coarse_preparation
    before = deepcopy(preparation.status())
    forwards = {'coarse': 0, 'base': 0, 'decoder': 0}
    hooks = []
    for stage, name in [('coarse', 'coarse_model'), ('base', 'base_model'), ('decoder', 'decoder_model')]:
        def count_forward(module, inputs, stage=stage):
            forwards[stage] += 1
        hooks.append(getattr(world, name).register_forward_pre_hook(count_forward))
    try:
        elevation, climate, stage = server.sample_physical(world, 42, 'natural', 2, -14, 10)
        if torch.cuda.is_initialized():
            torch.cuda.synchronize()
    finally:
        for hook in hooks:
            hook.remove()
    after = deepcopy(preparation.status())
    return world, np.asarray(elevation).copy(), np.asarray(climate).copy(), {
        'stage': stage, 'world_object_id': id(world),
        'world_precision': str(world._dtype),
        'model_forward_invocations_during_sample': forwards,
        'model_forward_count_scope': 'top-level model calls during sample only; creation/load warmup excluded; graph wrapper calls include replays',
        'coarse_preparation_before': before, 'coarse_preparation_after': after,
        'coarse_disk_hits_added': after['disk_hits']-before['disk_hits'],
        'coarse_network_windows_added': after['network_windows']-before['network_windows'],
        'seconds_including_world_creation': time.perf_counter()-started,
        'world_manifest': deepcopy(world._terrain_manifest)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', type=Path, default=Path('E:/TerrainDiffusionRuntime/runtime-lod-optimization/baseline.npz'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    start = time.perf_counter()
    # Save exactly the harness used before importing the GPU-owning server.
    source_bytes = Path(__file__).read_bytes()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = args.output.with_name(args.output.stem+'-harness.py')
    snapshot.write_bytes(source_bytes)
    with np.load(args.reference, allow_pickle=False) as reference:
        baseline = {name: reference['lod2_case0_'+name].copy() for name in ('elev', 'climate')}

    import torch
    import terrain_server as server
    from terrain_manifest import world_identity

    first_world = second_world = None
    first_closed = False
    try:
        with server.gpu_lock, torch.inference_mode():
            first_world, first_elev, first_climate, first = _sample_actual_server(server, torch)
            first_checks = {name: _compare(actual, baseline[name]) for name, actual in
                            [('elev', first_elev), ('climate', first_climate)]}
            # Retain the Python object to prove recreation while clearing its
            # tensor/store caches. Neural weights remain shared on the device.
            first_world.empty_cache()
            first_world.close()
            first_closed = True
            server.worlds.pop(('natural', 42), None)
            if server.shared_pipeline is not None:
                server.shared_pipeline.empty_cache()
            gc.collect()
            second_world, second_elev, second_climate, second = _sample_actual_server(server, torch)
            recreated = second_world is not first_world
            second_checks = {name: _compare(actual, baseline[name]) for name, actual in
                             [('elev', second_elev), ('climate', second_climate)]}
            replay_checks = {name: _compare(actual, expected) for name, actual, expected in
                             [('elev', second_elev, first_elev), ('climate', second_climate, first_climate)]}
            manifest = deepcopy(server.world_manifest(42, 'natural'))
            admitted_manifest_hash = world_identity(manifest)

        coarse_reused = (second['coarse_disk_hits_added'] > 0 and
                         second['coarse_network_windows_added'] == 0 and
                         second['model_forward_invocations_during_sample']['coarse'] == 0)
        manifest_stable = first['world_manifest'] == second['world_manifest'] == manifest
        precision_matches_server = (first['world_precision'] == second['world_precision'] == 'torch.bfloat16' and
                                    manifest['generation']['precision'] == 'bf16')
        checks = {'first_matches_saved_baseline': all(row['byte_exact'] for row in first_checks.values()),
                  'recreated_world_matches_saved_baseline': all(row['byte_exact'] for row in second_checks.values()),
                  'recreated_world_matches_first': all(row['byte_exact'] for row in replay_checks.values()),
                  'actual_world_recreated': recreated, 'coarse_windows_reused_without_coarse_forwards': coarse_reused,
                  'admitted_manifest_stable': manifest_stable, 'actual_server_bf16_precision': precision_matches_server,
                  'harness_unchanged_during_run': Path(__file__).read_bytes() == source_bytes}
        report = {'schema': 'actual-server-natural-runtime-reference-v2', 'passed': all(checks.values()),
                  'gpu_used': True, 'world_creation': "terrain_server.get_world(42, 'natural') on both passes",
                  'sampling': "terrain_server.sample_physical(world, 42, 'natural', 2, -14, 10)",
                  'reconstruction_scope': 'new server world, closed first store and cleared first/shared RAM tensor caches; neural weights remain shared; persisted coarse files retained',
                  'reference': str(args.reference.resolve()), 'reference_sha256': _sha(args.reference.read_bytes()),
                  'checks': checks, 'first_baseline_checks': first_checks, 'second_baseline_checks': second_checks,
                  'replay_checks': replay_checks, 'passes': {'first': first, 'recreated': second},
                  'physical_payload_bytes': first_elev.nbytes+first_climate.nbytes,
                  'seconds_including_server_import_and_model_load': time.perf_counter()-start,
                  'world_manifest_hash': admitted_manifest_hash, 'world_manifest': manifest,
                  'manifest_scope': 'actual server admission manifest and BF16 world, replacing the prior generic QA neural_manifest of a directly loaded pipeline',
                  'harness_sha256': _sha(source_bytes), 'harness_snapshot': str(snapshot.resolve()),
                  'cache_claim': 'zero newly executed coarse forwards/windows required on recreation; base/decoder forwards are recorded and may recompute after RAM clearing'}
        args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'passed': report['passed'], 'bytes': report['physical_payload_bytes'],
                          'checks': checks, 'second_forwards': second['model_forward_invocations_during_sample'],
                          'second_coarse_disk_hits': second['coarse_disk_hits_added'],
                          'world_manifest_hash': admitted_manifest_hash}, indent=2))
        raise SystemExit(0 if report['passed'] else 1)
    finally:
        for world in ((None if first_closed else first_world), second_world):
            if world is not None:
                world.empty_cache()
                world.close()
        server.worlds.pop(('natural', 42), None)


if __name__ == '__main__':
    main()

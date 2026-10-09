"""Bounded serial CUDA proof for UI generation settings, not visual-quality QA."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    harness = Path(__file__).read_bytes()
    import torch
    import terrain_server as server
    from terrain_generation import register_generation, resolve_generation
    from terrain_conditioning import make_conditioning_factory

    variants = [
        ('natural-noise', {'cond_snr': [.2, .5, .5, .5, .5]}, 3),
        ('natural-continental', {'height_source': 'natural-continental',
                                'continental_strength': .8}, 2),
    ]
    results = []
    started = time.perf_counter()
    with server.gpu_lock, torch.inference_mode():
        for label, overrides, lod in variants:
            profile = register_generation('natural', overrides)
            settings = resolve_generation(profile).settings
            factory = make_conditioning_factory(42, profile)
            world = server.get_world(42, profile)
            counts = dict(coarse=0, base=0, decoder=0)
            hooks = []
            for stage, name in [('coarse', 'coarse_model'), ('base', 'base_model'), ('decoder', 'decoder_model')]:
                def count(module, inputs, stage=stage):
                    counts[stage] += 1
                hooks.append(getattr(world, name).register_forward_pre_hook(count))
            sample_start = time.perf_counter()
            try:
                elevation, climate, stage = server.sample_physical(world, 42, profile, lod, -14, 10)
                torch.cuda.synchronize()
            finally:
                for hook in hooks:
                    hook.remove()
            row = dict(label=label, profile=profile, settings=settings, lod=lod,
                       stage=stage, model_forwards=counts, sample_seconds=time.perf_counter()-sample_start,
                       world_identity=server.world_identity(world._terrain_manifest),
                       shape=list(elevation.shape), climate_shape=list(climate.shape),
                       elevation_sha256=hashlib.sha256(elevation.tobytes()).hexdigest(),
                       climate_sha256=hashlib.sha256(climate.tobytes()).hexdigest())
            row['checks'] = dict(
                finite=bool(np.isfinite(elevation).all() and np.isfinite(climate).all()),
                admitted_settings=world._terrain_manifest['conditioning']['generation_settings'] == settings,
                actual_snr=world.kwargs['cond_snr'] == settings['cond_snr'],
                actual_frequency=world.kwargs['frequency_mult'] == settings['frequency_mult'],
                actual_factory=world.synthetic_map_factory._terrain_generation_profile == profile,
                preview_factory=factory._terrain_generation_profile == profile,
                neural_executed=sum(counts.values()) > 0,
            )
            results.append(row)
            world.empty_cache()
            world.close()
            server.worlds.pop((profile, 42), None)
    checks = dict(variants_pass=all(all(row['checks'].values()) for row in results),
                  isolated_worlds=len({row['world_identity'] for row in results}) == len(results),
                  harness_unchanged=Path(__file__).read_bytes() == harness)
    report = dict(passed=all(checks.values()), gpu_used=True, checks=checks, variants=results,
                  seconds=time.perf_counter()-started,
                  scope='Two bounded actual-server neural crops; no visual-quality or latency certification.',
                  harness_sha256=hashlib.sha256(harness).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.with_suffix('.harness.py').write_bytes(harness)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['passed'] else 1)


if __name__ == '__main__':
    main()

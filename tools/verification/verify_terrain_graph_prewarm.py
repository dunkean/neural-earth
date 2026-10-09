"""Real-checkpoint gate: startup capture, dynamic replay and reuse across seeds.

Run with the terrain server stopped to keep the GPU available.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from dataclasses import replace
import json
import time

import torch

from terrain_app import load_pipeline
from terrain_inference import configure_world, prewarm_coarse_graphs
from terrain_cuda_graphs import prewarm_base_forms, prewarm_decoder_form
from terrain_diffusion.inference.world_pipeline import WorldPipeline


@torch.inference_mode()
def main():
    started = time.perf_counter()
    shared = load_pipeline(42)
    warmed = dict(coarse=prewarm_coarse_graphs(shared),
                  base=prewarm_base_forms(shared.base_model, shared._terrain_profile.latent_batch),
                  decoder=prewarm_decoder_form(shared.decoder_model, shared.decoder_tile_size))
    assert all(result['fully_warmed'] for result in warmed.values()), warmed
    assert shared.tile_store._bytes == 0, 'Prewarm must not generate terrain'
    owner = shared.coarse_model
    def captures():
        return dict(base=shared.base_model.stats()['capture_calls'],
                    decoder=shared.decoder_model.stats()['capture_calls'],
                    solver=[graph.stats()['capture_calls'] for graph in owner._terrain_solver_graph_cache.values()],
                    streams=[graph.stats()['capture_calls'] for _, graph in owner._terrain_stream_pool[1].slots])
    before = captures()
    original_pool = owner._terrain_stream_pool[1]
    print(json.dumps(dict(stage='warmed', captures=before)), flush=True)
    cases = []
    for seed in (0, 43):
        config = {key: value for key, value in dict(shared.config).items() if not key.startswith('_')}
        config.update(seed=seed, dtype='bf16', latents_batch_size=16,
                      cache_limit=512*1024**2, torch_compile=False)
        world = WorldPipeline(**config)
        world.coarse_model, world.base_model, world.decoder_model = shared.coarse_model, shared.base_model, shared.decoder_model
        configure_world(world, shared._terrain_profile)
        world.bind()
        output = world.get(-128, -128, 128, 128, with_climate=True)
        assert torch.isfinite(output['elev']).all() and torch.isfinite(output['climate']).all()
        expected = {key: value.cpu().clone() for key, value in output.items()}
        after = captures()
        if after != before:
            print(json.dumps(dict(stage='unexpected_capture', before=before, after=after)), flush=True)
        assert after == before, 'A first request captured another graph after prewarm'
        assert owner._terrain_stream_pool[1] is original_pool, 'A first request replaced the prewarmed streams'
        # Fresh stores and eager networks, with the same physical profile/batches.
        configure_world(world, replace(shared._terrain_profile, cuda_graphs=False))
        actual = world.get(-128, -128, 128, 128, with_climate=True)
        assert all(torch.equal(expected[key], actual[key].cpu()) for key in expected), 'Prewarming changed terrain'
        world.empty_cache()
        world.close()
        cases.append(dict(seed=seed, no_new_captures=True, exact_eager_match=True))
    print(json.dumps(dict(passed=True, seconds=round(time.perf_counter()-started, 3),
                         models=warmed, cases=cases), indent=2), flush=True)


if __name__ == '__main__':
    main()

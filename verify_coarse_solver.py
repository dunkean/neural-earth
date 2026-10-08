"""Exclusive GPU gate for whole-solver graphs; no server or concurrent NN work."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from terrain_app import RUNTIME, load_pipeline
from terrain_coarse_graph import clear_coarse_solver
from terrain_inference import configure_world
from terrain_diffusion.inference.world_pipeline import WorldPipeline


@torch.inference_mode()
def main():
    path = RUNTIME / 'coarse-solver-shared-graph-gate.json'
    report = dict(state='running', samples=[], sources={name: hashlib.sha256(
        (Path(__file__).resolve().parent/name).read_bytes()).hexdigest() for name in ('terrain_coarse_graph.py',
        'terrain_inference.py', 'terrain_cuda_graphs.py', 'terrain_nn_constants.py',
        'verify_coarse_solver.py')}, notes=[
        'Server stopped, one terrain NN worker; desktop GPU load uncontrolled.',
        'Direct weighted coarse model windows; not viewport or cold-store benchmark.',
        'Alternate baseline/candidate order; first world candidate includes capture.',
        'Solver pool shared by worlds using one coarse model; seed rebuild keeps capture.'])
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    worlds = []
    try:
        loaded = load_pipeline(42)
        config = {k: v for k, v in dict(loaded.config).items() if not k.startswith('_')}
        for profile in ('natural', 'terrestrial-earthlike'):
            for seed in (42, 0):
                world = WorldPipeline(**(config | dict(seed=seed, torch_compile=False,
                    dtype='bf16', latents_batch_size=16, cache_limit=512*1024**2)))
                worlds.append(world)
                world.coarse_model, world.base_model, world.decoder_model = (
                    loaded.coarse_model, loaded.base_model, loaded.decoder_model)
                base_profile = replace(loaded._terrain_profile, coarse_solver_graphs=False)
                configure_world(world, base_profile, world_profile=profile)
                world.bind()
                for ordinal, index in enumerate(((0, 0, 0), (0, -2, 7), (0, 4, -3), (0, 0, 0))):
                    outputs, times = {}, {}
                    for candidate in ((False, True) if ordinal % 2 == 0 else (True, False)):
                        world._terrain_profile = replace(base_profile, coarse_solver_graphs=candidate)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        output = world.coarse._f([index])[0]
                        torch.cuda.synchronize()
                        times['graph' if candidate else 'baseline'] = time.perf_counter()-start
                        outputs[candidate] = output.cpu().numpy()
                    exact = outputs[False].tobytes() == outputs[True].tobytes()
                    item = dict(profile=profile, seed=seed, index=index,
                        first_use=ordinal == 0, runtime_first_use=len(report['samples']) == 0,
                        seconds=times, byte_exact=exact,
                        graph=world._terrain_coarse_solver_graph.stats())
                    report['samples'].append(item)
                    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
                    if not exact:
                        raise AssertionError(f'Whole coarse solver changed bytes at {profile}/{seed}/{index}')
                clear_coarse_solver(world)
                world.close()
                worlds.remove(world)
        report.update(state='complete', passed=True,
            all_captured=all(item['graph']['capture_calls'] > 0 for item in report['samples']),
            one_shared_capture=all(item['graph']['capture_calls'] == 1 for item in report['samples']))
        warm = [item for item in report['samples'] if not item['first_use']]
        report['warm_medians'] = {name: statistics.median(item['seconds'][name] for item in warm)
                                  for name in ('baseline', 'graph')}
    except BaseException as error:
        report.update(state='failed', passed=False, error=f'{type(error).__name__}: {error}')
        raise
    finally:
        for world in worlds:
            clear_coarse_solver(world)
            world.close()
        path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(dict(path=str(path), state=report['state'],
            passed=report.get('passed'), all_captured=report.get('all_captured'),
            warm_medians=report.get('warm_medians'), error=report.get('error'))), flush=True)


if __name__ == '__main__':
    main()

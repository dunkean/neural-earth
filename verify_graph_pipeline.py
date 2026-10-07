"""Whole-pipeline gate for selective graph replay, including all climate fields."""
import json
import os
import time
import numpy as np
import torch
from terrain_app import load_pipeline, RUNTIME
from terrain_cuda_graphs import CudaGraphModel
from terrain_nn_constants import prepare_constants

os.environ['TERRAIN_CUDA_GRAPHS'] = '0'


@torch.inference_mode()
def main():
    world = load_pipeline(42)
    samples = []
    def crop(seed, bounds):
        world.change_seed(seed)
        world.rebuild()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = world.get(*bounds, with_climate=True)
        arrays = (output['elev'].cpu().numpy(), output['climate'].cpu().numpy())
        return arrays, time.perf_counter() - started
    cases = [(42, (2560, -3584, 3072, -3072)), (43, (2560, -3584, 2816, -3328)),
             (0, (-128, -128, 128, 128)), (42, (-512, 512, -256, 768))]
    reference = [crop(*case) for case in cases]
    prepare_constants(world)
    world.coarse_model = CudaGraphModel(world.coarse_model, max_buckets=1)
    world.base_model = CudaGraphModel(world.base_model, max_buckets=4, max_batch=4)
    for idx, case in enumerate(cases):
        actual, seconds = crop(*case)
        expected, ref_seconds = reference[idx]
        elev_err = float(np.abs(actual[0] - expected[0]).max())
        climate_err = [float(np.abs(actual[1][c] - expected[1][c]).max()) for c in range(5)]
        assert elev_err == 0 and all(err == 0 for err in climate_err), 'Graphs changed generated terrain/climate'
        item = {'seed': case[0], 'bounds': case[1], 'eager_seconds': ref_seconds,
                'graph_seconds': seconds, 'speedup': ref_seconds / seconds,
                'includes_first_graph_capture': idx == 0, 'elevation_max_error_m': elev_err,
                'climate_max_errors': climate_err}
        samples.append(item)
        print(json.dumps(item), flush=True)
    report = {'passed': True, 'samples': samples,
              'coarse_graph': world.coarse_model.stats(), 'base_graph': world.base_model.stats(),
              'peak_allocated_bytes': torch.cuda.max_memory_allocated()}
    path = RUNTIME / 'inference-graph-pipeline.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'report': str(path), 'passed': True}), flush=True)


if __name__ == '__main__':
    main()

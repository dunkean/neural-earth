"""Small exclusive-GPU seed and CUDA-graph gate; does not enable graphs."""
import json
import os
import time
import statistics
import torch
import numpy as np
from terrain_app import load_pipeline, RUNTIME
from terrain_cuda_graphs import CudaGraphModel
from terrain_nn_constants import prepare_constants
from terrain_diffusion.inference.portable_rng import standard_normal

os.environ['TERRAIN_CUDA_GRAPHS'] = '0'


@torch.inference_mode()
def main():
    world = load_pipeline(42)
    assert world.seed == 42
    def crop():
        result = world.get(2560, -3584, 2816, -3328, with_climate=True)
        return result['elev'].cpu().numpy(), result['climate'].cpu().numpy()
    a, climate = crop()
    world.change_seed(43)
    b, _ = crop()
    world.change_seed(42)
    repeated, clim_repeat = crop()
    assert np.array_equal(a, repeated) and np.array_equal(climate, clim_repeat)
    assert not np.array_equal(a, b)
    world.change_seed(0)
    zero, zero_clim = crop()
    world.change_seed(43)
    world.change_seed(0)
    zero_repeat, zero_clim_repeat = crop()
    assert np.array_equal(zero, zero_repeat) and np.array_equal(zero_clim, zero_clim_repeat)
    report = {'seed_42_reset_exact': True, 'seed_0_reset_exact': True,
              'all_five_climate_outputs_finite': bool(np.isfinite(climate).all()), 'graphs': {}}
    for name, model, channels, size, count in (
        ('coarse', world.coarse_model, 11, 64, 1),
        ('base', world.base_model, 5, 64, 1), ('base16', world.base_model, 5, 64, 16),
        ('decoder', world.decoder_model, 5, 512, 1),
    ):
        x = torch.from_numpy(standard_normal(17, (count, channels, size, size))).to(world.device, dtype=world._dtype)
        labels = torch.ones(count, device=world.device, dtype=world._dtype)
        conditions = ([torch.zeros(count, 58, device=world.device, dtype=world._dtype)] if name.startswith('base') else
                      [torch.zeros(count, device=world.device, dtype=world._dtype) for _ in range(5)] if name == 'coarse' else [])
        expected = model(x, noise_labels=labels, conditional_inputs=conditions)
        prepare_constants(world)
        constants_result = model(x, noise_labels=labels, conditional_inputs=conditions)
        assert torch.equal(expected, constants_result), 'Cached constants changed the network'
        timings = []
        for _ in range(6):
            torch.cuda.synchronize()
            started = time.perf_counter()
            model(x, noise_labels=labels, conditional_inputs=conditions)
            torch.cuda.synchronize()
            timings.append(time.perf_counter() - started)
        wrapped = CudaGraphModel(model, max_buckets=2)
        captured_start = time.perf_counter()
        actual = wrapped(x, noise_labels=labels, conditional_inputs=conditions)
        torch.cuda.synchronize()
        capture_seconds = time.perf_counter() - captured_start
        graph_times = []
        for _ in range(6):
            torch.cuda.synchronize()
            started = time.perf_counter()
            wrapped(x, noise_labels=labels, conditional_inputs=conditions)
            torch.cuda.synchronize()
            graph_times.append(time.perf_counter() - started)
        max_error = float((expected - actual).abs().max())
        saved = actual.clone()
        changed = wrapped(x + .1, noise_labels=labels, conditional_inputs=conditions)
        assert torch.equal(saved, actual), 'Graph replay overwrote a caller output'
        assert not torch.equal(actual, changed)
        assert max_error == 0, 'Graph changed the network'
        report['graphs'][name] = {'stats': wrapped.stats(), 'max_abs': max_error,
                                  'capture_seconds': capture_seconds, 'eager_seconds': timings,
                                  'replay_seconds': graph_times,
                                  'speedup_median': statistics.median(timings) / statistics.median(graph_times)}
        report['graphs'][name]['capture_validated'] = wrapped.capture_calls > 0 and max_error == 0
        print(json.dumps({name: report['graphs'][name]}), flush=True)
        del wrapped
    path = RUNTIME / 'inference-runtime-verification.json'
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'seed_validation_passed': True,
                      'all_graphs_captured': all(item['capture_validated'] for item in report['graphs'].values()),
                      'path': str(path)}), flush=True)


if __name__ == '__main__':
    main()

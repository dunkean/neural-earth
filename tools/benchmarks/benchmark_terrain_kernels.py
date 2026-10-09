"""Paired CUDA-graph microbenchmark of real kernels, not a terrain speedup."""

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

import torch
import terrain_cuda_kernels as kernels


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = dict(samples=[], gpu=torch.cuda.get_device_name(), torch=str(torch.__version__),
                  note='Exclusive terrain GPU worker; desktop GPU load uncontrolled. Graph replay timing, not full inference.')
    kernels.prepare('cuda')
    for shape in ((16, 192, 64, 64), (1, 768, 64, 64), (1, 64, 512, 512)):
        a = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
        b = torch.randn_like(a)
        weights = torch.tensor([.3, .7], device='cuda', dtype=a.dtype)
        norm = torch.linalg.vector_norm(weights)
        factors = [weights[0], weights[1]]
        for name, original, candidate in (
            ('sum', lambda: ((a * weights[0] + b * weights[1]) + 0.) / norm,
             lambda: kernels.binary_sum(a, b, weights, norm)),
            ('concat', lambda: torch.cat([a * factors[0], b * factors[1]], 1),
             lambda: kernels.binary_concat(a, b, factors, 1)),
            ('silu', lambda: torch.nn.functional.silu(a) / .596, lambda: kernels.silu_scaled(a))):
            expected, actual = original(), candidate()
            exact = torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
            assert exact, (shape, name)
            graphs, outputs, times = {}, {}, {'reference': [], 'kernel': []}
            for key, forward in (('reference', original), ('kernel', candidate)):
                graphs[key] = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graphs[key]):
                    outputs[key] = forward()
            for repetition in range(4):
                for key in (('reference', 'kernel') if repetition % 2 == 0 else ('kernel', 'reference')):
                    for _ in range(8): graphs[key].replay()
                    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    begin.record()
                    for _ in range(64): graphs[key].replay()
                    end.record(); end.synchronize()
                    times[key].append(begin.elapsed_time(end) / 64)
            medians = {key: statistics.median(values) for key, values in times.items()}
            item = dict(shape=shape, operation=name, byte_exact=bool(exact), milliseconds=medians,
                        repetitions=times, speedup=medians['reference'] / medians['kernel'])
            report['samples'].append(item)
            print(json.dumps(item), flush=True)
            del outputs, graphs, expected, actual
    report['sources'] = {name: hashlib.sha256((source_path(name, root=_REPO_ROOT)).read_bytes()).hexdigest()
                         for name in ('terrain_cuda_kernels.py', 'tools/benchmarks/benchmark_terrain_kernels.py')}
    report['kernel'] = kernels.status()
    report['passed'] = True
    path = Path(args.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

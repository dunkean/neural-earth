"""Base-model throughput per 64x64 latent window: dtype, memory format, batch.

Isolated model forwards with the runtime's cached weights and constant patches;
no terrain windows, scheduling or IO. Desktop GPU load is uncontrolled.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import argparse
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import torch
from torch.utils.flop_counter import FlopCounterMode
from terrain_app import resolve_model_source  # also exposes the terrain-diffusion submodule
import terrain_inference
import terrain_nn_constants
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.models.mp_layers import MPConv, MPConvResample


def inputs(config, n, res, dtype, channels_last):
    x = torch.randn(n, config.in_channels, res, res, device='cuda', dtype=dtype)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    labels = torch.full((n,), 0.7, device='cuda', dtype=dtype)
    kinds = [c[0] for c in config.conditional_inputs]
    conds = [torch.randn(n, c[1], device='cuda', dtype=dtype) if kind == 'tensor'
             else torch.rand(n, device='cuda', dtype=dtype) for kind, c in zip(kinds, config.conditional_inputs)]
    return x, labels, conds


def timeit(fn, iters=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(3):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end) / iters)
    return min(times)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='base_model', choices=['base_model', 'coarse_model'])
    parser.add_argument('--res', type=int, default=64)
    parser.add_argument('--batches', default='16,32')
    parser.add_argument('--dtypes', default='bf16,fp16')
    parser.add_argument('--exact-kernels', type=int, default=1, choices=[0, 1])
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    with torch.inference_mode(False):  # weight version counters feed the eval cache
        model = EDMUnet2D.from_pretrained(str(Path(resolve_model_source()) / args.model)).to('cuda').eval()
    config = model.config
    x, labels, conds = inputs(config, 1, args.res, torch.float32, False)
    with FlopCounterMode(display=False) as counter:
        model(x, noise_labels=labels, conditional_inputs=conds)
    gflop = counter.get_total_flops() / 1e9
    terrain_nn_constants.prepare_constants(
        SimpleNamespace(device='cuda', coarse_model=model, base_model=model, decoder_model=model),
        exact_kernels=bool(args.exact_kernels))
    for module in model.modules():
        if isinstance(module, (MPConv, MPConvResample)):
            module.__dict__['_terrain_original_forward'] = module.forward
            module.forward = MethodType(terrain_inference._mp_forward, module)
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, model=args.model, res=args.res,
                  params_m=round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
                  gflop_per_window=round(gflop, 1), samples=[])
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}), flush=True)
    for name in args.dtypes.split(','):
        dtype = dict(bf16=torch.bfloat16, fp16=torch.float16, fp32=torch.float32)[name]
        model = model.to(dtype)
        for module in model.modules():
            module.__dict__.pop('_terrain_weight_cache', None)
        for channels_last in (False, True):
            model = model.to(memory_format=torch.channels_last if channels_last else torch.contiguous_format)
            for n in map(int, args.batches.split(',')):
                x, labels, conds = inputs(config, n, args.res, dtype, channels_last)
                embeds = model.compute_embeddings(labels, conds)
                forward = lambda: model(x, noise_labels=labels, conditional_inputs=conds, precomputed_embeds=embeds)
                eager = timeit(forward)
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        forward()
                torch.cuda.current_stream().wait_stream(stream)
                with torch.cuda.graph(graph):
                    forward()
                replay = timeit(graph.replay)
                sample = dict(dtype=name, channels_last=channels_last, batch=n,
                              eager_ms_per_window=round(eager / n, 3), graph_ms_per_window=round(replay / n, 3),
                              graph_tflops=round(gflop * n / replay, 1))
                report['samples'].append(sample)
                print(json.dumps(sample), flush=True)
                del graph
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()

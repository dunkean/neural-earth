"""CUDA Graph timing, including actual base halo overhead and FP8 GEMM bounds.

FP8 raw GEMM is reported separately: an FP8 dtype on a Conv2d is not an FP8
implementation. Full-model FP8 is only claimed when a real backend is installed.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import torch

from distill.common import DATA, REPO, atomic_json, external_path
from distill.bench_pipeline import require_idle_gpu
from distill.student import load_student


@torch.inference_mode()
def time_cuda(fn, iterations=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(3):
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end)/iterations)
    return min(times)


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


@torch.inference_mode()
def fp8_bounds():
    if not hasattr(torch, '_scaled_mm'):
        return dict(available=False, reason='torch._scaled_mm unavailable')
    results = []
    for m, n, k in ((4096, 384, 384), (4096, 768, 768), (4096, 1728, 1728)):
        a = torch.randn(m, k, device='cuda', dtype=torch.bfloat16)
        b = torch.randn(n, k, device='cuda', dtype=torch.bfloat16).T
        aa, bb = a.to(torch.float8_e4m3fn), b.to(torch.float8_e4m3fn)
        scale = torch.ones((), device='cuda')
        reference = lambda: torch.mm(a, b)
        candidate = lambda: torch._scaled_mm(aa, bb, scale_a=scale, scale_b=scale,
                                            out_dtype=torch.bfloat16, use_fast_accum=True)
        try:
            bg, fg = capture(reference), capture(candidate)
            bf16, fp8 = time_cuda(bg.replay), time_cuda(fg.replay)
            results.append(dict(m=m, n=n, k=k, bf16_ms=bf16, fp8_ms=fp8, speedup=bf16/fp8,
                                fp8_tflops=2*m*n*k/(fp8*1e9)))
        except (RuntimeError, TypeError) as exc:
            results.append(dict(m=m, n=n, k=k, error=f'{type(exc).__name__}: {exc}'))
    return dict(available=True, kind='raw GEMM; excludes quantization and convolution lowering', samples=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--tile', type=int, default=512)
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fp8-only', action='store_true')
    args = parser.parse_args()
    require_idle_gpu()
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, samples=[], whole_system_idle=True)
    if not args.fp8_only:
        if not args.checkpoint:
            parser.error('--checkpoint is required unless --fp8-only.')
        payload = args.checkpoint.read_bytes()
        with torch.inference_mode(False):
            model, saved = load_student(io.BytesIO(payload), 'cuda')
        stage = model.config.stage
        tile = args.tile if stage == 'base' else (64 if stage == 'coarse' else 512)
        halo = model.halo if stage == 'base' else 0
        total = tile+2*halo
        if tile % model.alignment:
            parser.error('Tile does not satisfy model alignment.')
        report.update(stage=stage, step=saved['step'], tile=tile, halo=halo,
                      params=sum(p.numel() for p in model.parameters()), checkpoint=str(args.checkpoint.resolve()))
        report['checkpoint_digest'] = hashlib.sha256(payload).hexdigest()
        report['student_source_digests'] = {
            'code:'+name: hashlib.sha256((REPO/name).read_bytes()).hexdigest()
            for name in ('distill/student.py', 'distill/features.py', 'distill/inference.py', 'distill/coarse_solver.py')}
        for channels_last in (False, True):
            model.to(memory_format=torch.channels_last if channels_last else torch.contiguous_format)
            inputs = torch.randn(args.batch, model.config.in_channels, total, total, device='cuda')
            if channels_last:
                inputs = inputs.contiguous(memory_format=torch.channels_last)
            def forward():
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    return model(inputs)
            eager = time_cuda(forward)
            # The short coarse solver constructs a CPU scheduler at each call.
            # Its scalar transfers cannot be captured; time the actual eager
            # path without changing model numerics just for this benchmark.
            graph_supported = not (stage == 'coarse' and model.config.solver_steps)
            graph_ms = None
            if graph_supported:
                graph = capture(forward)
                graph_ms = time_cuda(graph.replay)
                del graph
            equivalent = args.batch*tile**2/64**2
            item = dict(dtype='bf16', channels_last=channels_last, batch=args.batch,
                        eager_ms=eager, graph_ms=graph_ms,
                        graph_supported=graph_supported,
                        graph_unavailable_reason=None if graph_supported else
                            'Coarse solver constructs CPU schedule and transfers unpinned scalars each call.',
                        graph_ms_per_64_surface=graph_ms/equivalent if graph_ms is not None else None,
                        includes_halo=True, excludes_feature_construction=True)
            report['samples'].append(item)
            print(json.dumps(item), flush=True)
    report['fp8_raw_gemm'] = fp8_bounds()
    atomic_json(external_path(args.output), report)
    print(json.dumps(report['fp8_raw_gemm']), flush=True)


if __name__ == '__main__':
    main()

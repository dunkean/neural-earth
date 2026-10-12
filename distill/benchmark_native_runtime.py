"""Matched original / baked NF4 inference timings, excluding weight loading.

These are isolated forwards, not full-view or browser timings. NF4 is expanded
to BF16 for this runtime; compressed download size is not resident GPU memory.
"""
import argparse
import gc
from pathlib import Path
from types import SimpleNamespace

import torch

from distill.common import atomic_json, external_path, REVISION
from distill.native_base import teacher_source
from distill.native_inference import load_native
from distill.native_weight_quant import file_sha
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_inference import prepare_models


def measure(fn, iterations):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(5):
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)
    return dict(median_ms=sorted(samples)[2], samples_ms=samples)


@torch.no_grad()
def run(args):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Keep prior measurements; choose another output.')
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
        teacher_revision=REVISION, checkpoint=str(args.checkpoint),
        checkpoint_sha256=file_sha(args.checkpoint), scope='isolated 64x64 forwards',
        note=args.note, dtype='BF16', channels_last=False,
        embeddings='precomputed identically for both models',
        compressed_weights_expanded_to_bf16=True, iterations=args.iterations,
        exact_cuda_kernels=False, attention_backend='reference',
        samples=[], completed=False)
    for variant in ('original', 'distilled'):
        gc.collect()
        torch.cuda.empty_cache()
        initial = torch.cuda.memory_allocated()
        if variant == 'original':
            model = EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
            prepare_models(SimpleNamespace(coarse_model=model, base_model=model, decoder_model=model))
            source = teacher_source() / 'diffusion_pytorch_model.safetensors'
            report['original_file_bytes'] = source.stat().st_size
        else:
            model, saved = load_native(args.checkpoint, 'cuda')
            report['distilled_step'] = saved['step']
            del saved
        # Tensor constants must be materialized on CUDA before graph capture.
        # Preserve the baked NF4 convolution forwards in the distilled model.
        from terrain_nn_constants import prepare_constants
        prepare_constants(SimpleNamespace(device='cuda', coarse_model=model,
            base_model=model, decoder_model=model), exact_kernels=False)
        report[variant + '_params'] = sum(p.numel() for p in model.parameters())
        for batch in args.batches:
            # Regenerate identical inputs; randomness is outside timed regions.
            torch.manual_seed(20261012)
            x = torch.randn(batch, 5, 64, 64, device='cuda', dtype=torch.bfloat16)
            condition = [torch.randn(batch, 58, device='cuda', dtype=torch.bfloat16)]
            for phase, level in enumerate((1.5625, .609375)):
                labels = torch.full((batch,), level, device='cuda', dtype=torch.bfloat16)
                embeddings = model.compute_embeddings(labels, condition)
                def forward():
                    return model(x, noise_labels=labels, conditional_inputs=condition,
                                 precomputed_embeds=embeddings)
                torch.cuda.reset_peak_memory_stats()
                eager = measure(forward, args.iterations)
                graph = torch.cuda.CUDAGraph()
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        forward()
                torch.cuda.current_stream().wait_stream(stream)
                with torch.cuda.graph(graph):
                    prediction = forward()
                replay = measure(graph.replay, args.iterations)
                row = dict(variant=variant, batch=batch, phase=phase,
                    eager=eager, cuda_graph=replay,
                    eager_ms_per_window=eager['median_ms']/batch,
                    graph_ms_per_window=replay['median_ms']/batch,
                    graph_windows_per_second=1000*batch/replay['median_ms'],
                    peak_allocated_bytes=torch.cuda.max_memory_allocated()-initial,
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                    finite_output=bool(torch.isfinite(prediction).all()))
                if not row['finite_output']:
                    raise ValueError('Nonfinite inference output.')
                report['samples'].append(row)
                atomic_json(output, report)
                print(f"{variant} batch={batch} phase={phase}: eager {row['eager_ms_per_window']:.3f}, graph {row['graph_ms_per_window']:.3f} ms/window", flush=True)
                del graph, prediction, forward, embeddings, labels
            del x, condition
        del model
        gc.collect()
        torch.cuda.empty_cache()
    for row in report['samples']:
        if row['variant'] == 'distilled':
            reference = next(r for r in report['samples'] if r['variant']=='original'
                and r['batch']==row['batch'] and r['phase']==row['phase'])
            row['graph_speedup'] = reference['graph_ms_per_window']/row['graph_ms_per_window']
            row['eager_speedup'] = reference['eager_ms_per_window']/row['eager_ms_per_window']
    report['completed'] = True
    atomic_json(output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batches', type=lambda s: [int(n) for n in s.split(',')], default=[1,16])
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--note', required=True, help='Record concurrent jobs / display load.')
    args = parser.parse_args()
    if args.iterations < 1 or not args.batches or min(args.batches) < 1:
        raise ValueError('Positive iterations and batches required.')
    run(args)


if __name__ == '__main__':
    main()

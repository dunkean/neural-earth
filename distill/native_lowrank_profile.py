"""Profile actual cached BF16 layers before selecting teacher compression levels."""
import argparse
import gc
import json
from types import SimpleNamespace

import numpy as np
import torch

from distill.common import DATA, REPO, atomic_json, external_path
from distill.native_base import NativeBase, teacher_source
from distill.native_lowrank import NativeLowrankBase, eligible
from distill.native_lowrank_inference import freeze
from distill.wet_mountains import digest
from terrain_diffusion.models.edm_unet import EDMUnet2D


@torch.no_grad()
def timings(net, inputs, labels, cond, iterations=30):
    def forward():
        return net(inputs, noise_labels=labels, conditional_inputs=[cond])
    for _ in range(5):
        output = forward()
    torch.cuda.synchronize()
    if not torch.isfinite(output).all():
        raise FloatingPointError('Nonfinite cached inference.')
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): forward()
    torch.cuda.current_stream().wait_stream(stream)
    with torch.cuda.graph(graph, stream=stream): output = forward()
    samples = []
    for _ in range(3):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations): graph.replay()
        end.record(); torch.cuda.synchronize()
        samples.append(start.elapsed_time(end)/iterations/len(inputs))
    return dict(ms_per_window=float(np.median(samples)), repeats=samples,
                batch=len(inputs), execution='cached-BF16-CUDA-Graphs')


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fractions', type=float, nargs='+', default=[1., 5/6, .5])
    args = parser.parse_args()
    torch.set_num_threads(8); torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    source = EDMUnet2D.from_pretrained(str(teacher_source())).cuda().eval()
    original = NativeBase(source.config).cuda().eval().requires_grad_(False)
    original.net.load_state_dict(source.state_dict())
    del source
    root = DATA/'crops/native-mixed-scenes-v2'
    row = next(r for r in json.loads((root/'manifest.json').read_text())['scenes']
               if r['metadata']['split']=='train' and r['coarse_variant']=='coarse4')
    with np.load(root/row['file']) as sample:
        inputs, labels, cond = (torch.from_numpy(sample[k][:16].copy()).cuda().bfloat16()
                               for k in ('inputs', 'labels', 'condition'))
    reference = original(inputs, labels, [cond])
    net = freeze(original)
    actual = net(inputs, noise_labels=labels, conditional_inputs=[cond]).float()
    error = actual-reference
    equiv = dict(relative_mse=float(error.square().mean()/reference.square().mean()),
                 max_abs=float(error.abs().max()))
    if equiv['max_abs'] != 0.:
        raise ValueError(f'Cached teacher copy differs from the numerical wrapper: {equiv}')
    from terrain_nn_constants import prepare_constants
    prepare_constants(SimpleNamespace(device='cuda', coarse_model=net, base_model=net, decoder_model=net), exact_kernels=False)
    teacher_time = timings(net, inputs, labels, cond)
    # Scope event pairs only around warmed MP convolutions; this eager layer
    # timing chooses levels, while full CUDA Graph timing validates the gain.
    events, handles = {}, []
    for name, module in net.named_modules():
        if not eligible(name, module): continue
        events[name] = []
        def before(module, values, name=name):
            start = torch.cuda.Event(enable_timing=True); start.record()
            events[name].append([start, None])
        def after(module, values, output, name=name):
            end = torch.cuda.Event(enable_timing=True); end.record()
            events[name][-1][1] = end
        handles += [module.register_forward_pre_hook(before), module.register_forward_hook(after)]
    for _ in range(10): net(inputs, noise_labels=labels, conditional_inputs=[cond])
    torch.cuda.synchronize()
    for handle in handles: handle.remove()
    layers = {name: float(np.median([a.elapsed_time(b) for a, b in pairs])/len(inputs))
              for name, pairs in events.items()}
    levels = {}
    for name, elapsed in layers.items():
        side, block, _ = name.split('.', 2)
        key = side+'.'+block.split('_')[0]
        levels[key] = levels.get(key, 0.)+elapsed
    selected, covered = [], 0.
    for key, elapsed in sorted(levels.items(), key=lambda r:-r[1]):
        selected.append(key); covered += elapsed
        if covered >= .7*sum(levels.values()): break
    selection = [name for name in layers if any(name.startswith(k+'_') for k in selected)]
    saved = torch.load(DATA/'eval/native-lowrank-fp64-basis.pt', map_location='cpu', weights_only=True)
    results = []
    for fraction in args.fractions:
        if not 0 < fraction <= 1:
            raise ValueError('Rank fractions must be in (0,1].')
        ranks = {name: min(saved['factor_ranks'][name], max(16, int(np.ceil(saved['factor_ranks'][name]*fraction/16))*16))
                 for name in selection}
        model = NativeLowrankBase(saved['config'], ranks).cuda().eval().requires_grad_(False)
        state = {f'net.{name}': value for name, value in original.net.state_dict().items()
                 if name.rsplit('.',1)[0] not in ranks}
        for name, rank in ranks.items():
            state[f'net.{name}.spatial'] = saved['ema'][f'net.{name}.spatial'][:rank]
            state[f'net.{name}.pointwise'] = saved['ema'][f'net.{name}.pointwise'][:, :rank]
        model.load_state_dict(state)
        expected = model(inputs, labels, [cond])
        cached = freeze(model, constants=True)
        result = cached(inputs, noise_labels=labels, conditional_inputs=[cond]).float()
        mse = float((result-expected).square().mean()/expected.square().mean())
        if mse > 1e-5:
            raise ValueError(f'Cached factor copy differs beyond the runtime constant floor: {mse}')
        result = dict(rank_fraction=fraction, parameters=sum(p.numel() for p in model.parameters()),
                      wrapper_equivalence_mse=mse, **timings(cached, inputs, labels, cond))
        result['speedup_vs_teacher'] = teacher_time['ms_per_window']/result['ms_per_window']
        results.append(result); print(json.dumps(result), flush=True)
        del model, cached; gc.collect(); torch.cuda.empty_cache()
    atomic_json(external_path(args.output), dict(status='complete', gpu=torch.cuda.get_device_name(),
        teacher_time=teacher_time, teacher_cache_equivalence=equiv, selected_levels=selected,
        selected_layers=selection, selected_event_time_fraction=covered/sum(levels.values()),
        layer_ms=layers, level_ms=levels, results=results,
        sources_sha256={name:digest(REPO/'distill'/name) for name in
                       ('native_lowrank_profile.py','native_lowrank_inference.py','native_lowrank.py')},
        note='Layer events select levels; cached CUDA Graph totals measure actual gains. No training or terrain acceptance.', accepted=False))


if __name__ == '__main__': main()

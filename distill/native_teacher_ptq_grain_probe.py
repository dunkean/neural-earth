"""TRAIN-only same-input teacher PTQ probe for NF4 block size and global RMS."""
import argparse
import json
import math
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F

from distill.common import HOLDOUT_SEEDS, REPO, atomic_json, external_path
from distill.native_base import teacher_source
from distill.native_decoded import DecodedRegion, decoded_region_forward
from distill.native_grain_budget_cpu import bands, projection
from distill.native_weight_quant import encode, file_sha
from terrain_diffusion.models.mp_layers import MPConv, normalize


class TeacherBF16(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, noise_labels, conditional_inputs):
        return self.net(inputs.bfloat16(), noise_labels=noise_labels.bfloat16(),
                        conditional_inputs=[value.bfloat16() for value in conditional_inputs]).float()


def ptq_forward(layer, inputs, gain=1):
    weight = (layer._ptq_weight * gain).to(inputs.dtype)
    return F.conv2d(inputs, weight,
                    padding=0 if layer.no_padding else weight.shape[-1] // 2,
                    groups=layer.groups)


def internal_convolutions(model):
    first = 'enc.' + next(iter(model.enc))
    return [(name, module) for name, module in model.named_modules()
            if isinstance(module, MPConv) and module.weight.ndim == 4
            and name not in (first, 'out_conv')]


def install_effective(layers, block=None, preserve_global_norm=False):
    for _, layer in layers:
        raw = layer.weight.bfloat16().float()
        effective = normalize(raw) / math.sqrt(raw[0].numel())
        if block is None:
            selected = effective
        else:
            selected = encode(effective, 'nf4', block)[2]
            if preserve_global_norm:
                selected = selected * (torch.linalg.vector_norm(effective) /
                                       torch.linalg.vector_norm(selected))
        layer._ptq_weight = selected
        if '_ptq_original_forward' not in layer.__dict__:
            layer._ptq_original_forward = layer.forward
            layer.forward = MethodType(ptq_forward, layer)


def restore(layers):
    for _, layer in layers:
        if '_ptq_original_forward' in layer.__dict__:
            layer.forward = layer.__dict__.pop('_ptq_original_forward')
        layer.__dict__.pop('_ptq_weight', None)


def metrics(core, target):
    q = (core[4].double() * 38.6 - 31.4).cpu().numpy()
    qt = (target[4].double() * 38.6 - 31.4).cpu().numpy()
    result = dict(root_bias=float((q-qt)[16:48, 16:48].mean()),
                  root_mae=float(np.abs(q-qt)[16:48, 16:48].mean()))
    for size in (32, 48):
        margin = (64-size)//2
        crop = lambda v: v[margin:64-margin, margin:64-margin]
        for i, (actual, reference) in enumerate(zip(bands(q), bands(qt), strict=True), 1):
            result[f'band{i}_size{size}'] = projection(crop(actual), crop(reference))
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--diagnostic', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve the previous PTQ probe.')
    report = json.loads((args.diagnostic/'report.json').read_text())
    if report['status'] != 'complete':
        raise ValueError('A complete frozen TRAIN panel is required.')
    selected = []
    for kind in dict.fromkeys(row['kind'] for row in report['rows']):
        row = next(row for row in report['rows'] if row['kind'] == kind and row['coarse'] == 'coarse4')
        selected.append((report['contract']['scenes'][row['index']], row))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    net = EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    model = TeacherBF16(net)
    layers = internal_convolutions(net)
    if not layers:
        raise ValueError('Teacher has no eligible internal MP convolutions.')
    parameter_count = sum(layer.weight.numel() for _, layer in layers)
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64, 'cuda', torch.float32)
    variants = [('effective_no_quant', None, False), ('nf4_block512', 512, False),
                ('nf4_block128', 128, False), ('nf4_block64', 64, False),
                ('nf4_block512_globalnorm', 512, True),
                ('nf4_block64_globalnorm', 64, True)]
    rows = []
    try:
        for record, row in selected:
            path = Path(record['root'])/record['file']
            if (record['metadata']['split'] != 'train' or
                    record['metadata']['seed'] in HOLDOUT_SEEDS or
                    file_sha(path) != record['capture_sha256']):
                raise ValueError('Independent immutable TRAIN capture required.')
            restore(layers)
            region = DecodedRegion(path, 'cuda')
            region.refresh_noise(model, weight, row['noise_seed'])
            target = region.target.detach()
            base = decoded_region_forward(model, region, weight)['core']
            repeat_error = float((base-target).abs().max())
            if repeat_error > 1e-5:
                raise ValueError(f'Exact teacher repeat failed: {repeat_error}')
            comparisons = {}
            for name, block, globalnorm in variants:
                install_effective(layers, block, globalnorm)
                candidate = decoded_region_forward(model, region, weight)['core']
                comparisons[name] = metrics(candidate, target)
                if block is None and not torch.equal(candidate, base):
                    raise ValueError('Unquantized effective-kernel patch changed the teacher.')
                restore(layers)
            rows.append(dict(kind=row['kind'], seed=row['seed'], noise_seed=row['noise_seed'],
                             split='train', source_capture_sha256=record['capture_sha256'],
                             repeat_error=repeat_error, variants=comparisons))
            print(json.dumps(dict(kind=row['kind'], seed=row['seed'],
                                  band1_orthogonal={name: value['band1_size48']['orthogonal_energy_ratio']
                                                    for name, value in comparisons.items()})), flush=True)
    finally:
        restore(layers)
    atomic_json(output, dict(status='complete', rows=rows, updates=0,
        gpu=torch.cuda.get_device_name(), eligible_layers=len(layers),
        eligible_parameters=parameter_count, blocks=[512, 128, 64],
        global_norm='Per eligible convolution tensor; same global RMS, not per-filter normalization.',
        diagnostic_sha256=file_sha(args.diagnostic/'report.json'),
        teacher_weights_sha256=file_sha(teacher_source()/'diffusion_pytorch_model.safetensors'),
        source_code_sha256=file_sha(REPO/'distill/native_teacher_ptq_grain_probe.py'),
        scope='Five immutable TRAIN captures, fresh same-noise teacher chain; NF4 PTQ of internal 4D MPConv only. No fitting, no serving change.'))


if __name__ == '__main__':
    main()

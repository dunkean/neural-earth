"""TRAIN-only selective INT8 inference probe; leaves the paired training untouched."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from distill.common import atomic_json, external_path, REPO
from distill.native_basis_training import inventory
from distill.native_decoded import DecodedRegion, OriginalDecoder, decoded_region_forward
from distill.decoded_loss import reconstructed_height
from distill.native_field_auxiliary import low_height
from distill.native_scalar_precision import runtime_scalar_model
from distill.native_weight_quant import encode, file_sha
from distill.terrain_shape import compare_shape
from distill.wet_mountains import KIND
from terrain_diffusion.models.mp_layers import normalize


class FrozenForward(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, inputs, noise_labels, conditional_inputs):
        return self.net(inputs.bfloat16(), noise_labels=noise_labels.bfloat16(),
                        conditional_inputs=[v.bfloat16() for v in conditional_inputs]).float()


def field_metrics(core, target, high, target_high):
    actual_low = low_height(core[None])[0, 0, 16:48, 16:48]
    reference_low = low_height(target[None])[0, 0, 16:48, 16:48]
    result = {}
    for lod, actual, reference in [(3, actual_low, reference_low), (0, high, target_high)]:
        delta = actual-reference
        result[str(lod)] = dict(bias_m=float(delta.mean()), mae_m=float(delta.abs().mean()),
            centered_mae_m=float((delta-delta.mean()).abs().mean()),
            coast_disagreement=float(((actual > 0) != (reference > 0)).float().mean()))
    q, qt = [v[4].float().cpu().numpy()*38.6-31.4 for v in (core, target)]
    bands = [(v-gaussian_filter(v, 1., mode='reflect'))[16:48, 16:48] for v in (q, qt)]
    result['root_highpass_rms_ratio'] = float(np.sqrt(np.mean(bands[0]**2)/np.mean(bands[1]**2)))
    result['root_highpass_correlation'] = float(np.corrcoef(bands[0].ravel(), bands[1].ravel())[0, 1])
    shape = compare_shape(high.cpu().numpy(), target_high.cpu().numpy())
    result['decoded_morphology'] = shape
    return result


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve previous probes.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = torch.load(args.source, map_location='cpu', weights_only=True, mmap=True)
    anchor = float(source['ema']['net.out_gain'].bfloat16().float())
    saved = source | dict(architecture='native-weight-quant-scalar',
                         scalar_precision='anchored-fp32-v1', gain_anchor=anchor)
    model = FrozenForward(runtime_scalar_model(saved, 'cuda'))
    name = 'dec.512x512_block3.conv_res1'
    if name not in source['quantized_layers']:
        raise ValueError('The selected convolution must be NF4 in the source.')
    layer = model.net.get_submodule(name)
    baseline_weight = layer.weight.clone()
    raw = source['ema']['net.'+name+'.weight'].cuda().bfloat16().float()
    effective = normalize(raw)/math.sqrt(raw[0].numel())
    int8_weight = encode(effective, 'int8', source['block'])[2].bfloat16()
    extra_bytes = math.ceil(raw.numel()/source['block'])*source['block']//2
    records, _, _ = inventory()
    selected = []
    for kind in ('desert-plain', 'temperate-plain', 'wet-complex-coast', 'low-plain-sea', KIND):
        pool = [r for r in records if r['case_kind'] == kind and r['coarse_variant'] == 'coarse4']
        selected += sorted(pool, key=lambda r: (r['metadata']['seed'], r['file']))[:3 if kind == KIND else 1]
    decoder = OriginalDecoder('cuda')
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64, 'cuda', torch.float32)
    rows = []
    for record in selected:
        if record['metadata']['split'] != 'train':
            raise ValueError('TRAIN only.')
        region = DecodedRegion(Path(record['root'])/record['file'], 'cuda')
        target_high = reconstructed_height(decoder(region.target[None], region.decoder_noise),
                                          region.target[None])[0, 0, 64:-64, 64:-64]
        variants = {}
        for label, kernel in [('nf4', baseline_weight), ('last_int8', int8_weight)]:
            layer.weight.copy_(kernel)
            core = decoded_region_forward(model, region, weight)['core']
            high = reconstructed_height(decoder(core[None], region.decoder_noise), core[None])[0, 0, 64:-64, 64:-64]
            variants[label] = field_metrics(core, region.target, high, target_high)
        rows.append(dict(kind=record['case_kind'], seed=record['metadata']['seed'], split='train',
                         root=record['root'], file=record['file'], variants=variants))
        print(json.dumps(dict(kind=record['case_kind'], seed=record['metadata']['seed'],
            highpass={k:v['root_highpass_rms_ratio'] for k,v in variants.items()},
            bias_lod0={k:v['0']['bias_m'] for k,v in variants.items()})), flush=True)
    layer.weight.copy_(baseline_weight)
    atomic_json(output, dict(status='complete', rows=rows, source_sha256=file_sha(args.source),
        selected_layer=name, extra_payload_bytes=extra_bytes, weight_updates=0,
        gpu=torch.cuda.get_device_name(), source_code_sha256=file_sha(REPO/'distill/native_last_layer_probe.py'),
        scope='Selective inference-only INT8 on seven TRAIN captures. No fitting, export, served model or A/B training changes.'))


if __name__ == '__main__':
    main()

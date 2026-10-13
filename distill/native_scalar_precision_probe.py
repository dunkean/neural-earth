"""TRAIN-only exact-anchor identity and sub-BF16-step physical response probe."""
import argparse
import json
from pathlib import Path

import torch

from distill.common import REPO, atomic_json, external_path
from distill.native_basis_training import inventory
from distill.native_decoded import DecodedRegion, OriginalDecoder, decoded_region_forward
from distill.decoded_loss import reconstructed_height
from distill.native_field_auxiliary import low_height
from distill.native_scalar_precision import load_scalar_precision
from distill.native_weight_quant import file_sha
from distill.wet_mountains import KIND


def metrics(actual, target):
    delta = actual.float() - target.float()
    return dict(bias_m=float(delta.mean()), mae_m=float(delta.abs().mean()),
                centered_mae_m=float((delta-delta.mean()).abs().mean()))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve previous precision evidence.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = torch.load(args.source, map_location='cpu', weights_only=True, mmap=True)
    anchor = float(source['ema']['net.out_gain'].bfloat16().float())
    model = load_scalar_precision(source, 'cuda')
    records, _, _ = inventory()
    selected = []
    for kind in ('desert-plain', 'temperate-plain', 'wet-complex-coast', 'low-plain-sea', KIND):
        pool = [r for r in records if r['case_kind'] == kind and r['coarse_variant'] == 'coarse4']
        selected += sorted(pool, key=lambda r: (r['metadata']['seed'], r['file']))[:3 if kind == KIND else 1]
    decoder = OriginalDecoder('cuda')
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64, 'cuda', torch.float32)
    # 1/16 and 1/4 of the local BF16 spacing at this checkpoint's gain.
    spacing = float(torch.nextafter(torch.tensor(anchor).bfloat16(),
                    torch.tensor(float('inf')).bfloat16()).float()) - anchor
    rows = []
    variants = [('bf16', 0.), ('fp32', 0.), ('anchored', 0.)] + [
        ('anchored', spacing*f) for f in (-.25, -.0625, .0625, .25)]
    for record in selected:
        if record['metadata']['split'] != 'train':
            raise ValueError('TRAIN only.')
        region = DecodedRegion(Path(record['root'])/record['file'], 'cuda')
        target_low = low_height(region.target[None])[0, 0, 16:48, 16:48]
        target_high = reconstructed_height(decoder(region.target[None], region.decoder_noise),
                                          region.target[None])[0, 0, 64:-64, 64:-64]
        if target_high.shape != (256, 256):
            raise ValueError('Matched footprint changed.')
        results = []
        baseline = None
        for mode, increment in variants:
            model.set_precision(mode, anchor if mode == 'anchored' else None)
            model.net.out_gain.fill_(anchor + increment)
            with model.cached_weights():
                result = decoded_region_forward(model, region, weight)
            low = low_height(result['core'][None])[0, 0, 16:48, 16:48]
            high = reconstructed_height(decoder(result['core'][None], region.decoder_noise),
                                        result['core'][None])[0, 0, 64:-64, 64:-64]
            if baseline is None:
                baseline = dict(result=result, low=low, high=high)
            differences = {k: float((result[k]-baseline['result'][k]).abs().max())
                           for k in ('velocity0', 'inputs1', 'velocity1', 'core')}
            differences.update(low_height=float((low-baseline['low']).abs().max()),
                               decoded_height=float((high-baseline['high']).abs().max()))
            if mode == 'anchored' and increment == 0 and any(differences.values()):
                raise ValueError(f'Anchored initialization must be exact: {differences}')
            results.append(dict(mode=mode, increment=increment, gain=anchor+increment,
                metrics={'3': metrics(low, target_low), '0': metrics(high, target_high)},
                max_difference_from_baseline=differences))
        row = dict(kind=record['case_kind'], seed=record['metadata']['seed'], split='train',
                   root=record['root'], file=record['file'], variants=results)
        rows.append(row)
        print(json.dumps(row), flush=True)
    model.set_precision('bf16')
    model.net.out_gain.fill_(anchor)
    atomic_json(output, dict(status='complete', rows=rows, anchor=anchor, bf16_spacing=spacing,
        source_sha256=file_sha(args.source), weight_updates=0, gpu=torch.cuda.get_device_name(),
        code_sha256={name: file_sha(REPO/'distill'/name) for name in
                     ('native_scalar_precision.py', 'native_scalar_precision_probe.py')},
        footprint='Central32x32 latent /256x256 decoded pixels; identical physical footprint.',
        scope='TRAIN-only inference probe. No calibration, optimizer, export or served weight changes.'))


if __name__ == '__main__':
    main()

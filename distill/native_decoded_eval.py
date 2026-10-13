"""Measure native checkpoints on independently qualified rare validation cores.

Numeric internal evidence only: the original decoder is frozen, the physical
proxy is the training one, and full held-out pipeline/seam checks still follow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA, REPO, HOLDOUT_SEEDS, atomic_json, external_path
from distill.native_base import NativeBase, teacher_source
from distill.native_decoded import OriginalDecoder, DecodedRegion, decoded_region_forward
from distill.native_decoded_train import validate
from distill.decoded_loss import reconstructed_height
from distill.terrain_shape import compare_shape


@torch.no_grad()
def shape_validation(model, files, weight, decoder):
    rows = []
    for path in files:
        region = DecodedRegion(path, 'cuda')
        result = decoded_region_forward(model, region, weight)
        heights = []
        for latent in (result['core'], region.target):
            decoded = reconstructed_height(decoder(latent[None], region.decoder_noise), latent[None])[0, 0]
            low = latent[4]*38.6-31.4
            heights.append((decoded.cpu().numpy(), (low.sign()*low.square()).cpu().numpy()))
        rows.append(dict(file=path.name,
            decoded_lod0=compare_shape(heights[0][0], heights[1][0]),
            native_lod3=compare_shape(heights[0][1], heights[1][1])))
    return rows


def qualified_validation(root):
    if json.loads((root/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Finish the rare native capture and physical qualification first.')
    rows = json.loads((root/'qualification.json').read_text())['rows']
    selected = {row['file']: row for row in rows if row['split'] == 'val' and row['physical_qualified']}
    if {row['requested'] for row in selected.values()} != {
            'desert-plain', 'temperate-plain', 'wet-complex-coast', 'low-plain-sea'}:
        raise ValueError('Independent physically qualified validation must cover all four case kinds.')
    if any(row['seed'] in HOLDOUT_SEEDS for row in selected.values()):
        raise ValueError('Reserved final test worlds must not enter this validation.')
    return sorted(root/name for name in selected), selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATA/'crops/native-rare-scenes-v1')
    parser.add_argument('--statistics', type=Path, default=DATA/'eval/native-decoded-probe/channel-statistics.json')
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shape', action='store_true', help='Also measure same-position valleys and gradient directions.')
    args = parser.parse_args()
    files, classes = qualified_validation(args.dataset)
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = True
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64, 'cuda', torch.float32)
    decoder = OriginalDecoder('cuda')
    stats = json.loads(args.statistics.read_text())
    if stats['split'] != 'train':
        raise ValueError('Use fixed global TRAIN variance, never validation variance.')
    variance = torch.tensor(stats['variance'], device='cuda')
    results = []
    for path in args.checkpoints:
        checkpoint_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['contract']['statistics_sha256'] != hashlib.sha256(args.statistics.read_bytes()).hexdigest():
            raise ValueError('Evaluation variance differs from the fixed source training contract.')
        model = NativeBase(saved['config']).cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['ema'])
        details = validate(model, files, weight, decoder, variance)
        for row in details['rows']:
            row['kind'] = classes[row['file']]['requested']
            row['climate_archetype'] = classes[row['file']]['climate_archetype']
        by_kind = {}
        for kind in sorted({row['kind'] for row in details['rows']}):
            selected = [row for row in details['rows'] if row['kind'] == kind]
            by_kind[kind] = dict(count=len(selected), worlds=len({row['seed'] for row in selected}),
                median_height_mae_m=float(np.median([row['height_mae_m'] for row in selected])),
                mean_height_gradient=float(np.mean([row['height_gradient'] for row in selected])),
                mean_height_tpi=float(np.mean([row['height_tpi'] for row in selected])),
                feature_relative_mse=float(np.mean([row['feature_relative_mse'] for row in selected])))
        results.append(dict(checkpoint=str(path), sha256=checkpoint_sha, step=saved['step'],
                            source_step=saved['contract']['source_step'], by_kind=by_kind,
                            shapes=shape_validation(model, files, weight, decoder) if args.shape else None,
                            **details))
        print(json.dumps(dict(step=saved['step'], by_kind=by_kind)), flush=True)
        del model, saved
        torch.cuda.empty_cache()
    atomic_json(external_path(args.output), dict(status='complete', gpu=torch.cuda.get_device_name(),
        dataset_manifest_sha256=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest(),
        qualification_sha256=hashlib.sha256((args.dataset/'qualification.json').read_bytes()).hexdigest(),
        statistics_sha256=hashlib.sha256(args.statistics.read_bytes()).hexdigest(),
        decoder_sha256=hashlib.sha256((teacher_source().parent/'decoder_model/diffusion_pytorch_model.safetensors').read_bytes()).hexdigest(),
        producer_sha256={name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in (
            'distill/native_decoded_eval.py', 'distill/native_decoded.py', 'distill/native_decoded_train.py')},
        results=results, accepted=False,
        note='Independent rare validation, frozen original-decoder interior proxy. No training, no reserved test world, no performance claim or final comparison plates.'))


if __name__ == '__main__':
    main()

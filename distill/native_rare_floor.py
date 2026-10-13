"""Measure natural FP32/BF16 shape floors on the new rare validation cores.

Both variants retain original coarse/decoder models and identical native query
geometry. An original-BF16 replay also measures capture-to-current rounding.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA, REPO, atomic_json, external_path
from distill.native_decoded import DecodedRegion, OriginalDecoder
from distill.native_decoded_eval import qualified_validation
from distill.decoded_loss import reconstructed_height
from distill.terrain_shape import compare_shape


def views(latents, region, decoder):
    q = latents[4]*38.6-31.4
    low = q.sign()*q.square()
    height = reconstructed_height(decoder(latents[None], region.decoder_noise), latents[None])[0, 0]
    return dict(native_lod3=low.cpu().numpy(), decoded_lod0=height.cpu().numpy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=DATA/'crops/native-rare-scenes-v1')
    parser.add_argument('--output', type=Path, default=DATA/'eval/native-rare-fp32-floor.json')
    args = parser.parse_args()
    files, classes = qualified_validation(args.dataset)
    # Must precede Worlds: the tool bootstrap changes cwd, whereas terrain_app
    # chooses the upstream relative geography directory on its first import.
    from tools.verification.compare_base_variants import Cast, load_copy
    from distill.teacher import Worlds
    torch.set_num_threads(8)
    torch.backends.cudnn.benchmark = False
    first = DecodedRegion(files[0])
    worlds = Worlds(first.metadata['seed'])
    fp32 = Cast(load_copy('base_model', torch.float32), torch.float32)
    decoder = OriginalDecoder('cuda', model=worlds.loaded.decoder_model)
    rows = []
    with torch.inference_mode():
        for path in files:
            region = DecodedRegion(path, 'cuda')
            reference = views(region.target, region, decoder)
            row = dict(file=path.name, seed=region.metadata['seed'], profile=region.metadata['profile'],
                       kind=classes[path.name]['requested'], core=region.core, variants={})
            for variant in ('bf16_replay', 'fp32base'):
                world = worlds.create(region.metadata['seed'], region.metadata['profile'])
                if variant == 'fp32base':
                    world.base_model = fp32
                y, x = region.metadata['y'], region.metadata['x']
                try:
                    # Replay the complete128-square native capture first,
                    # including its native batch grouping and dependencies.
                    weighted = world.latents[:, y:y+128, x:x+128].float()
                    oy, ox = region.core[0]-y, region.core[1]-x
                    weighted = weighted[:, oy:oy+64, ox:ox+64]
                    latent = weighted[:-1]/weighted[-1:]
                    actual = views(latent, region, decoder)
                    result = dict(latent_max_abs=float((latent-region.target).abs().max()), views={})
                    for view in reference:
                        error = actual[view].astype(np.float64)-reference[view]
                        result['views'][view] = dict(mae_m=float(np.mean(np.abs(error))),
                            rmse_m=float(np.sqrt(np.mean(error**2))), shape=compare_shape(actual[view], reference[view]))
                    row['variants'][variant] = result
                finally:
                    world.close()
            rows.append(row)
            print(json.dumps(dict(file=path.name, kind=row['kind'],
                fp32_mae_m={view: value['mae_m'] for view, value in row['variants']['fp32base']['views'].items()},
                replay_mae_m={view: value['mae_m'] for view, value in row['variants']['bf16_replay']['views'].items()})), flush=True)
    atomic_json(external_path(args.output), dict(status='complete', gpu=torch.cuda.get_device_name(), rows=rows,
        dataset_manifest_sha256=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest(),
        qualification_sha256=hashlib.sha256((args.dataset/'qualification.json').read_bytes()).hexdigest(),
        producer_sha256={name: hashlib.sha256((REPO/name).read_bytes()).hexdigest() for name in (
            'distill/native_rare_floor.py', 'distill/native_decoded.py', 'distill/terrain_shape.py',
            'tools/verification/compare_base_variants.py')},
        accepted=False, note='Natural original-base precision floor and same-GPU BF16 capture replay; shape correlations on flat terrain must be interpreted against this floor. Aligned decoder interior proxy, no final plates.'))


if __name__ == '__main__':
    main()

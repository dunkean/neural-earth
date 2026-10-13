"""Same-scene BF16/FP32 floor for progressive-width validation, no training."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch

from distill.common import DATA, REPO, atomic_json, external_path
from distill.decoded_loss import reconstructed_height
from distill.native_base import teacher_source
from distill.native_decoded import DecodedRegion, OriginalDecoder, decoded_region_forward
from distill.native_progress_images import LABELS
from distill.native_progressive_prepare import configure
from distill.native_weight_quant import file_sha
from distill.terrain_shape import compare_shape
from distill.wet_mountains import ROOT
from distill.wet_mountain_progress import selected_validation
from terrain_diffusion.models.edm_unet import EDMUnet2D


class TypedTeacher:
    def __init__(self, net, dtype):
        self.net, self.dtype = net, dtype

    def __call__(self, inputs, labels, conditions):
        return self.net(inputs.to(self.dtype), noise_labels=labels.to(self.dtype),
                        conditional_inputs=[value.to(self.dtype) for value in conditions]).float()


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve the previous precision-floor measurement.')
    configure()
    bf16 = TypedTeacher(EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False), torch.bfloat16)
    fp32 = TypedTeacher(EDMUnet2D.from_pretrained(str(teacher_source())).cuda().float().eval().requires_grad_(False), torch.float32)
    decoder = OriginalDecoder('cuda')
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64, 'cuda', torch.float32)
    main_root = DATA/'crops/native-mixed-scenes-v2'
    manifest = json.loads((main_root/'manifest.json').read_text())
    rows, fields, repeats = [], {}, []
    for coarse in ('coarse4', 'coarse2'):
        selected = []
        for kind in LABELS:
            choices = [r for r in manifest['scenes'] if r['metadata']['split'] == 'val'
                       and r['case_kind'] == kind and r['coarse_variant'] == coarse]
            selected.append((main_root, sorted(choices, key=lambda r:(r['metadata']['seed'], r['file']))[0]))
        selected += [(ROOT, row) for row in selected_validation(ROOT, coarse)[0]]
        for root, record in selected:
            path = root/record['file']
            region = DecodedRegion(path, 'cuda')
            reference = decoded_region_forward(bf16, region, weight)['core']
            repeated = decoded_region_forward(bf16, region, weight)['core']
            if not torch.equal(reference, repeated):
                raise ValueError('Same-input BF16 teacher repeat differed: '+str(path))
            repeats.append(dict(file=str(path), exact=True))
            actual = decoded_region_forward(fp32, region, weight)['core']
            heights = []
            for latent in (reference, actual):
                q = latent[4].float()*38.6-31.4
                low = (q.sign()*q.square()).cpu().numpy()[16:48, 16:48]
                high = reconstructed_height(decoder(latent[None], region.decoder_noise), latent[None])[0, 0].cpu().numpy()[64:-64, 64:-64]
                heights.append({3:low, 0:high})
            for lod in (3, 0):
                a, b = heights[0][lod], heights[1][lod]
                delta = b-a
                hp = [v-gaussian_filter(v, 1., mode='reflect') for v in (a, b)]
                land = a > 0
                land[:3] = land[-3:] = False
                land[:, :3] = land[:, -3:] = False
                energy = float(np.mean(hp[0][land]**2)) if land.any() else 0.
                row = dict(kind=record['case_kind'], seed=record['metadata']['seed'], coarse=coarse, lod=lod,
                    mae_m=float(np.abs(delta).mean()), height_bias_m=float(delta.mean()),
                    coast_disagreement=float(np.mean((a > 0) != (b > 0))),
                    high_frequency_rms_ratio=float(np.sqrt(np.mean(hp[1][land]**2)/energy)) if energy > 1e-12 else None,
                    shape=compare_shape(b, a), capture=str(path), capture_sha256=file_sha(path))
                rows.append(row)
                prefix = f'{coarse}_{record["case_kind"]}_{record["metadata"]["seed"]}_lod{lod}'
                if prefix+'_bf16' in fields:
                    raise ValueError('Repeated field identifier: '+prefix)
                fields[prefix+'_bf16'] = a
                fields[prefix+'_fp32'] = b
    if len(rows) != 28 or len(fields) != 56 or not all(np.isfinite(r['mae_m']) for r in rows):
        raise ValueError('Incomplete precision floor.')
    output.mkdir(parents=True)
    np.savez_compressed(output/'arrays.npz', **fields)
    flat = [{k:v for k, v in r.items() if k != 'shape'} for r in rows]
    with (output/'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader();writer.writerows(flat)
    summaries = {coarse:{str(lod):dict(
        mae_m=float(np.mean([r['mae_m'] for r in rows if r['coarse']==coarse and r['lod']==lod])),
        abs_bias_m=float(np.mean([abs(r['height_bias_m']) for r in rows if r['coarse']==coarse and r['lod']==lod])))
        for lod in (3, 0)} for coarse in ('coarse4', 'coarse2')}
    atomic_json(output/'report.json', dict(status='complete', rows=rows, summaries=summaries,
        gpu=torch.cuda.get_device_name(), teacher_repeat_checks=repeats,
        teacher_sha256=file_sha(teacher_source()/'diffusion_pytorch_model.safetensors'),
        source_sha256=file_sha(REPO/'distill/native_progressive_precision_floor.py'),
        scope='Both teacher precisions recomputed with identical noise/conditioning on same GPU; original decoder unchanged. Same seven validation scenes and central32/256 fields as progressive evaluator.',
        note='Numerical reference floor, not an acceptance rule based only on average height error.'))
    print(json.dumps(dict(status='complete', summaries=summaries)), flush=True)


if __name__ == '__main__':
    main()

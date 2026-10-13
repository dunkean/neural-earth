"""Paired same-GPU terrain progress for gradual native channel removal."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
from scipy.ndimage import gaussian_filter
import torch

from distill.common import DATA, atomic_json, atomic_write, append_json, external_path
from distill.decoded_loss import reconstructed_height
from distill.native_base import teacher_source
from distill.native_decoded import DecodedRegion, OriginalDecoder, decoded_region_forward
from distill.native_progress_images import LABELS, shade
from distill.native_progressive_prepare import configure
from distill.native_progressive_width import load_progressive
from distill.native_weight_quant import file_sha
from distill.terrain_shape import compare_shape
from distill.wet_mountains import ROOT, KIND
from distill.wet_mountain_progress import selected_validation
from terrain_diffusion.models.edm_unet import EDMUnet2D


class TeacherForward:
    def __init__(self, net):
        self.net = net

    def __call__(self, inputs, labels, conditions):
        return self.net(inputs.bfloat16(), noise_labels=labels.bfloat16(),
                        conditional_inputs=[value.bfloat16() for value in conditions]).float()


@torch.no_grad()
def evaluate(checkpoint, output, coarse='coarse4', label='progressive160'):
    configure()
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    stem = output/f'base-{label}-{saved["step"]:06d}-{coarse}'
    if stem.with_suffix('.json').exists():
        raise ValueError('Existing progress report must be preserved.')
    model = load_progressive(saved, 'cuda')
    teacher = TeacherForward(EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False))
    decoder = OriginalDecoder('cuda')
    main = DATA/'crops/native-mixed-scenes-v2'
    manifest = json.loads((main/'manifest.json').read_text())
    selected = []
    for kind in LABELS:
        choices = [row for row in manifest['scenes'] if row['metadata']['split'] == 'val'
                   and row['case_kind'] == kind and row['coarse_variant'] == coarse]
        selected.append((main, sorted(choices, key=lambda row:(row['metadata']['seed'], row['file']))[0]))
    selected += [(ROOT, row) for row in selected_validation(ROOT, coarse)[0]]
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64, 'cuda', torch.float32)
    rows, fields = [], []
    with model.cached_weights():
        for root, record in selected:
            region = DecodedRegion(root/record['file'], 'cuda')
            cores = [decoded_region_forward(net, region, weight)['core'] for net in (teacher, model)]
            heights = []
            for latent in cores:
                q = latent[4].float()*38.6-31.4
                low = (q.sign()*q.square()).cpu().numpy()[16:48, 16:48]
                high = reconstructed_height(decoder(latent[None], region.decoder_noise), latent[None])[0, 0].cpu().numpy()[64:-64, 64:-64]
                heights.append({3:low, 0:high})
            for lod in (3, 0):
                reference, actual = heights[0][lod], heights[1][lod]
                delta = actual-reference
                hp = [value-gaussian_filter(value, 1., mode='reflect') for value in (reference, actual)]
                land = reference > 0
                land[:3] = land[-3:] = False;land[:, :3] = land[:, -3:] = False
                energy = float(np.mean(hp[0][land]**2)) if land.any() else 0.
                shape = compare_shape(actual, reference)
                row = dict(step=saved['step'], amount=saved['amount'], kind=record['case_kind'],
                    seed=record['metadata']['seed'], coarse=coarse, lod=lod,
                    mae_m=float(np.abs(delta).mean()), height_bias_m=float(delta.mean()),
                    coast_disagreement=float(np.mean((actual > 0) != (reference > 0))),
                    high_frequency_rms_ratio=float(np.sqrt(np.mean(hp[1][land]**2)/energy)) if energy > 1e-12 else None)
                rows.append(row|dict(shape=shape));fields.append((reference, actual))
    size, left, gap, header, row_height = 512, 245, 12, 125, 556
    image = Image.new('RGB', (left+2*(size+gap), header+len(rows)*row_height), (24, 24, 24))
    draw = ImageDraw.Draw(image);font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    draw.text((12, 10), f'Reduction192 vers160 | etape {saved["step"]} | retrait {saved["amount"]:.3f} | {coarse}', fill='white', font=font)
    draw.text((12, 40), 'Teacher recalcule meme GPU/bruit, decoder original. Champs32/256px agrandis; pas de lissage ajoute.', fill='white', font=font)
    draw.text((left, 83), 'Teacher BF16', fill='white', font=font)
    selected_weights = saved.get('selected_weights', 'ema')
    caption = {'ema':'Eleve progressif EMA', 'raw':'Eleve, poids bruts',
               'untrained':'Teacher projete, sans entrainement'}.get(selected_weights, selected_weights)
    draw.text((left+size+gap, 83), caption, fill='white', font=font)
    names = LABELS|{KIND:'Haute montagne humide'}
    for i, (row, pair) in enumerate(zip(rows, fields, strict=True)):
        y = header+i*row_height
        draw.multiline_text((12, y+140), f'{names[row["kind"]]}\nLOD {row["lod"]}\nSeed {row["seed"]}\nErreur {row["mae_m"]:.2f} m\nRelief x1.5', fill='white', font=font, spacing=8)
        for col, field in enumerate(pair):
            image.paste(shade(field, 30*2**row['lod'], 1.5).resize((size, size), Image.Resampling.NEAREST), (left+col*(size+gap), y))
    metadata = PngImagePlugin.PngInfo();metadata.add_text('CheckpointSHA256', file_sha(checkpoint))
    atomic_write(stem.with_suffix('.png'), lambda handle:image.save(handle, format='PNG', pnginfo=metadata))
    flat = [{key:value for key, value in row.items() if key != 'shape'} for row in rows]
    with stem.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    stem.with_suffix('.txt').write_text('Comparaison appariee meme GPU/bruit, teacher BF16 recalcule, decoder original. Champs natifs32/256px agrandis. CSV/JSON donnent les metriques et la morphologie.\n')
    report = dict(status='complete', checkpoint=str(checkpoint), checkpoint_sha256=file_sha(checkpoint),
        gpu=torch.cuda.get_device_name(), step=saved['step'], amount=saved['amount'], rows=rows,
        selected_weights=selected_weights,
        reference_scope='Teacher original BF16 recomputed on same GPU, source noise and conditioning.', accepted=False)
    atomic_json(stem.with_suffix('.json'), report)
    append_json(DATA/'_base/image-publications.jsonl', dict(image=str(stem.with_suffix('.png')), step=saved['step'], coarse=coarse, label='progressive160'))
    print(json.dumps(dict(image=str(stem.with_suffix('.png')), step=saved['step'], coarse=coarse)), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=DATA/'_base')
    parser.add_argument('--coarse', choices=('coarse4', 'coarse2'), default='coarse4')
    parser.add_argument('--label', default='progressive160')
    args = parser.parse_args()
    evaluate(args.checkpoint, external_path(args.output), args.coarse, args.label)


if __name__ == '__main__':
    main()

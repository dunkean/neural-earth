"""Publish real matched native validation relief in the user's flat _base folder.

Same four teacher-selected rare scenes at every checkpoint, identical noise,
coarse branch, frozen original decoder and fixed lighting. No denoising or
student-based scene selection. This interior validation view is distinct from
the full application and cold tile continuity checks.
"""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import torch

from distill.common import DATA, REPO, REVISION, atomic_json, atomic_write, external_path
from distill.native_base import NativeBase
from distill.native_decoded import DecodedRegion, OriginalDecoder, decoded_region_forward
from distill.decoded_loss import reconstructed_height
from distill.native_multilod import low_height
from distill.native_multilod_eval import validation_records
from distill.native_mixed_bank import KINDS
from distill.terrain_shape import compare_shape

LABELS = {'desert-plain': 'Plaine desertique', 'temperate-plain': 'Plaine temperee',
          'wet-complex-coast': 'Cote humide torturee', 'low-plain-sea': 'Plaine basse / mer'}


def shade(elevation, spacing, exaggeration):
    gy, gx = np.gradient(elevation.astype(np.float64)*exaggeration, spacing)
    slope, aspect = np.arctan(np.hypot(gx, gy)), np.arctan2(-gx, gy)
    light = np.clip(np.sin(np.pi/4)*np.cos(slope)+np.cos(np.pi/4)*np.sin(slope)*
                    np.cos(np.radians(315)-aspect), 0, 1)
    colour = np.where((elevation > 0)[..., None], [200, 185, 150], [120, 150, 190])/255.
    return Image.fromarray((colour*(.25+.75*light[..., None])*255).astype(np.uint8))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--dataset', type=Path, default=DATA/'crops/native-mixed-scenes-v2')
    parser.add_argument('--output', type=Path, default=DATA/'_base')
    parser.add_argument('--coarse', choices=('original', 'coarse4', 'coarse2'), default='coarse4')
    parser.add_argument('--label', default='', help='Distinct strategy prefix when pilot step numbers restart.')
    parser.add_argument('--mountains', type=Path, default=DATA/'crops/native-wet-mountains-v1',
                        help='Independent additional wet high-mountain bank; publish a separate comparison.')
    args = parser.parse_args()
    if args.label and any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.label):
        parser.error('Strategy label must contain only lowercase letters, digits, - or _.')
    root, output = external_path(args.dataset), external_path(args.output)
    manifest = json.loads((root/'manifest.json').read_text())
    statistics = json.loads((DATA/'eval/native-decoded-probe/channel-statistics.json').read_text())
    records = validation_records(manifest, statistics)
    if json.loads((root/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Complete independent matched bank required.')
    selected = []
    for kind in KINDS:
        pool = sorted((r for r in records if r['coarse_variant'] == args.coarse and r['case_kind'] == kind),
                      key=lambda r: (r['metadata']['seed'], r['file']))
        if not pool:
            raise ValueError('Each rare category must be represented.')
        selected.append(pool[0])
    torch.set_num_threads(8); torch.backends.cudnn.benchmark = True
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64, 'cuda', torch.float32)
    decoder = OriginalDecoder('cuda')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    for path in args.checkpoints:
        path = path.expanduser().resolve()
        payload = path.read_bytes(); sha = hashlib.sha256(payload).hexdigest()
        saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=True)
        if (saved.get('teacher_revision') or saved.get('dataset_manifest', {}).get('revision')) != REVISION:
            raise ValueError('Checkpoint teacher revision differs.')
        if saved.get('architecture') == 'native-lowrank':
            from distill.native_lowrank import NativeLowrankBase
            model = NativeLowrankBase(saved['config'], saved['factor_ranks'], saved['factor_mode'])
        else:
            model = NativeBase(saved['config'])
        model = model.cuda().eval().requires_grad_(False)
        model.load_state_dict(saved['ema'])
        step = saved['step']; rows, fields = [], []
        stem = f'{args.label}-{step:06d}' if args.label else f'{step:06d}'
        for record in selected:
            region = DecodedRegion(root/record['file'], 'cuda')
            result = decoded_region_forward(model, region, weight)
            heights = []
            for latent in (region.target, result['core']):
                fine = reconstructed_height(decoder(latent[None], region.decoder_noise), latent[None])[0, 0]
                heights.append({3: low_height(latent)[0, 0].cpu().numpy(),
                               0: fine[64:-64, 64:-64].cpu().numpy()})
            for lod in (3, 0):
                reference, candidate = heights[0][lod], heights[1][lod]
                morphology = compare_shape(candidate, reference)
                tpi = next(r for r in morphology['topographic_position'] if r['sigma_pixels'] == 2.)
                rows.append(dict(step=step, kind=record['case_kind'], file=record['file'],
                    seed=record['metadata']['seed'], profile=record['metadata']['profile'],
                    coarse=args.coarse, lod=lod, native_y=region.core[0], native_x=region.core[1],
                    mae_m=float(np.abs(candidate-reference).mean()),
                    coast_disagreement=float(np.mean((candidate > 0) != (reference > 0))),
                    tpi_correlation=tpi.get('tpi_correlation'), valley_retention=tpi.get('valley_retention'),
                    tpi_rms_ratio=tpi.get('tpi_rms_ratio')))
                fields.append((reference, candidate))
        size, left, pad, row_height, header = 512, 250, 12, 556, 140
        image = Image.new('RGB', (left+2*(size+pad), header+len(rows)*row_height), (24, 24, 24))
        draw = ImageDraw.Draw(image)
        source_step = saved.get('contract', {}).get('source_step')
        context = f' | {args.label}' if args.label else ''
        if args.label and source_step is not None:
            context += f', source {source_step}'
        candidate_label = 'Teacher dans NativeBase' if saved.get('evaluation_only') else f'Eleve, 2 passes, etape {step}'
        if saved.get('architecture') == 'native-lowrank':
            candidate_label = (f'Teacher factorise, rang {saved["rank_fraction"]:.3f}, sans entrainement'
                if saved.get('evaluation_only') else f'Eleve factorise, rang {saved["rank_fraction"]:.3f}, etape {step}')
        draw.text((12, 10), f'Etape {step}{context} | Base {model.config["model_channels"]}x{model.config["layers_per_block"]} | {args.coarse}', fill='white', font=font)
        draw.text((12, 39), 'Validation interne, decoder original fige. Meme bruit et memes scenes. Checkpoint '+sha[:12], fill='white', font=small)
        draw.text((12, 66), 'Ombrage fixe, sans lissage. LOD3 : 64 pixels natifs ; LOD0 : interieur 384 pixels.', fill='white', font=small)
        draw.text((left, 102), 'Teacher original, 2 passes', fill='white', font=font)
        draw.text((left+size+pad, 102), candidate_label, fill='white', font=font)
        for index, (row, pair) in enumerate(zip(rows, fields)):
            y = header+index*row_height
            lod = row['lod']; exaggeration = 3. if lod == 3 else 1.5
            title = LABELS[row['kind']]+f'\nLOD {lod}\nSeed {row["seed"]}\nErreur : {row["mae_m"]:.2f} m\nRelief exagere x{exaggeration:g}'
            draw.multiline_text((12, y+125), title, fill='white', font=small, spacing=8)
            for column, elevation in enumerate(pair):
                panel = shade(elevation, 30*2**lod, exaggeration).resize((size, size), Image.Resampling.NEAREST)
                image.paste(panel, (left+column*(size+pad), y))
        description = (f'Etape {step}. Teacher et eleve sur les memes quatre scenes rares de validation, '
                       f'coarse {args.coarse}, decoder original. Les valeurs exactes figurent dans {stem}.csv. '
                       'Ce rendu ne certifie pas encore la chaine complete de l application ou les raccords de tuiles.')
        meta = PngImagePlugin.PngInfo(); meta.add_text('Description', description); meta.add_text('CheckpointSHA256', sha)
        image_path = output/f'{stem}.png'
        if image_path.exists():
            with Image.open(image_path) as existing:
                if existing.info.get('CheckpointSHA256') != sha:
                    raise ValueError('Refuse to overwrite another model at the same step.')
        atomic_write(image_path, lambda handle: image.save(handle, format='PNG', pnginfo=meta))
        csv_path = output/f'{stem}.csv'
        with csv_path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        (output/f'{stem}.txt').write_text(description+'\n', encoding='utf-8')
        atomic_json(output/f'{stem}.json', dict(checkpoint=str(path), checkpoint_sha256=sha,
            strategy=args.label, source_step=source_step,
            step=step, gpu=torch.cuda.get_device_name(), dataset_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
            producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), rows=rows, accepted=False))
        print(json.dumps(dict(step=step, image=str(image_path))), flush=True)
        del model, saved, payload
        torch.cuda.empty_cache()
        mountain_status = args.mountains/'status.json'
        if mountain_status.exists() and json.loads(mountain_status.read_text())['status'] == 'complete':
            from distill.wet_mountain_progress import publish
            publish(path, args.label or 'base', args.mountains, output, args.coarse)
        else:
            print(json.dumps(dict(step=step, wet_mountain_comparison='pending teacher bank capture')), flush=True)


if __name__ == '__main__':
    main()

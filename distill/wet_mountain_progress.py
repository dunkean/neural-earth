"""Same-site mountain comparisons at every checkpoint, without selecting students.

Only physically qualified VAL scenes participate. Reserved final worlds are
kept for the final application comparison, never a progressive training gate.
"""
import argparse
import csv
import gc
import hashlib
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
from distill.native_progress_images import shade
from distill.terrain_shape import compare_shape
from distill.wet_mountains import ROOT, check_splits, digest


def selected_validation(root, coarse):
    manifest = json.loads((root/'manifest.json').read_text())
    if json.loads((root/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Finish mountain teacher capture before comparisons.')
    check_splits(manifest['scenes'], [])
    qualified = {r['file']: r for r in json.loads((root/'qualification.json').read_text())['rows']
                 if r['physical_qualified'] and r['split'] == 'val' and r['coarse_variant'] == coarse}
    selected = {}
    for row in sorted(manifest['scenes'], key=lambda r: r['file']):
        if row['file'] in qualified:
            q = qualified[row['file']]
            if (row['metadata']['split'] != 'val' or row['coarse_variant'] != coarse or
                    q['seed'] != row['metadata']['seed']):
                raise ValueError('Qualification and captured scene identities differ.')
            seed = row['metadata']['seed']
            # Stable teacher-only selection; identical scenes for every student.
            if seed not in selected:
                selected[seed] = row
    if len(selected) < 3:
        raise ValueError(f'Need three independent physically qualified wet-mountain VAL worlds, found {len(selected)}.')
    return list(selected.values()), qualified


@torch.no_grad()
def publish(checkpoint, label, root=ROOT, output=DATA/'_base', coarse='coarse4'):
    root, output = external_path(root), external_path(output)
    selected, qualifications = selected_validation(root, coarse)
    checkpoint = Path(checkpoint).expanduser().resolve()
    checkpoint_sha = digest(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if (saved.get('teacher_revision') or saved.get('dataset_manifest', {}).get('revision')) != REVISION:
        raise ValueError('Checkpoint teacher revision differs.')
    if saved.get('architecture') == 'native-lowrank':
        from distill.native_lowrank import NativeLowrankBase
        model = NativeLowrankBase(saved['config'], saved['factor_ranks'], saved['factor_mode'])
        candidate_name = f'Rang {saved["rank_fraction"]:.3f}, '+('controle non entraine' if saved.get('evaluation_only') else f'etape {saved["step"]}')
    else:
        model = NativeBase(saved['config'])
        candidate_name = f'Eleve {saved["config"]["model_channels"]}x{saved["config"]["layers_per_block"]}, etape {saved["step"]}'
    model.load_state_dict(saved['ema'])
    model = model.cuda().eval().requires_grad_(False)
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight, decoder = linear_weight_window(64, 'cuda', torch.float32), OriginalDecoder('cuda')
    rows, fields = [], []
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
            row = dict(step=saved['step'], file=record['file'], seed=record['metadata']['seed'],
                coarse=coarse, lod=lod, mae_m=float(np.abs(candidate-reference).mean()),
                shape=compare_shape(candidate, reference), teacher=qualifications[record['file']])
            rows.append(row); fields.append((reference, candidate))
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 19)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    size, left, gap, header, row_height = 768, 260, 12, 140, 812
    image = Image.new('RGB', (left+2*(size+gap), header+len(rows)*row_height), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    draw.text((12, 10), f'Haute montagne humide | {label} | etape {saved["step"]} | {coarse}', font=font, fill='white')
    draw.text((12, 40), '3 mondes VAL independants. Bruit identique, decoder original fige, sans lissage. Controle de morphologie interne.', font=small, fill='white')
    draw.text((12, 69), 'LOD0 : interieur384 pixels, 11,52km. LOD3 : 64 latents, 15,36km. Relief exagere x1.5.', font=small, fill='white')
    draw.text((left, 105), 'Teacher original, 2 passes', font=font, fill='white')
    draw.text((left+size+gap, 105), candidate_name, font=font, fill='white')
    flat = []
    for index, (row, pair) in enumerate(zip(rows, fields)):
        y = header+index*row_height
        tpi = next(r for r in row['shape']['topographic_position'] if r['sigma_pixels'] == 8.)
        annotation = (f'Haute montagne humide\nSeed {row["seed"]}\nLOD {row["lod"]}\nErreur {row["mae_m"]:.2f} m\n'
            f'Pluie {row["teacher"]["climate"]["rain"]:.0f} mm/an\n'
            f'Altitude p90 {row["teacher"]["decoded"]["land_height_p90"]:.0f} m\n'
            f'Correlation TPI8 {tpi.get("tpi_correlation", 0):.3f}\n'
            f'Rappel vallees {tpi.get("valley_retention", 0):.3f}\n'
            f'Amplitude TPI {tpi.get("tpi_rms_ratio", 0):.3f}')
        draw.multiline_text((12, y+160), annotation, font=small, fill='white', spacing=8)
        for col, elevation in enumerate(pair):
            panel = shade(elevation, 30*2**row['lod'], 1.5).resize((size, size), Image.Resampling.NEAREST)
            image.paste(panel, (left+col*(size+gap), y))
        for metric in row['shape']['topographic_position']:
            flat.append({k: row[k] for k in ('step', 'file', 'seed', 'coarse', 'lod', 'mae_m')} | metric)
    stem = f'mountains-{label}-{saved["step"]:06d}'
    png = output/f'{stem}.png'
    if png.exists():
        with Image.open(png) as previous:
            if previous.info.get('CheckpointSHA256') != checkpoint_sha:
                raise ValueError('Preserve another checkpoint image at this step.')
    description = ('Comparaison des memes trois hautes montagnes humides VAL, teacher original T2 '
        'et candidat, aux LOD3/0. Les metriques TPI mesurent la morphologie, pas le drainage. '
        'Images sans lissage ; inspection des vallees et raccords de la chaine complete encore requise.')
    meta = PngImagePlugin.PngInfo()
    meta.add_text('Description', description); meta.add_text('CheckpointSHA256', checkpoint_sha)
    atomic_write(png, lambda handle: image.save(handle, format='PNG', pnginfo=meta))
    columns = sorted(set().union(*(r.keys() for r in flat)))
    with (output/f'{stem}.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader(); writer.writerows(flat)
    (output/f'{stem}.txt').write_text(description+'\n')
    atomic_json(output/f'{stem}.json', dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_sha,
        step=saved['step'], label=label, dataset_manifest_sha256=digest(root/'manifest.json'),
        qualification_sha256=digest(root/'qualification.json'),
        producer_sha256=digest(REPO/'distill/wet_mountain_progress.py'),
        sources_sha256={name: digest(REPO/'distill'/name) for name in (
            'wet_mountains.py', 'native_decoded.py', 'decoded_loss.py', 'native_multilod.py',
            'native_progress_images.py', 'terrain_shape.py', 'native_base.py', 'native_lowrank.py')},
        gpu=torch.cuda.get_device_name(), rows=rows, accepted=False))
    print(json.dumps(dict(image=str(png), step=saved['step'])), flush=True)
    del model, saved, decoder; gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--dataset', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=DATA/'_base')
    parser.add_argument('--coarse', choices=('original', 'coarse4', 'coarse2'), default='coarse4')
    parser.add_argument('--wait-job')
    args = parser.parse_args()
    if not args.label or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.label):
        parser.error('Safe strategy label required.')
    if args.wait_job:
        from distill.run_training import wait_job
        wait_job(args.wait_job)
    torch.set_num_threads(8); torch.backends.cudnn.benchmark = True
    publish(args.checkpoint, args.label, args.dataset, args.output, args.coarse)


if __name__ == '__main__':
    main()

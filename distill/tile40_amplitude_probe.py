"""TRAIN-fitted gain and explicit oracle swaps diagnose decoded relief loss.

Oracle swaps isolate low-frequency height from the four detail latents. They
cannot be deployed: they use teacher latents and are labeled diagnostic only.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin
import torch

from distill.common import DATA, atomic_json, atomic_write, external_path
from distill.decoded_loss import decode, reconstructed_height
from distill.native_basis_training import inventory
from distill.native_decoded import DecodedRegion, OriginalDecoder
from distill.native_onepass import noise_windows
from distill.native_onepass_residual_pack import ResidualPackedRuntime
from distill.native_progress_images import LABELS, shade
from distill.native_progressive_prepare import configure
from distill.native_weight_quant import file_sha
from distill.student import load_student
from distill.terrain_shape import compare_shape
from distill.wet_mountain_progress import selected_validation
from distill.wet_mountains import ROOT, KIND


def fit_gain(pairs):
    numerator, denominator = 0., 0.
    for actual, reference in pairs:
        actual = np.asarray(actual, np.float64)
        reference = np.asarray(reference, np.float64)
        if actual.shape != reference.shape or not np.isfinite(actual).all() or not np.isfinite(reference).all():
            raise ValueError('Finite paired height fields required.')
        a, r = actual-actual.mean(), reference-reference.mean()
        # Equal scene weighting avoids choosing the gain from the largest mountain.
        scale = max(float(np.mean(r*r)), 1.)
        numerator += float(np.mean(a*r))/scale
        denominator += float(np.mean(a*a))/scale
    if denominator <= 1e-12 or numerator <= 0:
        raise ValueError('Gain correction cannot repair absent or reversed structure.')
    return numerator/denominator


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--packed', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--compact-decoder', type=Path, default=DATA/'final/weights/candidate-1/decoder.pt')
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve prior diagnostic artifacts.')
    output.mkdir(parents=True)
    configure()
    student = ResidualPackedRuntime(args.packed, 'cuda')
    decoder = OriginalDecoder('cuda')
    compact, _ = load_student(args.compact_decoder, 'cuda')
    compact.requires_grad_(False)
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64, 'cuda', torch.float32)

    def generate(root, record):
        region = DecodedRegion(root/record['file'], 'cuda')
        noises = noise_windows(record['metadata']['seed'], region.ref_origins, 'cuda')
        core = region.core_crop(student(noises, region.stages[1]['condition']), weight)
        return region, core

    def height(core, region, compact_model=False):
        if compact_model:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                residual = decode(compact, core[None], region.decoder_noise)
        else:
            residual = decoder(core[None], region.decoder_noise)
        return reconstructed_height(residual, core[None])[0, 0, 64:-64, 64:-64].cpu().numpy()

    records, _, _ = inventory()
    training = sorted([record for record in records if record['case_kind'] == KIND and
                       record['coarse_variant'] == 'coarse4'], key=lambda row:(row['metadata']['seed'], row['file']))[:6]
    if len(training) < 6 or any(record['metadata']['split'] != 'train' for record in training):
        raise ValueError('Six TRAIN mountain regions required for gain fitting.')
    pairs = []
    for record in training:
        region, core = generate(Path(record['root']), record)
        pairs.append((height(core, region), height(region.target, region)))
    gain = fit_gain(pairs)
    atomic_json(output/'gain-fit.json', dict(gain=gain, split='train',
        records=[dict(root=record['root'], file=record['file'], seed=record['metadata']['seed']) for record in training]))
    main_root = DATA/'crops/native-mixed-scenes-v2'
    manifest = json.loads((main_root/'manifest.json').read_text())
    selected = []
    for kind in LABELS:
        choices = [row for row in manifest['scenes'] if row['metadata']['split'] == 'val' and
                   row['case_kind'] == kind and row['coarse_variant'] == 'coarse4']
        selected.append((main_root, sorted(choices, key=lambda row:(row['metadata']['seed'], row['file']))[0]))
    selected.extend((ROOT, row) for row in selected_validation(ROOT, 'coarse4')[0])
    fields, rows = [], []
    for root, record in selected:
        region, core = generate(root, record)
        reference, current = height(region.target, region), height(core, region)
        low_swap = core.clone(); low_swap[4:] = region.target[4:]
        detail_swap = core.clone(); detail_swap[:4] = region.target[:4]
        variants = dict(current=current, train_gain=(current-current.mean())*gain+current.mean(),
            oracle_low=height(low_swap, region), oracle_details=height(detail_swap, region),
            teacher_compact_decoder=height(region.target, region, True))
        measurements = {name:dict(mae_m=float(np.abs(value-reference).mean()),
            bias_m=float((value-reference).mean()), shape=compare_shape(value, reference))
            for name, value in variants.items()}
        row = dict(kind=record['case_kind'], seed=record['metadata']['seed'], split='val', metrics=measurements)
        rows.append(row)
        fields.append((reference, *variants.values()))
        print(json.dumps(dict(kind=row['kind'], seed=row['seed'],
            mae={name:value['mae_m'] for name,value in measurements.items()})), flush=True)
        atomic_write(output/f'{row["kind"]}-{row["seed"]}.npz', lambda handle:
            np.savez_compressed(handle, reference=reference, **variants))
    titles = ['Original', 'NF4 actuel', f'Gain TRAIN ×{gain:.2f}', 'Oracle altitude', 'Oracle détails', 'Decoder compact']
    image = Image.new('RGB', (6*268, 110+len(rows)*300), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
    draw.text((12, 10), 'Diagnostic amplitude finale • mêmes bruit/climat • oracles non déployables', font=font, fill='white')
    draw.text((12, 40), 'Gain ajusté sur TRAIN uniquement • scènes VAL • pixels natifs256px • relief ×1,5', font=font, fill='white')
    for column, title in enumerate(titles):
        draw.text((column*268+6, 75), title, font=font, fill='white')
    for index, (row, panels) in enumerate(zip(rows, fields, strict=True)):
        y = 110+index*300
        for column, panel in enumerate(panels):
            image.paste(shade(panel, 30., 1.5), (column*268+6, y))
        draw.text((6, y+260), f'{row["kind"]} • seed {row["seed"]}', font=font, fill='white')
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text('Scope', 'Causal diagnostic;oracle variants require teacher features;not product evidence.')
    atomic_write(output/'amplitude-diagnostic.png', lambda handle:image.save(handle, format='PNG', pnginfo=metadata))
    packed = json.loads((args.packed/'manifest.json').read_text())
    atomic_json(output/'report.json', dict(status='complete', gain=gain, gain_fit='six TRAIN mountains',
        rows=rows, packed_weights_sha256=packed['sha256'], decoder_sha256=file_sha(args.compact_decoder),
        gpu=torch.cuda.get_device_name(), teacher_decoder='original', product_accepted=False,
        scope='Existing interior regions only;40km sketches and five climate families remain to test.'))


if __name__ == '__main__':
    main()

"""Review full40km paired reference terrain without displaying hidden network stages."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from distill.common import atomic_json, atomic_write, external_path
from distill.native_progress_images import shade
from distill.tile40_conditioning import CLIMATES, RELIEFS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.dataset, external_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root/'manifest.json').read_text())
    selected = []
    size, gap, header, row_height = 384, 12, 100, 450
    image = Image.new('RGB', (len(CLIMATES)*(size+gap), header+len(RELIEFS)*row_height), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    target = 'Sketche conservé + détails du decoder original' if manifest['target'] != 'original-teacher' else 'Teacher original'
    draw.text((12, 10), f'Tuiles40×40km • {target}', font=font, fill='white')
    draw.text((12, 40), 'Hauteurs calculées • ombrage identique • relief×1,5 • ce sont des cibles TRAIN, pas un nouvel élève', font=font, fill='white')
    for column, climate in enumerate(CLIMATES):
        draw.text((column*(size+gap)+8, 70), climate, font=font, fill='white')
    for index, relief in enumerate(RELIEFS):
        for column, climate in enumerate(CLIMATES):
            choices = sorted([row for row in manifest['cases'] if row['split'] == 'train' and row['relief'] == relief and
                              row['climate'] == climate and (root/row['file']).exists()], key=lambda row:row['seed'])
            if not choices:
                raise ValueError('All five climates and three relief classes need real target tiles.')
            row = choices[0]
            with np.load(root/row['file'], allow_pickle=False) as arrays:
                height = arrays['height_m']
            stats = json.loads((root/row['file']).with_suffix('.json').read_text())['stats']
            x, y = column*(size+gap), header+index*row_height
            panel = shade(height, manifest['spacing_m'], 1.5).resize((size, size), Image.Resampling.LANCZOS)
            image.paste(panel, (x, y))
            draw.text((x+8, y+size+6), f'{relief} • relief {stats["relief_p90_p10_m"]:.0f}m', font=font, fill='white')
            draw.text((x+8, y+size+30), f'PenteP90 {stats["slope_p90_degrees"]:.1f}°', font=font, fill='white')
            selected.append(dict(case=row, stats=stats))
    atomic_write(output/'fifteen-reference-tiles.png', lambda handle:image.save(handle, format='PNG'))
    atomic_json(output/'preview.json', dict(dataset=str(root), target=manifest['target'], cases=selected,
                scope='TRAIN target review;not student quality or held-out product proof.'))
    print(output/'fifteen-reference-tiles.png')


if __name__ == '__main__':
    main()

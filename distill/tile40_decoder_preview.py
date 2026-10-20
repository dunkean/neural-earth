"""Real VAL decoded-height comparisons across15 relief/climate cases."""
import argparse
import csv
import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from distill.common import DATA, atomic_json, atomic_write, external_path
from distill.native_progress_images import shade
from distill.student import load_student
from distill.tile40_conditioning import CLIMATES, RELIEFS
from distill.tile40_decoder_pilot import References, root_decode, physical_loss


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = DATA/'tile40-20261014'
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve previous comparisons;use a fresh output.')
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    refs = References(root/'reference-guided-v2')
    model, saved = load_student(root/'decoder16-pilot-v1/step-020000.pt')
    teacher, _ = load_student(DATA/'final/weights/candidate-1/decoder.pt')
    size, gap, row_height, header = 240, 12, 290, 130
    image = Image.new('RGB', (len(CLIMATES)*(2*size+gap), header+len(RELIEFS)*row_height), (24, 24, 24))
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    draw.text((12, 10), f'Decoder16 • EMA{saved["step"]} • 0,895Mo • comparaison de hauteurs finales VAL', font=font, fill='white')
    draw.text((12, 40), 'Gauche : référence guidée / Droite : decoder entraîné • même ombrage, relief×1,5', font=font, fill='white')
    draw.text((12, 70), 'Fenêtres11,52×11,52km • latents teacher fournis • pas une génération complète depuis sketch', font=font, fill='white')
    rows = []
    for column, climate in enumerate(CLIMATES):
        draw.text((column*(2*size+gap)+8, 100), climate, font=font, fill='white')
    for index, relief in enumerate(RELIEFS):
        for column, climate in enumerate(CLIMATES):
            row = min((r for r in refs.splits['val'] if r['case']['relief']==relief and r['case']['climate']==climate),
                      key=lambda r:r['case']['seed'])
            inputs, latents, target_root = refs.window(row, 384, 384, 'cpu')
            reference_residual, actual_residual = teacher(inputs), model(inputs)
            prediction_root = target_root+root_decode(actual_residual, latents)-root_decode(reference_residual, latents)
            _, values = physical_loss(prediction_root, target_root)
            target = (target_root.sign()*target_root.square())[0, 0].numpy()
            prediction = (prediction_root.sign()*prediction_root.square())[0, 0].numpy()
            x, y = column*(2*size+gap), header+index*row_height
            for offset, height in ((0, target), (size, prediction)):
                panel = shade(height, 30., 1.5).resize((size, size), Image.Resampling.LANCZOS)
                image.paste(panel, (x+offset, y))
            metrics = {k:float(v) for k,v in values.items()}
            draw.text((x+8, y+size+4), f'{relief} • MAE {metrics["height_mae_m"]:.2f}m', font=font, fill='white')
            draw.text((x+8, y+size+26), f'Écart amplitude {100*metrics["amplitude_gap"]:.1f}%', font=font, fill='white')
            rows.append(dict(**row['case'], **metrics, window_side_km=11.52))
    atomic_write(output/'fifteen-decoder-comparisons.png', lambda handle:image.save(handle, format='PNG'))
    atomic_json(output/'comparison.json', dict(step=saved['step'], rows=rows,
        scope='CPU FP32 VAL decoder component with oracle teacher latents and fixed guided-root context;not full40km generation.'))
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic_write(output/'comparison.csv', lambda handle:handle.write(table.getvalue().encode()))
    atomic_write(output/'description.txt', lambda handle:handle.write(
        ('Comparaison de15fenêtres de validation : cinq climats et trois reliefs.\n'
         'Chaque paire compare la référence guidée à la hauteur obtenue en remplaçant\n'
         'son decoder64 par le decoder16 EMA20k entraîné ; les latents sont fournis.\n'
         'Ombrage calculé avec un pas30m et une exagération identique1,5.\n'
         'Le tableauCSV donne les erreurs en mètres et les écarts d’amplitude.\n'
         'Ces fenêtres11,52km ne prouvent pas la génération complète40km depuis sketch.\n').encode()))
    print(output/'fifteen-decoder-comparisons.png', flush=True)


if __name__ == '__main__':
    main()

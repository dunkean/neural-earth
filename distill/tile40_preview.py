"""Render measured full40km VAL DEMs with common lighting and text/CSV alternatives."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from distill.common import atomic_json, atomic_write, external_path
from distill.native_progress_images import shade
from distill.native_weight_quant import file_sha
from distill.tile40_conditioning import CLIMATES, RELIEFS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--all-geometries', action='store_true', help='Show all six VAL sketches across the five climates.')
    args = parser.parse_args()
    report_path = args.evaluation/'report.json'
    report = json.loads(report_path.read_text())
    if report['status'] != 'complete':
        raise ValueError('Complete numerical evaluation required.')
    selected = []
    for relief in RELIEFS:
        seeds = sorted({row['case']['seed'] for row in report['rows'] if row['case']['relief']==relief})
        if not seeds:
            raise ValueError('All three relief groups must be represented.')
        for seed in seeds if args.all_geometries else seeds[:1]:
            for climate in CLIMATES:
                available = [row for row in report['rows'] if row['case']['relief']==relief
                             and row['case']['seed']==seed and row['case']['climate']==climate]
                if len(available)!=1:
                    raise ValueError('Each selected sketch must cover all five climates exactly once.')
                selected.append(available[0])
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve old plates;use a fresh output.')
    output.mkdir(parents=True)
    panel, gap, header, row_height = 240, 16, 130, 324
    width = len(CLIMATES)*(2*panel+gap)
    rows_count = len(selected)//len(CLIMATES)
    plate = Image.new('RGB', (width, header+rows_count*row_height), (24, 24, 24))
    predictions = Image.new('RGB', (len(CLIMATES)*(panel+gap), header+rows_count*row_height), (24,24,24))
    draw = ImageDraw.Draw(plate)
    prediction_draw = ImageDraw.Draw(predictions)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 18)
    draw.text((12, 10), args.label+f' | {len(selected)} tuiles VAL de 40 × 40 km', font=font, fill='white')
    draw.text((12, 40), 'Gauche : référence guidée issue du décodage original | Droite : génération depuis sketch/climat/seed', font=font, fill='white')
    draw.text((12, 70), 'Ombrage identique, relief ×1,5. Prototype expérimental : qualité non acceptée. Aucun coarse/base affiché.', font=font, fill='white')
    prediction_draw.text((12,10), args.label, font=font, fill='white')
    prediction_draw.text((12,40), f'{rows_count} sketches × 5 climats | 40 × 40 km | hauteurs finales', font=font, fill='white')
    prediction_draw.text((12,70), 'Ombrage ×1,5 | Prototype expérimental, qualité non acceptée.', font=font, fill='white')
    table_rows = []
    for index, row in enumerate(selected):
        r, c = divmod(index, len(CLIMATES))
        case = row['case']
        source = args.evaluation/case['file']
        with np.load(source, allow_pickle=False) as data:
            target, prediction = data['reference_height_m'], data['height_m']
        if target.shape != (1334, 1334) or prediction.shape != target.shape:
            raise ValueError('Only complete contracted40km tiles may be shown.')
        x, y = c*(2*panel+gap), header+r*row_height
        for offset, height in ((0, target), (panel, prediction)):
            plate.paste(shade(height, 40000./1334, 1.5).resize((panel, panel), Image.Resampling.LANCZOS), (x+offset, y))
        px = c*(panel+gap)
        predictions.paste(plate.crop((x+panel,y,x+2*panel,y+panel)),(px,y))
        if r == 0:
            draw.text((x+8, 100), case['climate'], font=font, fill='white')
            prediction_draw.text((px+8,100),case['climate'],font=font,fill='white')
        bands = row['metrics']['bands']
        draw.text((x+8, y+panel+5), f'{case["relief"]} | seed {case["seed"]}', font=font, fill='white')
        draw.text((x+8, y+panel+29), f'Amplitude 60 m : {bands[0]["amplitude_ratio"]:.2f}× | 240 m : {bands[1]["amplitude_ratio"]:.2f}×', font=font, fill='white')
        draw.text((x+8, y+panel+53), f'Corrélation 240 m : {bands[1]["correlation"]:.2f}', font=font, fill='white')
        prediction_draw.text((px+8,y+panel+5),case['relief'],font=font,fill='white')
        prediction_draw.text((px+8,y+panel+29),f'seed {case["seed"]}',font=font,fill='white')
        record = dict(relief=case['relief'], climate=case['climate'], seed=case['seed'], side_m=40000,
                      height_file=source.name, height_file_sha256=file_sha(source), height_mae_m=row['metrics']['height_mae_m'])
        for band in bands:
            scale = int(round(band['sigma_m']))
            record[f'amplitude_ratio_{scale}m'] = band['amplitude_ratio']
            record[f'correlation_{scale}m'] = band['correlation']
        table_rows.append(record)
    image_name='all-full40km-comparisons.png' if args.all_geometries else 'fifteen-full40km-comparisons.png'
    atomic_write(output/image_name, lambda handle:plate.save(handle, format='PNG'))
    atomic_write(output/'generated-full40km.png',lambda handle:predictions.save(handle,format='PNG'))
    table = io.StringIO()
    writer = csv.DictWriter(table, fieldnames=list(table_rows[0]))
    writer.writeheader()
    writer.writerows(table_rows)
    atomic_write(output/'comparison.csv', lambda handle:handle.write(table.getvalue().encode()))
    description = (f'{args.label}. {len(selected)} paires de tuiles complètes de40×40km.\n'
        'Colonnes : aride, froid, tropical, tempéré, océanique.\n'
        'Lignes groupées : montagnes, collines, plaines. Même sketch et même seed entre climats pour chaque ligne.\n'
        'À gauche, référence guidée construite à partir du décodage original.\n'
        'À droite, hauteurs générées uniquement depuis sketch, climat et seed.\n'
        'Les deux images utilisent la même lumière et le même facteur de relief1,5.\n'
        'Un rapport d’amplitude inférieur à1 mesure une atténuation ; il ne suffit pas à juger la morphologie.\n'
        'Le CSV fournit les amplitudes et corrélations pour chaque terrain. Qualité expérimentale non acceptée.\n')
    atomic_write(output/'description.txt', lambda handle:handle.write(description.encode()))
    atomic_json(output/'manifest.json', dict(evaluation_sha256=file_sha(report_path), label=args.label,
        scope=report['scope'], selected_weights=report.get('selected_weights', {'encoder':'ema', 'decoder':'ema'}),
        rows=table_rows, quality_accepted=False))
    print(output/image_name, flush=True)


if __name__ == '__main__':
    main()

"""Assemble30 real browser DEMs into a6x5plate,with physical climate diagnostics."""
import argparse
import csv
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter

from distill.common import atomic_json, atomic_write, external_path
from distill.native_weight_quant import file_sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads((args.capture/'report.json').read_text())
    if report['status']!='complete' or len(report['rows'])!=30:
        raise ValueError('Complete30case browser matrix required.')
    output=external_path(args.output)
    if output.exists():
        raise ValueError('Preserve old plates;use a fresh output.')
    output.mkdir(parents=True)
    climates=['arid','cold','tropical','temperate','oceanic']
    labels=['Aride','Froid','Tropical','Tempéré','Océanique']
    presets=list(dict.fromkeys(row['preset'] for row in report['rows']))
    size,gap,header,row_height=300,16,144,386
    plate=Image.new('RGB',(5*(size+gap),header+6*row_height),(24,24,24))
    draw=ImageDraw.Draw(plate);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    draw.text((12,12),'30 tuiles de 40 × 40 km réellement générées dans Edge / CPU',font=font,fill='white')
    draw.text((12,42),'Même croquis et même graine pour les cinq climats de chaque ligne. Ombrage fixe, relief ×1,5.',font=font,fill='white')
    draw.text((12,72),'Prototype encodeur20k / decoder8 original2k brut : qualité non acceptée. Aucun champ intermédiaire affiché.',font=font,fill='white')
    for c,label in enumerate(labels):draw.text((c*(size+gap)+8,112),label,font=font,fill='white')
    table=[]
    for r,preset in enumerate(presets):
        selected=[next(row for row in report['rows'] if row['preset']==preset and row['climate']==climate) for climate in climates]
        fields={}
        for row in selected:
            file=args.capture/(row['prefix']+'.f32')
            if file_sha(file)!=row['heightSha256']:
                raise ValueError('A captured browser DEM changed.')
            fields[row['climate']]=np.fromfile(file,dtype='<f4').reshape(1334,1334).astype(np.float64)
        baseline=fields['temperate']
        baseline_high=baseline-gaussian_filter(baseline,64.)
        for c,row in enumerate(selected):
            field=fields[row['climate']]
            bands={}
            for sigma in (2.,8.,32.,64.):
                high=field-gaussian_filter(field,sigma);margin=int(sigma*3)
                bands[int(round(sigma*40000./1334))]=float(np.sqrt(np.mean(high[margin:-margin,margin:-margin]**2)))
            high=field-gaussian_filter(field,64.)
            change=float(np.sqrt(np.mean((high[192:-192,192:-192]-baseline_high[192:-192,192:-192])**2)))
            x,y=c*(size+gap),header+r*row_height
            with Image.open(args.capture/(row['prefix']+'.png')) as image:
                if image.size!=(1334,1334):raise ValueError('Incomplete terrain PNG.')
                plate.paste(image.resize((size,size),Image.Resampling.LANCZOS),(x,y))
            relief=row['summary']['reliefM']
            draw.text((x+8,y+size+5),f'{preset} | relief P90−P10 : {relief:.0f} m',font=font,fill='white')
            draw.text((x+8,y+size+28),f'RMS 60 m : {bands[60]:.2f} m',font=font,fill='white')
            draw.text((x+8,y+size+51),f'Écart HF / tempéré : {change:.2f} m',font=font,fill='white')
            table.append(dict(preset=preset,climate=row['climate'],seed=row['seed'],
                relief_p90_p10_m=relief,climate_highpass_delta_rmse_m=change,
                worst_generation_seconds=max(m['generationMs'] for m in row['measurements'])/1000,
                worst_preview_ready_seconds=max(m['previewReadyMs'] for m in row['measurements'])/1000,
                **{f'rms_{scale}m':value for scale,value in bands.items()},height_sha256=row['heightSha256']))
    atomic_write(output/'six-sketches-five-climates.png',lambda handle:plate.save(handle,format='PNG'))
    text=io.StringIO();writer=csv.DictWriter(text,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    atomic_write(output/'measurements.csv',lambda handle:handle.write(text.getvalue().encode()))
    description=('Trente tuiles complètes de40×40km, calculées dans le navigateur.\n'
        'Lignes : plaine, collines, massif, crête, vallée, plateau. Colonnes : aride, froid, tropical, tempéré, océanique.\n'
        'Un même croquis et une même graine sont conservés entre les climats de chaque ligne.\n'
        'Les couleurs et la lumière restent fixes : les différences viennent des hauteurs calculées.\n'
        'Le CSV indique le relief P90−P10, les RMS par bande et le changement de détail par rapport au climat tempéré.\n'
        'Le RMS climatique est mesuré après retrait du fond à environ1920m, hors des marges. Il ne prouve pas un climat réaliste.\n'
        'La qualité du prototype reste non acceptée. Les références originales ne figurent pas sur cette planche.\n')
    atomic_write(output/'description.txt',lambda handle:handle.write(description.encode()))
    atomic_json(output/'report.json',dict(status='complete',capture_report_sha256=file_sha(args.capture/'report.json'),
        rows=table,quality_accepted=False,scope=report['scope']))
    print(output/'six-sketches-five-climates.png',flush=True)


if __name__=='__main__':main()

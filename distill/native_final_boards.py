"""Export paired 512px physical terrain boards and exact metric alternatives."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from distill.check_physical_seams import shade
from distill.common import atomic_json, atomic_write, external_path
from distill.native_site_manifest import digest
from distill.terrain_shape import compare_shape


def build(source,output):
    source=Path(source);output=external_path(output)
    report=json.loads((source/'report.json').read_text())
    if report['status']!='complete':raise ValueError('Completed physical seam evidence required.')
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    title_font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    rows=[]
    for site in report['sites']:
        capture=Path(site['source_scene'])
        bank=json.loads((capture.parent/'manifest.json').read_text())
        record=next(r for r in bank['scenes'] if r['file']==capture.name)
        kind=record['case_kind']
        for lod in report['lods']:
            name=f"{site['name']}-lod{lod}"
            paths={variant:source/f'{name}-{variant}.npz' for variant in ('reference','native_student')}
            fields={}
            for variant,path in paths.items():
                with np.load(path,allow_pickle=False) as arrays:fields[variant]=arrays['whole'].copy()
            reference,candidate=fields['reference'],fields['native_student']
            if reference.shape!=(512,512) or candidate.shape!=reference.shape:
                raise ValueError('Native 512px physical fields required; do not upscale.')
            difference=candidate.astype(np.float64)-reference
            shape=compare_shape(candidate,reference)
            row=dict(site=site['name'],kind=kind,seed=site['seed'],lod=lod,
                mae_m=float(np.abs(difference).mean()),height_bias_m=float(difference.mean()),
                max_abs_m=float(np.abs(difference).max()),
                coast_disagreement_pct=100*float(np.mean((candidate>0)!=(reference>0))),
                shape=shape,source_sha256={k:digest(p) for k,p in paths.items()})
            rows.append(row)
            stem=output/f'base-final-070000-{capture.stem}-lod{lod}'
            image=Image.new('RGB',(1060,662),(24,24,24));draw=ImageDraw.Draw(image)
            draw.text((12,10),f'{kind} | seed {site["seed"]} | LOD {lod} | {30*2**lod} m/pixel',font=title_font,fill='white')
            draw.text((12,43),'Coarse 4 et decoder livres identiques. Echelle et lumiere identiques, sans lissage ajoute.',font=font,fill='white')
            draw.text((12,74),'Base original BF16',font=title_font,fill='white')
            draw.text((542,74),'Base NF4 EMA 70 000 | ~150 Mo',font=title_font,fill='white')
            image.paste(Image.fromarray(shade(reference,lod)),(12,108))
            image.paste(Image.fromarray(shade(candidate,lod)),(542,108))
            draw.text((12,632),f'MAE {row["mae_m"]:.2f} m | biais {row["height_bias_m"]:+.2f} m | terre/mer en desaccord {row["coast_disagreement_pct"]:.2f} %',font=font,fill='white')
            metadata=PngImagePlugin.PngInfo();metadata.add_text('Description',
                f'Paired original / NF4 70k physical terrain. {kind}, seed {site["seed"]}, LOD {lod}. Native 512px fields. Metrics in adjacent JSON/CSV.')
            metadata.add_text('SourceReportSHA256',digest(source/'report.json'))
            atomic_write(stem.with_suffix('.png'),lambda f:image.save(f,format='PNG',pnginfo=metadata))
            atomic_json(stem.with_suffix('.json'),row)
            stem.with_suffix('.txt').write_text(f'{kind}, seed {site["seed"]}, LOD{lod}. Original vs NF4 EMA70k.\nMAE {row["mae_m"]:.6g} m; biais {row["height_bias_m"]:+.6g} m; desaccord terre/mer {row["coast_disagreement_pct"]:.6g} %.\nLe relief ombre montre les formes; les metriques ne constituent pas une acceptation.\n')
    atomic_json(output/'base-final-070000-physical-boards.json',dict(rows=rows,
        source=str(source),source_report_sha256=digest(source/'report.json'),accepted=False))
    with (output/'base-final-070000-physical-boards.csv').open('w',newline='') as handle:
        columns=['site','kind','seed','lod','mae_m','height_bias_m','max_abs_m','coast_disagreement_pct']
        writer=csv.DictWriter(handle,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    return len(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();print('boards',build(args.source,args.output))


if __name__=='__main__':main()

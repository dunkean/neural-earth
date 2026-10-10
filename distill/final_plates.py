"""Export native-resolution comparison plates from completed cold-tile checks.

All columns show the same 512x512 physical terrain footprint. Images are
measurements, with common shading, detail crops and signed-error scales.
The CSV and manifest provide the textual alternative and exact provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from distill.common import atomic_json, atomic_write, external_path
from distill.check_physical_seams import shade
from distill.rare_cases import local_errors
from tools.verification.compare_base_variants import describe, errors


KINDS = {'desert-plain': 'Plaine désertique', 'wet-complex-coast': 'Côte humide découpée',
         'low-plain-sea': 'Plaine basse / mer', 'temperate-plain': 'Plaine tempérée'}


def load_bank(directory):
    directory = Path(directory).resolve()
    payload = (directory/'report.json').read_bytes()
    report = json.loads(payload)
    if report.get('status') != 'complete' or not report.get('fresh_world_per_viewer_tile'):
        raise ValueError('Completed independent cold-tile evidence is required.')
    expected = {(s['name'], lod, variant) for s in report['sites'] for lod in report['lods']
                for variant in ('reference', 'student_all')}
    actual = {(r['site'], r['lod'], r['variant']) for r in report['rows']}
    if actual != expected or len(report['rows']) != len(expected):
        raise ValueError('Incomplete or duplicate physical views.')
    fields, sources = {}, []
    for site, lod, variant in sorted(expected):
        path = directory/f'{site}-lod{lod}-{variant}.npz'
        with np.load(path, allow_pickle=False) as arrays:
            value = arrays['tiled'].copy()
        if value.shape != (512, 512) or not np.isfinite(value).all():
            raise ValueError('Expected finite native 512x512 physical terrain.')
        fields[(site, lod, variant)] = value
        sources.append(dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return report, fields, dict(directory=str(directory), report_sha256=hashlib.sha256(payload).hexdigest(),
                               arrays=sources, checkpoint_digests=report['checkpoint_digests'])


def validate_banks(banks):
    first_report, first_fields, _ = banks[0]
    for report, fields, _ in banks[1:]:
        for key in ('gpu', 'sites', 'lods', 'teacher_sources', 'site_manifest_digest', 'tile_size', 'halo'):
            if report[key] != first_report[key]:
                raise ValueError('Comparison contexts differ: '+key)
        for key, value in first_fields.items():
            if key[2] == 'reference' and not np.array_equal(value, fields[key]):
                raise ValueError('Reference physical terrains differ across candidates.')


def error_rgb(delta, scale):
    normalized = np.clip(delta/scale, -1, 1)[..., None]
    neutral = np.full((*delta.shape, 3), 235.)
    positive, negative = np.array([185., 45., 45.]), np.array([40., 95., 190.])
    target = np.where(normalized >= 0, positive, negative)
    return (neutral+(target-neutral)*np.abs(normalized)).astype(np.uint8)


def build(candidates, output):
    if not candidates or len({label for label, _ in candidates}) != len(candidates):
        raise ValueError('Provide distinct candidate labels.')
    output = external_path(output)
    output.mkdir(parents=True, exist_ok=True)
    banks = [load_bank(directory) for _, directory in candidates]
    validate_banks(banks)
    font = ImageFont.truetype('DejaVuSans.ttf', 20)
    small = ImageFont.truetype('DejaVuSans.ttf', 16)
    title_font = ImageFont.truetype('DejaVuSans-Bold.ttf', 27)
    report, fields, _ = banks[0]
    rows, pages, links = [], [], []
    seam_rows = [{(r['site'], r['lod'], r['variant']): r for r in bank[0]['rows']} for bank in banks]
    for site in report['sites']:
        for lod in report['lods']:
            name = site['name']
            reference = fields[(name, lod, 'reference')]
            values = [bank[1][(name, lod, 'student_all')] for bank in banks]
            deltas = [v.astype(np.float64)-reference for v in values]
            # One symmetric scale shared by all candidates in this footprint.
            scale = max(float(np.quantile(np.abs(np.stack(deltas)), .98)), .01)
            panel, gap, margin = 512, 18, 28
            width = 2*margin+(panel+gap)*(1+len(candidates))-gap
            page = Image.new('RGB', (width, 1895), '#f4f2ee')
            draw = ImageDraw.Draw(page)
            heading = f"{KINDS.get(site['kind'], site['kind'])} · LOD {lod}"
            draw.text((margin, 20), heading, font=title_font, fill='#17242e')
            draw.text((margin, 60), f"{name} · {site['profile']} · seed {site['seed']} · {30*2**lod} m/pixel", font=font, fill='#334957')
            draw.text((margin, 95), 'Même cadrage : quatre tuiles froides, 512 × 512 pixels natifs. Même lumière, aucun lissage ajouté.', font=small, fill='#334957')
            ref_stats = describe(reference, lod)
            for col, (label, value) in enumerate([('Référence BF16', reference)]+list(zip([c[0] for c in candidates], values))):
                x = margin+col*(panel+gap)
                draw.text((x, 135), label, font=font, fill='#17242e')
                shaded = shade(value, lod)
                page.paste(Image.fromarray(shaded), (x, 175))
                draw.text((x, 701), 'Détail central : crop 256 × 256, zoom ×2', font=small, fill='#334957')
                page.paste(Image.fromarray(shaded[128:384, 128:384]).resize((512, 512), Image.Resampling.NEAREST), (x, 730))
                delta = np.zeros_like(reference) if col == 0 else deltas[col-1]
                draw.text((x, 1260), f'Erreur signée : échelle commune ±{scale:.2f} m', font=small, fill='#334957')
                page.paste(Image.fromarray(error_rgb(delta, scale)), (x, 1290))
                if col:
                    stats, error = describe(value, lod), errors(value, reference)
                    seam = seam_rows[col-1][(name, lod, 'student_all')]
                    local = local_errors(value, reference, lod)
                    slope = stats['slope_mean']/ref_stats['slope_mean'] if ref_stats['slope_mean'] else None
                    bands = [a/b if b else None for a, b in zip(stats['psd'], ref_stats['psd'])]
                    rows.append(dict(candidate=label, site=name, kind=site['kind'], lod=lod, seed=site['seed'],
                        mae_m=error['mae'], rmse_m=error['rmse'], p99_m=error['p99'], max_m=error['max'],
                        coast_sign_disagreement_pct=100*error['coast'], slope_ratio=slope,
                        land_mae_m=local['land']['mae_m'], low_land_mae_m=local['low_land_0_20m']['mae_m'],
                        coast_300m_mae_m=local['coast_300m']['mae_m'],
                        cold_tile_max_m=seam['max_m'], cold_tile_jump_max_m=seam['max_jump_error_m'],
                        shared_halo_max_m=seam['overlap_max_m'], error_colour_scale_m=scale,
                        colour_clipped_pct=float(100*np.mean(np.abs(delta)>scale)),
                        **{f'psd_band_{i+1}_ratio': v for i, v in enumerate(bands)}))
                    draw.text((x, 1815), f"MAE {error['mae']:.2f} m · côte {100*error['coast']:.2f} %", font=font, fill='#17242e')
                    draw.text((x, 1845), f"Pente ×{slope:.2f} · joint max {seam['max_m']:.3g} m", font=small, fill='#334957')
            draw.text((margin, 1815), 'Bleu : plus bas · rouge : plus haut', font=small, fill='#334957')
            draw.text((margin, 1845), 'Couleurs saturées au-delà de l’échelle.', font=small, fill='#334957')
            filename=f'{name}-lod{lod}.png'
            page.save(output/filename)
            pages.append(page)
            links.append((heading, filename))
    buffer=io.StringIO()
    writer=csv.DictWriter(buffer, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    atomic_write(output/'metrics-512.csv', lambda f:f.write(buffer.getvalue().encode('utf-8')))
    atomic_json(output/'manifest.json', dict(candidates=[dict(label=c[0], **b[2]) for c,b in zip(candidates,banks)],
        native_pixels=512, detail_crop_pixels=256, detail_display_zoom=2,
        physical_reference='Four independent cold viewer tiles', accepted=False))
    pages[0].save(output/'comparisons.pdf', save_all=True, append_images=pages[1:], resolution=150)
    content=''.join(f'<li><a href="{html.escape(f)}">{html.escape(h)}</a></li>' for h,f in links)
    page='''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Comparaisons finales</title><style>body{font:18px system-ui;max-width:950px;margin:40px auto;padding:0 20px;line-height:1.6}a{color:#175c85}</style>
<h1>Terrains contrastés : référence et candidats</h1><p>Chaque planche contient le relief natif, un détail au même endroit et l’erreur en mètres.
La référence, la lumière et l’échelle d’erreur sont identiques pour toutes les colonnes d’une planche.
Ces images proviennent des contrôles de raccords : quatre tuiles calculées indépendamment.</p>
<p><a href="comparisons.pdf">Toutes les planches (PDF)</a> · <a href="metrics-512.csv">Métriques par planche (CSV)</a> · <a href="manifest.json">Poids et provenance</a></p><ul>'''+content+'</ul></html>'
    atomic_write(output/'index.html', lambda f:f.write(page.encode('utf-8')))
    return dict(plates=len(pages), measurements=len(rows), output=str(output), accepted=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', nargs=2, action='append', required=True, metavar=('LABEL','SEAM_DIR'))
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    print(json.dumps(build(args.candidate, args.output)))


if __name__=='__main__':
    main()

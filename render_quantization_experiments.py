"""Render measured offline quantization DEM arrays; CPU-only, no synthetic art."""
import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

from render_inference_comparisons import render_case, digest
from terrain_climate import colorize, expand_climate


def render_climate(ref, ref_climate, candidate, candidate_climate, variant, output):
    c, rc = expand_climate(candidate_climate,candidate.shape), expand_climate(ref_climate,ref.shape)
    temp, baseline = c[0]+c[4]*np.maximum(candidate,0), rc[0]+rc[4]*np.maximum(ref,0)
    delta = temp-baseline
    lo,hi = min(float(temp.min()),float(baseline.min())),max(float(temp.max()),float(baseline.max()))
    fig,axes = plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    axes[0,0].imshow(baseline,vmin=lo,vmax=hi,cmap='coolwarm')
    axes[0,0].set_title('Température affichée BF16 (°C)')
    view=axes[0,1].imshow(temp,vmin=lo,vmax=hi,cmap='coolwarm')
    axes[0,1].set_title('Température candidat (°C)')
    fig.colorbar(view,ax=axes[0,:2],shrink=.7)
    limit=max(.01,float(np.abs(delta).max()))
    view=axes[0,2].imshow(delta,vmin=-limit,vmax=limit,cmap='RdBu_r')
    axes[0,2].set_title(f'Écart : max {float(np.abs(delta).max()):.4g} °C')
    fig.colorbar(view,ax=axes[0,2],shrink=.7)
    axes[1,0].imshow(colorize(ref,ref_climate,'biomes'))
    axes[1,0].set_title('Palette biomes BF16 (heuristique existante)')
    axes[1,1].imshow(colorize(candidate,candidate_climate,'biomes'))
    axes[1,1].set_title('Palette biomes candidat')
    axes[1,2].imshow((ref>0)!=(candidate>0),cmap='gray',vmin=0,vmax=1)
    axes[1,2].set_title('Changement terre / océan (blanc)')
    fig.suptitle(f'{variant} · seed101 · LOD3 · mêmes champs climat coarse\nTempérature reconstruite : BIO1 niveau mer + pente thermique × max(DEM,0)')
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.savefig(output/'lod3_case0-climate.png',dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--climate-only', action='store_true')
    args = parser.parse_args()
    source, output = args.input.resolve(), args.output.resolve()
    report = json.loads((source/'physical.json').read_text(encoding='utf-8'))
    if report['state'] != 'complete' or report['runtime_enabled'] is not False:
        raise ValueError('Expected completed isolated offline measurement')
    with np.load(source/'physical.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k].astype(np.float64) for k in archive.files}
    all_elev = [v for k,v in arrays.items() if k.endswith('_elev')]
    norm = TwoSlopeNorm(vmin=min(-1., min(float(x.min()) for x in all_elev)), vcenter=0.,
                        vmax=max(1., max(float(x.max()) for x in all_elev)))
    cmap = plt.get_cmap('terrain')
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for variant, results in report['variants'].items():
        if variant == 'reference':
            continue
        directory = output/variant
        directory.mkdir(exist_ok=True)
        render_climate(arrays['reference_0_elev'],arrays['reference_0_climate'],
                       arrays[f'{variant}_0_elev'],arrays[f'{variant}_0_climate'],variant,directory)
        if args.climate_only:
            continue
        for index, item in enumerate(results):
            ref, actual = arrays[f'reference_{index}_elev'], arrays[f'{variant}_{index}_elev']
            key = f'lod3_case{index}'
            case = render_case(key, ref, actual, item['site'], directory, norm, cmap,
                               max(1., float(np.max(np.abs(actual-ref)))), 45.)
            entries.append(dict(variant=variant, index=index, site=item['site'],
                                gate_passed=item['physical_gate_passed'],
                                metrics=item['elevation_error_m'], case=case))
    if args.climate_only:
        print(json.dumps(dict(state='complete',climate_figures=len(report['variants'])-1)))
        return
    evidence = dict(source=str(source), source_npz_sha=digest(source/'physical.npz'),
                    source_receipt_sha=digest(source/'physical.json'), entries=entries)
    (output/'comparison.json').write_text(json.dumps(evidence, indent=2), encoding='utf-8')
    lines = ['<!doctype html><html lang="fr"><meta charset="utf-8"><title>Quantification terrain — mesures</title>',
             '<style>body{font:16px system-ui;max-width:1500px;margin:30px auto;padding:12px;background:#101720;color:#e8edf4}img{max-width:100%}a{color:#8dccff}</style>',
             '<h1>Quantification du modèle latent : mesures offline</h1>',
             '<p>Vrais tableaux DEM à 240 m/pixel. Même échelle d’altitude par galerie ; écart et pente identifiés par figure. Aucun candidat activé dans le serveur. Les images ne remplacent pas le gate numérique de 1 m.</p>']
    for entry in entries:
        variant, index = entry['variant'], entry['index']
        lines += [f'<h2>{html.escape(variant)} · seed {entry["site"]["seed"]} · gate {entry["gate_passed"]}</h2>',
                  f'<p>Max {entry["metrics"]["max_abs"]:.6g} m ; MAE {entry["metrics"]["mae"]:.6g} m.</p>',
                  f'<img loading="lazy" src="{variant}/lod3_case{index}-overview.png">',
                  f'<p><a href="{variant}/lod3_case{index}-zooms.png">Détails</a> · <a href="{variant}/lod3_case{index}-profiles.png">Profils</a></p>']
        if index == 0:
            lines.append(f'<p><a href="{variant}/lod3_case0-climate.png">Température reconstruite et palette biomes</a></p>')
    lines.append('</html>')
    (output/'index.html').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(dict(entries=len(entries), output=str(output))), flush=True)


if __name__ == '__main__':
    main()

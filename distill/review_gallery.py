"""Build an offline reference/student wipe gallery with physical metric tables.

Source reports and arrays stay immutable. The gallery is a review artifact,
not a statement that a candidate passed quality or continuity checks.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from distill.common import atomic_json, atomic_write, external_path
from distill.check_physical_seams import shade


PAGE = '''<!doctype html>
<html lang="fr"><meta charset="utf-8"><title>Distillation — comparaison visuelle</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
body{font:16px system-ui;background:#182128;color:#eee;margin:24px auto;max-width:1050px;padding:0 16px}
select,button,input{font:inherit} select{max-width:100%;margin:6px 8px 6px 0;padding:6px}
button{padding:6px 14px} a{color:#9dd8ff} #viewer{position:relative;width:min(100%,750px);aspect-ratio:1;margin:auto;background:#333}
#viewer img{position:absolute;inset:0;width:100%;height:100%;image-rendering:auto}
#candidate{clip-path:inset(0 50% 0 0)} #divider{position:absolute;top:0;bottom:0;left:50%;width:2px;background:white}
#slider{display:block;width:min(100%,750px);margin:14px auto} .muted{color:#c0ccd3} table{border-collapse:collapse;margin:16px 0;width:100%}
td,th{text-align:left;border-bottom:1px solid #4a5862;padding:7px} #metrics{overflow-x:auto} #metrics table{min-width:650px}
</style>
<h1>Référence / élève</h1>
<p>Relief ombré à la même échelle pour les deux images. Déplace le curseur :
0 % montre la référence, 100 % l’élève. Les côtes, les petits reliefs et les lignes
alignées se comparent au même endroit. Le rendu de l’application doit aussi être jugé.</p>
<label>Comparaison <select id="group"></select></label><br>
<label>Lieu et niveau <select id="view"></select></label>
<button id="previous" aria-label="Vue précédente">←</button> <button id="next" aria-label="Vue suivante">→</button>
<p id="context" class="muted"></p>
<p>Élève à gauche du curseur, référence à droite.</p>
<div id="viewer"><img id="reference" alt=""><img id="candidate" alt=""><div id="divider"></div></div>
<label for="slider">Part de l’élève : <output id="amount">50 %</output></label>
<input id="slider" type="range" min="0" max="100" value="50">
<div id="metrics"></div>
<p class="muted">Les mesures décrivent cette vue ; elles ne constituent pas une acceptation.
La continuité entre tuiles est vérifiée dans des rapports séparés.</p>
<p>Sources et empreintes : <a href="manifest.json">manifest.json</a></p>
<script>
const groups=__DATA__, group=document.getElementById('group'), view=document.getElementById('view');
const slider=document.getElementById('slider');
function wipe(){document.getElementById('candidate').style.clipPath=`inset(0 ${100-slider.value}% 0 0)`;
 document.getElementById('divider').style.left=slider.value+'%';document.getElementById('amount').textContent=slider.value+' %';}
function show(){const g=groups[+group.value],v=g.views[+view.value];
 document.getElementById('reference').src=v.reference;document.getElementById('reference').alt='Référence — '+v.label;
 document.getElementById('candidate').src=v.candidate;document.getElementById('candidate').alt='Élève — '+v.label;
 document.getElementById('context').textContent=v.context;document.getElementById('metrics').innerHTML=v.table;wipe();}
function chooseGroup(){view.replaceChildren();groups[+group.value].views.forEach((v,i)=>{
 const o=new Option(v.label,String(i));view.add(o);});show();}
groups.forEach((g,i)=>group.add(new Option(g.label,String(i))));
group.onchange=chooseGroup;view.onchange=show;slider.oninput=wipe;
function move(delta){view.selectedIndex=(view.selectedIndex+delta+view.options.length)%view.options.length;show();}
document.getElementById('previous').onclick=()=>move(-1);document.getElementById('next').onclick=()=>move(1);
chooseGroup();
</script></html>'''


def table(row, reference):
    error = row['vs_reference']
    ratios = [a/b if b else None for a, b in zip(row['psd'], reference['psd'])]
    slope = row['slope_mean']/reference['slope_mean'] if reference['slope_mean'] else None
    values = [('Erreur moyenne (m)', error['mae']), ('Erreur maximale (m)', error['max']),
              ('Pente moyenne / référence', slope)]
    values += [(f'Puissance / référence — bande {i+1}', v) for i, v in enumerate(ratios)]
    return '<table><tr><th>Mesure</th><th>Valeur</th></tr>' + ''.join(
        '<tr><td>'+html.escape(name)+'</td><td>'+('—' if value is None else f'{value:.4g}')+'</td></tr>'
        for name, value in values) + '</table>'


def build(directories, output):
    output = external_path(output)
    output.mkdir(parents=True, exist_ok=True)
    groups, manifests = [], []
    for directory in directories:
        directory = Path(directory).resolve()
        payload = (directory/'report.json').read_bytes()
        report = json.loads(payload)
        reference = {(r['site'], r['lod']):r for r in report['variants']['reference']}
        site_info = {s['name']:s for s in report['sites']}
        manifest = dict(directory=str(directory), report_sha256=hashlib.sha256(payload).hexdigest(),
                        arrays_sha256=hashlib.sha256((directory/'arrays.npz').read_bytes()).hexdigest(),
                        checkpoint_digests=report['checkpoint_digests'], source_digests=report['source_digests'])
        manifests.append(manifest)
        with np.load(directory/'arrays.npz', allow_pickle=False) as arrays:
            for variant, rows in report['variants'].items():
                if not variant.startswith('student'):
                    continue
                stage_label = {'student': 'Base', 'student_coarse': 'Coarse',
                               'student_decoder': 'Decoder', 'student_all': 'Trois élèves ensemble'}[variant]
                group = dict(label=stage_label+' · '+directory.name, views=[])
                for row in rows:
                    site, lod = row['site'], row['lod']
                    if 'vs_reference' not in row:
                        raise ValueError('Candidate has no physical comparison metrics.')
                    images = {}
                    for which in ('reference', variant):
                        key = f'{which}|{site}|{lod}'
                        image_name = hashlib.sha256((str(directory)+'|'+key+'|'+manifest['arrays_sha256']).encode()).hexdigest()[:24]+'.png'
                        Image.fromarray(shade(arrays[key], lod)).save(output/image_name)
                        images[which] = image_name
                    info = site_info[site]
                    context = f"Seed {info['seed']} · {info['profile']} · LOD {lod} · {30*2**lod} m/pixel"
                    climate = info.get('conditioning', {})
                    if 'temp' in climate and 'rain' in climate:
                        context += f" · {climate['temp']:.1f} °C · {climate['rain']:.0f} mm/an"
                    group['views'].append(dict(label=f'{site} · LOD {lod}', context=context,
                        reference=images['reference'], candidate=images[variant], table=table(row, reference[(site,lod)])))
                if group['views']:
                    groups.append(group)
    if not groups:
        raise ValueError('No student comparisons found.')
    data = json.dumps(groups, ensure_ascii=False).replace('<', '\\u003c')
    atomic_write(output/'index.html', lambda f: f.write(PAGE.replace('__DATA__', data).encode()))
    atomic_json(output/'manifest.json', dict(reports=manifests, accepted=False, kind='Offline visual review'))
    return sum(len(g['views']) for g in groups)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(dict(views=build(args.report_dir, args.output), output=str(args.output))))


if __name__ == '__main__':
    main()

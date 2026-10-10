"""Assemble final physical comparisons, speed tables and preserved EMA exports.

Never associates a speed measurement with different weights or inference code.
Quality CSVs describe all historical and rare cases; plates use larger native
512px footprints from the independent cold-tile checks.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path
import shutil

import numpy as np

from distill.common import atomic_json, atomic_write, external_path
from distill.final_plates import build as plates
from distill.rare_cases import local_errors
from distill.review_gallery import build as gallery


def save_csv(path, rows):
    stream=io.StringIO()
    writer=csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    atomic_write(path, lambda f:f.write(stream.getvalue().encode('utf-8')))


def read_quality(label, directory, seam):
    directory=Path(directory)
    report=json.loads((directory/'report.json').read_text())
    digests=report['checkpoint_digests']['student_all']
    if digests != seam['checkpoint_digests']:
        raise ValueError('Physical and cold-tile evidence use different weights/code.')
    references={(r['site'], r['lod']):r for r in report['variants']['reference']}
    floors={(r['site'], r['lod']):r for r in report['variants']['fp32base']}
    actual={(r['site'], r['lod']) for r in report['variants']['student_all']}
    expected={(s['name'], lod) for s in report['sites'] for lod in (0,3)}
    if actual != expected or set(references) != expected or set(floors) != expected:
        raise ValueError('The final quality bank must be complete.')
    sites={s['name']:s for s in report['sites']}
    result=[]
    with np.load(directory/'arrays.npz', allow_pickle=False) as arrays:
        for row in report['variants']['student_all']:
            name,lod=row['site'],row['lod']
            ref, floor=references[(name,lod)],floors[(name,lod)]
            candidate,reference=arrays[f'student_all|{name}|{lod}'],arrays[f'reference|{name}|{lod}']
            if candidate.shape != reference.shape or not np.isfinite(candidate).all() or not np.isfinite(reference).all():
                raise ValueError('Invalid physical comparison arrays.')
            local=local_errors(candidate,reference,lod)
            error=row['vs_reference']
            result.append(dict(candidate=label, site=name, kind=sites[name].get('kind','historical'),
                profile=sites[name]['profile'], seed=sites[name]['seed'], lod=lod,
                mae_m=error['mae'], rmse_m=error['rmse'], p99_m=error['p99'], max_m=error['max'],
                bf16_fp32_mae_m=floor['vs_reference']['mae'], coast_sign_disagreement_pct=100*error['coast'],
                slope_ratio=row['slope_mean']/ref['slope_mean'] if ref['slope_mean'] else None,
                land_mae_m=local['land']['mae_m'], low_land_0_20m_mae_m=local['low_land_0_20m']['mae_m'],
                coast_300m_mae_m=local['coast_300m']['mae_m'],
                **{f'psd_band_{i+1}_ratio':a/b if b else None for i,(a,b) in enumerate(zip(row['psd'],ref['psd']))}))
    return result


def read_speed(label, directory, shared, seam):
    result=[]
    for stage in ('base','coarse','decoder'):
        root=Path(directory) if stage=='base' else Path(shared)
        for gpu in (0,1):
            path=root/f'{stage}-stage-gpu{gpu}.json'
            report=json.loads(path.read_text())
            expected={k:v for k,v in seam['checkpoint_digests'].items() if k.startswith('code:')}
            if report['checkpoint_digest'] != seam['checkpoint_digests'][stage] or report['student_source_digests'] != expected:
                raise ValueError('Speed evidence uses different weights or inference code.')
            if report.get('status') != 'complete' or report.get('repeats',0) < 3 or not all(report.get(k) for k in ('whole_system_idle','fresh_world_per_sample','includes_feature_construction','includes_transfers')):
                raise ValueError('Final speed evidence must include orchestration on an idle system.')
            for comparison in report['comparisons']:
                result.append(dict(candidate=label, gpu=report['gpu'], stage=stage, pixels_per_side=comparison['size'],
                    teacher_seconds=comparison['teacher_seconds'], student_seconds=comparison['student_seconds'],
                    speedup=comparison['speedup'], checkpoint_sha256=report['checkpoint_digest'],
                    report_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return result


def build(candidates, shared, output, additional=None):
    output=external_path(output)
    output.mkdir(parents=True,exist_ok=True)
    quality,speed,sizes,sources=[],[],[],[]
    galleries=[]
    for index,(label,seam_dir,prefix,bench_dir) in enumerate(candidates):
        seam=json.loads((Path(seam_dir)/'report.json').read_text())
        evidence=output/'evidence'/f'candidate-{index+1}'
        evidence.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path(seam_dir)/'report.json',evidence/'cold-tiles.json')
        for suffix in ('','-rare'):
            directory=Path(str(prefix)+suffix)
            quality.extend(read_quality(label,directory,seam))
            galleries.append(directory)
            shutil.copy2(directory/'report.json',evidence/('quality-rare.json' if suffix else 'quality-historical.json'))
            if (directory/'optimization-equivalence.json').exists():
                shutil.copy2(directory/'optimization-equivalence.json',evidence/('equivalence-rare.json' if suffix else 'equivalence-historical.json'))
        speed.extend(read_speed(label,bench_dir,shared,seam))
        for stage in ('base','coarse','decoder'):
            root=Path(bench_dir) if stage=='base' else Path(shared)
            for gpu in (0,1):
                shutil.copy2(root/f'{stage}-stage-gpu{gpu}.json',evidence/f'{stage}-stage-gpu{gpu}.json')
        if additional is not None:
            if len(additional)!=len(candidates):
                raise ValueError('One additional cold-tile bank per candidate is required.')
            shutil.copy2(Path(additional[index])/'report.json',evidence/'cold-tiles-historical.json')
        weights=Path(bench_dir)/'weights'
        manifest=json.loads((weights/'manifest.json').read_text())
        if any(manifest['sources'][stage]['sha256'] != seam['checkpoint_digests'][stage] for stage in ('base','coarse','decoder')):
            raise ValueError('Exported weights differ from the reviewed candidate.')
        target=output/'weights'/f'candidate-{index+1}'
        target.mkdir(parents=True,exist_ok=True)
        for stage,artifact in manifest['exports'].items():
            path=weights/artifact['file']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=artifact['sha256']:
                raise ValueError('Exported weight bytes changed.')
            shutil.copy2(path,target/path.name)
            sizes.append(dict(candidate=label,stage=stage,step=manifest['sources'][stage]['step'],
                weight_elements=artifact['tensor_elements'],fp32_bytes=artifact['bytes'],fp32_mb=artifact['bytes']/1e6,
                path=str(target.relative_to(output)/path.name),sha256=artifact['sha256']))
        shutil.copy2(weights/'manifest.json',target/'manifest.json')
        sources.append(dict(label=label,seam=str(seam_dir),quality=str(prefix),benchmark=str(bench_dir),digests=seam['checkpoint_digests'],
            historical_cold_tiles=str(additional[index]) if additional is not None else None,
            evidence=str(evidence.relative_to(output))))
    save_csv(output/'quality-all-sites.csv',quality)
    save_csv(output/'performance.csv',speed)
    save_csv(output/'model-sizes.csv',sizes)
    plates([(c[0],c[1]) for c in candidates],output/'plates',additional)
    count=gallery(galleries,output/'gallery')
    atomic_json(output/'manifest.json',dict(candidates=sources,gallery_views=count,accepted=False,
        performance_scope='Each neural stage separately; dependencies prefetched. Not end-to-end viewer latency.',
        quality_scope='All historical and rare viewer tiles; 512px plates have their own metrics.'))
    table='<table><tr><th>Candidat</th><th>GPU</th><th>Étape</th><th>Taille</th><th>Référence (s)</th><th>Élève (s)</th><th>Gain</th></tr>'
    for r in speed:
        table+='<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in [r['candidate'],r['gpu'],r['stage'],r['pixels_per_side'],
            f"{r['teacher_seconds']:.3f}",f"{r['student_seconds']:.3f}",f"×{r['speedup']:.2f}"])+'</tr>'
    table+='</table>'
    page='''<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Distillation : éléments de décision</title><style>body{font:17px system-ui;max-width:1200px;margin:40px auto;padding:0 20px;line-height:1.5}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}table{border-collapse:collapse}a{color:#175c85}.scroll{overflow:auto}</style>
<h1>Distillation : comparer les compromis</h1><p>Les quatre candidats sont conservés avec les mêmes élèves coarse et decoder.
La sélection pratique dépend du rendu ; les critères stricts initiaux restent des diagnostics.</p>
<p><a href="plates/index.html">Planches contrastées, PNG et PDF</a> · <a href="gallery/index.html">Comparateur interactif : tous les sites</a></p>
<p><a href="quality-all-sites.csv">Qualité, cas par cas</a> · <a href="performance.csv">Vitesse</a> · <a href="model-sizes.csv">Poids et taille des modèles</a> · <a href="manifest.json">Provenance</a></p>
<h2>Vitesse mesurée</h2><p>Chaque étape est chronométrée séparément, sur des champs neufs, construction des entrées et transferts inclus.
Les dépendances sont préchargées, les poids résidents et le warmup exclu. Médiane de trois répétitions, deux GPU libres.
Ces gains ne sont pas une mesure du temps d’affichage complet.</p><div class="scroll">'''+table+'</div></html>'
    atomic_write(output/'index.html',lambda f:f.write(page.encode('utf-8')))
    return dict(output=str(output),quality_rows=len(quality),speed_rows=len(speed),gallery_views=count,accepted=False)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',nargs=4,action='append',required=True,metavar=('LABEL','SEAM_DIR','QUALITY_PREFIX','BENCH_DIR'))
    parser.add_argument('--shared-bench',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--additional-seam-dir',nargs='+',type=Path)
    args=parser.parse_args()
    print(json.dumps(build(args.candidate,args.shared_bench,args.output,args.additional_seam_dir)))


if __name__=='__main__':
    main()

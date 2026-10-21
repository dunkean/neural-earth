"""Evaluate every packed native checkpoint;brief scores every two checkpoints."""
import argparse
import json
from pathlib import Path
import statistics
import signal
import subprocess
import sys
import time

import numpy as np

from distill.common import DATA, atomic_json, atomic_write, external_path
from distill.jobs import state
from distill.tile40_climate_run import terrain_summary
from distill.tile40_conditioning import CLIMATES


def physical_summary(report,directory):
    summary=terrain_summary(report)
    summary['height_mae_m']=statistics.mean(row['metrics']['height_mae_m'] for row in report['rows'])
    summary['mountain_band_correlation']={climate:statistics.mean(
        row['paired_spatial_band_correlation'] for row in report['rows']
        if row['case']['relief']=='mountains' and row['case']['climate']==climate) for climate in CLIMATES}
    mountains=[row for row in report['rows'] if row['case']['relief']=='mountains']
    maps={}
    for row in mountains:
        case=row['case']
        with np.load(Path(directory)/case['file'],allow_pickle=False) as values:
            maps[(case['seed'],case['climate'])]=(values['height_m'].astype(np.float64),values['reference_height_m'].astype(np.float64))
    responses={}
    for climate in CLIMATES:
        if climate=='temperate':continue
        rows=[]
        for seed in sorted({row['case']['seed'] for row in mountains}):
            actual,reference=maps[(seed,climate)]
            temperate,target=maps[(seed,'temperate')]
            a,b=actual-temperate,reference-target
            denominator=max(float(np.mean(b*b)),1.)
            ac,bc=a-a.mean(),b-b.mean()
            norm=float(np.linalg.norm(ac)*np.linalg.norm(bc))
            rows.append(dict(relative_rmse=float(np.sqrt(np.mean((a-b)**2)/denominator)),
                correlation=float(np.sum(ac*bc)/norm) if norm>1e-8 else 0.))
        responses[climate]={key:statistics.mean(row[key] for row in rows) for key in rows[0]}
    summary['climate_height_response']=responses
    summary['quality_gap']=(summary['groups']['mountains']+
        .25*(1-min(summary['mountain_band_correlation'].values()))+
        .25*statistics.mean(row['relative_rmse'] for row in responses.values()))
    # Numeric proximity to the original,not automatic visual acceptance.
    summary['near_original_numeric']=(summary['groups']['mountains']<.05 and
        all(value>=.95 for value in summary['mountain_band_correlation'].values()) and
        all(value['climate_spectral_response_error']<=.025 for value in summary['mountain_climates'].values()) and
        all(value['relative_rmse']<=.35 and value['correlation']>=.9 for value in responses.values()))
    return summary


def render(history,original,through):
    candidates=[row for row in history if row['step']<=through]
    best=min(candidates,key=lambda row:row['summary']['quality_gap'])
    recent=[row for row in candidates if row['step']>max(0,through-2000)]
    lines=[f'Base native comprimée — point à {through} pas',
        'Plus bas=mieux pour distances et erreurs ; plus haut=mieux pour corrélations.',
        f"Original répété : global {original['score']:.4f}, montagnes {original['groups']['mountains']:.4f}.",
        'Meilleur ancien distillé48/brut8k : global0.1654,montagnes0.1004 (décodeur compact).',
        'La nouvelle base conserve le décodeur original ; les comparaisons portent sur les terrains complets.',
        'Pas | Poids | Global | Montagnes | MAE mètres | Écart qualité | Proximité numérique original']
    for row in recent:
        s=row['summary'];lines.append(f"{row['step']} | {row['selection']} | {s['score']:.4f} | {s['groups']['mountains']:.4f} | {s['height_mae_m']:.2f} | {s['quality_gap']:.4f} | {s['near_original_numeric']}")
    lines += ['',f"Meilleur écart observé : {best['step']} pas,{best['selection']},base {best['bytes']/1e6:.2f}Mo.",
        'Climat | Distance montagnes | Corrélation spatiale détails | Erreur réponse spectrale | RMSE relative de réponse / corrélation']
    s=best['summary']
    for climate,values in s['mountain_climates'].items():
        response=s['climate_height_response'].get(climate)
        suffix=f"{response['relative_rmse']:.3f}/{response['correlation']:.3f}" if response else 'référence tempérée'
        lines.append(f"{climate} | {values['morphology_distance']:.4f} | {s['mountain_band_correlation'][climate]:.3f} | {values['climate_spectral_response_error']:.4f} | {suffix}")
    lines += ['','Validation visuelle requise ; aucun remplacement automatique des modèles distribués.']
    return '\n'.join(lines)+'\n'


def briefing_marks(history,terminal=False):
    complete=sorted(step for step in {r['step'] for r in history} if step>0 and
        {r['selection'] for r in history if r['step']==step}=={'raw','ema'})
    marks=complete[1::2]
    if terminal and complete and complete[-1] not in marks:marks.append(complete[-1])
    return marks


def persist_briefings(history,original,summaries,terminal=False):
    for mark in briefing_marks(history,terminal):
        path=summaries/f'point-{mark:06d}.json'
        if path.exists():continue
        retained=[r for r in history if r['step']<=mark]
        brief=dict(through_step=mark,history=retained,original_control=original,
            best_observed=min(retained,key=lambda row:row['summary']['quality_gap']),quality_accepted=False)
        content=render(retained,original,mark)
        atomic_json(path,brief)
        atomic_json(summaries/'latest.json',brief)
        atomic_write(path.with_suffix('.txt'),lambda handle:handle.write(content.encode()))
        atomic_write(summaries/'latest.txt',lambda handle:handle.write(content.encode()))
        print(content,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--training-job',required=True)
    parser.add_argument('--original-control',type=Path,required=True)
    parser.add_argument('--initial-evaluation',type=Path)
    parser.add_argument('--dataset',type=Path,default=DATA/'tile40-20261014/reference-coarse2-guided-expanded-v5')
    args=parser.parse_args()
    root=external_path(args.root)
    root.mkdir(parents=True,exist_ok=True)
    original_report=json.loads((args.original_control/'report.json').read_text())
    if original_report['status']!='complete':raise ValueError('Complete original repeat required.')
    original=physical_summary(original_report,args.original_control)
    history_path=root/'native-selection.json'
    history=json.loads(history_path.read_text())['history'] if history_path.exists() else []
    seen={(row['step'],row['selection']) for row in history}
    summaries=root/'score-reports';summaries.mkdir(exist_ok=True)
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    while not stopped:
        training=state(DATA/'jobs'/args.training_job)
        if training['status']=='missing':raise ValueError('Training job is missing.')
        paths=sorted((root/'packed').glob('step-*/manifest.json'))
        for path in paths:
            if stopped:break
            metadata=json.loads(path.read_text());key=(metadata['step'],metadata['selection'])
            if key in seen:continue
            output=args.initial_evaluation if key[0]==0 and args.initial_evaluation else root/f"val-{key[0]:06d}-{key[1]}"
            if not (output/'report.json').exists():
                if output.exists():raise ValueError('Incomplete existing physical evaluation;inspect before retry.')
                atomic_json(root/'evaluation-status.json',dict(status='evaluating',step=key[0],selection=key[1]))
                subprocess.run([sys.executable,'-u','-m','distill.tile40_native3_evaluate','--dataset',str(args.dataset),
                    '--packed',str(path.parent),'--output',str(output),'--limit','30'],check=True)
            report=json.loads((output/'report.json').read_text())
            if report['status']!='complete':raise ValueError('Incomplete physical evaluation.')
            if report['base_sha256']!=metadata['sha256']:
                raise ValueError('Physical evaluation uses different packed weights.')
            for field in ('dataset_manifest_sha256','coarse_sha256','decoder_sha256'):
                if report[field]!=original_report[field]:
                    raise ValueError(f'Unmatched original control:{field}')
            summary=physical_summary(report,output)
            row=dict(step=key[0],selection=key[1],summary=summary,bytes=report['packed_bytes'],report=str(output/'report.json'),packed=str(path.parent))
            history.append(row);seen.add(key)
            atomic_json(history_path,dict(history=history,original_control=original,
                best_observed=min(history,key=lambda row:row['summary']['quality_gap']),quality_accepted=False))
            persist_briefings(history,original,summaries)
        terminal=not training['live'] and training['status'] in ('complete','failed')
        remaining=any((json.loads(p.read_text())['step'],json.loads(p.read_text())['selection']) not in seen for p in (root/'packed').glob('step-*/manifest.json'))
        if terminal and not remaining:
            persist_briefings(history,original,summaries,terminal=True)
            atomic_json(root/'evaluation-status.json',dict(status='complete' if training['status']=='complete' else 'training_failed',evaluated=len(history),quality_accepted=False))
            return
        atomic_json(root/'evaluation-status.json',dict(status='waiting_snapshot',evaluated=len(history),training_status=training['status']))
        time.sleep(10)
    atomic_json(root/'evaluation-status.json',dict(status='interrupted',evaluated=len(history),quality_accepted=False))


if __name__=='__main__':main()

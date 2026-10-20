"""Persist score briefings every two fully evaluated checkpoints,without GPUs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import time

from distill.common import atomic_json, atomic_write, external_path


CLIMATE_LABELS = dict(arid='aride',cold='froid',tropical='tropical',temperate='tempéré',oceanic='océanique')


def completed_steps(history):
    versions = {}
    for entry in history:
        versions.setdefault(entry['step'],set()).add(entry['weights'])
    return sorted(step for step,weights in versions.items() if weights=={'raw','ema'})


def relative_gain(value, reference):
    return 100*(1-value/reference) if reference>0 else None


def briefing(history, baseline, through_step, after_step=0):
    complete = set(completed_steps(history))
    retained = [row for row in history if row['step'] in complete and row['step']<=through_step]
    new = [row for row in retained if row['step']>after_step]
    if not new or through_step not in complete:
        raise ValueError('A briefing requires new checkpoints with both raw and EMA evaluation complete.')
    rows = [dict(step=row['step'],weights=row['weights'],global_score=row['summary']['score'],
        mountain_score=row['summary']['groups']['mountains'],
        global_gain_percent=relative_gain(row['summary']['score'],baseline['score']),
        mountain_gain_percent=relative_gain(row['summary']['groups']['mountains'],baseline['groups']['mountains']),
        admissible_vs_baseline=row['admissible_vs_baseline'],report=row['report'])
        for row in sorted(new,key=lambda row:(row['step'],row['weights']))]
    eligible = [row for row in retained if row['admissible_vs_baseline']]
    best = min(eligible,key=lambda row:row['summary']['groups']['mountains']) if eligible else None
    latest = min((row for row in retained if row['step']==through_step),
        key=lambda row:(not row['admissible_vs_baseline'],row['summary']['groups']['mountains']))
    climates = [dict(climate=climate,score=values['morphology_distance'],
        baseline_score=baseline['mountain_climates'][climate]['morphology_distance'],
        gain_percent=relative_gain(values['morphology_distance'],baseline['mountain_climates'][climate]['morphology_distance']),
        response_error=values['climate_spectral_response_error'],
        baseline_response_error=baseline['mountain_climates'][climate]['climate_spectral_response_error'])
        for climate,values in latest['summary']['mountain_climates'].items()]
    return dict(through_step=through_step,after_step=after_step,rows=rows,
        baseline=dict(global_score=baseline['score'],mountain_score=baseline['groups']['mountains']),
        latest_selection=dict(step=latest['step'],weights=latest['weights']),mountain_climates=climates,
        best_admissible=best,quality_accepted=False,
        metric='Spectral amplitude plus directional coherence distance;lower is better;positive percentage means improvement.',
        scope='Decoded VAL terrain diagnostics,not training loss or proof of visual quality.')


def render(report):
    base = report['baseline']
    lines = [f"Point scores — checkpoints jusqu'à {report['through_step']:,} pas".replace(',',' '),
        'Distance spectrale + directionnelle : plus bas = mieux. Gain positif = amélioration.',
        f"Meilleur distillé existant48/brut8k : global {base['global_score']:.4f} ; montagnes {base['mountain_score']:.4f}.",
        '', 'Pas | Poids | Global | Montagnes | Gain montagnes | Admissible / référence']
    for row in report['rows']:
        lines.append(f"{row['step']} | {row['weights']} | {row['global_score']:.4f} | {row['mountain_score']:.4f} | "
            f"{row['mountain_gain_percent']:+.1f} % | {'oui' if row['admissible_vs_baseline'] else 'non'}")
    selected=report['latest_selection']
    lines += ['',f"Montagnes par climat — {selected['step']} pas, {selected['weights']} :",
        'Climat | Score | Référence | Gain | Erreur de réponse climatique / référence']
    for row in report['mountain_climates']:
        lines.append(f"{CLIMATE_LABELS[row['climate']]} | {row['score']:.4f} | {row['baseline_score']:.4f} | "
            f"{row['gain_percent']:+.1f} % | {row['response_error']:.4f} / {row['baseline_response_error']:.4f}")
    best = report['best_admissible']
    lines += ['',f"Meilleur candidat admissible : {best['step']} pas, {best['weights']}." if best else
        'Aucun candidat admissible pour le moment ; la référence48/8k reste retenue.',
        'Diagnostics sur les terrains VAL décodés ; la qualité visuelle reste à examiner.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--every',type=int,default=2)
    parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    if args.every<1:
        raise ValueError('A positive checkpoint cadence is required.')
    root=external_path(args.root)
    request=json.loads((root/'request.json').read_text())
    output=root/'score-reports'
    stopped=False
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    for signum in (signal.SIGTERM,signal.SIGINT):
        signal.signal(signum,stop)
    while not stopped:
        state=json.loads((root/'status.json').read_text())
        terminal=state['status'] in ('complete','failed','interrupted')
        selection=root/'selection.json'
        history=json.loads(selection.read_text())['history'] if selection.exists() else []
        steps=completed_steps(history)
        marks=steps[args.every-1::args.every]
        if terminal and steps and steps[-1] not in marks:
            marks.append(steps[-1])
        after=0
        for mark in marks:
            path=output/f'point-{mark:06d}.json'
            if not path.exists():
                report=briefing(history,request['baseline'],mark,after)
                report.update(created_at=datetime.now(timezone.utc).isoformat(),training_status=state['status'])
                atomic_json(path,report)
                content=render(report)
                atomic_write(path.with_suffix('.txt'),lambda handle:handle.write(content.encode()))
                atomic_json(output/'latest.json',report)
                atomic_write(output/'latest.txt',lambda handle:handle.write(content.encode()))
                print(content,flush=True)
            after=mark
        atomic_json(root/'scorewatch-status.json',dict(status='complete' if terminal else 'watching',
            every_checkpoints=args.every,evaluated_steps=steps,last_briefing=marks[-1] if marks else None,
            training_status=state['status']))
        if terminal or args.once:
            return
        time.sleep(10)


if __name__=='__main__':
    main()

"""Audit directional texture in complete captured40km height comparisons."""
import argparse
import json
from pathlib import Path
import numpy as np
from distill.common import atomic_json, external_path
from distill.tile40_evaluate import directional_coherence


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=external_path(args.output)
    if output.exists():raise ValueError('Use a fresh audit path.')
    report=json.loads((args.evaluation/'report.json').read_text())
    if report['status']!='complete':raise ValueError('Complete tile evaluation required.')
    rows=[]
    for row in report['rows']:
        with np.load(args.evaluation/row['case']['file'],allow_pickle=False) as data:
            prediction,reference=data['height_m'],data['reference_height_m']
        rows.append(dict(case=row['case'],prediction=directional_coherence(prediction),reference=directional_coherence(reference)))
    groups={relief:{side:float(np.mean([r[side] for r in rows if r['case']['relief']==relief])) for side in ('prediction','reference')}
            for relief in ('mountains','hills','plain')}
    atomic_json(output,dict(status='complete',rows=rows,groups=groups,
        scope='Complete40km height fields,960m highpass then150m local gradient tensor;energy-weighted orientation coherence;diagnostic only.'))
    print(json.dumps(groups),flush=True)


if __name__=='__main__':main()

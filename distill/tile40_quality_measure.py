"""Diagnostic sampled-DEM score;images and held-out acceptance remain required."""
import argparse
import json
import math
from pathlib import Path
import statistics

from distill.common import DATA


def measure(report):
    if report['status'] != 'complete' or not report['rows']:
        raise ValueError('A complete decoded-tile evaluation is required.')
    groups = {}
    for relief in ('mountains','hills','plain'):
        rows = [r for r in report['rows'] if r['case']['relief'] == relief]
        if not rows:
            raise ValueError('All three relief groups must be represented.')
        gaps = []
        for row in rows:
            metrics = row['metrics']
            spectral = statistics.mean(abs(math.log(max(b['amplitude_ratio'],1e-6))) for b in metrics['bands'])
            coherence = metrics['directional_coherence']
            directional = abs(math.log(max(coherence['prediction'],1e-6)/max(coherence['reference'],1e-6)))
            gaps.append(spectral+directional)
        groups[relief] = statistics.mean(gaps)
    return statistics.mean(groups.values()), groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path, default=DATA/'tile40-20261014/full-tile-flow32-depth3-colored20k-val30-cpu-v1/report.json')
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    score, groups = measure(report)
    print(json.dumps(dict(report=str(args.report),cases=len(report['rows']),groups=groups,
        scope='Spectral-amplitude plus directional-coherence diagnostic;cannot prove ridge beauty,climate control,latency or held-out generalization.')))
    print(f'METRIC decoded_morphology_distance={score}')


if __name__ == '__main__':
    main()

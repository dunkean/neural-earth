"""Emit existing shoreline trial measurements without running a second GPU job.

Autoresearch records these as diagnostic measurements, not fresh repetitions
or promotions. Weight and source fingerprints must still match the evidence.
"""
import argparse
import json
import math
from pathlib import Path

from distill.common import REPO
from distill.shore_probe import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    args = parser.parse_args()
    report_path = Path(json.loads(args.selection.read_text())['report'])
    report = json.loads(report_path.read_text())
    for identity in (report['checkpoint'], report['decoder']):
        if fingerprint(identity['path']) != identity['sha256']:
            raise ValueError('Measurement weights changed; rerun validation.')
    for name, expected in report['code_sha256'].items():
        if fingerprint(REPO/'distill'/name) != expected:
            raise ValueError('Measurement source changed; rerun validation.')
    if report['windows'] != len(report['rows']) or not report['coastal_windows'] or report['accepted']:
        raise ValueError('Expected a complete, non-promotional proxy measurement.')
    for name, value in report['metrics'].items():
        if not math.isfinite(value):
            raise ValueError('Non-finite metric.')
        print(f'METRIC {name}={value:.12g}', flush=True)
    print(f'ARTIFACT shore_validation={report_path}', flush=True)
    print('Existing measurement; no fresh repetition and no physical acceptance.', flush=True)


if __name__ == '__main__':
    main()

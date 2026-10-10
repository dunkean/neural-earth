"""Measure coherent horizontal/vertical 32/64-latent patterns on LOD 3 views.

This is a diagnostic, not an acceptance gate: natural terrain can contain axial
structure. Compare the reference, candidate and error on the same physical view.
The viewer halo is excluded after high-pass filtering the complete saved field.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

from distill.common import atomic_json, external_path


def axis_grid_rms(field, periods=(32, 64)):
    field = np.asarray(field, dtype=np.float64)
    power = 0.
    for profile in (field.mean(axis=0), field.mean(axis=1)):
        n = len(profile)
        for period in periods:
            if n % period:
                raise ValueError('The analyzed core must contain complete grid periods.')
            basis = np.exp(-2j*np.pi*np.arange(n)/period)
            amplitude = 2*abs(np.mean(profile*basis))
            power += amplitude**2/2
    return float(np.sqrt(power))


def highpass_core(field):
    field = np.asarray(field, dtype=np.float64)
    if field.shape != (304, 304):
        raise ValueError('Expected a complete 256-pixel view with its 24-pixel halo.')
    return (field-gaussian_filter(field, sigma=4))[24:-24, 24:-24]


def analyze(directory):
    payload = (directory/'report.json').read_bytes()
    report = json.loads(payload)
    rows = []
    with np.load(directory/'arrays.npz', allow_pickle=False) as arrays:
        for variant, views in report['variants'].items():
            if not variant.startswith('student'):
                continue
            for view in views:
                if view['lod'] != 3:
                    continue
                key = view['site']+'|3'
                reference = highpass_core(arrays['reference|'+key])
                candidate = highpass_core(arrays[variant+'|'+key])
                rows.append(dict(report=directory.name, variant=variant, site=view['site'], lod=3,
                    reference_highpass_rms_m=float(np.sqrt(np.mean(reference**2))),
                    student_highpass_rms_m=float(np.sqrt(np.mean(candidate**2))),
                    reference_grid_rms_m=axis_grid_rms(reference),
                    student_grid_rms_m=axis_grid_rms(candidate),
                    error_grid_rms_m=axis_grid_rms(candidate-reference)))
    return dict(directory=str(directory), report_sha256=hashlib.sha256(payload).hexdigest(),
                arrays_sha256=hashlib.sha256((directory/'arrays.npz').read_bytes()).hexdigest(),
                checkpoint_digests=report['checkpoint_digests'], rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report-dir', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    reports = [analyze(path) for path in args.report_dir]
    rows = [row for report in reports for row in report['rows']]
    if not rows:
        parser.error('No complete LOD 3 student views were found.')
    atomic_json(output/'report.json', dict(reports=reports, accepted=False,
        note='Coherent axial Fourier projections at 32 and 64 latents. Diagnostic only; inspect actual terrain.'))
    with (output/'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(views=len(rows), output=str(output))), flush=True)


if __name__ == '__main__':
    main()

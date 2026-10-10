"""Verify that an inference optimization preserves a complete physical case bank."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from distill.common import atomic_json, external_path


def compare(before, after, variant='student_all'):
    reports = [json.loads((directory/'report.json').read_text()) for directory in (before, after)]
    a, b = reports
    for key in ('gpu', 'torch', 'sites', 'site_manifest_digest', 'source_digests'):
        if a.get(key) != b.get(key):
            raise ValueError(f'Physical reference identity changed: {key}.')
    fingerprints = [report['checkpoint_digests'][variant] for report in reports]
    for stage in ('base', 'coarse', 'decoder'):
        if fingerprints[0].get(stage) != fingerprints[1].get(stage) or not fingerprints[0].get(stage):
            raise ValueError(f'{stage} checkpoint changed; this is not an inference-only comparison.')
    for name in ('distill/student.py', 'distill/features.py'):
        if fingerprints[0].get('code:'+name) != fingerprints[1].get('code:'+name):
            raise ValueError(f'Training architecture/schema changed: {name}.')
    result = dict(before=str(before), after=str(after), variant=variant,
                  checkpoint_digests=fingerprints, rows=[], accepted=False)
    with np.load(before/'arrays.npz', allow_pickle=False) as old, np.load(after/'arrays.npz', allow_pickle=False) as new:
        keys = sorted(key for key in old.files if key.startswith(variant+'|'))
        if not keys or set(keys) != {key for key in new.files if key.startswith(variant+'|')}:
            raise ValueError('Physical case banks differ or contain no candidate views.')
        for report in reports:
            expected = {f"{variant}|{row['site']}|{row['lod']}" for row in report['variants'][variant]}
            if expected != set(keys):
                raise ValueError('Report and saved physical arrays disagree.')
        for key in keys:
            x, y = old[key], new[key]
            if x.shape != y.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f'Invalid physical arrays: {key}.')
            error = np.abs(x.astype(np.float64)-y.astype(np.float64))
            result['rows'].append(dict(view=key, exact=bool(np.array_equal(x, y)),
                                       max_abs_m=float(error.max()), mean_abs_m=float(error.mean())))
    result['exact_passed'] = all(row['exact'] for row in result['rows'])
    result['files'] = [{name: hashlib.sha256((directory/name).read_bytes()).hexdigest()
                        for name in ('report.json', 'arrays.npz')} for directory in (before, after)]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('--variant', default='student_all')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = compare(args.before, args.after, args.variant)
    atomic_json(external_path(args.output), result)
    print(json.dumps(dict(exact_passed=result['exact_passed'], views=len(result['rows']))), flush=True)
    if not result['exact_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

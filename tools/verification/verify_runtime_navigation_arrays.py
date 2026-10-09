"""Compare physical native payloads captured by the real navigation probe."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import argparse
import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


def payloads(report):
    result = {}
    for response in report['responses']:
        parts = urlparse(response['url']).path.split('/')
        if response.get('payload_path') and parts[4] == '0':
            data = Path(response['payload_path']).read_bytes()
            assert hashlib.sha256(data).hexdigest() == response['payload_sha256']
            headers = response['headers']
            width = int(headers['x-terrain-width'])
            climate_size = 5 * int(headers['x-terrain-climate-width']) * int(headers['x-terrain-climate-height'])
            values = np.frombuffer(data, dtype='<f4')
            assert values.size == width * width + climate_size and np.isfinite(values).all()
            key = (int(parts[5]), int(parts[6].split('.')[0]))
            result[key] = (values[:width * width], values[width * width:])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--max-height-error', type=float, default=1.)
    args = parser.parse_args()
    reference = json.loads(Path(args.reference).read_text())
    candidate = json.loads(Path(args.candidate).read_text())
    assert reference['passed'] and candidate['passed']
    for field in ('profile', 'seed', 'centre'):
        assert reference[field] == candidate[field], field
    before, after = payloads(reference), payloads(candidate)
    assert before and before.keys() == after.keys(), 'Native coordinate coverage differs'
    samples = []
    for key in sorted(before):
        old_height, old_climate = before[key]
        height, climate = after[key]
        delta = np.abs(height.astype(np.float64) - old_height)
        samples.append(dict(coordinates=key, height_byte_exact=height.tobytes() == old_height.tobytes(),
                            height_max_abs_m=float(delta.max()), height_mean_abs_m=float(delta.mean()),
                            climate_byte_exact=climate.tobytes() == old_climate.tobytes(),
                            climate_max_abs=float(np.abs(climate.astype(np.float64) - old_climate).max())))
    report = dict(reference=args.reference, candidate=args.candidate, samples=samples,
                  max_height_error_m=max(s['height_max_abs_m'] for s in samples),
                  all_height_byte_exact=all(s['height_byte_exact'] for s in samples),
                  all_climate_byte_exact=all(s['climate_byte_exact'] for s in samples))
    report['passed'] = report['max_height_error_m'] <= args.max_height_error and report['all_climate_byte_exact']
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}))
    assert report['passed'], 'Physical navigation fidelity gate failed'


if __name__ == '__main__':
    main()

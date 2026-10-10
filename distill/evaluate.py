"""Audit physical acceptance; absence of evidence can never count as a pass.

python -m distill.evaluate --report ~/data/distill/eval/student/report.json
                         --variant student_all --output ~/data/distill/eval/student/acceptance.json

Use the same-site measured FP32/BF16 floor. Speed and seam checks are explicit
additional requirements, not inferred from a low latent validation loss.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from distill.common import atomic_json, external_path


def ratio(candidate, reference):
    if reference == 0:
        return 1. if candidate == 0 else float('inf')
    return candidate/reference


def audit(report, variant, baseline='fp32base'):
    expected = {(site['name'], lod) for site in report['sites'] for lod in (3, 0)}
    if len(expected) != 14:
        raise ValueError('Acceptance requires all seven sites at both LODs.')
    indexed = {}
    for name in ('reference', baseline, variant):
        rows = report.get('variants', {}).get(name, [])
        table = {(r['site'], r['lod']): r for r in rows}
        if len(rows) != 14 or set(table) != expected or any('error' in row for row in rows):
            raise ValueError(f'{name}: complete successful 7-site × 2-LOD evidence is required.')
        indexed[name] = table
    results = []
    for key in sorted(expected):
        reference, floor, candidate = (indexed[n][key] for n in ('reference', baseline, variant))
        metrics = candidate.get('vs_reference')
        threshold = floor.get('vs_reference')
        if metrics is None or threshold is None:
            raise ValueError(f'{key}: missing physical comparison errors.')
        slopes = {name: ratio(candidate[name], reference[name]) for name in ('slope_mean', 'slope_p90')}
        spectral = [ratio(c, r) for c, r in zip(candidate['psd'], reference['psd'])]
        if len(spectral) != 5:
            raise ValueError('All five spectral bands are required.')
        checks = dict(mae=math.isfinite(metrics['mae']) and metrics['mae'] <= threshold['mae'],
                      coast=math.isfinite(metrics['coast']) and metrics['coast'] <= threshold['coast'],
                      slopes=all(math.isfinite(v) and .95 <= v <= 1.05 for v in slopes.values()),
                      spectrum=all(math.isfinite(v) and .95 <= v <= 1.05 for v in spectral))
        results.append(dict(site=key[0], lod=key[1], mae_m=metrics['mae'], floor_mae_m=threshold['mae'],
                            coast=metrics['coast'], floor_coast=threshold['coast'], slope_ratios=slopes,
                            psd_ratios=[v if math.isfinite(v) else None for v in spectral],
                            checks=checks, passed=all(checks.values())))
    return dict(variant=variant, baseline=baseline, physical_passed=all(r['passed'] for r in results), rows=results)


def speed_audit(base_path, student_path, minimum=10.):
    reference = json.loads(Path(base_path).read_text())
    student = json.loads(Path(student_path).read_text())
    if reference['gpu'] != student['gpu']:
        raise ValueError('Speed comparison must use the same physical GPU.')
    if student['stage'] != 'base':
        raise ValueError('The ×10 field criterion applies to the base student.')
    times = [s['graph_ms_per_window'] for s in reference['samples'] if s['dtype'] == 'bf16' and s['batch'] == 16]
    candidates = [s['graph_ms_per_64_surface'] for s in student['samples']
                  if s['dtype'] == 'bf16' and s['includes_halo']]
    if not times or not candidates:
        raise ValueError('Missing BF16 graph timings including the student halo.')
    # At stride 32, each 64² of final surface requires four windows × two steps.
    # Also report the stricter isolated-forward comparison to avoid ambiguity.
    isolated = min(times)/min(candidates)
    return dict(gpu=reference['gpu'], isolated_forward_speedup=isolated,
                final_field_speedup=8*isolated, minimum_field_speedup=minimum,
                passed=8*isolated >= minimum,
                note='Field ratio includes overlap/two passes; excludes feature construction and I/O.')


def artifact_audit(report, variant, benchmark, seams):
    """Bind all evidence to the same models/code; a timing fixture is not a candidate."""
    fingerprint = report.get('checkpoint_digests', {}).get(variant, {})
    sources = {key: value for key, value in fingerprint.items() if key.startswith('code:')}
    visual = seams.get('visual_review', {})
    checks = dict(
        physical_weights_present=bool(fingerprint.get('base')) and len(sources) == 3,
        benchmark_weights=bool(fingerprint.get('base')) and benchmark.get('checkpoint_digest') == fingerprint.get('base'),
        benchmark_code=bool(sources) and benchmark.get('student_source_digests') == sources,
        benchmark_trained=benchmark.get('step', 0) > 0,
        benchmark_gpu=bool(report.get('gpu')) and benchmark.get('gpu') == report.get('gpu'),
        seams_weights=bool(fingerprint.get('base')) and seams.get('checkpoint_digest') == fingerprint.get('base'),
        seams_code=bool(sources) and seams.get('student_source_digests') == sources,
        seams_gpu=bool(report.get('gpu')) and seams.get('gpu') == report.get('gpu'),
        seams_validation=seams.get('split') == 'val' and len(seams.get('rows', [])) >= 12,
        seams_numerical=seams.get('numerical_passed') is True and all(row.get('passed') is True for row in seams.get('rows', [])),
        seams_visual=visual.get('passed') is True and visual.get('checkpoint_digests') == fingerprint,
    )
    return dict(checks=checks, passed=all(checks.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--variant', default='student_all')
    parser.add_argument('--baseline', default='fp32base')
    parser.add_argument('--base-benchmark', type=Path)
    parser.add_argument('--student-benchmark', type=Path)
    parser.add_argument('--seams', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(json.loads(args.report.read_text()), args.variant, args.baseline)
    result['speed'] = (speed_audit(args.base_benchmark, args.student_benchmark)
                       if args.base_benchmark and args.student_benchmark else dict(passed=False, reason='not measured'))
    result['seams'] = json.loads(args.seams.read_text()) if args.seams else dict(passed=False, reason='not verified')
    benchmark = json.loads(args.student_benchmark.read_text()) if args.student_benchmark else {}
    result['artifacts'] = artifact_audit(json.loads(args.report.read_text()), args.variant, benchmark, result['seams'])
    result['accepted'] = (result['physical_passed'] and result['speed']['passed'] and
                          result['seams'].get('passed', False) and result['artifacts']['passed'])
    atomic_json(external_path(args.output), result)
    print(json.dumps(dict(variant=args.variant, physical_passed=result['physical_passed'],
                         accepted=result['accepted'], failing_sites=sum(not r['passed'] for r in result['rows']))))
    raise SystemExit(0 if result['accepted'] else 1)


if __name__ == '__main__':
    main()

"""Alternate fresh-process exact/reference pairs; retain child fidelity receipts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def verify_child(child, backend, reference, sources):
    if child.get('status') != 'complete' or not child.get('require_exact'):
        raise ValueError('Child must complete its byte-exact physical gate')
    if child['reference']['sha256'] != reference['sha256']:
        raise ValueError('Reference archive changed during the series')
    if sources is not None and child['source_sha256'] != sources:
        raise ValueError('Child source files changed during the series; restart the pairs')
    evidence = []
    for sample in child['samples']:
        kernels = sample['after_inference']['pointwise_kernels']
        if backend == 'exact':
            if not kernels['requested'] or not kernels['effective_by_device'] or not all(kernels['effective_by_device'].values()):
                raise ValueError('Exact backend fell back to Torch; speedup is not admissible')
        elif kernels['requested']:
            raise ValueError('Reference process enabled the custom kernels')
        evidence.append(dict(key=sample['key'], kernels=kernels))
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--pairs', type=int, default=5)
    args = parser.parse_args()
    if args.pairs < 5:
        parser.error('At least five alternating pairs are required')
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    receipt = output / 'paired-summary.json'
    reference = dict(path=str(Path(args.reference).resolve()),
                     sha256=hashlib.sha256(Path(args.reference).read_bytes()).hexdigest())
    report = dict(status='running', pairs=[], reference=reference,
                  method='Fresh processes, AB/BA alternating. One runtime-first-use footprint and one warmed fresh-store footprint per LOD per child. Same sites and original physical NPZ reference. No Terrain server or other Terrain NN work. Desktop GPU activity is uncontrolled.',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    def save():
        receipt.write_text(json.dumps(report, indent=2), encoding='utf-8')
    save()
    try:
        for index in range(args.pairs):
            order = ['reference', 'exact'] if index % 2 == 0 else ['exact', 'reference']
            pair = dict(index=index, order=order, runs={})
            report['pairs'].append(pair)
            save()
            for backend in order:
                label = f'pair-{index}-{backend}'
                env = os.environ.copy()
                env.update(TERRAIN_EXACT_KERNELS=str(int(backend == 'exact')),
                           TERRAIN_PROFILE='0', TERRAIN_PROFILE_CUDA='0',
                           TERRAIN_ATTENTION_BACKEND='reference', TERRAIN_PREWARM_BASE='0')
                command = [sys.executable, str(Path(__file__).with_name('benchmark_runtime_lod.py')),
                           '--label', label, '--output', str(output), '--reference', args.reference,
                           '--require-exact', '--repeats', '1', '--lods', '3', '2']
                with (output / (label+'.log')).open('w', encoding='utf-8') as log:
                    subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
                child = json.loads((output/(label+'.json')).read_text(encoding='utf-8'))
                evidence = verify_child(child, backend, reference, report.get('child_source_sha256'))
                report['child_source_sha256'] = child['source_sha256']
                pair.setdefault('backend_evidence', {})[backend] = evidence
                pair['runs'][backend] = child['summary']
                save()
                print(label+' complete', flush=True)
            pair['speedup'] = {lod: pair['runs']['reference'][lod]['median_seconds'] /
                               pair['runs']['exact'][lod]['median_seconds'] for lod in ('3', '2')}
            save()
        report['summary'] = {lod: dict(
            median_paired_speedup=statistics.median(p['speedup'][lod] for p in report['pairs']),
            paired_speedups=[p['speedup'][lod] for p in report['pairs']],
            reference_median_seconds=statistics.median(p['runs']['reference'][lod]['median_seconds'] for p in report['pairs']),
            exact_median_seconds=statistics.median(p['runs']['exact'][lod]['median_seconds'] for p in report['pairs']))
            for lod in ('3', '2')}
        report['status'] = 'complete'
    except Exception as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        save()
    print(json.dumps(report['summary']), flush=True)


if __name__ == '__main__':
    main()

"""Measure preserved candidates sequentially, after successful GPU jobs finish.

The coordinator uses CPU only. Each measurement starts in its own process and
refuses competing CUDA work. Durable reports can be reused only for identical
checkpoint and inference code hashes. No quality decision is made here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from distill.common import REPO, atomic_json, external_path
from distill.export import export_bundle
from distill.jobs import state
from distill.run_training import wait_job


def wait_seam_diagnostic(name, report_path):
    """A completed numerical rejection is evidence; a crashed check is not."""
    try:
        wait_job(name)
    except RuntimeError:
        from distill.common import DATA
        job = state(DATA/'jobs'/name)
        path = Path(report_path)
        if job['live'] or job.get('returncode') != 1 or not path.exists():
            raise
        report = json.loads(path.read_text())
        sources = {'code:'+name: hashlib.sha256((REPO/name).read_bytes()).hexdigest()
                   for name in ('distill/student.py', 'distill/features.py', 'distill/inference.py', 'distill/coarse_solver.py')}
        if (path.stat().st_mtime < job['started_at'] or report.get('stage') != 'base' or
                report.get('numerical_passed') is not False or len(report.get('rows', [])) != 12 or
                report.get('student_source_digests') != sources):
            raise
        for row in report['rows']:
            if not isinstance(row.get('passed'), bool) or not all(
                    math.isfinite(row.get(key, float('nan')))
                    for key in ('max_abs', 'mean_abs', 'max_jump_error')):
                raise
        print(f'NUMERICAL REJECTION {name}: retained in {path}; benchmarks still needed.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for stage in ('base', 'coarse', 'decoder'):
        parser.add_argument('--'+stage+'-inspection', type=Path, required=True)
    parser.add_argument('--wait-job', nargs='+', default=[])
    parser.add_argument('--wait-seam', nargs=2, action='append', default=[], metavar=('JOB', 'REPORT'))
    parser.add_argument('--gpus', nargs='+', type=int, choices=(0, 1), default=[0, 1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    for name in args.wait_job:
        wait_job(name)
    for name, path in args.wait_seam:
        wait_seam_diagnostic(name, path)
    paths, digests = {}, {}
    for stage in ('base', 'coarse', 'decoder'):
        inspection = json.loads(getattr(args, stage+'_inspection').read_text())
        path = Path(inspection['candidate'])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != inspection['sha256']:
            raise ValueError(f'{stage}: preserved candidate differs from its inspection.')
        paths[stage], digests[stage] = path, digest
    sources = {'code:'+name: hashlib.sha256((REPO/name).read_bytes()).hexdigest()
               for name in ('distill/student.py', 'distill/features.py', 'distill/inference.py', 'distill/coarse_solver.py')}
    manifest = export_bundle(paths, output/'weights')
    summary = dict(status='running', sources=manifest['sources'], checkpoint_digests=digests,
                   student_source_digests=sources, measurements=[], accepted=False)
    for gpu in dict.fromkeys(args.gpus):
        environment = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=str(gpu),
                           TERRAIN_CUDA_DEVICE='0', TERRAIN_GPU_MODE='single', TERRAIN_CUDA_DEVICES='0')
        for stage in ('base', 'coarse', 'decoder'):
            for module, kind in [('distill.bench_student', 'network'), ('distill.bench_pipeline', 'stage')]:
                path = output/f'{stage}-{kind}-gpu{gpu}.json'
                reuse = False
                if path.exists():
                    old = json.loads(path.read_text())
                    if old['checkpoint_digest'] != digests[stage] or old['student_source_digests'] != sources:
                        raise ValueError('Benchmark identity changed; use a separate output directory.')
                    # Network reports are written only after their measurement.
                    reuse = old.get('whole_system_idle') is True and (
                        kind == 'network' or old.get('status') == 'complete')
                if not reuse:
                    print(f'BENCH gpu={gpu} stage={stage} kind={kind}', flush=True)
                    subprocess.run([sys.executable, '-m', module, '--checkpoint', str(paths[stage]),
                                    '--output', str(path)], cwd=REPO, env=environment, check=True)
                report = json.loads(path.read_text())
                summary['measurements'].append(dict(gpu=gpu, stage=stage, kind=kind,
                    report=str(path), comparisons=report.get('comparisons'), samples=report.get('samples')))
                atomic_json(output/'summary.json', summary)
    atomic_json(output/'summary.json', summary | dict(status='complete'))


if __name__ == '__main__':
    main()

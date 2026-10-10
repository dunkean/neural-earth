"""Insert a short GPU probe between training stages, then restore coordination.

The current trainer is never interrupted. Failed/interrupted training requires
inspection and is not automatically relaunched. After successful training,
even a failed probe restores the independent training workflow.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from distill.common import DATA, REPO, atomic_json
from distill.jobs import state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-job', required=True)
    parser.add_argument('--training-status', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    progress = DATA/'idle-probe.json'
    atomic_json(progress, dict(status='waiting', wait_job=args.wait_job, updated_at=time.time()))
    print(f'Waiting for {args.wait_job}; checks once per minute.', flush=True)
    while True:
        current = state(DATA/'jobs'/args.wait_job)
        if not current['live']:
            if current['status'] in ('starting', 'running'):
                time.sleep(1)
                current = state(DATA/'jobs'/args.wait_job)
            if not current['live']:
                if current['status'] != 'complete' or current.get('returncode') != 0:
                    raise RuntimeError(f'Trainer needs inspection: {current}')
                break
        time.sleep(60)
    completed = json.loads(args.training_status.read_text())
    if completed.get('status') != 'complete' or completed.get('step', 0) < completed.get('target_steps', 1):
        raise RuntimeError(f'Trainer did not finish; no automatic restart: {completed}')
    if state(DATA/'jobs/training-workflow')['live']:
        raise RuntimeError('The coordinator is still live and may start another GPU job; do not probe.')
    try:
        atomic_json(progress, dict(status='benchmarking', updated_at=time.time()))
        subprocess.run([sys.executable, '-m', 'distill.bench_pipeline', '--checkpoint', str(args.checkpoint),
                        '--output', str(args.output)], cwd=REPO, check=True)
        atomic_json(progress, dict(status='complete', output=str(args.output), updated_at=time.time()))
    except Exception as exc:
        atomic_json(progress, dict(status='probe-failed', error=str(exc), updated_at=time.time()))
        raise
    finally:
        if not state(DATA/'jobs/training-workflow')['live']:
            subprocess.run([sys.executable, '-m', 'distill.jobs', 'start', 'training-workflow', '--gpu', 'cpu',
                            '--', sys.executable, '-m', 'distill.run_training'], cwd=REPO, check=True)
            print('Training coordination restored.', flush=True)


if __name__ == '__main__':
    main()

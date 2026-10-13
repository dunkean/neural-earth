"""Dispatch one authorized Opus review after a known quota reset time."""
import argparse
from datetime import datetime
import os
from pathlib import Path
import sys
import time

from distill.common import atomic_json, external_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--not-before', required=True, help='ISO8601 timestamp with timezone')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    target = datetime.fromisoformat(args.not_before)
    if target.tzinfo is None:
        raise ValueError('An explicit timezone is required.')
    output = external_path(args.output)
    if output.exists() or not args.report.is_file():
        raise ValueError('Require a report and a fresh review output directory.')
    status = output.with_name(output.name+'.deferred.json')
    if status.exists():
        raise ValueError('Preserve an existing deferred review schedule.')
    contract = dict(not_before=target.isoformat(), report=str(args.report), output=str(output),
                    attempts=1, reason='Explicit retry after a verified session quota reset.')
    atomic_json(status, dict(status='waiting', **contract))
    print('Review scheduled for '+target.isoformat(), flush=True)
    while True:
        remaining = target.timestamp()-time.time()
        if remaining <= 0:
            break
        time.sleep(min(60., remaining))
    atomic_json(status, dict(status='dispatched', dispatched_at=time.time(), **contract))
    # Replace this process so job supervision and the review's signal handlers
    # remain authoritative. There is no retry loop after this single attempt.
    os.execv(sys.executable, [sys.executable, '-u', '-m', 'distill.opus_stage_review',
                            '--report', str(args.report), '--output', str(output)])


if __name__ == '__main__':
    main()

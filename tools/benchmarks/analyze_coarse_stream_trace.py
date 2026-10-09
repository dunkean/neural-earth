"""Summarize actual cross-stream kernel overlap in an exported Nsight SQLite trace."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sqlite3


def overlap(rows):
    events = defaultdict(list)
    for start, end, stream in rows:
        if end > start:
            events[start].append((stream, 1))
            events[end].append((stream, -1))
    active = defaultdict(int)
    busy = concurrent = peak = 0
    previous = None
    for timestamp, changes in sorted(events.items()):
        count = sum(n > 0 for n in active.values())
        if previous is not None:
            interval = timestamp-previous
            if count:
                busy += interval
            if count > 1:
                concurrent += interval
        for stream, delta in changes:
            active[stream] += delta
        peak = max(peak, sum(n > 0 for n in active.values()))
        previous = timestamp
    return dict(kernel_count=len(rows), stream_ids=sorted({row[2] for row in rows}),
                peak_simultaneous_kernel_streams=peak,
                busy_seconds=busy/1e9, concurrent_seconds=concurrent/1e9,
                fraction_of_busy_time_with_cross_stream_kernels=concurrent/busy if busy else 0.)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sqlite', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with sqlite3.connect(args.sqlite) as connection:
        ranges = connection.execute("SELECT start,end,text FROM NVTX_EVENTS WHERE text LIKE 'coarse/%' AND end IS NOT NULL ORDER BY start").fetchall()
        if not ranges:
            raise ValueError('No coarse benchmark NVTX ranges found')
        results = {}
        for start, end, label in ranges:
            rows = connection.execute('SELECT start,end,streamId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start >= ? AND end <= ? ORDER BY start', (start,end)).fetchall()
            if not rows:
                raise ValueError(f'No traced kernels for {label}')
            results[label] = overlap(rows)
    report = dict(sqlite=str(args.sqlite.resolve()), sqlite_sha256=hashlib.sha256(args.sqlite.read_bytes()).hexdigest(),
                  method='Sweep kernel start/end timestamps inside benchmark NVTX ranges; count distinct streams with kernels in flight. Profiling perturbs timing; use this evidence only for overlap, not speedup.',
                  ranges=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

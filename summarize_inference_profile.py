"""Summarize real CUDA kernels inside warmed graphs from a Chrome profiler trace."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def summarize(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    totals=defaultdict(float)
    counts=Counter()
    groups=defaultdict(float)
    for event in data['traceEvents']:
        if event.get('cat')!='kernel' or 'dur' not in event:
            continue
        name=event['name']
        duration=float(event['dur'])
        totals[name]+=duration
        counts[name]+=1
        if 'nchwToNhwc' in name or 'nhwcToNchw' in name:
            group='layout conversions'
        elif 'cudnn' in name and any(marker in name for marker in ('fprop','implicit_gemm','winograd')):
            group='recognized convolution kernels'
        elif name in ('silu_scaled','binary_sum','binary_concat'):
            group=name
        else:
            group='other/unclassified kernels'
        groups[group]+=duration
    total=sum(totals.values())
    if not total:
        raise ValueError('No CUDA kernel durations in this trace')
    return dict(trace_path=str(Path(path).resolve()),trace_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        total_kernel_ms=total/1000,
        note='Summed instrumented kernel durations, not wall time or critical path. CUDA graph kernels are included. Name-based groups are partial attribution, not per-network measurements.',
        groups={name:dict(ms=duration/1000,percent=100*duration/total) for name,duration in groups.items()},
        kernels=[dict(name=name,ms=duration/1000,percent=100*duration/total,calls=counts[name])
                 for name,duration in sorted(totals.items(),key=lambda pair:-pair[1])])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    report=summarize(args.trace)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(total_kernel_ms=report['total_kernel_ms'],groups=report['groups'])))

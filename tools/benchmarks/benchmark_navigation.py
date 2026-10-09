"""Small, repeatable HTTP benchmark of the full local generation path.

Use a new seed to measure cold world caches; weights can already be resident.
This is intentionally sequential and does not mix browser activity into results.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import argparse
import json
import time
import urllib.request
import urllib.parse
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8765')
    parser.add_argument('--seed', type=int, default=2026100702)
    parser.add_argument('--x', type=int, default=-14)
    parser.add_argument('--y', type=int, default=10)
    parser.add_argument('--world-profile', choices=('natural','terrestrial-earthlike','terrestrial-gondwana','terrestrial-continents','terrestrial-archipelago'), default='natural')
    parser.add_argument('--format', choices=('height','png'), default='height')
    parser.add_argument('--repetitions', type=int, default=3)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.repetitions < 1 or not 0 <= args.seed < 2**64:
        parser.error('repetitions must be positive and seed must be uint64')
    query = urllib.parse.urlencode({'world_profile':args.world_profile})
    with urllib.request.urlopen(f'{args.url}/api/world?seed={args.seed}&{query}', timeout=30) as response:
        world = json.load(response)
    rows = []
    cases = [('native-cold', 0, args.x, args.y),
             ('native-adjacent', 0, args.x+1, args.y),
             ('native-cache', 0, args.x, args.y),
             ('coarse-cold', 5, -1, 0)]
    for repetition in range(args.repetitions):
        # Only the first pass can be world-cold. Repetitions deliberately report
        # cache revisits separately; they are never folded into the cold median.
        for label, lod, tx, ty in cases:
            start = time.perf_counter()
            endpoint, extension = ('height','bin') if args.format=='height' else ('tiles','png')
            parameters=urllib.parse.urlencode(dict(world_profile=args.world_profile,
                profile=world['cache_profile'],climate='1',mode='relief'))
            url = f"{args.url}/{endpoint}/{world['version']}/{args.seed}/{lod}/{tx}/{ty}.{extension}?{parameters}"
            with urllib.request.urlopen(url, timeout=300) as response:
                payload = response.read()
                headers = {k: v for k, v in response.headers.items() if k.lower().startswith('x-terrain')}
            row = dict(label=label if repetition==0 else label+'-revisit', repetition=repetition,
                       wall_seconds=round(time.perf_counter()-start, 6),bytes=len(payload), headers=headers)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    with urllib.request.urlopen(f'{args.url}/api/status', timeout=30) as response:
        status = json.load(response)
    summaries={}
    for hit in ('hit','miss'):
        timings=sorted(r['wall_seconds'] for r in rows if r['headers'].get('X-Terrain-Cache')==hit)
        if timings:
            summaries[hit]=dict(samples=len(timings),median=statistics.median(timings),
                p95=timings[max(0,__import__('math').ceil(.95*len(timings))-1)],
                p95_method='nearest rank; small corpus, not a sustained navigation claim')
    report = dict(seed=str(args.seed), version=world['version'], world_profile=args.world_profile,
                  cache_profile=world['cache_profile'],world_manifest=world.get('world_manifest'),
                  transport=args.format, cases=rows, summaries=summaries,status=status,
                  note='HTTP caméra → champs reçus ; exclut présentation GPU. Miss ne prouve pas un monde froid : lire les fenêtres/cache dans status. Poids initiaux non isolés. Fermer les autres onglets.')
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

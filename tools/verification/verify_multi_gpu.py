"""Live multi-GPU check: cold single-GPU versus multi-GPU servers on one workload.

Each mode runs in its own server process with an empty output cache, so both
compute every tile. The browser protocol is replayed: one camera view, then
concurrent height requests. The report gives wall time per mode, the GPU that
produced each tile, whether the detail GPU ever ran the coarse network, and
height differences against the single-GPU reference, grouped by producing GPU.

  python tools/verification/verify_multi_gpu.py --output OUTPUT_DIR
  python tools/verification/verify_multi_gpu.py --output OUTPUT_DIR --modes single throughput consistent
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from urllib import request as http
import uuid

import numpy as np

ROOT = _REPO_ROOT


def call(base, path, data=None, timeout=600):
    body = None if data is None else json.dumps(data).encode()
    req = http.Request(base+path, data=body, method='GET' if data is None else 'POST',
                       headers={'Content-Type': 'application/json'} if data is not None else {})
    with http.urlopen(req, timeout=timeout) as response:
        return response.status, dict(response.headers), response.read()


def wait_ready(base, process, timeout=900):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError('Server exited during startup')
        try:
            _, _, raw = call(base, '/api/gpu', timeout=3)
            status = json.loads(raw)
            states = [d['preload']['state'] for d in status['devices'] if d['roles'] and d['preload']]
            if states and all(state in ('ready', 'failed') for state in states):
                return status
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError('Server did not finish model preloading')


def workload(lods, size):
    tiles = []
    for lod in lods:
        for ty in range(-(size[1]//2), size[1]-size[1]//2):
            for tx in range(-(size[0]//2), size[0]-size[0]//2):
                # Coarser levels first, then outward from the centre, like the viewer.
                tiles.append(dict(lod=lod, tx=tx, ty=ty, priority=1000*(4-min(lod, 3))//2+abs(tx)+abs(ty)))
    return tiles


def run_mode(mode, args, tiles, output, prime=()):
    port = args.port
    base = f'http://127.0.0.1:{port}'
    cache = Path(tempfile.mkdtemp(prefix=f'multi-gpu-{mode}-', dir=output))
    env = dict(os.environ, TERRAIN_GPU_MODE=mode, TERRAIN_OUTPUT_ROOT=str(cache))
    env.pop('TERRAIN_CUDA_DEVICE', None)
    log = (output/f'server-{mode}.log').open('wb')
    process = subprocess.Popen([sys.executable, '-u', str(ROOT/'backend'/'terrain_server.py'), '--port', str(port)],
                               cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        before = wait_ready(base, process)
        _, _, raw = call(base, f'/api/world?seed={args.seed}')
        world = json.loads(raw)
        session = 'verify' + uuid.uuid4().hex[:12]
        view = dict(session=session, epoch=1, seed=str(args.seed), world_profile='natural', tiles=list(prime))
        stop = threading.Event()
        def keep_alive():
            while not stop.wait(10):
                call(base, '/api/view', view)
        keeper = threading.Thread(target=keep_alive, daemon=True)
        keeper.start()
        results = {}
        def fetch(tile):
            started = time.perf_counter()
            path = (f"/height/natural-v1/{args.seed}/{tile['lod']}/{tile['tx']}/{tile['ty']}.bin"
                    f"?session={session}&epoch={view['epoch']}&profile={world['cache_profile']}")
            for _ in range(5):
                try:
                    status, headers, raw = call(base, path)
                    break
                except http.HTTPError as exc:
                    if exc.code not in (429, 409):
                        raise
                    time.sleep(1)
            width = int(headers['X-Terrain-Width'])
            results[(tile['lod'], tile['tx'], tile['ty'])] = dict(
                heights=np.frombuffer(raw, '<f4')[:width*width].reshape(width, width).copy(),
                device=headers.get('X-Terrain-Device', ''), stage=headers['X-Terrain-Stage'],
                compute=float(headers.get('X-Terrain-Compute-Seconds', 0)), latency=time.perf_counter()-started)
        if prime:
            # Untimed: learned coarse windows of the region (warm navigation).
            call(base, '/api/view', view)
            with ThreadPoolExecutor(args.concurrency) as pool:
                list(pool.map(fetch, prime))
            results.clear()
        view.update(epoch=2, tiles=tiles)
        call(base, '/api/view', view)
        started = time.perf_counter()
        with ThreadPoolExecutor(args.concurrency) as pool:
            list(pool.map(fetch, tiles))
        wall = time.perf_counter()-started
        stop.set()
        _, _, raw = call(base, '/api/gpu')
        after = json.loads(raw)
        return dict(mode=mode, wall_seconds=round(wall, 3), plan=after['plan'], cache_profile=world['cache_profile'],
                    shared_windows=after.get('shared_windows'),
                    world_identity=world['world_identity'], devices=after['devices'],
                    preload_graphs={d['name']: (d['preload'] or {}).get('cuda_graphs', {}).get('fully_warmed')
                                    for d in before['devices'] if d['roles']}), results
    finally:
        process.terminate()
        try:
            process.wait(30)
        except subprocess.TimeoutExpired:
            process.kill()
        log.close()


def compare(reference, results):
    groups = {}
    for key, value in results.items():
        base = reference[key]['heights']
        diff = np.abs(value['heights']-base)
        group = groups.setdefault((value['stage'], value['device'] or 'cpu'), dict(tiles=0, max=0., mean=0., bitwise=0))
        group['tiles'] += 1
        group['max'] = max(group['max'], float(diff.max()))
        group['mean'] += float(diff.mean())
        group['bitwise'] += int(np.array_equal(value['heights'], base))
    for group in groups.values():
        group['mean'] = round(group['mean']/group['tiles'], 4)
        group['max'] = round(group['max'], 4)
    return {f'{stage} · {device}': value for (stage, device), value in sorted(groups.items())}


def seams(results, lod):
    """Height step across interior tile borders, split by same/different GPU."""
    steps = dict(same_gpu=[], cross_gpu=[])
    for (level, tx, ty), value in results.items():
        if level != lod:
            continue
        right = results.get((lod, tx+1, ty))
        if right is None:
            continue
        halo = (value['heights'].shape[0]-256)//2
        a = value['heights'][halo:-halo, -halo-1]
        b = right['heights'][halo:-halo, halo]
        inner = np.abs(value['heights'][halo:-halo, -halo-2]-a).mean()
        steps['same_gpu' if value['device'] == right['device'] else 'cross_gpu'].append(
            float(np.abs(a-b).mean()-inner))
    return {k: dict(borders=len(v), mean_excess_step_m=round(float(np.mean(v)), 4) if v else None)
            for k, v in steps.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--modes', nargs='+', default=['single', 'throughput'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lods', nargs='+', type=int, default=[5, 3, 0])
    parser.add_argument('--prime-lods', nargs='*', type=int, default=[],
                        help='Untimed levels requested first, e.g. 5 to measure warm-coarse navigation')
    parser.add_argument('--size', nargs=2, type=int, default=[8, 5], metavar=('W', 'H'))
    parser.add_argument('--concurrency', type=int, default=24)
    parser.add_argument('--port', type=int, default=8791)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tiles = workload(args.lods, args.size)
    prime = workload(args.prime_lods, args.size)
    reports, arrays = [], {}
    for mode in args.modes:
        report, results = run_mode(mode, args, tiles, args.output, prime)
        arrays[mode] = results
        report['tiles_by_device'] = {}
        for value in results.values():
            name = value['device'] or 'cpu'
            report['tiles_by_device'][name] = report['tiles_by_device'].get(name, 0)+1
        report['detail_gpu_coarse_forwards'] = {d['name']: d['forward_calls'].get('coarse', 0)
                                                for d in report['devices'] if 'coarse' not in d['roles'] and d['roles']}
        if mode != args.modes[0]:
            report['difference_vs_'+args.modes[0]] = compare(arrays[args.modes[0]], results)
        report['lod0_border_steps'] = seams(results, 0) if 0 in args.lods else None
        reports.append(report)
        print(json.dumps({k: v for k, v in report.items() if k != 'devices'}, indent=1), flush=True)
    summary = dict(tiles=len(tiles), lods=args.lods, prime_lods=args.prime_lods, size=args.size, seed=args.seed, reports=reports)
    if len(reports) > 1:
        summary['speedup_vs_'+reports[0]['mode']] = {r['mode']: round(reports[0]['wall_seconds']/r['wall_seconds'], 3)
                                                     for r in reports[1:]}
    (args.output/'multi-gpu-report.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'reports'}, indent=1))


if __name__ == '__main__':
    main()

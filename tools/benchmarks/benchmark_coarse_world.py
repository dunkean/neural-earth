"""Measure finite learned-world preparation and a learned overview read."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()
from terrain_paths import RUNTIME_ROOT

import json
from pathlib import Path
import time
import urllib.request

BASE='http://127.0.0.1:8765'
OUTPUT=(RUNTIME_ROOT / 'audit-implementation/coarse-world.json')

def read(path,data=None):
    request=urllib.request.Request(BASE+path,
        data=json.dumps(data).encode() if data is not None else None,
        headers={'Content-Type':'application/json'} if data is not None else {})
    with urllib.request.urlopen(request,timeout=300) as response:
        return json.load(response)

def main():
    world=read('/api/world?seed=42&world_profile=natural')
    start=time.perf_counter()
    task=read('/api/coarse/prepare',dict(seed='42',world_profile='natural',max_windows=8192))
    samples=[]
    last=-1
    while True:
        status=read('/api/status')
        task=status['coarse_preparation']
        if task.get('seed')!='42' or task.get('world_profile')!='natural':
            raise RuntimeError('Another client replaced the measured preparation')
        progress=task.get('progress',{})
        complete=progress.get('complete_windows',0)
        if complete//500!=last or task['state']!='running':
            last=complete//500
            samples.append(dict(wall_seconds=time.perf_counter()-start,task=task))
            print(json.dumps(dict(wall_seconds=round(time.perf_counter()-start,2),
                                 complete=complete,total=progress.get('total_windows'),state=task['state'])),flush=True)
        report=dict(world=world,samples=samples,wall_seconds=time.perf_counter()-start,
                    task=task,status=status,
                    note='Weights already loaded; native navigation persisted some windows before this run. One 6160-window finite plan including climate apron. This is learned coarse, not final 30m DEM coverage.')
        OUTPUT.write_text(json.dumps(report,indent=2),encoding='utf-8')
        if task['state']!='running':
            break
        if time.perf_counter()-start>1800:
            raise TimeoutError('Preparation exceeded 30 minutes')
        time.sleep(1)
    if progress.get('complete_windows')!=progress.get('total_windows') or task.get('state')=='failed':
        raise AssertionError('Full learned coarse coverage was not attained')
    before=status['cuda_forward_calls']
    url=f"/height/{world['version']}/42/12/0/0.bin?world_profile=natural&climate=1&profile={world['cache_profile']}"
    begin=time.perf_counter()
    with urllib.request.urlopen(BASE+url,timeout=300) as response:
        payload=response.read();headers=dict(response.headers)
    after=read('/api/status')['cuda_forward_calls']
    report['learned_overview_read']=dict(wall_seconds=time.perf_counter()-begin,
        bytes=len(payload),headers=headers,nn_calls_before=before,nn_calls_after=after,
        no_extra_nn=before==after)
    report['passed']=bool(before==after and headers.get('X-Terrain-Stage')=='coarse-area-mean')
    OUTPUT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(OUTPUT),passed=report['passed'],wall_seconds=report['wall_seconds'],overview_seconds=report['learned_overview_read']['wall_seconds'])),flush=True)
    if not report['passed']:
        raise AssertionError('Prepared learned overview ran extra neural work or used preview')

if __name__=='__main__':main()

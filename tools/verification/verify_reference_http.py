"""Compare the final natural HTTP packet against the saved pre-audit packet."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path

import hashlib
import json
from pathlib import Path
import time
import urllib.request
import numpy as np

ROOT=_REPO_ROOT
OUTPUT=Path('E:/TerrainDiffusionRuntime/audit-implementation')

def main():
    baseline=OUTPUT/'baseline-natural.bin'
    start=time.perf_counter()
    with urllib.request.urlopen('http://127.0.0.1:8765/height/natural-v1/20261007092/0/-14/10.bin?world_profile=natural&climate=1',timeout=300) as response:
        payload=response.read()
        headers=dict(response.headers)
    expected=np.frombuffer(baseline.read_bytes(),dtype='<f4')
    actual=np.frombuffer(payload,dtype='<f4')
    if actual.shape!=expected.shape:
        raise AssertionError('Reference packet dimensions changed')
    pixels=304*304
    height_error=float(np.max(np.abs(actual[:pixels]-expected[:pixels])))
    climate_error=np.max(np.abs(actual[pixels:]-expected[pixels:]).reshape(5,33,33),axis=(1,2)).tolist()
    report=dict(seed='20261007092',lod=0,tx=-14,ty=10,world_profile='natural',
                baseline_sha256=hashlib.sha256(baseline.read_bytes()).hexdigest(),
                actual_sha256=hashlib.sha256(payload).hexdigest(),
                height_max_m=height_error,climate_max=climate_error,headers=headers,
                wall_seconds=time.perf_counter()-start,
                source_sha256={name:hashlib.sha256((source_path(name, root=ROOT)).read_bytes()).hexdigest() for name in
                    ('terrain_server.py','terrain_inference.py','terrain_window_scheduler.py','terrain_climate.py')},
                passed=height_error==0 and all(value==0 for value in climate_error),
                note='One saved 304-square height+compact-climate HTTP packet; not a general proof of order invariance.')
    (OUTPUT/'natural-reference-fidelity.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report),flush=True)
    if not report['passed']:
        raise AssertionError('Natural packet differs from saved pre-audit reference')

if __name__=='__main__':main()

"""Measure native initialization, verified disk reload and resident lookup.

No neural models or GPU operations. Seed is fresh/random unless explicitly given.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import argparse
import hashlib
import json
from pathlib import Path
import secrets
import time

import terrain_bootstrap as bootstrap


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    seed=secrets.randbits(64) if args.seed is None else args.seed
    report={'seed_u64':str(seed),'gpu_used':False,'uncontrolled_desktop_load':True,
            'harness_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'implementation':bootstrap.implementation_identity(),'styles':{}}
    for style in bootstrap.STYLES:
        start=time.perf_counter()
        heightmap=bootstrap.get_heightmap(seed,style)
        first=time.perf_counter()-start
        start=time.perf_counter()
        replay=bootstrap.WorldHeightmap(seed,style)
        disk=time.perf_counter()-start
        start=time.perf_counter()
        resident=bootstrap.get_heightmap(seed,style)
        lookup=time.perf_counter()-start
        if heightmap.height_m.tobytes()!=replay.height_m.tobytes() or resident is not heightmap:
            raise RuntimeError('Replay changed native height')
        report['styles'][style]={'first_creation_s':first,'verified_disk_reload_s':disk,
            'resident_lookup_s':lookup,'native_generation_s':heightmap.metadata['generation_seconds'],
            'selected_attempt':heightmap.metadata['selected_attempt'],
            'cache_path':str(heightmap.cache_path),'height_sha256':heightmap.metadata['height_sha256']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(args.output),'seed':str(seed),'styles':report['styles']},indent=2))


if __name__=='__main__':
    main()

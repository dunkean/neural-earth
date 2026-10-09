"""Live CUDA polar latent/decoder reads, including an adjacent-tile seam."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import json
import math
import urllib.parse
import urllib.request

import numpy as np


def main(seed=42,profile='natural'):
    base='http://127.0.0.1:8765'
    with urllib.request.urlopen(base+'/api/world?'+urllib.parse.urlencode(dict(seed=seed,world_profile=profile)),timeout=120) as r:
        world=json.load(r)
    params=urllib.parse.urlencode(dict(profile=world['cache_profile'],world_profile=world['generation_profile'],
        world_identity=world['world_identity'],neural_chart='polar',climate=1,mode='relief'))
    def read(lod,tx,ty):
        with urllib.request.urlopen(f'{base}/height/{world["version"]}/{seed}/{lod}/{tx}/{ty}.bin?{params}',timeout=180) as r:
            width=int(r.headers['X-Terrain-Width']);stage=r.headers['X-Terrain-Stage']
            data=np.frombuffer(r.read(),dtype='<f4')
            assert np.isfinite(data).all()
            height=data[:width*width].reshape(width,width)
            print(json.dumps(dict(lod=lod,tx=tx,ty=ty,stage=stage,min=float(height.min()),max=float(height.max()))),flush=True)
            return height,stage
    pole_x=(world['world_bounds'][2]-world['world_bounds'][0])/4
    tx=math.floor(-pole_x/(256*240))
    north_a,stage=read(3,tx,-1);assert stage=='latent'
    north_b,stage=read(3,tx,0);assert stage=='latent'
    # Identical physical cells in the shared halo, excluding its filter edge.
    np.testing.assert_allclose(north_a[-43:-5,5:-5],north_b[5:43,5:-5],atol=1e-4,rtol=0)
    _,stage=read(3,math.floor(pole_x/(256*240)),0);assert stage=='latent'
    _,stage=read(0,math.floor(-pole_x/(256*30)),0);assert stage=='decoder'
    _,stage=read(0,math.floor(pole_x/(256*30)),0);assert stage=='decoder'
    print('Real CUDA polar latents, decoder and shared seam: OK',flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--profile',default='natural')
    args=parser.parse_args();main(args.seed,args.profile)

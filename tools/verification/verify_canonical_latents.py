"""GPU gate for the optional scalar-base canonical numerical profile."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from terrain_app import RUNTIME, load_pipeline
from terrain_inference import configure_world


CASES = (
    (0, (-128, -128, 128, 128)),
    (42, (2560, -3584, 2816, -3328)),
    (43, (2560, -3584, 2816, -3328)),
    (20261007092, (2560, -3584, 2816, -3328)),
)
PERMUTATIONS = ((0, 1, 2, 3), (3, 0, 2, 1), (1, 3, 0, 2))


def _read(world, bounds):
    result = world.get(*bounds, with_climate=True)
    return result['elev'].float().cpu().numpy(), result['climate'].float().cpu().numpy()


def _split(world, bounds, permutation):
    i1, j1, i2, j2 = bounds
    im, jm = (i1+i2)//2, (j1+j2)//2
    patches = ((i1,j1,im,jm), (i1,jm,im,j2), (im,j1,i2,jm), (im,jm,i2,j2))
    height = np.empty((i2-i1,j2-j1), np.float32)
    climate = np.empty((5,i2-i1,j2-j1), np.float32)
    for k in permutation:
        pi1,pj1,pi2,pj2 = patches[k]
        h,c = _read(world, patches[k])
        height[pi1-i1:pi2-i1,pj1-j1:pj2-j1] = h
        climate[:,pi1-i1:pi2-i1,pj1-j1:pj2-j1] = c
    return height,climate


def _error(a,b):
    return dict(height_max_m=float(np.max(np.abs(a[0]-b[0]))),
                climate_max=[float(np.max(np.abs(a[1][k]-b[1][k]))) for k in range(5)],
                coast_disagreements=int(np.count_nonzero((a[0]>=0)!=(b[0]>=0))))


def _evict_graph(tensor, seen=None):
    if seen is None:
        seen=set()
    if tensor.uuid in seen:
        return
    seen.add(tensor.uuid)
    for upstream in tensor.args:
        _evict_graph(upstream,seen)
    tensor.clear_cache()


@torch.inference_mode()
def main():
    world = load_pipeline(42)
    natural_profile = world._terrain_profile
    if natural_profile.canonical_latents or natural_profile.latent_batch != 16:
        raise ValueError('Run the canonical gate from the validated natural batch-16 baseline')
    canonical_profile = replace(natural_profile, latent_batch=1, canonical_latents=True)
    rows=[]
    for seed,bounds in CASES:
        configure_world(world,natural_profile,world_profile='natural')
        world.change_seed(seed)
        world.rebuild()
        baseline=_read(world,bounds)
        configure_world(world,canonical_profile,world_profile='natural')
        world.rebuild()
        canonical=_read(world,bounds)
        variants=[]
        for permutation in PERMUTATIONS:
            world.rebuild()
            output=_split(world,bounds,permutation)
            variants.append(dict(order=permutation, **_error(canonical,output)))
        _evict_graph(world.residual)
        revisited=_read(world,bounds)
        row=dict(seed=seed,bounds=bounds,
                 baseline_vs_canonical=_error(baseline,canonical),
                 canonical_permutations=variants,
                 canonical_after_eviction=_error(canonical,revisited),
                 finite=all(np.isfinite(arr).all() for arr in (baseline+canonical+revisited)))
        rows.append(row)
    root=_REPO_ROOT
    source_hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                   for name in ('terrain_inference.py','terrain_window_scheduler.py',
                                'tools/verification/verify_canonical_latents.py')}
    order_ok=all(v['height_max_m']==0 and all(x<=.001 for x in v['climate_max'])
                 for row in rows for v in row['canonical_permutations']+[row['canonical_after_eviction']])
    baseline_gate=all(row['baseline_vs_canonical']['height_max_m']<=1 and
                      row['baseline_vs_canonical']['coast_disagreements']==0
                      for row in rows)
    report=dict(profile=asdict(canonical_profile),cases=rows,
                thresholds=dict(canonical_order_height_m=0,canonical_order_climate=.001,
                                baseline_height_m=1,baseline_coast_disagreements=0),
                source_sha256=source_hashes,torch=torch.__version__,cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(),
                passed=bool(order_ok and baseline_gate and all(row['finite'] for row in rows)))
    path=RUNTIME/'canonical-latent-fidelity.json'
    path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(report=str(path),passed=report['passed'],cases=rows)),flush=True)
    if not report['passed']:
        raise AssertionError('Optional canonical latent profile gate failed')


if __name__=='__main__':
    main()

"""Direct production lake raster on a real cached decoder tile, no inference.

This numerical lake-only probe omits reach geometry deliberately. It is not
a final image or a claim that the source-lake shoreline is visually realistic.
"""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'backend'))
from terrain_hydrology_service import HydrologyService, DISPLAY_LAKES

ROOT = Path(__file__).resolve().parents[2]
out = ROOT/'output/render-v2-hydrology-reference'
manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
site = next(s for s in manifest['sites'] if s['id'] == 'lake-temperate')
capture = json.loads((out/'captures-reference/manifest.json').read_text(encoding='utf-8'))
assert capture['worldIdentity'] == manifest['worldIdentity']
x, y = site['center_m']
tx, ty = int(np.floor(x/7680)), int(np.floor(y/7680))
path = ROOT/'generated/natural-v1'/capture['runtimeCacheProfile']/manifest['generationProfile']/'physical-v1'/manifest['seed']/'0'/f'{tx}_{ty}.npy'
assert path.exists(), 'New-identity native tile missing: this numerical check never starts inference'
receipt = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
assert receipt['world_identity'] == manifest['worldIdentity']
assert receipt['stage'] == 'decoder' and receipt['resolution'] == 30
raw = np.load(path, allow_pickle=False)
world = HydrologyService._load(Path(manifest['currentWorldCache']))
assert world is not None
world.reaches = []  # Only the in-memory record is isolated; no persisted edit.
service = HydrologyService(lambda *_: None)
entry = dict(status='ready', key='real-lake-only-readonly', world=world,
             source=dict(settings={'world_topology': 'sphere'}), water_grid=None,
             reach_bounds=np.empty((0, 4)))
tx, ty = map(int, path.stem.split('_'))
bounds = (tx*7680-720, ty*7680-720, tx*7680+8400, ty*7680+8400)
try:
    before, old, _ = service.physical(entry, raw, bounds, lod=0)
    height, water, report = service.physical(entry, raw, bounds, lod=0, render_lakes=True, min_lake_area_m2=1e6)
    after, unchanged, _ = service.physical(entry, raw, bounds, lod=0)
    np.testing.assert_array_equal(height, raw)
    np.testing.assert_array_equal(before, after)
    np.testing.assert_array_equal(old, unchanged)
    wet = water[..., 2] == 2
    assert wet.any() and (raw[wet] < water[..., 0][wet]).all()
    np.testing.assert_array_equal(water[..., 3], 1)
    assert report['hydro_max_cut_m'] == 0 and not DISPLAY_LAKES
    x, y = site['center_m']
    ix, iy = int(np.floor(x/30))-tx*256+24, int(np.floor(y/30))-ty*256+24
    assert water[iy, ix, 2] == 2 and water[iy, ix, 1] > .99
    x0, y0 = site['source_center_m']
    jx, jy = int(np.floor(x0/30))-tx*256+24, int(np.floor(y0/30))-ty*256+24
    source_emerged = bool(raw[jy, jx] >= site['level_m'])
    if source_emerged:
        assert water[jy, jx, 2] != 2
    point_evidence = dict(path=str(path), world_identity=manifest['worldIdentity'],
        height_m=float(raw[iy, ix]), water_level_m=site['level_m'],
        depth_m=float(site['level_m']-raw[iy, ix]), resolution_m=30, stage='decoder')
    result = dict(ok=True, label='Real cached decoder30m, production lake-only raster, no visual acceptance',
                  tile=str(path), tileSha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  worldIdentity=manifest['worldIdentity'], nativeResolutionMetres=30,
                  receipt=str(path.with_suffix('.json')),
                  bounds_m=bounds, pixels=int(raw.size), wetLakePixels=int(wet.sum()),
                  wetLakeFraction=float(wet.mean()), demExactlyUnchanged=True, legacyExactlyUnchanged=True,
                  authorityEverywhere=True, maxCutMetres=report['hydro_max_cut_m'],
                  submergedCentre=point_evidence, sourceCentreHeightMetres=float(raw[jy, jx]),
                  sourceCentreEmerged=source_emerged, emergedOriginalStaysDry=source_emerged,
                  report=report)
    (out/('real-lake-raster-'+manifest['worldIdentity'][:8]+'.json')).write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('report','submergedCentre')}))
finally:
    service.pool.shutdown(wait=True)

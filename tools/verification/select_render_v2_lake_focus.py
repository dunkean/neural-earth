"""Find a real wet shore focus from existing native receipts; no server/model calls."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from terrain_hydrology_service import HydrologyService
from terrain_hydrology_geometry import prepare_water_grid, rasterize_water


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    base = ROOT / 'output/render-v2-hydrology-reference/captures-reference'
    report = json.loads((base / 'report.json').read_text(encoding='utf-8'))
    case = next(s for s in report['results'] if s['id'] == 'lake-arid-closed')
    frame = next(f for f in case['frames'] if f['zoom'] == '40km' and f['tag'] == 'v2-july')
    state, site = frame['state'], case['site']
    service = HydrologyService(lambda *args: None)
    world_identity = report['frozen']['worldIdentity']
    cache = service.root / hashlib.sha256(f'{world_identity}/{service.world_implementation}'.encode()).hexdigest() / 'world.npz'
    world = service._load(cache)
    service.pool.shutdown()
    assert world is not None, 'Existing hydrological world cache required'
    grid = prepare_water_grid(world.height_m, world.bounds, lakes=world.lakes,
                              lake_ids=world.lake_ids, ocean=world.ocean,
                              wrap_x=world.topology == 'sphere')
    profile = state['cache']['keys'][0].split('/')[3]
    tiles = ROOT / 'generated/natural-v1' / profile / state['generationProfile'] / 'physical-v1' / str(state['world']) / '0'
    old = np.asarray(site['center_m'])
    bounds = site['bounds_m']
    candidates, wet_count, measured, seen = [], 0, [], set()
    for key in state['cache']['keys']:
        parts = key.split('/')
        if 'physical' not in parts:
            continue
        k = parts.index('physical')
        if parts[k + 2] != '0':
            continue
        tx, ty = map(int, parts[k + 3:k + 5])
        if (tx, ty) in seen:
            continue
        seen.add((tx, ty))
        path = tiles / f'{tx}_{ty}.npy'
        receipt = json.loads(path.with_suffix('.json').read_text(encoding='utf-8'))
        assert receipt['world_identity'] == world_identity
        assert receipt['stage'] == 'decoder' and receipt['resolution'] == 30
        height = np.load(path, mmap_mode='r')[24:280, 24:280]
        tile_bounds = (tx * 7680, ty * 7680, (tx + 1) * 7680, (ty + 1) * 7680)
        xs = tile_bounds[0] + (np.arange(256) + .5) * 30
        ys = tile_bounds[1] + (np.arange(256) + .5) * 30
        inside = ((xs[None, :] >= bounds[0]) & (xs[None, :] < bounds[2])
                  & (ys[:, None] >= bounds[1]) & (ys[:, None] < bounds[3]))
        if not inside.any():
            continue
        water = rasterize_water([], world.lakes, height.shape, tile_bounds,
                                water_grid=grid, terrain=height,
                                sphere_radius_m=service.radius(world))
        owned = water['lake_index'] == site['record_id']
        depth = site['level_m'] - height
        wet = owned & (water['class'] == 2) & (depth > 0)
        wet_count += int((wet & inside).sum())
        # Stay within one native cell of a genuinely dry neighbour, so the
        # fixed 160m view can show both sides. Reject tile-border artefacts.
        distance = distance_transform_edt(wet) * 30
        choose = wet & inside & (distance <= 30) & (depth >= .25) & (depth <= 2)
        choose[:3] = choose[-3:] = False
        choose[:, :3] = choose[:, -3:] = False
        rr, cc = np.nonzero(choose)
        for row, col in zip(rr, cc):
            centre = [float(xs[col]), float(ys[row])]
            candidates.append(dict(center_m=centre, native_height_m=float(height[row, col]),
                                   depth_m=float(depth[row, col]), lake_index=int(water['lake_index'][row, col]),
                                   lake_coverage=float(water['still_coverage'][row, col]),
                                   offset_from_source_m=float(np.linalg.norm(np.asarray(centre) - old)),
                                   tile=str(path.relative_to(ROOT)), row=int(row + 24), column=int(col + 24),
                                   receipt=str(path.with_suffix('.json').relative_to(ROOT)),
                                   receipt_sha256=digest(path.with_suffix('.json'))))
        measured.append(str(path.relative_to(ROOT)))
    candidates.sort(key=lambda c: c['offset_from_source_m'])
    if candidates:
        selected = candidates[0]
        a = np.load(ROOT / selected['tile'], mmap_mode='r')
        row, col = selected['row'], selected['column']
        window = a[row - 2:row + 3, col - 2:col + 3]
        selected['native_150m_window'] = dict(
            shape=[5, 5], sample_spacing_m=30,
            pixels_below_lake_level=int((window < site['level_m']).sum()),
            pixels_at_or_above_lake_level=int((window >= site['level_m']).sum()),
            height_min_m=float(window.min()), height_max_m=float(window.max()))
    result = dict(label='CPU cached native wet-shore selection only; no visual acceptance',
                  source_run=report['runId'], world_identity=world_identity,
                  world_cache=str(cache), world_cache_sha256=digest(cache),
                  source_site=site, native_resolution_m=30, lake_id=site['record_id'],
                  level_m=site['level_m'], wet_native_pixels=wet_count,
                  measured_tiles=measured, selected=candidates[0] if candidates else None,
                  alternative_candidates=candidates[:8],
                  limitations=['No new terrain inference or server requests.',
                               'A native wet pixel proves level/ownership, not subpixel visual shore quality.',
                               'Original frozen six-site manifest and dry-centre review remain unchanged.'])
    out = ROOT / 'output/render-v2-hydrology-reference/arid-lake-wet-focus.json'
    out.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(out), wet_native_pixels=wet_count, selected=result['selected'])))


if __name__ == '__main__':
    main()

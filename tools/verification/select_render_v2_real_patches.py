"""Read cached Orogen arrays to shortlist real 40 km Render v2 review sites.

Never imports the server/generator, runs inference, opens a browser or changes
the cache. A 19.53 km source atlas is a selection aid, not decoder validation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import uniform_filter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from terrain_koppen import CODES, classify
from terrain_polar import to_chart


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--atlas', type=Path, required=True)
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--world-info', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/render-v2-review/real-patches-40km.json')
    args = parser.parse_args()
    recipe = json.loads(args.recipe.read_text(encoding='utf-8-sig'))
    info = json.loads(args.world_info.read_text(encoding='utf-8-sig'))
    settings, seed = recipe['generation']['settings'], recipe['generation']['seed']
    with np.load(args.atlas, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata'].item()))
        assert str(metadata['namespace']['request']['seed']) == str(seed)
        assert metadata['stage_state'] == info['generation_stages'], 'Atlas does not match loaded world stages'
        assert metadata['height_sha256'] == info['world_manifest']['bootstrap']['height_sha256']
        config = metadata['namespace']['request']['config']
        assert all(config[k.removeprefix('orogen_')] == v for k, v in settings.items()
                   if k.removeprefix('orogen_') in config), 'Atlas settings differ'
        height = archive['height_m']
        assert hashlib.sha256(np.ascontiguousarray(height).tobytes()).hexdigest() == metadata['height_sha256']
        reference = np.maximum(archive['layer_climate_height_m'] if 'layer_climate_height_m' in archive else height, 0)
        sand = archive['layer_soil_sand']
        fields = []
        for season in ('summer', 'winter'):
            rain = np.maximum(archive['layer_precip_' + season], 0)
            lapse = (settings['orogen_temperature_wet_lapse'] + settings['orogen_temperature_dry_lapse_extra'] * (1 - np.clip(rain, 0, 1))) / 1000
            sea_level = -45 + np.clip(archive['layer_temperature_' + season], 0, 1) * 90 + lapse * reference
            fields.extend((sea_level, rain * 1000, lapse))
    fields = np.asarray(fields, np.float32)
    classes = classify(height, fields, settings)
    h, w = height.shape
    bounds = info['world_bounds']
    dx, dy = (bounds[2]-bounds[0])/w, (bounds[3]-bounds[1])/h
    xx = bounds[0] + (np.arange(w)+.5)*dx
    yy = bounds[1] + (np.arange(h)+.5)*dy
    latitude = -yy[:, None]/(bounds[3]-bounds[1])*180
    land = height >= 0
    land_fraction = uniform_filter(land.astype(float), size=3, mode=('nearest', 'wrap'))
    smooth_height = uniform_filter(np.maximum(height, 0), size=3, mode=('nearest', 'wrap'))
    smooth_sand = uniform_filter(sand, size=3, mode=('nearest', 'wrap'))
    codes = ['ocean'] + [c.upper() for c in CODES]
    ids = {code: i for i, code in enumerate(codes)}

    def mask(*names):
        return np.isin(classes, [ids[n.upper()] for n in names])

    def sample(array, x, y):
        ix = ((x-bounds[0])/dx-.5) % w
        iy = np.clip((y-bounds[1])/dy-.5, 0, h-1)
        j, i = np.floor(ix).astype(int), np.floor(iy).astype(int)
        fx, fy = ix-j, iy-i
        a = array[..., i, j]*(1-fx)+array[..., i, (j+1) % w]*fx
        b = array[..., np.minimum(i+1, h-1), j]*(1-fx)+array[..., np.minimum(i+1, h-1), (j+1) % w]*fx
        return np.asarray(a*(1-fy)+b*fy, np.float32)

    cache = ROOT / 'generated/natural-v1' / info['cache_profile'] / info['generation_profile'] / 'physical-v1' / str(seed) / '0'
    available = {p.stem for p in cache.glob('*.npy') if not p.name.endswith('.climate.npy')}
    targets = [
        ('tropical-rainforest-low', mask('af','am'), 200, 500, None),
        ('tropical-savanna', mask('aw'), 350, 800, None),
        ('humid-temperate-low', mask('cfa','cfb'), 200, 500, None),
        ('temperate-montane', mask('cfb','cfc'), 1800, 700, None),
        ('mediterranean', mask('csa','csb'), 350, 700, None),
        ('continental-low', mask('dfa','dfb'), 350, 700, None),
        ('taiga-low', mask('dfc','dsc'), 400, 700, None),
        ('hot-desert', mask('bwh'), 350, 700, None),
        ('sandy-desert', mask('bwh','bwk'), 400, 1000, 'sand'),
        ('cold-desert-high', mask('bwk'), 1800, 1000, None),
        ('steppe', mask('bsh','bsk'), 650, 1000, None),
        ('high-mountain', land & (height > 4000), 5000, 1300, None),
        ('alpine-tundra', mask('et') & (height > 2500), 3500, 1200, None),
        ('polar-tundra-low', mask('et') & (np.abs(latitude) > 55), 250, 600, None),
        ('polar-ice', mask('ef') & (np.abs(latitude) > 60), 1200, 1500, None),
        ('warm-coast', mask('af','am','aw','cfa','cfb','csa') & (np.abs(latitude) < 50), 40, 300, 'coast'),
    ]
    selected = []
    delta = np.linspace(-20000, 20000, 33)
    for name, family, altitude, altitude_scale, special in targets:
        purity = uniform_filter(family.astype(float), size=3, mode=('nearest', 'wrap'))
        eligible = family & (land_fraction > .95)
        if special == 'coast':
            eligible = family & (land_fraction > .25) & (land_fraction < .85) & (height < 300)
        score = purity*4 - np.abs(smooth_height-altitude)/altitude_scale
        if special == 'sand':
            score += smooth_sand*8
        if special == 'coast':
            score -= np.abs(land_fraction-.6)*4
        for previous in selected:
            if not previous['available']:
                continue
            px, py = previous['center_m']
            distance = np.hypot(np.minimum(np.abs(xx[None,:]-px), bounds[2]-bounds[0]-np.abs(xx[None,:]-px)), yy[:,None]-py)
            eligible &= distance > 200000
        score = np.where(eligible, score, -np.inf)
        if not np.isfinite(score).any():
            selected.append(dict(id=name, available=False, reason='No eligible source-atlas site'))
            continue
        flat = int(np.argmax(score))
        row, col = np.unravel_index(flat, height.shape)
        x, y = float(xx[col]), float(yy[row])
        sx, sy = np.meshgrid(x+delta, y+delta)
        elevations = sample(height, sx, sy)
        climate = sample(fields, sx, sy)
        biomes = classify(elevations, climate, settings)
        land_pixels = elevations >= 0
        count = np.bincount(biomes.ravel(), minlength=31)
        dominant = int(np.argmax(count[1:])+1)
        positive = elevations[land_pixels]
        ts = np.clip(climate[0]-climate[2]*np.maximum(elevations,0), -45,45)
        tw = np.clip(climate[3]-climate[5]*np.maximum(elevations,0), -45,45)
        stats = lambda a: dict(min=float(np.min(a)), mean=float(np.mean(a)), max=float(np.max(a)))
        tx0, tx1 = int(np.floor((x-20000)/7680)), int(np.floor((x+20000)/7680))
        ty0, ty1 = int(np.floor((y-20000)/7680)), int(np.floor((y+20000)/7680))
        required = [f'{tx}_{ty}' for ty in range(ty0,ty1+1) for tx in range(tx0,tx1+1)]
        pcx, pcy = to_chart(x,y,settings)
        selected.append(dict(id=name, available=True, center_m=[x,y], bounds_m=[x-20000,y-20000,x+20000,y+20000],
            longitude_deg=x/(bounds[2]-bounds[0])*360, latitude_deg=-y/(bounds[3]-bounds[1])*180,
            polar_chart_center_m=[float(pcx),float(pcy)], source_atlas_pixel=[int(col),int(row)],
            land_fraction=float(land_pixels.mean()), dominant_biome=codes[dominant],
            dominance_land=float(count[dominant]/max(1,land_pixels.sum())),
            biome_fractions={codes[i]:float(n/count.sum()) for i,n in enumerate(count) if n},
            elevation_m=stats(positive) if len(positive) else None, ocean_depth_m=stats(-elevations[~land_pixels]) if np.any(~land_pixels) else None,
            temperature_north_summer_c=stats(ts[land_pixels]), temperature_north_winter_c=stats(tw[land_pixels]),
            rain_north_summer_halfyear_mm=stats(climate[1][land_pixels]), rain_north_winter_halfyear_mm=stats(climate[4][land_pixels]),
            sand_fraction=stats(sample(sand,sx,sy)[land_pixels]),
            native30m_cache=dict(path=str(cache), tiles_required=len(required), tiles_cached=sum(p in available for p in required),
                complete=all(p in available for p in required), missing_tiles=[p for p in required if p not in available])))
    inventory = []
    altitude_edges = [0, 200, 1000, 2500, 4000, float('inf')]
    for i, code in enumerate(codes[1:],1):
        values = height[(classes == i) & land]
        inventory.append(dict(biome=code, land_source_pixels=int(len(values)), present=bool(len(values)),
            altitude_source_pixels={f'{int(lo)}-{int(hi) if np.isfinite(hi) else "inf"}m':int(np.count_nonzero((values>=lo)&(values<hi)))
                for lo,hi in zip(altitude_edges[:-1],altitude_edges[1:])},
            elevation_m=None if not len(values) else dict(min=float(values.min()), median=float(np.median(values)), max=float(values.max()))))
    report = dict(seed=seed, world_identity=info['world_identity'], recipe_world_identity=recipe['generation']['worldIdentity'],
        generation_profile=info['generation_profile'], source_atlas=str(args.atlas.resolve()), source_cache_key=metadata['cache_key'],
        source_height_sha256=metadata['height_sha256'], stages=metadata['stage_state'], source_resolution_m=[dx,dy],
        patch_width_m=40000, patch_sampling_grid=33, source_verified=True,
        limitations=['Source selection only: 19.53 km source samples cannot validate 30 m decoder terrain or 40 km visual detail.',
            'All statistics refer to geographic chart squares, with 33x33 bilinear source samples; these are not 1089 independent physical samples.',
            'Polar-chart centres are supplied as optional camera equivalents; square extents and cache coverage here refer to the ordinary geographic chart.',
            'Season labels are northern-calendar half-years, not local summer/winter.',
            'Native cache availability refers only to the current exact runtime/generation namespace and ordinary chart, at selection time.'],
        inventory=inventory, patches=selected)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(args.output)
    for p in selected:
        if p['available']:
            print(f"{p['id']:25} {p['center_m']} {p['dominant_biome']:3} {p['dominance_land']:.1%} land {p['land_fraction']:.1%} h {p['elevation_m']['mean']:.0f}m sand {p['sand_fraction']['mean']:.2f} cache {p['native30m_cache']['tiles_cached']}/{p['native30m_cache']['tiles_required']}")


if __name__ == '__main__':
    main()

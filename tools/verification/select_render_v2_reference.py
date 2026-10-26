"""Prepare real-world Render v2 reference sites, or measure completed native DEM.

Source samples only select centres; they never count as visual/native proof.
This tool reads cached arrays and receipts and does not call a server/model.
"""
import copy
import json
from pathlib import Path
import sys

import numpy as np
import select_render_v2_real_matrix as qa

OUT = qa.ROOT/'output/render-v2-reference'
WORLD_INFO = Path(sys.argv[sys.argv.index('--world-info')+1]).resolve() if '--world-info' in sys.argv else None
runtime_world = json.loads(WORLD_INFO.read_text(encoding='utf-8')) if WORLD_INFO else qa.world
assert runtime_world['generation_stages'] == qa.world['generation_stages'], 'Source stage changed'
assert runtime_world['generation_settings'] == qa.world['generation_settings'], 'Generation settings changed'


def summary(a):
    a = np.asarray(a)
    if not a.size:
        return None
    q = np.quantile(a, [.1, .5, .6, .9, .95])
    return dict(min=float(a.min()), mean=float(a.mean()), max=float(a.max()), std=float(a.std()),
                p10=float(q[0]), median=float(q[1]), p60=float(q[2]), p90=float(q[3]), p95=float(q[4]))


def source_stats(center):
    x, y = center
    u, v = np.meshgrid(x+np.linspace(-20000, 20000, 33), y+np.linspace(-20000, 20000, 33))
    h = qa.sample(qa.height, u, v)
    f = qa.sample(qa.fields, u, v)
    c = qa.classify(h, f, qa.settings)
    land = h >= 0
    gy, gx = np.gradient(h, 1250, 1250)
    count = np.bincount(c[land], minlength=len(qa.codes))
    fractions = {qa.codes[i]: float(n/max(1, land.sum())) for i, n in enumerate(count) if i and n}
    dominant = max(fractions, key=fractions.get) if fractions else 'OCEAN'
    ts = f[0]-f[2]*np.maximum(h, 0)
    tw = f[3]-f[5]*np.maximum(h, 0)
    return dict(
        status='source-preselection-only-native-and-visual-review-required',
        resolution_m=[qa.dx, qa.dy], sampling_grid=[33, 33], independent_samples=False,
        land_fraction=float(land.mean()), elevation_land_m=summary(h[land]), ocean_depth_m=summary(-h[~land]),
        slope_land=summary(np.hypot(gx, gy)[land]), dominant_biome=dominant,
        biome_fractions_land=fractions, dominance_land=fractions.get(dominant, 0),
        temperature_north_july_c=summary(ts), temperature_north_january_c=summary(tw),
        rain_north_july_halfyear_mm=summary(f[1]), rain_north_january_halfyear_mm=summary(f[4]))


def measure_native(case):
    state = case['frames'][0]['state']
    site = case.get('site', case.get('selection'))
    bounds = site['bounds_m']
    profile = state['cache']['keys'][0].split('/')[3]
    base = qa.ROOT/'generated/natural-v1'/profile/state['generationProfile']/'physical-v1'/str(state['world'])/'0'
    values, slopes, climate_classes, receipts = [], [], [], []
    measured_tiles = set()
    for key in state['cache']['keys']:
        parts = key.split('/')
        physical = parts.index('physical') if 'physical' in parts else -1
        if physical < 0 or parts[physical+2] != '0':
            continue
        tx, ty = map(int, parts[physical+3:physical+5])
        if (tx, ty) in measured_tiles:
            continue
        measured_tiles.add((tx, ty))
        p = base/f'{tx}_{ty}'
        if not p.with_suffix('.npy').exists():
            continue
        receipt = json.loads(p.with_suffix('.json').read_text(encoding='utf-8'))
        assert receipt['stage'] == 'decoder' and receipt['resolution'] == 30
        assert receipt['world_identity'] == runtime_world['world_identity']
        a = np.load(p.with_suffix('.npy'), mmap_mode='r')
        core = a[24:280, 24:280]
        x = (tx*256+np.arange(256)+.5)*30
        y = (ty*256+np.arange(256)+.5)*30
        keep = ((y[:, None] >= bounds[1]) & (y[:, None] < bounds[3])
                & (x[None, :] >= bounds[0]) & (x[None, :] < bounds[2]))
        if not keep.any():
            continue
        gy, gx = np.gradient(a, 30, 30)
        values.append(core[keep])
        slopes.append(np.hypot(gx[24:280, 24:280], gy[24:280, 24:280])[keep])
        receipts.append(dict(path=str(p.with_suffix('.json')), stage=receipt['stage'],
                             resolution_m=receipt['resolution'], world_identity=receipt['world_identity']))
        ux, vy = np.meshgrid(x[::4], y[::4])
        climate_classes.append(qa.classify(core[::4, ::4], qa.sample(qa.fields, ux, vy), qa.settings)[keep[::4, ::4]])
    assert values, 'No completed decoder data for native measurement'
    h, slope = np.concatenate(values), np.concatenate(slopes)
    assert np.isfinite(h).all() and np.isfinite(slope).all(), 'Nonfinite native data'
    expected = ((int(np.ceil(bounds[2]/30-.5))-int(np.ceil(bounds[0]/30-.5)))
                *(int(np.ceil(bounds[3]/30-.5))-int(np.ceil(bounds[1]/30-.5))))
    assert h.size == expected, ('Native measurement is incomplete', h.size, expected)
    land = h >= 0
    hl, sl = h[land], slope[land]
    hs, ss = summary(hl), summary(sl)
    count = np.bincount(np.concatenate(climate_classes), minlength=len(qa.codes))
    fractions = {qa.codes[i]: float(n/max(1, count[1:].sum())) for i, n in enumerate(count) if i and n}
    dominant = max(fractions, key=fractions.get) if fractions else 'OCEAN'
    upper_flat = float(np.mean((hl >= hs['p60']) & (sl < .06))) if hs else None
    stats = dict(
        resolution_m=30, pixels=int(h.size), land_pixels=int(land.sum()), land_fraction=float(land.mean()),
        elevation_land_m=hs, ocean_depth_m=summary(-h[~land]), slope_land=ss,
        elevation_min_m=hs['min'] if hs else None, elevation_mean_m=hs['mean'] if hs else None,
        elevation_max_m=hs['max'] if hs else None, elevation_std_m=hs['std'] if hs else None,
        relief_p90_p10_m=hs['p90']-hs['p10'] if hs else None,
        slope_median=ss['median'] if ss else None, slope_p95=ss['p95'] if ss else None,
        upper_flat_fraction=upper_flat, steep_fraction=float(np.mean(sl > .2)) if sl.size else None,
        dominant_biome=dominant, biome_fractions_land=fractions,
        climate_classification_stride_m=120,
        climate_method='Continuous cached Orogen seasons at actual decoder30m heights, sampled every120m',
        receipts=receipts, receipt_stages=['decoder'])
    relief = qa.native_relief(stats) if hs else 'water'
    # Keep mixed coasts explicit, not an accidental mountain/plain verdict
    # based on a handful of emergent pixels.
    if 0 < stats['land_fraction'] < .5:
        stats['land_relief_only'] = relief
        relief = 'coast'
    return dict(id=case['id'], center_m=site['center_m'], stats=stats, relief=relief)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if '--measure' in sys.argv:
        source = Path(sys.argv[sys.argv.index('--measure')+1])
        data = json.loads(source.read_text(encoding='utf-8'))
        result = [measure_native(c) for c in data['results']]
        target = source.with_name(source.stem+'-metrics.json')
        target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(target)
        return
    paths = [
        ('matrix', 'output/render-v2-40km/real-matrix-24-selection.json', 'slots'),
        ('initial', 'output/render-v2-review/real-patches-40km.json', 'patches'),
        ('candidate', 'output/render-v2-40km/real-matrix/source-replacement-candidates.json', 'slots')]
    sites, by_center = [], {}

    def add(site, origin, path=None):
        site = copy.deepcopy(site)
        key = tuple(round(float(x), 3) for x in site['center_m'])
        evidence = dict(origin=origin, id=site['id'], source_file=path)
        if key in by_center:
            by_center[key]['aliases'].append(evidence)
            return
        x, y = site['center_m']
        stats = source_stats(site['center_m'])
        family = site.get('biomeFamily')
        if not family:
            family = next((name for name, codes in qa.families.items() if stats['dominant_biome'] in codes), stats['dominant_biome'])
        r = dict(id=site['id'], center_m=site['center_m'], bounds_m=[x-20000, y-20000, x+20000, y+20000],
                 aliases=[evidence], biomeFamily=family,
                 biome_codes=list(site.get('biome_codes', qa.families.get(family, (stats['dominant_biome'],)))),
                 requestedRelief=site.get('relief'), reliefCriteria=site.get('criteria'),
                 source_stats=stats, nativeStatsBaseline=site.get('native_stats'),
                 nativeStatus='pending-final-native-measurement', hydro=0,
                 dateConvention='Northern calendar: July0.5 / January0, not local summer/winter')
        if site.get('featureProvenance'):
            r['featureProvenance'] = site['featureProvenance']
        if WORLD_INFO:
            r['historicalNativeIdentity'] = qa.selection['world_identity']
            r['nativeStatus'] = 'new-runtime-native-measurement-required'
            if r.get('featureProvenance'):
                r['featureProvenance']['historicalWorldIdentity'] = qa.selection['world_identity']
                r['featureProvenance']['nativeStatus'] = 'historical-target-new-native-validation-required'
        sites.append(r)
        by_center[key] = r

    for origin, path, key in paths:
        data = json.loads((qa.ROOT/path).read_text(encoding='utf-8'))
        for site in data[key]:
            if site.get('available'):
                add(site, origin, path)
    for name, center in [
        ('user-green', [1881981.912660809, 150833.7780589524]),
        ('user-arid', [-2187601.526072075, -4076795.334798676]),
        ('user-pack', [14657326.440143395, 7251946.615556731])]:
        add(dict(id=name, center_m=center), 'exact-user-url')
    assert len(sites) == 55, ('Unexpected deduplication; inspect corpus', len(sites))
    add(dict(id='user-pack-ocean', center_m=[14657326.440143395, 7263946.615556731],
             featureProvenance=dict(
                 parentSite='user-compact-pack', offset_from_original_m=[0, 12000],
                 baselineImage='output/render-v2-40km/real/user-compact-pack-v2-july.png',
                 nativePointHeightMetres=-178.9532012939453,
                 nativeTile='generated/natural-v1/dd32b8d700669d12/orogen--g46261a320a644f43189ea849/physical-v1/7985403534069300490/0/1908_945.npy',
                 target='Actual ocean pack, not the inland original centre; austral winter July / summer January')),
        'targeted-existing-real-capture')
    add(dict(id='warm-coast-beach-detail', center_m=[3445155., 847305.],
             featureProvenance=dict(
                 parentSite='warm-coast', baselineImage='output/render-v2-40km/real/warm-coast-render3d-july.png',
                 nativePointHeightMetres=.12599632143974304, nativePointSlope=.00572418887168169,
                 nativeNearestOceanDistanceMetres=42.42640687119285,
                 nativeTile='generated/natural-v1/dd32b8d700669d12/orogen--g46261a320a644f43189ea849/physical-v1/7985403534069300490/0/448_110.npy',
                 target='Wide beach in the real Render3D baseline;160m view centred on the shoreline material')),
        'targeted-existing-real-capture')
    with np.load(qa.selection['source_atlas'], allow_pickle=False) as data:
        sand = data['layer_soil_sand'].copy()
    coast = (qa.land3 > .12) & (qa.land3 < .88)
    specs = [
        ('coast-arid', coast & np.isin(qa.classes, [qa.codes.index(c) for c in ('BWH', 'BWK')]), 1-abs(qa.land3-.5), 'coast'),
        ('coast-polar', coast & np.isin(qa.classes, [qa.codes.index(c) for c in ('ET', 'EF')]), 1-abs(qa.land3-.5), 'coast'),
        ('arid-sand-rich', (qa.land3 > .97) & np.isin(qa.classes, [qa.codes.index(c) for c in ('BWH', 'BWK')]), sand, None)]
    for name, eligible, score, relief in specs:
        for site in sites:
            x, y = site['center_m']
            eligible &= np.hypot(qa.xx[None, :]-x, qa.yy[:, None]-y) > 150000
        indices = np.flatnonzero(eligible)
        for index in indices[np.argsort(score.ravel()[indices])[::-1]][:1000]:
            row, col = np.unravel_index(index, qa.height.shape)
            center = [float(qa.xx[col]), float(qa.yy[row])]
            stats = source_stats(center)
            if relief == 'coast' and not (.1 < stats['land_fraction'] < .9):
                continue
            if relief == 'water' and stats['land_fraction'] != 0:
                continue
            add(dict(id=name, center_m=center, relief=relief), 'targeted-source-extra')
            break
        else:
            raise RuntimeError('No honest source candidate for '+name)
    assert len(sites) == 60
    manifest = dict(
        label='REAL GENERATED TERRAIN REFERENCE; source labels remain provisional until native measurement',
        seed=qa.selection['seed'], worldIdentity=runtime_world['world_identity'],
        generationProfile=runtime_world['generation_profile'], sourceAtlas=qa.selection['source_atlas'],
        sourceHeightSha256=qa.selection['source_height_sha256'], sourceCacheKey=qa.selection['source_cache_key'],
        generationStages=qa.meta['stage_state'], sourceResolutionMetres=[qa.dx, qa.dy],
        worldInfoFile=str(WORLD_INFO) if WORLD_INFO else None, runtimeCacheProfile=runtime_world['cache_profile'],
        provenanceTransition=dict(historicalWorldIdentity=qa.selection['world_identity'],
                                  currentWorldIdentity=runtime_world['world_identity'],
                                  sourceStagesUnchanged=True, neuralPixelIdentityClaim=False,
                                  oldReceipts='Historical only; never relabelled or migrated'),
        share='gHY4ZFzncjegep8JTpf.rulIR78fFpNhu4Tj2Y2', sites=sites,
        size=1024, zooms=[dict(id='153.6km-lod3', extentMetres=153600, lod=3, sourceResolutionMetres=240),
                          *[dict(id=name, extentMetres=extent, lod=0, sourceResolutionMetres=30)
                            for name, extent in [('40km', 40000), ('10km', 10000), ('2.56km', 2560), ('160m', 160)]]],
        styles=['v2-july', 'render3d-july', 'v2-january'], expectedFrames=900,
        limitations=[
            'All images must come from real generated DEM tiles; source statistics are preselection only.',
            '153.6km uses forcedLOD3 actual latent240m data, not enlarged decoder0.',
            '40km,10km,2.56km,160m all use decoder30m LOD0.160m at0.15625m/px enlarges shader detail, not submeter terrain.',
            'Native biome classification reconstructs cached source climate on decoder heights every120m; it is a measurement proxy, not field truth.',
            'Threshold morphology is recorded separately from visual acceptance; requested matrix reliefs are never silently relabelled as passes.',
            'Dates are northern calendar January/July. Southern seasonal reversal is expected.',
            'Hydrology disabled for this corpus; sea/DEM coastline remain visible. No inferred lake/river appearance claim.',
            'Render sea is enabled so pure-ocean cases also require actual requested sourceLOD; sea-freeze previews cannot pass.',
            'No final image is accepted if production code, world identity, source stage, camera, sourceLOD or pending state changes.'])
    target = OUT/'manifest.json'
    if target.exists():
        old = json.loads(target.read_text(encoding='utf-8'))
        if old['worldIdentity'] != manifest['worldIdentity']:
            archive = OUT/('manifest-historical-'+old['worldIdentity'][:8]+'.json')
            if not archive.exists():
                archive.write_text(json.dumps(old, indent=2)+'\n', encoding='utf-8')
    target.write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(target, 'sites', len(sites), 'expectedFrames', 900)


if __name__ == '__main__':
    main()

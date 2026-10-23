"""Read-only selection from the current persisted real-world water network.

No global or regional generation is started. API confirmations use LOD10 so
they cannot subscribe Base regional work. Missing monthly evidence stays a gap.
"""
import hashlib
import json
from pathlib import Path
import sys
import urllib.parse
import urllib.request

import numpy as np

import select_render_v2_real_matrix as qa
from terrain_hydrology_service import HydrologyService, DISPLAY_LAKES

OUT = qa.ROOT/'output/render-v2-hydrology-reference'
WORLD_INFO = Path(sys.argv[sys.argv.index('--world-info')+1]).resolve() if '--world-info' in sys.argv else None
runtime_world = json.loads(WORLD_INFO.read_text(encoding='utf-8')) if WORLD_INFO else qa.world
assert runtime_world['generation_stages'] == qa.world['generation_stages']
assert runtime_world['generation_settings'] == qa.world['generation_settings']
IDENTITY = runtime_world['world_identity']
SEED = str(qa.selection['seed'])
PROFILE = qa.selection['generation_profile']


def climate_at(point, height):
    x, y = np.asarray(point)
    f = qa.sample(qa.fields, x, y)
    h = np.float32(max(float(height), 0))
    c = int(qa.classify(np.asarray(h), f, qa.settings))
    return dict(biome=qa.codes[c], july_c=float(f[0]-f[2]*h), january_c=float(f[3]-f[5]*h),
                annual_precip_mm=float(f[1]+f[4]), height_m=float(height))


def main():
    OUT.mkdir(exist_ok=True)
    if '--prepare-only' in sys.argv:
        result = json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
        reference = json.loads((qa.ROOT/'output/render-v2-reference/manifest.json').read_text(encoding='utf-8'))
        prepare_capture(result, reference)
        print('Prepared capture manifest without API request')
        return
    service = HydrologyService(lambda *_: None)
    expected = service.root/hashlib.sha256(f'{IDENTITY}/{service.world_implementation}'.encode()).hexdigest()/'world.npz'
    impl = service.implementation
    service.pool.shutdown(wait=False)
    assert expected.exists(), 'Current global hydro cache missing: this selector refuses to initiate a build'
    world = HydrologyService._load(expected)
    assert world is not None
    assert int(world.metadata['parameters']['seed']) == int(SEED)
    np.testing.assert_array_equal(world.height_m, qa.height)
    reference = json.loads((qa.ROOT/'output/render-v2-reference/manifest.json').read_text(encoding='utf-8'))
    previous_centers = np.asarray([s['center_m'] for s in reference['sites']])

    def near_existing(point):
        return float(np.min(np.linalg.norm(previous_centers-np.asarray(point), axis=1)))

    rivers = []
    for reach in world.reaches:
        xy = np.asarray(reach['xy'])
        if len(xy) < 2 or reach.get('flow_m3s', 0) < .1:
            continue
        i = len(xy)//2
        point = xy[i]
        h = np.asarray(reach['water_m'])[i]
        climate = climate_at(point, h)
        q = np.asarray(reach.get('monthly_flow_m3s', ()), float)
        monthly = q.size == 12 and not np.allclose(q, q[0], rtol=1e-6, atol=1e-9)
        dry = np.flatnonzero(q < .01)+1 if monthly else np.array([], int)
        rivers.append(dict(id=int(reach['id']), key=str(reach['key']), center_m=point.tolist(), climate=climate,
                           annual_flow_m3s=float(reach['flow_m3s']), width_m=float(reach['width_m']),
                           depth_m=float(reach['depth_m']), monthly_flow_m3s=q.tolist(),
                           monthly_variation_proven=bool(monthly), dry_months=dry.tolist(),
                           nearest_reference_center_m=near_existing(point)))
    lakes = []
    for lake in world.lakes:
        if lake['area_m2'] < 1e6 or lake['max_depth_m'] < 2:
            continue
        cells = np.asarray(lake['cells'], int)
        if not cells.size:
            continue
        # Aim at an actual owned wet source cell, not a centroid on an island.
        local = int(np.argmin(world.height_m.ravel()[cells]))
        row, col = divmod(int(cells[local]), world.shape[1])
        point = [float(qa.xx[col]), float(qa.yy[row])]
        climate = climate_at(point, lake['level_m'])
        lakes.append(dict(id=int(lake['id']), key=str(lake['key']), center_m=point,
                          climate=climate, level_m=float(lake['level_m']),
                          area_m2=float(lake['area_m2']), max_depth_m=float(lake['max_depth_m']),
                          closed=bool(lake['closed']), inflow_m3s=float(lake['inflow_m3s']),
                          outflow_m3s=float(lake['outflow_m3s']),
                          nearest_reference_center_m=near_existing(point)))
    chosen, gaps = [], []
    specs = [
        ('river-humid', rivers, lambda r: r['climate']['biome'] in ('AF', 'AM', 'CFB') and r['annual_flow_m3s'] > 20
         and r['monthly_variation_proven'] and not r['dry_months']),
        ('river-temperate', rivers, lambda r: r['climate']['biome'] in ('CFB', 'CFC', 'DFA', 'DFB', 'DFC')
         and r['annual_flow_m3s'] > 20 and r['climate']['january_c'] < 5),
        ('river-arid-intermittent', rivers, lambda r: r['climate']['biome'] in ('BWH', 'BWK', 'BSH', 'BSK')
         and r['monthly_variation_proven'] and 0 < len(r['dry_months']) < 12 and max(r['monthly_flow_m3s']) > .1),
        ('lake-temperate', lakes, lambda r: r['climate']['biome'] in ('CFB', 'CFC', 'DFA', 'DFB', 'DFC')
         and max(r['climate']['july_c'], r['climate']['january_c']) > 7),
        ('lake-cold', lakes, lambda r: r['climate']['biome'] in ('ET', 'EF')
         and max(r['climate']['july_c'], r['climate']['january_c']) < 8),
        ('lake-arid-closed', lakes, lambda r: r['climate']['biome'] in ('BWH', 'BWK', 'BSH', 'BSK') and r['closed'])]
    for name, pool, predicate in specs:
        candidates = [r for r in pool if predicate(r)]
        if not candidates:
            gaps.append(dict(id=name, reason='No current cached record meets the climate/regime criterion'))
            continue
        selected = min(candidates, key=lambda r: r['nearest_reference_center_m'])
        site = dict(selected, id=name, record_id=selected['id'], record_type='lake' if name.startswith('lake') else 'river')
        if name == 'lake-temperate':
            if IDENTITY != qa.selection['world_identity']:
                site['source_center_m'] = site['center_m']
                site['center_m'] = [-4170045., -6748035.]
                site['historicalTarget'] = dict(world_identity=qa.selection['world_identity'],
                    note='Submerged point in the historical decoder; current-version native shore validation required')
                proof_path = OUT/('real-lake-raster-'+IDENTITY[:8]+'.json')
                if proof_path.exists():
                    proof = json.loads(proof_path.read_text(encoding='utf-8'))
                    assert proof['worldIdentity'] == IDENTITY and proof['ok']
                    site['nativePointEvidence'] = proof['submergedCentre']
                    site['nativePointEvidence']['proof'] = str(proof_path)
                    site['nativePointEvidence']['source_center_emerged_height_m'] = proof['sourceCentreHeightMetres']
            # Actual cached decoder point 124m from the source-cell centre.
            # The original centre is emerged and must remain dry.
            native = qa.ROOT/'generated/natural-v1/dd32b8d700669d12'/PROFILE/'physical-v1'/SEED/'0/-543_-879.npy'
            if native.exists() and IDENTITY == qa.selection['world_identity']:
                a = np.load(native, mmap_mode='r')
                point = [-4170045., -6748035.]
                ix = int(np.floor(point[0]/30))-(-543)*256+24
                iy = int(np.floor(point[1]/30))-(-879)*256+24
                actual = float(a[iy, ix])
                assert actual < site['level_m']-.5
                site['source_center_m'] = site['center_m']
                site['center_m'] = point
                site['nativePointEvidence'] = dict(path=str(native), height_m=actual,
                    water_level_m=site['level_m'], depth_m=site['level_m']-actual,
                    resolution_m=30, stage='decoder', world_identity=IDENTITY,
                    source_center_emerged_height_m=284.7728271484375,
                    note='Source centre stays dry; actual submerged point selected from same cached tile')
        x, y = site['center_m']
        bounds = [x-20000, y-20000, x+20000, y+20000]
        query = dict(world_profile=PROFILE, world_identity=IDENTITY, bounds=','.join(map(str, bounds)), lod=10, month=7)
        api_url = 'http://127.0.0.1:8765/api/hydrology/'+SEED+'?'+urllib.parse.urlencode(query)
        with urllib.request.urlopen(api_url, timeout=45) as response:
            data = json.load(response)
            site['apiEvidence'] = dict(url=api_url, http=response.status, status=data.get('status'), hydro_id=data.get('hydro_id'),
                                       reaches=len(data.get('reaches', [])), lakes=len(data.get('lakes', [])))
            records = data.get('lakes' if site['record_type'] == 'lake' else 'reaches', [])
            same = [r for r in records if r.get('id') == site['record_id']]
            site['apiEvidence']['selectedRecordDelivered'] = bool(same)
            (OUT/(name+'-network.json')).write_text(json.dumps(data, indent=2), encoding='utf-8')
        view = dict(share=reference['share'], gpu=1, layer='render-v2', view='map', position=','.join(map(str, site['center_m'])),
                    zoom=40000/1024, lod=0, size='1024,1024', v2season=.5, hydro=1, hydroMonth=7,
                    hydroIntermittent=1 if 'intermittent' in name else 0, hydroDetail=0, cache_gib=1, coarse_gpu=1)
        site['url'] = 'http://127.0.0.1:8765/?'+urllib.parse.urlencode(view)
        site['status'] = 'current-network-record-confirmed-native-shore-and-render-pending'
        chosen.append(site)
    result = dict(label='CURRENT REAL HYDRO NETWORK PRESELECTION; not final images or validated neural shorelines',
                  seed=SEED, worldIdentity=IDENTITY, generationProfile=PROFILE,
                  currentWorldCache=str(expected), worldCacheSha256=hashlib.sha256(expected.read_bytes()).hexdigest(),
                  displayImplementation=impl, sourceHeightExactMatch=True,
                  sourceHeightSha256=qa.selection['source_height_sha256'], physicalLakeRasterEnabled=DISPLAY_LAKES,
                  seasonality=world.metadata.get('seasonality'), sites=chosen, gaps=gaps,
                  limitations=['Monthly hydrographs are modeled synthetic unit forcing, not observations.',
                               'Global lake levels/ownership derive from19.53km source and do not validate30m neural shorelines.',
                               'DISPLAY_LAKES remains false for legacy; Render v2 uses a separate opt-in source-lake display grid.',
                               'Viewer ready, no fetch and pending0 do not prove physical tiles ready: require X-Terrain-Hydro-State ready.'])
    manifest_path = OUT/'manifest.json'
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text(encoding='utf-8'))
        if old['worldIdentity'] != IDENTITY:
            archive = OUT/('manifest-historical-'+old['worldIdentity'][:8]+'.json')
            if not archive.exists():
                archive.write_text(json.dumps(old, indent=2), encoding='utf-8')
    manifest_path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    prepare_capture(result, reference)
    print(json.dumps(dict(sites=len(chosen), gaps=gaps, output=str(OUT/'manifest.json'))))


def prepare_capture(result, reference):
    from select_render_v2_reference import source_stats
    capture_sites = []
    for site in result['sites']:
        x, y = site['center_m']
        capture_sites.append(dict(site, bounds_m=[x-20000, y-20000, x+20000, y+20000],
            hydro=1, hydroIntermittent='intermittent' in site['id'],
            biomeFamily=site['climate']['biome'], biome_codes=[site['climate']['biome']],
            requestedRelief=None, aliases=[dict(origin='current-physical-hydrology-record', id=site['record_id'])],
            source_stats=source_stats(site['center_m'])))
    capture = dict(reference, label='WORKING REAL HYDROLOGICAL COMPLEMENT; not the final900-frame corpus',
                   sites=capture_sites, expectedFrames=len(capture_sites)*15, hydrologyRequired=True,
                   hydrologyNetwork=result,
                   limitations=reference['limitations']+[
                       'This complement enables hydrology. Uploaded tile headers must be ready, independently of visible.pending.',
                       'Monthly flows are modeled, not observed; source river paths and lake levels may disagree with fine neural relief.',
                       'Lake raster is source-only, no Base pond activation. Raw DEM is unchanged and emerged cells stay dry.',
                       'Native-coarse LOD4 and above retains the legacy vector overlay, not V2 seasonal lake ice.',
                       'Screenshots composite the actual viewport including annotation overlay; renderer-only screenshots would omit legacy water.'])
    capture_dir = OUT/'captures-reference'
    capture_dir.mkdir(exist_ok=True)
    (capture_dir/'manifest.json').write_text(json.dumps(capture, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()

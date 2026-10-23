"""Search real cached 40km crops near a failed source target, no requests.

Proposals remain separate from the manifest and require a real visual replay.
Incomplete shifted windows are rejected by the canonical native measurement.
"""
import argparse
import copy
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--world-info', required=True)
parser.add_argument('--warm-dir', required=True)
parser.add_argument('--case', required=True)
args = parser.parse_args()
import select_render_v2_reference as ref

base = Path(args.warm_dir).resolve()
report = json.loads((base/'report.json').read_text(encoding='utf-8'))
manifest = json.loads((ref.qa.ROOT/'output/render-v2-reference/manifest.json').read_text(encoding='utf-8'))
assert report['selection']['world_identity'] == ref.runtime_world['world_identity'] == manifest['worldIdentity']
original = next(r for r in report['results'] if r['id']==args.case and r.get('ok'))
site = next(s for s in manifest['sites'] if s['id']==args.case)
family_codes = dict(AF=['AF','AM'], CFB=['CFB','CFC'], BSK=['BSK','BSH'],
                    ARID=['BWK','BWH'], DFC=['DFC','DSC'], POLAR=['ET','EF'])[site['biomeFamily']]
proposals, failures = [], []
for dy in [-2000, 0, 2000]:
    for dx in [-2000, 0, 2000]:
        if dx==0 and dy==0:
            continue
        x, y = site['center_m'][0]+dx, site['center_m'][1]+dy
        shifted = copy.deepcopy(site)
        shifted['center_m'] = [x, y]
        shifted['bounds_m'] = [x-20000, y-20000, x+20000, y+20000]
        case = dict(original, site=shifted, id=f'{site["id"]}-offset-{dx}-{dy}')
        try:
            measured = ref.measure_native(case)
            share = sum(measured['stats']['biome_fractions_land'].get(c, 0) for c in family_codes)
            proposals.append(dict(measured, originalId=site['id'], offset_m=[dx,dy],
                requestedRelief=site['requestedRelief'], familyDominance=share,
                targetNativeMatched=measured['relief']==site['requestedRelief'] and share>=.8,
                visualAcceptance='not-captured-pending-review'))
            print(json.dumps(dict(offset=[dx,dy], nativeRelief=measured['relief'],
                upperFlat=measured['stats']['upper_flat_fraction'], share=share)), flush=True)
        except AssertionError as error:
            failures.append(dict(offset_m=[dx,dy], reason=str(error)))
result = dict(label='REAL CACHED DEM CROP PROPOSALS; not captured or added to manifest',
    worldIdentity=manifest['worldIdentity'], originalId=site['id'], sourceProvenance=site['aliases'],
    proposals=proposals, incompleteWindows=failures)
target = base/(site['id']+'-cached-offset-proposals.json')
target.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps(dict(output=str(target), matched=sum(p['targetNativeMatched'] for p in proposals))))

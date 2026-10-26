"""Read-only source preselection for unfilled real-matrix morphology slots.

The 19.53 km source cannot prove native morphology. These candidates require
decoded DEM measurements and real captures before any acceptance or gallery.
No request is made to a server and no generation is launched.
"""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter

import select_render_v2_real_matrix as qa


def main():
    report = json.loads((qa.OUT/'real-matrix/report.json').read_text(encoding='utf-8'))
    gaps = [r for r in report['results']
            if not (r['acceptance']['nativeReliefMatched'] and r['representativeFamily'])]
    avoided = [r['selection']['center_m'] for r in report['results']]
    results = []
    for gap in gaps:
        family = gap['selection']['biomeFamily']
        relief = gap['selection']['relief']
        names = qa.families[family]
        family_mask = np.isin(qa.classes, [qa.codes.index(n) for n in names]) & (qa.height >= 0)
        purity = uniform_filter(family_mask.astype(float), size=3, mode=qa.mode)
        valid = family_mask & (qa.land3 > .97) & (purity > .70)
        if relief == 'plain':
            # Avoid clipped high plateaux: the decoder turned the old flat
            # 6000 m EF source into mountains, not a native plain.
            valid &= (qa.height < 1000) & (qa.std3 < 40) & (qa.range3 < 150) & (qa.max_slope < .008)
            score = purity*8-qa.std3/30-qa.height/2000
        elif relief == 'plateau':
            valid &= ((qa.tpi > 120) & (qa.slope < .010) & (qa.range3 > 350)
                      & (qa.range3 < 1400) & (qa.max_slope > 2.5*np.maximum(qa.slope, .002)))
            if family == 'POLAR':
                valid &= qa.height < 4500
            score = purity*8+np.minimum(qa.tpi, 700)/500-qa.slope*60
        else:
            continue
        for x, y in avoided:
            valid &= np.hypot(qa.xx[None, :]-x, qa.yy[:, None]-y) > 100000
        indices = np.flatnonzero(valid)
        indices = indices[np.argsort(score.ravel()[indices])[::-1]][:500]
        candidates = []
        for index in indices:
            row, col = np.unravel_index(index, qa.height.shape)
            center = [float(qa.xx[col]), float(qa.yy[row])]
            if any(np.hypot(center[0]-c['center_m'][0], center[1]-c['center_m'][1]) < 120000
                   for c in candidates):
                continue
            stats = qa.source_stats(center)
            share = sum(stats['biome_fractions_land'].get(c, 0) for c in names)
            if share < .8 or stats['land_fraction'] < .97:
                continue
            if relief == 'plain' and stats['relief_p90_p10_m'] >= 130:
                continue
            if relief == 'plateau':
                if stats['relief_p90_p10_m'] < 150:
                    continue
                # Reject low basin centres even when the broad TPI was high.
                if qa.height[row, col] < stats['elevation_mean_m']:
                    continue
            candidates.append(dict(
                id=gap['id']+'-candidate-'+str(len(candidates)+1).zfill(2),
                replaces=gap['id'], center_m=center, biomeFamily=family,
                biome_codes=list(names), relief=relief,
                bounds_m=[center[0]-20000, center[1]-20000, center[0]+20000, center[1]+20000],
                source_stats=stats, family_dominance_source=share,
                criteria=gap['selection']['criteria'], available=True,
                status='source-preselection-only-native-morphology-and-visual-review-required'))
            if len(candidates) == 3:
                break
        results.append(dict(id=gap['id'], biomeFamily=family, relief=relief,
                            candidates=candidates, available=bool(candidates)))
        print(gap['id'], len(candidates), 'source candidates; native unverified', flush=True)
    result = dict(
        label='SOURCE PRESELECTION ONLY; not validation and not generated native tiles',
        world_identity=qa.selection['world_identity'],
        source_atlas=qa.selection['source_atlas'],
        source_height_sha256=qa.selection['source_height_sha256'],
        generation_stages=qa.meta['stage_state'], source_resolution_m=[qa.dx, qa.dy],
        native_cache_search='Seven missing slots had no matching fully cached40km crop; raw failures remain unchanged.',
        families={k: list(v) for k, v in qa.families.items()}, reliefs=qa.reliefs,
        results=results, slots=[c for r in results for c in r['candidates']])
    target = qa.OUT/'real-matrix/source-replacement-candidates.json'
    target.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(target)


if __name__ == '__main__':
    main()

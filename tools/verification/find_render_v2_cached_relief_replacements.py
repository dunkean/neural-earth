"""Search real completed decoder rectangles for missing matrix reliefs.

All candidate40km crops fit existing cached30m tiles. Screening uses integral
sums; acceptance is recomputed by the original exact native measurement.
"""
import copy
import json
from pathlib import Path

import numpy as np
import select_render_v2_real_matrix as qa

OUT=qa.OUT/'real-matrix'
report=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
old=json.loads((qa.OUT/'real/report.json').read_text(encoding='utf-8'))
oldstats={n['id']:n['stats'] for n in json.loads((qa.OUT/'real/readonly-dem-metrics.json').read_text(encoding='utf-8'))}
targets=[r for r in report['results'] if not(r['acceptance']['nativeReliefMatched'] and r['representativeFamily'])]
pool=[];seen=set()
for c in report['results']+old['results']:
    if not c.get('frames'):continue
    center=tuple(c['selection']['center_m'])
    if center in seen:continue
    seen.add(center);c=copy.deepcopy(c);c['nativeStats']=c.get('nativeStats') or oldstats.get(c['id'])
    if c['nativeStats']:pool.append(c)


def windows(a,n):
    p=np.pad(np.cumsum(np.cumsum(a,dtype=np.float64,axis=0),dtype=np.float64,axis=1),((1,0),(1,0)))
    return p[n:,n:]-p[:-n,n:]-p[n:,:-n]+p[:-n,:-n]


results=[]
for target in targets:
    family=target['selection']['biome_codes'];relief=target['selection']['relief'];candidates=[];tested=0
    for parent in pool:
        share=sum(parent['nativeStats']['biome_fractions_land'].get(c,0) for c in family)
        if share<.4:continue
        state=parent['frames'][0]['state'];coords=[tuple(map(int,k.split('/')[-2:])) for k in state['cache']['keys']]
        xmin=min(x for x,y in coords);xmax=max(x for x,y in coords);ymin=min(y for x,y in coords);ymax=max(y for x,y in coords)
        grid=np.full(((ymax-ymin+1)*256,(xmax-xmin+1)*256),np.nan,np.float32)
        base=qa.ROOT/'generated/natural-v1'/state['cache']['keys'][0].split('/')[3]/state['generationProfile']/'physical-v1'/str(state['world'])/'0'
        for tx,ty in coords:
            p=base/f'{tx}_{ty}.npy'
            if p.exists():grid[(ty-ymin)*256:(ty-ymin+1)*256,(tx-xmin)*256:(tx-xmin+1)*256]=np.load(p,mmap_mode='r')[24:280,24:280]
        valid=np.isfinite(grid);clean=np.nan_to_num(grid);gy,gx=np.gradient(clean,30,30);slope=np.hypot(gx,gy)
        n=int(np.ceil(40000/30));count=n*n
        if min(grid.shape)<n:continue
        coverage=windows(valid,n);mean=windows(clean,n)/count
        std=np.sqrt(np.maximum(0,windows(clean.astype(np.float64)**2,n)/count-mean**2))
        land=windows(clean>=0,n)/count;eligible=(coverage==count)&(land>.97)
        if relief=='plain':
            steep=windows(slope>.15,n)/count;eligible&=(std<45)&(steep<.05)
            score=-steep*10-std/500
        elif relief=='plateau':
            q60=np.quantile(grid[valid&(grid>=0)],.6)
            flat=windows((clean>=q60)&(slope<.06),n)/count;steep=windows(slope>.2,n)/count
            eligible&=(flat>.15)&(steep>.06)&(std>100)
            score=flat-np.abs(steep-.18)*.1
        else:continue
        score=np.where(eligible,score,-np.inf);indices=np.flatnonzero(np.isfinite(score));indices=indices[np.argsort(score.ravel()[indices])[::-1]]
        trial_centres=[]
        for index in indices:
            row,col=np.unravel_index(index,score.shape);center=[xmin*7680+int(col)*30+20000,ymin*7680+int(row)*30+20000]
            if any(np.hypot(center[0]-p[0],center[1]-p[1])<1200 for p in trial_centres):continue
            trial_centres.append(center);tested+=1
            trial=copy.deepcopy(parent);trial['id']=target['id'];trial['selection']['center_m']=center
            trial['selection']['bounds_m']=[center[0]-20000,center[1]-20000,center[0]+20000,center[1]+20000]
            native=qa.native_stats(trial);actual=qa.native_relief(native)
            family_share=sum(native['biome_fractions_land'].get(c,0) for c in family)
            if actual==relief and family_share>=.8:
                candidates.append(dict(id=target['id'],center_m=center,bounds_m=trial['selection']['bounds_m'],native_stats=native,source_stats=qa.source_stats(center),
                    family_dominance=family_share,parent_cached_case=parent['id'],shift_m=float(np.hypot(center[0]-parent['selection']['center_m'][0],center[1]-parent['selection']['center_m'][1])),
                    status='native-measured-matching-crop-needs-live-capture-and-visual-review'))
                break
            if len(trial_centres)>=6:break
    results.append(dict(id=target['id'],relief=relief,biomeFamily=target['selection']['biomeFamily'],exact_native_crops_tested=tested,candidates=candidates))
    print(target['id'],'tested',tested,'matched',[(c['center_m'],c['parent_cached_case']) for c in candidates],flush=True)
target=OUT/'cached-relief-replacements.json';target.write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8');print(target)

"""Search already-decoded cache rectangles for strict replacement plain crops.

No model/server imports. Keeps the existing native plain thresholds; it does
not relabel a rolling site. Returned centres still require exact measurement.
"""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/render-v2-40km/real-matrix'
report=json.loads((OUT/'report.json').read_text(encoding='utf-8'))


def windows(a,n):
    p=np.pad(np.cumsum(np.cumsum(a,dtype=np.float64,axis=0),dtype=np.float64,axis=1),((1,0),(1,0)))
    return p[n:,n:]-p[:-n,n:]-p[n:,:-n]+p[:-n,:-n]


output=[]
for case in report['results']:
    if case['selection']['relief']!='plain' or not case.get('protocolOk') or case.get('acceptance',{}).get('nativeReliefMatched'):continue
    state=case['frames'][0]['state'];coords=[tuple(map(int,k.split('/')[-2:])) for k in state['cache']['keys']]
    xmin=min(x for x,y in coords);xmax=max(x for x,y in coords);ymin=min(y for x,y in coords);ymax=max(y for x,y in coords)
    grid=np.full(((ymax-ymin+1)*256,(xmax-xmin+1)*256),np.nan,np.float32)
    base=ROOT/'generated/natural-v1'/state['cache']['keys'][0].split('/')[3]/state['generationProfile']/'physical-v1'/str(state['world'])/'0'
    for tx,ty in coords:
        a=np.load(base/f'{tx}_{ty}.npy',mmap_mode='r')
        grid[(ty-ymin)*256:(ty-ymin+1)*256,(tx-xmin)*256:(tx-xmin+1)*256]=a[24:280,24:280]
    valid=np.isfinite(grid);clean=np.nan_to_num(grid);gy,gx=np.gradient(clean,30,30);slope=np.hypot(gx,gy)
    n=int(np.ceil(40000/30));count=n*n
    coverage=windows(valid,n);mean=windows(clean,n)/count;std=np.sqrt(np.maximum(0,windows(clean.astype(np.float64)**2,n)/count-mean**2))
    steep=windows(slope>.15,n)/count;land=windows(clean>=0,n)/count
    eligible=(coverage==count)&(std<45)&(steep<.05)&(land>.97)
    score=np.where(eligible,-steep*10-std/500,-np.inf);indices=np.flatnonzero(np.isfinite(score));indices=indices[np.argsort(score.ravel()[indices])[::-1]]
    candidates=[]
    for index in indices:
        row,col=np.unravel_index(index,score.shape);center=[xmin*7680+col*30+20000,ymin*7680+row*30+20000]
        if any(np.hypot(center[0]-c['center_m'][0],center[1]-c['center_m'][1])<1200 for c in candidates):continue
        candidates.append(dict(center_m=center,bounds_m=[center[0]-20000,center[1]-20000,center[0]+20000,center[1]+20000],
            window_std_m=float(std[row,col]),window_fraction_slope_over_15percent=float(steep[row,col]),window_land_fraction=float(land[row,col])))
        if len(candidates)==6:break
    output.append(dict(id=case['id'],candidates=candidates,cache_bounds_m=[xmin*7680,ymin*7680,(xmax+1)*7680,(ymax+1)*7680],
        note='Full40km crops inside existing decoder cache. Prefix screen only; exact p10/p90 and family reconstructed on native heights still required.'))
target=OUT/'cached-plain-replacement-candidates.json';target.write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')
print(target)
for case in output:print(case['id'],len(case['candidates']),[(c['center_m'],round(c['window_fraction_slope_over_15percent'],4)) for c in case['candidates']])

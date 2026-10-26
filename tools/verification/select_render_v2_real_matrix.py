"""Select an explicit 6-biome-family x 4-relief matrix from cached data only.

Source-only morphology is provisional at 19.53 km. Missing slots remain
missing, rather than relabelling an unrelated climate or relief as a pass.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import uniform_filter, minimum_filter, maximum_filter, map_coordinates

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from terrain_koppen import CODES, classify

OUT = ROOT/'output/render-v2-40km'
selection_file = Path(os.environ.get('REVIEW_SELECTION', ROOT/'output/render-v2-review/real-patches-40km.json'))
selection = json.loads(selection_file.read_text(encoding='utf-8-sig'))
for target, source in (('world_identity', 'worldIdentity'), ('source_atlas', 'sourceAtlas'),
                       ('source_height_sha256', 'sourceHeightSha256'), ('source_cache_key', 'sourceCacheKey')):
    if target not in selection and source in selection:
        selection[target] = selection[source]
world_file = Path(selection.get('worldInfoFile', ROOT/'output/render-v2-review/selection-world-info.json'))
world = json.loads(world_file.read_text(encoding='utf-8-sig'))
measure_path = Path(sys.argv[sys.argv.index('--measure')+1]) if '--measure' in sys.argv else None
completed = json.loads((measure_path or OUT/'real/report.json').read_text(encoding='utf-8'))
settings = world['generation_settings']
with np.load(selection['source_atlas'],allow_pickle=False) as z:
    meta=json.loads(str(z['metadata'].item()))
    assert meta['stage_state']==world['generation_stages']
    assert meta['height_sha256']==selection['source_height_sha256']
    height=z['height_m']
    reference=np.maximum(z['layer_climate_height_m'] if 'layer_climate_height_m' in z else height,0)
    fields=[]
    for season in ('summer','winter'):
        rain=np.maximum(z['layer_precip_'+season],0)
        lapse=(settings['orogen_temperature_wet_lapse']+settings['orogen_temperature_dry_lapse_extra']*(1-np.clip(rain,0,1)))/1000
        fields.extend((-45+np.clip(z['layer_temperature_'+season],0,1)*90+lapse*reference,rain*1000,lapse))
fields=np.asarray(fields,np.float32)
classes=classify(height,fields,settings)
codes=['ocean']+[c.upper() for c in CODES]
families={
    'AF':('AF','AM'), 'CFB':('CFB','CFC'), 'BSK':('BSK','BSH'),
    'ARID':('BWH','BWK'), 'DFC':('DFC','DSC'), 'POLAR':('ET','EF'),
}
reliefs=['plain','rolling','plateau','mountains']
b=world['world_bounds'];hh,ww=height.shape;dx=(b[2]-b[0])/ww;dy=(b[3]-b[1])/hh
xx=b[0]+(np.arange(ww)+.5)*dx;yy=b[1]+(np.arange(hh)+.5)*dy
positive=np.maximum(height,0);mode=('nearest','wrap')
mean3=uniform_filter(positive,size=3,mode=mode)
std3=np.sqrt(np.maximum(0,uniform_filter(positive*positive,size=3,mode=mode)-mean3*mean3))
range3=maximum_filter(positive,size=3,mode=mode)-minimum_filter(positive,size=3,mode=mode)
tpi=positive-uniform_filter(positive,size=7,mode=mode)
gy,gx=np.gradient(height,dy,dx);slope=np.hypot(gx,gy)
max_slope=maximum_filter(slope,size=3,mode=mode)
land3=uniform_filter((height>=0).astype(float),size=3,mode=mode)


def sample(a,x,y):
    ix=((x-b[0])/dx-.5)%ww;iy=np.clip((y-b[1])/dy-.5,0,hh-1)
    j,i=np.floor(ix).astype(int),np.floor(iy).astype(int);fx,fy=ix-j,iy-i
    top=a[...,i,j]*(1-fx)+a[...,i,(j+1)%ww]*fx
    bot=a[...,np.minimum(i+1,hh-1),j]*(1-fx)+a[...,np.minimum(i+1,hh-1),(j+1)%ww]*fx
    return np.asarray(top*(1-fy)+bot*fy,np.float32)


def source_stats(center):
    x,y=center;u,v=np.meshgrid(x+np.linspace(-20000,20000,33),y+np.linspace(-20000,20000,33))
    h=sample(height,u,v);f=sample(fields,u,v);c=classify(h,f,settings);land=h>=0
    gy,gx=np.gradient(h,1250,1250);s=np.hypot(gx,gy);a=h[land]
    q=np.quantile(a,[.1,.5,.9]);count=np.bincount(c.ravel(),minlength=31);dominant=int(np.argmax(count[1:])+1)
    row=int(np.clip(round((y-b[1])/dy-.5),0,hh-1));col=int(round((x-b[0])/dx-.5))%ww
    ts=f[0]-f[2]*np.maximum(h,0);tw=f[3]-f[5]*np.maximum(h,0)
    return dict(land_fraction=float(land.mean()),elevation_min_m=float(a.min()),elevation_mean_m=float(a.mean()),elevation_max_m=float(a.max()),
        elevation_std_m=float(a.std()),relief_p90_p10_m=float(q[2]-q[0]),slope_median=float(np.median(s[land])),slope_p95=float(np.quantile(s[land],.95)),
        center_slope=float(slope[row,col]),regional_tpi_136km_m=float(tpi[row,col]),context_range_58km_m=float(range3[row,col]),
        dominant_biome=codes[dominant],dominance_land=float(count[dominant]/max(1,land.sum())),
        biome_fractions_land={codes[i]:float(n/max(1,land.sum())) for i,n in enumerate(count) if i and n},
        temperature_north_summer_c=float(np.mean(ts[land])),temperature_north_winter_c=float(np.mean(tw[land])),
        rain_north_summer_halfyear_mm=float(np.mean(f[1][land])),rain_north_winter_halfyear_mm=float(np.mean(f[4][land])))


def native_stats(case):
    state=case['frames'][0]['state'];bounds=case['selection']['bounds_m']
    base=ROOT/'generated/natural-v1'/state['cache']['keys'][0].split('/')[3]/state['generationProfile']/'physical-v1'/str(state['world'])/'0'
    values=[];slopes=[];climate_classes=[];receipts=[]
    for key in state['cache']['keys']:
        tx,ty=map(int,key.split('/')[-2:]);p=base/f'{tx}_{ty}'
        if not p.with_suffix('.npy').exists():return None
        receipt=json.loads(p.with_suffix('.json').read_text());assert receipt['stage']=='decoder' and receipt['resolution']==30 and receipt['world_identity']==selection['world_identity']
        a=np.load(p.with_suffix('.npy'),mmap_mode='r');core=a[24:280,24:280]
        x=(tx*256+np.arange(256)+.5)*30;y=(ty*256+np.arange(256)+.5)*30
        keep=(y[:,None]>=bounds[1])&(y[:,None]<bounds[3])&(x[None,:]>=bounds[0])&(x[None,:]<bounds[2])
        if not keep.any():continue
        gy,gx=np.gradient(a,30,30);s=np.hypot(gx[24:280,24:280],gy[24:280,24:280])
        values.append(core[keep]);slopes.append(s[keep]);receipts.append(str(p.with_suffix('.json')))
        # Sample the same continuous source climate at the actual decoder
        # heights, every 120 m. No quantised renderer aux textures involved.
        ux,vy=np.meshgrid(x[::4],y[::4]);small=core[::4,::4];valid=keep[::4,::4]
        climate_classes.append(classify(small,sample(fields,ux,vy),settings)[valid])
    if not values:return None
    h=np.concatenate(values);s=np.concatenate(slopes);land=h>=0;a=h[land];sl=s[land]
    q=np.quantile(a,[.1,.5,.6,.9]);count=np.bincount(np.concatenate(climate_classes),minlength=31)
    d=int(np.argmax(count[1:])+1)
    return dict(elevation_min_m=float(a.min()),elevation_mean_m=float(a.mean()),elevation_max_m=float(a.max()),elevation_std_m=float(a.std()),
        relief_p90_p10_m=float(q[3]-q[0]),slope_median=float(np.median(sl)),slope_p95=float(np.quantile(sl,.95)),
        upper_flat_fraction=float(np.mean((h>=q[2])&(s<.06))),steep_fraction=float(np.mean(s>.2)),land_fraction=float(land.mean()),
        dominant_biome=codes[d],biome_fractions_land={codes[i]:float(n/max(1,count[1:].sum())) for i,n in enumerate(count) if i and n},
        climate_classification_stride_m=120,climate_method='Continuous cached Orogen seasons reconstructed on actual 30m DEM; sampled every120m',
        pixels=int(len(h)),receipts=len(receipts),receipt_stages=['decoder'],resolution_m=30)


def native_relief(n):
    if n['elevation_std_m']<45 and n['relief_p90_p10_m']<130 and n['slope_p95']<.15:return 'plain'
    if n['upper_flat_fraction']>.19 and n['steep_fraction']>.08 and n['relief_p90_p10_m']>300:return 'plateau'
    if n['elevation_std_m']>160 and n['relief_p90_p10_m']>450 and n['slope_p95']>.30:return 'mountains'
    if n['elevation_std_m']>=40 and n['relief_p90_p10_m']>=120:return 'rolling'
    return None


def main():
    native=[]
    for case in completed['results']:
        if case.get('ok'):
            n=native_stats(case)
            if n:native.append(dict(id=case['id'],center_m=case['selection']['center_m'],stats=n,relief=native_relief(n),source_stats=source_stats(case['selection']['center_m'])))
    native_target = measure_path.with_name('native-metrics.json') if measure_path else OUT/'real/readonly-dem-metrics.json'
    native_target.write_text(json.dumps(native,indent=2)+'\n',encoding='utf-8')
    if measure_path:
        print(native_target)
        sys.exit(0)

    criteria={
        'plain':'Source context58km std<45m, range<180m, maximum slope<.008; exact40km relief p90-p10<130m. Native: std<45m, relief<130m, slopep95<.15.',
        'rolling':'Source context58km std45..250m and range150..1000m; exact40km relief100..700m. Native: std>=40m and relief>=120m, below mountain thresholds.',
        'plateau':'Source raised flat-centre proxy: regionalTPI136km>80m, centre slope<.012, context range300..1400m and slope contrast>2. Exact40km relief>=120m. Native: >=19% upper-flat area, >=8% slopes>.2, relief>300m. Requires visual confirmation of a plateau.',
        'mountains':'Source context58km std>300m, range>1000m, max slope>.015; exact40km relief>500m. Native: std>160m, relief>450m and slopep95>.30.',
    }
    slots=[];used=[]
    for family,names in families.items():
        ids=[codes.index(n) for n in names];family_mask=np.isin(classes,ids)&(height>=0)
        purity=uniform_filter(family_mask.astype(float),size=3,mode=mode)
        for relief in reliefs:
            slot=dict(id=family.lower()+'-'+relief,biomeFamily=family,biome_codes=list(names),relief=relief,criteria=criteria[relief])
            existing=[n for n in native if n['relief']==relief and sum(n['stats']['biome_fractions_land'].get(c,0) for c in names)>=.8 and n['id'] not in [u.get('reused_site') for u in slots]]
            if existing:
                n=max(existing,key=lambda n:sum(n['stats']['biome_fractions_land'].get(c,0) for c in names))
                slot.update(available=True,status='native-measured-needs-visual-relief-review',center_m=n['center_m'],source_stats=source_stats(n['center_m']),native_stats=n['stats'],reused_site=n['id'])
            else:
                valid=family_mask&(land3>.97)&(purity>.55)
                if relief=='plain':valid&=(std3<45)&(range3<180)&(max_slope<.008);score=purity*5-std3/45
                elif relief=='rolling':valid&=(std3>=45)&(std3<250)&(range3>150)&(range3<1000);score=purity*5-np.abs(std3-130)/200
                elif relief=='plateau':valid&=(tpi>80)&(slope<.012)&(range3>300)&(range3<1400)&(max_slope>2*np.maximum(slope,.003));score=purity*5+np.minimum(tpi,500)/500-slope*50
                else:valid&=(std3>300)&(range3>1000)&(max_slope>.015);score=purity*5+np.minimum(std3,900)/900
                for x,y in used:
                    valid&=np.hypot(xx[None,:]-x,yy[:,None]-y)>60000
                indices=np.flatnonzero(valid)
                candidates=indices[np.argsort(score.ravel()[indices])[-200:][::-1]] if len(indices) else []
                chosen=None
                for index in candidates:
                    row,col=np.unravel_index(index,height.shape);center=[float(xx[col]),float(yy[row])];s=source_stats(center)
                    family_share=sum(s['biome_fractions_land'].get(c,0) for c in names)
                    if family_share<.8 or s['land_fraction']<.97:continue
                    r=s['relief_p90_p10_m']
                    if relief=='plain' and r>=130:continue
                    if relief=='rolling' and not 100<=r<=700:continue
                    if relief=='plateau' and r<120:continue
                    if relief=='mountains' and r<=500:continue
                    chosen=(center,s);break
                if chosen:slot.update(available=True,status='source-only-provisional-needs-decoder-and-visual-review',center_m=chosen[0],source_stats=chosen[1],native_stats=None)
                else:slot.update(available=False,status='missing',center_m=None,source_stats=None,native_stats=None,reason='No candidate satisfying family dominance and relief criteria in cached source. Do not substitute an incompatible biome or morphology.')
            if slot['available']:
                x,y=slot['center_m'];used.append((x,y));slot['bounds_m']=[x-20000,y-20000,x+20000,y+20000]
                slot['family_dominance']=sum((slot['native_stats'] or slot['source_stats'])['biome_fractions_land'].get(c,0) for c in names)
            slots.append(slot)
    report=dict(worldIdentity=selection['world_identity'],seed=selection['seed'],sourceCacheKey=selection['source_cache_key'],sourceResolutionMetres=[dx,dy],patchExtentMetres=40000,
        families=families,reliefs=reliefs,criteria=criteria,slots=slots,
        limitations=['24 explicit crossed slots, not24 arbitrary sites. Missing slots are deliberately unfilled.',
            'Source-only shape is provisional: 19.53km atlas cannot certify40km meso/micro-relief. Context metrics use58.59/136.72km windows; selected patch metrics use40km.',
            'Native morphology thresholds are screening measurements; a geological plateau especially requires visual confirmation.',
            'Source climate classifications are reconstructed on decoder heights when cached. Effective biome may differ from source-atlas labels.',
            'No inference, server mutation, browser or GPU work performed by this selector.'])
    target=OUT/'real-matrix-24-selection.json';target.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(target)
    for s in slots:print(s['id'],s['status'],s.get('center_m'),None if not s['available'] else (s['native_stats'] or s['source_stats'])['dominant_biome'])


if __name__ == '__main__':
    main()

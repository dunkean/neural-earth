"""CPU reference / PNG fallback for the GPU surface material compositor."""
import json
import numpy as np
from terrain_soil import coordinates, noise, seed_value, sample_soil
from terrain_koppen import sample_classes, seasonal_fields

DEFAULTS = dict(season=.5,variation=1.,forest=1.,rock_slope=40.,snow=1.,
                vegetation_tint=[1.,1.,1.],rock_tint=[1.,1.,1.],snow_color=[.94,.96,.97])
LIMITS = dict(season=(0,1),variation=(0,2),forest=(0,2),rock_slope=(15,75),snow=(0,2))
# Eight muted material families, independent of the informational biome palette.
FAMILIES = np.array([[.12,.27,.13,.94,.08],[.35,.40,.19,.55,.15],
                     [.65,.56,.37,.08,0],[.47,.46,.26,.42,.1],
                     [.20,.32,.16,.86,.72],[.16,.27,.19,.88,.15],
                     [.40,.43,.28,.5,0],[.39,.40,.37,0,0]],np.float32)
FAMILY_IDS = np.array([4,0,0,1,2,2,3,3,4,4,4,1,1,1,4,4,4,
                       4,4,5,5,4,4,5,5,4,4,5,5,6,7])


def parse_settings(value=None):
    if isinstance(value,str):
        value = json.loads(value)
    if value is not None and not isinstance(value,dict):
        raise ValueError('Render settings must be an object')
    value = value or {}
    if set(value)-set(DEFAULTS):
        raise ValueError('Unknown Render settings')
    result = {k:list(v) if isinstance(v,list) else v for k,v in DEFAULTS.items()}
    for key,val in value.items():
        if key in LIMITS:
            lo,hi = LIMITS[key]
            if isinstance(val,bool) or not isinstance(val,(int,float)) or not np.isfinite(val) or not lo<=val<=hi:
                raise ValueError('Invalid Render '+key)
            result[key] = float(val)
        else:
            if not isinstance(val,(tuple,list)) or len(val)!=3 or any(isinstance(v,bool) or not isinstance(v,(int,float)) or not np.isfinite(v) or not 0<=v<=1 for v in val):
                raise ValueError('Invalid Render '+key)
            result[key] = list(val)
    return result


def smooth(a,b,x):
    t = np.clip((x-a)/(b-a),0,1)
    return t*t*(3-2*t)


def mix(a,b,t):
    return a*(1-t[...,None])+b*t[...,None]


def surface_material(ids, height, slope, curvature, point, footprint, soil, seasons,
                     alpine, snowline, salt, settings=None, north=None, water=None):
    """No hydrology is fabricated. Optional water = proximity, wetness in [0,1]."""
    s = parse_settings(settings)
    broad = np.full_like(height,.5)
    if footprint<6000:
        broad = .5+(noise(point/12000,salt)-.5)*(1-smooth(3000,6000,footprint))
    patch = np.full_like(height,.5);holes=patch.copy();fine=patch.copy();micro=patch.copy()
    for scale,offset,name in ((1600,11,'patch'),(180,23,'holes'),(25,37,'fine'),(12,53,'micro')):
        if footprint<scale/2:
            warped = point+(broad[...,None]-.5)*np.array([900.,675.,-225.]) if name=='patch' else point
            value = .5+(noise(warped/scale+offset,salt+offset)-.5)*(1-smooth(scale/4,scale/2,footprint))
            if name=='patch':patch=value
            elif name=='holes':holes=value
            elif name=='fine':fine=value
            else:micro=value
    fam = FAMILIES[FAMILY_IDS[ids]]
    ts = np.clip(seasons[0]-seasons[2]*np.maximum(height,0),-45,45)
    tw = np.clip(seasons[3]-seasons[5]*np.maximum(height,0),-45,45)
    phase = .5-.5*np.cos(2*np.pi*s['season'])
    temp = tw*(1-phase)+ts*phase
    moisture = np.clip((seasons[1]+seasons[4])/2200,0,1)
    grade = np.linalg.norm(slope,axis=-1)
    concave = np.clip(curvature,-1,1)
    wet = np.zeros_like(height) if water is None else np.clip(water[1],0,1)
    bank = np.zeros_like(height) if water is None else np.clip(water[0],0,1)
    soil_color = soil[:3].transpose(1,2,0)*(1-.2*wet[...,None]-.08*bank[...,None])
    # Clearings include low herbs, not uniformly naked brown earth. Substrate
    # humus supports that cover; dry / cold areas still expose their true soil.
    herbs = smooth(-6,8,np.maximum(ts,tw))*moisture*(.35+.25*soil[8])*(1-smooth(.45,1.4,grade))
    soil_color = mix(soil_color,np.array([.40,.43,.25]),herbs)
    rock = soil[3:6].transpose(1,2,0)*np.asarray(s['rock_tint'])
    vegetation = fam[...,:3]*np.asarray(s['vegetation_tint'])
    local_season = (s['season']+np.where(ts<tw,.5,0))%1
    autumn = smooth(.58,.72,local_season)*(1-smooth(.83,.97,local_season))*fam[...,4]
    vegetation = mix(vegetation,np.array([.47,.34,.15]),autumn*.45)
    dry = np.clip((temp-12)/22,0,1)*(1-moisture)
    vegetation = mix(vegetation,np.array([.51,.46,.25]),dry*.35)
    forest_alt = (1-smooth(np.maximum(alpine*1000-500,0),np.maximum(alpine*1000+500,1000),height))*smooth(0,12,np.maximum(ts,tw))
    density = np.clip(fam[...,3]*s['forest']*(.72+.28*moisture)*forest_alt*(.65+.7*broad)+.1*bank,0,1)
    clearing = smooth(.19,.74,.62*patch+.38*holes+.2*(broad-.5))
    cover = np.clip(density*smooth(.12,.72,clearing+density*.3-.16)*(1-smooth(.45,1.4,grade)),0,1)
    color = mix(soil_color,vegetation,cover)
    threshold = np.tan(np.deg2rad(s['rock_slope']))
    exposed = smooth(threshold*.65,threshold*1.5,grade)
    highland = smooth(np.maximum(alpine*1000,0),np.maximum(alpine*1000+1600,1600),height)*.65
    rock_cover = np.clip(np.maximum(exposed,highland)+.12*concave+.12*(holes-.5),0,1)
    # Small strata / regional variation, not a second independent noise stack.
    rock *= (1+s['variation']*(.12*(broad-.5)+.08*(fine-.5)))[...,None]
    color = mix(color,rock,rock_cover)
    aspect = np.zeros_like(height) if north is None else np.sum(slope*north,axis=-1)/np.maximum(grade,1e-5)*np.sign(point[...,1])
    cold = smooth(3,-5,temp)
    local_snowline = snowline*1000+(holes-.5)*300+(broad-.5)*400
    altitude = smooth(np.maximum(local_snowline-250,0),np.maximum(local_snowline+700,700),height)
    snow_potential = np.clip(np.maximum(cold*.85,altitude)*s['snow']*(1+.15*aspect-.12*concave+.25*(holes-.5)),0,1)
    # Cold high mountains retain patchy snow to ~70 degrees; bare cliffs still
    # emerge. Concavities accumulate, convex exposed ridges shed it.
    slip = .65+.55*altitude+.25*cold
    retention = 1-smooth(slip,slip+1.4,grade)
    snow = np.clip(snow_potential*retention*(1-.2*wet),0,1)
    color = mix(color,np.asarray(s['snow_color']),snow)
    color *= (1+s['variation']*(.15*(broad-.5)+.12*(patch-.5)+.08*(fine-.5)+.04*(micro-.5)))[...,None]
    ocean = mix(np.array([.24,.40,.49]),np.array([.035,.10,.22]),smooth(0,4500,-height))
    return np.clip(np.where((height<0)[...,None],ocean,color),0,1)


def colorize_surface(world,xs,ys,elevation,resolution,generation_settings,settings=None,*,polar=False,mode='render'):
    soil = sample_soil(world,xs,ys,generation_settings,polar=polar)
    if mode=='soil':
        ocean = np.broadcast_to([.06,.17,.27],(*elevation.shape,3))
        return np.where((elevation<0)[...,None],ocean,soil[:3].transpose(1,2,0))
    from terrain_biomes import sample_biome_fields
    seasons = seasonal_fields(world,xs,ys,generation_settings,polar=polar)
    # Material families/treelines describe the regional climate before its
    # altitude adjustment. Reclassifying into ET/EF first would suddenly lower
    # the snowline and create hard rings at the class temperature boundaries.
    regional_height = np.zeros_like(elevation)
    ids = sample_classes(world,xs,ys,generation_settings,regional_height,polar=polar)
    biome = sample_biome_fields(world,xs,ys,generation_settings,regional_height,polar=polar)
    point = coordinates(xs,ys,world.bounds,generation_settings.get('world_topology','sphere')!='plane',polar)
    # numpy gradient edge values are outside the cropped tile's physical halo.
    spacing = (resolution,resolution) if np.isscalar(resolution) else (resolution[1],resolution[0])
    dy,dx = np.gradient(elevation,*spacing)
    slope = np.stack((dx,dy),axis=-1)
    step = max(spacing)
    curvature = (4*elevation-np.roll(elevation,1,0)-np.roll(elevation,-1,0)-np.roll(elevation,1,1)-np.roll(elevation,-1,1))/step
    ny,nx = np.gradient(point[...,1],*spacing)
    north = np.stack((nx,ny),axis=-1)
    north /= np.maximum(np.linalg.norm(north,axis=-1,keepdims=True),1e-5)
    if generation_settings.get('world_topology','sphere')=='plane':north.fill(0)
    return surface_material(ids,elevation,slope,curvature,point,step,soil,seasons,
                            biome[3],biome[4],seed_value(getattr(getattr(world,'source',world),'seed',0)),settings,north)

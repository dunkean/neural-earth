"""CPU reference / PNG fallback for the GPU surface material compositor.

The WGSL in ``terrain_renderer.js`` mirrors this module operation for operation.
Materials derive from continuous climate (no class IDs, so no hard biome
borders), terrain derivatives (slope, local relief position) and gradient noise
in canonical geographic coordinates. Continuous coverage is modulated gently
at its edges; unresolved detail converges to the underlying coverage.
"""

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path
import json
import hashlib
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from terrain_soil import coordinates, seed_value, sample_soil
from terrain_koppen import seasonal_fields

DEFAULTS = dict(season=.5,variation=1.,forest=1.,moisture=0.,rock_slope=40.,snow=1.,
                vegetation_tint=[1.,1.,1.],rock_tint=[1.,1.,1.],snow_color=[.94,.96,.97])
LIMITS = dict(season=(0,1),variation=(0,2),forest=(0,2),moisture=(-1,1),rock_slope=(15,75),snow=(0,2))
NOISE_NORM = 5.25  # unit standard deviation for the gradient noise below
F = np.float32


def appearance_identity(root=None):
    """Display revision, separate from the immutable neural terrain identity."""
    root=Path(root) if root is not None else REPO_ROOT
    digest=hashlib.sha256()
    for name in ('terrain_render.py','terrain_renderer.js','terrain_render_controls.js','terrain_soil.py','terrain_pedology.py'):
        digest.update(name.encode());digest.update((source_path(name, root=root)).read_bytes())
    return digest.hexdigest()[:16]


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
    t = np.clip((np.asarray(x,F)-F(a))/F(b-a),0,1)
    return t*t*(3-2*t)


def mix(a,b,t):
    t = np.asarray(t,F)
    return np.asarray(a,F)*(1-t[...,None])+np.asarray(b,F)*t[...,None]


def _hash(c, salt):
    v = c.astype(np.uint32)
    with np.errstate(over='ignore'):
        h = v[...,0]*np.uint32(0x9e3779b9)^v[...,1]*np.uint32(0x85ebca6b)^v[...,2]*np.uint32(0xc2b2ae35)^np.uint32(salt&0xffffffff)
        h = (h^(h>>16))*np.uint32(0x7feb352d)
        h = (h^(h>>15))*np.uint32(0x846ca68b)
        return h^(h>>16)


def gradient_noise(p, salt):
    """3D gradient noise with quintic fade, unit standard deviation."""
    p = np.asarray(p,F)
    fl = np.floor(p); c = fl.astype(np.int32); f = p-fl
    u = f*f*f*(f*(f*6-15)+10)

    def corner(x,y,z):
        o = np.array([x,y,z],np.int32)
        h = _hash(c+o,salt)
        g = np.stack((h&1023,(h>>10)&1023,(h>>20)&1023),axis=-1).astype(F)/F(511.5)-1
        return np.sum(g*(f-o.astype(F)),axis=-1)
    lerp = lambda a,b,t:a+(b-a)*t
    a = lerp(lerp(corner(0,0,0),corner(1,0,0),u[...,0]),lerp(corner(0,1,0),corner(1,1,0),u[...,0]),u[...,1])
    b = lerp(lerp(corner(0,0,1),corner(1,0,1),u[...,0]),lerp(corner(0,1,1),corner(1,1,1),u[...,0]),u[...,1])
    return lerp(a,b,u[...,2])*F(NOISE_NORM)


def fbm(point, scale, octaves, footprint, salt):
    """Returns (unit-variance pattern of resolved octaves, retained variance fraction).

    Octaves under 8 samples per wavelength fade out, then are skipped.
    """
    value = np.zeros(point.shape[:-1],F); kept = total = 0.; s = scale; a = 1.
    for i in range(octaves):
        total += a*a
        if footprint < s*.25:
            fade = 1-float(smooth(s*.125,s*.25,footprint))
            value = value+F(a*fade)*gradient_noise(point/F(s),salt+i*0x9e37)
            kept += (a*fade)**2
        a *= .5; s *= .5
    return (value/F(np.sqrt(kept)) if kept>0 else value), F(kept/total)


def cover(fraction, pattern, sharpness=.35):
    """Bounded, continuous variation, strongest at partial coverage.

    Unlike a thresholded noise mask this cannot turn half cover into binary
    blotches. Symmetric noise has zero mean, including while octaves fade out.
    ``sharpness`` is the modulation amplitude (0..1), not a threshold width.
    """
    value, detail = pattern
    c = np.clip(np.asarray(fraction,F),0,1)
    v = value/np.sqrt(1+value*value)*np.sqrt(detail)
    return c+F(np.clip(sharpness,0,1))*c*(1-c)*v


def surface_material(height, gradient, tpi, point, north, footprint, soil, seasons,
                     salt, settings=None, bare=False, pedology=None):
    """Satellite-like albedo. ``seasons`` are sea-level ts, ps, lapse_s, tw, pw, lapse_w.

    ``bare`` returns the substrate (soil + exposed bedrock) used by the Soil layer.
    """
    s = parse_settings(settings)
    var, forest_amount, rock_tan = F(s['variation']), F(s['forest']), F(np.tan(np.deg2rad(s['rock_slope'])))
    fp = float(footprint)
    # Palette feedback concerns LOD 1/2 (60/120 m), not the broader view.
    material_detail = F(float(smooth(30,60,fp))*(1-float(smooth(120,240,fp))))
    phase = F(.5-.5*np.cos(2*np.pi*s['season']))
    h = np.maximum(height,0).astype(F)
    macro = fbm(point,64000.,4,fp,salt+1)
    macro2 = fbm(point+np.array([5101.,-2297.,3313.],F),48000.,4,fp,salt+2)
    patch = fbm(point,4800.,4,fp,salt+3)
    crag = fbm(point,700.,4,fp,salt+4)
    fine = fbm(point,240.,4,fp,salt+5)
    province = fbm(point,900000.,3,fp,salt+6)
    m1 = macro[0]*np.sqrt(macro[1]); m2 = macro2[0]*np.sqrt(macro2[1])
    pv = province[0]*np.sqrt(province[1])
    ts0, ps, ls, tw0, pw, lw = seasons
    ts = np.clip(ts0-ls*h,-45,45); tw = np.clip(tw0-lw*h,-45,45)
    # At map scale the climate/relief supplies the structure, not noise blobs.
    locality = F(1-float(smooth(600,6000,fp)))
    terrain = smooth(.015,.12,np.abs(tpi))
    jitter = F(.8)*m2*var*locality*(F(.25)+F(.75)*terrain)
    t_hot = np.maximum(ts,tw)+jitter; t_cold = np.minimum(ts,tw)+jitter
    t_now = tw+(ts-tw)*phase
    # Koppen-like aridity threshold; valleys gather water, ridges drain.
    pth = np.maximum(F(20.)*(ts+tw)*F(.5)+F(280.),F(120.))
    wet_shift = np.exp(F(.6)*F(s['moisture'])+F(.10)*var*m1*locality).astype(F)
    valley = smooth(0,-.12,tpi); ridge = smooth(0,.12,tpi)
    humid = np.maximum(ps+pw,0)/pth*wet_shift
    humid = humid*(1+F(.7)*valley*(1-smooth(1.5,3,humid))-F(.18)*ridge)
    humid_now = np.maximum(pw+(ps-pw)*phase,0)*F(2.)/pth*wet_shift*(1+F(.7)*valley-F(.18)*ridge)
    grade = np.linalg.norm(gradient,axis=-1).astype(F)*F(max(fp,30.)/30.)**F(.3)
    arid = 1-smooth(.35,1.,humid)
    w_trop = smooth(13,19,t_cold); w_boreal = 1-smooth(-14,-3,t_cold)

    # Substrate: transported regional soil, zonal soils, geological provinces.
    ground = np.moveaxis(soil[:3] if bare or pedology is None else pedology[:3],0,-1).astype(F)
    sand = mix([.80,.66,.46],[.70,.50,.33],smooth(-.3,.5,pv))
    desert = mix([.55,.46,.36],sand,smooth(-.4,.4,m2+F(1.5)*soil[6]-F(.5)-F(.8)*grade/rock_tan))
    # Soil retains its exposed-substrate rendering. Render's earth hues come
    # from regional pedology, rather than replacing it with a climate palette.
    ground = mix(ground,desert,arid*F(.8 if bare else 0))
    ground = mix(ground,[.50,.29,.17],w_trop*smooth(1,2,humid)*F(.6 if bare else 0))
    steppe = (1-arid)*(1-w_trop)*(1-w_boreal)*smooth(.8,1.2,humid)*(1-smooth(1.6,2.4,humid))
    ground = mix(ground,[.25,.21,.16],steppe*F(.5 if bare else 0))
    ground = ground*(1-F(.12)*valley+F(.08)*ridge)[...,None]
    ground = ground*(1+var*(F(.035)*m1*locality+F(.025)*patch[0]*np.sqrt(patch[1])+F(.02)*fine[0]*np.sqrt(fine[1])))[...,None]
    rock = mix([.50,.48,.45],[.30,.29,.28],smooth(.15,.6,pv))
    rock = mix(rock,[.66,.62,.55],smooth(-.15,-.6,pv))
    # Arid outcrops: dark desert varnish, or red sandstone in some provinces.
    rock = mix(rock,mix([.36,.30,.25],[.60,.38,.25],smooth(.1,.6,m2)),arid*F(.85))
    rock = mix(rock,np.moveaxis(soil[3:6],0,-1),F(.35))
    if not bare:
        # Keep regional mineral hues, but halve their chroma at equal luminance.
        neutral = np.sum(rock*np.array([.2126,.7152,.0722],F),axis=-1)
        rock = mix(rock,neutral[...,None],F(.5)*material_detail)
    rock = rock*np.asarray(s['rock_tint'],F)
    # Horizontal strata, resolved only on fine tiles; strongest in arid ranges.
    strata = np.sin(F(2*np.pi)*(h+F(120.)*patch[0])/F(90.))*F(1-float(smooth(25,45,fp)))
    rock = rock*(1+var*F(.10)*strata*(F(.15)+F(.85)*arid)+var*F(.07)*crag[0]*np.sqrt(crag[1]))[...,None]
    rock_tan_eff = rock_tan*(F(.55)+F(.45)*smooth(.3,1.3,humid))
    alpine = 1-smooth(2,9,t_hot)
    exposure = (smooth(rock_tan_eff*.5,rock_tan_eff*1.3,grade)+F(.45)*alpine*smooth(rock_tan_eff*.15,rock_tan_eff*.7,grade)
                +F(.35)*ridge*smooth(.35,.9,grade/rock_tan)-F(.3)*valley)
    outcrop = cover(np.clip(exposure,0,1),(F(.75)*crag[0]+F(.25)*patch[0]*np.sqrt(patch[1]),crag[1]))
    color = mix(ground,rock,outcrop)
    if bare:
        return _ocean(height,color,t_now,m1,var,s['snow'],phase)

    # Vegetation: grasses/shrubs over soil, canopy patches over grasses.
    tint = np.asarray(s['vegetation_tint'],F)
    green = smooth(.5,1.4,humid_now)*smooth(3,12,t_now)
    grass = mix([.60,.53,.34],[.29,.37,.16],green)
    grass = mix(grass,[.40,.38,.27],1-smooth(7,14,t_hot))*tint
    herbs = smooth(.2,.85,humid)*smooth(0,7,t_hot)*(1-smooth(rock_tan*.8,rock_tan*1.4,grade))
    # Grasses and shrubs thin out diffusely; only canopy forms crisp stands.
    sward = fine if fine[1]>0 else patch
    color = mix(color,grass,cover(herbs*(1-outcrop),sward,.12*float(var)))
    dry_forest = 1-smooth(1.3,3,humid)
    canopy = mix([.20,.28,.14],[.16,.255,.13],w_trop)
    canopy = mix(canopy,[.15,.22,.16],w_boreal)
    canopy = mix(canopy,[.27,.29,.17],dry_forest*F(.75))
    deciduous = (1-w_trop)*(1-F(.7)*w_boreal)*(1-F(.5)*dry_forest)
    local = (F(s['season'])+np.where(ts>=tw,F(0),F(.5)))%1
    autumn = smooth(.66,.76,local)*(1-smooth(.84,.92,local))
    canopy = mix(canopy,[.42,.27,.11],autumn*deciduous*F(.6))
    leafless = smooth(8,2,t_now)*deciduous
    canopy = mix(canopy,[.36,.33,.28],leafless*F(.75))
    highland = (1-smooth(10,17,t_hot))*material_detail
    # Reuse resolved rock/canopy noise: no extra FBM evaluations or reads.
    canopy_detail = np.clip(F(.12)*highland*crag[0]*np.sqrt(crag[1])
                           +F(.025)*fine[0]*np.sqrt(fine[1]),-.18,.18)
    canopy = canopy*tint*((1-F(.18)*highland)*(1+var*canopy_detail))[...,None]
    trees = (smooth(.65,2.1,humid)*smooth(5.5,13.5,t_hot+F(.8)*valley)*forest_amount*
             (1-smooth(rock_tan*.9,rock_tan*1.6,grade))*(1-outcrop))
    # Terrain carries valley stands; noise only adds gentle local clearings.
    trees = trees*(1-F(.22)*ridge)+F(.14)*valley*smooth(2,10,t_hot)*smooth(.4,1.2,humid)*(1-outcrop)*forest_amount
    woods = cover(np.clip(trees,0,1),patch,.65*float(var))
    color = mix(color,canopy,woods)
    beach = (1-smooth(1.5,6,height))*(1-smooth(.05,.2,grade))*F(1-float(smooth(40,120,fp)))
    color = mix(color,[.76,.70,.56],beach*F(.8)*(1-F(.5)*w_boreal))

    # Snow: seasonal and perennial, exposure-aware, sheds from steep ridges.
    gnorm = np.linalg.norm(gradient,axis=-1)
    # Solar preference reverses continuously across the equator, not by sign.
    hemisphere = point[...,1]/np.maximum(np.sqrt(point[...,1]**2+F(.0064)*np.sum(point*point,axis=-1)),1)
    aspect = np.sum(gradient*north,axis=-1)/np.maximum(gnorm,1e-5)*hemisphere
    sun = (aspect*np.minimum(gnorm,1)).astype(F)
    t_snow = t_now+F(2.2)*sun+F(.45)*m1*var*locality+F(.25)*ridge-F(.35)*valley
    # Snow needs supply: cold deserts stay mostly bare, wet ranges glaciate.
    supply = smooth(60,400,np.maximum(ps+pw,0)*wet_shift*(1+F(.5)*valley-F(.25)*ridge))
    perennial = smooth(2.5,-2.5,np.maximum(ts,tw)+F(1.5)*sun+F(.35)*m1*var*locality)*supply
    seasonal = smooth(1,-4,t_snow)*smooth(.15,.5,humid)*supply
    snow = np.clip(np.maximum(seasonal,perennial)*F(s['snow']),0,1)
    # Physical slope, NOT LOD-amplified grade: a coarse ridge is not a cliff.
    # Broad retention curve and adhered snow avoid bare, contour-like rings.
    slip = smooth(F(1.25)+F(.65)*perennial,F(3.5)+perennial,gnorm)
    snow = snow*(1-(F(.88)-F(.18)*perennial)*slip)*(1-F(.3)*woods*(1-perennial)*(1-F(.7)*leafless))
    color = mix(color,np.asarray(s['snow_color'],F),cover(snow,crag,.35*float(var)))
    color = color*(1-F(.06)*valley)[...,None]
    return _ocean(height,color,t_now,m1,var,s['snow'],phase)


def _ocean(height, color, t_now, m1, var, snow_amount, phase):
    d = np.maximum(-np.asarray(height,F),0)
    water = mix([.10,.30,.36],[.045,.16,.27],smooth(5,120,d))
    water = mix(water,[.018,.065,.15],smooth(150,3000,d))
    # Seasonal sea ice from the sea-level temperature (t_now is at sea level here).
    ice = np.clip(smooth(-1.5,-6,t_now+F(.4)*m1)*F(snow_amount),0,1)
    water = mix(water,[.80,.84,.88],ice*F(.9))*(1+F(.04)*var*m1)[...,None]
    return np.clip(np.where((np.asarray(height)<0)[...,None],water,color),0,1)


def terrain_derivatives(elevation, resolution):
    """Gradient (m/m) and local relief position, matching the GPU tile path."""
    spacing = (resolution,resolution) if np.isscalar(resolution) else (resolution[1],resolution[0])
    dy,dx = np.gradient(elevation.astype(np.float64),*spacing)
    small = gaussian_filter(elevation.astype(np.float64),1.2)
    large = gaussian_filter(elevation.astype(np.float64),6.)
    tpi = ((small-large)/(6*max(spacing))).astype(F)
    return np.stack((dx,dy),axis=-1).astype(F),tpi


def colorize_surface(world,xs,ys,elevation,resolution,generation_settings,settings=None,*,polar=False,mode='render'):
    from terrain_pedology import sample_pedology, colorize_pedology
    if mode=='pedology':
        return colorize_pedology(world,xs,ys,generation_settings,polar=polar)
    soil = sample_soil(world,xs,ys,generation_settings,polar=polar)
    pedology = sample_pedology(world,xs,ys,generation_settings,polar=polar)
    seasons = seasonal_fields(world,xs,ys,generation_settings,polar=polar)
    spherical = generation_settings.get('world_topology','sphere')!='plane'
    point = coordinates(xs,ys,world.bounds,spherical,polar)
    gradient,tpi = terrain_derivatives(elevation,resolution)
    step = resolution if np.isscalar(resolution) else max(resolution)
    spacing = (resolution,resolution) if np.isscalar(resolution) else (resolution[1],resolution[0])
    ny,nx = np.gradient(point[...,1],*spacing)
    north = np.stack((nx,ny),axis=-1)
    north /= np.maximum(np.linalg.norm(north,axis=-1,keepdims=True),1e-5)
    if not spherical:north.fill(0)
    return surface_material(elevation,gradient,tpi,point,north,step,soil,seasons,
                            seed_value(getattr(getattr(world,'source',world),'seed',0)),settings,mode=='soil',pedology)

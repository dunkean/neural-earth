"""Inspect persisted Orogen plate/tectonic fields without running the NN."""
import colorsys
import numpy as np

MODES = ('orogen-plates', 'orogen-crust', 'orogen-boundaries',
         'orogen-convergence', 'orogen-uplift', 'orogen-height')
LEGENDS = {
    'orogen-plates': 'Plate IDs · distinct colors; dark lines show plate borders',
    'orogen-crust': 'Continental crust: ochre · oceanic crust: blue',
    'orogen-boundaries': 'Convergence: red · divergence: cyan · transform: yellow',
    'orogen-convergence': 'Relative normal velocity · compression: red · extension: blue',
    'orogen-uplift': 'Continental mountain uplift · black: 0 m · orange: 3,000 m · white: 6,000 m',
    'orogen-height': 'Source elevation after selected erosion · signed metres · source resolution 19.53 km',
}
EXTRA = {
    'orogen-pressure-summer': ('pressureSummer', 'Summer atmospheric pressure'),
    'orogen-pressure-winter': ('pressureWinter', 'Winter atmospheric pressure'),
    'orogen-rain-shadow': ('rainShadowSummer', 'Summer rain shadow'),
    'orogen-erosion': ('erosionDelta', 'Relief change from the selected erosion engine'),
    'orogen-continentality': ('continentality', 'Distance from ocean influence'),
    'orogen-back-arc': ('backArc', 'Back arc contribution'),
    'orogen-fold-ridges': ('foldRidge', 'Folded mountain ridge contribution'),
    'orogen-superplates': ('superPlates', 'Superplate groups driving broad mountain belts'),
    'orogen-hotspots': ('hotspot', 'Original hotspot contribution'),
    'orogen-orogeny': ('orogenicPower', 'Original orogenic power'),
    'orogen-biomes': ('biomes', 'Biomes · classification and altitude following terrain LOD'),
    'render': ('surface', 'Render · vegetation, substrate, rock and seasonal snow'),
    'soil': ('substrate', 'Soils · soil and exposed rock on terrain'),
    'pedology': ('pedology', 'Pedology · soil provinces on a 200 km grid: sandy, calcareous, clay-rich, ferrallitic, organic, podzolic, mineral'),
    'orogen-koppen': ('koppen', 'Köppen following terrain LOD and altitude'),
    'orogen-temperature-summer': ('temperature_summer', 'Northern summer temperature · -45 to +45 °C'),
    'orogen-temperature-winter': ('temperature_winter', 'Northern winter temperature · -45 to +45 °C'),
    'orogen-precip-summer': ('precip_summer', 'Northern summer half-year precipitation · logarithmic 0–6,000 mm scale'),
    'orogen-precip-winter': ('precip_winter', 'Northern winter half-year precipitation · logarithmic 0–6,000 mm scale'),
    'orogen-wind-summer': ('wind', 'Northern summer winds · direction hue, speed brightness'),
    'orogen-wind-winter': ('wind', 'Northern winter winds · direction hue, speed brightness'),
    'orogen-currents-summer': ('current', 'Northern summer currents · direction hue, speed brightness'),
    'orogen-currents-winter': ('current', 'Northern winter currents · direction hue, speed brightness'),
}
MODES += tuple(EXTRA)
LEGENDS.update({k:v[1] for k,v in EXTRA.items()})
LEGENDS['orogen-convergence']='Original propagated Orogent stress · positive compression'
LEGENDS['orogen-uplift']='Original Orogent elevation contribution · internal units'
_PLATE_COLORS = np.array([colorsys.hsv_to_rgb((i*.61803398875)%1, .55, .9)
                          for i in range(120)], np.float32)


def sample(world, field, xs, ys, *, categorical=False):
    xs, ys = np.asarray(xs, np.float64), np.asarray(ys, np.float64)
    x0,y0,x1,y1 = world.bounds
    periodic = getattr(world, 'periodic_longitude', True)
    xx = ((xs-x0)%(x1-x0))/(x1-x0)*world.width-.5 if periodic else np.clip(
        (xs-x0)/(x1-x0)*world.width-.5, 0, world.width-1)
    yy = np.clip((ys-y0)/(y1-y0)*world.height-.5, 0, world.height-1)
    if categorical:
        return field[np.rint(yy).astype(np.int64)[:,None],
                     (np.floor(xx+.5).astype(np.int64)%world.width)[None,:]]
    ix, iy = np.floor(xx).astype(np.int64), np.floor(yy).astype(np.int64)
    j0 = ix % world.width if periodic else np.clip(ix, 0, world.width-1)
    j1 = (ix+1) % world.width if periodic else np.clip(ix+1, 0, world.width-1)
    tx, ty = (xx-ix)[None,:], (yy-iy)[:,None]
    a = field[iy[:,None], j0[None,:]]*(1-tx)+field[
        iy[:,None], j1[None,:]]*tx
    by = np.minimum(iy+1, world.height-1)[:,None]
    b = field[by, j0[None,:]]*(1-tx)+field[
        by, j1[None,:]]*tx
    return np.asarray(a*(1-ty)+b*ty, np.float32)


def render(world, mode, xs, ys):
    if mode not in MODES:
        raise ValueError(f'Unknown Orogen diagnostic layer {mode!r}')
    if mode=='pedology':
        from terrain_pedology import colorize_pedology
        settings={'world_topology':'sphere' if getattr(world,'periodic_longitude',True) else 'plane'}
        return colorize_pedology(world,xs,ys,settings)
    if mode in EXTRA:
        field=EXTRA[mode][0]
        if field=='koppen':
            return np.stack([sample(world,world.layers['koppen_color_'+str(i)],xs,ys,categorical=True) for i in range(3)],axis=-1)
        if field=='biomes':
            return np.stack([sample(world,world.layers['biome_'+str(i)],xs,ys) for i in range(3)],axis=-1)
        if field in ('wind','current'):
            season=mode.rsplit('-',1)[1];prefix='wind' if field=='wind' else 'ocean_current'
            east=sample(world,world.layers[prefix+'_east_'+season],xs,ys)
            north=sample(world,world.layers[prefix+'_north_'+season],xs,ys)
            angle=(np.arctan2(north,east)+np.pi)/(2*np.pi)
            # Vector direction in RGB color wheel; fixed speed transfer.
            phase=np.stack([angle,angle+1/3,angle+2/3],axis=-1)
            rgb=.5+.5*np.cos(phase*2*np.pi)
            return rgb*(.15+.85*np.clip(np.hypot(east,north),0,1))[...,None]
        city_delta=field=='erosionDelta' and 'city_erosion_delta_m' in world.layers
        value=sample(world,world.layers.get('city_erosion_delta_m' if city_delta else field,world.layers['plates']*0),xs,ys,categorical=field in ('superPlates','koppen'))
        if field in ('superPlates','koppen'): return _PLATE_COLORS[value.astype(np.int64)%len(_PLATE_COLORS)]
        if field.startswith('pressure') or field=='erosionDelta':
            t=np.clip(np.abs(value)*(.001 if city_delta else 3 if field=='erosionDelta' else 1),0,1)[...,None]
            return (1-t)*np.array([.12,.12,.15])+t*np.where(value[...,None]>=0,np.array([1.,.25,.1]),np.array([.1,.4,1.]))
        t=np.clip(value,0,1)[...,None]
        if field.startswith('temperature'): return (1-t)*np.array([.08,.25,.85])+t*np.array([1.,.25,.05])
        if field.startswith('precip'):
            t=np.clip(np.log1p(np.maximum(value,0)*1000)/np.log1p(6000),0,1)[...,None]
            return (1-t)*np.array([.75,.55,.28])+t*np.array([.05,.35,.7])
        return np.clip(t*3,0,1)*np.array([1.,.45,.08])
    if mode == 'orogen-height':
        from terrain_diffusion.inference.relief_map import get_relief_map
        height = world.sample_height_m(xs, ys)
        spacing = abs(xs[1]-xs[0]) if len(xs)>1 else (world.bounds[2]-world.bounds[0])/world.width
        return get_relief_map(height, None, None, None, resolution=spacing, vmin=0, vmax=6500)
    if mode == 'orogen-plates':
        ids = sample(world, world.layers['plates'], xs, ys, categorical=True)
        rgb = _PLATE_COLORS[ids.astype(np.int64)%len(_PLATE_COLORS)].copy()
        border = sample(world, world.layers['boundaries'], xs, ys, categorical=True)>0
        rgb[border] *= .35
        return rgb
    if mode == 'orogen-crust':
        ocean = sample(world, world.layers['crust'], xs, ys, categorical=True)>0
        return np.where(ocean[...,None], np.array([.07,.24,.44]), np.array([.77,.58,.28]))
    if mode == 'orogen-boundaries':
        kind = sample(world, world.layers['boundaries'], xs, ys, categorical=True)
        palette = np.array([[.06,.10,.15], [.95,.22,.13], [.12,.75,.95], [.96,.82,.20]])
        return palette[kind.astype(np.int64)]
    if mode == 'orogen-convergence':
        value = sample(world, world.layers['convergence'], xs, ys)
        t = np.clip(np.abs(value)/2, 0, 1)[...,None]
        positive = np.array([.98,.18,.08]); negative = np.array([.1,.5,1.])
        return (1-t)*np.array([.06,.1,.15])+t*np.where(value[...,None]>=0, positive, negative)
    value = sample(world, world.layers['uplift'], xs, ys)
    t = np.clip(value/.5, 0, 1)[...,None]
    lo = np.array([.035,.05,.07]); mid = np.array([1.,.40,.04]); hi = np.array([1.,1.,.94])
    return np.where(t<.5, lo+(mid-lo)*t*2, mid+(hi-mid)*(t-.5)*2)

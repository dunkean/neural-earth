"""Regional soil provinces and substrate from one fast, coarse evaluation.

This visual approximation is a composition map, not a soil survey. Climate
(moisture index, warmth, frost), regional relief (ruggedness, altitude,
lowland basins) and parent-material noise give seven soil mixtures. Their RGB
is shared by Pedology and Render. Soils' sand/clay/humus texture follows the
same mixtures; the exposed-rock tint varies with parent noise and climate.
"""
import numpy as np
from scipy.ndimage import gaussian_filter
from terrain_soil import coordinates, noise, seed_value
from terrain_orogen_layers import sample

VERSION = 'pedology-v2-coarse'
TYPES = ('sandy', 'calcareous', 'clayey', 'ferrallitic', 'organic', 'podzolic', 'mineral')
LABELS = ('Sandy', 'Calcareous', 'Clay-rich', 'Ferrallitic', 'Organic', 'Podzolic', 'Mineral')
PALETTE = np.array([[.69,.57,.39],[.58,.53,.43],[.48,.35,.25],[.56,.30,.18],
                    [.29,.25,.18],[.43,.43,.37],[.50,.47,.42]],np.float32)
NAMES = ('pedology_red','pedology_green','pedology_blue')+tuple('pedology_'+n for n in TYPES)
OCEAN = np.array([.10,.22,.30],np.float32)
# Topsoil sand, clay and humus of each mixture.
TEXTURE = np.array([[.82,.08,.10],[.38,.37,.25],[.18,.66,.16],[.22,.68,.10],
                    [.14,.18,.68],[.62,.13,.25],[.55,.33,.12]],np.float32)
SOIL_RGB = np.array([[.64,.53,.36],[.43,.30,.21],[.22,.20,.14]],np.float32)
SOIL_NAMES = ('soil_red','soil_green','soil_blue','rock_red','rock_green',
              'rock_blue','soil_sand','soil_clay','soil_humus')


def _factor(h, w):
    # Half resolution: provinces are regional and source cells are ~20 km.
    # The 2x2 mean also cancels single-pixel source relief.
    return 2 if h >= 64 and h % 2 == 0 and w % 2 == 0 else 1


def cell_metres(shape, bounds):
    h, w = shape
    return (bounds[2]-bounds[0])/(w//_factor(h, w))


def _smooth(a, b, x):
    t = np.clip((x-a)/(b-a), 0, 1)
    return t*t*(3-2*t)


def generate_substrate(height, layers, seed, bounds, spherical=True):
    """Soil and pedology layers, evaluated coarse and resampled bilinearly."""
    from terrain_array_io import POOL, parallel_rows
    h, w = height.shape
    k = _factor(h, w)
    ch, cw = h//k, w//k
    down = lambda a: np.asarray(a, np.float32).reshape(ch, k, cw, k).mean(axis=(1, 3), dtype=np.float32)
    field = lambda name, value: down(layers[name] if name in layers else np.full((h, w), value, np.float32))
    x0, y0, x1, y1 = bounds
    salt = seed_value(seed)
    # Parent material varies over hundreds of kilometres: evaluate its noise
    # on a lattice halved until it has about 128 rows, then resample.
    halvings = 0
    while ch >> halvings >= 256 and (ch >> halvings) % 2 == 0 and (cw >> halvings) % 2 == 0:
        halvings += 1
    nx = x0+(np.arange(cw >> halvings)+.5)*(x1-x0)/(cw >> halvings)
    ny = y0+(np.arange(ch >> halvings)+.5)*(y1-y0)/(ch >> halvings)
    def channels(a, b):
        p = coordinates(nx, ny[a:b], bounds, spherical)
        return dict(calcareous=.6*noise(p/1.1e6, salt+101)+.4*noise(p/3.7e5, salt+102),
                    texture=.6*noise(p/6.5e5, salt+103)+.4*noise(p/2.2e5, salt+104),
                    region=noise(p/350000, salt), local=noise(p/120000+17, salt+31))
    parent = parallel_rows(channels, len(ny), chunks=8)
    for _ in range(halvings):
        parent = dict(zip(parent, POOL.map(lambda plane: _upsample2(plane, spherical), parent.values())))
    ts = field('temperature_summer', .65)*90-45
    tw = field('temperature_winter', .45)*90-45
    rain = (field('precip_summer', .5)+field('precip_winter', .5))*1000
    uplift = np.clip(field('uplift', 0.)*2, 0, 1)
    # Relief from the regional surface only: smoothed mean and deviation.
    mode = ('nearest', 'wrap' if spherical else 'nearest')
    elevation = down(height)
    mean = gaussian_filter(elevation, 1.5, mode=mode)
    deviation = np.sqrt(np.maximum(gaussian_filter(elevation*elevation, 1.5, mode=mode)-mean*mean, 0))
    rugged = _smooth(150, 900, deviation)
    alpine = _smooth(1800, 4200, mean)
    lowland = (1-_smooth(150, 700, mean))*(1-rugged)
    tm = (ts+tw)*.5
    moisture = rain/(150+55*np.maximum(tm+5, 0))
    wet = _smooth(.45, 1.3, moisture)
    arid = 1-_smooth(.12, .55, moisture)
    steppe = _smooth(.15, .45, moisture)*(1-_smooth(.8, 1.3, moisture))
    temperate = _smooth(0, 12, tm)
    warm = _smooth(5, 22, tm)
    tropical = _smooth(14, 20, np.minimum(ts, tw))
    boreal = _smooth(-10, -2, tm)*(1-_smooth(6, 14, tm))
    polar = 1-_smooth(-14, -6, tm)
    calc = _smooth(.3, .7, parent['calcareous'])
    clay_parent = _smooth(.3, .7, parent['texture'])
    sand_parent = 1-clay_parent
    weights = np.stack((
        .02+arid*(.35+.65*sand_parent)*(1-.7*rugged)+.15*sand_parent*(1-wet),
        .02+steppe*(.35+.65*calc)*(1-.8*polar)+.25*calc*(1-wet)*rugged,
        .03+temperate*(1-arid)*(1-tropical)*(.3+.7*clay_parent)*(.4+.6*lowland),
        .01+tropical*wet*(1-.6*rugged)*(1-alpine),
        .01+wet*lowland*(boreal+.35*polar+.3*tropical),
        .01+wet*boreal*(.4+.6*sand_parent)*(1-alpine),
        .03+.85*np.maximum(rugged, alpine)+.5*polar+.3*arid*clay_parent*(1-lowland)))
    # Sharpen toward provinces while keeping soft transitions.
    weights **= 4
    weights /= weights.sum(axis=0)
    texture = np.einsum('kyx,kc->cyx', weights, TEXTURE)
    shift = .3*(parent['region']-.5)
    texture[0] += shift
    texture[1] -= shift
    texture = np.maximum(texture, .01)
    texture /= texture.sum(axis=0)
    iron = np.clip(.15+.55*parent['region']+.18*warm-.25*uplift, 0, 1)
    tint = .9+.2*parent['local']
    # Exposed rock follows parent lithology: pale limestone, dark basic or
    # metamorphic rock in uplands, iron-stained red where warm.
    lime = .7*calc*(1-iron)
    dark = .6*clay_parent*(1-calc)*(.4+.6*rugged)
    rock = [((grey*(1-iron)+red*iron)*(1-lime-dark)+pale*lime+basic*dark)*tint for grey, red, pale, basic in
            zip((.43,.44,.42), (.56,.40,.29), (.66,.64,.58), (.30,.30,.30))]
    # Mixing is linear and bilinear weights sum to one: compose coarse, then
    # resample; sums and palette products hold to float rounding.
    planes = [*np.einsum('kyx,kc->cyx', weights, PALETTE), *weights,
              *np.einsum('kyx,kc->cyx', texture, SOIL_RGB), *np.clip(rock, 0, 1), *texture]
    planes = [np.asarray(plane, np.float32) for plane in planes]
    if k > 1:
        planes = list(POOL.map(lambda plane: _upsample2(plane, spherical), planes))
    return dict(zip(NAMES+SOIL_NAMES, planes))


def _upsample2(plane, periodic):
    """Bilinear 2x resampling between pixel-centre grids (weights 3/4, 1/4)."""
    def axis(a, wrap):
        # Doubles axis 0; edges clamp unless the axis wraps.
        quarter = a*np.float32(.25)
        out = np.empty((2*a.shape[0], *a.shape[1:]), np.float32)
        out[0::2] = out[1::2] = a*np.float32(.75)
        out[2::2] += quarter[:-1]
        out[1:-1:2] += quarter[1:]
        out[0] += quarter[-1] if wrap else quarter[0]
        out[-1] += quarter[0] if wrap else quarter[-1]
        return out
    return np.ascontiguousarray(axis(axis(plane, False).T, periodic).T)


def generate_pedology(height, layers, seed, bounds, spherical=True):
    result = generate_substrate(height, layers, seed, bounds, spherical)
    return {name: result[name] for name in NAMES}




def pedology_atlas(world, settings):
    source=getattr(world,'source',world)
    if all(name in source.layers for name in NAMES):
        return source.layers
    key=(tuple(world.bounds),settings.get('world_topology','sphere'),VERSION)
    cache=getattr(source,'_terrain_pedology',{})
    if key not in cache:
        height=getattr(source,'height_m',np.zeros((world.height,world.width),np.float32))
        cache[key]=generate_pedology(height,source.layers,getattr(source,'seed',0),world.bounds,
                                    settings.get('world_topology','sphere')!='plane')
        source._terrain_pedology=cache
    return cache[key]


def sample_pedology(world,xs,ys,settings,*,polar=False):
    """Four display planes: RGB composition and the original world land mask."""
    source=getattr(world,'source',world)
    fields=pedology_atlas(world,settings)
    height=getattr(source,'height_m',np.zeros((world.height,world.width),np.float32))
    rasters=np.stack([fields[n] for n in NAMES[:3]]+[(height>=0).astype(np.float32)])
    if polar:
        from terrain_polar import from_chart,sample_raster
        x,y=from_chart(*np.meshgrid(xs,ys),settings)
        return sample_raster(rasters,x,y,world.bounds)
    return np.stack([sample(world,r,xs,ys) for r in rasters]).astype(np.float32)


def colorize_pedology(world,xs,ys,settings,*,polar=False):
    values=sample_pedology(world,xs,ys,settings,polar=polar)
    return np.where((values[3]>=.5)[...,None],np.moveaxis(values[:3],0,-1),OCEAN)

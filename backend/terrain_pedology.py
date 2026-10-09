"""Regional soil provinces on the source world, independent of rendered relief.

This visual approximation is a coarse composition map, not a soil survey.
The seven mixtures and their RGB are shared by Pedology and Render. Soil's
existing exposed-substrate material remains a separate display.
"""
from types import SimpleNamespace
import numpy as np
from scipy.ndimage import gaussian_filter
from terrain_soil import coordinates, noise, seed_value
from terrain_orogen_layers import sample

VERSION = 'pedology-v1-200km'
CELL_METRES = 200000.
TYPES = ('sandy', 'calcareous', 'clayey', 'ferrallitic', 'organic', 'podzolic', 'mineral')
LABELS = ('Sandy', 'Calcareous', 'Clay-rich', 'Ferrallitic', 'Organic', 'Podzolic', 'Mineral')
PALETTE = np.array([[.69,.57,.39],[.58,.53,.43],[.48,.35,.25],[.56,.30,.18],
                    [.29,.25,.18],[.43,.43,.37],[.50,.47,.42]],np.float32)
NAMES = ('pedology_red','pedology_green','pedology_blue')+tuple('pedology_'+n for n in TYPES)
OCEAN = np.array([.10,.22,.30],np.float32)


def generate_pedology(height, layers, seed, bounds, spherical=True):
    """Evaluate climate/parent material at ~200 km; interpolate compositions.

    The small bounded grid makes the map insensitive to fine DEM detail.
    Source coastline is applied separately, never blurred into soil provinces.
    """
    h,w = height.shape
    x0,y0,x1,y1 = bounds
    cw = min(w,256,max(2,int(np.ceil((x1-x0)/CELL_METRES))))
    ch = min(h,128,max(2,int(np.ceil((y1-y0)/CELL_METRES))))
    source = SimpleNamespace(width=w,height=h,bounds=bounds,periodic_longitude=spherical)
    xs=x0+(np.arange(cw)+.5)*(x1-x0)/cw
    ys=y0+(np.arange(ch)+.5)*(y1-y0)/ch
    coarse=lambda field:sample(source,gaussian_filter(np.asarray(field,np.float32),
        sigma=(.45*h/ch,.45*w/cw),mode=('nearest','wrap' if spherical else 'nearest')),xs,ys)
    ts=coarse(layers.get('temperature_summer',np.full_like(height,.65)))*90-45
    tw=coarse(layers.get('temperature_winter',np.full_like(height,.45)))*90-45
    rain=coarse(layers.get('precip_summer',np.full_like(height,.5))+layers.get('precip_winter',np.full_like(height,.5)))*1000
    elev=np.maximum(coarse(height),0)
    p=coordinates(xs,ys,bounds,spherical)
    parent=noise(p/1200000,seed_value(seed))
    smooth=lambda a,b,x:(lambda t:t*t*(3-2*t))(np.clip((x-a)/(b-a),0,1))
    warm=smooth(5,24,(ts+tw)*.5)
    tropical=smooth(12,22,np.minimum(ts,tw))
    wet=smooth(250,1800,rain)
    cold=1-smooth(-8,8,(ts+tw)*.5)
    alpine=smooth(1800,4500,elev)
    weights=np.stack((.08+.8*(1-wet)*(.4+.6*warm)*(.65+.35*parent),
        .10+.35*(1-wet)*(1-parent),
        .14+.35*wet*warm*(1-.7*tropical),
        .015+.9*tropical*wet,
        .03+.65*wet*(1-tropical)*(1-cold)*(1-alpine),
        .04+.7*cold*wet*(1-alpine),
        .04+.9*alpine+.3*cold*(1-wet)),axis=-1).astype(np.float32)
    weights/=weights.sum(axis=-1,keepdims=True)
    coarse_world=SimpleNamespace(width=cw,height=ch,bounds=bounds,periodic_longitude=spherical)
    target_x=x0+(np.arange(w)+.5)*(x1-x0)/w
    target_y=y0+(np.arange(h)+.5)*(y1-y0)/h
    fractions=np.stack([sample(coarse_world,weights[...,i],target_x,target_y) for i in range(len(TYPES))],axis=-1).astype(np.float32)
    fractions/=fractions.sum(axis=-1,keepdims=True)
    rgb=fractions@PALETTE
    values=np.concatenate((rgb,fractions),axis=-1)
    return {name:values[...,i] for i,name in enumerate(NAMES)}


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

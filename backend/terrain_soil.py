"""Generated, deliberately approximate substrate: sand / clay / humus and colors.

This is an appearance atlas, not a geological survey. It is composed with the
world and persisted alongside its other rasters; old worlds can derive it once
without rebuilding their retained relief or climate stages.
"""
import hashlib
import numpy as np

NAMES = ('soil_red', 'soil_green', 'soil_blue', 'rock_red', 'rock_green',
         'rock_blue', 'soil_sand', 'soil_clay', 'soil_humus')
VERSION = 'substrate-v2-pedology'
TRANSPORT_LAYERS = 50


def seed_value(seed):
    return int.from_bytes(hashlib.sha256(str(seed).encode()).digest()[:3], 'little')


def noise(point, salt=0):
    """Integer-hashed 3D value noise, shared with the material WGSL."""
    p = np.asarray(point, np.float32)
    cell = np.floor(p).astype(np.int32)
    f = p-cell
    f = f*f*(3-2*f)
    result = np.zeros(p.shape[:-1], np.float32)
    with np.errstate(over='ignore'):
        for z in (0, 1):
            for y in (0, 1):
                for x in (0, 1):
                    c = (cell+np.array([x,y,z],np.int32)).astype(np.uint32)
                    h = c[...,0]*np.uint32(0x9e3779b9) ^ c[...,1]*np.uint32(0x85ebca6b) ^ c[...,2]*np.uint32(0xc2b2ae35) ^ np.uint32(salt)
                    h = (h^(h>>16))*np.uint32(0x7feb352d)
                    h = (h^(h>>15))*np.uint32(0x846ca68b)
                    h ^= h>>16
                    weight = (f[...,0] if x else 1-f[...,0])*(f[...,1] if y else 1-f[...,1])*(f[...,2] if z else 1-f[...,2])
                    result += (h>>8).astype(np.float32)/16777216*weight
    return result


def coordinates(xs, ys, bounds, spherical=True, polar=False):
    x,y = np.meshgrid(xs,ys)
    if not spherical:
        return np.stack((x,y,np.zeros_like(x)),axis=-1).astype(np.float32)
    x0,y0,x1,y1 = bounds
    lon = (x-x0)/(x1-x0)*2*np.pi-np.pi
    lat = np.pi/2-(y-y0)/(y1-y0)*np.pi
    xyz = np.stack((np.cos(lat)*np.cos(lon),np.sin(lat),np.cos(lat)*np.sin(lon)),axis=-1)
    if polar:
        xyz = np.stack((xyz[...,0],-xyz[...,2],xyz[...,1]),axis=-1)
    return (xyz*((x1-x0)/(2*np.pi))).astype(np.float32)


def generate_soil(height, layers, seed, bounds, spherical=True):
    # One coarse model shared with Pedology; see terrain_pedology.
    from terrain_pedology import generate_substrate
    result = generate_substrate(height, layers, seed, bounds, spherical)
    return {name: result[name] for name in NAMES}


def soil_atlas(world, settings):
    source = getattr(world,'source',world)
    if all(name in source.layers for name in NAMES):
        return source.layers
    # Compatibility for immutable saved atlases made before this display field.
    key = (tuple(world.bounds),settings.get('world_topology','sphere'))
    cache = getattr(source,'_terrain_soil',{})
    if key not in cache:
        height = getattr(source,'height_m',np.zeros((world.height,world.width),np.float32))
        cache[key] = generate_soil(height,source.layers,getattr(source,'seed',0),world.bounds,
                                   settings.get('world_topology','sphere')!='plane')
        source._terrain_soil = cache
    return cache[key]


def sample_soil(world, xs, ys, settings, *, polar=False):
    from terrain_orogen_layers import sample
    fields = soil_atlas(world,settings)
    if polar:
        from terrain_polar import from_chart, sample_raster
        x,y = from_chart(*np.meshgrid(xs,ys),settings)
        return sample_raster(np.stack([fields[n] for n in NAMES]),x,y,world.bounds)
    return np.stack([sample(world,fields[n],xs,ys) for n in NAMES])


def sample_soil_transport(world, xs, ys, settings, *, polar=False):
    fields = sample_soil(world,xs,ys,settings,polar=polar)
    meta = np.zeros((1,len(ys),len(xs)),np.float32)
    # Exact, non-interpolated metadata. All noise uses canonical geographic
    # coordinates, including across the map seam and the rotated polar chart.
    meta[0,0,:7] = [*world.bounds,settings.get('world_topology','sphere')!='plane',
                     polar,seed_value(getattr(getattr(world,'source',world),'seed',0))]
    from terrain_pedology import sample_pedology
    return np.concatenate((fields,meta,sample_pedology(world,xs,ys,settings,polar=polar)))

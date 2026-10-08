"""A rotated spherical chart for neural detail, with regular polar distances.

The source continents stay in their original atlas. Only the NN coordinate
frame rotates by 90 degrees: both geographic poles become chart-equator
points. Physical cell sizes remain 7680 / 240 / 30 metres. All chart reads are
global and bounded, so adjacent output tiles use identical neural samples.
"""
from functools import lru_cache

import numpy as np

from terrain_geometry import world_bounds

VERSION = 'rotated-polar-chart-v1'


def _directions(x, y, bounds):
    x0, y0, x1, y1 = bounds
    lon = (np.asarray(x, np.float64)-x0)/(x1-x0)*2*np.pi-np.pi
    lat = np.pi/2-(np.asarray(y, np.float64)-y0)/(y1-y0)*np.pi
    cp = np.cos(lat)
    return cp*np.cos(lon), np.sin(lat), cp*np.sin(lon)


def _coordinates(nx, ny, nz, bounds):
    x0, y0, x1, y1 = bounds
    return (x0+(np.arctan2(nz, nx)+np.pi)/(2*np.pi)*(x1-x0),
            y0+(np.pi/2-np.arctan2(ny, np.hypot(nx, nz)))/np.pi*(y1-y0))


def to_chart(x, y, settings):
    bounds = world_bounds(settings)
    nx, ny, nz = _directions(x, y, bounds)
    return _coordinates(nx, nz, -ny, bounds)


def from_chart(x, y, settings):
    bounds = world_bounds(settings)
    nx, ny, nz = _directions(x, y, bounds)
    return _coordinates(nx, -nz, ny, bounds)


def sample_raster(fields, x, y, bounds):
    """Bilinear point sampling, with a single value at each spherical pole."""
    h, w = fields.shape[-2:]
    x0, y0, x1, y1 = bounds
    xx = ((x-x0)/(x1-x0)*w-.5) % w
    yy = np.clip((y-y0)/(y1-y0)*h-.5, 0, h-1)
    ix, iy = np.floor(xx).astype(np.int64), np.floor(yy).astype(np.int64)
    fx, fy = xx-ix, yy-iy
    a = fields[..., iy, ix]*(1-fx)+fields[..., iy, (ix+1)%w]*fx
    b = fields[..., np.minimum(iy+1,h-1), ix]*(1-fx)+fields[..., np.minimum(iy+1,h-1), (ix+1)%w]*fx
    value = a*(1-fy)+b*fy
    # Pixel-centred atlases stop half a cell short of the pole. Continue that
    # half cell to its longitude mean instead of extruding meridian spokes.
    row = (y-y0)/(y1-y0)*h
    for edge, gain in ((0, np.clip(1-2*row,0,1)), (-1, np.clip(1-2*(h-row),0,1))):
        if np.any(gain):
            pole = fields[..., edge, :].mean(axis=-1)
            pole = pole[(...,)+(None,)*np.ndim(x)]
            value = value*(1-gain)+pole*gain
    return np.asarray(value, np.float32)


class PolarConditioning:
    """Rotate the physical source fields; do not repeat climate transforms."""
    def __init__(self, source, settings, *, width=2048, height=1024):
        self.source = source
        self.settings = self.generation_settings = settings
        self.bounds = world_bounds(settings)
        x0,y0,x1,y1 = self.bounds
        xs = x0+(np.arange(width)+.5)*(x1-x0)/width
        ys = y0+(np.arange(height)+.5)*(y1-y0)/height
        self.fields = np.ascontiguousarray(source.sample(xs,ys), dtype=np.float32)
        if self.fields.shape != (5,height,width) or not np.isfinite(self.fields).all():
            raise ValueError('Polar chart needs finite five-channel physical conditioning')
        self.fields.flags.writeable = False

    def __getattr__(self, name):
        # Admission verifies the original generator profile and immutable
        # bootstrap height receipt even when its sampling frame is rotated.
        return getattr(self.source, name)

    def sample_raw(self, cj0, ci0, cj1, ci1):
        return self.sample((np.arange(cj0,cj1,dtype=np.float64)+.5)*7680,
                           (np.arange(ci0,ci1,dtype=np.float64)+.5)*7680)

    def sample(self, xs, ys):
        x,y = np.meshgrid(xs,ys)
        gx,gy = from_chart(x,y,self.settings)
        return sample_raster(self.fields,gx,gy,self.bounds)

    def finalize(self, fields):
        return np.asarray(fields,np.float32).copy()

    def __call__(self, cj0, ci0, cj1, ci1):
        import torch
        fields = self.sample_raw(cj0,ci0,cj1,ci1)
        fields[0] = np.sign(fields[0])*np.sqrt(np.abs(fields[0]))
        return torch.from_numpy(fields)

    def latitude(self, x, y):
        from terrain_geometry import world_latitude
        _,gy = from_chart(x,y,self.settings)
        return float(world_latitude(gy,self.settings))


@lru_cache(maxsize=2)
def conditioning(seed, profile):
    from terrain_conditioning import make_conditioning_factory
    from terrain_generation import resolve_generation
    return PolarConditioning(make_conditioning_factory(seed,profile),resolve_generation(profile).settings)


def chart_manifest(manifest):
    import hashlib
    import json
    from copy import deepcopy
    result = deepcopy(manifest)
    result['geography']['neural_chart'] = VERSION
    result['geography']['coordinate_system'] = 'rotated-sphere-metre, y-down'
    payload = {k:v for k,v in result.items() if k!='world_hash'}
    result['world_hash'] = hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()).hexdigest()
    return result

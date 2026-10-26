"""Bounded C2 interpolation below empirically chosen relief floors.

RELIEF_FLOOR is the only default policy table. TERRAIN_RELIEF_FLOORS accepts
JSON, e.g. {"classic":-3,"hybrid":-4}, for repeatable generator comparisons.
The quintic Hermite patches share nodal derivatives through order (2,2).
A shared nodal limiter bounds every Bezier control point by its cell corners:
unlike an unrestricted cardinal cubic, cliff ringing is bounded to 2 cm.
Coordinates are global cell centres, with integer division before conversion.
"""
from collections import OrderedDict
import json
import os
import threading

import numpy as np

MIN_LOD = -12
VERSION = 'v1'
# Measured in tmp/e2e/interp/diagnosis; -6 is also the generators' current limit.
RELIEF_FLOOR = {'classic': -6, 'hybrid': -6, 'phasor': -6, 'mse': -6}
BLOCK = 128
_lock = threading.RLock()
_limiter_kernel = None
_limiter_error = None


def _cuda_limiter(values, fields, device):
    """The local control-net bounds, one independent FP64 thread per node."""
    import torch  # Load the runtime's NVRTC before CuPy on mixed CUDA installs.
    import cupy as cp
    global _limiter_kernel
    with cp.cuda.Device(device):
        if _limiter_kernel is None:
            _limiter_kernel=cp.RawKernel(r'''
            extern "C" __global__ void limiter(const double* raw,const double* d,double* out,int channels,int h,int w){
                int k=blockDim.x*blockIdx.x+threadIdx.x,n=channels*h*w;
                if(k>=n)return;int c=k/(h*w),y=(k/w)%h,x=k%w;
                int stride=w+2,at=(c*(h+2)+y+1)*stride+x+1;
                double f=raw[at],alpha=1.,tol=c==0?.02:1e-9;
                for(int sy=-1;sy<=1;sy+=2)for(int sx=-1;sx<=1;sx+=2){
                    double lo=f,hi=f;
                    for(int iy=0;iy<=1;iy++)for(int ix=0;ix<=1;ix++){
                        double a=raw[at+iy*sy*stride+ix*sx];lo=a<lo?a:lo;hi=a>hi?a:hi;
                    }
                    lo=lo-f-tol;hi=hi-f+tol;
                    for(int iy=0;iy<3;iy++)for(int ix=0;ix<3;ix++){
                        double ay=.2*iy*sy,by=iy==2?.05:0.,ax=.2*ix*sx,bx=ix==2?.05:0.;
                        double v=ay*d[k]+by*d[n+k];
                        v=v+ax*d[2*n+k];v=v+bx*d[3*n+k];
                        v=v+(ay*ax)*d[4*n+k];v=v+(ay*bx)*d[5*n+k];
                        v=v+(by*ax)*d[6*n+k];v=v+(by*bx)*d[7*n+k];
                        double b=v>0.?hi/v:v<0.?lo/v:1.;alpha=b<alpha?b:alpha;
                    }
                }out[k]=alpha<0.?0.:alpha>1.?1.:alpha;
            }''','limiter',options=('--fmad=false',))
        derivatives=np.stack([fields[k] for k in ((1,0),(2,0),(0,1),(0,2),(1,1),(1,2),(2,1),(2,2))])
        shape=fields[0,0].shape;out=cp.empty(shape,cp.float64)
        _limiter_kernel(((out.size+127)//128,),(128,),(cp.asarray(values),cp.asarray(derivatives),out,*shape))
        return cp.asnumpy(out)


def relief_floor(mode='classic'):
    policy = dict(RELIEF_FLOOR)
    raw = os.environ.get('TERRAIN_RELIEF_FLOORS', '')
    if raw:
        overrides = json.loads(raw)
        if not isinstance(overrides, dict):
            raise ValueError('TERRAIN_RELIEF_FLOORS must be a JSON object')
        for name, value in overrides.items():
            if name not in policy or isinstance(value, bool) or not isinstance(value, int) or not -6 <= value <= -1:
                raise ValueError('Relief floors must name a mode and be integers in -6..-1')
            policy[name] = value
    return policy[mode]


def interpolation_suffix(lod, mode='classic'):
    floor = relief_floor(mode)
    return f'/interp-from{floor}-{VERSION}' if lod < floor else ''


def _numpy(value):
    return np.asarray(value.cpu().numpy() if hasattr(value, 'cpu') else value, dtype=np.float64)


def _cache(world):
    with _lock:
        if not hasattr(world, '_terrain_lod_interpolation'):
            world._terrain_lod_interpolation = OrderedDict()
        return world._terrain_lod_interpolation


def remember_floor(world, mode, identity, y0, x0, heights, surface=None):
    """Reuse complete canonical blocks already covered by a generated floor tile."""
    floor=relief_floor(mode)
    values=_numpy(heights)[None] if surface is None else np.concatenate((_numpy(heights)[None],_numpy(surface)))
    y1,x1=y0+values.shape[1],x0+values.shape[2]
    cache=_cache(world)
    with _lock:
        for by in range(-(-y0//BLOCK),y1//BLOCK):
            for bx in range(-(-x0//BLOCK),x1//BLOCK):
                key=((mode,floor,identity),by,bx)
                if key not in cache:cache[key]=values[:,by*BLOCK-y0:(by+1)*BLOCK-y0,bx*BLOCK-x0:(bx+1)*BLOCK-x0].copy()
        while len(cache)>128:cache.popitem(last=False)


def _read(world, reader, key, y0, x0, y1, x1, check):
    cache = _cache(world)
    out = None
    for by in range(y0//BLOCK, (y1-1)//BLOCK+1):
        for bx in range(x0//BLOCK, (x1-1)//BLOCK+1):
            if check:
                check()
            identity = (key, by, bx)
            with _lock:
                block = cache.get(identity)
                if block is not None:
                    cache.move_to_end(identity)
            if block is None:
                h, s = reader(by*BLOCK, bx*BLOCK, (by+1)*BLOCK, (bx+1)*BLOCK)
                block = _numpy(h)[None] if s is None else np.concatenate((_numpy(h)[None], _numpy(s)))
                with _lock:
                    cache[identity] = block
                    while len(cache) > 128:
                        cache.popitem(last=False)
            if out is None:
                out = np.empty((len(block), y1-y0, x1-x0), np.float64)
            ya, xa = max(y0, by*BLOCK), max(x0, bx*BLOCK)
            yb, xb = min(y1, (by+1)*BLOCK), min(x1, (bx+1)*BLOCK)
            out[:, ya-y0:yb-y0, xa-x0:xb-x0] = block[:, ya-by*BLOCK:yb-by*BLOCK, xa-bx*BLOCK:xb-bx*BLOCK]
    return out


def _fields(values, device=None):
    global _limiter_error
    # Fixed-order local stencils; no solve or boundary condition depends on a tile.
    fields = {}
    taps = ((0., 1., 0.), (-.5, 0., .5), (1., -2., 1.))
    for p in range(3):
        for q in range(3):
            field = np.zeros_like(values[:, 1:-1, 1:-1])
            for i, wy in enumerate(taps[p]):
                for j, wx in enumerate(taps[q]):
                    if wy*wx:
                        field += values[:, i:i+field.shape[1], j:j+field.shape[2]]*(wy*wx)
            fields[p, q] = field
    # One limiter per node, shared by all incident patches and all derivatives.
    f = fields[0, 0]
    alpha = np.ones_like(f)
    if device is not None and f.size>4096 and _limiter_error is None:
        try:
            alpha=_cuda_limiter(values,fields,device)
        except Exception as error:
            _limiter_error=f'{type(error).__name__}: {error}'
            import logging
            logging.getLogger(__name__).warning('C2 limiter uses CPU fallback: %s',_limiter_error)
        else:
            for order in fields:
                if order!=(0,0):fields[order]*=alpha
            return fields
    for sy in (-1, 1):
        for sx in (-1, 1):
            corners = [values[:, 1+dy:values.shape[1]-1+dy, 1+dx:values.shape[2]-1+dx]
                       for dy in (0, sy) for dx in (0, sx)]
            # Two centimetres of height tolerance avoids crushing valid slopes
            # for sub-ULP mixed derivatives on an otherwise smooth surface.
            tolerance=np.full((len(f),1,1),1e-9);tolerance[0]=.02
            low, high = np.minimum.reduce(corners)-f-tolerance, np.maximum.reduce(corners)-f+tolerance
            for ay, by in ((0., 0.), (.2*sy, 0.), (.4*sy, .05)):
                for ax, bx in ((0., 0.), (.2*sx, 0.), (.4*sx, .05)):
                    change = (ay*fields[1, 0]+by*fields[2, 0]+ax*fields[0, 1]+bx*fields[0, 2]+
                              ay*ax*fields[1, 1]+ay*bx*fields[1, 2]+by*ax*fields[2, 1]+by*bx*fields[2, 2])
                    bound = np.ones_like(change)
                    np.divide(high, change, out=bound, where=change > 0)
                    np.divide(low, change, out=bound, where=change < 0)
                    alpha = np.minimum(alpha, bound)
    for order in fields:
        if order != (0, 0):
            fields[order] *= np.clip(alpha, 0., 1.)
    return fields


def _basis(t):
    # Hermite basis: values, first and second derivatives at the two endpoints.
    t2, t3, t4, t5 = t*t, t**3, t**4, t**5
    return ((1-10*t3+15*t4-6*t5, t-6*t3+8*t4-3*t5, (t2-3*t3+3*t4-t5)/2),
            (10*t3-15*t4+6*t5, -4*t3+7*t4-3*t5, (t3-2*t4+t5)/2))


def evaluate(values, y0, x0, ys, xs, device=None):
    """Evaluate global parent-index positions, including exact parent knots."""
    iy, ix = np.floor(ys).astype(np.int64), np.floor(xs).astype(np.int64)
    wy, wx = _basis(ys-iy), _basis(xs-ix)
    fields = _fields(values,device)
    result = np.zeros((len(values), len(ys), len(xs)), np.float64)
    for q in range(3):
        rows=np.zeros((len(values),len(ys),next(iter(fields.values())).shape[2]),np.float64)
        for ey in range(2):
            for p in range(3):
                rows+=fields[p,q][:,iy+ey-y0-1]*wy[ey][p][None,:,None]
        for ex in range(2):
            result+=rows[:,:,ix+ex-x0-1]*wx[ex][q][None,None,:]
    return result


def sample_interpolated(world, lod, i1, j1, i2, j2, *, reader, mode='classic', identity='', check=None):
    floor = relief_floor(mode)
    if not MIN_LOD <= lod < floor:
        raise ValueError('Interpolation requires a LOD below its relief floor')
    scale = 2**(floor-lod)
    def axis(a, b):
        cells = np.arange(a, b, dtype=np.int64)
        return cells//scale + ((cells % scale)+.5)/scale-.5
    ys, xs = axis(i1, i2), axis(j1, j2)
    y0, x0 = int(np.floor(ys[0]))-2, int(np.floor(xs[0]))-2
    y1, x1 = int(np.floor(ys[-1]))+4, int(np.floor(xs[-1]))+4
    values = _read(world, reader, (mode, floor, identity), y0, x0, y1, x1, check)
    device=getattr(world,'device',None)
    ordinal=(getattr(device,'index',None) or 0) if str(device).startswith('cuda') else None
    out = evaluate(values, y0, x0, ys, xs,ordinal)
    return np.ascontiguousarray(out[0]), None if len(out) == 1 else np.ascontiguousarray(out[1:], dtype=np.float32)

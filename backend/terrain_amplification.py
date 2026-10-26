"""Windowed multi-scale erosion of the sub-30 m procedural reconstruction.

Adapted from Schott et al., Terrain Amplification using Multi-Scale Erosion
(TOG 2024; MIT reference implementation, see THIRD_PARTY_NOTICES.md). Weighted
stream-power incision, conservative talus transport, then capacity-limited
sediment settling run in metres, in that order. This is a short-budget visual
prototype, not a converged catchment simulation. The reconstruction's new
octaves supply the upsampling/noise seed; no domain warp moves its coastline.

Native drainage has 160 Jacobi steps (4.8 km reach) without moving heights.
Erosion inherits that contributing area as a persistent lower bound. Stage
collars contain 2*(erosion+thermal+deposition) stencil steps, radius-six activity,
twelve isotropic slope-envelope sweeps, two binomial passes, eight positive
downhill averages and three extrema guards, plus the largest final filter /
material stencil and three final safety guards. _reach
asserts this budget against every geometry. No-flux edges cannot reach a centre.
Only the completed stage's new increment is projected once, by one radial
parent-band convolution after a positive fine-grid Gaussian. Gain is one;
no filtered residual feeds back into erosion or a nonlinear amplifier. The
Gaussian rejects 95% of power at four-cell wavelengths and >99.99% near
Nyquist. No decimation, parent-cell statistics or block limiters stamp a
lattice. Notched C2 splines carry height and stream
confidence; material/area guides use positive splines. Shoreline safety wins.
Linear partition-of-unity blending retains exact locality across request shapes.
Cached numerators use dyadic fixed-point values in FP64: their sums are exact
in either tile store, even when windows arrive in a different order. Public
heights and surface guides remain contiguous FP32 on the world's device.

sample_surface independently diagnoses any final-height provider using native
catchment context and finite geometry stencils. It never runs erosion or writes
callback heights; callers own the final-mask cache and its SURFACE_VERSION key.
"""
from __future__ import annotations

from contextlib import nullcontext
from collections import OrderedDict
from functools import lru_cache
from threading import local
import math

import torch
import torch.nn.functional as F
from infinite_tensor import InfiniteTensor, TensorWindow
from terrain_diffusion.inference.world_pipeline import linear_weight_window

import terrain_refinement as refinement

VERSION = 'mse-v5'
SURFACE_VERSION = 'surface-v1'
STAGE_LEVELS = (2, 4, 6)
SURFACE_CHANNELS = ('flow', 'deposit', 'scree', 'rock')
MAX_INTENSITY = 2.0
WINDOW, STRIDE, MARGIN = 384, 320, 296
ITERATIONS = {2: (64, 20, 32), 4: (48, 16, 24), 6: (32, 16, 20)}
DRAIN_ITERATIONS = 160
GEOMETRY = {0: (192, 160, 168), 2: (WINDOW, STRIDE, MARGIN),
            4: (256, 192, 240), 6: (256, 192, 224)}
RELIEF_SHARE = {2: .38, 4: .28, 6: .22}
DETAIL_GAIN = {2: 1., 4: 1., 6: 1.}
DETAIL_CAP_METRES = {2: 9., 4: 3., 6: .75}
PROJECTION_SIGMA = 1.1  # Final positive Gaussian: wavelengths >= four cells.
PROJECTION_RADIUS = 16
PROJECTION_CUTOFF = .215  # Parent Nyquist .125; box restriction includes sidelobes.
DETAIL_SMOOTH_RADIUS = 5
AREA_REFERENCE_M2 = 10000.
EROSION_RATE = {2: .045, 4: .035, 6: .025}
TRANSFER_SIGMA = .7
TRANSFER_ORDER = 6
# One final linear projection, with no gain/cap/filter feedback. A radial
# Blackman-windowed jinc removes the parent band; positive Gaussian smoothing
# rejects the finest grid frequencies. Neither filter enters transport passes.
_DIRECTIONS = ((-1, -1), (-1, 0), (-1, 1), (0, -1),
               (0, 1), (1, -1), (1, 0), (1, 1))
_DISTANCES = tuple(math.hypot(y, x) for y, x in _DIRECTIONS)
_FLOW_CELLS = 1048576
_FLOW_LOG = math.log1p(_FLOW_CELLS)
_FLOW_AREA = (30 / 2**STAGE_LEVELS[0])**2
_state = local()


def _reach(stage):
    # Incoming flux reads a neighbour's routing/flux, hence two cells per
    # dynamic pass. Fixed native routing needs only its initial extra cell.
    # Native appearance adds two binomial passes and a radius-two ridge test.
    # Fine channel position uses a 6 m Gaussian after the 2 m material filter.
    # Six-cell process gate, twelve envelopes, two binomial passes, eight
    # downhill averages and three raw guards precede BOTH display branches.
    # Projection adds radius 21 plus three guards; appearance can reach 53
    # cells at stage 6. Conservative reaches: 165/287/231/220 cells.
    cell = 30/2**stage
    appearance = math.ceil(3*max(.8,2/cell))+math.ceil(3*max(1.5,6/cell))
    return (DRAIN_ITERATIONS+5 if stage == 0 else
            2*sum(ITERATIONS[stage])+6+12+2+8+3+
            max(DETAIL_SMOOTH_RADIUS+PROJECTION_RADIUS+3,appearance+1))


assert all(geometry[2] >= _reach(stage) for stage, geometry in GEOMETRY.items())


def _reference(world, level, i1, j1, i2, j2):
    """Respect the refinement output budget, even with a larger future collar."""
    return torch.cat([torch.cat([
        refinement.sample_refined(world, level, y, x, min(y+1024, i2), min(x+1024, j2),
                                  check=getattr(_state, 'check', None))
        for x in range(j1, j2, 1024)], dim=1) for y in range(i1, i2, 1024)], dim=0)


def _gaussian_taps(sigma, radius):
    taps = [math.exp(-.5*(x/sigma)**2) for x in range(-radius,radius+1)]
    total = math.fsum(taps)
    return [tap/total for tap in taps]


@lru_cache(maxsize=1)
def _projection_terms():
    """Separable eigenterms of one radial FIR, not repeated high-pass passes.

    The small CPU coefficient solve/decomposition is geometry only. Keeping every even term
    above roundoff reconstructs the radial kernel (rotation error <1e-7).
    Fixed-order FP64 filtering avoids shape-dependent cuDNN convolution sums.
    """
    import numpy as np
    from scipy.special import j1
    r = PROJECTION_RADIUS
    y,x = np.mgrid[-r:r+1,-r:r+1]
    distance = np.hypot(y,x)
    f = PROJECTION_CUTOFF
    kernel = np.divide(f*j1(2*np.pi*f*distance),distance,
                       out=np.full_like(distance,np.pi*f*f),where=distance>0)
    window = .42+.5*np.cos(np.pi*distance/(r+1))+.08*np.cos(2*np.pi*distance/(r+1))
    kernel *= np.where(distance<=r,window,0)
    kernel /= kernel.sum()
    # A short radial jinc has ~1e-3 passband ripple. Broad metre-scale erosion
    # can leak through that ripple even at gain one. Flatten only its native
    # passband using a small radial Gaussian basis, then decompose the single
    # resulting FIR. This is coefficient design, never terrain-dependent
    # iteration or repeated high-passing. All basis functions have unit DC.
    basis = np.stack([np.exp(-.5*(distance/s)**2)*(distance<=r)
                      for s in (.5,.7,1.,1.4,2.,2.8,4.,5.5)])
    basis /= basis.sum((1,2))[:,None,None]
    frequencies = np.linspace(0,.5,301)
    angles = np.linspace(0,np.pi/4,4)
    cosine = np.cos(2*np.pi*frequencies[:,None,None,None]*(
        np.cos(angles)[None,:,None,None]*x+np.sin(angles)[None,:,None,None]*y))
    responses = np.einsum('faij,kij->fak',cosine,basis).reshape(-1,len(basis))
    error = (1-np.einsum('faij,ij->fa',cosine,kernel)).ravel()
    passband = np.repeat(frequencies<=.125,4)
    weights = np.repeat(np.where(frequencies<=.125,1000.,np.where(frequencies>=.19,1.,.01)),4)
    coefficients = np.linalg.lstsq(np.vstack((responses*weights[:,None],np.eye(len(basis))*.001)),
        np.r_[np.where(passband,error,0)*weights,np.zeros(len(basis))],rcond=None)[0]
    kernel += np.einsum('k,kij->ij',coefficients,basis)
    kernel /= kernel.sum()
    eigenvalues,vectors = np.linalg.eigh(kernel)
    terms = [(float(w),((v+v[::-1])*.5).tolist())
             for w,v in zip(eigenvalues,vectors.T) if abs(w)>1e-10 and abs(v.sum())>1e-9]
    normalization = math.fsum(w*math.fsum(v)**2 for w,v in terms)
    return [(w/normalization,v) for w,v in terms]


def _high_pass(values):
    """Subtract one radial parent-band convolution on the final increment."""
    r = PROJECTION_RADIUS
    result = values.double()[r:-r,r:-r]
    for weight,taps in _projection_terms():
        _check()
        result = result-weight*refinement._filter(values.double(),taps)
    return result


def _project_increment(values, origin, margin, size, base, inherited, stage):
    """One linear parent-band removal after all physical transport is finished.

    Smooth away wavelengths below four cells first. There is no residual gain,
    nonlinear cap between filters, or feedback of projected heights into any
    process. Safety guards operate only on the finished surface.
    """
    smooth = refinement._filter(values.double(),_gaussian_taps(PROJECTION_SIGMA,DETAIL_SMOOTH_RADIUS))
    high = _high_pass(smooth)*DETAIL_GAIN[stage]
    radius = DETAIL_SMOOTH_RADIUS+PROJECTION_RADIUS
    offset = margin-radius
    inner = (slice(radius,-radius),)*2
    centre = base[inner].double()
    weight = refinement._smoothstep(0,12,centre)*(centre>0)
    delta = (inherited[inner].double()+high)*weight
    budget = centre.abs()*.49/MAX_INTENSITY
    height = (centre+delta.clamp(min=-budget,max=budget)).float()[None]
    guarded = _remove_extrema(height,base[inner][None],.1*30/2**stage,_cuda_pack(values.device))[0]
    delta = (guarded-base[inner]).clamp(min=-budget.float(),max=budget.float())
    delta = torch.where(base[inner]>0,delta,torch.zeros_like(delta))
    return delta[offset:offset+size,offset:offset+size].contiguous()


def parse_intensity(raw) -> float:
    """Canonical slider/cache-key value; reject non-finite and malformed input."""
    if raw is None or (isinstance(raw, str) and raw.strip().lower() in ('', 'off')):
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError('Invalid terrain amplification intensity') from exc
    if not math.isfinite(value):
        raise ValueError('Terrain amplification intensity must be finite')
    return round(min(MAX_INTENSITY, max(0.0, value)), 2)


def _check():
    callback = getattr(_state, 'check', None)
    if callback is not None:
        callback()


def _neighbours(value):
    """Eight batched neighbour views, replicated height at a no-flux edge."""
    padded = F.pad(value[:, None], (1, 1, 1, 1), mode='replicate')[:, 0]
    h, w = value.shape[-2:]
    return [padded[:, 1+y:1+y+h, 1+x:1+x+w] for y, x in _DIRECTIONS]


def _shift_zero(value, y, x):
    padded = F.pad(value, (1, 1, 1, 1))
    h, w = value.shape[-2:]
    return padded[..., 1+y:1+y+h, 1+x:1+x+w]


def _sum(values):
    result = values[0]
    for value in values[1:]:
        result = result + value
    return result


def _routing(height, land, cell, prior=None, exponent=None):
    neighbours = _neighbours(height)
    land_neighbours = _neighbours(land)
    drops = [(height - other).clamp_min(0) * land * mask
             for other, mask in zip(neighbours, land_neighbours)]
    weights = [(drop / (cell * distance)).pow(1.3 if exponent is None else exponent)
               for drop, distance in zip(drops, _DISTANCES)]
    if prior is not None:
        weights = [weight*(1+8*(other-prior)).clamp(.35,2.5)
                   for weight,other in zip(weights,_neighbours(prior))]
    total = _sum(weights).clamp_min(1e-12)
    weights = [weight / total for weight in weights]
    slope = torch.zeros_like(height)
    drop_max = torch.zeros_like(height)
    for drop, distance in zip(drops, _DISTANCES):
        slope = torch.maximum(slope, drop / (cell * distance))
        drop_max = torch.maximum(drop_max, drop)
    return weights, slope, drop_max


def _incoming(value, weights):
    return _sum([_shift_zero(value * weight, -y, -x)
                 for weight, (y, x) in zip(weights, _DIRECTIONS)])


def _slope_limit(delta, cell):
    """Symmetric Lipschitz envelopes: suppress isolated cuts without blurring ribs.

    The new stage's range is at most six cell widths in metres. Twelve
    eight-neighbour relaxations cover every possible constraint at .5 m/m;
    beyond twelve cells even the largest difference is admissible. Euclidean
    neighbour lengths avoid the diamond cones of a Manhattan envelope.
    The mean of lower/upper envelopes preserves both cuts and positive fans.
    """
    pack = _cuda_pack(delta.device)
    if pack is not None:
        lower, upper = delta.clone(), delta.clone()
        next_lower, next_upper = torch.empty_like(delta), torch.empty_like(delta)
        p = torch.tensor([cell], device=delta.device)
        for _ in range(12):
            _check()
            pack.run('envelope', [lower, upper, next_lower, next_upper, p],
                     [*delta.shape[-2:], delta.numel()], delta.numel())
            lower, next_lower, upper, next_upper = next_lower, lower, next_upper, upper
        return (lower+upper)*.5
    lower, upper = delta, delta
    for _ in range(12):
        _check()
        # Zero-order extension is no-flux; these artificial edges lie
        # outside the retained collar, as for the physical iterations.
        low = _neighbours(lower)
        high = _neighbours(upper)
        for index, distance in enumerate(_DISTANCES):
            lower = torch.minimum(lower, low[index]+cell*.5*distance)
            upper = torch.maximum(upper, high[index]-cell*.5*distance)
    return (lower+upper)*.5


def _smooth_increment(delta, cell):
    """Two fixed-order binomial passes round envelope facets, preserving bounds.

    Positive unit-sum weights retain the global amplitude and Lipschitz bounds.
    Smoothing only the new increment leaves the reference's own rock intact.
    """
    pack = _cuda_pack(delta.device)
    if pack is not None:
        out = torch.empty_like(delta)
        p = torch.tensor([cell], device=delta.device)
        for _ in range(2):
            _check()
            pack.run('rounding', [delta, out, p], [*delta.shape[-2:], delta.numel()], delta.numel())
            delta, out = out, delta
        return delta
    for _ in range(2):
        _check()
        values = [delta*.25]
        values += [value*(.125 if distance == 1 else .0625)
                   for value, distance in zip(_neighbours(delta), _DISTANCES)]
        delta = _sum(values)
    return delta


def _process_strength(initial, parent, cell):
    """Inherited slope/drainage, not fine residual amplitude, sets activity.

    Gentle dry ground has zero activity. Channels may incise gentler slopes;
    thermal erosion still requires a steep face in its physical transport.
    The positive 5-cell Gaussian prevents octave noise from opening this gate.
    """
    taps = _gaussian_taps(1.1,5)
    smooth = torch.stack([refinement._filter(F.pad(v.double(),(5,)*4),taps) for v in initial])
    n = _neighbours(smooth)
    gy,gx = (n[6]-n[1])/(2*cell),(n[4]-n[3])/(2*cell)
    slope = torch.hypot(gx,gy)
    face = refinement._smoothstep(.20,.65,slope)
    channel = refinement._smoothstep(.04,.20,slope)*parent[:,1].double().square()
    return torch.maximum(face,channel).float(),gx,gy


def _along_slope(values, gx, gy):
    """Positive eight-step flux stabilisation along the inherited downhill axis.

    This averages transport noise; it cannot create a new extremum, oscillation
    or wavelength. Cross-slope channel contrast survives preferentially.
    """
    norm = torch.hypot(gx,gy).clamp_min(1e-12)
    weights = [torch.exp(-((x*gy-y*gx)/(norm*d)).square()/.18)/d
               for (y,x),d in zip(_DIRECTIONS,_DISTANCES)]
    total = 1+_sum(weights)
    result = values.double()
    for _ in range(8):
        _check()
        result = (result+_sum([v*w for v,w in zip(_neighbours(result),weights)]))/total
    return result.float()


# New CUDA implementation of the tensor stencils below, not upstream shader
# text. PyTorch owns every allocation and stream. No atomics/reductions across
# pixels: each thread visits the same eight neighbours in the same order.
_CUDA_SOURCE = r'''
__device__ int dy(int d) { return d<3?-1:(d<5?0:1); }
__device__ int dx(int d) { return d==0||d==3||d==5?-1:(d==1||d==6?0:1); }
__device__ float dist(int d) { return (d==0||d==2||d==5||d==7)?1.4142135623730951f:1.f; }
__device__ long long neighbour(long long id,int d,int h,int w) {
    int y=(id/w)%h+dy(d), x=id%w+dx(d);
    return y<0||y>=h||x<0||x>=w ? -1 : id+dy(d)*w+dx(d);
}
__device__ float clampf(float v,float lo,float hi) { return fminf(hi,fmaxf(lo,v)); }
__device__ float smooth(float lo,float hi,float x) { float t=clampf((x-lo)/(hi-lo),0.f,1.f); return t*t*(3.f-2.f*t); }
extern "C" __global__ void route(const float* z,const float* base,const float* prior,
        const float* exponent,float* r,const float* p,
        long long h,long long w,long long n) {
    long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(id>=n)return;
    float weights[8],sum=0.f,slope=0.f,drop=0.f;
    for(int d=0;d<8;d++) {
        long long q=neighbour(id,d,h,w);
        float dz=q<0||base[id]<=0.f||base[q]<=0.f?0.f:fmaxf(0.f,z[id]-z[q]);
        float s=dz/(p[0]*dist(d));
        float bias=q<0?1.f:clampf(1.f+8.f*(prior[q]-prior[id]),.35f,2.5f);
        weights[d]=powf(s,exponent[id])*bias; sum+=weights[d];
        slope=fmaxf(slope,s); drop=fmaxf(drop,dz);
    }
    sum=fmaxf(sum,1.e-12f);
    for(int d=0;d<8;d++)r[d*n+id]=weights[d]/sum;
    r[8*n+id]=slope; r[9*n+id]=drop;
}
__device__ float incoming(const float* v,const float* r,long long id,int h,int w,long long n) {
    float value=0.f;
    // Tensor incoming visits outgoing directions 0..7, hence source
    // neighbours 7..0. Retain that FP32 addition order too.
    for(int d=7;d>=0;d--) { long long q=neighbour(id,d,h,w); if(q>=0)value+=v[q]*r[(7-d)*n+q]; }
    return value;
}
extern "C" __global__ void drain(const float* local,const float* r,const float* base,
        float* out,const float* p,long long h,long long w,long long n) {
    long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(id>=n)return;
    out[id]=base[id]>0.f?fminf(p[1],1.f+incoming(local,r,id,h,w,n)):0.f;
}
extern "C" __global__ void incision(const float* z,const float* initial,const float* cap,
        const float* coast,const float* hard,const float* inherited,const float* local,
        const float* r,const float* base,float* sed,float* removed,float* out,float* next,
        const float* p,long long h,long long w,long long n) {
    long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(id>=n)return;
    float rain=base[id]>0.f?1.f:0.f;
    float flow=fminf(p[1],rain+incoming(local,r,id,h,w,n)); next[id]=flow;
    float stream=fmaxf(flow,inherited[id]), slope=r[8*n+id];
    float cut=p[0]*p[2]*sqrtf(stream*p[3])*powf(fminf(slope,1.f),1.6f)/hard[id];
    cut=fminf(cut,r[9*n+id]*.22f)*coast[id]*smooth(.025f,.12f,slope);
    float height=initial[id]+clampf(z[id]-cut-initial[id],-4.f*cap[id],4.f*cap[id]);
    float actual=fmaxf(0.f,z[id]-height); sed[id]+=actual; removed[id]+=actual; out[id]=height;
}
__device__ float flux(float za,float zb,float repose,float coast,float cell,int d) {
    return fminf(fmaxf(za-zb-repose*(cell*dist(d)),0.f)*.035f,cell*.035f)*coast;
}
extern "C" __global__ void thermal(const float* z,const float* initial,const float* cap,
        const float* coast,const float* repose,const float* base,float* talus,float* out,
        const float* p,long long h,long long w,long long n) {
    long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(id>=n)return;
    float receive=0.f,send=0.f;
    for(int d=0;d<8;d++) { long long q=neighbour(id,d,h,w);
        if(q>=0&&base[id]>0.f&&base[q]>0.f) {
            send+=flux(z[id],z[q],repose[id],coast[id],p[0],d);
        }
    }
    for(int d=7;d>=0;d--) { long long q=neighbour(id,d,h,w);
        if(q>=0&&base[id]>0.f&&base[q]>0.f)
            receive+=flux(z[q],z[id],repose[q],coast[q],p[0],d);
    }
    talus[id]+=receive;out[id]=initial[id]+clampf(z[id]+receive-send-initial[id],-4.f*cap[id],4.f*cap[id]);
}
extern "C" __global__ void settling(const float* z,const float* initial,const float* cap,
        const float* coast,const float* inherited,const float* local,const float* r,
        const float* sed,float* deposit,float* out,float* next,const float* p,
        long long h,long long w,long long n) {
    long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(id>=n)return;
    float sum=0.f;for(int d=0;d<8;d++)sum+=r[d*n+id];
    float transported=sed[id]*fmaxf(0.f,1.f-sum)+incoming(sed,r,id,h,w,n);
    float slope=r[8*n+id],stream=fmaxf(inherited[id],local[id]);
    float capacity=p[0]*.025f*powf(stream*p[3],.3f)*fminf(slope*slope,1.f);
    float amount=fmaxf(transported-capacity,0.f)*.45f*(1.f-smooth(.35f,.9f,slope))*coast[id];
    float height=initial[id]+clampf(z[id]+amount-initial[id],-4.f*cap[id],4.f*cap[id]);
    float actual=fmaxf(height-z[id],0.f);deposit[id]+=actual;
    next[id]=fmaxf(transported-actual,0.f);out[id]=height;
}
extern "C" __global__ void envelope(const float* low,const float* high,float* outlow,
        float* outhigh,const float* p,long long h,long long w,long long n) {
    long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(id>=n)return;
    float lo=low[id],hi=high[id];
    for(int d=0;d<8;d++) { long long q=neighbour(id,d,h,w);if(q>=0) {
        float length=p[0]*.5f*dist(d);
        lo=fminf(lo,low[q]+length);hi=fmaxf(hi,high[q]-length);
    }} outlow[id]=lo;outhigh[id]=hi;
}
extern "C" __global__ void rounding(const float* z,float* out,const float* p,
        long long h,long long w,long long n) {
    long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(id>=n)return;
    int y=(id/w)%h,x=id%w;float value=z[id]*.25f;
    for(int d=0;d<8;d++) {
        int yy=max(0,min((int)h-1,y+dy(d))),xx=max(0,min((int)w-1,x+dx(d)));
        long long q=id+(yy-y)*w+(xx-x);
        value+=z[q]*(dist(d)==1.f?.125f:.0625f);
    }out[id]=value;
}
extern "C" __global__ void extrema(const float* z,const float* base,float* out,const float* p,
        long long h,long long w,long long n) {
    long long id=(long long)blockIdx.x*blockDim.x+threadIdx.x;if(id>=n)return;
    float lo=1.e30f,hi=-1.e30f,bl=1.e30f,bh=-1.e30f;
    for(int d=0;d<8;d++) {long long q=neighbour(id,d,h,w);if(q>=0) {
        lo=fminf(lo,z[q]);hi=fmaxf(hi,z[q]);bl=fminf(bl,base[q]);bh=fmaxf(bh,base[q]);
    }}
    out[id]=clampf(z[id],lo-fmaxf(bl-base[id],0.f)-.01f*p[0],
                            hi+fmaxf(base[id]-bh,0.f)+.01f*p[0]);
}
'''
_cuda_packs, _cuda_errors = {}, {}


def _cuda_pack(device):
    """Optional fused stencils; the same plain tensor algorithm is the fallback."""
    if torch.device(device).type != 'cuda':
        return None
    index = torch.device(device).index
    index = torch.cuda.current_device() if index is None else index
    if index in _cuda_packs or index in _cuda_errors:
        return _cuda_packs.get(index)
    import ctypes as C
    from terrain_cuda_kernels import KernelPack, _cuda_libraries
    try:
        # Reuse the repository's DLL discovery, signatures, launch validation
        # and Torch stream ownership; compile only this module's new kernels.
        pack = KernelPack.__new__(KernelPack)
        pack.device = torch.device('cuda', index)
        with torch.cuda.device(pack.device):
            torch.empty(0, device=pack.device)
            pack.nvrtc, pack.driver = _cuda_libraries()
            pack._bind()
            context = C.c_void_p()
            pack.check(pack.ctx_current(C.byref(context)))
            pack.context = context.value
            program = C.c_void_p()
            pack.check_nvrtc(pack.create(C.byref(program), _CUDA_SOURCE.encode(), b'amplification.cu', 0, None, None))
            major, minor = torch.cuda.get_device_capability(pack.device)
            options = [f'--gpu-architecture=compute_{major}{minor}'.encode(),
                       b'--std=c++14', b'--fmad=false']
            try:
                result = pack.compile(program, len(options), (C.c_char_p*len(options))(*options))
                if result:
                    size = C.c_size_t()
                    pack.log_size(program, C.byref(size))
                    log = C.create_string_buffer(size.value)
                    pack.log(program, log)
                    raise RuntimeError(log.value.decode(errors='replace'))
                size = C.c_size_t()
                pack.check_nvrtc(pack.ptx_size(program, C.byref(size)))
                ptx = C.create_string_buffer(size.value)
                pack.check_nvrtc(pack.ptx(program, ptx))
                pack.module = C.c_void_p()
                pack.check(pack.load(C.byref(pack.module), C.cast(ptx, C.c_void_p), 0, None, None))
                pack.functions = {}
                for name in ('route', 'drain', 'incision', 'thermal', 'settling', 'envelope', 'rounding', 'extrema'):
                    function = C.c_void_p()
                    pack.check(pack.get_function(C.byref(function), pack.module, name.encode()))
                    pack.functions[name] = function
            finally:
                pack.destroy(C.byref(program))
        _cuda_packs[index] = pack
    except Exception as exc:
        _cuda_errors[index] = f'{type(exc).__name__}: {exc}'
    return _cuda_packs.get(index)


def kernel_status():
    return dict(devices=sorted(_cuda_packs), errors=dict(_cuda_errors), compiler='torch-bundled-nvrtc')


def _remove_extrema(height, base, cell, pack=None):
    """Do not introduce isolated pits/peaks more prominent than the input's."""
    # The projection supplies sliced interiors. CUDA addresses flat buffers;
    # retaining a larger source row stride would corrupt the reference stencil.
    height,base = height.contiguous(),base.contiguous()
    if pack is not None:
        out = torch.empty_like(height)
        p = torch.tensor([cell], device=height.device)
        for _ in range(3):
            _check()
            pack.run('extrema', [height, base, out, p], [*height.shape[-2:], height.numel()], height.numel())
            height, out = out, height
        return height
    low_base, high_base = _neighbours(base)[0], _neighbours(base)[0]
    for value in _neighbours(base)[1:]:
        low_base, high_base = torch.minimum(low_base, value), torch.maximum(high_base, value)
    for _ in range(3):
        _check()
        low, high = _neighbours(height)[0], _neighbours(height)[0]
        for value in _neighbours(height)[1:]:
            low, high = torch.minimum(low, value), torch.maximum(high, value)
        height = height.clamp(min=low-(low_base-base).clamp_min(0)-cell*.01,
                              max=high+(base-high_base).clamp_min(0)+cell*.01)
    return height


def _simulate_cuda(initial, base, cap, coast, hardness, repose, inherited, prior, exponent, stage, pack):
    """Two launches per hydraulic step, one per thermal step; Torch buffers only."""
    shape, n = initial.shape, initial.numel()
    p = torch.tensor([30/2**stage, _FLOW_CELLS*(_FLOW_AREA/(30/2**stage)**2),
                      EROSION_RATE[stage],(30/2**stage)**2/AREA_REFERENCE_M2], device=base.device)
    integers = [*shape[-2:], n]
    height, out = initial.clone(), torch.empty_like(initial)
    local, next_local = (base > 0).float(), torch.empty_like(initial)
    sediment, next_sediment = torch.zeros_like(initial), torch.empty_like(initial)
    removed, talus, deposited = (torch.zeros_like(initial) for _ in range(3))
    routing = torch.empty((10, *shape), device=base.device)
    for step in range(ITERATIONS[stage][0]):
        _check()
        # Four explicit transport substeps per routing update reduce expensive
        # powers without increasing the conservative two-cell reach budget.
        if step % 4 == 0:
            pack.run('route', [height, base, prior, exponent, routing, p], integers, n)
        pack.run('incision', [height, initial, cap, coast, hardness, inherited, local, routing,
                             base, sediment, removed, out, next_local, p], integers, n)
        height, out, local, next_local = out, height, next_local, local
    for _ in range(ITERATIONS[stage][1]):
        _check()
        pack.run('thermal', [height, initial, cap, coast, repose, base, talus, out, p], integers, n)
        height, out = out, height
    for step in range(ITERATIONS[stage][2]):
        _check()
        if step % 4 == 0:
            pack.run('route', [height, base, prior, exponent, routing, p], integers, n)
        pack.run('settling', [height, initial, cap, coast, inherited, local, routing,
                             sediment, deposited, out, next_sediment, p], integers, n)
        height, out, sediment, next_sediment = out, height, next_sediment, sediment
    return height, torch.maximum(local, inherited), removed, talus, deposited


def _native_drainage(base, origins=None, seed=0):
    """Stationary MFD at 30 m: contributing area only, no terrain alteration."""
    land = (base > 0).float()
    local = land.clone()
    origins = origins or [(0,0)]*base.shape[0]
    noise = _physical_noise(base,origins,seed,30.,((1.,900.,41),))
    exponent = 1.3+.18*noise
    prior = torch.zeros_like(base)
    limit = _FLOW_CELLS*_FLOW_AREA/900
    pack = _cuda_pack(base.device)
    if pack is not None:
        routing = torch.empty((10, *base.shape), device=base.device)
        out = torch.empty_like(base)
        p = torch.tensor([30., limit], device=base.device)
        integers = [*base.shape[-2:], base.numel()]
        pack.run('route', [base, base, prior, exponent, routing, p], integers, base.numel())
        for _ in range(DRAIN_ITERATIONS):
            _check()
            pack.run('drain', [local, routing, base, out, p], integers, base.numel())
            local, out = out, local
    else:
        weights, _, _ = _routing(base, land, 30.,prior,exponent)
        for _ in range(DRAIN_ITERATIONS):
            _check()
            local = (land+_incoming(local, weights)).clamp_max(limit)*land
    flow = torch.log1p(local*(900/_FLOW_AREA))/_FLOW_LOG*land
    zeros = torch.zeros_like(base)
    return torch.stack((zeros, flow, zeros, zeros, zeros, flow), dim=1)


def _physical_noise(base, origins, seed, cell, octaves=((.5,180.,31),(.35,48.,32),(.15,12.,33))):
    rows = torch.arange(base.shape[-2],device=base.device,dtype=torch.int64)[None,:,None]
    cols = torch.arange(base.shape[-1],device=base.device,dtype=torch.int64)[None,None,:]
    oy = torch.tensor([o[0] for o in origins],device=base.device)[:,None,None]
    ox = torch.tensor([o[1] for o in origins],device=base.device)[:,None,None]
    py,px = (rows+oy).double()*cell,(cols+ox).double()*cell
    return sum(weight*refinement._value_noise(seed&0xffffffff,py,px,metres,stream)
               for weight,metres,stream in octaves).float()


def _erode(base, parent, origins, seed, stage):
    """Whole-tensor stencils; batch dimension never participates in arithmetic."""
    cell = 30.0 / 2**stage
    initial = base + parent[:, 0]
    height = initial.clone()
    land = (base > 0).float()
    coast = refinement._smoothstep(0, 12, base) * land
    # Relief is measured over 9 cells; each new stage's incremental slope
    # budget shrinks in flat terrain. Parent erosion must survive: clipping the
    # total to a finer stage's metre budget would erase ravines and terrace
    # their sides. Bounds apply about I_s, never about R_s.
    hi = F.max_pool2d(base[:, None], 9, 1, 4)[:, 0]
    lo = -F.max_pool2d(-base[:, None], 9, 1, 4)[:, 0]
    cap = torch.minimum((hi-lo)*RELIEF_SHARE[stage],
                        torch.full_like(base,min(cell*3,DETAIL_CAP_METRES[stage]))) * coast
    activity,gx,gy = _process_strength(initial,parent,cell)
    cap = cap*activity
    # Smooth planes/glacier-like ramps should not be incised merely because
    # the native drainage guide is large. Average absolute curvature, rather
    # than signed curvature, so saddles and inflection lines do not punch
    # isolated holes in the resistance field (radius 1 + radius 4 = 5).
    neighbours = _neighbours(base)
    bend = (base-_sum([neighbours[k]*.25 for k in (1,6,3,4)])).abs()/cell
    bend = F.avg_pool2d(bend[:,None], 9, 1, 4)[:,0]
    cap = cap*refinement._smoothstep(.0003, .006, bend)
    # Physical coordinates and common hash streams keep strata coherent across
    # scales. No single-cell white noise: the finest wavelength is 12 metres.
    noise = _physical_noise(base,origins,seed,cell)
    repose = .72 + .16*noise
    # A periodic height cosine cuts regular contour bands into even a smooth
    # hillside. Aperiodic, seed-hashed layers retain broad geological variation
    # without introducing a preferred vertical spacing or a sinusoidal carrier.
    strata = refinement._value_noise(seed&0xffffffff,base.double(),noise.double()*12,12.,47).float()
    hardness = .95+.2*strata+.2*noise
    # Sheet incision of a uniform slope mostly exposes the reconstruction's
    # tiny residuals. Confine hydraulic work to the inherited drainage lines;
    # thermal transport remains free to act on steep faces independently.
    hardness = hardness/(.04+.96*parent[:,1].clamp(0,1).square())
    # The guide encodes physical contributing area in stage-2 cell units.
    # Decode m², then rescale to this stage's cell area (x16 per stage). A
    # fixed physical log range keeps the guide comparable at every LOD.
    prior = parent[:,5 if parent.shape[1]>5 else 1].contiguous()
    exponent = 1.3+.18*noise
    inherited = torch.expm1(prior.clamp(0, 1)*_FLOW_LOG) * (_FLOW_AREA/cell**2) * land
    stream_limit = _FLOW_CELLS * (_FLOW_AREA/cell**2)
    local = land.clone()
    stream = torch.maximum(inherited, local)
    sediment = torch.zeros_like(height)
    removed = torch.zeros_like(height)
    talus = torch.zeros_like(height)
    deposited = torch.zeros_like(height)

    def bounded(candidate):
        # Only a generous emergency bound during transport: repeatedly applying
        # a soft limiter to the accumulated delta would itself lift incised
        # ground and manufacture sediment during the subsequent passes.
        delta = (candidate-initial).clamp(-4*cap, 4*cap)
        return initial + delta

    pack = _cuda_pack(base.device)
    if pack is not None:
        height, stream, removed, talus, deposited = _simulate_cuda(
            initial, base, cap, coast, hardness, repose, inherited.contiguous(), prior, exponent, stage, pack)
    else:
        for step in range(ITERATIONS[stage][0]):
            _check()
            if step % 4 == 0:
                weights, slope, drop = _routing(height, land, cell,prior,exponent)
            local = (land + _incoming(local, weights)).clamp_max(stream_limit)
            stream = torch.maximum(inherited, local)
            # Stream power sqrt(A/hectare)*S^1.6. The guide is rescaled to
            # fine cells for accumulation, but physical area in this law must
            # be invariant under that rescaling. Otherwise fine stages drive
            # most cells into the relief cap and expose its residual texture.
            area = stream*(cell**2/AREA_REFERENCE_M2)
            cut = cell*EROSION_RATE[stage] * area.sqrt() * slope.clamp_max(1).pow(1.6) / hardness
            cut = torch.minimum(cut, drop*.22) * coast * refinement._smoothstep(.025, .12, slope)
            next_height = bounded(height-cut)
            actual = (height-next_height).clamp_min(0)
            sediment = sediment + actual
            removed = removed + actual
            height = next_height

        for _ in range(ITERATIONS[stage][1]):
            _check()
            others = _neighbours(height)
            masks = _neighbours(land)
            flux = [torch.minimum((height-other-repose*(cell*d)).clamp_min(0)*.035,
                                  torch.full_like(height, cell*.035))*land*mask*coast
                    for other, mask, d in zip(others, masks, _DISTANCES)]
            receive = _sum([_shift_zero(f, -y, -x) for f, (y, x) in zip(flux, _DIRECTIONS)])
            height = bounded(height + receive - _sum(flux))
            talus = talus + receive

        for step in range(ITERATIONS[stage][2]):
            _check()
            if step % 4 == 0:
                weights, slope, _ = _routing(height, land, cell,prior,exponent)
            # Carry actual incised sediment; settle when transport capacity falls.
            outgoing = _sum(weights)
            sediment = sediment*(1-outgoing).clamp_min(0) + _incoming(sediment, weights)
            capacity = cell*.025 * (stream*(cell**2/AREA_REFERENCE_M2)).pow(.3) * slope.square().clamp_max(1)
            # The short budget otherwise drops the whole incised load back onto
            # steep source slopes before it reaches a fan. Stable settling fades
            # out between ~19° and ~42°; scree has its separate thermal transport.
            settling = 1-refinement._smoothstep(.35, .9, slope)
            settle = (sediment-capacity).clamp_min(0)*.45*settling*coast
            next_height = bounded(height+settle)
            actual = (next_height-height).clamp_min(0)
            deposited = deposited + actual
            sediment = (sediment-actual).clamp_min(0)
            height = next_height

    increment = cap*torch.tanh((height-initial)/(cap+1e-9))
    increment = _slope_limit(increment, cell)
    # A neighbourhood may reach across a coast/flat boundary; retain the local
    # relief bound there too. The parent is already a smooth bounded field.
    increment = _smooth_increment(increment.clamp(-cap, cap), cell)
    increment = _along_slope(increment,gx,gy)
    height = initial + increment
    height = _remove_extrema(height, base, cell, pack)
    _, slope, _ = _routing(height, land, cell)
    flow = torch.log1p(stream*(cell**2/_FLOW_AREA)) / _FLOW_LOG * coast
    deposit = deposited / (deposited + cell*.12)
    scree = (talus / (talus + cell*.15) +
             .15*torch.exp(-((slope-repose)/.18).square())).clamp(0, 1)*coast
    rock = refinement._smoothstep(.45, 1.4, slope) * (1-deposit) * (
        .35 + .65*removed/(removed+cell*.1)) * coast
    # This is D_s = E_s-R_s, with coast/sign safety included in E_s itself.
    delta = (height-base)*coast
    sea_budget = base.abs()*.49/MAX_INTENSITY
    delta = delta.clamp(min=-sea_budget, max=sea_budget)*land
    return torch.stack((delta, flow, deposit, scree, rock, flow), dim=1)


def _upsample(world, source, target, i1, j1, i2, j2):
    """Notched C2 height/stream interpolation; positive material/area splines.

    The old block-mean inverse amplified a lattice carrier. The existing
    reconstruction filter preserves resolved amplitudes but rejects that
    carrier at source Nyquist; no mean constraints or decimation are involved.
    """
    scale = 2**(target-source)
    # A separable spline leaves weak square spectral images at fine zooms.
    # LP=I-(I-G)^6 is radial, flat through resolved source frequencies and
    # rolls off across source Nyquist. Apply it on the fine grid, not by
    # decimating or constraining any parent cells. Convolving C2 stays C2.
    sigma = TRANSFER_SIGMA*scale
    anti = math.ceil(4*sigma*math.sqrt(TRANSFER_ORDER)) if source else 0
    r0,r1,rows,fy = refinement._axis(i1-anti,i2+anti,scale,world.device)
    c0,c1,cols,fx = refinement._axis(j1-anti,j2+anti,scale,world.device)
    radius = refinement.SPLINE_RADIUS
    values = _read_stage(world,source,r0-radius,c0-radius,r1+radius+1,c1+radius+1)
    taps_y,taps_x = refinement._cubic_taps(rows,fy),refinement._cubic_taps(cols,fx)
    coefficients = refinement._filter(values[0].double(),refinement._INTERPOLATE)
    height = refinement._evaluate(coefficients,taps_y,taps_x)
    if anti:
        filtered = torch.zeros((i2-i1,j2-j1),device=world.device,dtype=torch.float64)
        for k in range(1,TRANSFER_ORDER+1):
            taps = _gaussian_taps(sigma*math.sqrt(k),anti)
            filtered += math.comb(TRANSFER_ORDER,k)*(-1)**(k+1)*refinement._filter(height,taps)
        height = filtered
        first,_,guide_rows,guide_fy = refinement._axis(i1,i2,scale,world.device)
        left,_,guide_cols,guide_fx = refinement._axis(j1,j2,scale,world.device)
        taps_y = refinement._cubic_taps(guide_rows+first-r0,guide_fy)
        taps_x = refinement._cubic_taps(guide_cols+left-c0,guide_fx)
    result = [height]
    for index,channel in enumerate(values[1:],1):
        if index==1:
            # Preserve a narrow stream's confidence through repeated stages;
            # smoothing its samples at every x4 transfer erased small streams.
            channel = refinement._filter(channel.double(),refinement._INTERPOLATE)
            result.append(refinement._evaluate(channel,taps_y,taps_x).clamp(0,1))
        else:
            channel = channel[radius:-radius,radius:-radius]
            result.append(refinement._evaluate(channel.double(),taps_y,taps_x))
    return torch.stack(result).float()


def _native_position(world, stage, origin, shape):
    """C2 terrain position over a fixed 150 m native stencil, in slope units."""
    y,x = origin
    h,w = shape
    r0,r1,rows,fy = refinement._axis(y,y+h,2**stage,world.device)
    c0,c1,cols,fx = refinement._axis(x,x+w,2**stage,world.device)
    native = _reference(world,0,r0-2,c0-2,r1+3,c1+3)
    mean = F.avg_pool2d(native[None,None],5,1)[0,0]
    position = (mean-native[2:-2,2:-2])/30
    return refinement._evaluate(position.double(),refinement._cubic_taps(rows,fy),
                                refinement._cubic_taps(cols,fx)).float()


def _material_smooth(values, cell, *, full_precision=False):
    """Fixed-order isotropic Gaussian, >=2 m; no batch-dependent convolution."""
    sigma = max(.8,2/cell)
    radius = math.ceil(3*sigma)
    taps = _gaussian_taps(sigma,radius)
    result = torch.stack([refinement._filter(F.pad(value.double(),(radius,)*4),taps)
                          for value in values])
    return result if full_precision else result.float()


def _channel_shape(height, gx, gy, cell):
    """Cross-slope concavity, not circular pits or contour-parallel rock bands."""
    n = _neighbours(height)
    dxx = (n[3]+n[4]-2*height)/cell**2
    dyy = (n[1]+n[6]-2*height)/cell**2
    dxy = (n[0]+n[7]-n[2]-n[5])/(4*cell**2)
    norm = (gx*gx+gy*gy).clamp_min(1e-12)
    cross = (gy*gy*dxx+gx*gx*dyy-2*gx*gy*dxy)/norm
    along = (gx*gx*dxx+gy*gy*dyy+2*gx*gy*dxy)/norm
    elongation = (cross-along.clamp_min(0))/(cross.abs()+along.abs()+1e-10)
    return refinement._smoothstep(.15,.75,elongation)


def _drainage_lines(guide):
    """Native flow ridges, classified before C2 magnification, never after it.

    Differencing a quantised, magnified guide and multiplying by scale squared
    amplifies tiny interpolation errors into grid-aligned dashes. Classifying
    on its source grid avoids that alias and retains drainage across zooms.
    """
    smooth = _smooth_increment(guide,30.)
    broad = F.avg_pool2d(smooth[:,None],5,1,2)[:,0]
    contrast = (smooth-broad).clamp_min(0)
    return refinement._smoothstep(.38,.64,smooth)*refinement._smoothstep(.001,.025,contrast)


def _calibrate_surface(world, base, raw, parent, origins, stage):
    """Physical, request-independent material masks; raw guide stays separate.

    Cumulative transported sediment is not coverage. Fans require a low slope
    and positive native terrain position; scree occupies the repose apron;
    exposed rock occupies steep net-eroded faces. Parent C2 drainage supplies
    thin flow lines without displaying the finest-grid MFD counter pattern.
    """
    cell = 30/2**stage
    neighbours = _neighbours(base)
    gy = (neighbours[6]-neighbours[1])/(2*cell)
    gx = (neighbours[4]-neighbours[3])/(2*cell)
    slope = (gy*gy+gx*gx).sqrt()
    position = torch.stack([_native_position(world,stage,origin,base.shape[-2:]) for origin in origins])
    f = _material_smooth(raw[:,2],cell).clamp(0,1)
    sediment = f/(f+8*(1-f)+1e-9)
    valley = refinement._smoothstep(-.03,.09,position)
    deposit = sediment*valley*(1-refinement._smoothstep(.45,1.05,slope))
    # A pre-existing fan remains material when a finer pass adds only rills.
    deposit = torch.maximum(deposit,.75*parent[:,2])
    foot = refinement._smoothstep(-.06,.12,position)
    scree = (_material_smooth(raw[:,3],cell)*foot*refinement._smoothstep(.2,.55,slope)*
             (1-refinement._smoothstep(.85,1.25,slope)))
    erosion = _material_smooth((parent[:,0]-raw[:,0]).clamp_min(0)/cell,cell)
    exposed = .3+.7*refinement._smoothstep(.03,.8,erosion)
    rock = refinement._smoothstep(.45,1.1,slope)*exposed*(1-.85*deposit)*(1-.35*scree)
    # Parent lines survive every scale. Local concavity on the eroded surface
    # also reveals sub-native rills: their true area can be below one native
    # cell, so thresholding the inherited area alone would hide them forever.
    eroded = _material_smooth(base+raw[:,0],cell,full_precision=True)
    sigma = max(1.5,6/cell)
    radius = math.ceil(3*sigma)
    taps = _gaussian_taps(sigma,radius)
    broad = torch.stack([refinement._filter(F.pad(value,(radius,)*4),taps)
                         for value in eroded])
    concavity = (broad-eroded)/max(6,1.5*cell)
    channel = (refinement._smoothstep(.016,.11,concavity)*
               refinement._smoothstep(.03,.3,erosion)*refinement._smoothstep(.04,.2,slope)*
               _channel_shape(eroded,gx.double(),gy.double(),cell))
    flow = torch.maximum(parent[:,1],channel*(1-.35*deposit))
    coast = refinement._smoothstep(0,12,base)*(base>0)
    return torch.stack((flow,deposit,scree,rock),dim=1).clamp(0,1)*coast[:,None]


def _stage_field(world, stage):
    fields = world.__dict__.setdefault('_terrain_amplification_fields', {})
    key = (VERSION,stage)
    if key in fields:
        return fields[key]
    profile = getattr(world, '_terrain_profile', None)
    cache_device = world.device if getattr(profile, 'gpu_windows', False) else 'cpu'
    size, stride, margin = GEOMETRY[stage]
    assert margin >= _reach(stage), 'Amplification collar does not contain its stencil reach'
    # Fixed-point products/sums stay below 2**53 (total delta < 30 metres).
    weight = linear_weight_window(size, world.device, torch.float64)
    weight = (weight*65536).round().clamp_min(1) / 65536

    def generate(ctxs):
        _check()
        origins = [(ctx[1]*stride-margin, ctx[2]*stride-margin) for ctx in ctxs]
        count = size + 2*margin
        bases, parents = [], []
        for y, x in origins:
            bases.append(_reference(world, stage, y, x, y+count, x+count))
            parents.append(_upsample(world, stage-2, stage, y, x, y+count, x+count)
                           if stage > 0 else None)
        bases = torch.stack(bases)
        if stage > 0:
            parents = torch.stack(parents)
            raw = _erode(bases,parents,origins,int(getattr(world,'seed',0)),stage)
            surface = _calibrate_surface(world,bases,raw,parents,origins,stage)
            deltas = [_project_increment(raw[k,0]-parents[k,0],origin,margin,size,
                        bases[k],parents[k,0],stage) for k,origin in enumerate(origins)]
            values = torch.cat((torch.stack(deltas)[:,None],surface[...,margin:-margin,margin:-margin],
                                raw[:,5:6,margin:-margin,margin:-margin]),dim=1).double()
        else:
            values = _native_drainage(bases,origins,int(getattr(world,'seed',0)))
            values[:,1] = _drainage_lines(values[:,5])
            values = values[...,margin:-margin,margin:-margin].double()
        values = (values*65536).round() / 65536
        return [torch.cat((value*weight, weight[None]), dim=0).to(cache_device).contiguous()
                for value in values]

    # Delta, four public materials, hidden log contributing area, PoU weight.
    # Display calibration must never overwrite the inherited physical guide.
    tensor = InfiniteTensor(shape=(7, None, None), f=generate,
                            output_window=TensorWindow(size=(7, size, size),
                                                       stride=(7, stride, stride)),
                            batch_size=4, device=cache_device, dtype=torch.float64,
                            tile_store=world.tile_store,
                            tensor_id=f'amplification_{VERSION}_{refinement.VERSION}_{stage}_'
                                      f'{"fused" if _cuda_pack(world.device) is not None else "tensor"}_'
                                      f'{int(getattr(world, "seed", 0))}')
    observer = getattr(world, '_terrain_window_scheduler', None)
    if observer is not None:
        observer._install_recursive(tensor)
    fields[key] = tensor
    return tensor


def _read_stage(world, stage, i1, j1, i2, j2):
    fused = _stage_field(world, stage)[:, i1:i2, j1:j2].to(world.device)
    return (fused[:6] / fused[6:]).float()


@torch.inference_mode()
def sample_drainage(world, i1, j1, i2, j2) -> torch.Tensor:
    """Native 30 m contributing area in m², FP32 contiguous on world.device.

    Bounds are integer native cell edges, as for sample_refined at level 0;
    at most 1024² cells may be read. The cached, seed/version-scoped 160-step
    multiple-flow field has exact overlap/negative-coordinate invariance.
    Oceans return zero. This is finite-horizon drainage, not a global solve.
    The fixed-point log cache has <0.012% relative area quantisation error.
    Reading drainage never applies erosion or changes the native DEM.
    """
    if any(not isinstance(v,int) for v in (i1,j1,i2,j2)):
        raise ValueError('Drainage coordinates must be integers')
    if i2<=i1 or j2<=j1 or (i2-i1)*(j2-j1)>1024**2:
        raise ValueError('Drainage read exceeds bounded output budget')
    _check()
    observer = getattr(world,'_terrain_window_scheduler',None)
    with observer.scope(getattr(_state,'check',None)) if observer is not None else nullcontext():
        guide = _read_stage(world,0,i1,j1,i2,j2)[5].clamp(0,1)
    return (torch.expm1(guide*_FLOW_LOG)*_FLOW_AREA).float().contiguous()


def _surface_blur(values, sigma):
    """Fixed-order positive FIR; artificial edges stay outside the read collar."""
    radius = math.ceil(3*sigma)
    return refinement._filter(F.pad(values.double(),(radius,)*4),_gaussian_taps(sigma,radius))


def _surface_margin(level):
    # Broad shape/gradient radius 10, six uphill bilinear traces, final blur.
    # Flow's cross-slope ridge test plus three longitudinal averages fit too.
    step = max(1.,2/(30/2**level))
    reach = 10+6*(math.ceil(step)+1)+4
    return 8*math.ceil(reach/8)


def _surface_native(world, a, b, c, d):
    """Globally anchored square native reads, independent of request shape.

    Some real-world decoder adapters resize a rectangular native read's short
    side. Fixed 128-cell squares avoid that ambiguity in terrain position and
    land masks, matching the coherent native datum of the drainage phasor.
    The immutable-world cache holds at most 32 FP32 blocks (2 MiB); its key
    includes SURFACE_VERSION. Only the result buffer is written. Callback
    heights and final masks never enter this cache.
    """
    result = torch.empty((c-a,d-b),device=world.device,dtype=torch.float64)
    size = 128
    cache = world.__dict__.setdefault('_terrain_surface_native_blocks',OrderedDict())
    for y in range(a//size*size,c,size):
        for x in range(b//size*size,d,size):
            _check()
            key = (SURFACE_VERSION,refinement.VERSION,type(world).__module__,type(world).__qualname__,
                   str(world.device),int(getattr(world,'seed',0)),y,x)
            # pop/reinsert also tolerates another reader evicting this key;
            # duplicate cold reads are harmless, with no lock around inference.
            block = cache.pop(key,None)
            if block is None:
                block = _reference(world,0,y,x,y+size,x+size).to(world.device).float().contiguous()
            cache[key] = block
            if len(cache)>32:
                cache.popitem(last=False)
            top,left,bottom,right = max(a,y),max(b,x),min(c,y+size),min(d,x+size)
            result[top-a:bottom-a,left-b:right-b] = block[top-y:bottom-y,left-x:right-x]
    return result


def _surface_guides(world, level, bounds):
    """Positive C2 anti-imaging of native log area, ridge contrast and position.

    A small positive source Gaussian followed by the cubic B-spline suppresses
    spectral images without the interpolating inverse's ringing. Classification
    follows interpolation so narrow drainage peaks survive fine zooms. Native
    land/sea stays independent of an arbitrary datum offset in height_fn.
    """
    i1,j1,i2,j2 = bounds
    scale = 2**level
    r0,r1,rows,fy = refinement._axis(i1,i2,scale,world.device)
    c0,c1,cols,fx = refinement._axis(j1,j2,scale,world.device)
    margin = 7
    a,b,c,d = r0-margin,c0-margin,r1+margin+1,c1+margin+1
    area = torch.cat([torch.cat([sample_drainage(world,y,x,min(y+1024,c),min(x+1024,d))
                    for x in range(b,d,1024)],dim=1) for y in range(a,c,1024)],dim=0).double()
    native = _surface_native(world,a,b,c,d)
    log_area = torch.log1p(area/900)
    contrast = (log_area-_surface_blur(log_area,1.)).clamp_min(0)
    position = (_surface_blur(native,1.5)-native)/30
    taps_y = refinement._cubic_taps(rows+margin,fy)
    taps_x = refinement._cubic_taps(cols+margin,fx)
    fields = [refinement._evaluate(_surface_blur(v,.45),taps_y,taps_x)
              for v in (log_area,contrast,position,refinement._signed_sqrt(native))]
    return fields[0],fields[1],fields[2],fields[3]>0


def _surface_gather(values, dy, dx):
    """Bilinear upstream gather, with request-independent fractional arithmetic.

    Split displacement before adding integer indices: normalised grid_sample
    coordinates and floating local indices otherwise introduce shape-dependent
    rounding at overlaps. Clamped artificial edges are outside the collar.
    """
    h,w = values.shape
    iy,ix = torch.floor(dy).long(),torch.floor(dx).long()
    fy,fx = dy-iy,dx-ix
    y = torch.arange(h,device=values.device)[:,None]+iy
    x = torch.arange(w,device=values.device)[None,:]+ix
    a = values[y.clamp(0,h-1),x.clamp(0,w-1)]
    b = values[y.clamp(0,h-1),(x+1).clamp(0,w-1)]
    c = values[(y+1).clamp(0,h-1),x.clamp(0,w-1)]
    d = values[(y+1).clamp(0,h-1),(x+1).clamp(0,w-1)]
    return (a*(1-fx)+b*fx)*(1-fy)+(c*(1-fx)+d*fx)*fy


@torch.inference_mode()
def sample_surface(world, level, i1, j1, i2, j2, height_fn, *, check=None) -> torch.Tensor:
    """Diagnose FINAL metre heights: flow/deposit/scree/rock, FP32 (4,H,W).

    Integer cell edges and the 1024² output budget follow sample_refined;
    levels 0..6 work, including intermediate negative LODs. height_fn receives
    a 32..56-cell collar containing every stencil (large reads are split into
    bounded pieces). Fixed-order FIRs and integer/fraction split gathers give
    exact overlaps, including negative indices, for a height_fn that agrees on
    its own overlaps; no request statistics enter the masks.
    Heights are never written. Native drainage/geometry are cached; final masks are not,
    because a callback can change its heights. External caches must include
    SURFACE_VERSION AND their height-provider identity/intensity. Land/sea is
    read from the native world, so changing height_fn's datum has no effect;
    FP32 inputs retain their usual altitude-dependent rounding precision.
    """
    if not isinstance(level,int) or not 0<=level<=6:
        raise ValueError('Unsupported refinement level')
    if any(not isinstance(v,int) for v in (i1,j1,i2,j2)):
        raise ValueError('Refinement coordinates must be integers')
    if i2<=i1 or j2<=j1 or (i2-i1)*(j2-j1)>1024**2:
        raise ValueError('Refinement read exceeds bounded output budget')
    if not callable(height_fn):
        raise ValueError('height_fn must return final displayed heights')
    previous = getattr(_state,'check',None)
    _state.check = check
    observer = getattr(world,'_terrain_window_scheduler',None)
    try:
        with observer.scope(check) if observer is not None else nullcontext():
            _check()
            margin = _surface_margin(level)
            bounds = (i1-margin,j1-margin,i2+margin,j2+margin)
            rows = []
            for y in range(bounds[0],bounds[2],1024):
                pieces = []
                for x in range(bounds[1],bounds[3],1024):
                    _check()
                    y2,x2 = min(y+1024,bounds[2]),min(x+1024,bounds[3])
                    piece = height_fn(y,x,y2,x2)
                    if not isinstance(piece,torch.Tensor) or piece.shape!=(y2-y,x2-x):
                        raise ValueError('height_fn returned an invalid height window')
                    pieces.append(piece.to(device=world.device,dtype=torch.float64))
                rows.append(torch.cat(pieces,dim=1))
            height = torch.cat(rows,dim=0)
            cell = 30/2**level
            fine,broad = _surface_blur(height,1.1),_surface_blur(height,3.)
            n = _neighbours(fine[None])
            gy,gx = (n[6][0]-n[1][0])/(2*cell),(n[4][0]-n[3][0])/(2*cell)
            slope = torch.hypot(gx,gy)
            bn = _neighbours(broad[None])
            by,bx = (bn[6][0]-bn[1][0])/(2*cell),(bn[4][0]-bn[3][0])/(2*cell)
            norm = torch.hypot(bx,by).clamp_min(1e-12)
            uy,ux = by/norm,bx/norm
            concavity = (broad-fine)/(3*cell)
            log_area,contrast,position,land = _surface_guides(world,level,bounds)
            _check()
            ss = refinement._smoothstep
            native_flow = ss(3.3,6.,log_area)*ss(.08,.45,contrast)
            shape = _channel_shape(fine[None],bx[None],by[None],cell)[0]
            across = max(2.,1.5/cell)
            shoulders = (_surface_gather(concavity,ux*across,-uy*across)+
                         _surface_gather(concavity,-ux*across,uy*across))*.5
            prominence = (concavity-shoulders).clamp_min(0)
            rills = (ss(.001+.0005*cell,.020+.008*cell,concavity)*shape.square()*
                     ss(.0003,.006,prominence)*ss(.06,.25,slope))
            # A broad bowl is not a stream line. The cross-slope ridge test
            # rejects it; positive longitudinal averages reject isolated dots.
            for _ in range(3):
                _check()
                rills = (rills+_surface_gather(rills,uy*2,ux*2)+
                         _surface_gather(rills,-uy*2,-ux*2))/3
            flow = torch.maximum(native_flow,rills)
            water = torch.maximum(ss(1.8,4.7,log_area),.75*rills)
            noise = _physical_noise(height[None],[bounds[:2]],int(getattr(world,'seed',0)),cell,
                                    ((.65,32.,81),(.35,96.,82)))[0].double()
            face = ss(.55+.07*noise,.95+.07*noise,slope)
            crest = ss(.32,.7,slope)*ss(.01,.08,-concavity)
            rock = torch.maximum(face,.9*crest)*(1-.35*water)*ss(-.15,.35,noise)
            valley = torch.maximum(ss(.001,.025,concavity),ss(0,.08,position))
            deposit = water*valley*(1-ss(.45,.85,slope))*(1-.65*rock)
            step = max(1.,2/cell)
            dy,dx = uy*step,ux*step
            source = ss(.12,.5,rock)
            apron = source
            for _ in range(6):
                _check()
                apron = torch.maximum(source,.85*_surface_gather(apron,dy,dx))
            scree = (ss(.09,.48,apron)*ss(.40,.53,slope)*(1-ss(.75,.93,slope))*(1-.75*rock))
            fields = torch.stack([_surface_blur(v,1.1) for v in (flow,deposit,scree,rock)])
            fields[0] = ss(.10+.012*cell,.36+.045*cell,fields[0])
            fields = fields.clamp(0,1)*land[None]
            return fields[:,margin:-margin,margin:-margin].float().contiguous()
    finally:
        _state.check = previous


@torch.inference_mode()
def _amplified_height(world, level, i1, j1, i2, j2, *, intensity=1.0, check=None):
    """The mse-v5 height path, kept independent of displayed surface diagnostics."""
    intensity = parse_intensity(intensity)
    # Delegate validation, native convention and bounds to the existing entry.
    base = refinement.sample_refined(world, level, i1, j1, i2, j2, check=check)
    if level < 2 or intensity == 0:
        return base
    observer = getattr(world, '_terrain_window_scheduler', None)
    previous = getattr(_state, 'check', None)
    _state.check = check
    try:
        with observer.scope(check) if observer is not None else nullcontext():
            fields = (_read_stage(world, level, i1, j1, i2, j2) if level in STAGE_LEVELS else
                      _upsample(world, level-1, level, i1, j1, i2, j2))
            _check()
            # At intermediate levels a parent footprint can straddle the fine
            # coast. Its safety guard only acts there; stage interiors display
            # precisely R_L + intensity*D_L, without intensity-dependent work.
            delta = fields[0]
            if level not in STAGE_LEVELS:
                delta = delta * refinement._smoothstep(0, 12, base)
            delta = delta.clamp(min=-base.abs()*.49/MAX_INTENSITY,
                                max=base.abs()*.49/MAX_INTENSITY)
            height = torch.where(base > 0, base + intensity*delta, base)
            return height.float().contiguous()
    finally:
        _state.check = previous


@torch.inference_mode()
def sample_amplified(world, level, i1, j1, i2, j2, *, intensity=1.0, check=None
                     ) -> tuple[torch.Tensor,torch.Tensor]:
    """mse-v5 heights plus shared final-height diagnostics.

    Preserve the experimental mode's intensity-independent surface contract:
    classify its canonical intensity-one heights. Hybrid callers can instead
    pass their exact displayed-height provider to sample_surface directly.
    """
    intensity = parse_intensity(intensity)
    height = _amplified_height(world,level,i1,j1,i2,j2,intensity=intensity,check=check)
    if level<2 or intensity==0:
        return height,torch.zeros((4,*height.shape),device=world.device,dtype=torch.float32)
    height_fn = lambda a,b,c,d: _amplified_height(world,level,a,b,c,d,intensity=1.,check=check)
    surface = sample_surface(world,level,i1,j1,i2,j2,height_fn,check=check)
    # Match the displayed coast even where the native diagnostic spline's
    # zero contour differs slightly from refinement's sub-native coast bend.
    return height,surface*(height>0)

"""Continuous sub-30 m ravines from the equations of Grenier et al. (2024).

Grenier, Guerin, Galin and Sauvage, ``Real-time Terrain Enhancement with
Controlled Procedural Patterns'', Computer Graphics Forum 43(1), e14992,
doi:10.1111/cgf.14992 (paper CC BY-NC). Independent implementation of
equations 6--16; no author source code is used.

Gabor directions are evaluated at impulses, not at the query: rotating a
query-local carrier breaks long waves and requires derivatives of theta.
Fixed impulse carriers instead give the analytic Jacobian of eq. 14. A C2
compact taper between 2 and 3 sigma replaces the infinite Gaussian tail;
this makes finite-support evaluation continuous even across hash cells.
Sigma is one nominal wavelength, with one signed, jittered impulse per cell.
Carriers have +/-log(1.25) frequency jitter and +/-18 degree orientation
jitter, frozen at impulses; signed cubic-tail weights break regular spacing.
The cascade signal stays nonnegative (eq. 6). The paper's .2*parent amplitude
is further gated by C2 hashed patches, parent head/floor masks and level shares
(1, .7, .3), so it vanishes at parent bottoms AND starts below parent crests.
Height displacement instead uses
a*(H-profile_mean). Cascade directions use derivatives of that displacement.

Wavelengths are 60, 20, 20/3 m. They fade in over LODs 0--1, 2--3, 4--5,
respectively, and only above four samples/wavelength. First-level amplitude
is 0--.20*wavelength (0--12 m), capped by .7 of local relief and twice the
native high-pass RMS; subsequent maxima are 1.68 and .101 m before head gates.
Epsilons .06/.10/.16 and cosine-profile shares .20/.45/.75 soften the
near-Nyquist octaves. These profiles are fixed when a level first resolves:
zooming never retunes its parent. Crest fBm (.20 of amplitude) and small
isotropic rock octaves (8/4/2 m wavelengths, .22/.10/.05 m maxima) carry
fine texture, each faded out while unresolved. Intensity 2 maps to 4/3 depth;
larger strengths asymptote to 1.5 rather than doubling flutes.
The default base is the refinement C2 coastal spline plus decoder octave;
KEEP_PROCEDURAL_BASE opts into the original procedural octaves as well.
Intensity zero and level zero always return sample_refined unaltered.

The first displacement is projected off its 30 m cell averages, measured
on the fixed 7.5 m grid. A compact inverse cell-average filter and a C2
native spline reconstruct the component to subtract. This is a globally
anchored linear operation, never a tile mean or an arrival-dependent cache.
It preserves the native relief without discontinuous per-cell corrections.

The native amplitude proxy uses smoothed slope, local relief, high-pass
roughness and a concavity/channel-distance proxy, all with fixed support.
It suppresses near-affine/glacier-like smooth surfaces, flats and coasts,
but does not identify ice. USE_NATIVE_DRAINAGE optionally reads the public
30 m contributing-area field from terrain_amplification: channels exceed
28800 m2, distance is capped at 240 m, and incision fades within 30--180 m.
The proxy remains default after comparison on all three seed-42 sites.
Neither path recovers a global drainage network. As in paper
section 5.4, plateaus/cliffs and sedimentary flood plains remain limitations.

The explicit ``drainage=True`` path is PHASOR_DRAINAGE_VERSION, distinct
from that legacy flag and from the byte-compatible v3 default. It concentrates
amplitude in a 45--240 m corridor around channels above 28800 m2, grows heads
over 28800--86400 m2, and suppresses points 15--60 m upstream of a head.
Smoothed native downhill routing, biased toward increasing area, supplies the
channel direction. Bank gradients blend toward that tangent; first-level
carrier phases place bottoms at the nearby channel instead of random crests.
Slope, roughness, ridge/floor and patch gates remain; the same conservative
projection removes native-cell displacement means. The mode is still a local
finite-horizon procedural enhancement, not a new global drainage simulation.
Small isotropic bedrock texture can remain on rough steep interfluves; flats
and smooth plateaus do not receive that texture.

Only this new path samples native heights through globally anchored 128-cell
square reads. This isolates an upstream rectangular-read resize ambiguity
that otherwise moves heights when a material halo changes the read shape.
Its bounded 4 MiB per-world native cache affects cost, never numeric results;
CUDA events make cached data safe across streams. The unchanged legacy off
read can consequently differ slightly from this coherent native C2 base.
Conservation diagnostics explicitly use the coherent base, and report the
legacy-frame difference separately. Intensity zero still returns the exact
original sample_refined result, including its original native read policy.
"""
from __future__ import annotations

from contextlib import nullcontext
from collections import OrderedDict
import math
from threading import Lock

import numpy as np
import torch

import terrain_refinement as refinement

VERSION = 'phasor-v3'
PHASOR_DRAINAGE_VERSION = 'phasor-drainage-v1'
SURFACE_CHANNELS = ('flow', 'deposit', 'scree', 'rock')
KEEP_PROCEDURAL_BASE = False
WAVELENGTHS = (60., 20., 20/3)
PROJECT_NATIVE_MEAN = True
USE_NATIVE_DRAINAGE = False
CASCADE_GAIN = .2
PROFILE_EPSILON = .01
RADIAL_C = 8.
FREQUENCY_JITTER = math.log(1.25)
ORIENTATION_JITTER = math.radians(18)
GULLY_SHARE = (1., .7, .3)
PROFILE_EPSILONS = (.06, .10, .16)
PROFILE_ROUNDING = (.20, .45, .75)
DRAINAGE_THRESHOLD_M2 = 32*900.
DRAINAGE_DISTANCE_CELLS = 8
_ROOT_EPS = math.sqrt(PROFILE_EPSILON)
_PROFILE_K = 1 + 2*_ROOT_EPS
# Exact integral of eq. 6 over [-pi, pi], divided by 2*pi.
_PROFILE_MEAN = (.5*(1 + _ROOT_EPS) + PROFILE_EPSILON /
                 (2*math.sqrt(_PROFILE_K))*math.asinh(math.sqrt(_PROFILE_K / PROFILE_EPSILON))
                 - _ROOT_EPS)
_KERNELS = {}

# Box average of a cubic B-spline over the four level-2 sample centres.
# Its symmetric five-tap stencil has a stable inverse. Truncate at twelve
# cells, like refinement's spline prefilter; normalize its DC gain exactly.
PROJECTION_RADIUS = 12
_BOX_TAPS = []
for _offset in range(-2, 3):
    _distances = np.abs(np.array([-.375, -.125, .125, .375])-_offset)
    _BOX_TAPS.append(float(np.where(_distances < 1, (4-6*_distances**2+3*_distances**3)/6,
                                   np.where(_distances < 2, (2-_distances)**3/6, 0)).mean()))
_omega = 2*math.pi*np.arange(8192)/8192
_inverse = np.fft.ifft(1/sum(tap*np.cos(offset*_omega)
                           for offset, tap in zip(range(-2, 3), _BOX_TAPS))).real
_PROJECT_INVERSE = [float(_inverse[offset % len(_inverse)])
                    for offset in range(-PROJECTION_RADIUS, PROJECTION_RADIUS+1)]
_PROJECT_INVERSE = [tap/math.fsum(_PROJECT_INVERSE) for tap in _PROJECT_INVERSE]


def _profile_mean(epsilon=PROFILE_EPSILON, rounding=0.):
    root = math.sqrt(epsilon)
    k = 1+2*root
    triangle = .5*(1+root)+epsilon/(2*math.sqrt(k))*math.asinh(math.sqrt(k/epsilon))-root
    return (1-rounding)*triangle+rounding*.5


def _profile(phase, epsilon=PROFILE_EPSILON, rounding=0.):
    k, root_eps = 1+2*math.sqrt(epsilon), math.sqrt(epsilon)
    root = (k*(phase/math.pi)**2+epsilon).sqrt()
    value = (1-rounding)*(root-root_eps)+rounding*(1-phase.cos())*.5
    derivative = (1-rounding)*k*phase/(math.pi**2*root)+rounding*phase.sin()*.5
    return value, derivative


def _transfer(real, imag, epsilon=PROFILE_EPSILON, rounding=0.):
    """H and its analytic derivatives in the complex plane (eqs. 6--8)."""
    radius2 = real*real + imag*imag
    radius = radius2.sqrt()
    phase = torch.atan2(imag, real)
    profile, dp = _profile(phase, epsilon, rounding)
    mean = _profile_mean(epsilon, rounding)
    rho = (2/math.pi)*torch.atan(RADIAL_C*radius)
    drho = (2/math.pi)*RADIAL_C/(1 + RADIAL_C**2*radius2)
    safe_r = radius.clamp_min(1e-30)
    radial = drho*(profile - mean)/safe_r
    angular = rho*dp/radius2.clamp_min(1e-60)
    return (rho*profile + (1-rho)*mean,
            radial*real - angular*imag, radial*imag + angular*real, phase, rho)


def _derivative_taps(index, t):
    return [(index-1, -.5*(1-t)**2), (index, 1.5*t*t-2*t),
            (index+1, -1.5*t*t+t+.5), (index+2, .5*t*t)]


def _step_gradient(low, high, value):
    t = ((value-low)/(high-low)).clamp(0, 1)
    return t*t*(3-2*t), 6*t*(1-t)/(high-low)


def _noise_gradient(seed, y, x, spacing, stream):
    """C2 value noise and its analytic physical gradient, in fixed hash order."""
    yy, xx = y/spacing, x/spacing
    iy, ix = yy.floor(), xx.floor()
    fy, fx = yy-iy, xx-ix
    sy = fy*fy*fy*(fy*(fy*6-15)+10)
    sx = fx*fx*fx*(fx*(fx*6-15)+10)
    dy, dx = 30*fy*fy*(1-fy)*(1-fy)/spacing, 30*fx*fx*(1-fx)*(1-fx)/spacing
    corner = lambda r, c: 2*refinement._uniform(refinement._hash(
        refinement._hash(seed ^ ((iy.long()+r) & refinement._MASK)) ^
        ((ix.long()+c) & refinement._MASK)), stream)-1
    v00, v01, v10, v11 = corner(0, 0), corner(0, 1), corner(1, 0), corner(1, 1)
    top, bottom = v00+(v01-v00)*sx, v10+(v11-v10)*sx
    return top+(bottom-top)*sy, (bottom-top)*dy, ((v01-v00)*(1-sy)+(v11-v10)*sy)*dx


def _patch(seed, index, y, x):
    a = _noise_gradient(seed+811, y, x, 3*WAVELENGTHS[index], 101+index)
    b = _noise_gradient(seed+811, y, x, 1.3*WAVELENGTHS[index], 111+index)
    value, dy, dx = [.75*aa+.25*bb for aa, bb in zip(a, b)]
    patch, derivative = _step_gradient(-.20, .38, value)
    return patch, derivative*dy, derivative*dx


def _head_gate(parent):
    """Start below parent crests and taper into rounded parent valley bottoms."""
    amp = parent[5].clamp_min(1e-20)
    h = parent[0]/amp
    dy = (parent[1]-h*parent[6])/amp
    dx = (parent[2]-h*parent[7])/amp
    bottom, db = _step_gradient(.06, .22, h)
    crest, dc = _step_gradient(.52, .88, h)
    gate = bottom*(1-crest)
    derivative = db*(1-crest)-bottom*dc
    return gate, derivative*dy, derivative*dx


def _drainage_distance(world, n0, m0, n1, m1):
    """Capped distance to contributing-area channels; fixed native halo, exact seams."""
    from terrain_amplification import sample_drainage
    from scipy.ndimage import distance_transform_edt
    r = DRAINAGE_DISTANCE_CELLS
    area = sample_drainage(world, n0-r, m0-r, n1+r, m1+r)
    channels = area.detach().cpu().numpy() >= DRAINAGE_THRESHOLD_M2
    distances = (distance_transform_edt(~channels).clip(0, r) if channels.any()
                 else np.full(channels.shape, r, dtype=np.float64))
    return torch.tensor(distances[r:-r, r:-r]*30, device=world.device, dtype=torch.float64)


def _drainage_controls(world, n0, m0, n1, m1, check=None, *, native=None):
    """Channel corridor, downhill direction and offset, on the native grid.

    Uses only the documented m2 reader. Routing prefers downhill neighbours
    with increasing area, then averages vectors on channel cells. A capped
    nearest-channel gather extends that direction into its banks. The 11-cell
    read halo is 8 distance cells + 2 vector-smoothing cells + 1 routing cell;
    an outside channel cannot affect the capped interior. No request-wide
    normalization or unbounded distance transform enters the result.
    """
    from terrain_amplification import sample_drainage
    from scipy.ndimage import distance_transform_edt
    r, smooth_radius = DRAINAGE_DISTANCE_CELLS, 2
    halo = r+smooth_radius+1
    bounds = n0-halo, m0-halo, n1+halo, m1+halo
    area = sample_drainage(world, *bounds).double()
    height = (refinement.sample_refined(world,0,*bounds,check=check).double()
              if native is None else native(*bounds))
    if check is not None:
        check()
    center, local = height[1:-1, 1:-1], area[1:-1, 1:-1]
    fy, fx = torch.zeros_like(center), torch.zeros_like(center)
    for dy, dx in ((-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)):
        distance = math.hypot(dy, dx)
        neighbour = height[1+dy:height.shape[0]-1+dy, 1+dx:height.shape[1]-1+dx]
        downstream = area[1+dy:area.shape[0]-1+dy, 1+dx:area.shape[1]-1+dx]
        drop = ((center-neighbour)/(30*distance)).clamp_min(0)
        preference = .2+.8*refinement._smoothstep(.7, 1.2, downstream/(local+1))
        score = (drop*preference).square()
        fy, fx = fy+score*(dy/distance), fx+score*(dx/distance)
    length = (fy*fy+fx*fx).sqrt().clamp_min(1e-20)
    weight = refinement._smoothstep(DRAINAGE_THRESHOLD_M2/2, 2*DRAINAGE_THRESHOLD_M2, local)
    taps = [1/16, 4/16, 6/16, 4/16, 1/16]
    fy = refinement._filter(fy/length*weight, taps)
    fx = refinement._filter(fx/length*weight, taps)
    # All arrays now have exactly the 8-cell nearest-channel halo.
    crop = smooth_radius+1
    native_area = area[crop:-crop, crop:-crop].detach().cpu().numpy()
    channels = native_area >= DRAINAGE_THRESHOLD_M2
    shape = n1-n0, m1-m0
    if not channels.any():
        zeros = torch.zeros(shape, device=world.device, dtype=torch.float64)
        return (zeros,)*6
    distances, indices = distance_transform_edt(~channels, return_indices=True)
    rows, cols = indices
    flow_y, flow_x = fy.detach().cpu().numpy()[rows, cols], fx.detach().cpu().numpy()[rows, cols]
    norm = np.sqrt(flow_y*flow_y+flow_x*flow_x)
    flow_y, flow_x = flow_y/np.maximum(norm, 1e-20), flow_x/np.maximum(norm, 1e-20)
    yy, xx = np.indices(channels.shape)
    offset_y, offset_x = (rows-yy)*30., (cols-xx)*30.
    selected = (distances <= r) & (norm > 1e-12)
    convert = lambda value: torch.tensor(np.where(selected, value, 0)[r:-r, r:-r].copy(),
                                         device=world.device, dtype=torch.float64)
    dist, nearest_area = convert(np.minimum(distances, r)*30), convert(native_area[rows, cols])
    fy, fx, oy, ox, valid = [convert(value) for value in (flow_y, flow_x, offset_y, offset_x, selected)]
    near = (1-refinement._smoothstep(45, 240, dist))*valid
    head_area = refinement._smoothstep(DRAINAGE_THRESHOLD_M2, 3*DRAINAGE_THRESHOLD_M2, nearest_area)
    # Points upstream of a channel's head do not inherit full incision merely
    # because its nearest contributing cell is a short distance downhill.
    along = -oy*fy-ox*fx
    head_position = refinement._smoothstep(-60, -15, along)
    gain = near*head_area*head_position
    if check is not None:
        check()
    return gain, fy, fx, oy, ox, near


def _intensity_gain(intensity):
    # C1 at one, asymptote 1.5; 2.0 maps to 4/3 rather than twice the depth.
    return intensity if intensity <= 1 else 1+.5*(1-1/(1+2*(intensity-1)))


class _CanonicalNative:
    """Globally anchored native reads for the drainage path, with a bounded LRU.

    Upstream _compute_elev's short-side resize is shape-dependent for some
    rectangular reads. Fixed square 128-cell reads (origins divisible by the
    decoder scale) avoid that ambiguity. Every native pixel has exactly one
    source read; copies and crops perform no floating arithmetic. This does
    not change the legacy v3 path, neural windows, or the world object.
    """
    def __init__(self, world, check=None):
        self.world, self.check = world, check
        self.tiles,self.lock = world.__dict__.setdefault('_terrain_phasor_canonical_native',(OrderedDict(),Lock()))
        get = world.get
        self.identity = (PHASOR_DRAINAGE_VERSION,refinement.VERSION,int(getattr(world,'seed',0)),
            str(world.device),
            float(world.kwargs.get('residual_std',0)),float(world.kwargs.get('residual_mean',0)),
            id(getattr(get,'__func__',get)),id(getattr(world,'residual',None)),id(getattr(world,'latents',None)))

    def __call__(self, a, b, c, d):
        size = 128
        if self.check is not None:
            self.check()
        out = torch.empty((c-a,d-b),device=self.world.device,dtype=torch.float64)
        for y in range(a//size*size,c,size):
            for x in range(b//size*size,d,size):
                key = self.identity,y,x
                with self.lock:
                    entry = self.tiles.get(key)
                    if entry is not None:
                        self.tiles.move_to_end(key)
                if entry is None:
                    tile = refinement.sample_refined(self.world,0,y,x,y+size,x+size,check=self.check).double()
                    ready = None
                    if tile.is_cuda:
                        ready = torch.cuda.Event()
                        ready.record(torch.cuda.current_stream(tile.device))
                    entry = tile,ready
                    with self.lock:
                        self.tiles[key] = entry
                        while len(self.tiles)>32:  # <=4 MiB of FP64 native data per world
                            self.tiles.popitem(last=False)
                tile,ready = entry
                if tile.is_cuda:
                    stream = torch.cuda.current_stream(tile.device)
                    stream.wait_event(ready)
                    tile.record_stream(stream)
                ya,xa,yb,xb = max(a,y),max(b,x),min(c,y+size),min(d,x+size)
                out[ya-a:yb-a,xa-b:xb-b] = tile[ya-y:yb-y,xa-x:xb-x]
        return out


class _NativeWorldView:
    """Read-only native adapter; existing neural/drainage caches stay shared."""
    __slots__ = ('source','native')

    def __init__(self, world, check=None):
        self.source, self.native = world, _CanonicalNative(world,check)

    @property
    def __dict__(self):
        return self.source.__dict__

    def __getattr__(self, name):
        return getattr(self.source,name)

    def get(self, i1, j1, i2, j2, *, with_climate=True):
        if with_climate:
            return self.source.get(i1,j1,i2,j2,with_climate=True)
        return {'elev':self.native(i1,j1,i2,j2).float()}


def _canonical_reader(world, check=None):
    return world.native if isinstance(world,_NativeWorldView) else _CanonicalNative(world,check)


def _drainage_world_view(world, level, i1, j1, check=None):
    if isinstance(world,_NativeWorldView):
        return world
    from terrain_amplification import sample_drainage
    scale = 2**level
    # Bind A's cached native tensor to the original source before any caller
    # asks for native surface diagnostics through the coherent read view.
    sample_drainage(world,i1//scale,j1//scale,i1//scale+1,j1//scale+1)
    if check is not None:
        check()
    return _NativeWorldView(world,check)


class _Controls:
    """Fixed native stencils followed by C2 gathers, independent of read size."""

    def __init__(self, world, bounds, check, *, base_bounds=None, drainage=False):
        y0, x0, y1, x1 = bounds
        self.y0, self.x0 = math.floor(y0/30)-3, math.floor(x0/30)-3
        end_y, end_x = math.ceil(y1/30)+4, math.ceil(x1/30)+4
        r = refinement.SMOOTH_RADIUS
        lo, left, hi, right = self.y0-r-1, self.x0-r-1, end_y+r+1, end_x+r+1
        native_bounds = lo-r, left-r, hi+r, right+r
        if base_bounds is not None:
            native_bounds = (min(native_bounds[0], base_bounds[0]), min(native_bounds[1], base_bounds[1]),
                             max(native_bounds[2], base_bounds[2]), max(native_bounds[3], base_bounds[3]))
        self.native = (_canonical_reader(world,check) if drainage else
                       refinement._NativeWindow(world,*native_bounds,check))
        raw = self.native(lo, left, hi, right)
        smooth = refinement._filter(raw, refinement._SMOOTH)
        gy = (smooth[2:, 1:-1]-smooth[:-2, 1:-1])/60
        gx = (smooth[1:-1, 2:]-smooth[1:-1, :-2])/60
        center = smooth[1:-1, 1:-1]
        lap = (smooth[2:, 1:-1]+smooth[:-2, 1:-1]+smooth[1:-1, 2:]+
               smooth[1:-1, :-2]-4*center)/900
        # Smooth squared residuals with a second stencil: use a separate,
        # identically anchored read, rather than any tile-wide statistic.
        raw2 = self.native(lo-r, left-r, hi+r, right+r)
        smooth2 = refinement._filter(raw2, refinement._SMOOTH)
        residual2 = raw2[r:-r, r:-r]-smooth2
        rms = refinement._filter(residual2*residual2, refinement._SMOOTH)[1:-1, 1:-1].sqrt()
        slope = (gy*gy+gx*gx).sqrt()
        relief = (rms*rms + (90*slope)**2).sqrt()
        # Positive curvature marks a valley/channel. Its inverse is a cheap
        # local proxy for distance from channels, not accumulated drainage.
        concavity = (90*lap/(slope+.03)).clamp_min(0)
        channel_distance = 1/(1+4*concavity*concavity)
        self.channel_control = None
        if drainage:
            gain, flow_y, flow_x, oy, ox, near = _drainage_controls(
                world, self.y0, self.x0, end_y, end_x, check, native=self.native)
            channel_distance = gain*(.35+.65*channel_distance)
            # Preserve slope magnitude while rotating the contour-normal
            # field toward the actual channel tangent (sign is immaterial).
            uy, ux = -flow_y, -flow_x
            sign = torch.where(uy*gy+ux*gx < 0, -1., 1.)
            uy, ux = uy*sign, ux*sign
            blend = .8*near*refinement._smoothstep(0, .4, gain)
            by = (1-blend)*gy/(slope+1e-20)+blend*uy
            bx = (1-blend)*gx/(slope+1e-20)+blend*ux
            norm = (by*by+bx*bx).sqrt().clamp_min(1e-20)
            gy, gx = by/norm*slope, bx/norm*slope
            self.channel_control = _Controls.__new__(_Controls)
            self.channel_control.y0, self.channel_control.x0 = self.y0, self.x0
            self.channel_control.fields = torch.stack((oy, ox, near)).contiguous()
        elif USE_NATIVE_DRAINAGE:
            distance = _drainage_distance(world, self.y0, self.x0, end_y, end_x)
            channel_distance = refinement._smoothstep(30, 180, distance)
        # Native convex ridges are gully heads, rather than fully incised
        # flutes running all the way to a divide. Concave floors fade above.
        ridge = refinement._smoothstep(.08, .5, -90*lap/(slope+.03))
        channel_distance = channel_distance*(1-.9*ridge)
        amplitude = torch.minimum(torch.minimum(.20*WAVELENGTHS[0]*slope/(slope+.25), .7*relief), 2*rms)
        amplitude = (amplitude * refinement._smoothstep(.06, .28, slope) *
                     refinement._smoothstep(.03, .6, rms) * channel_distance *
                     refinement._smoothstep(4, 35, center))
        self.fields = torch.stack((amplitude, gy, gx)).contiguous()
        self.rock_control = None
        if drainage:
            # Keep small isotropic bedrock texture on rough steep interfluves,
            # without creating off-channel gullies or roughening plateaus.
            rock = (refinement._smoothstep(.35,.9,slope)*
                    refinement._smoothstep(.3,2.,rms)*refinement._smoothstep(4,35,center))
            self.rock_control = _Controls.__new__(_Controls)
            self.rock_control.y0,self.rock_control.x0 = self.y0,self.x0
            self.rock_control.fields = torch.stack((rock,torch.zeros_like(rock),torch.zeros_like(rock))).contiguous()

    def sample(self, y, x):
        kernel = _kernel(y.device, 'controls')
        if kernel is not None:
            y, x = torch.broadcast_tensors(y, x)
            out = torch.empty((5, *y.shape), device=y.device, dtype=torch.float64)
            refinement._launch(kernel, y.shape, y.contiguous(), x.contiguous(), self.fields,
                               np.int32(y.shape[0]), np.int32(y.shape[1]),
                               np.int32(self.fields.shape[1]), np.int32(self.fields.shape[2]),
                               float(self.y0), float(self.x0), out)
            return tuple(out)
        # Physical cell centres match refinement._axis, including negatives.
        yy, xx = y/30-.5-self.y0, x/30-.5-self.x0
        iy, ix = yy.floor().long(), xx.floor().long()
        fy, fx = yy-iy, xx-ix
        ty, tx = refinement._cubic_taps(iy, fy), refinement._cubic_taps(ix, fx)
        dy, dx = _derivative_taps(iy, fy), _derivative_taps(ix, fx)

        def gather(field, rows, cols):
            total = None
            for row, wy in rows:
                part = None
                for col, wx in cols:
                    term = field[row, col]*wx
                    part = term if part is None else part+term
                term = part*wy
                total = term if total is None else total+term
            return total

        a, gy, gx = [gather(f, ty, tx) for f in self.fields]
        return a, gather(self.fields[0], dy, tx)/30, gather(self.fields[0], ty, dx)/30, gy, gx


_GABOR_SOURCE = r'''
extern "C" __global__ void gabor(
 const double* y,const double* x,const double* params,
 const long long* iy,const long long* ix,int h,int w,int ph,int pw,int planes,
 double sigma,double* output) {
 int col=blockIdx.x*blockDim.x+threadIdx.x, row=blockIdx.y*blockDim.y+threadIdx.y;
 if(row>=h||col>=w)return;
 int p=row*w+col, n=h*w, pn=ph*pw;
 double re=0,im=0,ry=0,rx=0,jy=0,jx=0;
 for(int dr=-3;dr<=3;dr++)for(int dc=-3;dc<=3;dc++) {
  int k=(iy[p]+dr)*pw+ix[p]+dc;
  double dy=(y[p]-params[k])/sigma, dx=(x[p]-params[pn+k])/sigma;
  double t=dy*dy+dx*dx;
  double u=fmin(1.,fmax(0.,(t-4.)/5.));
  double taper=1.-u*u*u*(10.+u*(-15.+6.*u));
  double dt=-6.*u*u*(1.-u)*(1.-u);
  double env=params[2*pn+k]*exp(-.5*t);
  double angle=params[3*pn+k]*dy+params[4*pn+k]*dx;
  if(planes>5)angle+=params[5*pn+k];
  double c=cos(angle);
  double s=sin(angle);
  double v=env*taper, ay=env*(2.*dt-taper)*dy/sigma, ax=env*(2.*dt-taper)*dx/sigma;
  double ky=params[3*pn+k]/sigma, kx=params[4*pn+k]/sigma;
  re+=v*c; im+=v*s;
  ry+=ay*c-v*s*ky; rx+=ax*c-v*s*kx;
  jy+=ay*s+v*c*ky; jx+=ax*s+v*c*kx;
 }
 output[p]=re;output[n+p]=im;output[2*n+p]=ry;
 output[3*n+p]=rx;output[4*n+p]=jy;output[5*n+p]=jx;
}
'''

_CONTROLS_SOURCE = r'''
extern "C" __global__ void controls(const double* y,const double* x,const double* f,
 int h,int w,int fh,int fw,double y0,double x0,double* out) {
 int col=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y*blockDim.y+threadIdx.y;
 if(row>=h||col>=w)return;
 int p=row*w+col,n=h*w,fn=fh*fw;
 double yy=y[p]/30.-.5-y0,xx=x[p]/30.-.5-x0;
 int iy=(int)floor(yy),ix=(int)floor(xx);
 double ty=yy-iy,tx=xx-ix,wy[4],wx[4],dy[4],dx[4];
 double t[2]={ty,tx};
 for(int d=0;d<2;d++) {
  double v=t[d],v2=v*v,v3=v2*v;
  double* a=d==0?wy:wx; double* b=d==0?dy:dx;
  a[0]=pow(1.-v,3.)/6.;a[1]=(3.*v3-6.*v2+4.)/6.;
  a[2]=(-3.*v3+3.*v2+3.*v+1.)/6.;a[3]=v3/6.;
  b[0]=-.5*pow(1.-v,2.);b[1]=1.5*v*v-2.*v;
  b[2]=-1.5*v*v+v+.5;b[3]=.5*v*v;
 }
 for(int k=0;k<5;k++) {
  int channel=k<3?0:k-2; double sum=0;
  for(int r=0;r<4;r++) {
   double part=0;
   for(int c=0;c<4;c++)part+=f[channel*fn+(iy-1+r)*fw+ix-1+c]*(k==2?dx[c]:wx[c]);
   sum+=part*(k==1?dy[r]:wy[r]);
  }
  out[k*n+p]=k==1||k==2?sum/30.:sum;
 }
}
'''

_FINISH_SOURCE = r'''
__device__ unsigned long long hash32(unsigned long long v) {
 v&=0xffffffffULL;v^=v>>16;v=(v*0x7feb352dULL)&0xffffffffULL;
 v^=v>>15;v=(v*0x1b873593ULL)&0xffffffffULL;return v^(v>>16);
}
__device__ double corner(long long y,long long x,unsigned long long seed,int stream) {
 unsigned long long h=hash32(hash32(seed^((unsigned long long)y&0xffffffffULL))^((unsigned long long)x&0xffffffffULL));
 return 2.*(double)(hash32(h+stream)&0xffffffULL)/16777216.-1.;
}
__device__ double noise(double y,double x,double spacing,unsigned long long seed,int stream) {
 double gy=y/spacing,gx=x/spacing,iy=floor(gy),ix=floor(gx),fy=gy-iy,fx=gx-ix;
 double sy=fy*fy*(3.-2.*fy),sx=fx*fx*(3.-2.*fx);
 double v00=corner((long long)iy,(long long)ix,seed,stream),v01=corner((long long)iy,(long long)ix+1,seed,stream);
 double v10=corner((long long)iy+1,(long long)ix,seed,stream),v11=corner((long long)iy+1,(long long)ix+1,seed,stream);
 double top=v00+(v01-v00)*sx,bottom=v10+(v11-v10)*sx;return top+(bottom-top)*sy;
}
__device__ double step(double lo,double hi,double x) {
 double t=fmin(1.,fmax(0.,(x-lo)/(hi-lo)));return t*t*(3.-2.*t);
}
extern "C" __global__ void finish(const float* base,const double* patterns,const double* crests,const double* gy,const double* gx,
 const double* y,const double* x,int h,int w,unsigned long long seed,double intensity,
 const double* correction,const double* rocks,const double* native_rock,int drainage,
 double w0,double w1,double w2,float* height,float* surface) {
 int col=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y*blockDim.y+threadIdx.y;
 if(row>=h||col>=w)return;
 int p=row*w+col,n=h*w;
 double delta=0.,v[4]={0.,0.,0.,0.},weights[3]={w0,w1,w2},waves[3]={60.,20.,20./3.};
 double steep=step(.15,.65,sqrt(gy[p]*gy[p]+gx[p]*gx[p]));
 for(int k=0;k<3;k++) {
  if(weights[k]==0.)continue;
  double r=patterns[(k*4)*n+p],phase=patterns[(k*4+1)*n+p];
  double rho=patterns[(k*4+2)*n+p],amp=patterns[(k*4+3)*n+p];
  double crest=step(3.141592653589793/4.,3.*3.141592653589793/4.,fabs(phase));
  double fbm=0.,spacing[3]={6.,12.,24.};
  for(int j=0;j<3;j++)fbm+=crests[k*3+j]*noise(y[p]/30.,x[p]/30.,waves[k]/30./spacing[j],seed,71+j);
  delta+=weights[k]*(r-(k==0?correction[p]:0.)+.20*amp*crest*rho*fbm);
  double strength=fmin(1.,fmax(0.,amp/(.12*waves[k])))*rho*weights[k];
  double bottom=1.-step(.12,.85,fabs(phase)),flank=step(.25,.9,fabs(phase))*(1.-crest);
  double signals[4]={bottom,bottom*(1.-steep),steep*flank,steep*crest};
  for(int j=0;j<4;j++)v[j]=fmax(v[j],signals[j]*strength);
 }
 double rocky=0.,rock_wave[3]={8.,4.,2.};
 for(int k=0;k<3;k++)rocky+=rocks[k]*noise(y[p],x[p],rock_wave[k]/2.,seed+1709,131+k);
 double rock_amount=v[3]*fmin(1.,patterns[3*n+p]/4.);
 if(drainage) {
  double patch=step(-.1,.4,noise(y[p],x[p],120.,seed+2641,117));
  rock_amount=fmax(rock_amount,native_rock[p]*patch);
  v[3]=fmax(v[3],native_rock[p]*patch);
 }
 if(drainage)delta+=rocky*rock_amount;
 else delta+=rocky*v[3]*fmin(1.,patterns[3*n+p]/4.);
 double b=base[p],coast=step(4.,35.,b),change=intensity*delta*coast;
 height[p]=(float)(b+change*fabs(b)/fmax(sqrt(b*b+4.*change*change),2.2250738585072014e-308));
 for(int j=0;j<4;j++)surface[j*n+p]=(float)fmin(1.,fmax(0.,v[j]*coast));
}
'''

_FIELD_SOURCE = _FINISH_SOURCE.split('extern "C"')[0]+r'''
__device__ void ngrad(double y,double x,double spacing,unsigned long long seed,int stream,double* out) {
 double yy=y/spacing,xx=x/spacing,iy=floor(yy),ix=floor(xx),fy=yy-iy,fx=xx-ix;
 double sy=fy*fy*fy*(fy*(fy*6.-15.)+10.),sx=fx*fx*fx*(fx*(fx*6.-15.)+10.);
 double dy=30.*fy*fy*(1.-fy)*(1.-fy)/spacing,dx=30.*fx*fx*(1.-fx)*(1.-fx)/spacing;
 double v00=corner((long long)iy,(long long)ix,seed,stream),v01=corner((long long)iy,(long long)ix+1,seed,stream);
 double v10=corner((long long)iy+1,(long long)ix,seed,stream),v11=corner((long long)iy+1,(long long)ix+1,seed,stream);
 double top=v00+(v01-v00)*sx,bottom=v10+(v11-v10)*sx;
 out[0]=top+(bottom-top)*sy;out[1]=(bottom-top)*dy;out[2]=((v01-v00)*(1.-sy)+(v11-v10)*sy)*dx;
}
__device__ double dstep(double lo,double hi,double value) {
 double t=fmin(1.,fmax(0.,(value-lo)/(hi-lo)));return 6.*t*(1.-t)/(hi-lo);
}
extern "C" __global__ void field(const double* g,const double* amps,const double* parent,
 const double* y,const double* x,int h,int w,int index,unsigned long long seed,
 double wave,double share,double root_eps,double epsilon,double k,double rounding,double mean,double* out) {
 int col=blockIdx.x*blockDim.x+threadIdx.x,row=blockIdx.y*blockDim.y+threadIdx.y;
 if(row>=h||col>=w)return;
 int p=row*w+col,n=h*w;
 double a[3],b[3];ngrad(y[p],x[p],3.*wave,seed,101+index,a);ngrad(y[p],x[p],1.3*wave,seed,111+index,b);
 double value=.75*a[0]+.25*b[0],ny=.75*a[1]+.25*b[1],nx=.75*a[2]+.25*b[2];
 double patch=step(-.20,.38,value),py=dstep(-.20,.38,value)*ny,px=dstep(-.20,.38,value)*nx;
 if(index) {
  double pa=fmax(parent[3*n+p],1e-20),q=parent[p]/pa;
  double qy=(parent[n+p]-q*parent[4*n+p])/pa,qx=(parent[2*n+p]-q*parent[5*n+p])/pa;
  double bottom=step(.06,.22,q),crest=step(.52,.88,q),gate=bottom*(1.-crest);
  double derivative=dstep(.06,.22,q)*(1.-crest)-bottom*dstep(.52,.88,q);
  py=py*gate+patch*(derivative*qy);px=px*gate+patch*(derivative*qx);patch*=gate;
 }
 patch*=share;py*=share;px*=share;
 double amp=amps[p]*patch,ay=amps[n+p]*patch+amps[p]*py,ax=amps[2*n+p]*patch+amps[p]*px;
 double re=g[p],im=g[n+p],r2=re*re+im*im,r=sqrt(r2),phase=atan2(im,re),pi=3.141592653589793;
 double t=phase/pi,root=sqrt(k*(t*t)+epsilon);
 double profile=(1.-rounding)*(root-root_eps)+rounding*(1.-cos(phase))*.5;
 double dp=(1.-rounding)*k*phase/(pi*pi*root)+rounding*sin(phase)*.5;
 double rho=(2./pi)*atan(8.*r),drho=(2./pi)*8./(1.+64.*r2);
 double radial=drho*(profile-mean)/fmax(r,1e-30),angular=rho*dp/fmax(r2,1e-60);
 double hr=radial*re-angular*im,hi=radial*im+angular*re,v=rho*profile+(1.-rho)*mean;
 out[p]=amp*v;out[n+p]=ay*v+amp*(hr*g[2*n+p]+hi*g[4*n+p]);
 out[2*n+p]=ax*v+amp*(hr*g[3*n+p]+hi*g[5*n+p]);
 out[3*n+p]=phase;out[4*n+p]=rho;out[5*n+p]=amp;out[6*n+p]=ay;out[7*n+p]=ax;
}
'''


def _kernel(device, name='gabor'):
    """Optional NVRTC fusion; PyTorch is the complete supported fallback."""
    device = torch.device(device)
    if device.type != 'cuda':
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    key = (index, name)
    if key not in _KERNELS:
        try:
            import cupy as cp
            with cp.cuda.Device(index):
                source = {'gabor': _GABOR_SOURCE, 'controls': _CONTROLS_SOURCE,
                          'finish': _FINISH_SOURCE, 'field': _FIELD_SOURCE}[name]
                kernel = cp.RawKernel(source, name, options=('--fmad=false',))
                kernel.compile()
            _KERNELS[key] = kernel
        except Exception:
            _KERNELS[key] = None
    return _KERNELS[key]


def _gabor(y, x, sigma, params, origin):
    """Complex Gaussian field and physical Jacobian; all sums in fixed order."""
    y, x = torch.broadcast_tensors(y, x)
    iy = (y/sigma).floor().long()-origin[0]
    ix = (x/sigma).floor().long()-origin[1]
    kernel = _kernel(y.device)
    if kernel is not None:
        out = torch.empty((6, *y.shape), dtype=torch.float64, device=y.device)
        refinement._launch(kernel, y.shape, y.contiguous(), x.contiguous(), params,
                           iy.contiguous(), ix.contiguous(), np.int32(y.shape[0]), np.int32(y.shape[1]),
                           np.int32(params.shape[1]), np.int32(params.shape[2]), np.int32(params.shape[0]), sigma, out)
        return tuple(out)
    out = [torch.zeros_like(y) for _ in range(6)]
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            py, px, weight, ky, kx = params[:5, iy+dr, ix+dc]
            dy, dx = (y-py)/sigma, (x-px)/sigma
            t = dy*dy+dx*dx
            u = ((t-4)/5).clamp(0, 1)
            taper = 1-u*u*u*(10+u*(-15+6*u))
            dt = -6*u*u*(1-u)*(1-u)
            env = weight*torch.exp(-.5*t)
            angle = ky*dy+kx*dx
            if params.shape[0] > 5:
                angle = angle+params[5, iy+dr, ix+dc]
            c, s = angle.cos(), angle.sin()
            v = env*taper
            ay, ax = env*(2*dt-taper)*dy/sigma, env*(2*dt-taper)*dx/sigma
            ky, kx = ky/sigma, kx/sigma
            terms = (v*c, v*s, ay*c-v*s*ky, ax*c-v*s*kx,
                     ay*s+v*c*ky, ax*s+v*c*kx)
            out = [a+b for a, b in zip(out, terms)]
    return tuple(out)


class _Patterns:
    def __init__(self, world, controls, check=None, *, bounds=None, last=2, drainage=False):
        self.world, self.controls, self.check = world, controls, check
        self.bounds, self.last = bounds, last
        self.parameters, self.samples = {}, {}
        self.projection = None
        self.first_bounds = None
        self.drainage = drainage

    def _parameters(self, index, y, x):
        if self.bounds is not None and index in self.parameters:
            return self.parameters[index]
        sigma = WAVELENGTHS[index]
        if self.bounds is None:
            y0, x0, y1, x1 = float(y.min()), float(x.min()), float(y.max()), float(x.max())
        else:
            margin = 4*sum(WAVELENGTHS[index+1:self.last+1])
            y0, x0, y1, x1 = self.bounds
            y0, x0, y1, x1 = y0-margin, x0-margin, y1+margin, x1+margin
        if index == 0 and self.first_bounds is not None:
            yy0, xx0, yy1, xx1 = self.first_bounds
            y0, x0, y1, x1 = min(y0, yy0), min(x0, xx0), max(y1, yy1), max(x1, xx1)
        a0, b0 = math.floor(y0/sigma)-3, math.floor(x0/sigma)-3
        a1, b1 = math.floor(y1/sigma)+4, math.floor(x1/sigma)+4
        a = torch.arange(a0, a1, device=y.device, dtype=torch.int64)[:, None]
        b = torch.arange(b0, b1, device=y.device, dtype=torch.int64)[None, :]
        seed = (int(getattr(self.world, 'seed', 0))*1000003 + index*7919 + 3737) & refinement._MASK
        hashed = refinement._hash(refinement._hash(a ^ seed) ^ b)
        py = (a.double()+refinement._uniform(hashed, 1))*sigma
        px = (b.double()+refinement._uniform(hashed, 2))*sigma
        u = 2*refinement._uniform(hashed, 3)-1
        weight = (.3*u+.7*u*u*u)/math.sqrt(math.pi*(.09/3+.42/5+.49/7))
        _, _, _, gy, gx = self.controls.sample(py, px)
        for ancestor in range(index):
            parent = self.sample(ancestor, py, px)
            dy, dx = parent[1]-parent[8]*parent[6], parent[2]-parent[8]*parent[7]
            if ancestor == 0 and self.projection is not None:
                _, cy, cx, _, _ = self.projection.sample(py, px)
                dy, dx = dy-cy, dx-cx
            gy, gx = gy+dy, gx+dx
        slope = (gy*gy+gx*gx).sqrt().clamp_min(1e-30)
        # theta=atan2(gy,gx)+pi/2; frequency is perpendicular to slope.
        ky, kx = gx/slope, -gy/slope
        angle = (2*refinement._uniform(hashed, 4)-1)*ORIENTATION_JITTER
        channel = None
        if self.drainage and index == 0:
            channel = self.controls.channel_control.sample(py, px)
            angle = angle*(1-.6*channel[4])
        frequency = 2*math.pi*torch.exp((2*refinement._uniform(hashed, 5)-1)*FREQUENCY_JITTER)
        # Both jittered carrier components are frozen at the impulse, so the
        # existing Gabor Jacobian differentiates precisely this carrier.
        ky, kx = (ky*angle.cos()+kx*angle.sin())*frequency, (kx*angle.cos()-ky*angle.sin())*frequency
        if channel is not None:
            # Positive impulses with phase zero at the nearby channel place
            # a ravine bottom there. Carrier phase remains impulse-local,
            # therefore the existing analytic Gaussian Jacobian is exact.
            phase = -(ky*channel[0]+kx*channel[3])/sigma
            params = torch.stack((py, px, weight.abs(), ky, kx, phase)).contiguous()
        else:
            params = torch.stack((py, px, weight, ky, kx)).contiguous()
        result = params, (a0, b0)
        if self.bounds is not None:
            self.parameters[index] = result
        return result

    def sample(self, index, y, x):
        """r, analytic gradient, phase, attenuation and amplitude at arbitrary p.

        The full product rule includes grad(a), unlike the paper's locally
        constant amplitude approximation. Orientation is frozen at impulses.
        Only request-local memoization is used; it cannot affect other tiles.
        """
        key = index, id(y), id(x)
        if key in self.samples:
            return self.samples[key][2]
        params, (a0, b0) = self._parameters(index, y, x)
        sigma = WAVELENGTHS[index]
        gaussian = _gabor(y, x, sigma, params, (a0, b0))
        epsilon, rounding = PROFILE_EPSILONS[index], PROFILE_ROUNDING[index]
        mean = _profile_mean(epsilon, rounding)
        if index:
            parent = self.sample(index-1, y, x)
            amp, ay, ax = [CASCADE_GAIN*v for v in parent[:3]]
        else:
            amp, ay, ax, _, _ = self.controls.sample(y, x)
        kernel = _kernel(y.device, 'field')
        if kernel is not None:
            yy, xx = torch.broadcast_tensors(y, x)
            out = torch.empty((8, *yy.shape), device=y.device, dtype=torch.float64)
            amps = torch.stack((amp, ay, ax)).contiguous()
            p = torch.stack(tuple(parent[j] for j in (0, 1, 2, 5, 6, 7))) if index else amps
            refinement._launch(kernel, yy.shape, torch.stack(gaussian), amps, p,
                               yy.contiguous(), xx.contiguous(), np.int32(yy.shape[0]), np.int32(yy.shape[1]),
                               np.int32(index), np.uint64((int(getattr(self.world, 'seed', 0))+811)&refinement._MASK),
                               sigma, GULLY_SHARE[index], math.sqrt(epsilon), epsilon,
                               1+2*math.sqrt(epsilon), rounding, mean, out)
            result = (*tuple(out), mean)
            self.samples[key] = y, x, result
            if self.check is not None:
                self.check()
            return result
        real, imag, ry, rx, jy, jx = gaussian
        h, hr, hi, phase, rho = _transfer(real, imag, epsilon, rounding)
        patch, py, px = _patch(int(getattr(self.world, 'seed', 0)), index, y, x)
        if index:
            head, hy, hx = _head_gate(parent)
            patch, py, px = patch*head, py*head+patch*hy, px*head+patch*hx
        patch, py, px = [GULLY_SHARE[index]*v for v in (patch, py, px)]
        amp, ay, ax = amp*patch, ay*patch+amp*py, ax*patch+amp*px
        if self.check is not None:
            self.check()
        result = (amp*h, ay*h+amp*(hr*ry+hi*jy), ax*h+amp*(hr*rx+hi*jx), phase, rho, amp, ay, ax, mean)
        self.samples[key] = y, x, result
        return result


def _project_first(patterns, bounds, last, device):
    """C2 reconstruction of native-cell means of the first height displacement.

    A coefficient at native n is a fixed FIR of globally anchored native
    averages. Its support is 12 cells per axis plus cubic taps and 4*60 m
    impulse reach. Extra descendant impulse queries extend the coefficient
    region by 4*sum(descendant wavelengths), independent of request shape.
    """
    margin = 4*sum(WAVELENGTHS[1:last+1])
    y0, x0, y1, x1 = bounds
    n0, m0 = math.floor((y0-margin)/30)-3, math.floor((x0-margin)/30)-3
    n1, m1 = math.floor((y1+margin)/30)+4, math.floor((x1+margin)/30)+4
    radius = PROJECTION_RADIUS
    ya, xa, yb, xb = n0-radius, m0-radius, n1+radius, m1+radius
    patterns.first_bounds = (ya*30+3.75, xa*30+3.75, yb*30-3.75, xb*30-3.75)
    y = (torch.arange(ya*4, yb*4, device=device, dtype=torch.float64)+.5)[:, None]*7.5
    x = (torch.arange(xa*4, xb*4, device=device, dtype=torch.float64)+.5)[None, :]*7.5
    values = patterns.sample(0, y, x)
    displacement = values[0]-values[8]*values[5]
    # Fixed gather order: not an adaptive reduction over a request window.
    average = None
    for row in range(4):
        for col in range(4):
            term = displacement[row::4, col::4]/16
            average = term if average is None else average+term
    coefficients = refinement._filter(average, _PROJECT_INVERSE)
    projection = _Controls.__new__(_Controls)
    projection.y0, projection.x0 = n0, m0
    projection.fields = torch.stack((coefficients, torch.zeros_like(coefficients), torch.zeros_like(coefficients)))
    patterns.projection = projection
    return projection


def _base(world, level, i1, j1, i2, j2, check, *, native=None):
    """Refinement's coast/C2/decoder path, without its procedural octaves.

    Keep this small explicit composition instead of changing a global flag
    or monkeypatching refinement while another request may be sampling it.
    """
    if KEEP_PROCEDURAL_BASE:
        return refinement.sample_refined(world, level, i1, j1, i2, j2, check=check)
    scale = 2**level
    r0, r1, rows, fy = refinement._axis(i1, i2, scale, world.device)
    c0, c1, cols, fx = refinement._axis(j1, j2, scale, world.device)
    lo, left, hi, right = refinement._base_bounds(level, i1, j1, i2, j2)
    lo, left, hi, right = min(lo, r0), min(left, c0), max(hi, r1+1), max(right, c1+1)
    margin = max(refinement.SPLINE_RADIUS, refinement.SMOOTH_RADIUS+refinement.HALF_BAND_RADIUS)
    if native is None:
        native = refinement._NativeWindow(world, lo-margin, left-margin, hi+margin, right+margin, check)
    _, decoder = refinement._roughness(world, native, lo, left, hi, right)
    base, coast = refinement._coastal_base(native, level, i1, j1, i2, j2)
    if level > refinement.COAST_LAG and bool((coast > 0).any()):
        base = base+coast*refinement._coast_offset(world, level, i1, j1, i2, j2)
    gain = refinement._decoder_gain(decoder[r0-lo:r1+1-lo, c0-left:c1+1-left])
    gain = refinement._evaluate(gain, refinement._cubic_taps(rows, fy), refinement._cubic_taps(cols, fx))
    detail = refinement._level_detail(world, 1, level, i1, j1, i2, j2)*.5
    result = refinement._limited(base, detail*float(world.kwargs['residual_std'])*gain)
    return (result*result.abs()).float().contiguous()


@torch.inference_mode()
def _drainage_base(world, level, i1, j1, i2, j2, check=None):
    """The new path's coherent native C2/decoder base, for diagnostics."""
    view = _drainage_world_view(world,level,i1,j1,check)
    return _base(view,level,i1,j1,i2,j2,check,native=_canonical_reader(view,check))


def _lod_weight(level, wavelength):
    # A one-LOD transition ending at the first resolved (>=4 sample) LOD.
    # At its unresolved integer endpoint the weight is exactly zero.
    first = max(0, math.ceil(math.log2(120/wavelength))-1)
    return float(refinement._smoothstep(first, first+1, torch.tensor(float(level))))


def _crest_weights(level, index):
    # A value-noise cell contributes principally around two cell widths.
    return tuple(weight*_lod_weight(level, 2*WAVELENGTHS[index]/spacing)
                 for spacing, weight in ((6, .57), (12, .29), (24, .14)))


def _rock_weights(level):
    return tuple(amp*_lod_weight(level, wave) for amp, wave in ((.22, 8.), (.10, 4.), (.05, 2.)))


@torch.inference_mode()
def sample_phasor(world, level, i1, j1, i2, j2, *, intensity=1.0, drainage=False, check=None):
    """Return metre heights (H,W) and flow/deposit/scree/rock (4,H,W).

    Coordinates, budgets and cancellation follow sample_refined. Intensity
    is a finite nonnegative multiplier; values above one are permitted.
    No global reductions, tile windows or request-order state enter the
    pattern. FP64 world coordinates retain negative/far-origin precision.
    ``drainage=True`` selects PHASOR_DRAINAGE_VERSION without changing v3's
    default numeric path or using mutable module flags per request.
    """
    try:
        intensity = float(intensity)
    except (TypeError, ValueError) as exc:
        raise ValueError('Phasor intensity must be finite and nonnegative') from exc
    if not math.isfinite(intensity) or intensity < 0:
        raise ValueError('Phasor intensity must be finite and nonnegative')
    intensity = _intensity_gain(intensity)
    # Validate without doing a full procedural baseline on the active path.
    if not isinstance(level, int) or not 0 <= level <= -refinement.MIN_LOD:
        raise ValueError('Unsupported refinement level')
    if any(not isinstance(v, int) for v in (i1, j1, i2, j2)):
        raise ValueError('Refinement coordinates must be integers')
    if i2 <= i1 or j2 <= j1 or (i2-i1)*(j2-j1) > 1024**2:
        raise ValueError('Refinement read exceeds bounded output budget')
    if check is not None:
        check()
    if intensity == 0 or level == 0:
        base = refinement.sample_refined(world, level, i1, j1, i2, j2, check=check)
        return base, torch.zeros((4, *base.shape), device=base.device, dtype=torch.float32)
    if drainage:
        world = _drainage_world_view(world,level,i1,j1,check)
    observer = getattr(world, '_terrain_window_scheduler', None)
    with observer.scope(check) if observer is not None else nullcontext():
        weights = [_lod_weight(level, wave) for wave in WAVELENGTHS]
        crest_weights = [_crest_weights(level, index) for index in range(3)]
        active = [i for i, weight in enumerate(weights) if weight > 0]
        if not active:
            base = _base(world, level, i1, j1, i2, j2, check)
            return base, torch.zeros((4, *base.shape), device=base.device, dtype=torch.float32)
        cell = 30/2**level
        y = (torch.arange(i1, i2, device=world.device, dtype=torch.float64)+.5)[:, None]*cell
        x = (torch.arange(j1, j2, device=world.device, dtype=torch.float64)+.5)[None, :]*cell
        # Recursive impulse orientations need only sum of ancestor supports.
        margin = 4*sum(WAVELENGTHS[:max(active)+1])+180
        if PROJECT_NATIVE_MEAN:
            margin += 30*(PROJECTION_RADIUS+3)
        bb = refinement._base_bounds(level, i1, j1, i2, j2)
        br = max(refinement.SPLINE_RADIUS, refinement.SMOOTH_RADIUS+refinement.HALF_BAND_RADIUS)
        controls = _Controls(world, ((i1+.5)*cell-margin, (j1+.5)*cell-margin,
                                     (i2-.5)*cell+margin, (j2-.5)*cell+margin), check,
                             base_bounds=(bb[0]-br, bb[1]-br, bb[2]+br, bb[3]+br), drainage=drainage)
        base = _base(world, level, i1, j1, i2, j2, check, native=controls.native)
        patterns = _Patterns(world, controls, check,
                             bounds=((i1+.5)*cell, (j1+.5)*cell, (i2-.5)*cell, (j2-.5)*cell),
                             last=max(active), drainage=drainage)
        correction = torch.zeros_like(base, dtype=torch.float64)
        if PROJECT_NATIVE_MEAN:
            projection = _project_first(patterns, patterns.bounds, max(active), base.device)
            correction = projection.sample(y, x)[0]
        _, _, _, gy, gx = controls.sample(y, x)
        native_rock = controls.rock_control.sample(y,x)[0] if drainage else correction
        kernel = _kernel(base.device, 'finish')
        if kernel is not None:
            packed = torch.zeros((3, 4, *base.shape), device=base.device, dtype=torch.float64)
            for index in active:
                values = patterns.sample(index, y, x)
                packed[index] = torch.stack((values[0]-values[8]*values[5], values[3], values[4], values[5]))
            yy, xx = torch.broadcast_tensors(y, x)
            height = torch.empty_like(base)
            material = torch.empty((4, *base.shape), device=base.device, dtype=torch.float32)
            crests = torch.tensor(crest_weights, device=base.device, dtype=torch.float64)
            rocks = torch.tensor(_rock_weights(level), device=base.device, dtype=torch.float64)
            refinement._launch(kernel, base.shape, base, packed, crests, gy.contiguous(), gx.contiguous(),
                               yy.contiguous(), xx.contiguous(), np.int32(base.shape[0]), np.int32(base.shape[1]),
                               np.uint64((int(getattr(world, 'seed', 0))+919) & refinement._MASK),
                               intensity, correction.contiguous(), rocks, native_rock.contiguous(),
                               np.int32(bool(drainage)), *weights, height, material)
            if check is not None:
                check()
            return height, material
        delta = torch.zeros_like(base, dtype=torch.float64)
        surface = torch.zeros((4, *base.shape), device=base.device, dtype=torch.float64)
        slope = (gy*gy+gx*gx).sqrt()
        steep = refinement._smoothstep(.15, .65, slope)
        for index in active:
            r, _, _, phase, rho, amp, _, _, mean = patterns.sample(index, y, x)
            crest = refinement._smoothstep(math.pi/4, 3*math.pi/4, phase.abs())
            fbm = sum(crest_weights[index][k]*refinement._value_noise(int(getattr(world, 'seed', 0))+919,
                       y/30, x/30, WAVELENGTHS[index]/30/spacing, 71+k)
                       for k, spacing in enumerate((6, 12, 24)))
            centered = r-amp*mean-(correction if index == 0 else 0)
            delta = delta+weights[index]*(centered+.20*amp*crest*rho*fbm)
            strength = (amp/(.12*WAVELENGTHS[index])).clamp(0, 1)*rho*weights[index]
            bottom = 1-refinement._smoothstep(.12, .85, phase.abs())
            flank = refinement._smoothstep(.25, .9, phase.abs())*(1-crest)
            signals = (bottom, bottom*(1-steep), steep*flank, steep*crest)
            surface = torch.maximum(surface, torch.stack(signals)*strength)
        rocky = sum(weight*refinement._value_noise(int(getattr(world, 'seed', 0))+2628,
                     y, x, wave/2, 131+k) for k, (weight, wave) in enumerate(zip(_rock_weights(level), (8., 4., 2.))))
        rock_amount = surface[3]*(patterns.sample(0,y,x)[5]/4).clamp(0,1)
        if drainage:
            patch = refinement._smoothstep(-.1,.4,refinement._value_noise(
                int(getattr(world,'seed',0))+3560,y,x,120.,117))
            rock_amount = torch.maximum(rock_amount,native_rock*patch)
            surface[3] = torch.maximum(surface[3],native_rock*patch)
        if drainage:
            delta = delta+rocky*rock_amount
        else:
            delta = delta+rocky*surface[3]*(patterns.sample(0,y,x)[5]/4).clamp(0,1)
        coast = refinement._smoothstep(4, 35, base.double())
        delta = intensity*delta*coast
        # Relative soft limiter preserves the sign even at large strengths.
        height = refinement._limited(base.double(), delta).float().contiguous()
        surface = (surface*coast).clamp(0, 1).float().contiguous()
        if check is not None:
            check()
        return height, surface

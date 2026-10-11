"""Experimental 15 m to 0.47 m refinement below the 30 m checkpoint.

The checkpoint is not a super-resolution model. LOD -1 reuses the 30 m
decoder once, with its appearance latents over half the footprint. Finer
levels would feed it latents stretched beyond its training distribution, so
LOD -2 to -6 add deterministic procedural octaves instead: sparse convolution
of zero-mean kernels, scaled from the local roughness of the decoder and
native octaves and rougher on steep and convex ground. Each finer octave is
markedly fainter in slope, so zooming in adds little grain, and patches set
by a low-frequency noise and the relief only modulate it gently. All
surfaces are continuous functions of position, sampled directly at the
requested spacing: hillshading shows no grid of the 30 m source or of
coarser levels. Composition happens in the checkpoint's signed-sqrt height.
Near sea level a smoothed base gives one coastline free of the 30 m raster,
the same at every level up to a turn of a few degrees; detail never flips
land and sea.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import numpy as np
import torch
import torch.nn.functional as F
from infinite_tensor import InfiniteTensor, TensorWindow
from terrain_diffusion.inference.world_pipeline import linear_weight_window
from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler

VERSION = 'decoder-v12-nyquist-notch'
MIN_LOD = -6
# Finest level whose octave comes from the decoder; finer ones are procedural.
DECODER_LEVELS = 1
# Truncated exact inverse of the cubic B-spline sampling filter [1, 4, 1]/6.
# Its taps decay as |sqrt(3)-2|**n: twelve per side leave < 1e-7 relative.
PREFILTER_RADIUS = 12
_POLE = math.sqrt(3) - 2
_PREFILTER = [_POLE**abs(n) for n in range(-PREFILTER_RADIUS, PREFILTER_RADIUS+1)]
_PREFILTER = [tap / math.fsum(_PREFILTER) for tap in _PREFILTER]
# Half-band low-pass (Blackman-windowed sinc, cutoff at the parent grid's
# Nyquist). Its complement keeps only the new octave of each level, so a
# finer level adds detail without moving its parent's coarser features.
HALF_BAND_RADIUS = 7
_HALF_BAND = [.5 if n == 0 else math.sin(math.pi*n/2)/(math.pi*n) *
              (.42 + .5*math.cos(math.pi*n/(HALF_BAND_RADIUS+1)) +
               .08*math.cos(2*math.pi*n/(HALF_BAND_RADIUS+1)))
              for n in range(-HALF_BAND_RADIUS, HALF_BAND_RADIUS+1)]
_HALF_BAND = [tap / math.fsum(_HALF_BAND) for tap in _HALF_BAND]
# Nyquist notch: identity minus a Blackman low-pass shifted to the grid's
# Nyquist (flat to 0.4 cycles/sample, zero at 0.5). The decoders leave
# alternating row and column stripes there; an interpolating spline keeps
# them, and magnified hillshading turned them into a 60 m and 30 m grid.
NOTCH_RADIUS = 12
_NOTCH = [.42 + .5*math.cos(math.pi*n/(NOTCH_RADIUS+1)) + .08*math.cos(2*math.pi*n/(NOTCH_RADIUS+1))
          for n in range(-NOTCH_RADIUS, NOTCH_RADIUS+1)]
_NOTCH = [(n == 0) - (-1)**n * tap / math.fsum(_NOTCH) for n, tap in zip(range(-NOTCH_RADIUS, NOTCH_RADIUS+1), _NOTCH)]
# Spline coefficients of the notched samples, as one FIR.
_INTERPOLATE = [math.fsum(_NOTCH[i]*_PREFILTER[k-i] for i in range(len(_NOTCH)) if 0 <= k-i < len(_PREFILTER))
                for k in range(len(_NOTCH) + len(_PREFILTER) - 1)]
SPLINE_RADIUS = PREFILTER_RADIUS + NOTCH_RADIUS
# Binomial smoothing (radius 4, ~120 m on the native grid) for terrain fields.
_SMOOTH = [math.comb(8, n) / 256 for n in range(9)]
SMOOTH_RADIUS = 4

# Procedural octaves: sparse convolution with one impulse per cell of the
# octave's own grid. An impulse carries either a radial kernel
# K(t) = (1-14t+28t^2)(1-t)^4, t = r^2/RADIAL_RADIUS^2, or a gully/rib kernel
# p(v)q(u) in the frame of the local downslope direction u, with
# p(v) = (1-26v^2+65v^4)(1-v^2)^4 and q(u) = (1-u^2)^4. Both have two
# vanishing moments (power ~k^4 near zero), so coarser features stay in
# place, and peak at 2.83 cells: the octave between this grid's and its
# parent's Nyquist. Only a few kernels at the coarsest octave follow
# slopes: more, or finer, they combed slopes into streaks. Only polynomials,
# division and square roots: results do not depend on tile shape.
RADIAL_RADIUS = 3.66
ACROSS_RADIUS = 3.55
ALONG_GAIN = 1.25                # along-slope elongation on steep ground
REACH = 6                        # cells searched each side: ACROSS_RADIUS*ALONG_GAIN + jitter
_RADIAL_POWER = math.pi * 5 / 143                            # int K^2 dA / radius^2
_ORIENTED_POWER = 0.4386831965240903 * 0.5990767402532109    # int p^2 * int q^2
# Per-octave amplitude decay 2**-H from the reference roughness. H = 1 keeps
# each octave's slope constant; above it, finer octaves fade in the hillshade.
# The newest octave of a level peaks at ~3 samples: at equal slope it read as
# pixel-scale orange peel. Steep ground decays a little more slowly. The reference adds NATIVE_SHARE of the 30 m
# octave to the LOD -1 decoder octave.
HURST_GENTLE, HURST_STEEP = 1.35, 1.25
NATIVE_SHARE = .3
# Soft cap on each octave's RMS slope, decoder octave included: a cliff's
# decoder ringing must not become a comb of metre-high ribs. Each procedural
# octave halves it: rough ground sits at the cap, and equal caps summed into a
# grain that grew with every level.
SLOPE_CAP = .4
CAP_DECAY = .5
# Patches. Finer octaves are not everywhere: a value noise of PATCH_NOISE
# octaves (spacing in native cells, weight), shared by every octave, plus a
# relief term (steep and convex ground) decides where they keep their
# amplitude. Octave o loses up to PATCH_CUT[o-1] of it elsewhere. Deeper cuts
# left blurred ground beside patches many times rougher.
PATCH_NOISE = ((4., .55), (1.7, .3), (.7, .15))
PATCH_LOW, PATCH_HIGH = -.05, .3
PATCH_CUT = (.15, .25, .3, .35, .4)
# Share of heavy-tailed (cubed uniform) impulse weights per octave. More
# read as scattered dots.
TAIL = (0., .1, .2, .25, .3)
# Coastline. Within COAST_NEAR native cells of sea level the base comes from
# signed-sqrt heights smoothed by COAST_SMOOTH and a cubic B-spline (sigma ~0.9
# cell), fading to the exact spline by COAST_FAR cells. That zero line is one
# continuous curve: every level shows the same coast, only sampled finer.
# Below ~9 m it then bends very slightly: isotropic octaves move it along its
# normal, COAST_LAG levels coarser than the display, with a total RMS turn of
# COAST_ANGLE degrees (a few degrees at most) and no new islands or pools.
COAST_SMOOTH = [.25, .5, .25]
COAST_NEAR, COAST_FAR = 1., 2.5
COAST_ANGLE = .75
COAST_LAG = 2
_MASK = 0xffffffff


def _axis(start, stop, scale, device):
    """Fine cells [start, stop) on a grid `scale` times coarser.

    Returns the first and last coarse indices a cubic B-spline reads, each
    cell's left coarse sample relative to the first, and its fraction in [0, 1). Integer origins
    keep power-of-two fractions exact at negative and far coordinates.
    """
    cells = torch.arange(start, stop, device=device, dtype=torch.int64)
    fraction = (cells.remainder(scale).double()+.5)/scale-.5
    left = cells.div(scale, rounding_mode='floor') - (fraction < 0).long()
    first = (2*start+1-scale) // (2*scale) - 1
    last = (2*(stop-1)+1-scale) // (2*scale) + 2
    return first, last, left - first, fraction.remainder(1)


def _cubic_taps(index, t):
    t2, t3 = t*t, t*t*t
    weights = ((1-t)**3/6, (3*t3-6*t2+4)/6, (-3*t3+3*t2+3*t+1)/6, t3/6)
    return [(index-1+k, w) for k, w in enumerate(weights)]


def _evaluate(values, rows, cols):
    """Separable weighted gathers in a fixed order: reads of any tile shape agree exactly."""
    out = None
    for index, weight in rows:
        term = values[index] * weight[:, None]
        out = term if out is None else out + term
    result = None
    for index, weight in cols:
        term = out[:, index] * weight[None]
        result = term if result is None else result + term
    return result


def _filter(values, taps):
    """Valid separable FIR in a fixed tap order (no shape-dependent kernels)."""
    radius = len(taps) // 2
    kernel = _kernel('fir', values.device) if values.dtype == torch.float64 else None
    if kernel is not None:
        weights = torch.tensor(taps, device=values.device, dtype=torch.float64)
        for axis in (0, 1):
            values = values.contiguous()
            h, w = values.shape
            out = torch.empty((h - 2*radius, w) if axis == 0 else (h, w - 2*radius),
                              device=values.device, dtype=torch.float64)
            _launch(kernel, out.shape, values, weights, len(taps), h, w, axis, out)
            values = out
        return values
    for dim in (0, 1):
        size = values.shape[dim] - 2*radius
        out = None
        for offset, tap in enumerate(taps):
            term = values.narrow(dim, offset, size) * tap
            out = term if out is None else out + term
        values = out
    return values


class _NativeWindow:
    """One float64 read of the native DEM shared by every neighbourhood of a tile."""

    def __init__(self, world, n0, m0, n1, m1, check=None):
        values = sample_refined(world, 0, n0, m0, n1, m1, check=check)
        self.values = values.to(world.device, dtype=torch.float64)
        self.n0, self.m0 = n0, m0

    def __call__(self, n0, m0, n1, m1):
        return self.values[n0-self.n0:n1-self.n0, m0-self.m0:m1-self.m0]


def _signed_sqrt(z):
    return z.sign() * z.abs().sqrt()


def _limited(value, change):
    """`value + change`, soft-limited to half the distance from sea level: never a new sign."""
    return value + change*value.abs() / (value*value + 4*change*change).sqrt().clamp_min(
        torch.finfo(value.dtype).tiny)


def _base_bounds(level, i1, j1, i2, j2):
    """Native cells a fine tile's base reads: cubic taps of every cell centre."""
    scale = 2**level
    return i1 // scale - 2, j1 // scale - 2, (i2-1) // scale + 3, (j2-1) // scale + 3


def _coastal_base(native, level, i1, j1, i2, j2):
    """Native DEM in signed-sqrt height s on the fine grid, with a refined coastline.

    The checkpoint generates heights as s*|s|; along coasts s crosses zero
    nearly linearly while metres flatten to zero. Inland: the C2 interpolating
    spline of s, limited to the sign of its bilinear surface (no new specks).
    Near the coast: a smoothed spline whose zero line ignores the 30 m raster
    (no staircases; one-cell inlets fade instead of becoming ponds). The
    surface does not depend on the level, so neither does the coastline.
    Also returns |grad s| in the low coastal band (zero elsewhere): adding it
    times an offset moves the zero line by that many native cells.
    """
    n0, m0, n1, m1 = _base_bounds(level, i1, j1, i2, j2)
    radius = SPLINE_RADIUS
    q = _signed_sqrt(native(n0-radius, m0-radius, n1+radius, m1+radius))
    interpolating = _filter(q, _INTERPOLATE)
    smooth = _filter(_trim(q, radius-2), COAST_SMOOTH)
    gy = (smooth[2:, 1:-1] - smooth[:-2, 1:-1]) / 2
    gx = (smooth[1:-1, 2:] - smooth[1:-1, :-2]) / 2
    gradient = (gy*gy + gx*gx).sqrt()
    smooth, raw = smooth[1:-1, 1:-1], _trim(q, radius)

    scale = 2**level
    device = q.device
    _, _, rows, fy = _axis(i1, i2, scale, device)
    _, _, cols, fx = _axis(j1, j2, scale, device)
    # _axis indexes from the first cell a cubic reads; the bounds start one earlier.
    first_r, first_c = _axis(i1, i2, scale, device)[0] - n0, _axis(j1, j2, scale, device)[0] - m0
    rows, cols = rows + first_r, cols + first_c
    ty, tx = _cubic_taps(rows, fy), _cubic_taps(cols, fx)
    linear = _evaluate(raw, [(rows, 1-fy), (rows+1, fy)], [(cols, 1-fx), (cols+1, fx)])
    exact = _limited(linear, _evaluate(interpolating, ty, tx) - linear)
    coastal = _evaluate(smooth, ty, tx)
    slope = _evaluate(gradient, ty, tx)
    near = 1 - _smoothstep(COAST_NEAR, COAST_FAR, coastal.abs() / (slope + 1e-9))
    band = near * (1 - _smoothstep(1, 3, coastal.abs()))
    return exact + (coastal - exact)*near, slope*band


def _level_detail(world, source, level, i1, j1, i2, j2):
    """Decoder band of `source` level as a C2 spline, sampled on the `level` grid.

    The band is the decoder sample minus its half-band low-pass: the octave
    between the parent's and this level's Nyquist frequencies. Prefiltered spline coefficients reproduce it on its own
    grid and interpolate it smoothly on every finer grid.
    """
    scale, radius = 2**(level-source), SPLINE_RADIUS + HALF_BAND_RADIUS
    r0, r1, rows, fy = _axis(i1, i2, scale, world.device)
    c0, c1, cols, fx = _axis(j1, j2, scale, world.device)
    fused = _detail_field(world, source)[:, r0-radius:r1+radius+1, c0-radius:c1+radius+1]
    fused = fused.to(world.device, dtype=torch.float64)
    detail = (fused[0] / fused[1]).clamp(-3, 3)
    band = detail[HALF_BAND_RADIUS:-HALF_BAND_RADIUS, HALF_BAND_RADIUS:-HALF_BAND_RADIUS] - _filter(detail, _HALF_BAND)
    coefficients = _filter(band, _INTERPOLATE)
    return _evaluate(coefficients, _cubic_taps(rows, fy), _cubic_taps(cols, fx))


def _smoothstep(low, high, x):
    t = ((x - low) / (high - low)).clamp(0, 1)
    return t*t*(3 - 2*t)


def _hash(x):
    """32-bit integer hash in int64 lanes; products stay below 2**63."""
    x = x & _MASK
    x = x ^ (x >> 16)
    x = (x * 0x7feb352d) & _MASK
    x = x ^ (x >> 15)
    x = (x * 0x1b873593) & _MASK
    return x ^ (x >> 16)


def _hash_int(x):
    x &= _MASK
    x ^= x >> 16
    x = (x * 0x7feb352d) & _MASK
    x ^= x >> 15
    x = (x * 0x1b873593) & _MASK
    return x ^ (x >> 16)


def _uniform(h, stream):
    """Uniform in [0, 1) with 24 bits: exact in float32 as well as float64."""
    return (_hash(h + stream) & 0xffffff).double() / 2**24


def _value_noise(seed, y, x, spacing, stream):
    """Value noise in [-1, 1] with C1 interpolation on a lattice of `spacing` native cells."""
    gy, gx = y / spacing, x / spacing
    iy, ix = gy.floor(), gx.floor()
    fy, fx = gy - iy, gx - ix
    sy, sx = fy*fy*(3 - 2*fy), fx*fx*(3 - 2*fx)
    r, c = iy.long() & _MASK, ix.long() & _MASK
    corner = lambda dr, dc: 2*_uniform(_hash(_hash(seed ^ ((r + dr) & _MASK)) ^ ((c + dc) & _MASK)), stream) - 1
    v00, v01, v10, v11 = corner(0, 0), corner(0, 1), corner(1, 0), corner(1, 1)
    top = v00 + (v01 - v00)*sx
    bottom = v10 + (v11 - v10)*sx
    return top + (bottom - top)*sy


def _patch_noise(seed, y, x):
    total = None
    for k, (spacing, weight) in enumerate(PATCH_NOISE):
        term = weight * _value_noise(seed, y, x, spacing, 11 + k)
        total = term if total is None else total + term
    return total


def _slope_cap(source):
    """Amplitude whose octave at 2.83 cells of level `source` has its capped RMS slope."""
    cap = SLOPE_CAP * CAP_DECAY**max(0, source - DECODER_LEVELS)
    return cap * 2**1.5 * (30 / 2**source) / (2*math.pi)


def _soft_cap(values, cap):
    ratio = values / cap
    return values / (1 + ratio*ratio).sqrt()


def _trim(values, by):
    return values[by:-by, by:-by] if by else values


def _roughness(world, native, n0, m0, n1, m1):
    """Local RMS in metres of the 30 m native octave and the raw LOD -1 decoder octave.

    Both on native cells [n0, n1) x [m0, m1), smoothed over ~120 m.
    """
    r, k = SMOOTH_RADIUS, HALF_BAND_RADIUS
    heights = native(n0-r-k, m0-r-k, n1+r+k, m1+r+k)
    native_band = _trim(heights, k) - _filter(heights, _HALF_BAND)
    lo, hi, left, right = n0 - r, n1 + r, m0 - r, m1 + r
    fused = _detail_field(world, 1)[:, 2*lo-k:2*hi+k, 2*left-k:2*right+k]
    fused = fused.to(world.device, dtype=torch.float64)
    detail = (fused[0] / fused[1]).clamp(-3, 3)
    band = detail[k:-k, k:-k] - _filter(detail, _HALF_BAND)
    # Metres as in sample_refined: band * 0.5 * 2*sqrt|z| * residual_std.
    power = band*band
    power = (power[0::2, 0::2] + power[0::2, 1::2] + power[1::2, 0::2] + power[1::2, 1::2]) / 4
    power = power * _trim(heights, k).abs() * float(world.kwargs['residual_std'])**2
    return _filter(native_band*native_band, _SMOOTH).sqrt(), _filter(power, _SMOOTH).sqrt()


def _decoder_gain(decoder):
    """Smooth gain bounding the decoder octave's local RMS slope (cliff ringing)."""
    return 1 / (1 + (decoder/_slope_cap(DECODER_LEVELS))**2).sqrt()


def _terrain_fields(native, native_rms, decoder_rms, n0, m0, n1, m1):
    """Slope, relative curvature and roughness on native cells [n0, n1) x [m0, m1).

    Roughness combines the LOD -1 decoder octave as it is displayed with a
    fraction of the native 30 m octave: the decoder's own octave is several
    times weaker, and alone would leave plateaus blurred at metre scale.
    """
    r = SMOOTH_RADIUS
    margin = 2*r + 1
    smooth = _filter(native(n0-margin, m0-margin, n1+margin, m1+margin), _SMOOTH)
    inner = smooth[1:-1, 1:-1]
    gy = (smooth[2:, 1:-1] - smooth[:-2, 1:-1]) / 60
    gx = (smooth[1:-1, 2:] - smooth[1:-1, :-2]) / 60
    laplacian = (smooth[2:, 1:-1] + smooth[:-2, 1:-1] + smooth[1:-1, 2:] + smooth[1:-1, :-2] - 4*inner) / 900
    spread = (_filter(laplacian*laplacian, _SMOOTH) + 1e-18).sqrt()
    curvature = (_trim(laplacian, r) / spread).clamp(-1, 1)
    shown = decoder_rms * _decoder_gain(decoder_rms)
    roughness = (shown*shown + (NATIVE_SHARE*native_rms)**2).sqrt()
    return _trim(gy, r), _trim(gx, r), curvature, roughness


def _bilinear(fields, y, x):
    """Fields at native-cell positions y, x (relative to the field origin, in cells)."""
    y, x = y - .5, x - .5
    iy, ix = y.floor(), x.floor()
    fy, fx = y - iy, x - ix
    iy, ix = iy.long(), ix.long()
    out = []
    for field in fields:
        top = field[iy, ix]*(1-fx) + field[iy, ix+1]*fx
        bottom = field[iy+1, ix]*(1-fx) + field[iy+1, ix+1]*fx
        out.append(top*(1-fy) + bottom*fy)
    return out


def _impulses(world, source, a0, b0, a1, b1, fields, origin):
    """Kernel parameters of the impulses in octave cells [a0, a1) x [b0, b1).

    Every value depends only on the cell's global index, the seed and the
    terrain fields at the impulse: identical for any tile that reads it.
    """
    device = world.device
    octave = source - DECODER_LEVELS
    seed = (int(getattr(world, 'seed', 0)) * 1000003 + source * 7919) & _MASK
    orient = .1 if octave == 1 else 0.
    patch_seed = _hash_int((int(getattr(world, 'seed', 0)) * 1000003 + 4241) & _MASK)
    cut = PATCH_CUT[octave-1]
    tail = TAIL[octave-1]
    gentle, steep_decay = 2**(-HURST_GENTLE*octave), 2**(-HURST_STEEP*octave)
    kernel = _kernel('impulses', device)
    if kernel is not None:
        out = torch.empty((7, a1 - a0, b1 - b0), device=device, dtype=torch.float32)
        fields = [f.contiguous() for f in fields]
        _launch(kernel, out.shape[1:], *fields, np.int32(fields[0].shape[1]), np.int64(a0), np.int64(b0),
                np.int32(a1 - a0), np.int32(b1 - b0), np.uint64(_hash_int(seed)), np.uint64(patch_seed), float(2**source),
                float(origin[0]), float(origin[1]), orient, gentle, steep_decay, _slope_cap(source),
                cut, PATCH_LOW, PATCH_HIGH, *[float(v) for o in PATCH_NOISE for v in o], tail,
                ACROSS_RADIUS, ALONG_GAIN - 1, 3**.5, (RADIAL_RADIUS**2 * _RADIAL_POWER)**.5, _ORIENTED_POWER, out)
        return list(out)
    a = torch.arange(a0, a1, device=device, dtype=torch.int64)[:, None]
    b = torch.arange(b0, b1, device=device, dtype=torch.int64)[None, :]
    h = _hash(_hash(_hash(torch.tensor(seed, device=device)) ^ (a & _MASK)) ^ (b & _MASK))
    jy, jx = _uniform(h, 1), _uniform(h, 2)
    # Impulse positions on the native grid, relative to the field origin.
    scale = 2**source
    py, px = (a.double() + jy) / scale, (b.double() + jx) / scale
    y, x = py - origin[0], px - origin[1]
    gy, gx, curvature, roughness = _bilinear(fields, y, x)
    slope = (gy*gy + gx*gx).sqrt()
    dy, dx = gy / (slope + 1e-12), gx / (slope + 1e-12)
    # Rare oriented kernels on slopes at 7.5 m, none from 3.75 m: they
    # streaked slopes, and finer debris has no preferred direction.
    steep = _smoothstep(.15, .7, slope)
    # Rocky patches: noise, raised on steep and convex ground. Outside them
    # finer octaves fade, and so do gullies and ribs down the slope.
    presence = _smoothstep(PATCH_LOW, PATCH_HIGH, _patch_noise(patch_seed, py, px) + (.1*steep - .2*curvature))
    gain = 1 - cut*(1 - presence)
    oriented = _uniform(h, 3) < _smoothstep(.02, .15, slope) * orient * gain
    along = ACROSS_RADIUS * (1 + (ALONG_GAIN-1)*_smoothstep(.05, .4, slope))
    decay = (1-steep)*gentle + steep*steep_decay
    # Valleys (positive Laplacian) stay smoother than ridges and convex slopes.
    amplitude = roughness * decay * (1 - .4*curvature)
    # After the cap: rough ground mostly sits at it, which made it uniform.
    amplitude = _soft_cap(amplitude, _slope_cap(source)) * gain
    # Heavy-tailed weights at fine octaves: a few distinct bumps among many
    # faint ones rather than an even grain; unit RMS at either end.
    s = 2*_uniform(h, 5) - 1
    weight = amplitude * ((1 - tail)*(3**.5*s) + tail*(7**.5*(s*s*s)))
    radial = torch.where(oriented, 0., weight / (RADIAL_RADIUS**2 * _RADIAL_POWER)**.5)
    weight = torch.where(oriented, weight / (along * ACROSS_RADIUS * _ORIENTED_POWER).sqrt(), 0.)
    return [v.float() for v in (jy, jx, dy, dx, 1 / along, weight, radial)]


# CUDA kernels (CuPy/NVRTC, optional). Each repeats the PyTorch fallback's
# operations in the same order without fused multiply-add, so both paths give
# identical results; one launch replaces hundreds of small elementwise ones.
_SOURCES = {
    'fir': r'''
extern "C" __global__ void fir(const double* v, const double* taps, int n, int h, int w, int axis,
                               double* out) {
  int x = blockIdx.x*blockDim.x + threadIdx.x, y = blockIdx.y*blockDim.y + threadIdx.y;
  int ow = axis ? w - n + 1 : w, oh = axis ? h : h - n + 1;
  if (x >= ow || y >= oh) return;
  int step = axis ? 1 : w;
  const double* s = v + (long long)y*w + x;
  double acc = s[0]*taps[0];
  for (int k = 1; k < n; ++k) acc = acc + s[(long long)k*step]*taps[k];
  out[(long long)y*ow + x] = acc;
}
''',
    'impulses': r'''
typedef unsigned long long u64;
__device__ u64 mix(u64 x) {
  const u64 M = 0xffffffffULL;
  x &= M; x ^= x >> 16; x = (x*0x7feb352dULL) & M; x ^= x >> 15; x = (x*0x1b873593ULL) & M;
  return x ^ (x >> 16);
}
__device__ double uniform(u64 h, u64 stream) { return (double)(mix(h + stream) & 0xffffffULL) / 16777216.0; }
__device__ double smooth(double low, double high, double x) {
  double t = (x - low) / (high - low); t = t < 0. ? 0. : (t > 1. ? 1. : t);
  return t*t*(3. - 2.*t);
}
__device__ double noise(u64 seed, double y, double x, double spacing, u64 stream) {
  const u64 M = 0xffffffffULL;
  double gy = y / spacing, gx = x / spacing;
  double iy = floor(gy), ix = floor(gx);
  double fy = gy - iy, fx = gx - ix;
  double sy = fy*fy*(3. - 2.*fy), sx = fx*fx*(3. - 2.*fx);
  u64 r = (u64)(long long)iy & M, c = (u64)(long long)ix & M;
  double v00 = 2.*uniform(mix(mix(seed ^ (r & M)) ^ (c & M)), stream) - 1.;
  double v01 = 2.*uniform(mix(mix(seed ^ (r & M)) ^ ((c + 1) & M)), stream) - 1.;
  double v10 = 2.*uniform(mix(mix(seed ^ ((r + 1) & M)) ^ (c & M)), stream) - 1.;
  double v11 = 2.*uniform(mix(mix(seed ^ ((r + 1) & M)) ^ ((c + 1) & M)), stream) - 1.;
  double top = v00 + (v01 - v00)*sx;
  double bottom = v10 + (v11 - v10)*sx;
  return top + (bottom - top)*sy;
}
__device__ double sample(const double* f, int fw, long long iy, long long ix, double fy, double fx) {
  double top = f[iy*fw + ix]*(1. - fx) + f[iy*fw + ix + 1]*fx;
  double bottom = f[(iy + 1)*fw + ix]*(1. - fx) + f[(iy + 1)*fw + ix + 1]*fx;
  return top*(1. - fy) + bottom*fy;
}
extern "C" __global__ void impulses(
    const double* gy_f, const double* gx_f, const double* curv_f, const double* rough_f, int fw,
    long long a0, long long b0, int na, int nb, u64 seed, u64 patch_seed, double scale, double oy0, double ox0,
    double orient, double gentle, double steep_decay, double cap, double cut, double patch_low, double patch_high,
    double s0, double w0, double s1, double w1, double s2, double w2, double tail, double across, double along_gain,
    double sqrt3, double radial_norm, double oriented_power, float* out) {
  int j = blockIdx.x*blockDim.x + threadIdx.x, i = blockIdx.y*blockDim.y + threadIdx.y;
  if (i >= na || j >= nb) return;
  const u64 M = 0xffffffffULL;
  long long a = a0 + i, b = b0 + j;
  u64 h = mix(mix(seed ^ ((u64)a & M)) ^ ((u64)b & M));
  double jy = uniform(h, 1), jx = uniform(h, 2);
  double py = ((double)a + jy) / scale, px = ((double)b + jx) / scale;
  double y = py - oy0 - .5, x = px - ox0 - .5;
  double iy = floor(y), ix = floor(x), fy = y - iy, fx = x - ix;
  double gy = sample(gy_f, fw, (long long)iy, (long long)ix, fy, fx);
  double gx = sample(gx_f, fw, (long long)iy, (long long)ix, fy, fx);
  double curvature = sample(curv_f, fw, (long long)iy, (long long)ix, fy, fx);
  double roughness = sample(rough_f, fw, (long long)iy, (long long)ix, fy, fx);
  double slope = sqrt(gy*gy + gx*gx);
  double dy = gy / (slope + 1e-12), dx = gx / (slope + 1e-12);
  double steep = smooth(.15, .7, slope);
  double patch = w0*noise(patch_seed, py, px, s0, 11);
  patch = patch + w1*noise(patch_seed, py, px, s1, 12);
  patch = patch + w2*noise(patch_seed, py, px, s2, 13);
  double presence = smooth(patch_low, patch_high, patch + (.1*steep - .2*curvature));
  double gain = 1. - cut*(1. - presence);
  bool oriented = uniform(h, 3) < smooth(.02, .15, slope) * orient * gain;
  double along = across * (1. + along_gain*smooth(.05, .4, slope));
  double decay = (1. - steep)*gentle + steep*steep_decay;
  double amplitude = roughness * decay * (1. - .4*curvature);
  double ratio = amplitude / cap;
  amplitude = amplitude / sqrt(1. + ratio*ratio) * gain;
  double s = 2.*uniform(h, 5) - 1.;
  double weight = amplitude * ((1. - tail)*(sqrt3*s) + tail*(2.6457513110645907*(s*s*s)));
  long long n = (long long)na*nb, k = (long long)i*nb + j;
  out[k] = (float)jy; out[n + k] = (float)jx; out[2*n + k] = (float)dy; out[3*n + k] = (float)dx;
  out[4*n + k] = (float)(1. / along);
  out[5*n + k] = oriented ? (float)(weight / sqrt(along * across * oriented_power)) : 0.f;
  out[6*n + k] = oriented ? 0.f : (float)(weight / radial_norm);
}
''',
    'octave': r'''
extern "C" __global__ void octave(
    const float* jy, const float* jx, const float* ky, const float* kx, const float* inv_along,
    const float* weight, const float* radial, int columns,
    const int* ir, const int* ic, const float* fy, const float* fx, int h, int w, int reach,
    float inv_across, float inv_radial, float* out) {
  int x = blockIdx.x*blockDim.x + threadIdx.x, y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x >= w || y >= h) return;
  float total = 0.f;
  for (int di = -reach; di <= reach; ++di) {
    float py = fy[y] - (float)di;
    int row = (ir[y] + di) * columns + ic[x];
    for (int dj = -reach; dj <= reach; ++dj) {
      int i = row + dj;
      float oy = py - jy[i];
      float ox = (fx[x] - (float)dj) - jx[i];
      float u = (oy*ky[i] + ox*kx[i]) * inv_along[i];
      float v = (ox*ky[i] - oy*kx[i]) * inv_across;
      float q = 1.f - u*u; q = q < 0.f ? 0.f : q;
      float p = 1.f - v*v; p = p < 0.f ? 0.f : p;
      q = q*q; p = p*p;
      float v2 = v*v;
      float t = (oy*oy + ox*ox) * inv_radial;
      float k = 1.f - t; k = k < 0.f ? 0.f : k;
      k = k*k;
      float oriented = weight[i] * ((1.f - 26.f*v2) + (65.f*v2)*v2) * (p*p) * (q*q);
      float round = radial[i] * ((1.f - 14.f*t) + (28.f*t)*t) * (k*k);
      total = total + (oriented + round);
    }
  }
  out[y*w + x] = total;
}
''',
}
_KERNELS = {}


def _kernel(name, device):
    """Compiled CuPy kernel `name` for a CUDA device, or None (CPU, or CuPy/NVRTC missing)."""
    device = torch.device(device)
    if device.type != 'cuda':
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    if (name, index) not in _KERNELS:
        try:
            import cupy as cp
            with cp.cuda.Device(index):
                kernel = cp.RawKernel(_SOURCES[name], name, options=('--fmad=false',))
                kernel.compile()
            _KERNELS[name, index] = kernel
        except Exception:
            _KERNELS[name, index] = None
    return _KERNELS[name, index]


def _launch(kernel, shape, *args):
    """Run a 2D kernel over `shape` (rows, columns) on PyTorch's current stream."""
    import cupy as cp
    device = next(a.device for a in args if isinstance(a, torch.Tensor))
    index = device.index if device.index is not None else torch.cuda.current_device()
    converted = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            converted.append(cp.from_dlpack(arg))
        elif isinstance(arg, float):
            converted.append(cp.float64(arg))
        else:
            converted.append(arg)
    with cp.cuda.Device(index), cp.cuda.ExternalStream(torch.cuda.current_stream(device).cuda_stream):
        kernel(((shape[1] + 15)//16, (shape[0] + 15)//16), (16, 16), tuple(converted))


def _procedural_octave(world, source, level, i1, j1, i2, j2, fields, origin):
    """Sparse-convolution octave `source` sampled on the `level` grid, in metres."""
    return _convolve(world, source, level, i1, j1, i2, j2,
                     lambda a0, b0, a1, b1: _impulses(world, source, a0, b0, a1, b1, fields, origin))


def _offset_impulses(world, source, a0, b0, a1, b1):
    """Isotropic impulses of one coastline octave, in native cells.

    An isotropic band with RMS amplitude a at wavelength L turns a line by an
    RMS of about 2*pi*a/L/sqrt(2) radians; the octaves share COAST_ANGLE.
    """
    device = world.device
    a = torch.arange(a0, a1, device=device, dtype=torch.int64)[:, None]
    b = torch.arange(b0, b1, device=device, dtype=torch.int64)[None, :]
    seed = (int(getattr(world, 'seed', 0)) * 1000003 + source * 7919 + 5003) & _MASK
    h = _hash(_hash(_hash(torch.tensor(seed, device=device)) ^ (a & _MASK)) ^ (b & _MASK))
    octaves = -MIN_LOD - COAST_LAG
    turn = math.tan(math.radians(COAST_ANGLE)) / octaves**.5
    wavelength = 2**1.5 / 2**source
    amplitude = turn * wavelength * 2**.5 / (2*math.pi)
    radial = amplitude * (3**.5 * (2*_uniform(h, 5) - 1)) / (RADIAL_RADIUS**2 * _RADIAL_POWER)**.5
    zero = torch.zeros_like(radial)
    return [v.float() for v in (_uniform(h, 1), _uniform(h, 2), zero, zero, zero, zero, radial)]


def _coast_offset(world, level, i1, j1, i2, j2):
    """Coastline offset in native cells: octaves 1..level-COAST_LAG, each at least 11 px long."""
    total = None
    for source in range(1, level - COAST_LAG + 1):
        octave = _convolve(world, source, level, i1, j1, i2, j2,
                           lambda a0, b0, a1, b1: _offset_impulses(world, source, a0, b0, a1, b1))
        total = octave if total is None else total + octave
    return total


def _convolve(world, source, level, i1, j1, i2, j2, make_impulses):
    """Sum the kernels of octave `source`'s impulses at the `level` grid's cells."""
    shift = level - source
    device = torch.device(world.device)
    rows = torch.arange(i1, i2, device=device, dtype=torch.int64)
    cols = torch.arange(j1, j2, device=device, dtype=torch.int64)
    cell_r, cell_c = rows >> shift, cols >> shift
    a0, b0 = int(cell_r[0]) - REACH, int(cell_c[0]) - REACH
    impulses = make_impulses(a0, b0, int(cell_r[-1]) + REACH + 1, int(cell_c[-1]) + REACH + 1)
    # Position inside the cell: small dyadic numbers, exact in float32.
    fy = ((rows - (cell_r << shift)).double() + .5) / 2**shift
    fx = ((cols - (cell_c << shift)).double() + .5) / 2**shift
    inv_across, inv_radial = 1 / ACROSS_RADIUS, 1 / RADIAL_RADIUS**2
    kernel = _kernel('octave', device)
    if kernel is not None:
        h, w = i2 - i1, j2 - j1
        out = torch.empty((h, w), device=device, dtype=torch.float32)
        _launch(kernel, (h, w), *[v.contiguous() for v in impulses], np.int32(impulses[0].shape[1]),
                (cell_r - a0).int().contiguous(), (cell_c - b0).int().contiguous(),
                fy.float().contiguous(), fx.float().contiguous(), np.int32(h), np.int32(w), np.int32(REACH),
                np.float32(inv_across), np.float32(inv_radial), out)
        return out.double()

    jy, jx, dy, dx, inv_along, weight, radial = impulses
    fy, fx = fy.float()[:, None], fx.float()[None, :]
    ir, ic = (cell_r - a0)[:, None], (cell_c - b0)[None, :]
    # One batch per impulse row, summed in a fixed column order.
    ic = ic[None] + torch.arange(-REACH, REACH + 1, device=device)[:, None, None]
    fx = fx[None] - torch.arange(-REACH, REACH + 1, device=device, dtype=torch.float32)[:, None, None]
    total = None
    for di in range(-REACH, REACH + 1):
        r = ir + di
        oy = (fy - di) - jy[r, ic]
        ox = fx - jx[r, ic]
        ky, kx = dy[r, ic], dx[r, ic]
        u = (oy*ky + ox*kx) * inv_along[r, ic]
        v = (ox*ky - oy*kx) * inv_across
        q = (1 - u*u).clamp_min(0)
        p = (1 - v*v).clamp_min(0)
        q, p = q*q, p*p
        v2 = v*v
        t = (oy*oy + ox*ox) * inv_radial
        k = (1 - t).clamp_min(0)
        k = k*k
        terms = (weight[r, ic] * (1 - 26*v2 + 65*v2*v2) * (p*p) * (q*q)
                 + radial[r, ic] * (1 - 14*t + 28*t*t) * (k*k))
        for term in terms:
            total = term if total is None else total + term
    return total.double()


def _procedural_box(level, i1, j1, i2, j2):
    """Native cells covering every impulse of every procedural octave, plus bilinear taps."""
    boxes = []
    for source in range(DECODER_LEVELS + 1, level + 1):
        shift, scale = level - source, 2**source
        boxes.append((((i1 >> shift) - REACH) // scale - 1, ((j1 >> shift) - REACH) // scale - 1,
                      (((i2-1) >> shift) + REACH + 1) // scale + 2, (((j2-1) >> shift) + REACH + 1) // scale + 2))
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _procedural_detail(world, level, i1, j1, i2, j2, *, native=None, roughness=None, check=None):
    """Sum of procedural octaves DECODER_LEVELS+1 .. level, in metres.

    `native` and `roughness` (native_rms, decoder_rms, n0, m0) let a caller
    share its reads; by default they cover exactly this octave set.
    """
    n0, m0, n1, m1 = _procedural_box(level, i1, j1, i2, j2)
    if native is None:
        margin = SMOOTH_RADIUS + HALF_BAND_RADIUS
        native = _NativeWindow(world, n0-margin, m0-margin, n1+margin, m1+margin, check)
    if roughness is None:
        roughness = (*_roughness(world, native, n0, m0, n1, m1), n0, m0)
    native_rms, decoder_rms, lo, left = roughness
    crop = lambda field: field[n0-lo:n1-lo, m0-left:m1-left]
    fields = _terrain_fields(native, crop(native_rms), crop(decoder_rms), n0, m0, n1, m1)
    total = None
    for source in range(DECODER_LEVELS + 1, level + 1):
        octave = _procedural_octave(world, source, level, i1, j1, i2, j2, fields, (n0, m0))
        total = octave if total is None else total + octave
        if check is not None:
            check()
    return total


def _fractional_latents(world, row, col, size, stride, scale):
    """Read a sub-cell footprint without rounding its global origin to FP32.

    Positions refer to the decoder's 64x64 conditioning sample centres. Only
    local offsets become FP32, retaining centimetre spacing far from the origin.
    Normalize the fused source before interpolating its physical latent values.
    """
    lc = world.latent_compression
    count = size // lc
    row0 = (row * stride + lc / 2) / (lc * scale) - .5
    col0 = (col * stride + lc / 2) / (lc * scale) - .5
    i1, j1 = math.floor(row0), math.floor(col0)
    i2 = math.floor(row0 + (count - 1) / scale) + 2
    j2 = math.floor(col0 + (count - 1) / scale) + 2
    fused = world.latents[:, i1:i2, j1:j2].to(world.device, dtype=torch.float32)
    values = (fused[:-1] / fused[-1:])[None]
    offsets = torch.arange(count, device=values.device, dtype=torch.float32) / scale
    ys = (offsets + (row0 - i1) + .5) * (2 / (i2 - i1)) - 1
    xs = (offsets + (col0 - j1) + .5) * (2 / (j2 - j1)) - 1
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack((xx, yy), dim=-1)[None]
    value = F.grid_sample(values, grid, mode='bilinear', padding_mode='border',
                          align_corners=False)[0]
    return torch.cat((value, torch.ones_like(value[:1])), dim=0)


def _detail_field(world, level):
    fields = world.__dict__.setdefault('_terrain_refinement_fields', {})
    if level in fields:
        return fields[level]
    from terrain_inference import _decoder_batch

    size, stride = world.decoder_tile_size, world.decoder_tile_stride
    scale, lc = 2**level, world.latent_compression
    if size % lc:
        raise ValueError('Refinement requires aligned decoder conditioning')
    aligned = size % (lc * scale) == 0 and stride % (lc * scale) == 0
    latent_size, latent_stride = size // (lc * scale), stride // (lc * scale)
    scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, sigma_max=80, sigma_data=0.5)
    cache_device = world.device if world._terrain_profile.gpu_windows else 'cpu'
    weight = linear_weight_window(size, 'cpu', torch.float32).to(cache_device)
    times = [torch.atan(scheduler.sigmas[0] / scheduler.config.sigma_data)]

    def generate(ctxs):
        latents = []
        for ctx in ctxs:
            if aligned:
                row, col = ctx[1] * latent_stride, ctx[2] * latent_stride
                value = world.latents[:, row:row + latent_size, col:col + latent_size]
                # Keep the original four refinement levels byte-for-byte.
                latents.append(value.repeat_interleave(scale, -2).repeat_interleave(scale, -1))
            else:
                latents.append(_fractional_latents(world, ctx[1], ctx[2], size, stride, scale))
        return _decoder_batch(world, ctxs, latents, scheduler, weight, times, size, stride,
                              noise_seed=world.seed + level * 104729)

    tensor = InfiniteTensor(
        shape=(2, None, None), f=generate,
        output_window=TensorWindow(size=(2, size, size), stride=(2, stride, stride)),
        batch_size=1, device=cache_device, tile_store=world.tile_store,
        tensor_id=f'refinement_residual_{level}',
    )
    observer = getattr(world, '_terrain_window_scheduler', None)
    if observer is not None:
        observer._install_recursive(tensor)
    fields[level] = tensor
    return tensor


@torch.inference_mode()
def sample_refined(world, level, i1, j1, i2, j2, *, check=None):
    """Read fine-grid cells; level 0 is the native 30 m DEM (coherent coast signs).

    Integer coordinates refer to cell edges at 30 / 2**level metres. Every
    output cell is a fixed-order sum over its own source neighbourhood, so
    overlapping tile requests agree exactly, including at negative coordinates.
    """
    if not 0 <= level <= -MIN_LOD:
        raise ValueError('Unsupported refinement level')
    if any(not isinstance(v, int) for v in (i1, j1, i2, j2)):
        raise ValueError('Refinement coordinates must be integers')
    if i2 <= i1 or j2 <= j1 or (i2 - i1) * (j2 - j1) > 1024**2:
        raise ValueError('Refinement read exceeds bounded output budget')
    if check is not None:
        check()
    observer = getattr(world, '_terrain_window_scheduler', None)
    with observer.scope(check) if observer is not None else nullcontext():
        if level == 0:
            # The same coherent-sign native DEM as LOD 0-2 decoder tiles.
            from terrain_coastline import read_native
            return read_native(world, i1, j1, i2, j2).float()

        # Each level is evaluated directly from continuous surfaces: no
        # recursion, block means or nearest repeats, so no 30 m lattice.
        # One native read and one roughness field serve every neighbourhood.
        device = torch.device(world.device)
        scale = 2**level
        r0, r1, rows, fy = _axis(i1, i2, scale, device)
        c0, c1, cols, fx = _axis(j1, j2, scale, device)
        lo, left, hi, right = _base_bounds(level, i1, j1, i2, j2)
        lo, left, hi, right = min(lo, r0), min(left, c0), max(hi, r1+1), max(right, c1+1)
        if level > DECODER_LEVELS:
            box = _procedural_box(level, i1, j1, i2, j2)
            lo, left, hi, right = min(lo, box[0]), min(left, box[1]), max(hi, box[2]), max(right, box[3])
        margin = max(SPLINE_RADIUS, SMOOTH_RADIUS + HALF_BAND_RADIUS, 2*SMOOTH_RADIUS + 1)
        native = _NativeWindow(world, lo-margin, left-margin, hi+margin, right+margin, check)
        native_rms, decoder_rms = _roughness(world, native, lo, left, hi, right)
        # Composition in the checkpoint's signed-sqrt height space s. Only the
        # base decides land and sea; detail never changes a sign.
        result, coast = _coastal_base(native, level, i1, j1, i2, j2)
        if level > COAST_LAG and bool((coast > 0).any()):
            result = result + coast*_coast_offset(world, level, i1, j1, i2, j2)
        detail = None
        for source in range(1, min(level, DECODER_LEVELS)+1):
            band = _level_detail(world, source, level, i1, j1, i2, j2) * (0.5**source)
            detail = band if detail is None else detail + band
            if check is not None:
                check()
        # Bound the decoder octave's local RMS slope (cliff ringing) with a
        # smooth gain: a C2 B-spline of the native-grid gain, no harmonics.
        gain = _decoder_gain(decoder_rms[r0-lo:r1+1-lo, c0-left:c1+1-left])
        residual = detail * float(world.kwargs['residual_std'])
        result = _limited(result, residual * _evaluate(gain, _cubic_taps(rows, fy), _cubic_taps(cols, fx)))
        if level > DECODER_LEVELS:
            # Procedural octaves are metres; d(s|s|)/ds = 2|s|, floored at
            # |s| = 1 (1 m) where the limiter takes over near sea level.
            extra = _procedural_detail(world, level, i1, j1, i2, j2, native=native,
                                       roughness=(native_rms, decoder_rms, lo, left), check=check)
            result = _limited(result, extra / (2*(result.abs() + 1)))
        result = result * result.abs()
        if check is not None:
            check()
        return result.float().contiguous()

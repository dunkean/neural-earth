"""Per-pixel resource evaluator: NumPy reference of web/terrain_resources_gpu.js.

The browser evaluates resource layers in WGSL inside the renderer's shade
pass, from the tile DEM/hydrology already on the GPU plus coarse LOD-10
fields served once per world and mode. This module mirrors that evaluator
operation for operation (float32, uint32 wrapping hashes) for parity tests,
the canonical rock realization and measurements. Canonical budgets (stocks, tonnages,
integer allocations) stay in terrain_resources.py and are exact.

Display model (doc RESOURCE_LAYERS.md §16):
- continuous fields: parent LOD-10 surface sampled at a smoothly warped
  position, times terrain terms of the displayed LOD and a low-frequency
  dominated keyed noise; display conservation is approximate and measured;
- rocks: realized geology, argmax over classes of (biased log expected share
  + beta x large-wavelength noise, belt-forming classes elongated along the
  structural trend); the canonical cell shares are area shares of the same
  function, so the display agrees with them at every LOD;
- tectonic contexts (soft memberships): dominant context, warped contacts;
- month-like layers: one neighbouring cell's value, chosen by its bilinear weight.
All noise is keyed by (seed, layer, octave LOD, lattice cell) on integer
sample indices, independent of tiles and request order.
"""
from __future__ import annotations

import math

import numpy as np

AUTH_LOD = 10
AUTH_SIZE = 30. * 2 ** AUTH_LOD
F32 = np.float32
U32 = np.uint32

# Display profiles: log-linear terrain sensitivity, noise sigma, warp floor.
# Coefficients apply to: slope (high-passed, degrees), valley, ridge, north,
# dz (metres above the 6-sample Gaussian), river (0..1).
PROFILE_NAMES = ('neutral', 'arable', 'pasture', 'forest', 'wet', 'rock', 'lowflat', 'cold', 'season',
                 'runoff', 'mineral', 'marine')
PROFILES = {
    'neutral': ((0., .35, -.25, 0., 0., 0.), .20),
    'arable':  ((-1 / 6, 1.0, -.8, 0., 0., .3), .18),
    'pasture': ((-1 / 14, .6, -.3, 0., 0., .2), .20),
    'forest':  ((-1 / 40, .6, -.5, .3, 0., .2), .22),
    'wet':     ((-1 / 4, 2.2, -1., 0., 0., 1.6), .25),
    'rock':    ((1 / 8, -1., 1.2, 0., 0., -.5), .20),
    'lowflat': ((-1 / 5, 1.4, -.6, 0., 0., .8), .22),
    'cold':    ((1 / 30, 0., .3, .5, .012, 0.), .10),
    'season':  ((0., 0., 0., -.2, -.006, 0.), .08),
    'runoff':  ((1 / 15, .5, 0., 0., 0., .5), .15),
    'mineral': ((1 / 25, -.3, .6, 0., 0., 0.), .08),
    'marine':  ((0., 0., 0., 0., 0., 0.), .25),
}
# Finest octave shown at a LOD: wavelength >= 4 samples (2 would alias into pixel grain).
OCTAVE_FLOOR = 2
KINDS = dict(ramp=0, log=1, mix=2, classes=3, cyclic=4, points=5, stock=6)
SHAPES = dict(vein=0, body=1, strata=2, placer=3)


# ---------------------------------------------------------------------------
# Hashing and value noise on integer sample lattices (bit-exact with WGSL)

def lowbias32(x):
    x = np.asarray(x, U32).copy()
    with np.errstate(over='ignore'):
        x ^= x >> U32(16)
        x *= U32(0x7feb352d)
        x ^= x >> U32(15)
        x *= U32(0x846ca68b)
        x ^= x >> U32(16)
    return x


def as_u32(i):
    return (np.asarray(i, np.int64) & 0xffffffff).astype(U32)


def rhash(i, j, salt):
    with np.errstate(over='ignore'):
        return lowbias32((as_u32(i) * U32(0x9e3779b1)) ^ lowbias32(as_u32(j) + U32(0x85ebca77)) ^ U32(salt))


def salt_of(seed32, layer, octave, comp=0):
    with np.errstate(over='ignore'):
        v = (U32(seed32) + U32(layer) * U32(0x632be5ab) + U32(octave + 64) * U32(0x27d4eb2d) + U32(comp) * U32(0x165667b1))
    return int(lowbias32(np.array(v, U32)))


def node_value(i, j, salt):
    return ((rhash(i, j, salt) >> U32(8)).astype(F32) * F32(1 / 16777216.)) * F32(2) - F32(1)


def _lowbias_inplace(x):
    x ^= x >> U32(16)
    x *= U32(0x7feb352d)
    x ^= x >> U32(15)
    x *= U32(0x846ca68b)
    x ^= x >> U32(16)
    return x


def _unit(h):
    return ((h >> U32(8)).astype(F32) * F32(1 / 16777216.)) * F32(2) - F32(1)


def _quad(ix, iy, ux, uy, salt):
    """Bilinear blend (weights already faded) of the four lattice nodes around
    (ix, iy): same values as node_value, with six hashes instead of eight."""
    ui = (np.asarray(ix, np.int64) & 0xffffffff).astype(U32)
    uj = (np.asarray(iy, np.int64) & 0xffffffff).astype(U32)
    s = U32(salt)
    with np.errstate(over='ignore'):
        a0 = ui * U32(0x9e3779b1)
        ui += U32(1)
        a1 = ui * U32(0x9e3779b1)
        b0 = _lowbias_inplace(uj + U32(0x85ebca77))
        uj += U32(1)
        b1 = _lowbias_inplace(uj + U32(0x85ebca77))
        n00 = _unit(_lowbias_inplace(a0 ^ b0 ^ s))
        n10 = _unit(_lowbias_inplace(a1 ^ b0 ^ s))
        n01 = _unit(_lowbias_inplace(a0 ^ b1 ^ s))
        n11 = _unit(_lowbias_inplace(a1 ^ b1 ^ s))
    a = n00 * (F32(1) - ux) + n10 * ux
    b = n01 * (F32(1) - ux) + n11 * ux
    return (a * (F32(1) - uy) + b * uy).astype(F32)


def fade(t):
    t = t.astype(F32)
    return t * t * t * (t * (t * F32(6) - F32(15)) + F32(10))


def lattice(g, lod, octave):
    """Integer cell index and fraction of global sample g at an octave >= lod."""
    k = octave - lod
    g = np.asarray(g, np.int64)
    if k <= 0:
        return g, np.full(g.shape, .5, F32)
    return g >> k, (((g & ((1 << k) - 1)).astype(F32) + F32(.5)) * F32(1. / (1 << k)))


def octave(gi, gj, lod, oct_, salt, wx=None, wy=None):
    """Quintic value noise of octave oct_ at samples (gi, gj); warp in octave units."""
    ix, fx = lattice(gi, lod, oct_)
    iy, fy = lattice(gj, lod, oct_)
    if wx is not None:
        qx, qy = fx + wx.astype(F32), fy + wy.astype(F32)
        fl_x, fl_y = np.floor(qx), np.floor(qy)
        ix, iy = ix + fl_x.astype(np.int64), iy + fl_y.astype(np.int64)
        fx, fy = (qx - fl_x).astype(F32), (qy - fl_y).astype(F32)
    return _quad(ix, iy, fade(fx), fade(fy), salt)


def lam(o):
    return F32(30. * 2. ** o)


def warp(gi, gj, lod, seed32, layer, lo, hi=13, amp=.3):
    """Multi-scale domain warp in metres: octaves max(lo, lod+2) .. hi."""
    wx = np.zeros(np.broadcast(gi, gj).shape, F32)
    wy = np.zeros_like(wx)
    for o in range(max(lo, lod + OCTAVE_FLOOR), hi + 1):
        a = F32(amp) * lam(o)
        wx = wx + a * octave(gi, gj, lod, o, salt_of(seed32, layer, o, 1))
        wy = wy + a * octave(gi, gj, lod, o, salt_of(seed32, layer, o, 2))
    return wx, wy


def fbm(gi, gj, lod, seed32, layer, lo, hi, decay=.8, wx=None, wy=None):
    """Normalised sum of octaves lo..hi (amplitude 2^(decay (o - hi)))."""
    total = np.zeros(np.broadcast(gi, gj).shape, F32)
    norm = 0.
    for o in range(max(lo, lod + OCTAVE_FLOOR), hi + 1):
        a = F32(2. ** (decay * (o - hi)))
        total = total + a * octave(gi, gj, lod, o, salt_of(seed32, layer, o),
                                   None if wx is None else wx / lam(o), None if wy is None else wy / lam(o))
        norm += float(a) ** 2
    return (total / F32(max(math.sqrt(norm), 1e-6) * .45)).astype(F32)


# ---------------------------------------------------------------------------
# Parent field sampling (LOD-10 texture: K layers, wrap in longitude)

class Field:
    def __init__(self, data, i0, j0, bounds=None):
        self.data = np.asarray(data, F32)          # (K, H, W)
        self.i0, self.j0 = int(i0), int(j0)
        self.bounds = None if bounds is None else np.asarray(bounds, F32) / F32(AUTH_SIZE)
        self.k, self.h, self.w = self.data.shape

    def cell_pos(self, g, lod, w):
        """Continuous position (cell units, cell centres at integers) of samples + warp metres."""
        g = np.asarray(g, np.int64)
        if lod <= AUTH_LOD:
            k = AUTH_LOD - lod
            base = (g >> k).astype(np.int64)
            frac = ((g & ((1 << k) - 1)).astype(F32) + F32(.5)) * F32(1. / (1 << k)) - F32(.5)
        else:
            pos = ((g.astype(F32) + F32(.5)) * F32(2 ** (lod - AUTH_LOD)) - F32(.5)).astype(F32)
            base = np.floor(pos).astype(np.int64)
            frac = (pos - np.floor(pos)).astype(F32)
        q = frac + (w * F32(1. / AUTH_SIZE)).astype(F32)
        fl = np.floor(q)
        return base + fl.astype(np.int64), (q - fl).astype(F32)

    def axis_pos(self, g, lod, w, axis):
        c, f = self.cell_pos(g, lod, w)
        if self.bounds is None:
            return c, f
        lo, hi = self.bounds[axis], self.bounds[axis + 2]
        first, last = (self.i0, self.i0 + self.w - 1) if axis == 0 else (self.j0, self.j0 + self.h - 1)
        a, b = (F32(first + 1) + lo) * F32(.5), (F32(last) + hi) * F32(.5)
        x = c.astype(F32) + f + F32(.5)
        x = (x - lo) % (hi - lo) + lo if axis == 0 else np.clip(x, a, b)
        c = np.clip(np.floor(x - F32(.5)).astype(np.int64), first, last - 1)
        left = np.where(c == first, a, c.astype(F32) + F32(.5))
        right = np.where(c + 1 == last, b, c.astype(F32) + F32(1.5))
        f = (x - left) / (right - left)
        if axis == 0:
            seam = (x < a) | (x >= b)
            f = np.where(seam, (np.where(x < a, x + hi - lo, x) - b) / (a + hi - lo - b), f)
            c = np.where(seam, last, c)
        return c, f.astype(F32)

    def seam(self, gi, lod):
        if self.bounds is None:
            return F32(1)
        x = (np.asarray(gi, F32) + F32(.5)) * F32(2. ** (lod - AUTH_LOD))
        lo, hi = self.bounds[0], self.bounds[2]
        x = (x - lo) % (hi - lo)
        return smoothstep(0, 2, np.minimum(x, hi - lo - x))

    def bilinear(self, layer, gi, gj, lod, wx, wy):
        ci, fx = self.axis_pos(gi, lod, wx, 0)
        cj, fy = self.axis_pos(gj, lod, wy, 1)
        c0, c1 = (ci - self.i0) % self.w, (ci + 1 - self.i0) % self.w
        r0, r1 = np.clip(cj - self.j0, 0, self.h - 1), np.clip(cj + 1 - self.j0, 0, self.h - 1)
        d = self.data[layer]
        a = d[r0, c0] * (F32(1) - fx) + d[r0, c1] * fx
        b = d[r1, c0] * (F32(1) - fx) + d[r1, c1] * fx
        return (a * (F32(1) - fy) + b * fy).astype(F32)

    def nearest(self, layer, gi, gj, lod, wx, wy):
        ci, fx = self.axis_pos(gi, lod, wx, 0)
        cj, fy = self.axis_pos(gj, lod, wy, 1)
        ci, cj = ci + (fx >= .5), cj + (fy >= .5)
        return self.data[layer][np.clip(cj - self.j0, 0, self.h - 1), (ci - self.i0) % self.w]


# ---------------------------------------------------------------------------
# Terrain features (same definitions as the shade pass)

def smoothstep(a, b, x):
    t = np.clip((np.asarray(x, F32) - F32(a)) / F32(b - a), 0, 1).astype(F32)
    return t * t * (F32(3) - F32(2) * t)


def features(height, gradient, gradient_large, small, large, resolution, river):
    """height, small/large Gaussians (renderer blur, metres); gradients dz/dx, dz/dy."""
    gx, gy = gradient
    slope = np.degrees(np.arctan(np.hypot(gx, gy))).astype(F32)
    slope_broad = np.degrees(np.arctan(np.hypot(*gradient_large))).astype(F32)
    tpi = ((small - large) / F32(6. * resolution)).astype(F32)
    norm = np.maximum(np.hypot(gx, gy), F32(1e-6))
    north = (-gy / norm * smoothstep(1, 6, slope)).astype(F32)
    return dict(slope=(slope - slope_broad).astype(F32), valley=smoothstep(0, -.10, tpi), ridge=smoothstep(0, .10, tpi),
                north=north, dz=(height - large).astype(F32), river=np.asarray(river, F32), raw_slope=slope)


def terrain_terms(feat, coef):
    keys = ('slope', 'valley', 'ridge', 'north', 'dz', 'river')
    t = np.zeros_like(feat['valley'])
    for k, c in zip(keys, coef):
        if c:
            t = t + F32(c) * feat[k]
    return np.clip(t, -3, 3).astype(F32)


# ---------------------------------------------------------------------------
# Evaluation of one mode at samples

def season_layers(season, k):
    m = F32(season) * F32(12) - F32(.5)
    m0 = math.floor(float(m))
    t = F32(float(m) - m0)
    return m0 % k, (m0 + 1) % k, t


def evaluate_continuous(pkg, gi, gj, lod, feat):
    """Value of a ramp/log mode at samples (feat may be None: no terrain terms)."""
    seed32, layer = pkg['seed32'], pkg['layer']
    field = pkg['field']
    wx, wy = warp(gi, gj, lod, seed32, layer * 4 + 1, 8, 10, .25)
    seam = field.seam(gi, lod)
    wx, wy = wx * seam, wy * seam
    if pkg['seasonal']:
        a, b, t = season_layers(pkg['season'], field.k)
        p = field.bilinear(a, gi, gj, lod, wx, wy) * (F32(1) - t) + field.bilinear(b, gi, gj, lod, wx, wy) * t
    else:
        p = field.bilinear(0, gi, gj, lod, wx, wy)
    coef, sigma = PROFILES[pkg['profile']]
    # Low-frequency variation only (3.8-15 km); finer detail comes from the terrain terms.
    n = fbm(gi, gj, lod, seed32, layer * 4 + 2, 7, 9, .8)
    terms = terrain_terms(feat, coef) if feat is not None else np.zeros_like(n)
    # Terrain terms fade from LOD 1 to 5: a coarse DEM texel's roughness is not landform.
    z = terms * (F32(1) - F32(.75) * smoothstep(1, 5, np.float32(lod))) + F32(sigma) * n * seam
    if pkg['bounded']:
        lo, hi = pkg.get('bounds', pkg['range'])     # physical bounds, not the colour range
        q = np.clip((p - F32(lo)) / F32(hi - lo), F32(1e-4), F32(1 - 1e-4))
        v = F32(lo) + F32(hi - lo) / (F32(1) + np.exp(-(np.log(q / (F32(1) - q)) + z)))
        v = np.where(p <= F32(lo), F32(lo), v)
    else:
        v = np.maximum(p, F32(0)) * np.exp(z - F32(.5) * F32(sigma) ** 2)
    return v.astype(F32)


# Strike-aligned noise: four integer orientations of the sample lattice
# (a, b) = (c0 gi + c1 gj, c2 gi + c3 gj + c4); the a axis is stretched x2
# (one octave coarser) and runs along PHI in the (i east, j south) frame.
DIRECTIONS = ((1, 0, 0, 1, 0), (0, 1, -1, 0, -1), (1, 1, -1, 1, 0), (1, -1, 1, 1, 0))
PHI = (0., math.pi / 2, math.pi / 4, 3 * math.pi / 4)
ZONE_WARP = (11, .15)        # contact warp: octaves up to 61 km, 0.15 wavelength
STRETCH = 1


def octave_stretched(a, b, lod, oct_, salt, wa, wb, stretch=STRETCH):
    """Value noise with the a axis 2^stretch times coarser; warp in metres along (a, b)."""
    ia, fa = lattice(a, lod, oct_ + stretch)
    ib, fb = lattice(b, lod, oct_)
    qa, qb = fa + (wa / lam(oct_ + stretch)).astype(F32), fb + (wb / lam(oct_)).astype(F32)
    la, lb = np.floor(qa), np.floor(qb)
    ia, ib = ia + la.astype(np.int64), ib + lb.astype(np.int64)
    fa, fb = (qa - la).astype(F32), (qb - lb).astype(F32)
    return _quad(ia, ib, fade(fa), fade(fb), salt)


def fbm_aligned(gi, gj, lod, seed32, layer, lo, hi, decay, wx, wy, theta):
    """fbm elongated along the local strike theta: blend of the two nearest orientations.
    Each orientation is evaluated only where its weight is positive (bit-identical
    to evaluating it everywhere and multiplying by zero)."""
    gi, gj, wx, wy, theta = np.broadcast_arrays(np.asarray(gi, np.int64), np.asarray(gj, np.int64),
                                                np.asarray(wx, F32), np.asarray(wy, F32), np.asarray(theta, F32))
    total = np.zeros(gi.shape, F32)
    weight2 = np.zeros_like(total)
    octaves = range(max(lo, lod + OCTAVE_FLOOR), hi + 1)
    norm = math.sqrt(sum(float(F32(2. ** (decay * (o - hi)))) ** 2 for o in octaves)) if len(octaves) else 0.
    for d, (c0, c1, c2, c3, c4) in enumerate(DIRECTIONS):
        w = np.maximum(np.cos(F32(2) * (theta - F32(PHI[d]))), F32(0)).astype(F32)
        m = w > 0
        if not m.any():
            continue
        g, h, u, v, wm = gi[m], gj[m], wx[m], wy[m], w[m]
        a, b = c0 * g + c1 * h, c2 * g + c3 * h + c4
        wa, wb = (F32(c0) * u + F32(c1) * v).astype(F32), (F32(c2) * u + F32(c3) * v).astype(F32)
        n = np.zeros(a.shape, F32)
        for o in octaves:
            n = n + F32(2. ** (decay * (o - hi))) * octave_stretched(a, b, lod, o, salt_of(seed32, layer, o, 3 + d), wa, wb)
        total[m] = total[m] + wm * (n / F32(max(norm, 1e-6) * .45))
        weight2[m] = weight2[m] + wm * wm
    return (total / np.sqrt(np.maximum(weight2, F32(1e-6)))).astype(F32)


def zone_warp(pkg, gi, gj, lod, warp_floor=None):
    lo = lod + 1 if warp_floor is None else warp_floor
    wx, wy = warp(gi, gj, lod, pkg['seed32'], pkg['layer'] * 4 + 1, lo, ZONE_WARP[0], ZONE_WARP[1])
    seam = pkg['field'].seam(gi, lod)
    return wx * seam, wy * seam


# Realized geology: class = argmax_k (log E_k + b1 * N_g(k) + b2 * N_k), where
# E_k is the expected share of class k (Orogen contexts x lithology table,
# LOD-10 field, warped bilinear), N_g a province noise shared by the classes
# of a group (plutonic, volcanic, metamorphic, sedimentary; octaves 12-15,
# 123-983 km) and N_k a class noise (octaves 11-14, 61-491 km). Belt-forming
# groups and classes use noise elongated along the structural trend. The
# same function gives the canonical cell shares (terrain_resources, on LOD-8
# samples) and the display at every LOD (GPU).
REALIZED_GROUP_OCTAVES = (12, 15)
REALIZED_CLASS_OCTAVES = (11, 14)
REALIZED_CLASS_DECAY = 1.
REALIZED_BETA = (1.6, .7)         # province (group) and class noise weights


def realized_class(pkg, gi, gj, lod, warp_floor=None):
    """Class index at samples; pkg field layers: K biased log-expected shares, cos 2θ, sin 2θ.
    pkg: groups (group index per class), group_aligned, aligned (per class), beta (b1, b2)."""
    seed32, layer, field = pkg['seed32'], pkg['layer'], pkg['field']
    kcls = pkg['classes']
    b1, b2 = (F32(b) for b in pkg.get('beta', REALIZED_BETA))
    wx, wy = zone_warp(pkg, gi, gj, lod, warp_floor)
    c2 = field.bilinear(kcls, gi, gj, lod, wx, wy)
    s2 = field.bilinear(kcls + 1, gi, gj, lod, wx, wy)
    theta = (F32(.5) * np.arctan2(s2, c2)).astype(F32)

    def noise(salt, octaves, decay, aligned):
        o0, o1 = octaves
        return (fbm_aligned(gi, gj, lod, seed32, salt, o0, o1, decay, wx, wy, theta) if aligned
                else fbm(gi, gj, lod, seed32, salt, o0, o1, decay, wx, wy)) * field.seam(gi, lod)

    province = [noise(layer * 64 + 48 + g, REALIZED_GROUP_OCTAVES, 1., a) for g, a in enumerate(pkg['group_aligned'])]
    best = np.full(np.broadcast(gi, gj).shape, -np.inf, F32)
    out = np.zeros(best.shape, np.int32)
    for k in range(kcls):
        n = noise(layer * 64 + 32 + k, REALIZED_CLASS_OCTAVES, REALIZED_CLASS_DECAY, pkg['aligned'][k])
        v = (field.bilinear(k, gi, gj, lod, wx, wy) + b1 * province[pkg['groups'][k]] + b2 * n).astype(F32)
        out = np.where(v > best, k, out)
        best = np.maximum(best, v)
    return out


def evaluate_dominant(pkg, gi, gj, lod):
    """Generalised class at small scales: the largest parent share (warped bilinear)."""
    wx, wy = warp(gi, gj, lod, pkg['seed32'], pkg['layer'] * 4 + 1, lod + 1, ZONE_WARP[0], ZONE_WARP[1])
    seam = pkg['field'].seam(gi, lod)
    wx, wy = wx * seam, wy * seam
    best = np.full(np.broadcast(gi, gj).shape, -1., F32)
    out = np.zeros(best.shape, np.int32)
    for k in range(pkg['classes']):
        v = pkg['field'].bilinear(k, gi, gj, lod, wx, wy)
        out = np.where(v > best, k, out)
        best = np.maximum(best, v)
    return out


# Uniformising knots of fbm(octaves 8-10, decay 0.7): quantiles at k/16
# (measured on 3.2 M samples, symmetrised), tails linear to +/-3.5.
SELECT_KNOTS = (-3.5, -1.5528, -1.1966, -0.9383, -0.7218, -0.5275, -0.346, -0.1713, 0.0,
                0.1713, 0.346, 0.5275, 0.7218, 0.9383, 1.1966, 1.5528, 3.5)


def uniformize(n):
    """Piecewise-linear map of the selection noise to ~U(0, 1)."""
    n = np.asarray(n, F32)
    knots = np.asarray(SELECT_KNOTS, F32)
    k = np.clip(np.searchsorted(knots, n, side='right') - 1, 0, 15)
    lo, hi = knots[k], knots[k + 1]
    return np.clip((k.astype(F32) + (n - lo) / (hi - lo)) / F32(16), F32(0), F32(1)).astype(F32)


def select_cell(pkg, gi, gj, lod, wx, wy):
    """One of the four LOD-10 cells around each sample, chosen with probability
    equal to its bilinear weight by comparing a smooth uniformised noise with
    the cumulative weights: each cell rules one cell-area in expectation, in
    organic blobs, and nothing is blended (categorical values such as months)."""
    field = pkg['field']
    zero = np.zeros(np.broadcast(gi, gj).shape, F32)
    ci, fx = field.axis_pos(gi, lod, zero, 0)
    cj, fy = field.axis_pos(gj, lod, zero, 1)
    u = uniformize(fbm(gi, gj, lod, pkg['seed32'], pkg['layer'] * 16 + 8, 8, 10, .7, wx, wy) * field.seam(gi, lod))
    w00 = (F32(1) - fx) * (F32(1) - fy)
    w10 = fx * (F32(1) - fy)
    w01 = (F32(1) - fx) * fy
    a, b, c = w00, w00 + w10, w00 + w10 + w01
    dx = np.where(u < a, 0, np.where(u < b, 1, np.where(u < c, 0, 1)))
    dy = np.where(u < b, 0, 1)
    return ci + dx, cj + dy


def evaluate_cyclic(pkg, gi, gj, lod):
    """Month-like layers: value of a selected neighbouring cell (no blending of months)."""
    wx, wy = warp(gi, gj, lod, pkg['seed32'], pkg['layer'] * 4 + 1, max(lod + 1, -5), 10, .25)
    seam = pkg['field'].seam(gi, lod)
    wx, wy = wx * seam, wy * seam
    ci, cj = select_cell(pkg, gi, gj, lod, wx, wy)
    f = pkg['field']
    return f.data[0][np.clip(cj - f.j0, 0, f.h - 1), (ci - f.i0) % f.w]


# ---------------------------------------------------------------------------
# Deposits (shapes and kernels)

def deposit_contribution(dep, gi, gj, lod, valley, bounds=None, screen_mpp=0.):
    """Alpha of the body, halo and gossan rim, and colour index, of one deposit.

    dep: dict(di, dj (LOD-0 cell indices), a, b (m), theta, shape, gossan).
    """
    r = F32(30. * 2. ** lod)
    if lod >= 0:
        dx = ((np.asarray(gi, np.int64) << lod) - dep['di']).astype(F32) * F32(30) + F32(15. * (2 ** lod) - 15.)
        dy = ((np.asarray(gj, np.int64) << lod) - dep['dj']).astype(F32) * F32(30) + F32(15. * (2 ** lod) - 15.)
    else:
        s = -lod
        dx = (np.asarray(gi, np.int64) - (dep['di'] << s)).astype(F32) * r + r * F32(.5) - F32(15)
        dy = (np.asarray(gj, np.int64) - (dep['dj'] << s)).astype(F32) * r + r * F32(.5) - F32(15)
    if bounds is not None:
        span = F32(bounds[2] - bounds[0])
        dx = dx - np.floor(dx / span + F32(.5)) * span
    a, b = F32(dep['a']), F32(dep['b'])
    c, s_ = F32(math.cos(dep['theta'])), F32(math.sin(dep['theta']))
    u, v = dx * c + dy * s_, -dx * s_ + dy * c
    dist = np.sqrt(dx * dx + dy * dy)
    rpx = F32(np.clip(dep['symbol'], 3., 7.)) * F32(max(1., screen_mpp/float(r)))
    disc = np.clip(rpx + F32(.5) - dist/r, 0, 1)
    disc_outer = np.clip(rpx + F32(1.5) - dist/r, 0, 1)
    curved_v = v - b * F32(.28) * np.sin(F32(3) * u / np.maximum(a, F32(1)))
    q = np.sqrt((u/a)**2 + (curved_v/b)**2)
    body = np.clip((F32(1)-q)*b/r + F32(.5), 0, 1)
    outer = np.clip((F32(1)-q)*b/r + F32(1.5), 0, 1)
    if dep['shape'] == SHAPES['vein']:
        body = np.clip((b+F32(.5)*r-np.abs(curved_v))/r, 0, 1)*np.clip((a+F32(.5)*r-np.abs(u))/r, 0, 1)
        outer = np.clip((b+F32(1.5)*r-np.abs(curved_v))/r, 0, 1)*np.clip((a+F32(1.5)*r-np.abs(u))/r, 0, 1)
    elif dep['shape'] == SHAPES['placer']:
        body = body * np.clip(F32(.6)+F32(.4)*valley, F32(.4), F32(1))
    fade = smoothstep(rpx, F32(2)*rpx, np.sqrt(a*b)/r)
    body = disc*(F32(1)-fade)+body*fade
    outer = disc_outer*(F32(1)-fade)+outer*fade
    return body.astype(F32), np.zeros_like(body), np.maximum(F32(0),outer-body).astype(F32)


def deposit_color(deps, gi, gj, lod, valley, palette, bounds=None, base=(.5,.5,.5), screen_mpp=0.):
    """Mass-priority composite, independent of the spatial bucket traversal."""
    rgb = np.array(base, np.float32)
    mass = -1.
    for d in deps:
        body, _, ring = deposit_contribution(d, gi, gj, lod, valley, bounds, screen_mpp)
        if float(body+ring) > 0 and d['mass'] > mass:
            mass = d['mass']
            rgb = np.asarray(base, np.float32)*(1-ring)+np.array([.035,.045,.055], np.float32)*ring
            rgb = rgb*(1-body)+np.asarray(palette[min(d['color'],31)], np.float32)*body
    return rgb


def stock_radii(a, b, lod):
    """Compact district -> cluster -> footprint support, metres (display only)."""
    r = F32(30. * 2. ** lod)
    district = F32(135000) * smoothstep(2, 6, F32(lod))
    return max(F32(2) * F32(a), F32(3) * r, district), max(F32(2) * F32(b), F32(3) * r, district / F32(3))


def stock_normalization(dj, theta, su, sv, bounds=None, offset_y=0.):
    """Integral against spherical area, including clipped polar support.

    The planar kernel is 3/(pi*a*b)*(1-q^2)^2 on q<1; its y marginal is
    16/(5*pi)*(1-t^2)^(5/2). Eight-point Gauss-Legendre integrates the marginal
    times cos(latitude), with the same float32 operations as WGSL.
    """
    if bounds is None:
        return F32(1)
    y = F32(dj) * F32(30) + F32(15) + F32(offset_y)
    radius = F32(bounds[2] - bounds[0]) / F32(6.2831853)
    cy = F32(bounds[1] + bounds[3]) * F32(.5)
    sy = np.sqrt((su * F32(math.sin(theta))) ** 2 + (sv * F32(math.cos(theta))) ** 2).astype(F32)
    lo, hi = max(F32(-1), (F32(bounds[1]) - y) / sy), min(F32(1), (F32(bounds[3]) - y) / sy)
    mid, half = (lo + hi) * F32(.5), (hi - lo) * F32(.5)
    out = F32(0)
    for node, weight in ((.183434642, .362683783), (.525532410, .313706646), (.796666477, .222381034), (.960289856, .101228536)):
        for sign in (-1, 1):
            t = mid + F32(sign * node) * half
            z = max(F32(0), F32(1) - t * t)
            out += F32(weight) * z * z * np.sqrt(z) * max(F32(0), np.cos((y + sy * t - cy) / radius))
    return max(F32(1e-8), out * half * F32(16 / (5 * math.pi)))


def stock_lobes(d, lod, bounds=None):
    """Positive partition of one mass into curved, overlapping soft lobes.

    Offsets and widths are keyed by the site, along its structural azimuth.
    Each lobe integrates to its fixed mass share, including polar clipping.
    """
    su, sv = stock_radii(d['a'], d['b'], lod)
    phase = node_value(d['di'], d['dj'], 1709) * F32(3)
    bend = F32(np.clip(d.get('access', 0), -1, 1))
    c, s = F32(math.cos(d['theta'])), F32(math.sin(d['theta']))
    for k, weight in enumerate((.13, .23, .28, .23, .13)):
        t = F32(k - 2) * F32(.33)
        u = t * su
        v = sv * (F32(.45)*np.sin(phase+F32(2.2)*F32(k)) + bend*t*t)
        ox, oy = u*c-v*s, u*s+v*c
        if bounds is not None:
            center = F32(d['dj'])*F32(30)+F32(15)
            oy = np.clip(center+oy, F32(bounds[1])+F32(1), F32(bounds[3])-F32(1))-center
        a, b = su*F32(.40), sv*(F32(.50)+F32(.15)*np.cos(phase+F32(k)))
        yield F32(ox), F32(oy), F32(a), F32(b), F32(weight)


def stock_density(deps, gi, gj, lod, bounds=None):
    """Stock per spherical km2: sum of smooth, normalized compact kernels.

    Positive lobes partition the mass exactly; their quadrature normalization
    includes spherical area. Fine relief affects appearance, never this density.
    """
    r = F32(30. * 2. ** lod)
    out = np.zeros(np.broadcast(gi, gj).shape, F32)
    for d in deps:
        dx = (np.asarray(gi, np.float64) + .5) * float(r) - (d['di'] + .5) * 30
        dy = (np.asarray(gj, np.float64) + .5) * float(r) - (d['dj'] + .5) * 30
        dx, dy = np.asarray(dx, F32), np.asarray(dy, F32)
        if bounds is not None:
            span = F32(bounds[2] - bounds[0])
            dx -= np.floor(dx / span + F32(.5)) * span
        c, s = F32(math.cos(d['theta'])), F32(math.sin(d['theta']))
        for ox, oy, su, sv, weight in stock_lobes(d, lod, bounds):
            lx, ly = dx-ox, dy-oy
            u, v = lx*c+ly*s, -lx*s+ly*c
            q = np.maximum(F32(0), F32(1) - u*u/(su*su) - v*v/(sv*sv))
            norm = stock_normalization(d['dj'], d['theta'], su, sv, bounds, oy)
            out += F32(d['mass']) * weight * F32(1e6) * F32(3 / math.pi) / (su * sv * norm) * q * q
    return out.astype(F32)


def stock_display(pkg, deps, gi, gj, lod):
    field = pkg.get('field')
    bounds = None if field is None or field.bounds is None else field.bounds * F32(AUTH_SIZE)
    return stock_density(deps, gi, gj, lod, bounds)

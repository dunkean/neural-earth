# SPDX-License-Identifier: GPL-3.0-only
"""Hydrology geometry: river vectors, lake shores, water fields and surface envelopes.

NumPy/SciPy functions (with a few sequential Numba kernels) on the record
interface of the hydrology core; no server, Torch or CUDA imports. Method note: ``docs/methods/hydrology-geometry.md``.

Conventions
-----------
* Map metres, x east, y south. ``bounds = (x0, y0, x1, y1)`` are cell-edge
  extents (as in ``terrain_coarse.CoarseGrid``); cell ``(r, c)`` has its centre
  at ``(x0 + (c + .5) dx, y0 + (r + .5) dy)``.
* Widths, depths, wavelengths and lengths of rivers are physical metres. With
  ``sphere_radius_m`` the east-west map scale ``cos(latitude)`` is applied
  (latitude ``= -y / R``); without it the map is treated as a plane.
* Reach records: ``id, source, target, basin_id, xy (N, 2), water_m (N),
  flow_m3s, area_m2, downstream_id``; optional ``pins`` (indices of fixed
  portals/nodes inside ``xy``), ``corridor_m`` and ``climate``.
* Lake records: ``id, level_m, closed, area_m2, cells`` (flat indices).
* Everything random is indexed by ``(seed, reach id, band)`` and continuous
  physical reach stations. Display LOD, tile origin and request order never
  enter a seed; tolerances only change how many vertices sample the same curve.

The bend generator adapts ``hydrology_meanders.rs`` from the local Magna Urbis
``city_generator`` (HEAD 1d245aa, GPL-3.0-only): C2 B-spline course noise,
modulated phase, bounded centripetal Hermite tangents. The surface envelopes
follow the bed/bank composition of its ``hydrologySurface.ts`` but use a fixed
projection instead of an additive cut.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from numba import njit
from scipy.ndimage import gaussian_filter1d, minimum_filter1d, maximum_filter1d

GEOMETRY_VERSION = 'hydro-geometry-v1'
DRY, OCEAN, LAKE, RIVER = 0, 1, 2, 3

# Practical bankfull power laws, Q in m3/s; provenance and caveats in the method note.
WIDTH_COEFFICIENT, WIDTH_EXPONENT = 7.2, 0.50
DEPTH_COEFFICIENT, DEPTH_EXPONENT = 0.27, 0.39
MEANDER_WAVELENGTH_WIDTHS = 11.0      # Leopold & Wolman (1960): 10-14 widths
MIN_BEND_RADIUS_WIDTHS = 1.5          # observed Rc/W mostly 1.5-4, median ~2.4
NESTED_BELT_WAVELENGTH_WIDTHS = 55.0
NESTED_FINE_WAVELENGTH_WIDTHS = 3.5
CHANNEL_ANGLE_RAD = 1.0               # direction-angle scale when fully mobile
MAX_ANGLE_RAD = 1.35                  # < pi/2: the course stays a graph over its axis
FINE_ANGLE_RAD = 0.35                 # nested short bends
BELT_AMPLITUDE_WIDTHS = 5.0
NOISE_STD = 0.42                      # std of B-spline noise with uniform controls
BAND_STD = 0.26                       # std of that noise after the band filter below
BEND_LATTICE = 3.6                    # lattice units per wavelength: mean bend period ~1 wavelength


# ----------------------------------------------------------------------------
# Deterministic noise (adapted from Magna hydrology_meanders.rs)

_MASK = np.uint64(0xFFFFFFFF)


def _seed32(*parts) -> int:
    text = '|'.join(str(int(p)) if isinstance(p, (np.integer, bool)) else str(p)
                    for p in parts)
    return int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), 'little')


def _hash01(index, seed: int):
    x = np.asarray(index, np.int64).astype(np.uint64) & _MASK
    x = (x * np.uint64(0x9E3779B9)) & _MASK
    x ^= np.uint64(seed & 0xFFFFFFFF)
    x ^= x >> np.uint64(16)
    x = (x * np.uint64(0x7FEB352D)) & _MASK
    x ^= x >> np.uint64(15)
    x = (x * np.uint64(0x846CA68B)) & _MASK
    x ^= x >> np.uint64(16)
    return x.astype(np.float64) / 4294967295.0


def _value_noise(u, seed):
    """C2 quintic value noise in [0, 1]; lattice unit is one physical wavelength."""
    u = np.asarray(u, np.float64)
    i = np.floor(u)
    t = u - i
    t = t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
    a = _hash01(i, seed)
    return a + (_hash01(i + 1, seed) - a) * t


def _spline_noise(u, seed):
    """Cubic B-spline noise with controls in [-1, 1]; |f''| <= 4 per unit u.

    Random neighbouring controls need not alternate in sign, so there is no
    periodic carrier and no forced bend at each wavelength.
    """
    u = np.asarray(u, np.float64)
    i = np.floor(u)
    t = u - i
    t2, t3 = t * t, t * t * t
    weights = ((1 - t) ** 3, 3 * t3 - 6 * t2 + 4, -3 * t3 + 3 * t2 + 3 * t + 1, t3)
    return sum(w * (_hash01(i + k - 1, seed) * 2 - 1) for k, w in enumerate(weights)) / 6.0


def _smoothstep(e0, e1, x):
    t = np.clip((np.asarray(x, np.float64) - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ----------------------------------------------------------------------------
# Grids, sampling and coordinates

def grid_spacing(shape, bounds):
    rows, cols = shape
    x0, y0, x1, y1 = bounds
    return (x1 - x0) / cols, (y1 - y0) / rows


def cell_centres(shape, bounds):
    """Cell-centre x (cols,) and y (rows,) arrays in float64 map metres."""
    rows, cols = shape
    dx, dy = grid_spacing(shape, bounds)
    return bounds[0] + (np.arange(cols) + 0.5) * dx, bounds[1] + (np.arange(rows) + 0.5) * dy


def _lattice(field_shape, bounds, x, y, wrap):
    rows, cols = field_shape
    dx, dy = grid_spacing(field_shape, bounds)
    u = (np.asarray(x, np.float64) - bounds[0]) / dx - 0.5
    v = (np.asarray(y, np.float64) - bounds[1]) / dy - 0.5
    j0, i0 = np.floor(u), np.floor(v)
    tx, ty = u - j0, v - i0
    j0, i0 = j0.astype(np.int64), i0.astype(np.int64)
    if wrap:
        j0 %= cols
        j1 = (j0 + 1) % cols
    else:
        tx = np.where(j0 < 0, 0.0, np.where(j0 >= cols - 1, 1.0, tx))
        j0 = np.clip(j0, 0, cols - 1)
        j1 = np.minimum(j0 + 1, cols - 1)
    ty = np.where(i0 < 0, 0.0, np.where(i0 >= rows - 1, 1.0, ty))
    i0 = np.clip(i0, 0, rows - 1)
    i1 = np.minimum(i0 + 1, rows - 1)
    return i0, i1, j0, j1, tx, ty, dx, dy


def sample_bilinear(field, bounds, x, y, *, wrap_x=False, gradient=False):
    """Bilinear sample at map positions; edge-clamped or x-periodic.

    Values outside the grid are the stable edge extension (or the periodic
    image), never dependent on which crop requested them. With ``gradient``
    also returns d/dx and d/dy in units per map metre.
    """
    field = np.asarray(field, np.float64)
    i0, i1, j0, j1, tx, ty, dx, dy = _lattice(field.shape, bounds, x, y, wrap_x)
    a, b, c, d = field[i0, j0], field[i0, j1], field[i1, j0], field[i1, j1]
    top, bottom = a + (b - a) * tx, c + (d - c) * tx
    value = top + (bottom - top) * ty
    if not gradient:
        return value
    gx = ((b - a) * (1 - ty) + (d - c) * ty) / dx
    gy = (bottom - top) / dy
    return value, gx, gy


def sample_nearest(field, bounds, x, y, *, wrap_x=False):
    field = np.asarray(field)
    rows, cols = field.shape
    dx, dy = grid_spacing(field.shape, bounds)
    j = np.floor((np.asarray(x, np.float64) - bounds[0]) / dx).astype(np.int64)
    i = np.clip(np.floor((np.asarray(y, np.float64) - bounds[1]) / dy).astype(np.int64), 0, rows - 1)
    j = j % cols if wrap_x else np.clip(j, 0, cols - 1)
    return field[i, j]


def x_scale(y, sphere_radius_m=None):
    """Physical metres per map metre east-west (``cos latitude``), clamped near poles."""
    y = np.asarray(y, np.float64)
    if not sphere_radius_m:
        return np.ones_like(y)
    return np.maximum(np.cos(y / float(sphere_radius_m)), 1e-3)


def to_local_float32(xy, origin=None, *, snap_m=1024.0):
    """Split float64 map positions into a float64 origin and float32 offsets.

    The default origin is the bounding-box centre snapped to ``snap_m`` so
    neighbouring requests usually share it. GPU code adds offsets to the
    origin in double or subtracts a shared camera origin first.
    """
    xy = np.asarray(xy, np.float64).reshape(-1, 2)
    if origin is None:
        centre = (xy.min(0) + xy.max(0)) * 0.5 if len(xy) else np.zeros(2)
        origin = np.round(centre / snap_m) * snap_m
    origin = np.asarray(origin, np.float64)
    return origin, (xy - origin).astype(np.float32)


def pack_reaches(refined, origin=None):
    """Concatenate refined reaches into typed arrays for transport."""
    xy = np.concatenate([r['xy'] for r in refined]) if refined else np.zeros((0, 2))
    origin, local = to_local_float32(xy, origin)
    offsets = np.zeros(len(refined) + 1, np.int64)
    offsets[1:] = np.cumsum([len(r['xy']) for r in refined])
    def cat(key):
        return (np.concatenate([np.asarray(r[key], np.float32) for r in refined])
                if refined else np.zeros(0, np.float32))
    return dict(version=GEOMETRY_VERSION, origin_m=origin, offsets=offsets, xy=local,
                station_m=cat('station_m'), water_m=cat('water_m'),
                width_m=cat('width_m'), depth_m=cat('depth_m'),
                ids=[r['id'] for r in refined])


# ----------------------------------------------------------------------------
# Parent preservation across globally aligned levels

def block_mean(field, factor):
    """Mean of each complete ``factor`` x ``factor`` block."""
    field = np.asarray(field, np.float64)
    rows, cols = field.shape
    if rows % factor or cols % factor:
        raise ValueError('field shape must be a multiple of factor')
    return field.reshape(rows // factor, factor, cols // factor, factor).mean(axis=(1, 3))


def aligned_window(start, size, factor, halo_blocks=0):
    """Expand a fine-index interval to complete, globally aligned parent blocks.

    ``start`` is the global fine index (e.g. 240 m row = floor(y / 240)). The
    lattices of this project share origin 0, so a fine index ``i`` has parent
    ``i // factor``. Returns ``(start, size)`` of the expanded fine window.
    """
    first = (int(start) // factor - halo_blocks) * factor
    last = -(-(int(start) + int(size)) // factor) + halo_blocks
    return first, last * factor - first


def _upsample(parent, factor):
    """Cell-centred, edge-clamped bilinear upsampling by an integer factor."""
    def axis(n):
        u = (np.arange(n * factor) + 0.5) / factor - 0.5
        i = np.floor(u).astype(np.int64)
        t = u - i
        return np.clip(i, 0, n - 1), np.clip(i + 1, 0, n - 1), t
    r0, r1, ty = axis(parent.shape[0])
    c0, c1, tx = axis(parent.shape[1])
    rows = parent[r0] * (1 - ty)[:, None] + parent[r1] * ty[:, None]
    return rows[:, c0] * (1 - tx) + rows[:, c1] * tx


def mean_preserving_correction(deficit, factor, *, iterations=12, omega=1.5, exact=True):
    """Smooth fine field whose block means equal ``deficit`` exactly.

    Richardson iteration on bilinear coefficients (block-mean operator symbol
    in [1/4, 1]); the last, tiny residual is added per block so equality holds
    to round-off. Comparable to Tobler's pycnophylactic interpolation.
    """
    deficit = np.asarray(deficit, np.float64)
    coefficients = deficit.copy()
    for _ in range(iterations):
        residual = deficit - block_mean(_upsample(coefficients, factor), factor)
        if np.max(np.abs(residual), initial=0.0) < 1e-9:
            break
        coefficients += omega * residual
    correction = _upsample(coefficients, factor)
    if not exact:
        return correction
    residual = deficit - block_mean(correction, factor)
    return correction + np.kron(residual, np.ones((factor, factor)))


def parent_envelope(parent, factor, margin_m=0.0):
    """Fine-resolution lower/upper bounds from each parent's 3x3 min/max +- margin."""
    from scipy.ndimage import maximum_filter, minimum_filter
    parent = np.asarray(parent, np.float64)
    lower = _upsample(minimum_filter(parent, 3, mode='nearest'), factor) - margin_m
    upper = _upsample(maximum_filter(parent, 3, mode='nearest'), factor) + margin_m
    return lower, upper


def preserve_parent(fine, parent, factor, *, lower=None, upper=None, iterations=12, rounds=6):
    """Constrain refined detail so each parent cell's mean survives.

    ``fine`` must cover complete parent blocks aligned with the global lattice
    (see :func:`aligned_window`); block ``(I, J)`` of ``fine`` is parent cell
    ``(I, J)`` of ``parent``. Partial blocks are rejected rather than claimed
    conserved. The fine residual relative to the parent is kept; only a smooth
    mean correction is added. Optional fine-resolution ``lower``/``upper``
    envelopes are hard bounds; means and bounds are alternated and the final
    clamp wins, with the remaining mean error reported.

    Separately processed windows agree only approximately in their interior
    (the correction depends on deficits within about ``iterations`` blocks);
    exact agreement requires the same complete aligned window.
    """
    fine = np.asarray(fine, np.float64)
    parent = np.asarray(parent, np.float64)
    if fine.shape != (parent.shape[0] * factor, parent.shape[1] * factor):
        raise ValueError('fine must be exactly parent.shape * factor (complete aligned blocks)')
    out = fine + mean_preserving_correction(parent - block_mean(fine, factor), factor,
                                            iterations=iterations)
    if lower is not None or upper is not None:
        lo = -np.inf if lower is None else np.asarray(lower, np.float64)
        hi = np.inf if upper is None else np.asarray(upper, np.float64)
        if np.any(np.asarray(lo) > np.asarray(hi)):
            raise ValueError('lower envelope exceeds upper envelope')
        lo = np.broadcast_to(lo, out.shape)
        hi = np.broadcast_to(hi, out.shape)
        for _ in range(rounds):
            out = np.clip(out, lo, hi)
            deficit = parent - block_mean(out, factor)
            if np.max(np.abs(deficit)) < 1e-6:
                break
            out = out + mean_preserving_correction(deficit, factor, iterations=iterations)
        out = np.clip(out, lo, hi)
        # Final exact step: distribute each block's remaining deficit in
        # proportion to the room left below U (or above L). This is exact and
        # stays inside the envelope whenever the parent mean is attainable.
        deficit = np.kron(parent - block_mean(out, factor), np.ones((factor, factor)))
        room = np.where(deficit > 0, hi - out, out - lo)
        room = np.where(np.isfinite(room), room, 1e12)
        total = np.kron(block_mean(room, factor), np.ones((factor, factor)))
        share = np.where(total > 0, np.minimum(np.abs(deficit) / np.maximum(total, 1e-300), 1.0), 0.0)
        out = np.clip(out + np.sign(deficit) * share * room, lo, hi)
    error = parent - block_mean(out, factor)
    return out, dict(max_mean_error_m=float(np.max(np.abs(error))),
                     max_correction_m=float(np.max(np.abs(out - fine))))


# ----------------------------------------------------------------------------
# Polyline helpers

def _cumlen(xy):
    s = np.zeros(len(xy))
    if len(xy) > 1:
        s[1:] = np.cumsum(np.hypot(*np.diff(xy, axis=0).T))
    return s


@njit(cache=True)
def _douglas_peucker(xy, mark, tolerance):
    # Split at the first farthest point from the clamped anchor segment. Dense
    # reaches have 1e5 samples; per-split Python overhead dominated regions.
    anchors = np.flatnonzero(mark)
    stack = np.empty((len(xy), 2), np.int64)
    top = 0
    for i in range(len(anchors) - 1):
        stack[top, 0], stack[top, 1] = anchors[i], anchors[i + 1]
        top += 1
    while top:
        top -= 1
        a, b = stack[top, 0], stack[top, 1]
        if b - a < 2:
            continue
        ax, ay = xy[a, 0], xy[a, 1]
        abx, aby = xy[b, 0] - ax, xy[b, 1] - ay
        den = max(abx * abx + aby * aby, 1e-30)
        best, k = -1.0, a + 1
        for i in range(a + 1, b):
            dx, dy = xy[i, 0] - ax, xy[i, 1] - ay
            t = min(max((dx * abx + dy * aby) / den, 0.0), 1.0)
            d = math.hypot(dx - t * abx, dy - t * aby)
            if d > best:
                best, k = d, i
        if best > tolerance:
            mark[k] = True
            stack[top, 0], stack[top, 1] = a, k
            stack[top + 1, 0], stack[top + 1, 1] = k, b
            top += 2
    return mark


def _simplify(xy, tolerance, keep=()):
    """Douglas-Peucker indices; ``keep`` indices (and ends) are always retained."""
    n = len(xy)
    mark = np.zeros(n, bool)
    mark[[0, n - 1]] = True
    mark[list(keep)] = True
    xy = np.ascontiguousarray(xy, np.float64)
    return np.flatnonzero(_douglas_peucker(xy, mark, float(tolerance)))


def _hermite(points, step, start_tangent=None, end_tangent=None):
    """Densely sampled Hermite/Bezier spline through ``points``.

    Shared bisector tangents with centripetal handles shorter than 0.85/3 of
    each chord (Magna ``smooth_polyline``). Returns dense xy and the dense
    index of every control point (controls are retained exactly) and the unit
    tangent direction at each control.
    """
    n = len(points)
    chords = np.diff(points, axis=0)
    lengths = np.maximum(np.hypot(*chords.T), 1e-12)
    units = chords / lengths[:, None]
    tangents = np.zeros_like(points)
    for i in range(n):
        before, after = max(i - 1, 0), min(i, n - 2)
        direction = units[before] + units[after]
        given = start_tangent if i == 0 else end_tangent if i == n - 1 else None
        if given is not None:
            direction = np.asarray(given, np.float64)
        norm = np.hypot(*direction)
        direction = direction / norm if norm > 1e-12 else units[after]
        tangents[i] = direction * math.sqrt(min(lengths[before], lengths[after])) * 0.85
    pieces, control_index, count = [], [0], 0
    for i in range(n - 1):
        interval = math.sqrt(lengths[i]) / 3.0
        a, b = points[i], points[i + 1]
        c, d = a + tangents[i] * interval, b - tangents[i + 1] * interval
        m = max(1, int(math.ceil(lengths[i] / step)))
        t = (np.arange(m) / m)[:, None]
        u = 1 - t
        pieces.append(u ** 3 * a + 3 * u * u * t * c + 3 * u * t * t * d + t ** 3 * b)
        count += m
        control_index.append(count)
    pieces.append(points[-1:])
    directions = tangents / np.maximum(np.hypot(*tangents.T), 1e-300)[:, None]
    return np.concatenate(pieces), np.asarray(control_index), directions


def _unit_normals(xy):
    tangent = np.gradient(xy, axis=0)
    tangent /= np.maximum(np.hypot(*tangent.T), 1e-12)[:, None]
    return tangent, np.stack([-tangent[:, 1], tangent[:, 0]], 1)


def _monotone_profile(values):
    """Pool-adjacent-violators least-squares non-increasing fit."""
    blocks = []
    for value in values:
        blocks.append([float(value), 1])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]:
            total, count = blocks.pop()
            blocks[-1][0] += total
            blocks[-1][1] += count
    return np.concatenate([np.full(c, t / c) for t, c in blocks])


def _moving_average(values, window):
    window = max(1, int(window) | 1)
    if window == 1 or len(values) < 3:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, pad, mode='edge')
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode='valid')


class _Frame:
    """Local physical frame X = (x - xc) cos(lat(y)), Y = y - yc about a reach."""

    def __init__(self, xy, sphere_radius_m):
        self.radius = sphere_radius_m
        self.centre = (xy.min(0) + xy.max(0)) * 0.5

    def forward(self, xy):
        out = xy - self.centre
        out[:, 0] *= x_scale(xy[:, 1], self.radius)
        return out

    def inverse(self, local):
        y = local[:, 1] + self.centre[1]
        return np.stack([local[:, 0] / x_scale(y, self.radius) + self.centre[0], y], 1)


def bankfull_geometry(flow_m3s, *, width_coefficient=WIDTH_COEFFICIENT,
                      width_exponent=WIDTH_EXPONENT, depth_coefficient=DEPTH_COEFFICIENT,
                      depth_exponent=DEPTH_EXPONENT):
    """Width and depth in metres from discharge (Andreadis et al. 2013 power laws)."""
    q = np.maximum(np.asarray(flow_m3s, np.float64), 0.0)
    return (np.maximum(width_coefficient * q ** width_exponent, 0.5),
            np.maximum(depth_coefficient * q ** depth_exponent, 0.05))


def nested_meander_gate(*, precip_mm_year, temperature_c, slope, flow_m3s, valley_widths):
    """0..1 eligibility for nested tropical alluvial belts.

    Humid (P >= ~1.2-2 m/yr), warm (mean T >= ~18-22 C), low gradient
    (S <~ 2e-4..6e-4), sizeable (Q >~ 20-200 m3/s) and unconfined
    (valley half-width >~ 3-8 channel widths). All criteria multiply.
    """
    return float(_smoothstep(1200, 2000, precip_mm_year) * _smoothstep(18, 22, temperature_c)
                 * (1 - _smoothstep(2e-4, 6e-4, slope)) * _smoothstep(20, 200, flow_m3s)
                 * _smoothstep(3, 8, valley_widths))


# ----------------------------------------------------------------------------
# Reach refinement

# Optional display cosmetics, all off by default (viewer option names).
COSMETICS = ('valleyFit', 'meanders', 'nestedMeanders', 'smoothBanks')


def cosmetic_options(cosmetics=()):
    """:func:`refine_reach` switches for enabled cosmetics; none = smoothed axis only."""
    names = set(cosmetics)
    return dict(valley_fit='valleyFit' in names, meanders='meanders' in names,
                nested='nestedMeanders' in names, limit_radius='smoothBanks' in names)


def _profile_with_pins(water, pins):
    """Non-increasing water profile pinned at ``pins``; returns (profile, conflict_m)."""
    water = np.asarray(water, np.float64).copy()
    levels = water[pins]
    pinned = np.minimum.accumulate(levels)
    conflict = float(np.max(levels - pinned, initial=0.0))
    out = water.copy()
    for k in range(len(pins) - 1):
        a, b = pins[k], pins[k + 1]
        segment = _monotone_profile(water[a:b + 1])
        out[a:b + 1] = np.clip(segment, pinned[k + 1], pinned[k])
        out[a], out[b] = pinned[k], pinned[k + 1]
    return out, conflict


def _valley_fit(axis, s, normals, pin_s, sampler, corridor, depth, step):
    """Viterbi lateral path inside the corridor following the valley floor.

    Node cost: height above the section minimum (incision proxy, in bankfull
    depths) plus a mild deviation penalty; transitions penalise lateral slope
    and forbid moves steeper than 0.7. Pins are forced to zero offset.
    """
    length = s[-1]
    m = max(3, int(math.ceil(length / step)) + 1)
    st = np.linspace(0.0, length, m)
    ds = st[1]
    k = int(min(8, math.floor(corridor / max(step * 0.5, 1e-9))))
    if k < 1:
        return st, np.zeros(m), {}
    offs = np.arange(-k, k + 1) * (corridor / k)
    p = np.stack([np.interp(st, s, axis[:, 0]), np.interp(st, s, axis[:, 1])], 1)
    nn = np.stack([np.interp(st, s, normals[:, 0]), np.interp(st, s, normals[:, 1])], 1)
    nn /= np.maximum(np.hypot(*nn.T), 1e-12)[:, None]
    px = p[:, None, 0] + nn[:, None, 0] * offs
    py = p[:, None, 1] + nn[:, None, 1] * offs
    heights = sampler(px.ravel(), py.ravel()).reshape(px.shape)
    node = (heights - heights.min(1, keepdims=True)) / max(depth, 1.0) + 0.25 * (offs / corridor) ** 2
    for station in pin_s:
        i = int(np.argmin(np.abs(st - station)))
        node[i] = np.inf
        node[i, k] = 0.0
    jump = offs[:, None] - offs[None, :]
    lateral = 2.0 * (jump / ds) ** 2
    lateral[np.abs(jump) > max(0.7 * ds, corridor / k * 1.001)] = np.inf
    cost = node[0].copy()
    back = np.zeros((m, 2 * k + 1), np.int64)
    for i in range(1, m):
        total = cost[:, None] + lateral
        back[i] = np.argmin(total, 0)
        cost = total[back[i], np.arange(2 * k + 1)] + node[i]
    path = np.empty(m, np.int64)
    path[-1] = int(np.argmin(cost))
    for i in range(m - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    raw = offs[path]
    fitted = gaussian_filter1d(raw, 2.0, mode='nearest')
    distance = np.min(np.abs(st[:, None] - np.asarray(pin_s)[None, :]), 1)
    fitted *= _smoothstep(0.0, 4.0 * ds, distance)
    fitted = np.clip(fitted, -corridor, corridor)
    ground0 = heights[:, k]
    ground1 = sampler(p[:, 0] + nn[:, 0] * fitted, p[:, 1] + nn[:, 1] * fitted)
    def adverse(g):
        g = _moving_average(g, 3)
        return float(np.sum(np.maximum(np.diff(g), 0.0)))
    info = dict(fit_mean_drop_m=float(ground0.mean() - ground1.mean()),
                fit_adverse_m=adverse(ground1), axis_adverse_m=adverse(ground0))
    if ground1.mean() > ground0.mean() or info['fit_adverse_m'] > info['axis_adverse_m'] * 1.1 + depth:
        info['fit_rejected'] = True
        return st, np.zeros(m), info
    return st, fitted, info


def _axis_curvature(axis, s):
    angle = np.unwrap(np.arctan2(*np.gradient(axis, axis=0)[:, ::-1].T))
    return np.abs(gaussian_filter1d(np.gradient(angle, s), 1.0, mode='nearest'))


def _limit_axis_radius(axis, s, pin_s, radius, passes=4):
    """Smooth the valley axis until its bend radius is at least ``radius``.

    The Hermite axis concentrates the parent's turning at its controls and
    the valley-fit lattice path at its lateral moves; a wide channel drawn on
    such corners shows chords joined by kinks tighter than its own width.
    Gaussian passes of ``radius / 2`` (on the physics-fixed axis stations)
    are blended in with a C1 weight that is zero, with zero slope, at every
    pin, so shared node positions and tangents stay exact. This only lowers
    curvature; it adds no bends.
    """
    ds = max(float(np.median(np.diff(s))), 1e-9)
    distance = np.min(np.abs(s[:, None] - np.asarray(pin_s)[None, :]), 1)
    weight = _smoothstep(0.0, 2.0 * radius, distance)[:, None]
    core = weight[:, 0] > 0.999
    for _ in range(passes):
        if not core.any() or _axis_curvature(axis, s)[core].max() * radius <= 1.0:
            break
        axis = axis + weight * (gaussian_filter1d(axis, 0.5 * radius / ds, axis=0, mode='nearest') - axis)
    return axis


def _channel_offsets(s, wavelength, width, mobility, bound, seed_r, r, gain, nested_intensity):
    """Lateral channel offset as a function of axis station ``s``.

    The channel direction angle relative to its axis follows irregular
    B-spline noise of a modulated phase (Langbein & Leopold's sine-generated
    curve with a stochastic carrier instead of a sinusoid, Magna's phase
    modulation). Integrating ``da = cos(theta) ds'`` and ``do = sin(theta) ds'``
    along the channel arc ``s'`` with |theta| < pi/2 keeps the course a
    single-valued graph over the axis (no loops, no self-intersection). A
    weak restoring term keeps ``o`` inside the free corridor ``bound``.
    """
    length = s[-1]
    h = max(float(np.median(np.diff(s))), 1e-6)
    steps = int(math.ceil(length / (h * math.cos(MAX_ANGLE_RAD)))) + 2
    arc = np.arange(steps) * h
    coord = r[0] * 1000 + arc / wavelength
    frequency = 2.0 ** ((_value_noise(coord * 0.38, _seed32(seed_r, 'f')) - 0.5) * 0.7)
    phase = r[1] * 1000 + np.r_[0.0, np.cumsum(0.5 * (frequency[1:] + frequency[:-1]) * h / wavelength)]
    modulation = 0.42 + 0.58 * _value_noise(coord * 0.4, _seed32(seed_r, 'a'))
    # Angle content longer than about one wavelength would integrate into an
    # unbounded lateral random walk; it is removed (broad wandering belongs to
    # the valley axis and the nested belt band).
    def band(values, cutoff):
        return (values - gaussian_filter1d(values, cutoff / h, mode='nearest')) / BAND_STD
    noise = band(_spline_noise(phase * BEND_LATTICE, _seed32(seed_r, "c")), 0.45 * wavelength) * modulation
    fine = np.zeros(steps)
    if nested_intensity:
        fine_wave = NESTED_FINE_WAVELENGTH_WIDTHS * width
        fine = band(_spline_noise(r[3] * 1000 + arc * BEND_LATTICE / fine_wave, _seed32(seed_r, "fine")),
                    0.45 * fine_wave) * FINE_ANGLE_RAD * nested_intensity \
            * (0.5 + _value_noise(coord * 0.7, _seed32(seed_r, 'fa')))
    omega = CHANNEL_ANGLE_RAD * np.sqrt(np.clip(mobility, 0, 1)) * gain
    f64 = lambda v: np.ascontiguousarray(v, np.float64)
    a_out, o_out = _integrate_angle(f64(s), f64(bound), f64(np.broadcast_to(omega, np.shape(s))),
                                    f64(noise), f64(fine),
                                    float(gain), float(width), float(h), float(length), steps)
    return np.interp(s, a_out, o_out)


@njit(cache=True)
def _integrate_angle(s, bound, omega, noise, fine, gain, width, h, length, steps):
    a_out, o_out = np.zeros(steps), np.zeros(steps)
    a = o = theta = 0.0
    turn = h / (MIN_BEND_RADIUS_WIDTHS * width)    # exact minimum bend radius
    j, last, n = 0, len(s) - 1, 1
    for k in range(steps - 1):
        while j < last and s[j + 1] <= a:
            j += 1
        b = max(min(bound[j], 4.0 * width), 1e-6)
        target = omega[j] * noise[k] + fine[k] * gain - 0.5 * o / b
        target = MAX_ANGLE_RAD * math.tanh(target / MAX_ANGLE_RAD)
        theta = min(max(target, theta - turn), theta + turn)
        a += h * math.cos(theta)
        o += h * math.sin(theta)
        a_out[n], o_out[n] = a, o
        n += 1
        if a >= length:
            break
    return a_out[:n], o_out[:n]


def refine_reach(reach, *, seed, terrain=None, terrain_bounds=None, sample_spacing_m=None,
                 climate=None, corridor_m=None, sphere_radius_m=None, wrap_period_m=None,
                 start_tangent=None, end_tangent=None, tolerance_m=None, meander_scale=1.0,
                 valley_fit=True, fit_profile=True, nested=True, smooth_axis=True,
                 meanders=True, limit_radius=True, support_bounds=None):
    """Refine one reach into a natural, pinned, physically sized centreline.

    Parameters
    ----------
    reach : dict
        Core reach record; ``xy`` is the authoritative parent course (e.g. D8
        cell centres), ``water_m`` its levels. ``pins`` (indices) mark portals
        or nodes that stay exact; ends are always pinned.
    seed : int
        World seed; combined with ``reach['id']`` and band tags.
    terrain, terrain_bounds : optional height grid used for valley fitting,
        valley confinement and terrain-following water.
    sample_spacing_m : terrain/target sampling scale; default terrain cell.
    climate : dict with scalar ``precip_mm_year`` and ``temperature_c``
        (or ``reach['climate']``); gates nested tropical belts. None = no nesting.
    corridor_m : half-width of the inherited parent corridor; default
        ``reach['corridor_m']`` or ``max(0.75 * input spacing, 2.5 * width)``.
        Valley fitting and all bends stay inside it.
    start_tangent, end_tangent : unit map-space directions shared at nodes.
    tolerance_m : geometric tessellation error in metres; it never changes
        the curve, only how many of its samples are retained.
    smooth_axis : False keeps the parent polyline itself as the axis (the
        last crossing-repair fallback of :func:`refine_network`).
    meanders : False returns the smoothed axis itself as the course: no
        generated bends, nested belt, confinement sampling or far-view belt
        (``nested`` is then ignored). ``valley_fit`` and ``limit_radius``
        (the bend-radius limiter of the axis) are independent switches.
    support_bounds : optional map-space envelope for the complete generated
        channel and belt. Total displacement from each original parent
        station is smoothly limited before lengths and widths are computed.
        Parent vertices must lie strictly inside this envelope; pins retain
        their positions and tangents. None preserves the unbounded API.

    Returns a dict with the passthrough record keys plus ``xy`` (M, 2) float64,
    ``station_m`` (channel arc length), ``axis_station_m`` (stable valley
    station), ``water_m``, ``width_m``, ``depth_m``, ``tangent``, the far-view
    ``belt_xy``/``belt_station_m``/``belt_water_m``/``belt_width_m``/
    ``belt_halfwidth_m``, ``wavelength_m``, ``morphology`` and ``diagnostics``.
    """
    xy_map = np.asarray(reach['xy'], np.float64).reshape(-1, 2).copy()
    if support_bounds is not None:
        support_bounds = np.asarray(support_bounds, np.float64)
        if (support_bounds.shape != (4,) or not np.isfinite(support_bounds).all() or
                np.any(xy_map <= support_bounds[:2]) or np.any(xy_map >= support_bounds[2:])):
            raise ValueError('support_bounds must strictly contain every parent vertex')
    water_in = np.asarray(reach.get('water_m', np.zeros(len(xy_map))), np.float64)
    if wrap_period_m:
        step = np.diff(xy_map[:, 0])
        xy_map[1:, 0] = xy_map[0, 0] + np.cumsum((step + wrap_period_m / 2) % wrap_period_m
                                                  - wrap_period_m / 2)
    pins = sorted({0, len(xy_map) - 1, *(int(p) % len(xy_map) for p in reach.get('pins', ()))})
    # Drop repeated vertices; a repeated pin maps to the retained copy.
    keep = np.ones(len(xy_map), bool)
    keep[1:] = np.hypot(*np.diff(xy_map, axis=0).T) > 1e-9
    index_map = np.cumsum(keep) - 1
    xy_map, water_in = xy_map[keep], water_in[keep]
    pins = sorted({int(index_map[p]) for p in pins})
    base = {key: reach.get(key) for key in
            ('id', 'source', 'target', 'basin_id', 'downstream_id', 'flow_m3s', 'area_m2')}
    flow = float(reach.get('flow_m3s') or 0.0)
    width, depth = (float(v) for v in bankfull_geometry(flow))
    water, conflict = _profile_with_pins(water_in, pins) if len(xy_map) > 1 else (water_in, 0.0)
    if len(xy_map) < 2 or _cumlen(xy_map)[-1] < 1e-6:
        return dict(base, xy=xy_map, station_m=np.zeros(len(xy_map)),
                    axis_station_m=np.zeros(len(xy_map)), water_m=water,
                    width_m=np.full(len(xy_map), width), depth_m=np.full(len(xy_map), depth),
                    tangent=np.zeros_like(xy_map), belt_xy=xy_map, belt_station_m=np.zeros(len(xy_map)),
                    belt_water_m=water, belt_width_m=np.full(len(xy_map), width),
                    belt_halfwidth_m=np.full(len(xy_map), width / 2),
                    wavelength_m=MEANDER_WAVELENGTH_WIDTHS * width, bankfull_width_m=width,
                    bankfull_depth_m=depth, pins=np.arange(len(xy_map)), morphology='point',
                    diagnostics=dict(profile_conflict_m=conflict), version=GEOMETRY_VERSION)

    frame = _Frame(xy_map, sphere_radius_m)
    local = frame.forward(xy_map)
    input_q = _cumlen(local)
    spacing_in = float(np.median(np.diff(input_q)))
    corridor = float(corridor_m or reach.get('corridor_m') or max(0.75 * spacing_in, 2.5 * width))
    wavelength = MEANDER_WAVELENGTH_WIDTHS * width

    def map_tangent(t, at):
        if t is None:
            return None
        t = np.asarray(t, np.float64).copy()
        t[0] *= x_scale(at[1], sphere_radius_m)
        return t

    sampler = None
    if terrain is not None:
        terrain = np.asarray(terrain, np.float64)
        wrap = bool(wrap_period_m) and abs((terrain_bounds[2] - terrain_bounds[0]) - wrap_period_m) < 1e-6
        def sampler(x, y):
            m = frame.inverse(np.stack([np.ravel(x), np.ravel(y)], 1))
            return sample_bilinear(terrain, terrain_bounds, m[:, 0], m[:, 1], wrap_x=wrap)
        if sample_spacing_m is None:
            sample_spacing_m = float(min(grid_spacing(terrain.shape, terrain_bounds)))

    # 1. Axis: stairs removed by pinned simplification, then bounded Hermite.
    # Without generated bends nothing needs wavelength sampling; 1/64 of the
    # parent spacing keeps the spline's chord error well below a decoder cell.
    axis_step = max(min(spacing_in / 8.0, wavelength / 8.0) if meanders else spacing_in / 64.0,
                    input_q[-1] / 50000.0, 0.05)
    if smooth_axis:
        # A channel W wide cannot express parent staircase detail finer than
        # its own half-width; such corners would otherwise cap every bend
        # through the 0.7 / curvature bound below.
        controls = _simplify(local, max(0.5 * spacing_in, 0.5 * width), pins)
        axis, control_dense, control_dir = _hermite(local[controls], axis_step,
                                                    map_tangent(start_tangent, xy_map[0]),
                                                    map_tangent(end_tangent, xy_map[-1]))
    else:
        controls = np.arange(len(local))
        counts = np.maximum(np.ceil(np.diff(input_q) / axis_step).astype(np.int64), 1)
        control_dense = np.r_[0, np.cumsum(counts)]
        q = np.concatenate([np.linspace(input_q[i], input_q[i + 1], c, endpoint=False)
                            for i, c in enumerate(counts)] + [input_q[-1:]])
        axis = np.stack([np.interp(q, input_q, local[:, 0]), np.interp(q, input_q, local[:, 1])], 1)
        control_dir, _ = _unit_normals(local)
    s_axis = _cumlen(axis)
    # q(s): parent input length coordinate, piecewise linear between controls.
    q_axis = np.interp(s_axis, s_axis[control_dense], input_q[controls])
    pin_dense = control_dense[np.searchsorted(controls, pins)]
    _, normals = _unit_normals(axis)
    diagnostics = dict(profile_conflict_m=conflict, corridor_m=corridor)

    # 2. Valley fit inside the parent corridor (optional).
    fit_offsets = np.zeros(len(axis))
    if sampler is not None and valley_fit and corridor > width / 2:
        st, fitted, info = _valley_fit(axis, s_axis, normals, s_axis[pin_dense], sampler,
                                       corridor - width / 2, depth,
                                       max(sample_spacing_m, s_axis[-1] / 4000.0))
        diagnostics.update(info)
        fit_offsets = np.interp(s_axis, st, fitted)
        diagnostics['valley_fit_max_shift_m'] = float(np.max(np.abs(fit_offsets)))
        if np.any(fit_offsets):
            axis = axis + normals * fit_offsets[:, None]
            s_axis = _cumlen(axis)
            _, normals = _unit_normals(axis)
    if smooth_axis and limit_radius:
        axis = _limit_axis_radius(axis, s_axis, s_axis[pin_dense], MIN_BEND_RADIUS_WIDTHS * width)
        s_axis = _cumlen(axis)
        _, normals = _unit_normals(axis)
        diagnostics['axis_min_radius_m'] = float(1.0 / max(_axis_curvature(axis, s_axis).max(), 1e-12))
    length = s_axis[-1]
    pin_s = s_axis[pin_dense]

    # 3. Water profile on axis stations; optionally follow the fine valley floor.
    def profile_at(s):
        return np.interp(np.interp(s, s_axis, q_axis), input_q, water)
    level_axis = profile_at(s_axis)
    if sampler is not None and fit_profile:
        floor = _moving_average(sampler(axis[:, 0], axis[:, 1]),
                                max(3, int(0.5 * wavelength / max(np.median(np.diff(s_axis)), 1e-9))))
        followed = np.minimum(level_axis, floor)
        for k in range(len(pin_dense) - 1):
            a, b = pin_dense[k], pin_dense[k + 1]
            top, bottom = level_axis[a], level_axis[b]
            segment = np.minimum.accumulate(followed[a:b + 1])
            level_axis[a:b + 1] = np.clip(segment, bottom, top)
            level_axis[a], level_axis[b] = top, bottom     # pinned levels are authority
        diagnostics['profile_lowered_max_m'] = float(np.max(profile_at(s_axis) - level_axis))
    if not meanders:
        return _axis_course(base, frame, axis, s_axis, level_axis, pin_dense, controls, pins,
                            control_dir, control_dense if not smooth_axis else [], width, depth,
                            wavelength, corridor, sphere_radius_m, reach, tolerance_m, diagnostics)

    # 4. Physical controls along the axis: slope, valley confinement, mobility.
    ds_axis = max(float(np.median(np.diff(s_axis))), 1e-9)
    slope_window = max(3, int(max(2 * wavelength, 4 * spacing_in) / ds_axis))
    slope = np.maximum(-np.gradient(_moving_average(level_axis, slope_window), s_axis), 0.0)
    slope = _moving_average(slope, slope_window)
    mean_slope = max((level_axis[0] - level_axis[-1]) / length, 0.0)
    threshold = 0.0125 * max(flow, 1e-3) ** -0.44     # Leopold & Wolman (1957), SI
    mobility = 1.0 / (1.0 + (slope / threshold) ** 2)
    free = np.full(len(axis), corridor) - np.abs(fit_offsets)
    valley_half = free.copy()
    if sampler is not None:
        lateral = np.linspace(-corridor, corridor, 17)
        stride = max(1, int(max(sample_spacing_m, wavelength / 8) / ds_axis))
        idx = np.unique(np.r_[np.arange(0, len(axis), stride), len(axis) - 1])
        px = axis[idx, None, 0] + normals[idx, None, 0] * lateral
        py = axis[idx, None, 1] + normals[idx, None, 1] * lateral
        h = sampler(px.ravel(), py.ravel()).reshape(px.shape)
        # The channel spans W and the parent course is often a one-cell
        # incised trench; neither is a valley wall. Walls rise 2 D above the
        # bank just outside that band (or above bed + 2 D in a canyon).
        rise = max(2 * depth, 2.0)
        inner = np.abs(lateral) <= max(width / 2, 0.5 * sample_spacing_m)
        m = int(np.sum(inner[9:]))
        bank = (np.minimum(h[:, 7 - m], h[:, 9 + m]) if m < 8 else h[:, 8] + rise)
        wall = (h > np.minimum(bank, h[:, 8] + rise)[:, None] + rise) & ~inner
        half = np.full((len(idx), 2), corridor)
        for side, columns in ((0, np.arange(7, -1, -1)), (1, np.arange(9, 17))):
            hit = wall[:, columns]
            first = np.where(hit.any(1), hit.argmax(1), len(columns))
            half[:, side] = np.where(first < len(columns), np.abs(lateral[columns][np.minimum(first, len(columns) - 1)]), corridor)
        valley_half = np.interp(s_axis, s_axis[idx], half.min(1))
    tangent_angle = np.unwrap(np.arctan2(*np.gradient(axis, axis=0)[:, ::-1].T))
    curvature = np.abs(gaussian_filter1d(np.gradient(tangent_angle, s_axis), 3.0, mode='nearest'))
    bound = np.minimum(np.minimum(free, valley_half) - width / 2, 0.7 / np.maximum(curvature, 1e-12))
    bound = np.maximum(bound, 0.0)

    # 5. Gate nested tropical belts by actual climate, slope, flow and confinement.
    climate = climate if climate is not None else reach.get('climate')
    seed_r = _seed32(seed, reach.get('id'), 'reach')
    intensity = 0.0
    if nested and climate:
        gate = nested_meander_gate(
            precip_mm_year=float(climate.get('precip_mm_year', 0.0)),
            temperature_c=float(climate.get('temperature_c', 0.0)), slope=mean_slope,
            flow_m3s=flow, valley_widths=float(np.median(valley_half)) / width)
        intensity = gate * (0.5 + 0.5 * float(_hash01([7], seed_r)[0]))
        intensity = intensity if intensity > 0.15 else 0.0
    diagnostics['nested_intensity'] = intensity

    # 6. Dense stations: fixed by reach physics, never by display LOD.
    h = wavelength / 24.0
    if intensity:
        h = min(h, NESTED_FINE_WAVELENGTH_WIDTHS * width / 24.0)
    h = max(min(h, spacing_in / 4.0), length / 200000.0, 0.05)
    corners = s_axis[control_dense] if not smooth_axis else ()
    s = np.unique(np.r_[np.arange(0.0, length, h), pin_s, corners, length])
    ax_xy = np.stack([np.interp(s, s_axis, axis[:, 0]), np.interp(s, s_axis, axis[:, 1])], 1)
    nrm = np.stack([np.interp(s, s_axis, normals[:, 0]), np.interp(s, s_axis, normals[:, 1])], 1)
    nrm /= np.maximum(np.hypot(*nrm.T), 1e-12)[:, None]
    b_s = np.interp(s, s_axis, bound)
    window = max(1, int(wavelength / h))
    # Half a wavelength each side: one confined section must not straighten
    # 16 km of a large river. min(b_smooth, b_s) and the limiter still apply.
    b_smooth = gaussian_filter1d(minimum_filter1d(b_s, window // 2 * 2 + 1, mode='nearest'),
                                 max(window / 4, 1.0), mode='nearest')
    b_smooth = np.maximum(np.minimum(b_smooth, b_s), 0.0)
    mob = np.interp(s, s_axis, mobility)

    # Nested belt: offset band of wavelength ~55 W (B-spline noise oscillates
    # about every 2.5 lattice units). Channel bends: angle integration below.
    r = _hash01(np.arange(8), seed_r)
    belt = np.zeros_like(s)
    if intensity:
        u_belt = r[2] * 1000 + s * 2.5 / (NESTED_BELT_WAVELENGTH_WIDTHS * width)
        belt = (0.8 * _spline_noise(u_belt, _seed32(seed_r, 'belt'))
                + 0.4 * _spline_noise(u_belt * 0.37, _seed32(seed_r, 'belt2'))) / NOISE_STD \
            * BELT_AMPLITUDE_WIDTHS * width * intensity * meander_scale
    channel = _channel_offsets(s, wavelength, width, mob, b_smooth, seed_r, r,
                               meander_scale * (1 + 0.3 * intensity), intensity)

    gaps = np.diff(pin_s)
    segment = np.clip(np.searchsorted(pin_s, s, side='right') - 1, 0, len(gaps) - 1)
    distance = np.minimum(s - pin_s[segment], pin_s[segment + 1] - s)
    taper_c = _smoothstep(0.0, np.maximum(np.minimum(0.6 * wavelength, 0.35 * gaps[segment]), 1e-9), distance)
    taper_b = _smoothstep(0.0, np.maximum(np.minimum(0.25 * NESTED_BELT_WAVELENGTH_WIDTHS * width,
                                                     0.35 * gaps[segment]), 1e-9), distance)

    def soft(x, limit):
        safe = np.maximum(limit, 1e-9)
        return np.where(limit > 1e-9, safe * np.tanh(x / safe), 0.0)

    scale = np.ones_like(s)
    for _ in range(4):
        # Bounded, C1-tapered offsets: zero offset and slope at every pin, so
        # shared node positions and axis tangents are exact.
        belt_off = soft(belt * taper_b * scale, b_smooth)
        total = soft((belt * taper_b + channel * taper_c) * scale, b_smooth)
        # The channel integration already bounds its own radius; this local
        # limiter only catches what tapers, clipping and the belt add, with a
        # 25 % allowance for finite-difference curvature on the dense samples.
        slope_o = np.gradient(total, s)
        ratio = (np.abs(np.gradient(slope_o, s)) / (1 + slope_o ** 2) ** 1.5
                 * 0.75 * MIN_BEND_RADIUS_WIDTHS * width)
        ratio = gaussian_filter1d(maximum_filter1d(ratio, window // 2 * 2 + 1, mode='nearest'),
                                  max(window / 4, 1.0), mode='nearest')
        if float(ratio.max()) <= 1.0:
            break
        scale = scale / np.maximum(ratio, 1.0)
    total = np.clip(total, -b_s, b_s)
    belt_off = np.clip(belt_off, -b_s, b_s)
    channel_xy = ax_xy + nrm * total[:, None]
    belt_xy = ax_xy + nrm * belt_off[:, None]
    if support_bounds is not None:
        q = np.interp(s, s_axis, q_axis)
        parent = np.column_stack([np.interp(q, input_q, xy_map[:, a]) for a in (0, 1)])
        def bounded(points):
            delta = frame.inverse(points)-parent
            room = np.where(delta >= 0, support_bounds[2:]-parent, parent-support_bounds[:2])
            ratio = np.max(np.abs(delta)/room, axis=1)
            scale = np.ones_like(ratio)
            near = ratio > .5
            # Identity below half the available room, then a C1 saturation
            # strictly inside the envelope. No edge clipping or new pins.
            scale[near] = (.5+.5*np.tanh(2*ratio[near]-1))/ratio[near]
            return frame.forward(parent+delta*scale[:, None])
        channel_xy, belt_xy = bounded(channel_xy), bounded(belt_xy)

    # 7. Far-view belt: effective wet width conserves channel area per belt length.
    s_channel, s_belt = _cumlen(channel_xy), _cumlen(belt_xy)
    k = max(3, 2 * window + 1)
    sinuosity = np.gradient(_moving_average(s_channel, k), s) / np.maximum(
        np.gradient(_moving_average(s_belt, k), s), 1e-9)
    sinuosity = np.clip(sinuosity, 1.0, None)
    separation = (np.linalg.norm(channel_xy-belt_xy, axis=1) if support_bounds is not None
                  else np.abs(total-belt_off))
    halfwidth = gaussian_filter1d(maximum_filter1d(separation, k, mode='nearest'),
                                  max(window / 4, 1.0), mode='nearest') + width / 2
    belt_width = np.minimum(width * sinuosity, 2 * halfwidth)
    if support_bounds is not None:
        # Bound the effective far-view kernel as well as its centreline.
        # Four channel widths retain strongly sinuous belts without making
        # delivery depend on an unbounded numerical arc-length derivative.
        belt_width = np.minimum(belt_width, 4*width)

    level = np.interp(s, s_axis, level_axis)
    if tolerance_m is None:
        tolerance_m = 0.05 * width if sample_spacing_m is None else min(0.05 * width, 0.25 * sample_spacing_m)
        tolerance_m = min(tolerance_m, float(reach.get('tolerance_m', np.inf)))
        tolerance_m = max(tolerance_m, 0.05)
    pin_keep = np.searchsorted(s, pin_s)
    kept = _simplify(channel_xy, tolerance_m, np.r_[pin_keep, np.searchsorted(s, corners)].astype(np.int64))
    kept_belt = _simplify(belt_xy, tolerance_m, pin_keep)
    tangent, _ = _unit_normals(channel_xy)
    # Offsets have zero slope at pins: the exact shared node tangent applies.
    tangent[pin_keep] = control_dir[np.searchsorted(controls, pins)]
    out_xy = frame.inverse(channel_xy[kept])
    tangent_map = tangent[kept].copy()
    tangent_map[:, 0] /= x_scale(out_xy[:, 1], sphere_radius_m)
    tangent_map /= np.maximum(np.hypot(*tangent_map.T), 1e-12)[:, None]
    n_axis = len(s_axis)
    sinuosity_total = s_channel[-1] / max(length, 1e-9)
    diagnostics.update(
        sinuosity=float(sinuosity_total), max_offset_m=float(np.max(np.abs(total))),
        curvature_scale=float(scale.min()), dense_samples=int(len(s)), axis_samples=int(n_axis),
        mean_slope=float(mean_slope), mobility_mean=float(np.mean(mob)))
    morphology = 'nested' if intensity else ('meandering' if sinuosity_total > 1.15 else 'straight')
    return dict(
        base, xy=out_xy, station_m=s_channel[kept], axis_station_m=s[kept],
        water_m=level[kept], width_m=np.full(len(kept), width), depth_m=np.full(len(kept), depth),
        tangent=tangent_map, pins=np.searchsorted(kept, pin_keep),
        belt_xy=frame.inverse(belt_xy[kept_belt]), belt_station_m=s[kept_belt],
        belt_water_m=level[kept_belt], belt_width_m=belt_width[kept_belt],
        belt_halfwidth_m=halfwidth[kept_belt], wavelength_m=wavelength,
        bankfull_width_m=width, bankfull_depth_m=depth, corridor_m=corridor,
        morphology=morphology, diagnostics=diagnostics, version=GEOMETRY_VERSION)


def _axis_course(base, frame, axis, s_axis, level_axis, pin_dense, controls, pins, control_dir,
                 corners, width, depth, wavelength, corridor, sphere_radius_m, reach, tolerance_m,
                 diagnostics):
    """The smoothed axis as the delivered course (no generated bends).

    Same record keys as :func:`refine_reach` except the far-view belt: the
    belt would equal the channel, so rasterization uses the channel alone.
    """
    if tolerance_m is None:
        # A quarter of a native decoder cell: the smooth spline needs no
        # finer tessellation, whatever the channel width.
        tolerance_m = max(min(7.5, float(reach.get('tolerance_m', np.inf))), 0.05)
    keep = np.r_[pin_dense, corners].astype(np.int64)
    kept = _simplify(axis, tolerance_m, keep)
    pin_keep = np.searchsorted(kept, pin_dense)
    tangent, _ = _unit_normals(axis)
    tangent = tangent[kept]
    tangent[pin_keep] = control_dir[np.searchsorted(controls, pins)]
    out_xy = frame.inverse(axis[kept])
    tangent[:, 0] /= x_scale(out_xy[:, 1], sphere_radius_m)
    tangent /= np.maximum(np.hypot(*tangent.T), 1e-12)[:, None]
    n = len(kept)
    diagnostics.update(sinuosity=1.0, max_offset_m=0.0, dense_samples=int(len(axis)),
                       axis_samples=int(len(axis)), output_vertices=int(n))
    return dict(base, xy=out_xy, station_m=s_axis[kept], axis_station_m=s_axis[kept],
                water_m=level_axis[kept], width_m=np.full(n, width), depth_m=np.full(n, depth),
                tangent=tangent, pins=pin_keep, wavelength_m=wavelength,
                bankfull_width_m=width, bankfull_depth_m=depth, corridor_m=corridor,
                morphology='axis', diagnostics=diagnostics, version=GEOMETRY_VERSION)


# ----------------------------------------------------------------------------
# Network: shared node tangents and crossing repair

def _sort_key(reach):
    return str(reach.get('id'))


def _node_tangents(reaches, sphere_radius_m):
    """Main-stem node tangents: bisector of the largest inflow and the outflow."""
    def chord(xy, start):
        xy = np.asarray(xy, np.float64)
        a, b = (xy[0], xy[min(2, len(xy) - 1)]) if start else (xy[max(-3, -len(xy))], xy[-1])
        d = b - a
        d[0] *= float(x_scale(a[1], sphere_radius_m))
        n = math.hypot(*d)
        return d / n if n > 1e-12 else None
    tangents, by_target = {}, {}
    for i, r in enumerate(reaches):
        by_target.setdefault(r.get('target'), []).append(i)
    for i, r in enumerate(reaches):
        incoming = by_target.get(r.get('source'), [])
        if not incoming or len(r['xy']) < 2:
            continue
        main = max(incoming, key=lambda k: (float(reaches[k].get('flow_m3s') or 0),
                                            str(reaches[k].get('id'))))
        a, b = chord(reaches[main]['xy'], False), chord(r['xy'], True)
        if a is None or b is None or float(a @ b) < -0.2:
            continue
        t = (a + b) / max(math.hypot(*(a + b)), 1e-12)
        y = float(np.asarray(r['xy'])[0][1])
        t_map = np.array([t[0] / float(x_scale(y, sphere_radius_m)), t[1]])
        t_map /= math.hypot(*t_map)
        tangents.setdefault(main, {})['end'] = t_map
        tangents.setdefault(i, {})['start'] = t_map
    return tangents


def find_crossings(refined, *, ignore_m=1e-6):
    """Pairs of reach indices whose centrelines properly intersect.

    Uses an axis-aligned bounding-box sweep. Touching at a shared node is not a
    crossing: intersections within ``ignore_m`` of a polyline end, or within
    the two bankfull half-widths of an end both reaches share (the tributary
    is already inside the receiving channel), are ignored.
    """
    segs, owner, local = [], [], []
    for i, r in enumerate(refined):
        xy = np.asarray(r['xy'], np.float64)
        if len(xy) < 2:
            continue
        segs.append(np.concatenate([xy[:-1], xy[1:]], 1))
        owner.append(np.full(len(xy) - 1, i))
        local.append(np.arange(len(xy) - 1))
    if not segs:
        return set()
    seg, owner, local = np.concatenate(segs), np.concatenate(owner), np.concatenate(local)
    lo = np.minimum(seg[:, :2], seg[:, 2:])
    hi = np.maximum(seg[:, :2], seg[:, 2:])
    ends = {i: (np.asarray(r['xy'])[0], np.asarray(r['xy'])[-1])
            for i, r in enumerate(refined) if len(r['xy'])}
    # Every segment pair is considered once. A long diagonal previously
    # occupied thousands of tiny bins and repeated the same pair tests in
    # every bin. The sweep has no dependence on tessellation/bin scale; it
    # runs along the axis on which fewer box intervals overlap.
    def overlaps(axis):
        start = np.sort(lo[:, axis])
        return int(np.sum(np.searchsorted(start, hi[:, axis], 'right')
                          - np.searchsorted(start, lo[:, axis], 'left')))
    axis = int(overlaps(1) < overlaps(0))
    order = np.argsort(lo[:, axis], kind='stable')
    a, b, t = _sweep_crossings(np.ascontiguousarray(seg), lo, hi, owner.astype(np.int64),
                               local.astype(np.int64), order, axis)
    found = set()
    for k in range(len(a)):
        point = seg[a[k], :2] + t[k] * (seg[a[k], 2:] - seg[a[k], :2])
        i, j = int(owner[a[k]]), int(owner[b[k]])
        pair = (min(i, j), max(i, j))
        if pair in found:
            continue
        joint = 0.5 * (float(refined[i].get('bankfull_width_m', 0.0))
                       + float(refined[j].get('bankfull_width_m', 0.0)))
        shared = [e for e in ends[i] if any(np.hypot(*(e - f)) <= 1e-6 for f in ends[j])]
        # A tributary may join a few cells from the receiver's start (for
        # example after an inserted portal point); its mouth is the junction.
        for x, y in ((i, j), (j, i)):
            if x != y and refined[x].get('id') is not None and \
                    refined[x].get('downstream_id') == refined[y].get('id'):
                shared.append(ends[x][1])
        if any(np.hypot(*(point - e)) <= ignore_m for e in (*ends[i], *ends[j])):
            continue
        if any(np.hypot(*(point - e)) <= joint for e in shared):
            continue
        found.add(pair)
    return found


@njit(cache=True)
def _sweep_crossings(seg, lo, hi, owner, local, order, axis):
    """Proper intersections ``(a, b, t)`` of overlapping boxes, swept along ``axis``.

    Excludes adjacent segments of one polyline; ``t`` is the parameter on ``a``.
    """
    other = 1 - axis
    active = np.empty(len(order), np.int64)
    first, second = np.empty(64, np.int64), np.empty(64, np.int64)
    where = np.empty(64)
    count = hits = 0
    for k in order:
        kept = 0
        for m in range(count):
            j = active[m]
            if hi[j, axis] < lo[k, axis]:
                continue
            active[kept] = j
            kept += 1
            if hi[j, other] < lo[k, other] or lo[j, other] > hi[k, other]:
                continue
            if owner[j] == owner[k] and abs(local[j] - local[k]) <= 1:
                continue
            px, py = seg[j, 0], seg[j, 1]
            rx, ry = seg[j, 2] - px, seg[j, 3] - py
            sx, sy = seg[k, 2] - seg[k, 0], seg[k, 3] - seg[k, 1]
            qx, qy = seg[k, 0] - px, seg[k, 1] - py
            den = rx * sy - ry * sx
            if abs(den) <= 1e-18:
                continue
            t = (qx * sy - qy * sx) / den
            u = (qx * ry - qy * rx) / den
            if not (0.0 < t < 1.0 and 0.0 < u < 1.0):
                continue
            if hits == len(first):
                first, second = np.concatenate((first, first)), np.concatenate((second, second))
                where = np.concatenate((where, where))
            first[hits], second[hits], where[hits] = j, k, t
            hits += 1
        count = kept
        active[count] = k
        count += 1
    return first[:hits], second[:hits], where[:hits]


def refine_network(reaches, *, seed, climate=None, max_repairs=8, **options):
    """Refine all reaches with shared main-stem tangents and crossing repair.

    Processing order is by reach id, independent of input order. When two
    refined centrelines cross, the smaller-flow reach is damped
    deterministically: half bends, no bends, no valley fit, then the parent
    polyline itself (steps of disabled features are skipped); then the
    larger reach is damped the same way.
    Remaining and inherited (parent-course) crossings are reported.
    ``climate`` may be a dict of scalars or a callable ``(reach) -> dict``.
    Other options are passed to :func:`refine_reach`.
    Returns ``(refined_in_input_order, diagnostics)``.
    """
    order = sorted(range(len(reaches)), key=lambda i: _sort_key(reaches[i]))
    tangents = _node_tangents(reaches, options.get('sphere_radius_m'))
    state = {i: dict(meander_scale=float(options.get('meander_scale', 1.0)),
                     valley_fit=options.get('valley_fit', True), smooth_axis=True) for i in order}
    kwargs = {k: v for k, v in options.items() if k not in ('valley_fit', 'meander_scale', 'smooth_axis')}
    meanders = options.get('meanders', True)

    def run(i):
        c = climate(reaches[i]) if callable(climate) else climate
        t = tangents.get(i, {})
        return refine_reach(reaches[i], seed=seed, climate=c,
                            start_tangent=t.get('start', reaches[i].get('start_tangent')),
                            end_tangent=t.get('end', reaches[i].get('end_tangent')), **state[i], **kwargs)

    def rank(k):
        return float(reaches[k].get('flow_m3s') or 0), str(reaches[k].get('id'))
    refined = {i: run(i) for i in order}
    # Crossings already present in the parent courses cannot be repaired here.
    inherited = find_crossings(reaches)
    repaired, crossings = [], set()
    for round_ in range(max_repairs + 1):
        crossings = find_crossings([refined[i] for i in range(len(reaches))]) - inherited
        if not crossings or round_ == max_repairs:
            break
        def exhausted(k):
            return not state[k]['smooth_axis']
        victims = set()
        for i, j in crossings:
            first, second = sorted((i, j), key=rank)
            victims.add(second if exhausted(first) else first)
        for i in sorted(victims, key=lambda k: _sort_key(reaches[k])):
            if exhausted(i):
                continue
            if meanders and state[i]['meander_scale'] > 0.5:
                state[i]['meander_scale'] = 0.5
            elif meanders and state[i]['meander_scale'] > 0:
                state[i]['meander_scale'] = 0.0
            elif state[i]['valley_fit']:
                state[i]['valley_fit'] = False
            else:
                state[i]['smooth_axis'] = False
            refined[i] = run(i)
            repaired.append((reaches[i].get('id'), dict(state[i])))
    ids = [r.get('id') for r in reaches]
    return [refined[i] for i in range(len(reaches))], dict(
        repaired=repaired, unresolved_crossings=[(ids[i], ids[j]) for i, j in sorted(crossings)],
        inherited_crossings=[(ids[i], ids[j]) for i, j in sorted(inherited)])


DISPLAY_FLOW_RTOL = 0.01      # annual/monthly change that still counts as one course


def compact_display_reaches(reaches, *, kinds=('trunk',), flow_rtol=DISPLAY_FLOW_RTOL):
    """Join unbranched record chains into longer display courses.

    Hydrologic records are cut wherever a potential tributary may join; once
    annual-dry tributaries are hidden, those cuts are artificial pins every few
    cells, and a reach shorter than its meander wavelength stays straight.
    Record ``A`` is joined with ``B = A['downstream_id']`` only when both are
    in ``reaches``, share ``parent_id`` and a kind in ``kinds``, ``A`` is the
    only listed record draining into ``B`` (so no displayed tributary joins
    the interior), ``A`` ends neither at a lake nor a portal, the junction
    point and level agree, annual and monthly flows agree within
    ``flow_rtol``, and ``navigation_months`` are identical (or absent on
    both). A flow, regime or navigation change keeps the cut rather than
    mixing or relabelling records.

    Returns new records (inputs are not modified) in input order of chain
    heads. A merged record keeps the head's ``id``, ``source``, start
    tangent/portal and navigation (equal on every member), and the tail's
    ``target``, ``downstream_id``, flow, area (drained area at the downstream
    end) and end portal/tangent/kind. ``length_m`` is the members' sum and
    ``slope`` the whole-course drop between the end water levels over it;
    both are dropped when not every member has a length. The duplicated
    junction vertex is dropped; only true interior pins survive. ``members``
    lists the original ids. ``downstream_id`` values pointing into a chain
    are remapped to its id.
    """
    by_id = {r.get('id'): k for k, r in enumerate(reaches)}
    upstream = {}
    for k, r in enumerate(reaches):
        if r.get('downstream_id') in by_id:
            upstream.setdefault(by_id[r['downstream_id']], []).append(k)

    def monthly(r):
        return np.asarray(r.get('monthly_flow_m3s', [r.get('flow_m3s') or 0.0]), np.float64)

    def joins(a, b):
        ra, rb = reaches[a], reaches[b]
        end_kind = str(ra.get('terminal_kind') or (ra.get('target') if isinstance(ra.get('target'), str) else ''))
        if (ra.get('kind') not in kinds or rb.get('kind') != ra.get('kind')
                or ra.get('parent_id') != rb.get('parent_id') or upstream.get(b) != [a]
                or ra.get('end_portal') is not None or rb.get('start_portal') is not None
                or end_kind.startswith(('lake:', 'parent_lake:'))
                or ra.get('end_link') is not None or rb.get('start_link') is not None):
            return False
        xa, xb = np.asarray(ra['xy'], np.float64), np.asarray(rb['xy'], np.float64)
        wa, wb = np.asarray(ra['water_m'], np.float64), np.asarray(rb['water_m'], np.float64)
        if np.hypot(*(xa[-1] - xb[0])) > 1e-3 or abs(wa[-1] - wb[0]) > 1e-3:
            return False
        na, nb = ra.get('navigation_months'), rb.get('navigation_months')
        if (na is None) != (nb is None) or (na is not None and not np.array_equal(np.asarray(na), np.asarray(nb))):
            return False
        qa, qb = monthly(ra), monthly(rb)
        scale = max(abs(float(ra.get('flow_m3s') or 0.0)), float(np.max(np.abs(qa), initial=0.0)), 1e-12)
        return (qa.shape == qb.shape and float(np.max(np.abs(qa - qb), initial=0.0)) <= flow_rtol * scale
                and abs(float(ra.get('flow_m3s') or 0.0) - float(rb.get('flow_m3s') or 0.0)) <= flow_rtol * scale)

    follows = {a: by_id[r['downstream_id']] for a, r in enumerate(reaches)
               if r.get('downstream_id') in by_id and joins(a, by_id[r['downstream_id']])}
    absorbed = set(follows.values())
    chains, owner = [], {}
    for head in range(len(reaches)):
        if head in absorbed:
            continue
        chain, k = [head], head
        while k in follows and follows[k] not in chain:
            k = follows[k]
            chain.append(k)
        for k in chain:
            owner[reaches[k].get('id')] = reaches[head].get('id')
        chains.append(chain)
    # A pure cycle of joinable records has no head; keep such records as is.
    for k in range(len(reaches)):
        if reaches[k].get('id') not in owner:
            owner[reaches[k].get('id')] = reaches[k].get('id')
            chains.append([k])

    out = []
    for chain in chains:
        head, tail = reaches[chain[0]], reaches[chain[-1]]
        merged = dict(head, members=[head.get('id')])
        if len(chain) > 1:
            parts = [reaches[k] for k in chain]
            xy, water, pins, profile, cells = [], [], [], [], []
            for n, r in enumerate(parts):
                skip = 1 if n else 0
                offset = sum(len(v) for v in xy) - skip
                p_xy = np.asarray(r['xy'], np.float64)
                xy.append(p_xy[skip:])
                water.append(np.asarray(r['water_m'], np.float64)[skip:])
                pins.extend(offset + int(p) % len(p_xy) for p in r.get('pins', ())
                            if 0 < int(p) % len(p_xy) < len(p_xy) - 1)
                if r.get('flow_profile_m3s') is not None:
                    profile.append(np.asarray(r['flow_profile_m3s'], np.float64)[skip:])
                if r.get('cells') is not None:
                    c = np.asarray(r['cells'])
                    cells.append(c[1:] if cells and len(c) and len(cells[-1]) and c[0] == cells[-1][-1] else c)
            xy = np.concatenate(xy)
            merged.update(xy=xy, water_m=np.concatenate(water), pins=[0, *pins, len(xy) - 1],
                          members=[r.get('id') for r in parts])
            if len(profile) == len(parts):
                merged['flow_profile_m3s'] = np.concatenate(profile)
            if len(cells) == len(parts):
                merged['cells'] = np.concatenate(cells)
            for key in ('target', 'downstream_id', 'flow_m3s', 'area_m2', 'monthly_flow_m3s', 'min_month_flow_m3s',
                        'end_portal', 'end_tangent', 'end_link', 'terminal_kind'):
                if key in tail or key in merged:
                    merged[key] = tail.get(key)
            if all(r.get('length_m') is not None for r in parts):
                merged['length_m'] = float(sum(float(r['length_m']) for r in parts))
                if any('slope' in r for r in parts):
                    drop = float(merged['water_m'][0] - merged['water_m'][-1])
                    merged['slope'] = max(drop, 0.0) / max(merged['length_m'], 1.0)
            else:
                merged.pop('length_m', None)
                merged.pop('slope', None)
        if merged.get('downstream_id') in owner:
            merged['downstream_id'] = owner[merged['downstream_id']]
        out.append(merged)
    return out


def simplify_reach(refined, tolerance_m):
    """Coarser display tessellation of the same refined curve (pins retained)."""
    xy = np.asarray(refined['xy'], np.float64)
    keep = _simplify(xy, tolerance_m, refined.get('pins', ()))
    out = dict(refined)
    for key in ('xy', 'station_m', 'axis_station_m', 'water_m', 'width_m', 'depth_m', 'tangent'):
        out[key] = np.asarray(refined[key])[keep]
    out['pins'] = np.searchsorted(keep, np.asarray(refined.get('pins', ()), np.int64))
    return out


# ----------------------------------------------------------------------------
# Lakes and still water

def lake_rings(field, bounds, *, level=None, region=None, margin_m=1e-3):
    """Shoreline rings by marching squares on the cell-centre lattice.

    ``field`` is either a boolean wet mask (contour at 0.5) or a height grid
    with ``level`` (wet where height < level). ``region`` restricts the wet
    side to one owned water body; dry cells enclosed by it become island
    holes. Rings are closed (the window is padded dry); each is a dict with
    ``xy`` (n, 2) map metres, signed ``area_m2`` (positive outer, negative
    hole), ``hole`` and ``touches_edge`` (closed artificially by the window).
    Overlapping windows of one grid give identical segments in shared cells.
    """
    field = np.asarray(field)
    if field.dtype == bool:
        v = field.astype(np.float64) - 0.5
    else:
        if level is None:
            raise ValueError('level is required for a height field')
        v = float(level) - field.astype(np.float64)
        if region is not None:
            v = np.where(region, np.maximum(v, margin_m), np.minimum(v, -margin_m))
    v = np.where(v == 0, -1e-12, v)
    rows, cols = v.shape
    dx, dy = grid_spacing(v.shape, bounds)
    p = np.full((rows + 2, cols + 2), -1.0)
    p[1:-1, 1:-1] = v

    def edge_point(r, c, horizontal):
        a = p[r, c]
        b = p[r, c + 1] if horizontal else p[r + 1, c]
        t = a / (a - b)
        rr, cc = (r, c + t) if horizontal else (r + t, c)
        return bounds[0] + (cc - 0.5) * dx, bounds[1] + (rr - 0.5) * dy

    inside = p > 0
    a, b = inside[:-1, :-1], inside[:-1, 1:]
    c, d = inside[1:, 1:], inside[1:, :-1]
    case = a * 1 + b * 2 + c * 4 + d * 8
    starts, stops, coords = [], [], {}
    for r, col in zip(*np.nonzero((case != 0) & (case != 15))):
        edges = {'T': ((r, col, True), a[r, col] != b[r, col]),
                 'R': ((r, col + 1, False), b[r, col] != c[r, col]),
                 'B': ((r + 1, col, True), c[r, col] != d[r, col]),
                 'L': ((r, col, False), d[r, col] != a[r, col])}
        crossed = [k for k in 'TRBL' if edges[k][1]]
        if len(crossed) == 4:   # saddle: decide by the cell-centre mean
            centre = p[r:r + 2, col:col + 2].mean() > 0
            pairs = (('T', 'R'), ('B', 'L')) if centre == bool(a[r, col]) else (('L', 'T'), ('R', 'B'))
        else:
            pairs = (tuple(crossed),)
        q = p[r:r + 2, col:col + 2]
        for e1, e2 in pairs:
            k1, k2 = edges[e1][0], edges[e2][0]
            for key in (k1, k2):
                if key not in coords:
                    coords[key] = edge_point(*key)
            p1, p2 = np.array(coords[k1]), np.array(coords[k2])
            mid = (p1 + p2) * 0.5
            u = (mid[0] - bounds[0]) / dx + 0.5 - col
            w = (mid[1] - bounds[1]) / dy + 0.5 - r
            gx = ((q[0, 1] - q[0, 0]) * (1 - w) + (q[1, 1] - q[1, 0]) * w) / dx
            gy = ((q[1, 0] - q[0, 0]) * (1 - u) + (q[1, 1] - q[0, 1]) * u) / dy
            delta = p2 - p1
            if delta[0] * gy - delta[1] * gx < 0:   # wet side always on one side
                k1, k2 = k2, k1
            starts.append(k1)
            stops.append(k2)
    following = {}
    for i, key in enumerate(starts):
        following.setdefault(key, []).append(i)
    used = np.zeros(len(starts), bool)
    rings = []
    lo_x, lo_y = bounds[0] + 0.5 * dx, bounds[1] + 0.5 * dy
    hi_x, hi_y = bounds[2] - 0.5 * dx, bounds[3] - 0.5 * dy
    for first in range(len(starts)):
        if used[first]:
            continue
        chain, i = [], first
        while not used[i]:
            used[i] = True
            chain.append(coords[starts[i]])
            options = [k for k in following.get(stops[i], ()) if not used[k]]
            if not options:
                break
            i = options[0]
        if len(chain) < 3:
            continue
        xy = np.asarray(chain, np.float64)
        area = 0.5 * float(np.sum(xy[:, 0] * np.roll(xy[:, 1], -1) - np.roll(xy[:, 0], -1) * xy[:, 1]))
        edge = bool(np.any((xy[:, 0] < lo_x - 1e-9) | (xy[:, 0] > hi_x + 1e-9)
                           | (xy[:, 1] < lo_y - 1e-9) | (xy[:, 1] > hi_y + 1e-9)))
        rings.append(dict(xy=xy, area_m2=area, touches_edge=edge))
    for ring in rings:
        # The wet side is kept on one side of travel, so outer rings have a
        # positive (x, y-south) shoelace area and island holes a negative one.
        ring['hole'] = ring['area_m2'] < 0
    rings.sort(key=lambda q: (-abs(q['area_m2']), float(q['xy'][:, 0].min()), float(q['xy'][:, 1].min())))
    return rings


def lake_shore_rings(lake, grid, *, margin_cells=2):
    """Rings of one lake record from the core grid (``height``/``height_m``,
    ``lake_ids`` or the lake's ``cells``, ``bounds``), in the lake's window."""
    height = np.asarray(grid.get('height_m', grid.get('height')), np.float64)
    rows, cols = height.shape
    cells = (np.asarray(lake['cells'], np.int64) if lake.get('cells') is not None
             else np.flatnonzero(np.asarray(grid['lake_ids']) == lake['id']))
    if not len(cells):
        return []
    r, c = np.divmod(cells, cols)
    r0, r1 = max(r.min() - margin_cells, 0), min(r.max() + margin_cells + 1, rows)
    c0, c1 = max(c.min() - margin_cells, 0), min(c.max() + margin_cells + 1, cols)
    dx, dy = grid_spacing(height.shape, grid['bounds'])
    x0, y0 = grid['bounds'][0], grid['bounds'][1]
    window = (x0 + c0 * dx, y0 + r0 * dy, x0 + c1 * dx, y0 + r1 * dy)
    region = np.zeros((r1-r0, c1-c0), bool)
    region[r-r0, c-c0] = True
    return lake_rings(height[r0:r1, c0:c1], window, level=lake['level_m'], region=region)


def _lake_cells(lake, lake_ids, shape):
    if lake.get('cells') is not None:
        mask = np.zeros(int(np.prod(shape)), bool)
        mask[np.asarray(lake['cells'], np.int64)] = True
        return mask.reshape(shape)
    return np.asarray(lake_ids) == lake['id']


def _shift(a, dr, dc, fill, wrap_x):
    """Value of the neighbour at (r + dr, c + dc) for each cell."""
    out = a
    if dc:
        if wrap_x:
            out = np.roll(out, -dc, 1)
        else:
            out = np.roll(out, -dc, 1)
            out[:, -dc if dc > 0 else 0:None if dc > 0 else -dc] = fill
    if dr:
        out = np.roll(out, -dr, 0)
        out[-dr if dr > 0 else 0:None if dr > 0 else -dr] = fill
    return out


def prepare_water_grid(height_m, bounds, *, lakes=(), lake_ids=None, ocean=None,
                       sea_level_m=0.0, wrap_x=False, margin_m=1e-3):
    """Signed still-water field on the authoritative (core) grid.

    ``v = level - ground`` is >= ``margin_m`` in owned wet cells (ocean
    components and lake cells) and <= ``-margin_m`` elsewhere, so inland
    depressions below sea level stay dry unless they are lakes. Dry cells
    within two cells of water inherit the neighbouring level (fixed neighbour
    priority) to place the zero crossing. Consumed by :func:`rasterize_water`.
    """
    height = np.asarray(height_m, np.float64)
    level = np.full(height.shape, np.nan)
    kind = np.zeros(height.shape, np.int8)
    index = np.full(height.shape, -1, np.int32)
    if ocean is not None:
        ocean = np.asarray(ocean, bool)
        level[ocean], kind[ocean] = sea_level_m, OCEAN
    for k, lake in enumerate(lakes):
        if lake.get('cells') is not None:
            cells = np.asarray(lake['cells'], np.int64)
            level.ravel()[cells], kind.ravel()[cells], index.ravel()[cells] = float(lake['level_m']), LAKE, k
        else:
            cells = _lake_cells(lake, lake_ids, height.shape)
            level[cells], kind[cells], index[cells] = float(lake['level_m']), LAKE, k
    wet = kind > 0
    known = wet.copy()
    neighbours = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    for _ in range(2):
        got = np.zeros(height.shape, bool)
        new = [level.copy(), kind.copy(), index.copy()]
        for dr, dc in neighbours:
            take = ~known & ~got & _shift(known, dr, dc, False, wrap_x)
            for array, source, fill in zip(new, (level, kind, index), (np.nan, 0, -1)):
                array[take] = _shift(source, dr, dc, fill, wrap_x)[take]
            got |= take
        level, kind, index = new
        known |= got
    v = np.where(wet, np.maximum(level - height, margin_m), -1e4)
    shore = known & ~wet
    v[shore] = np.minimum(level[shore] - height[shore], -margin_m)
    return dict(v=v, level=level, kind=kind, index=index, height=height,
                bounds=tuple(float(b) for b in bounds), wrap_x=bool(wrap_x),
                sea_level_m=float(sea_level_m))


# ----------------------------------------------------------------------------
# Rasterization into arbitrary target grids

def _ribbon_pixels(xy, half, extra, xs, ys, kx, shifts, max_pairs=4_000_000):
    """Nearest-segment search for pixels within ``half + extra`` of a polyline.

    Distances are physical (east-west scaled by ``kx`` per row). Returns flat
    pixel ids, distance, segment index and segment parameter (ties broken by
    distance, then segment index).
    """
    rows, cols = len(ys), len(xs)
    dx = xs[1] - xs[0] if cols > 1 else 1.0
    dy = ys[1] - ys[0] if rows > 1 else 1.0
    kmin = float(np.min(kx))
    radius = np.maximum(half[:-1], half[1:]) + extra
    parts = []
    for shift in shifts:
        a, b = xy[:-1] + (shift, 0.0), xy[1:] + (shift, 0.0)
        rx = radius / kmin
        j0 = np.clip(np.ceil((np.minimum(a[:, 0], b[:, 0]) - rx - xs[0]) / dx), 0, cols).astype(np.int64)
        j1 = np.clip(np.floor((np.maximum(a[:, 0], b[:, 0]) + rx - xs[0]) / dx), -1, cols - 1).astype(np.int64)
        i0 = np.clip(np.ceil((np.minimum(a[:, 1], b[:, 1]) - radius - ys[0]) / dy), 0, rows).astype(np.int64)
        i1 = np.clip(np.floor((np.maximum(a[:, 1], b[:, 1]) + radius - ys[0]) / dy), -1, rows - 1).astype(np.int64)
        nj, ni = np.maximum(j1 - j0 + 1, 0), np.maximum(i1 - i0 + 1, 0)
        counts = nj * ni
        segments = np.flatnonzero(counts)
        start = 0
        while start < len(segments):
            cum = np.cumsum(counts[segments[start:]])
            stop = start + max(1, int(np.searchsorted(cum, max_pairs, side='right')))
            group = segments[start:stop]
            start = stop
            c = counts[group]
            seg = np.repeat(group, c)
            offset = np.arange(len(seg)) - np.repeat(np.cumsum(c) - c, c)
            ii = i0[seg] + offset // nj[seg]
            jj = j0[seg] + offset % nj[seg]
            k = kx[ii]
            ex, ey = (b[seg, 0] - a[seg, 0]) * k, b[seg, 1] - a[seg, 1]
            qx, qy = (xs[jj] - a[seg, 0]) * k, ys[ii] - a[seg, 1]
            t = np.clip((qx * ex + qy * ey) / np.maximum(ex * ex + ey * ey, 1e-30), 0.0, 1.0)
            d = np.hypot(qx - t * ex, qy - t * ey)
            near = d <= half[seg] + (half[seg + 1] - half[seg]) * t + extra
            parts.append((ii[near] * cols + jj[near], d[near], seg[near], t[near]))
    if not parts:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0, np.int64), np.zeros(0)
    pix, d, seg, t = (np.concatenate(x) for x in zip(*parts))
    if not len(pix):
        return pix, d, seg, t
    order = np.lexsort((seg, d, pix))
    pix, d, seg, t = pix[order], d[order], seg[order], t[order]
    first = np.r_[True, pix[1:] != pix[:-1]]
    return pix[first], d[first], seg[first], t[first]


def _box_coverage(distance, width, cell):
    """Area fraction of a straight ribbon within a pixel's 1-D box footprint."""
    lo = np.maximum(distance - cell / 2, -width / 2)
    hi = np.minimum(distance + cell / 2, width / 2)
    return np.clip(hi - lo, 0.0, None) / cell


def _rasterize_reach(reach, xs, ys, kx, cell_flat, cell_typ, shifts_for, bank_widths):
    xy = np.asarray(reach['xy'], np.float64)
    flow = float(reach.get('flow_m3s') or 0.0)
    w_default, d_default = (float(v) for v in bankfull_geometry(flow))
    width = np.asarray(reach.get('width_m', np.full(len(xy), w_default)), np.float64)
    depth = np.asarray(reach.get('depth_m', np.full(len(xy), d_default)), np.float64)
    level = np.asarray(reach.get('water_m', np.zeros(len(xy))), np.float64)
    wavelength = float(reach.get('wavelength_m', MEANDER_WAVELENGTH_WIDTHS * width.mean()))
    w_far = float(_smoothstep(0.25 * wavelength, wavelength, cell_typ)) if 'belt_xy' in reach else 0.0
    reps = []
    if w_far < 1.0:
        reps.append((1.0 - w_far, xy, width, depth, level))
    if w_far > 0.0:
        belt = np.asarray(reach['belt_xy'], np.float64)
        reps.append((w_far, belt, np.asarray(reach['belt_width_m'], np.float64),
                     np.full(len(belt), float(depth.mean())), np.asarray(reach['belt_water_m'], np.float64)))
    shifts = shifts_for(min(r[1][:, 0].min() for r in reps), max(r[1][:, 0].max() for r in reps))
    acc = None
    for weight, pxy, pw, pd, pl in reps:
        extra = float(cell_flat.max()) + bank_widths * float(pw.max())
        pix, d, seg, t = _ribbon_pixels(pxy, pw / 2, extra, xs, ys, kx, shifts)
        if not len(pix):
            continue
        w = pw[seg] + (pw[seg + 1] - pw[seg]) * t
        tangent = pxy[seg + 1] - pxy[seg]
        tangent /= np.maximum(np.hypot(*tangent.T), 1e-12)[:, None]
        rep = dict(pix=pix, cov=_box_coverage(d, w, cell_flat[pix]) * weight, d=d, w=w,
                   D=pd[seg] + (pd[seg + 1] - pd[seg]) * t, level=pl[seg] + (pl[seg + 1] - pl[seg]) * t,
                   tx=tangent[:, 0], ty=tangent[:, 1], weight=np.full(len(pix), weight))
        if acc is None:
            acc = rep
            continue
        # Blend representations: coverage adds (weights sum to one), other
        # attributes come from the representation contributing more.
        union = np.union1d(acc['pix'], rep['pix'])
        merged = {}
        for key in rep:
            if key == 'pix':
                continue
            fill = np.inf if key == 'd' else 0.0
            pair = []
            for source in (acc, rep):
                values = np.full(len(union), fill)
                values[np.searchsorted(union, source['pix'])] = source[key]
                pair.append(values)
            merged[key] = pair
        prefer = (merged['cov'][1] > merged['cov'][0]) | (
            (merged['cov'][1] == merged['cov'][0]) & (merged['weight'][1] > merged['weight'][0]))
        acc = {key: np.where(prefer, b_, a_) for key, (a_, b_) in merged.items()}
        acc['cov'] = merged['cov'][0] + merged['cov'][1]
        acc['pix'] = union
    return acc, flow, w_far


def rasterize_water(reaches, lakes, shape, bounds, *, water_grid=None, grid=None, terrain=None,
                    sphere_radius_m=None, wrap_period_m=None, sea_level_m=0.0,
                    distance_limit_m=1.0e6, bank_widths=0.75, shore_band_m=None, shore_soft_m=1.0,
                    still_coordinates=None):
    """Water fields on any target grid (window, crop or mip) of the map.

    ``reaches`` are refined reach dicts (raw core reaches also work with their
    ``xy`` and bankfull width from flow). Still water comes from
    ``water_grid`` (:func:`prepare_water_grid`) or the core ``grid`` dict
    (``height``/``height_m``, ``lake_ids``, ``ocean``, ``bounds``) with
    ``lakes``. Values derive from global geometry and the pixel centre only,
    so overlapping crops on one lattice agree. ``terrain`` (target-grid
    ground) gives still-water depth and refines shorelines inside a fixed
    ``shore_band_m`` (default one parent cell) around the parent shore, with
    a ``shore_soft_m`` height ramp; level and owner never change.

    Rivers transition smoothly from the resolved channel (pixel <= wavelength
    / 4) to the belt (pixel >= wavelength), whose effective width conserves
    wet area: far views show averaged coverage, never a flooded belt.

    Fields: ``class`` uint8 (0 dry, 1 ocean, 2 lake, 3 river; the kind with
    the larger coverage), ``coverage`` [0, 1], ``water_level_m`` (NaN dry),
    ``depth_m``, ``flow_x``/``flow_y`` (m3/s times the downstream unit
    tangent, map axes), ``river_coverage``, ``distance_m`` (nearest river
    centreline, ``distance_limit_m`` beyond the search band), ``width_m``,
    ``bankfull_depth_m``, ``river_level_m``, ``resolved`` (0..1),
    ``reach_index``, ``still_coverage``, ``still_level_m``, ``lake_index``
    (-1 none) and per-row physical ``cell_m``.
    """
    rows, cols = shape
    xs, ys = cell_centres(shape, bounds)
    dx, dy = grid_spacing(shape, bounds)
    kx = x_scale(ys, sphere_radius_m)
    cell_row = np.sqrt(np.abs(dx * kx * dy))
    cell_flat = np.repeat(cell_row, cols)
    cell_typ = float(np.median(cell_row))
    n = rows * cols
    river_cov, distance = np.zeros(n), np.full(n, float(distance_limit_m))
    width_f, bank_d, resolved = np.zeros(n), np.zeros(n), np.zeros(n)
    river_level = np.full(n, np.nan)
    flow_x, flow_y = np.zeros(n), np.zeros(n)
    reach_index = np.full(n, -1, np.int32)

    def shifts_for(lo, hi):
        if not wrap_period_m:
            return [0.0]
        margin = 0.01 * wrap_period_m
        return [k * wrap_period_m for k in (-1, 0, 1)
                if hi + k * wrap_period_m >= bounds[0] - margin and lo + k * wrap_period_m <= bounds[2] + margin]

    for ri, reach in enumerate(reaches):
        if len(reach['xy']) < 2:
            continue
        acc, flow, w_far = _rasterize_reach(reach, xs, ys, kx, cell_flat, cell_typ, shifts_for, bank_widths)
        if acc is None:
            continue
        pix = acc['pix']
        better = (acc['cov'] > river_cov[pix]) | ((acc['cov'] == river_cov[pix]) & (acc['d'] < distance[pix]))
        p = pix[better]
        river_cov[p] = np.minimum(acc['cov'][better], 1.0)
        distance[p] = acc['d'][better]
        width_f[p] = acc['w'][better]
        bank_d[p] = acc['D'][better]
        river_level[p] = acc['level'][better]
        flow_x[p], flow_y[p] = acc['tx'][better] * flow, acc['ty'][better] * flow
        reach_index[p] = ri
        resolved[p] = _smoothstep(0.5, 2.0, acc['w'][better] / cell_flat[p]) * (1 - w_far)

    if water_grid is None and grid is not None:
        water_grid = prepare_water_grid(
            grid.get('height_m', grid.get('height')), grid['bounds'], lakes=lakes,
            lake_ids=grid.get('lake_ids'), ocean=grid.get('ocean'), sea_level_m=sea_level_m,
            wrap_x=bool(wrap_period_m))
    still_cov, still_level = np.zeros(n), np.full(n, np.nan)
    still_kind = np.zeros(n, np.int8)
    lake_index = np.full(n, -1, np.int32)
    ground = None
    if water_grid is not None:
        X, Y = (a.ravel() for a in np.meshgrid(xs, ys))
        wb, wrap = water_grid['bounds'], water_grid['wrap_x']
        source_x, source_y = still_coordinates(X, Y) if still_coordinates else (X, Y)
        v, gx, gy = sample_bilinear(water_grid['v'], wb, source_x, source_y, wrap_x=wrap, gradient=True)
        if still_coordinates:
            # Derivatives in the display chart, sampled from one geographic
            # authority. Rotating a second atlas would move the shoreline.
            def shifted(ox, oy):
                a, b = still_coordinates(X+ox, Y+oy)
                return sample_bilinear(water_grid['v'], wb, a, b, wrap_x=wrap)
            gx = (shifted(dx*.5, 0.)-shifted(-dx*.5, 0.))/dx
            gy = (shifted(0., dy*.5)-shifted(0., -dy*.5))/dy
        slope = np.hypot(gx, gy) * math.sqrt(abs(dx * dy))
        with np.errstate(divide='ignore', invalid='ignore'):
            cov = np.where(slope > 1e-12, np.clip(0.5 + v / slope, 0.0, 1.0), (v > 0).astype(np.float64))
        kind = sample_nearest(water_grid['kind'], wb, source_x, source_y, wrap_x=wrap)
        near_level = sample_nearest(water_grid['level'], wb, source_x, source_y, wrap_x=wrap)
        ground = sample_bilinear(water_grid['height'], wb, source_x, source_y, wrap_x=wrap)
        if terrain is not None:
            # Fine terrain refines the shore only inside a fixed band around
            # the parent shore; the level and the owner stay the parent's.
            ground = np.asarray(terrain, np.float64).ravel()
            band = shore_band_m or math.sqrt(abs(np.prod(grid_spacing(water_grid['v'].shape, wb))))
            with np.errstate(divide='ignore', invalid='ignore'):
                signed = np.where(np.hypot(gx, gy) > 1e-12, v / np.hypot(gx, gy), np.sign(v) * np.inf)
                fine = np.clip(0.5 + (near_level - ground) / shore_soft_m, 0.0, 1.0)
            weight = 1.0 - _smoothstep(0.5 * band, band, np.abs(signed))
            cov = np.where(np.isfinite(near_level), (1 - weight) * cov + weight * fine, cov)
        still_cov = np.where(kind > 0, cov, 0.0)
        still_kind = np.where(still_cov > 0, kind, 0).astype(np.int8)
        still_level = np.where(still_cov > 0, near_level, np.nan)
        lake_index = np.where(still_kind == LAKE, sample_nearest(water_grid['index'], wb, source_x, source_y, wrap_x=wrap),
                              -1).astype(np.int32)
    elif terrain is not None:
        ground = np.asarray(terrain, np.float64).ravel()

    is_river = (river_cov > still_cov) & (river_cov > 1e-4)
    is_still = ~is_river & (still_cov > 1e-4)
    klass = np.where(is_river, RIVER, np.where(is_still, still_kind, DRY)).astype(np.uint8)
    coverage = np.where(klass > 0, np.maximum(river_cov, still_cov), 0.0)
    water_level = np.where(is_river, river_level, np.where(is_still, still_level, np.nan))
    with np.errstate(divide='ignore', invalid='ignore'):
        section = np.where(width_f > 0, bank_d * np.clip(1 - (2 * distance / width_f) ** 2, 0, 1), 0.0)
    depth = np.where(is_river, section, 0.0)
    if ground is not None:
        depth = np.where(is_still, np.maximum(still_level - ground, 0.0), depth)
    f32 = lambda a: np.asarray(a, np.float32).reshape(shape)
    return {
        'class': klass.reshape(shape), 'coverage': f32(coverage),
        'water_level_m': f32(water_level), 'depth_m': f32(depth),
        'flow_x': f32(np.where(is_river, flow_x, 0.0)), 'flow_y': f32(np.where(is_river, flow_y, 0.0)),
        'river_coverage': f32(river_cov), 'distance_m': f32(distance), 'width_m': f32(width_f),
        'bankfull_depth_m': f32(bank_d), 'river_level_m': f32(river_level),
        'resolved': f32(resolved), 'reach_index': reach_index.reshape(shape),
        'still_coverage': f32(still_cov), 'still_level_m': f32(still_level),
        'lake_index': lake_index.reshape(shape), 'cell_m': cell_row.astype(np.float32),
        'version': GEOMETRY_VERSION,
    }


# ----------------------------------------------------------------------------
# Bounded surface conditioning by fixed admissible envelopes

def surface_envelopes(raw_height, fields, *, bank_slope=0.5, lake_depth_m=0.25, max_bank_fill_m=2.0,
                      max_cut_m=None, max_still_cut_m=None):
    """Fixed lower/upper admissible heights ``(L, U)`` from raw inputs.

    River: ``U = level - a D (1 - (2d/W)^2)`` inside the bankfull width and
    ``level + (d - W/2) bank_slope / a`` outside, where ``a`` is the section
    resolvability (0 for sub-pixel channels: no carve at far LOD; coverage
    carries them). Still water with coverage >= 0.5: ``U = level -
    lake_depth_m`` (one shared level; islands come from the shore field).
    Banks within ``0.75 W`` of resolved channels get ``L = level`` when that
    raises raw ground by at most ``max_bank_fill_m``. ``L <= U`` always.
    """
    raw = np.asarray(raw_height, np.float64)
    width = np.asarray(fields['width_m'], np.float64)
    d = np.asarray(fields['distance_m'], np.float64)
    level = np.asarray(fields['river_level_m'], np.float64)
    depth = np.asarray(fields['bankfull_depth_m'], np.float64)
    a = np.asarray(fields['resolved'], np.float64)
    river = (width > 0) & np.isfinite(level) & (a > 0)
    half = np.maximum(width / 2, 1e-9)
    with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        inner = level - a * depth * np.clip(1 - (d / half) ** 2, 0, 1)
        outer = level + (d - half) * bank_slope / np.maximum(a, 1e-6)
    upper = np.where(river, np.where(d < half, inner, outer), np.inf)
    still_level = np.asarray(fields['still_level_m'], np.float64)
    still = (np.asarray(fields['still_coverage']) >= 0.5) & np.isfinite(still_level)
    if max_still_cut_m is not None:
        still &= raw-still_level <= max_still_cut_m
    upper = np.where(still, np.minimum(upper, still_level - lake_depth_m), upper)
    bank = river & (a > 0.5) & (d >= half) & (d < half + 0.75 * width) & ~still
    bank &= (level - raw) <= max_bank_fill_m
    lower = np.minimum(np.where(bank, level, -np.inf), upper)
    if max_cut_m is not None:
        upper = np.maximum(upper, raw-max_cut_m)
    return lower, upper


def project_surface(height, lower, upper):
    """``Phi(h) = max(L, min(h, U))``; idempotent for fixed ``L <= U``."""
    return np.maximum(lower, np.minimum(np.asarray(height, np.float64), upper))


def condition_surface(raw_height, fields, *, envelopes=None, **options):
    """Project raw terrain onto fixed water envelopes and refresh water fields.

    Returns ``(height, fields, info)``. Delivered ``class``, ``coverage``,
    ``water_level_m`` and ``depth_m`` are recomputed from the projected
    height (still water whose ground stays at/above its level becomes dry).
    ``info['envelopes']`` are the fixed ``(L, U)``; passing them back
    re-applies the identical projection. Raw input arrays are not modified.
    """
    raw = np.asarray(raw_height, np.float64)
    lower, upper = envelopes if envelopes is not None else surface_envelopes(raw, fields, **options)
    height = project_surface(raw, lower, upper)
    out = dict(fields)
    klass = np.asarray(fields['class']).copy()
    level = np.asarray(fields['water_level_m'], np.float64)
    still = (klass == OCEAN) | (klass == LAKE)
    # A bounded excavation can leave a resolved bed above its prescribed
    # water surface. Report that conflict and keep the physical raster dry;
    # painting a zero-depth ribbon would conceal an impossible channel.
    resolved_river = (klass == RIVER) & (np.asarray(fields['resolved']) > .5)
    with np.errstate(invalid='ignore'):
        blocked_river = resolved_river & ~(height < level)
        klass[(still | resolved_river) & ~(height < level)] = DRY
    wet = klass > 0
    out['class'] = klass
    out['coverage'] = np.where(wet, fields['coverage'], 0).astype(np.float32)
    out['water_level_m'] = np.where(wet, level, np.nan).astype(np.float32)
    out['depth_m'] = np.where(wet, np.maximum(level - height, 0.0), 0.0).astype(np.float32)
    change = height - raw
    return height, out, dict(envelopes=(lower, upper), max_cut_m=float(max(-change.min(initial=0.0), 0.0)),
                             max_fill_m=float(max(change.max(initial=0.0), 0.0)),
                             unresolved_river_cells=int(np.count_nonzero(blocked_river)),
                             changed_cells=int(np.count_nonzero(change)))

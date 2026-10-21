"""Regional 240 m hydrology refinement inside inherited global ownership.

``refine_region`` refines one fixed, globally aligned piece of the global
``HydroWorld`` against canonical neural heights: the 7680 m coarse field and
the 240 m base reconstruction. The global graph stays authoritative:

* Routing height is the base detail aligned onto the coarse field: each
  7680 m block keeps (approximately) its coarse mean, the base adds its residual
  valleys. No new macro graph is solved on the base DEM.
* Ownership comes from the parent: each 7680 m block takes the basin and
  incremental unit (reach, lake or coastal/closed basin) of the source cell
  containing its centre. Water never crosses these labels except along
  inherited trunks and declared portals.
* Parent trunks are re-routed by a least-cost valley path inside their
  corridor, between shared portals (pure functions of the parent reach, the
  cut line and a fixed strip) or node anchors. Local tributaries are a
  Priority-Flood rooted at those trunks, parent lakes and the parent sea.
* Water is a partition of the parent lateral yield; trunk flows interpolate
  the parent flows between portals, so nothing is counted twice.
* A unit fragment whose own reach lies in another piece joins an upstream
  trunk of the same lineage inside the piece, or leaves through a shared
  unit link (pure block-level contract) that the neighbour injects.
* Parent lakes and the sea seed at their water levels; the conditioned
  surface is a bounded local carve whose edge strips use shared records only.

All coordinates are map metres, y south, lattices aligned at 0 (7680 = 32 x
240, 240 = 8 x 30). Method note: docs/methods/hydrology-regions.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import bisect
import hashlib
import json
import math
import time

import numpy as np
from numba import njit
from scipy import ndimage
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

REGIONS_VERSION = 'hydro-regions-v3'
COARSE_M, BASE_M, DECODER_M = 7680.0, 240.0, 30.0
BLOCK = 32                       # base cells per coarse block
POLYLINE_TIE_OFFSET_M = np.array([0.0137, 0.0291])
SEA_SEED, PARENT_LAKE_SEED, EDGE_SEED, TERMINAL_SEED, TRUNK_SEED, LINK_SEED = 1, 2, 3, 4, 5, 6
NETWORK, ORPHAN, TERMINAL = 1, 3, 4   # drain_class: network (incl. links) / unused since v2 / local pit
ORPHAN_KEY = 1 << 40             # flood keys of linked fragments (never a parent key)
UNIT_BLOCK_LIMIT = 20000         # larger units are not linked (pure: depends on the unit only)
DI = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
DJ = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
# Practical bankfull power laws, Q in m3/s; provenance in hydrology-geometry.md.
WIDTH_COEFFICIENT, WIDTH_EXPONENT = 7.2, 0.50
DEPTH_COEFFICIENT, DEPTH_EXPONENT = 0.27, 0.39

DEFAULTS = dict(
    corridor_m=None,            # trunk corridor half-width; default 0.75 source cell
    anchor_radius_m=None,       # node anchor search radius; default one coarse block
    valley_radius_cells=8,      # valley floor = local minimum over this radius
    relief_scale_m=10.0,        # height above valley floor costing one extra path length
    deviation_weight=2.0,       # cost at the corridor edge, in path lengths
    climb_weight=24.0,          # path metres charged per metre climbed downstream
    portal_deviation_m=20.0,    # height-equivalent penalty at the corridor edge (portals, anchors)
    max_incision_m=30.0,        # retained API option; coarse profiles own the incision budget
    strong_incision_m=60.0,     # incision above this is a flagged strong correction
    min_channel_area_m2=1e7,    # tributary initiation area
    min_lake_depth_m=1.0,
    min_lake_cells=4,
    bank_fill_m=2.0,            # displayed water may stand this far above its own channel ground
    bank_slope=0.1,             # upper-envelope rise per map metre away from a channel
    bank_radius_cells=3,        # support of a channel carve (cut tapers to 0)
    edge_strip_cells=4,         # local carve fades to 0 at piece edges; shared records only there
    smoothing_steps=3,          # Richardson steps of the coarse-mean correction
    align_coarse=True,          # False: route on the plain Base (vector geography, 1-block halo, no Coarse)
    omega=1.6)


@dataclass(eq=False)
class RegionPlan:
    """Committed regional package. Arrays cover the core piece, row 0 north.

    ``origin`` is the global 240 m (row, col) of cell [0, 0]; local flat index
    ``i`` is global ``(origin[0] + i // cols, origin[1] + i % cols)``.
    ``receivers``: local flat index, -1 terminal (sea/parent lake/pit), -2
    declared exit portal or unit link, -4 trunk handled by a neighbouring span.
    ``links``: shared unit-link records touching this core (identical in both
    pieces of a cut).
    """
    bounds: tuple
    origin: tuple
    shape: tuple
    reaches: list
    local_lakes: list
    portals: list
    routing_height_m: np.ndarray
    conditioned_base_m: np.ndarray
    upper_m: np.ndarray
    water_m: np.ndarray
    unit_key: np.ndarray
    basin_ids: np.ndarray
    receivers: np.ndarray
    area_m2: np.ndarray
    flow_m3s: np.ndarray
    drain_class: np.ndarray
    metadata: dict = field(default_factory=dict)
    links: list = field(default_factory=list)


# ----------------------------------------------------------------------------
# Globally aligned windows and residual composition

def _grid_origin(arr, bounds, cell, name):
    rows, cols = np.shape(arr)
    x0, y0, x1, y1 = (float(v) for v in bounds)
    if abs((x1 - x0) / cols - cell) > 1e-6 * cell or abs((y1 - y0) / rows - cell) > 1e-6 * cell:
        raise ValueError(f'{name} spacing must be {cell:g} m')
    r, c = y0 / cell, x0 / cell
    if abs(r - round(r)) > 1e-6 or abs(c - round(c)) > 1e-6:
        raise ValueError(f'{name} bounds must lie on the global {cell:g} m lattice at 0')
    return int(round(r)), int(round(c))


def _crop(arr, origin, r0, c0, r1, c1, name):
    i0, j0, i1, j1 = r0 - origin[0], c0 - origin[1], r1 - origin[0], c1 - origin[1]
    if i0 < 0 or j0 < 0 or i1 > arr.shape[0] or j1 > arr.shape[1]:
        raise ValueError(f'{name} does not cover global rows {r0}:{r1}, cols {c0}:{c1}')
    return arr[i0:i1, j0:j1]


def _stencil(c):
    """Block mean of the cell-centred bilinear upsampling, valid interior (even factor)."""
    c = .125 * c[:-2] + .75 * c[1:-1] + .125 * c[2:]
    return .125 * c[:, :-2] + .75 * c[:, 1:-1] + .125 * c[:, 2:]


def _bilinear_up(c, factor):
    """Bilinear samples of ``c`` (one-cell margin) at fine centres of its interior."""
    def axis(n):
        u = 1 + np.repeat(np.arange(n), factor) + np.tile((np.arange(factor) + .5) / factor - .5, n)
        i = np.floor(u).astype(np.int64)
        return i, u - i
    i, wy = axis(c.shape[0] - 2)
    j, wx = axis(c.shape[1] - 2)
    rows = c[i] * (1 - wy)[:, None] + c[i + 1] * wy[:, None]
    return rows[:, j] * (1 - wx) + rows[:, j + 1] * wx


def aligned_detail(parent, parent_origin, fine, fine_origin, factor, rect, *, steps=3, omega=1.6):
    """Fine detail aligned on the parent lattice over parent-cell ``rect``.

    ``rect = (r0, c0, r1, c1)`` in global parent indices; origins are global
    indices of element [0, 0]. Returns ``fine + up(C)`` where the smooth
    correction ``C`` makes each parent block mean approach the parent value
    (Richardson iteration on the exact block-mean stencil [1/8, 3/4, 1/8]).
    The result depends only on parent/fine data within ``rect`` expanded by
    ``1 + steps`` parent cells, never on the caller's window size. There is
    no piecewise-constant remainder: the residual block-mean error is smooth
    and measured (``quality['block_mean_error_max_m']``), never a terrace.
    """
    r0, c0, r1, c1 = rect
    m = 1 + int(steps)
    p = _crop(np.asarray(parent, np.float64), parent_origin, r0 - m, c0 - m, r1 + m, c1 + m, 'parent')
    f = _crop(np.asarray(fine, np.float64), fine_origin, (r0 - m) * factor, (c0 - m) * factor,
              (r1 + m) * factor, (c1 + m) * factor, 'fine')
    nr, nc = p.shape
    deficit = p - f.reshape(nr, factor, nc, factor).mean(axis=(1, 3))
    coeff = deficit
    for _ in range(int(steps)):
        coeff = coeff[1:-1, 1:-1] + omega * (deficit[1:-1, 1:-1] - _stencil(coeff))
        deficit = deficit[1:-1, 1:-1]
    inner = f[(m * factor):-(m * factor), (m * factor):-(m * factor)]
    return inner + _bilinear_up(coeff, factor)


def piece_windows(core_bounds, *, smoothing_steps=DEFAULTS['smoothing_steps'], align_coarse=True):
    """Minimum (base_bounds, coarse_bounds) that ``refine_region`` needs for a core.

    Without coarse alignment only core + one block of Base is read (shared
    strips); the coarse window is then unused."""
    halo = (2 + int(smoothing_steps)) * COARSE_M if align_coarse else COARSE_M
    x0, y0, x1, y1 = (float(v) for v in core_bounds)
    grown = (x0 - halo, y0 - halo, x1 + halo, y1 + halo)
    return grown, grown


def piece_at(x, y, *, piece_blocks=16):
    """Core bounds of the uniform aligned piece grid containing (x, y)."""
    size = piece_blocks * COARSE_M
    i, j = math.floor(y / size), math.floor(x / size)
    return (j * size, i * size, (j + 1) * size, (i + 1) * size)


# ----------------------------------------------------------------------------
# Parent labels and lateral yield on aligned coarse blocks

def _latitude_scale(world, y):
    if world.topology != 'sphere':
        return np.ones_like(np.asarray(y, np.float64))
    y0, y1 = world.bounds[1], world.bounds[3]
    lat = np.radians(90 - (np.asarray(y, np.float64) - y0) / (y1 - y0) * 180)
    return np.maximum(np.cos(lat), 0.0)


def _sample_blocks(world, gi, gj):
    """Parent labels of blocks (global indices), sampled at block centres."""
    gi, gj = np.asarray(gi, np.int64), np.asarray(gj, np.int64)
    s = world.sample((gj.ravel() + .5) * COARSE_M, (gi.ravel() + .5) * COARSE_M)
    nreach, nlake = len(world.reaches), len(world.lakes)
    shape = gi.shape
    out = dict(cell=s['cell'].reshape(shape), basin=s['basin_id'].reshape(shape).astype(np.int64),
               ocean=s['ocean'].reshape(shape).astype(bool), lake=s['lake_id'].reshape(shape).astype(np.int64),
               reach_cell=s['reach_id'].reshape(shape).astype(np.int64),
               filled=s['filled_m'].reshape(shape).astype(np.float64))
    reach = s['owner_reach'].reshape(shape).astype(np.int64)
    olake = s['owner_lake'].reshape(shape).astype(np.int64)
    key = np.where(reach >= 0, reach, np.where(olake >= 0, nreach + olake, nreach + nlake + out['basin']))
    key = np.where(out['lake'] >= 0, nreach + out['lake'], key)
    out['key'] = np.where(out['ocean'], -1, key)
    kinds = np.asarray(world.basins.get('kind', np.zeros(0, np.int8)))
    sea_basin = np.r_[kinds == 0, False]          # basin -1 maps to the final False
    basin = np.where((out['basin'] >= 0) & (out['basin'] < kinds.size), out['basin'], kinds.size)
    # Coastal fringe units (no reach, no lake) of sea basins may drain to the parent sea.
    out['to_sea'] = (out['key'] >= nreach + nlake) & sea_basin[basin]
    return out


def _block_area(world, gi):
    """Exact sum of the fine 240 m cell areas of blocks in global block rows ``gi``."""
    gi = np.asarray(gi, np.int64)
    rows = (gi[..., None] * BLOCK + np.arange(BLOCK) + .5) * BASE_M
    return BLOCK * (BASE_M ** 2 * _latitude_scale(world, rows)).sum(-1)


class _Blocks:
    """Parent records per 7680 m block, sampled at block centres (pure, global)."""

    def __init__(self, world, rect):
        r0, c0, r1, c1 = rect
        self.r0, self.c0 = r0, c0
        gi, gj = np.meshgrid(np.arange(r0, r1), np.arange(c0, c1), indexing='ij')
        for name, value in _sample_blocks(world, gi, gj).items():
            setattr(self, name, value)
        self.density = _cell_density(world, self.cell)

    def at(self, name, grow, gcol):
        return getattr(self, name)[np.asarray(grow) // BLOCK - self.r0, np.asarray(gcol) // BLOCK - self.c0]

    def fine(self, name, g_r0, g_c0, rows, cols):
        r = (g_r0 + np.arange(rows)) // BLOCK - self.r0
        c = (g_c0 + np.arange(cols)) // BLOCK - self.c0
        return getattr(self, name)[r[:, None], c[None, :]]


def _cell_density(world, cell):
    """Non-negative parent lateral yield per m^2 of the blocks assigned to each source cell.

    Elementwise in ``cell`` (any shape), so a cell gets identical values in
    every caller. Summed over all fine cells of its blocks it reproduces the
    cell's non-negative lateral yield.
    """
    cells = np.unique(cell)
    h, w = world.shape
    q = world.discharge_m3s.ravel()
    rcv = world.receivers.ravel()
    rows, cols = np.divmod(cells, w)
    local = q[cells].astype(np.float64)
    sphere = world.topology == 'sphere'
    for di, dj in zip(DI, DJ):
        nr, nc = rows + di, cols + dj
        ok = (nr >= 0) & (nr < h)
        nc = nc % w if sphere else nc
        ok &= (nc >= 0) & (nc < w)
        nb = np.where(ok, nr * w + np.clip(nc, 0, w - 1), 0)
        local -= np.where(ok & (rcv[nb] == cells), q[nb], 0.0)
    wet = world.ocean.ravel()[cells] | (world.lake_ids.ravel()[cells] >= 0)
    local = np.where(wet, 0.0, np.maximum(local, 0.0))
    dx = (world.bounds[2] - world.bounds[0]) / w
    dy = (world.bounds[3] - world.bounds[1]) / h

    def centres(a, b):   # block indices whose centre lies in [a, b)
        return np.arange(math.ceil(a / COARSE_M - .5), math.ceil(b / COARSE_M - .5))
    assigned = np.empty(cells.size)
    for k, (i, j) in enumerate(zip(rows, cols)):
        nx = centres(world.bounds[0] + j * dx, world.bounds[0] + (j + 1) * dx).size
        ly = centres(world.bounds[1] + i * dy, world.bounds[1] + (i + 1) * dy)
        assigned[k] = nx * _block_area(world, ly).sum()
    density = np.divide(local, assigned, out=np.zeros_like(local), where=assigned > 0)
    return density[np.searchsorted(cells, cell)]


# ----------------------------------------------------------------------------
# Compiled kernels: keyed Priority-Flood and accumulation

@njit(cache=True)
def _less(hf, hc, a, b):
    return hf[a] < hf[b] or (hf[a] == hf[b] and hc[a] < hc[b])


@njit(cache=True)
def _swap(hf, hc, hi, a, b):
    hf[a], hf[b] = hf[b], hf[a]
    hc[a], hc[b] = hc[b], hc[a]
    hi[a], hi[b] = hi[b], hi[a]


@njit(cache=True)
def _flood(h, key, sea, to_sea, rows, cols, visited, is_seed, filled, rcv, seeds, order, n_order):
    """Priority-Flood from ``seeds`` (levels preset in ``filled``).

    A neighbour joins only from a cell with the same parent key (or from a
    parent-sea seed if it is a sea-draining fringe cell). Equal levels are
    FIFO, so flats drain towards their entry along a breadth-first rank.
    """
    n = h.size
    hf = np.empty(n)
    hc = np.empty(n, np.int64)
    hi = np.empty(n, np.int64)
    size = 0
    counter = 0
    for s in seeds:
        hf[size], hc[size], hi[size] = filled[s], counter, s
        counter += 1
        k = size
        size += 1
        while k > 0 and _less(hf, hc, k, (k - 1) // 2):
            _swap(hf, hc, hi, k, (k - 1) // 2)
            k = (k - 1) // 2
    while size > 0:
        c = hi[0]
        size -= 1
        hf[0], hc[0], hi[0] = hf[size], hc[size], hi[size]
        k = 0
        while True:
            a, b, best = 2 * k + 1, 2 * k + 2, k
            if a < size and _less(hf, hc, a, best):
                best = a
            if b < size and _less(hf, hc, b, best):
                best = b
            if best == k:
                break
            _swap(hf, hc, hi, k, best)
            k = best
        if not is_seed[c]:
            order[n_order] = c
            n_order += 1
        r, q = c // cols, c % cols
        for d in range(8):
            nr, nq = r + DI[d], q + DJ[d]
            if nr < 0 or nr >= rows or nq < 0 or nq >= cols:
                continue
            nb = nr * cols + nq
            if visited[nb]:
                continue
            if key[nb] != key[c] and not (sea[c] and to_sea[nb]):
                continue
            visited[nb] = True
            filled[nb] = max(h[nb], filled[c])
            rcv[nb] = c
            hf[size], hc[size], hi[size] = filled[nb], counter, nb
            counter += 1
            k = size
            size += 1
            while k > 0 and _less(hf, hc, k, (k - 1) // 2):
                _swap(hf, hc, hi, k, (k - 1) // 2)
                k = (k - 1) // 2
    return n_order


@njit(cache=True)
def _inherit(order, rcv, cls):
    for c in order:          # receivers precede donors in pop order
        cls[c] = cls[rcv[c]]


@njit(cache=True)
def _inherit_still(order, rcv, filled, owner):
    """Propagate ``owner`` from receivers along flats of equal filled level."""
    for c in order:
        r = rcv[c]
        if r >= 0 and owner[c] < 0 and filled[c] == filled[r]:
            owner[c] = owner[r]


@njit(cache=True)
def _accumulate(order, n_order, rcv, seed, values):
    """Sum ``values`` downstream; flow arriving at a seed is returned separately."""
    acc = values.copy()
    into = np.zeros(values.size)
    for t in range(n_order - 1, -1, -1):
        c = order[t]
        r = rcv[c]
        if r < 0:
            continue
        if seed[r]:
            into[r] += acc[c]
        else:
            acc[r] += acc[c]
    return acc, into


# ----------------------------------------------------------------------------
# Parent reaches: crossings, portals, anchors, trunk paths

def _vertex_values(world, reach, xy):
    cells = world.cell_at(xy[:, 0], xy[:, 1])
    own = np.isin(cells, np.asarray(reach.get('cells', cells)))
    q = world.discharge_m3s.ravel()[cells].astype(np.float64)
    a = world.area_m2.ravel()[cells].astype(np.float64)
    idx = np.flatnonzero(own)
    if idx.size == 0:
        return np.full(len(xy), float(reach['flow_m3s'])), np.full(len(xy), float(reach['area_m2']))
    nearest = idx[np.clip(np.searchsorted(idx, np.arange(len(xy))), 0, idx.size - 1)]
    left = idx[np.clip(np.searchsorted(idx, np.arange(len(xy))) - 1, 0, idx.size - 1)]
    pick = np.where(np.abs(left - np.arange(len(xy))) < np.abs(nearest - np.arange(len(xy))), left, nearest)
    return np.where(own, q, q[pick]), np.where(own, a, a[pick])


def _crossings(xy, core):
    """Exact crossings of the parent polyline with the core edges, by station."""
    x0, y0, x1, y1 = core
    out = []
    for axis, line, lo, hi in ((0, x0, y0, y1), (0, x1, y0, y1), (1, y0, x0, x1), (1, y1, x0, x1)):
        a, b = xy[:-1, axis], xy[1:, axis]
        for i in np.flatnonzero((a < line) != (b < line)):
            t = (line - a[i]) / (b[i] - a[i])
            point = xy[i] + t * (xy[i + 1] - xy[i])
            point[axis] = line
            if lo <= point[1 - axis] < hi:
                out.append((i + t, axis, line, point, (lo, hi)))
    return sorted(out, key=lambda c: c[0])


def _at_station(xy, s):
    i = min(int(s), len(xy) - 2)
    return xy[i] + (s - i) * (xy[i + 1] - xy[i])


def _clip_polyline(xy, s0, s1):
    inner = xy[int(math.floor(s0)) + 1:int(math.ceil(s1))]
    return np.vstack([_at_station(xy, s0)[None], inner, _at_station(xy, s1)[None]])


def _segment_distance(px, py, poly):
    px, py = np.broadcast_arrays(np.asarray(px, np.float64), np.asarray(py, np.float64))
    best = np.full(px.shape, np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        t = np.clip(((px - a[0]) * ab[0] + (py - a[1]) * ab[1]) / max(float(ab @ ab), 1e-12), 0, 1)
        best = np.minimum(best, np.hypot(px - a[0] - t * ab[0], py - a[1] - t * ab[1]))
    return best


def _bankfull(q):
    q = np.maximum(np.asarray(q, np.float64), 0.0)
    return WIDTH_COEFFICIENT * q ** WIDTH_EXPONENT, DEPTH_COEFFICIENT * q ** DEPTH_EXPONENT


class _Region:
    """Working state of one refinement call."""

    def __init__(self, world, coarse, co, base, bo, core, p, water_profiles=None):
        self.world, self.p = world, p
        self.water_profiles = water_profiles
        self.core = core
        self.r0, self.c0, self.r1, self.c1 = (int(round(core[1] / COARSE_M)), int(round(core[0] / COARSE_M)),
                                              int(round(core[3] / COARSE_M)), int(round(core[2] / COARSE_M)))
        self.coarse, self.co, self.base, self.bo = coarse, co, base, bo
        self.g0 = (self.r0 * BLOCK, self.c0 * BLOCK)
        # Core plus one block: shared strips (portals, links, lake shores) read it.
        self.Hh = self.compose((self.r0 - 1, self.c0 - 1, self.r1 + 1, self.c1 + 1))
        self.H = np.ascontiguousarray(self.Hh[BLOCK:-BLOCK, BLOCK:-BLOCK])
        self.rows, self.cols = self.H.shape
        self.blocks = _Blocks(world, (self.r0 - 1, self.c0 - 1, self.r1 + 1, self.c1 + 1))
        fine = lambda name: self.blocks.fine(name, self.g0[0], self.g0[1], self.rows, self.cols)
        self.key, self.basin, self.ocean = fine('key'), fine('basin'), fine('ocean')
        self.lake_block, self.to_sea = fine('lake'), fine('to_sea')
        self.sea_level = float(world.metadata.get('parameters', {}).get('sea_level_m', 0.0))
        self.lake_level = np.array([float(lk['level_m']) for lk in world.lakes] + [np.nan])
        self.lake = self._lake_shores()
        y = (self.g0[0] + np.arange(self.rows) + .5) * BASE_M
        self.cell_area = np.repeat((BASE_M ** 2 * _latitude_scale(world, y))[:, None], self.cols, 1)
        self.yield_m3s = np.where(self.ocean | (self.lake_block >= 0), 0.0, fine('density') * self.cell_area)
        h, w = world.shape
        spacing = max((world.bounds[2] - world.bounds[0]) / w, (world.bounds[3] - world.bounds[1]) / h)
        self.corridor = float(p['corridor_m'] or 0.75 * spacing)
        r = int(p['valley_radius_cells'])
        self.valley = self.H - ndimage.minimum_filter(self.H, size=2 * r + 1, mode='nearest')
        self.used = np.zeros((self.rows, self.cols), bool)
        self.portal_cache, self.anchor_cache = {}, {}
        self.cx = (self.g0[1] + np.arange(self.cols) + .5) * BASE_M
        self.cy = (self.g0[0] + np.arange(self.rows) + .5) * BASE_M

    def compose(self, rect):
        if not self.p['align_coarse']:
            r0, c0, r1, c1 = rect
            return _crop(self.base, self.bo, r0 * BLOCK, c0 * BLOCK, r1 * BLOCK, c1 * BLOCK, 'base').copy()
        return aligned_detail(self.coarse, self.co, self.base, self.bo, BLOCK, rect,
                              steps=self.p['smoothing_steps'], omega=self.p['omega'])

    def level(self, reach, station):
        if self.water_profiles is not None:
            return self.water_profiles.level(reach['id'], station)
        raw = np.minimum.accumulate(np.asarray(reach['water_m'], np.float64))
        return np.interp(station, np.arange(len(raw)), raw)

    def _lake_shores(self):
        """Parent lake ids per cell, extended over the parent shore envelope.

        The envelope is every land block 8-adjacent to a block of parent lake L
        in L's basin. Its cells below L's level that connect to L's blocks
        inside the 3 x 3 block window centred on the cell's block join L at
        L's level: a pure function of that window, so both sides of a cut agree.
        """
        b = self.blocks
        lake = self.lake_block.copy()
        level = self.lake_level
        for i in range(1, b.lake.shape[0] - 1):
            for j in range(1, b.lake.shape[1] - 1):
                if b.lake[i, j] >= 0 or b.ocean[i, j]:
                    continue
                near = b.lake[i - 1:i + 2, j - 1:j + 2]
                cand = sorted({int(v) for v in near[near >= 0].ravel()
                               if int(self.world.lakes[int(v)]['basin_id']) == b.basin[i, j]})
                if not cand:
                    continue
                L = cand[0]
                own = np.kron(near == L, np.ones((BLOCK, BLOCK), bool))
                land = np.kron((near < 0) & ~b.ocean[i - 1:i + 2, j - 1:j + 2]
                               & (b.basin[i - 1:i + 2, j - 1:j + 2] == b.basin[i, j]), np.ones((BLOCK, BLOCK), bool))
                win = self.Hh[(i - 1) * BLOCK:(i + 2) * BLOCK, (j - 1) * BLOCK:(j + 2) * BLOCK]
                wet = own | (land & (win < level[L]))
                labels, _ = ndimage.label(wet, np.ones((3, 3), bool))
                joined = np.isin(labels, np.unique(labels[own]))
                centre = joined[BLOCK:2 * BLOCK, BLOCK:2 * BLOCK] & ~own[BLOCK:2 * BLOCK, BLOCK:2 * BLOCK]
                lake[(i - 1) * BLOCK:i * BLOCK, (j - 1) * BLOCK:j * BLOCK][centre] = L
        return lake

    def local(self, grow, gcol):
        return (grow - self.g0[0]) * self.cols + (gcol - self.g0[1])

    def xy_of(self, flat):
        flat = np.asarray(flat, np.int64)
        return np.stack((self.cx[flat % self.cols], self.cy[flat // self.cols]), -1)

    def gid(self, flat):
        return f'g{self.g0[0] + int(flat) // self.cols}_{self.g0[1] + int(flat) % self.cols}'

    # Shared portal: pure function of parent reach, cut line and the fixed strip.
    def portal(self, reach, xy, qv, av, crossing):
        s, axis, line, point, (lo, hi) = crossing
        pid = f"{reach['id']}|{'xy'[axis]}{line:.0f}|{int(s)}"
        if pid in self.portal_cache:
            return self.portal_cache[pid]
        p, corr = self.p, self.corridor
        t_p = point[1 - axis]
        lb = int(round(line / COARSE_M))
        b0 = max(int(math.floor((t_p - corr) / COARSE_M)), int(round(lo / COARSE_M)))
        b1 = min(int(math.floor((t_p + corr) / COARSE_M)) + 1, int(round(hi / COARSE_M)))
        rect = (b0, lb - 1, b1, lb + 1) if axis == 0 else (lb - 1, b0, lb + 1, b1)
        strip = self.compose(rect)
        strip = strip if axis == 0 else strip.T          # (along, across); cut between 31 | 32
        g = b0 * BLOCK + np.arange(strip.shape[0])
        t = (g + .5) * BASE_M
        gl, gh = lb * BLOCK - 1, lb * BLOCK
        cell_lo = (g, np.full_like(g, gl)) if axis == 0 else (np.full_like(g, gl), g)
        cell_hi = (g, np.full_like(g, gh)) if axis == 0 else (np.full_like(g, gh), g)
        i = min(int(s), len(xy) - 2)
        upstream_low = xy[i + 1, axis] > xy[i, axis]
        node = self.world.nodes[reach['target']] if reach.get('target') is not None else {}
        mouth = node.get('kind') == 'mouth'
        near = np.abs(t - t_p) <= corr
        sides = []
        for cell, downstream in ((cell_lo, not upstream_low), (cell_hi, upstream_low)):
            sea = self.blocks.at('ocean', *cell)
            own = (self.blocks.at('basin', *cell) == reach['basin_id']) & ~sea
            # A mouth just beyond the cut: the downstream side may be the parent sea.
            sides.append((own | (sea & mouth & downstream)) & (self.blocks.at('lake', *cell) < 0))
        # Tier 0: both sides carry the reach basin; tier 1: one side (diagonal
        # parent steps through source-cell corners); tier 2: none (flagged).
        tier = np.where(sides[0] & sides[1], 0, np.where(sides[0] | sides[1], 1, 2))
        tier = np.where(near, tier, 3)
        ok = tier == tier.min()
        level = .5 * (strip[:, BLOCK - 1] + strip[:, BLOCK])
        # A valley is low relative to the regional trend along the cut, not
        # merely the downslope end of the corridor: remove the linear trend.
        trend = np.zeros_like(level)
        if np.count_nonzero(ok) >= 3:
            slope, offset = np.polyfit(t[ok] - t_p, level[ok], 1)
            trend = slope * (t - t_p) + offset
        cost = level - trend + p['portal_deviation_m'] * ((t - t_p) / corr) ** 2
        k = int(np.argmin(np.where(ok, cost, np.inf)))
        down = cell_hi if upstream_low else cell_lo
        sea_side = bool(self.blocks.at('ocean', down[0][k], down[1][k]))
        ends = []
        for col in (BLOCK - 8, BLOCK + 7):
            win = slice(max(0, k - 8), min(strip.shape[0], k + 9))
            j = win.start + int(np.argmin(strip[win, col]))
            across = (lb * BLOCK - BLOCK + col + .5) * BASE_M
            ends.append(np.array([across, t[j]]) if axis == 0 else np.array([t[j], across]))
        tangent = ends[1] - ends[0] if upstream_low else ends[0] - ends[1]
        tangent /= max(np.hypot(*tangent), 1e-9)
        pos = np.array([line, t[k]]) if axis == 0 else np.array([t[k], line])
        qs = float(np.interp(s, np.arange(len(qv)), qv))
        a_s = float(np.interp(s, np.arange(len(av)), av))
        record = dict(id=pid, reach_id=reach['id'], basin_id=reach['basin_id'], axis='xy'[axis],
                      line_m=float(line), station=float(s), parent_xy=point.copy(), xy=pos,
                      cells=((int(cell_lo[0][k]), int(cell_lo[1][k])), (int(cell_hi[0][k]), int(cell_hi[1][k]))),
                      level_m=float(self.level(reach, s)), ground_m=float(level[k]),
                      tangent=tangent, flow_m3s=qs, area_m2=a_s,
                      upstream_side='low' if upstream_low else 'high', tier=int(tier[k]),
                      admissible=bool(tier[k] <= 1), downstream_sea=sea_side,
                      deviation_m=float(t[k] - t_p))
        self.portal_cache[pid] = record
        return record

    def portal_cell(self, record):
        for grow, gcol in record['cells']:
            if 0 <= grow - self.g0[0] < self.rows and 0 <= gcol - self.g0[1] < self.cols:
                return self.local(grow, gcol)
        raise RuntimeError('portal is not on this core edge')

    # Node anchor: lowest admissible cell near a parent node inside its owning core.
    def anchor(self, node_key, xy, kind, basin, lake):
        if node_key in self.anchor_cache:
            return self.anchor_cache[node_key]
        radius = float(self.p['anchor_radius_m'] or COARSE_M)
        reach_cells = int(2 * radius // BASE_M) + 2
        i = int(np.clip(math.floor(xy[1] / BASE_M) - self.g0[0], 0, self.rows - 1))
        j = int(np.clip(math.floor(xy[0] / BASE_M) - self.g0[1], 0, self.cols - 1))
        rs = slice(max(0, i - reach_cells), min(self.rows, i + reach_cells + 1))
        cs = slice(max(0, j - reach_cells), min(self.cols, j + reach_cells + 1))
        d = np.hypot(self.cx[None, cs] - xy[0], self.cy[rs, None] - xy[1])
        ocean, lakes = self.ocean[rs, cs], self.lake[rs, cs]
        own = (self.basin[rs, cs] == basin) & ~ocean & (lakes < 0)
        prefer = None
        if kind == 'mouth':
            prefer = ndimage.binary_dilation(ocean, np.ones((3, 3), bool))
        elif lake is not None and lake >= 0:
            prefer = ndimage.binary_dilation(lakes == lake, np.ones((3, 3), bool))
        fallback = False
        for scale in (1.0, 2.0):            # widen once before declaring a fallback
            cand = own & (d <= scale * radius)
            if cand.any():
                break
        else:
            fallback = True
            cand = d <= d.min()
        if prefer is not None and (cand & prefer).any():
            cand &= prefer
        h = self.H[rs, cs]
        trend = np.zeros_like(h)
        if np.count_nonzero(cand) >= 3:      # valley floor relative to the local plane
            dx, dy = self.cx[None, cs] - xy[0] + 0 * h, self.cy[rs, None] - xy[1] + 0 * h
            design = np.stack([np.ones(np.count_nonzero(cand)), dx[cand], dy[cand]], 1)
            coef = np.linalg.lstsq(design, h[cand], rcond=None)[0]
            trend = coef[0] + coef[1] * dx + coef[2] * dy
        cost = np.where(cand, h - trend + self.p['portal_deviation_m'] * (d / radius) ** 2, np.inf)
        a, b = np.unravel_index(int(np.argmin(cost)), cost.shape)
        cell = (rs.start + a) * self.cols + cs.start + b
        self.anchor_cache[node_key] = (cell, float(self.H.flat[cell]), fallback)
        return self.anchor_cache[node_key]

    def path(self, start, end, dist, allowed, limit):
        p = self.p
        mask = allowed & (dist <= limit)
        mask.flat[start] = mask.flat[end] = True
        idx = np.flatnonzero(mask)
        lookup = np.full(mask.size, -1, np.int64)
        lookup[idx] = np.arange(idx.size)
        h = self.H.flat[idx]
        node = (1 + self.valley.flat[idx] / p['relief_scale_m']
                + p['deviation_weight'] * np.minimum(dist.flat[idx] / self.corridor, 4) ** 2)
        r, c = np.divmod(idx, self.cols)
        src, dst, cost = [], [], []
        for di, dj in zip(DI, DJ):
            nr, nc = r + di, c + dj
            ok = (nr >= 0) & (nr < self.rows) & (nc >= 0) & (nc < self.cols)
            nb = lookup[np.where(ok, nr * self.cols + nc, 0)]
            ok &= nb >= 0
            a, b = np.flatnonzero(ok), nb[ok]
            length = BASE_M * math.hypot(di, dj)
            src.append(a)
            dst.append(b)
            cost.append(length * .5 * (node[a] + node[b]) + p['climb_weight'] * np.maximum(h[b] - h[a], 0))
        graph = csr_matrix((np.concatenate(cost), (np.concatenate(src), np.concatenate(dst))),
                           shape=(idx.size, idx.size))
        dist_to, pred = dijkstra(graph, indices=int(lookup[start]), return_predecessors=True)
        k = int(lookup[end])
        if not np.isfinite(dist_to[k]):
            return None
        out = [k]
        while out[-1] != lookup[start]:
            out.append(int(pred[out[-1]]))
        return idx[np.array(out[::-1])]

    def parent_line(self, poly):
        """Fallback: raster of the clipped parent polyline (cells forced)."""
        dense = [poly[0]]
        for a, b in zip(poly[:-1], poly[1:]):
            n = max(1, int(math.ceil(np.hypot(*(b - a)) / (.5 * BASE_M))))
            dense.extend(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
        dense = np.asarray(dense)
        col = np.clip(np.floor(dense[:, 0] / BASE_M).astype(np.int64) - self.g0[1], 0, self.cols - 1)
        row = np.clip(np.floor(dense[:, 1] / BASE_M).astype(np.int64) - self.g0[0], 0, self.rows - 1)
        flat = row * self.cols + col
        keep = np.r_[True, flat[1:] != flat[:-1]]
        return flat[keep]


# ----------------------------------------------------------------------------
# Public entry point

def refine_region(world, coarse_height, coarse_bounds, base_height, base_bounds, core_bounds, *,
                  seed=0, water_profiles=None, **options):
    """Refine the global hydrology over one fixed aligned core piece.

    Parameters
    ----------
    world : HydroWorld from ``terrain_hydrology.build_hydrology``.
    coarse_height, coarse_bounds : canonical 7680 m neural field on the global
        lattice, covering at least ``piece_windows(core_bounds)[1]``.
    base_height, base_bounds : canonical 240 m base reconstruction on the
        global lattice covering ``piece_windows(core_bounds)[0]`` (halo of
        ``2 + smoothing_steps`` coarse blocks around the core).
    core_bounds : owned piece, multiples of 7680 m. Neighbouring pieces must
        come from one uniform grid of this core's size and alignment (as
        ``piece_at``): cut edges coincide and unit links infer that grid.
        With ``align_coarse=False`` the coarse arguments may be None: routing
        uses the plain Base over core + one block (display vectors only).
    seed : world seed; enters identities only (there are no random draws).
    water_profiles : optional canonical object with ``level(reach_id, station)``;
        station is fractional original reach vertex index. Without it, the
        descending original source profile is used, never local node ground.
    options : see ``DEFAULTS``; neighbours must use identical options.

    Returns a :class:`RegionPlan`.
    """
    started = time.perf_counter()
    p = dict(DEFAULTS)
    unknown = set(options) - set(p)
    if unknown:
        raise TypeError(f'unknown options: {sorted(unknown)}')
    p.update(options)
    core = tuple(float(v) for v in core_bounds)
    for v in core:
        if abs(v / COARSE_M - round(v / COARSE_M)) > 1e-9:
            raise ValueError('core_bounds must lie on the 7680 m lattice')
    if core[2] <= core[0] or core[3] <= core[1]:
        raise ValueError('core_bounds must be (x0, y0, x1, y1) with x1 > x0, y1 > y0')
    if world.topology == 'sphere' and (core[0] < world.bounds[0] or core[2] > world.bounds[2]):
        raise ValueError('pieces crossing the antimeridian need the canonical seam strip (not implemented)')
    base = np.asarray(base_height, np.float64)
    bo = _grid_origin(base, base_bounds, BASE_M, 'base')
    need_base, need_coarse = piece_windows(core, smoothing_steps=p['smoothing_steps'],
                                           align_coarse=p['align_coarse'])
    coarse, co = None, None
    if p['align_coarse']:
        coarse = np.asarray(coarse_height, np.float64)
        co = _grid_origin(coarse, coarse_bounds, COARSE_M, 'coarse')
        _crop(coarse, co, *(int(round(need_coarse[i] / COARSE_M)) for i in (1, 0, 3, 2)), 'coarse')
    _crop(base, bo, *(int(round(need_base[i] / BASE_M)) for i in (1, 0, 3, 2)), 'base')
    R = _Region(world, coarse, co, base, bo, core, p, water_profiles)
    t_compose = time.perf_counter() - started

    spans = _route_trunks(R)
    t_trunks = time.perf_counter() - started - t_compose
    plan = _route_local(R, spans)
    plan.metadata['timings_s'] = dict(compose=round(t_compose, 4), trunks=round(t_trunks, 4),
                                      total=round(time.perf_counter() - started, 4))
    digest = hashlib.blake2b(digest_size=16)
    digest.update(json.dumps(dict(version=REGIONS_VERSION, hydro=world.metadata.get('hydro_id'),
                                  core=core, seed=int(seed), options={k: p[k] for k in sorted(p)}),
                             sort_keys=True, default=str).encode())
    for arr in (R.Hh,):
        digest.update(np.ascontiguousarray(arr, np.float64).tobytes())
    for span in spans:
        digest.update(str(span['reach']['id']).encode())
        digest.update(np.ascontiguousarray(span['water'], np.float64).tobytes())
    for link in plan.links:
        digest.update(f"{link['id']}:{link['level_m']:.17g}".encode())
    plan.metadata.update(region_id=digest.hexdigest(), version=REGIONS_VERSION, seed=int(seed),
                         options=p, corridor_m=R.corridor)
    # Uncarved routing terrain over core + one 240 m cell (a crop of the
    # piece-independent Hh). Display corrections interpolate across core edges.
    plan.metadata['routing_halo_m'] = R.Hh[BLOCK - 1:1 - BLOCK, BLOCK - 1:1 - BLOCK].astype(np.float32)
    plan.metadata['routing_halo_bounds'] = (core[0] - BASE_M, core[1] - BASE_M, core[2] + BASE_M, core[3] + BASE_M)
    return plan


_BOX_CACHE = []        # [(reaches list, boxes)], identity-checked, newest last


def _reach_boxes(world):
    """Per-reach (x0, y0, x1, y1) of the parent polylines, cached per reach list."""
    for reaches, boxes in _BOX_CACHE:
        if reaches is world.reaches:
            return boxes
    boxes = np.array([[*np.min(r['xy'], 0), *np.max(r['xy'], 0)] if len(r['xy']) > 1
                      else [np.inf, np.inf, -np.inf, -np.inf] for r in world.reaches]).reshape(-1, 4)
    _BOX_CACHE.append((world.reaches, boxes))
    del _BOX_CACHE[:-4]
    return boxes


def _route_trunks(R):
    """Valley-following trunk spans of every parent reach crossing the core."""
    world = R.world
    width = world.bounds[2] - world.bounds[0]
    x0, y0, x1, y1 = R.core
    boxes = _reach_boxes(world)
    shift = np.zeros(len(boxes))
    if world.topology == 'sphere':
        valid = np.isfinite(boxes).all(axis=1)
        shift[valid] = width * np.round((.5 * (x0 + x1) - .5 * (boxes[valid, 0] + boxes[valid, 2])) / width)
    c = R.corridor + 1.0
    hit = ((boxes[:, 2] + shift >= x0 - c) & (boxes[:, 0] + shift <= x1 + c)
           & (boxes[:, 3] >= y0 - c) & (boxes[:, 1] <= y1 + c))
    todo = []
    for k in np.flatnonzero(hit):
        reach = world.reaches[k]
        # A fixed sub-centimetre offset (identical for every piece) removes exact
        # corner/vertex coincidences with the 7680 m cut lattice.
        xy = np.asarray(reach['xy'], np.float64).reshape(-1, 2) + POLYLINE_TIE_OFFSET_M + [shift[k], 0]
        todo.append((reach, xy))
    todo.sort(key=lambda item: (-float(item[0]['flow_m3s']), str(item[0]['id'])))
    inside = lambda pt: x0 <= pt[0] < x1 and y0 <= pt[1] < y1
    spans = []
    for reach, xy in todo:
        qv, av = _vertex_values(world, reach, xy)
        crossings = _crossings(xy, R.core)
        state = inside(xy[0])
        bounds = [('node', 0.0, None)] if state else []
        for c in crossings:
            bounds.append(('portal', c[0], c))
        if inside(xy[-1]):
            bounds.append(('node', float(len(xy) - 1), None))
        # Boundaries alternate entry/exit, so consecutive pairs are the inside spans.
        for a, b in zip(bounds[0::2], bounds[1::2]):
            spans.append(_trunk_span(R, reach, xy, qv, av, a, b))
    return [s for s in spans if s is not None]


def _endpoint(R, reach, xy, qv, av, event, which):
    world = R.world
    kind, s, crossing = event
    if kind == 'portal':
        record = R.portal(reach, xy, qv, av, crossing)
        return dict(kind='portal', cell=R.portal_cell(record), level=record['level_m'],
                    flow=record['flow_m3s'], portal=record, key=f"p{record['id']}", fallback=not record['admissible'])
    node_id = reach['source'] if which == 'start' else reach['target']
    node = world.nodes[node_id] if node_id is not None and 0 <= node_id < len(world.nodes) else {}
    nkind = node.get('kind', 'source')
    lake = reach.get('leaves_lake') if which == 'start' else reach.get('enters_lake')
    point = xy[0] if which == 'start' else xy[-1]
    cell, _, fallback = R.anchor((node_id, round(point[0]), round(point[1])), point, nkind,
                                     reach['basin_id'], lake)
    flow = float(qv[0]) if which == 'start' else float(reach['flow_m3s'])
    # The anchor selects XY only. Shared reach profiles own all water levels.
    level = float(R.level(reach, 0 if which == 'start' else len(xy) - 1))
    return dict(kind='node', node_kind=nkind, cell=cell, level=level, flow=flow, key='src',
                fallback=fallback)


def _bounded_water(water, ground, fill_m, incision_m):
    """Lower a descending water line that floats over its own 240 m channel.

    Water may stand at most ``fill_m`` above any ground upstream (a lower
    point upstream bounds everything downstream), cut at most ``incision_m``
    below downstream ground, and never drop below its end. Every envelope
    descends, so the result descends; both pinned ends are kept exactly. An
    unchanged line is returned wherever it already fits.
    """
    water = np.asarray(water, np.float64)
    low = np.maximum(np.maximum.accumulate((ground - incision_m)[::-1])[::-1], water[-1])
    out = np.minimum(water, np.maximum(low, np.minimum.accumulate(ground + fill_m)))
    out[0], out[-1] = water[0], water[-1]
    return out


def _trunk_span(R, reach, xy, qv, av, a, b):
    p = R.p
    start = _endpoint(R, reach, xy, qv, av, a, 'start')
    end = _endpoint(R, reach, xy, qv, av, b, 'end')
    if start['kind'] == 'portal' and start['portal']['downstream_sea']:
        return None          # the reach already reached the parent sea at this cut
    poly = _clip_polyline(xy, a[1], b[1])
    lo = np.floor((poly.min(0) - 2 * R.corridor) / BASE_M).astype(np.int64)
    hi = np.ceil((poly.max(0) + 2 * R.corridor) / BASE_M).astype(np.int64)
    rs = slice(max(0, lo[1] - R.g0[0]), min(R.rows, hi[1] - R.g0[0]))
    cs = slice(max(0, lo[0] - R.g0[1]), min(R.cols, hi[0] - R.g0[1]))
    dist = np.full((R.rows, R.cols), np.inf)
    dist[rs, cs] = _segment_distance(R.cx[None, cs], R.cy[rs, None], poly)
    allowed = (R.basin == reach['basin_id']) & ~R.ocean & (R.lake < 0) & ~R.used
    stage = 'corridor'
    path = R.path(start['cell'], end['cell'], dist, allowed, R.corridor)
    # Endpoints on a label staircase (parent diagonal through a source-cell
    # corner) may sit in a neighbour's label: allow non-sea cells within one
    # block of them, counted as foreign cells.
    near = np.zeros_like(allowed)
    for cell in (start['cell'], end['cell']):
        i, j = divmod(cell, R.cols)
        near[max(0, i - BLOCK):i + BLOCK + 1, max(0, j - BLOCK):j + BLOCK + 1] = True
    relaxed = allowed | (near & ~R.ocean & (R.lake < 0) & ~R.used)
    for name, mask, limit in (('widened_corridor', allowed, 2 * R.corridor), ('basin', allowed, np.inf),
                              ('endpoint_relaxed', relaxed, 2 * R.corridor)):
        if path is None:
            stage = name
            path = R.path(start['cell'], end['cell'], dist, mask, limit)
    if path is None:
        stage = 'parent_line'
        line = np.vstack([R.xy_of([start['cell']]), poly, R.xy_of([end['cell']])])
        path = R.parent_line(line)
    foreign = int(np.count_nonzero(R.basin.flat[path] != reach['basin_id']))
    R.used.flat[path] = True
    ground = R.H.flat[path]
    path_xy = R.xy_of(path)
    distance = np.r_[0., np.cumsum(np.hypot(*np.diff(path_xy, axis=0).T))]
    fraction = distance / distance[-1] if distance[-1] > 0 else np.zeros(len(path))
    canonical = np.asarray(R.level(reach, a[1] + fraction * (b[1] - a[1])), np.float64)
    water = _bounded_water(canonical, ground, p['bank_fill_m'], p['strong_incision_m'])
    incision = float(np.max(ground - water, initial=0.0))
    deviation = _segment_distance(path_xy[:, 0], path_xy[:, 1], poly)
    return dict(reach=reach, path=path, water=water, start=start, end=end, ground=ground,
                diagnostics=dict(stage=stage, cells=int(path.size), foreign_basin_cells=foreign,
                                 max_incision_m=incision,
                                 max_fill_m=float(np.max(water - ground, initial=0.0)),
                                 fine_lowering_m=float(np.max(canonical - water, initial=0.0)),
                                 profile_rise_m=float(np.maximum(np.diff(water), 0.).max(initial=0.)),
                                 mean_deviation_m=float(deviation.mean()),
                                 max_deviation_m=float(deviation.max()),
                                 anchor_fallback=bool(start['fallback'] or end['fallback']),
                                 strong_correction=bool(incision > p['strong_incision_m']
                                                        or stage == 'parent_line' or foreign > 2)))


# ----------------------------------------------------------------------------
# Unit fragments across cuts: a pure block-level ownership contract

_POLY_BLOCKS = []      # [(reaches list, {reach id: blocks on its polyline})], newest last


def _reach_blocks(world, rid):
    for reaches, table in _POLY_BLOCKS:
        if reaches is world.reaches:
            break
    else:
        table = {}
        _POLY_BLOCKS.append((world.reaches, table))
        del _POLY_BLOCKS[:-4]
    if rid not in table:
        xy = np.asarray(world.reaches[rid]['xy'], np.float64).reshape(-1, 2) + POLYLINE_TIE_OFFSET_M
        pts = [xy[:1]]
        for a, b in zip(xy[:-1], xy[1:]):
            n = max(1, int(math.ceil(np.hypot(*(b - a)) / (.125 * COARSE_M))))
            pts.append(a + (b - a) * (np.arange(1, n + 1) / n)[:, None])
        pts = np.vstack(pts)
        table[rid] = set(zip(np.floor(pts[:, 1] / COARSE_M).astype(np.int64).tolist(),
                             np.floor(pts[:, 0] / COARSE_M).astype(np.int64).tolist()))
    return table[rid]


def _key_name(world, k):
    nreach, nlake = len(world.reaches), len(world.lakes)
    return str(k) if k < nreach else f'lake{k - nreach}' if k < nreach + nlake else f'basin{k - nreach - nlake}'


def _joins_unit(world, v, k):
    """Whether parent reach ``v`` delivers its water to unit ``k`` (its own network path)."""
    nreach, nlake = len(world.reaches), len(world.lakes)
    if k >= nreach + nlake:
        return int(world.reaches[v]['basin_id']) == k - nreach - nlake
    r, hops = v, 0
    while r is not None and hops < 4096:
        if r == k or (k >= nreach and world.reaches[r].get('enters_lake') == k - nreach):
            return True
        reach = world.reaches[r]
        r, hops = reach.get('downstream_id', None), hops + 1
        if r is None:
            r = reach.get('hydraulic_downstream_id')
    return False


def _block_neighbours(b):
    """Unit block graph: 8-connected, like the fine flood (diagonals meet at a corner cell pair)."""
    i, j = b
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if di or dj:
                yield (i + di, j + dj)


def _unit_component(world, k, start, extent):
    """Global connected blocks of unit ``k`` containing block ``start`` (None if too large)."""
    members, seen, frontier = {start}, {start: True}, [start]
    while frontier:
        cand = sorted({nb for b in frontier for nb in _block_neighbours(b)
                       if nb not in seen and extent[0] <= nb[0] < extent[2] and extent[1] <= nb[1] < extent[3]})
        if not cand:
            break
        arr = np.array(cand, np.int64)
        keys = _sample_blocks(world, arr[:, 0], arr[:, 1])['key']
        frontier = [b for b, kk in zip(cand, keys) if kk == k]
        seen.update((b, kk == k) for b, kk in zip(cand, keys))
        members.update(frontier)
        if len(members) > UNIT_BLOCK_LIMIT:
            return None
    return sorted(members)


def _unit_profile_ids(world, k, blocks):
    """Own sink and potential absorber reaches of one entire unit component."""
    arr = np.asarray(blocks, np.int64)
    x0, y0 = arr[:, 1].min() * COARSE_M, arr[:, 0].min() * COARSE_M
    x1, y1 = (arr[:, 1].max() + 1) * COARSE_M, (arr[:, 0].max() + 1) * COARSE_M
    boxes = _reach_boxes(world)
    near = np.flatnonzero((boxes[:, 2] >= x0) & (boxes[:, 0] <= x1)
                         & (boxes[:, 3] >= y0) & (boxes[:, 1] <= y1))
    candidates = [v for v in near.tolist() if v != k and _joins_unit(world, v, k)]
    return ([int(k)] if k < len(world.reaches) else []) + candidates


def required_profile_reaches(world, core_bounds):
    """Pure conservative sink/absorber profile prerequisites for a regional core.

    Includes whole connected units touching the core, with the same 20 000
    block limit as unit links. Call before refinement to batch coarse queries.
    """
    x0, y0, x1, y1 = core_bounds
    gi, gj = np.meshgrid(np.arange(round(y0 / COARSE_M), round(y1 / COARSE_M)),
                         np.arange(round(x0 / COARSE_M), round(x1 / COARSE_M)), indexing='ij')
    keys = _sample_blocks(world, gi, gj)['key']
    extent = (math.ceil(world.bounds[1] / COARSE_M - .5), math.ceil(world.bounds[0] / COARSE_M - .5),
              math.ceil(world.bounds[3] / COARSE_M - .5), math.ceil(world.bounds[2] / COARSE_M - .5))
    done, required = set(), set()
    for i, j, key in zip(gi.ravel(), gj.ravel(), keys.ravel()):
        start = (int(i), int(j))
        if key < 0 or start in done:
            continue
        blocks = _unit_component(world, int(key), start, extent)
        done.update(blocks or [start])
        if blocks is not None:
            required.update(_unit_profile_ids(world, int(key), blocks))
    return sorted(required)


def _piece_entry_station(world, rid, core):
    """Earliest possible trunk-span station in a piece, before any XY reroute."""
    xy = np.asarray(world.reaches[rid]['xy'], np.float64) + POLYLINE_TIE_OFFSET_M
    if world.topology == 'sphere':
        width = world.bounds[2] - world.bounds[0]
        shift = width * np.round((.5 * (core[0] + core[2]) - .5 * (xy[:, 0].min() + xy[:, 0].max())) / width)
        xy = xy + [shift, 0]
    if core[0] <= xy[0, 0] < core[2] and core[1] <= xy[0, 1] < core[3]:
        return 0.
    events = _crossings(xy, core)
    return float(events[0][0]) if events else None


def _unit_links(R):
    """Ownership of unit fragments in this core, and the unit links touching it.

    A unit fragment is a 4-connected set of blocks of one parent unit inside
    one piece of the uniform grid. It is ``own`` when it holds its unit's sink
    (the reach polyline, the lake blocks, a coastal block next to sea in the
    same piece) and ``absorbed`` when the polyline of a parent reach that
    delivers to the unit crosses it (the fragment joins that trunk upstream of
    the parent junction). Otherwise it is linked: it leaves through the block
    edge towards the fragment closest (in unit blocks) to a sink, carrying the
    exact parent yield of its blocks plus its inflowing links. Everything is a
    pure function of the parent world and the piece grid, so the neighbour
    computes the same record. Returns ({core block: flood key}, links, diag).
    """
    world = R.world
    prow, pcol = R.r1 - R.r0, R.c1 - R.c0
    piece = lambda b: ((b[0] - R.r0) // prow, (b[1] - R.c0) // pcol)
    extent = (math.ceil(world.bounds[1] / COARSE_M - .5), math.ceil(world.bounds[0] / COARSE_M - .5),
              math.ceil(world.bounds[3] / COARSE_M - .5), math.ceil(world.bounds[2] / COARSE_M - .5))
    nreach, nlake = len(world.reaches), len(world.lakes)
    keys = R.blocks.key[1:-1, 1:-1]
    override, links, diag = {}, [], dict(own=0, absorbed=0, linked=0, stranded=0,
        rejected_high_absorber_candidates=0, unresolved_authority_blocks=0)
    done = set()
    for (i, j) in sorted(zip(*np.nonzero(keys >= 0))):
        start = (R.r0 + int(i), R.c0 + int(j))
        if start in done:
            continue
        k = int(keys[i, j])
        blocks = _unit_component(world, k, start, extent)
        if blocks is None:
            done.add(start)
            diag['stranded'] += 1
            continue
        done.update(blocks)
        index = {b: n for n, b in enumerate(blocks)}
        arr = np.array(blocks, np.int64)
        s = _sample_blocks(world, arr[:, 0], arr[:, 1])
        land = len(blocks)
        own = np.zeros(land, bool)
        if k >= nreach + nlake:
            # Coastal fringe: sink if the parent sea touches it inside its piece;
            # parent sea across a cut becomes a terminal node (the parent outlet).
            for n in np.flatnonzero(s['to_sea']):
                ring = [nb for nb in _block_neighbours(blocks[n]) if nb not in index or index[nb] >= land]
                ocean = _sample_blocks(world, *np.array(ring, np.int64).T)['ocean'] if ring else []
                for nb, wet in zip(ring, ocean):
                    if wet and piece(nb) == piece(blocks[n]):
                        own[n] = True
                    elif wet and nb not in index:
                        index[nb] = len(blocks)
                        blocks.append(nb)
            arr = np.array(blocks, np.int64)
            s = _sample_blocks(world, arr[:, 0], arr[:, 1])
        # Fragments: connected unit blocks within one piece; each sea node alone.
        comp = np.full(len(blocks), -1, np.int64)
        ncomp = 0
        for n, b in enumerate(blocks):
            if comp[n] >= 0:
                continue
            if n >= land:
                comp[n], ncomp = ncomp, ncomp + 1
                continue
            comp[n], stack = ncomp, [b]
            while stack:
                cur = stack.pop()
                for nb in _block_neighbours(cur):
                    m = index.get(nb)
                    if m is not None and m < land and comp[m] < 0 and piece(nb) == piece(cur):
                        comp[m] = ncomp
                        stack.append(nb)
            ncomp += 1
        members = [np.flatnonzero(comp == c) for c in range(ncomp)]
        # Sinks of the unit itself.
        if k < nreach:
            own = np.array([b in _reach_blocks(world, k) for b in blocks])
        elif k < nreach + nlake:
            own = s['lake'] == k - nreach
        status = ['sea' if members[c][0] >= land else 'own' if own[members[c]].any() else None
                  for c in range(ncomp)]
        absorber = [None] * ncomp

        def entry_level(c, rid):
            pi, pj = piece(blocks[members[c][0]])
            r0, c0 = R.r0 + pi * prow, R.c0 + pj * pcol
            core = (c0 * COARSE_M, r0 * COARSE_M, (c0 + pcol) * COARSE_M, (r0 + prow) * COARSE_M)
            station = _piece_entry_station(world, rid, core)
            return None if station is None else float(R.level(world.reaches[rid], station))

        # Bound the extra lift from a distant upstream trunk relative to
        # this unit's own receiver. Otherwise one high absorber lifts every
        # shared link, even beside a low, physically compatible own reach.
        if k < nreach:
            authority = float(R.level(world.reaches[k], 0.))
        elif k < nreach + nlake:
            authority = float(world.lakes[k - nreach]['level_m'])
        else:
            authority = R.sea_level
        # Absorption by a trunk of the same lineage crossing the fragment.
        profile_ids = _unit_profile_ids(world, k, blocks)
        cands = sorted((v for v in profile_ids if v != k),
                       key=lambda v: (-float(world.reaches[v]['flow_m3s']), v))
        for c in range(ncomp):
            if status[c] is None:
                fragment = {blocks[n] for n in members[c]}
                for v in cands:
                    if not fragment & _reach_blocks(world, v):
                        continue
                    level = entry_level(c, v)
                    # Nearby upstream absorption is an intentional ownership
                    # approximation. Keep its established small relief range,
                    # but reject a remote trunk needing a strong correction.
                    if level is None or level > authority + R.p['strong_incision_m'] + 1e-8:
                        diag['rejected_high_absorber_candidates'] += 1
                        continue
                    absorber[c] = v
                    break
                if absorber[c] is not None:
                    status[c] = 'absorbed'
        # Bound each actual sink span at its sink piece's first raw entry.
        # Every rerouted span in that piece starts at this station or later,
        # so its canonical water cannot exceed that sink's ceiling.
        fixed = (float(world.lakes[k - nreach]['level_m']) if nreach <= k < nreach + nlake
                 else R.sea_level if k >= nreach + nlake else None)
        sink_level = [None] * ncomp
        for c in range(ncomp):
            rid = k if status[c] == 'own' and k < nreach else absorber[c]
            if rid is not None:
                sink_level[c] = entry_level(c, rid)
            elif status[c] == 'sea' or (status[c] == 'own' and fixed is not None):
                sink_level[c] = R.sea_level if status[c] == 'sea' else fixed
        reference = [v for v in sink_level if v is not None] + ([fixed] if fixed is not None else [])
        link_level = max(reference) if reference else float(s['filled'].max())
        # Block distance (unit block graph) to the nearest sink.
        delta = np.full(len(blocks), np.inf)
        front = [n for c in range(ncomp) if status[c] for n in members[c]]
        d = 0
        while front:
            delta[front] = d
            nxt = set()
            for n in front:
                for nb in _block_neighbours(blocks[n]):
                    m = index.get(nb)
                    if m is not None and not np.isfinite(delta[m]):
                        nxt.add(m)
            front, d = sorted(nxt), d + 1
        yields = _cell_density(world, s['cell']) * _block_area(world, arr[:, 0])
        areas = _block_area(world, arr[:, 0])
        inflow = [[] for _ in range(ncomp)]
        exit_of = {}
        for c in range(ncomp):
            if status[c] is None and np.isfinite(delta[members[c]].min()):
                exits = []            # block edge or corner on a cut
                for n in members[c]:
                    for nb in _block_neighbours(blocks[n]):
                        m = index.get(nb)
                        if m is not None and comp[m] != c and np.isfinite(delta[m]):
                            corner = nb[0] != blocks[n][0] and nb[1] != blocks[n][1]
                            exits.append((delta[m], corner, s['filled'][m], nb, blocks[n], m))
                exit_of[c] = min(exits)[3:]      # nearest sink, block edge before corner, lowest parent
        # One exit per fragment: links form trees with strictly lower sink
        # distance downstream, each rooted at one sink that bounds its level.
        tree_level = {}
        for c in sorted(range(ncomp), key=lambda c: (delta[members[c]].min(), int(members[c][0]))):
            tree_level[c] = (tree_level[comp[exit_of[c][2]]] if c in exit_of else
                             link_level if sink_level[c] is None else sink_level[c])
        order = sorted(range(ncomp), key=lambda c: (-delta[members[c]].min(), int(members[c][0])))
        for c in order:
            mine, flood_key = members[c][0], None
            if c in exit_of:
                nb, b, m = exit_of[c]
                q = math.fsum([*yields[members[c]], *(f for f, _ in inflow[c])])
                a = math.fsum([*areas[members[c]], *(x for _, x in inflow[c])])
                inflow[comp[m]].append((q, a))
                status[c] = 'linked'
                if piece(b) == (0, 0) or piece(nb) == (0, 0):
                    links.append(_link_site(R, k, b, nb, q, a, tree_level[c]))
                    if m >= land:              # into the parent sea across the cut: terminal
                        links[-1].update(downstream_sea=True, downstream_tributary=None)
                    if piece(b) == (0, 0):     # this core's fragment: flooded from its exit only
                        flood_key = ORPHAN_KEY + len(links) - 1
                        links[-1]['flood_key'] = flood_key
            if mine < land and piece(blocks[mine]) == (0, 0):
                diag[status[c] or 'stranded'] += len(members[c])
                if not np.isfinite(delta[members[c]].min()):
                    diag['unresolved_authority_blocks'] += len(members[c])
                for n in members[c]:
                    if status[c] == 'absorbed':
                        override[blocks[n]] = int(absorber[c])
                    elif flood_key is not None:
                        override[blocks[n]] = flood_key
    return override, links, diag


def _link_site(R, k, b, n, q, a, level):
    """Shared unit-link record on the block edge (or corner) between ``b`` (upstream) and ``n``."""
    (bi, bj), (ni, nj) = b, n
    world = R.world
    name = _key_name(world, k)
    channel = bool(a >= R.p['min_channel_area_m2'])
    record = dict(id=f'u{k}|{bi}_{bj}>{ni}_{nj}', kind='unit_link', unit_key=int(k), unit=name,
                  upstream_block=(int(bi), int(bj)), downstream_block=(int(ni), int(nj)),
                  flow_m3s=float(q), area_m2=float(a), channel=channel, downstream_sea=False)
    if bi != ni and bj != nj:                 # diagonal: the corner cell pair, fixed
        r0, c0 = min(bi, ni), min(bj, nj)
        strip = R.compose((r0, c0, r0 + 2, c0 + 2))
        corner = lambda i, j: ((BLOCK - 1 if i == r0 else BLOCK), (BLOCK - 1 if j == c0 else BLOCK))
        (ub, uc), (db, dc) = corner(bi, bj), corner(ni, nj)
        up = (r0 * BLOCK + ub, c0 * BLOCK + uc)
        down = (r0 * BLOCK + db, c0 * BLOCK + dc)
        record.update(axis='corner', line_m=None, xy=np.array([(c0 + 1) * COARSE_M, (r0 + 1) * COARSE_M]),
                      level_m=float(level), ground_m=float(.5 * (strip[ub, uc] + strip[db, dc])),
                      upstream_cell=up, downstream_cell=down,
                      downstream_tributary=f'{name}~g{down[0]}_{down[1]}' if channel else None)
        return record
    vertical = bj != nj                       # the cut is the line x = const
    strip = R.compose((min(bi, ni), min(bj, nj), max(bi, ni) + 1, max(bj, nj) + 1))
    strip = strip if vertical else strip.T    # (along, across); cut between 31 | 32
    line = (max(bj, nj) if vertical else max(bi, ni)) * COARSE_M
    g = (min(bi, ni) if vertical else min(bj, nj)) * BLOCK + np.arange(BLOCK)
    t = (g + .5) * BASE_M
    z = .5 * (strip[:, BLOCK - 1] + strip[:, BLOCK])
    # Keep clear of trunk portals on this edge: both sides hold the same portals.
    busy = np.zeros(BLOCK, bool)
    for rec in R.portal_cache.values():
        if rec['axis'] == ('x' if vertical else 'y') and rec['line_m'] == line:
            busy |= np.abs(t - rec['xy'][1 if vertical else 0]) <= 2.5 * BASE_M
    m = int(np.argmin(np.where(busy, np.inf, z) if not busy.all() else z))
    across = int(round(line / BASE_M))
    up_across = across - 1 if (bj < nj if vertical else bi < ni) else across
    down_across = across if up_across == across - 1 else across - 1
    cell = lambda c: (int(g[m]), c) if vertical else (c, int(g[m]))
    down = cell(down_across)
    record.update(axis='x' if vertical else 'y', line_m=float(line),
                  xy=np.array([line, t[m]]) if vertical else np.array([t[m], line]),
                  level_m=float(level), ground_m=float(z[m]),
                  upstream_cell=cell(up_across), downstream_cell=down,
                  downstream_tributary=f'{name}~g{down[0]}_{down[1]}' if channel else None)
    return record


@njit(cache=True)
def _channel_water(order, receivers, fixed, filled, ground, fill_m, incision_m):
    """Per-cell drawn water, the receiver-DAG form of ``_bounded_water``.

    At most ``fill_m`` above the lowest ground anywhere upstream, at most
    ``incision_m`` below the highest ground downstream, never below the fixed
    water it reaches and never above ``filled``. Cells with a finite ``fixed``
    level (seeds, drawn lakes) keep it. Each bound descends along receivers
    (a drawn lake never stands above its routed fill), so the result descends
    too and confluences agree by construction.
    """
    up = ground + fill_m
    low = filled.copy()
    out = filled.copy()
    for c in range(len(fixed)):
        if np.isfinite(fixed[c]):
            up[c] = low[c] = out[c] = fixed[c]
    for i in range(len(order) - 1, -1, -1):          # upstream first
        c = order[i]
        r = receivers[c]
        if r >= 0 and up[c] < up[r]:
            up[r] = up[c]
    for i in range(len(order)):                       # downstream first
        c = order[i]
        r = receivers[c]
        if np.isfinite(fixed[c]):
            continue
        low[c] = max(ground[c] - incision_m, low[r] if r >= 0 else filled[c])
        out[c] = min(filled[c], max(low[c], up[c]))
    return out, low


@njit(cache=True)
def _propagate_ceiling(order, receivers, ceiling):
    """Move each inlet ceiling downstream; receiver order is already acyclic."""
    for i in range(len(order) - 1, -1, -1):
        c = order[i]
        target = receivers[c]
        if target >= 0:
            ceiling[target] = min(ceiling[target], ceiling[c])


def _route_local(R, spans):
    p = R.p
    n = R.rows * R.cols
    H = R.H.ravel()
    key = R.key.ravel().astype(np.int64).copy()
    override, links, link_diag = _unit_links(R)
    exit_key = {rec['id']: rec.pop('flood_key') for rec in links if 'flood_key' in rec}
    flood = key.copy()                   # routing key: parent unit, absorbing trunk or linked fragment
    for (gi, gj), k in override.items():
        i, j = (gi - R.r0) * BLOCK, (gj - R.c0) * BLOCK
        flood.reshape(R.rows, R.cols)[i:i + BLOCK, j:j + BLOCK] = k
    filled = np.full(n, np.nan)
    rcv = np.full(n, -9, np.int64)       # -9: unvisited / ordinary routed cell marker
    seed_kind = np.zeros(n, np.int8)
    trunk_of = np.full(n, -1, np.int64)
    trunk_pos = np.full(n, -1, np.int64)
    basin = R.basin.ravel().copy()       # effective owner: trunk cells carry their reach's basin
    for k, span in enumerate(spans):
        path, water = span['path'], span['water']
        for i, c in enumerate(path):
            if trunk_of[c] >= 0:
                continue
            trunk_of[c], trunk_pos[c] = k, i
            key[c] = flood[c] = int(span['reach']['id'])
            basin[c] = int(span['reach']['basin_id'])
            filled[c] = water[i]
            seed_kind[c] = TRUNK_SEED
            rcv[c] = path[i + 1] if i + 1 < len(path) else (-2 if span['end']['kind'] == 'portal' else -1)
    # Unit links: exits seed at the shared level, inlets inject the shared flow.
    link_of, inlet_of = {}, {}
    inject_q, inject_a = np.zeros(n), np.zeros(n)
    for idx, rec in enumerate(links):
        if rec['id'] not in exit_key:
            cell = R.local(*rec['downstream_cell'])
            inject_q[cell] += rec['flow_m3s']
            inject_a[cell] += rec['area_m2']
            inlet_of[cell] = idx
            continue
        fk = exit_key[rec['id']]
        (gr, gc), vertical = rec['upstream_cell'], rec['axis'] == 'x'
        along = gr if vertical else gc
        lo = along // BLOCK * BLOCK
        cand = sorted(range(lo, lo + BLOCK), key=lambda t: (abs(t - along), t))
        cells = ([R.local(gr, gc)] if rec['axis'] == 'corner' else
                 [R.local(t, gc) if vertical else R.local(gr, t) for t in cand])
        free = [c for c in cells if seed_kind[c] == 0 and flood[c] == fk]
        if free:      # otherwise the fragment is stranded and handled below (flagged)
            c = free[0]
            seed_kind[c], filled[c], rcv[c] = LINK_SEED, rec['level_m'], -2
            link_of[c] = idx
    ocean = R.ocean.ravel()
    sea = ocean & (seed_kind == 0)
    seed_kind[sea], filled[sea], rcv[sea] = SEA_SEED, np.minimum(H[sea], R.sea_level), -1
    lake_id = R.lake.ravel()
    still = (lake_id >= 0) & (seed_kind == 0)
    seed_kind[still], filled[still], rcv[still] = PARENT_LAKE_SEED, R.lake_level[lake_id[still]], -1
    extension = still & (R.lake_block.ravel() < 0)
    key[extension] = len(R.world.reaches) + lake_id[extension]
    visited = seed_kind > 0
    order = np.empty(n, np.int64)
    sea = seed_kind == SEA_SEED
    to_sea = R.to_sea.ravel()
    n_order = _flood(H, flood, sea, to_sea, R.rows, R.cols, visited, seed_kind > 0, filled, rcv,
                     np.flatnonzero(visited), order, 0)
    phase_a = n_order
    # Stranded remnants (no own sink, link or absorbing trunk reachable inside
    # their key) join an adjacent connected area of the same basin, flagged.
    if not visited.all():
        eligible = visited & (flood < ORPHAN_KEY) & (seed_kind != LINK_SEED)
        ring = ndimage.binary_dilation(~visited.reshape(R.rows, R.cols), np.ones((3, 3), bool)).ravel()
        n_order = _flood(H, basin, sea, to_sea, R.rows, R.cols, visited, visited.copy(), filled, rcv,
                         np.flatnonzero(ring & eligible), order, n_order)
    stranded = order[phase_a:n_order].copy()
    terminals = 0
    while not visited.all():
        # Enclosed fragments without any inherited outlet: one local pit per
        # component (lowest cell, then lowest index), flagged in quality.
        labels, count = ndimage.label(~visited.reshape(R.rows, R.cols), np.ones((3, 3), bool))
        flat = labels.ravel()
        rest = np.flatnonzero(flat)
        rest = rest[np.lexsort((rest, H[rest], flat[rest]))]
        pits = rest[np.r_[True, flat[rest][1:] != flat[rest][:-1]]]
        seed_kind[pits], filled[pits], rcv[pits], visited[pits] = TERMINAL_SEED, H[pits], -1, True
        terminals += pits.size
        n_order = _flood(H, flood, sea, to_sea, R.rows, R.cols, visited, seed_kind > 0, filled, rcv,
                         pits, order, n_order)
    order = order[:n_order]
    seed = seed_kind > 0
    rcv[seed & (rcv == -9)] = -1
    # Link inlets are fixed boundary levels, not ground heights. An obstacle
    # downstream must be incised rather than silently raising the inlet join.
    ceiling = np.full(n, np.inf)
    for cell, idx in inlet_of.items():
        ceiling[cell] = min(ceiling[cell], links[idx]['level_m'])
    _propagate_ceiling(order, rcv, ceiling)
    conflict = seed & (seed_kind != TERMINAL_SEED) & (filled > ceiling + 1e-8)
    if conflict.any():
        raise ValueError('unit inlet lies below its fixed downstream water authority: '
                         f'{float(np.max(filled[conflict] - ceiling[conflict])):.6g} m')
    lowered = np.maximum(filled - ceiling, 0.)
    filled = np.minimum(filled, ceiling)
    inlet_diagnostics = dict(inlet_ceiling_lowering_max_m=float(lowered.max(initial=0.)),
                             inlet_ceiling_cells=int(np.count_nonzero(lowered > 0)),
                             inlet_ceiling_incision_max_m=float(np.maximum(H[lowered > 0] - filled[lowered > 0], 0.).max(initial=0.)))
    cls = np.where(seed_kind == TERMINAL_SEED, TERMINAL, NETWORK).astype(np.int8)
    _inherit(order, rcv, cls)
    source = R.yield_m3s.ravel() + inject_q
    area, _ = _accumulate(order, order.size, rcv, seed, R.cell_area.ravel() + inject_a)
    flow, flow_into = _accumulate(order, order.size, rcv, seed, source)

    # Beyond the shore envelope, water filled to a parent lake's level from
    # that lake is its continuation, never a separate pond level.
    from_parent = np.where(seed_kind == PARENT_LAKE_SEED, lake_id, -1).astype(np.int64)
    _inherit_still(order, rcv, filled, from_parent)
    # Drawn channel water: filled water would float over hidden ponds (rejected
    # key-wall impoundments, edge-touching floods), so channels cut their sills.
    # A pond drawn below its routed level must still drain downhill: drop it
    # (undrawn, reported) while the lowest water its outlet can reach is higher.
    rank = np.full(n, -1, np.int64)
    rank[order] = np.arange(order.size)
    _, floor = _channel_water(order, rcv, np.where(seed, filled, np.nan), filled, H,
                              p['bank_fill_m'], p['strong_incision_m'])
    lakes, lake_of, walls = _local_lakes(R, H, filled, seed, cls, trunk_of, flood,
                                         drain=(rcv, rank, floor), parent=from_parent)
    while True:
        fixed = np.where(seed, filled, np.nan)
        for lake in lakes:
            fixed[lake['cells']] = lake['level_m']
        shown, floor = _channel_water(order, rcv, fixed, filled, H, p['bank_fill_m'], p['strong_incision_m'])
        below = [int(rcv[lk['cells'][np.argmin(rank[lk['cells']])]]) for lk in lakes]
        keep = [lk for lk, b in zip(lakes, below) if lk['level_m'] >= lk['routed_level_m']
                or b < 0 or floor[b] <= lk['level_m'] + 1e-9]
        if len(keep) == len(lakes):
            break
        walls['undrawn'] += len(lakes) - len(keep)
        lakes = keep
    lake_of[:] = -1
    for lid, lake in enumerate(lakes):
        lake_of[lake['cells']] = lid
    for lake in lakes:
        parent = from_parent[lake['cells']]
        if (parent >= 0).any():
            lake.update(parent_lake_id=int(parent[parent >= 0].min()),
                        unit_key=len(R.world.reaches) + int(parent[parent >= 0].min()))
    below_parent = ~seed & (from_parent >= 0) & (lake_of < 0) & (filled - H >= p['min_lake_depth_m'])
    channel = (~seed & (cls == NETWORK) & (area >= p['min_channel_area_m2']) & (lake_of < 0))
    inlet = np.zeros(n, bool)
    inlet[list(inlet_of)] = True
    tribs = _tributaries(R, channel, rcv, lake_of, lakes, seed, inlet, order)
    # A shifted trunk or lake may already occupy an inlet cell. Keep a short
    # connector from the shared point/level to that sink rather than a dangling
    # downstream_tributary id (the two water authorities need not be equal).
    for cell, idx in inlet_of.items():
        if links[idx]['channel'] and not links[idx]['downstream_sea'] and not channel[cell]:
            tribs.append(dict(start=cell, cells=np.array([cell], np.int64), end=cell))
    tribs.sort(key=lambda t: t['start'])
    trunk_reaches, span_flow = _trunk_records(R, spans, tribs, flow, flow_into, trunk_of, trunk_pos)
    reaches = _finish_tributaries(R, tribs, trunk_reaches, spans, trunk_of, trunk_pos, lakes, lake_of,
                                  shown, flow, area, rcv, seed_kind, links, link_of, inlet_of)
    lakes += _shore_records(R, lake_id, extension)

    # Physical surface: bounded local carve D, U = H - D, conditioned = min(H, U).
    bed = np.full(n, np.inf)
    water_grid = np.full(n, np.nan)
    for span, q in zip(spans, span_flow):
        width, depth = _bankfull(q)
        cells = span['path']
        cut = span['water'] - depth * np.minimum(1.0, width / BASE_M)
        np.minimum.at(bed, cells, cut)
        water_grid[cells] = span['water']
    for rec in reaches:
        if rec['kind'] != 'tributary':
            continue
        cells = rec['cells']
        k0 = 1 if rec['start_link'] else 0
        water = rec['water_m'][k0:k0 + len(cells)]
        width, depth = _bankfull(flow[cells])
        np.minimum.at(bed, cells, water - depth * np.minimum(1.0, width / BASE_M))
        water_grid[cells] = water
    channel_fill = float(np.nanmax(water_grid - H, initial=0.))
    for lake in lakes:
        water_grid[lake['cells']] = lake['level_m']
    upper = _envelope(R, bed.reshape(R.rows, R.cols), list(R.portal_cache.values()) + links)
    conditioned = np.minimum(R.H, upper)

    flow_grid = np.where(seed, np.nan, flow)
    for span, q in zip(spans, span_flow):
        flow_grid[span['path']] = q
    quality = _quality(R, spans, key, basin, seed, rcv, cls, flow_into, seed_kind, conditioned,
                       terminals, phase_a, source, stranded, links, link_of, link_diag, extension,
                       below_parent)
    # Channel water left under the conditioned ground (edge strips fade the own carve).
    quality['hidden_water_max_m'] = float(np.nanmax(conditioned.ravel() - water_grid, initial=0.0))
    quality.update(inlet_diagnostics)
    quality['unit_link_fill_max_m'] = max((max(r['level_m'] - r['ground_m'], 0.) for r in links), default=0.)
    quality['channel_water_above_ground_max_m'] = channel_fill
    excess = _containment(water_grid.reshape(R.rows, R.cols), conditioned)
    quality.update(lateral_containment_max_m=float(np.nanmax(excess, initial=0.0)),
                   uncontained_cells=int(np.count_nonzero(excess > p['bank_fill_m'])),
                   key_wall_impoundments=walls['count'], key_wall_area_m2=walls['area_m2'],
                   key_wall_excess_max_m=walls['max_excess_m'], undrawn_ponds=walls['undrawn'])
    portals = sorted({pid: rec for pid, rec in R.portal_cache.items()}.values(), key=lambda r: r['id'])
    return RegionPlan(
        bounds=R.core, origin=R.g0, shape=(R.rows, R.cols), reaches=trunk_reaches + reaches,
        local_lakes=lakes, portals=portals, routing_height_m=R.H, conditioned_base_m=conditioned,
        upper_m=upper, water_m=water_grid.reshape(R.rows, R.cols),
        unit_key=key.reshape(R.rows, R.cols), basin_ids=basin.reshape(R.rows, R.cols),
        receivers=rcv.reshape(R.rows, R.cols), area_m2=area.reshape(R.rows, R.cols),
        flow_m3s=flow_grid.reshape(R.rows, R.cols), drain_class=cls.reshape(R.rows, R.cols),
        metadata=dict(quality=quality), links=sorted(links, key=lambda r: r['id']))


def _shore_records(R, lake_id, extension):
    """Parent-lake shore extensions: water at the parent level, owned by the parent lake."""
    out = []
    H = R.H.ravel()
    for L in np.unique(lake_id[extension]):
        cells = np.flatnonzero(extension & (lake_id == L))
        parent = R.world.lakes[int(L)]
        out.append(dict(id=f"P{int(L)}@{R.gid(int(cells[0]))[1:]}", parent_lake_id=int(L),
                        level_m=float(parent['level_m']), closed=bool(parent['closed']), flow_through=False,
                        area_m2=float(R.cell_area.ravel()[cells].sum()), cells=cells,
                        max_depth_m=float(parent['level_m'] - H[cells].min()),
                        basin_id=int(parent['basin_id']), unit_key=len(R.world.reaches) + int(L),
                        xy=R.xy_of(cells).mean(0), entry_reaches=[], exit_reach=None,
                        lowest_cell=int(cells[np.argmin(H[cells])]), outlet_cell=-1))
    return out


def _containment(water, ground):
    """Per wet cell: lowest adjacent water (3 x 3, itself included) minus the
    lowest adjacent dry ground. Positive values are water standing above a bank
    it would spill over, whatever the channel's own downstream slope."""
    rows, cols = water.shape
    wp = np.pad(water, 1, constant_values=np.nan)
    gp = np.pad(ground, 1, constant_values=np.inf)
    low_w, low_g = water.copy(), np.full(water.shape, np.inf)
    for di in range(3):
        for dj in range(3):
            nw, ng = wp[di:di + rows, dj:dj + cols], gp[di:di + rows, dj:dj + cols]
            low_w = np.fmin(low_w, nw)
            low_g = np.where(np.isnan(nw), np.minimum(low_g, ng), low_g)
    return np.where(np.isfinite(water), low_w - low_g, np.nan)


def _envelope(R, bed, shared):
    """Upper envelope from a bounded carve, with shared-only edge strips.

    Own channels carve ``D = max(0, min(H - bed - s d, cut (1 - d / r)))``
    around the nearest channel cell (``cut`` is that cell's H - bed, ``r`` is
    ``bank_radius_cells``), faded to zero across ``edge_strip_cells`` at every
    core edge. Shared records (trunk portals and unit links, identical in both
    pieces) carve the same function around their point on the cut, faded only
    towards the perpendicular piece lines. At a cut both sides therefore apply
    one function of position and of the pure H: no step. ``U = H - D`` where
    ``D > 0``, else +inf.
    """
    p, H = R.p, R.H
    radius, strip = float(p['bank_radius_cells']), float(p['edge_strip_cells'])
    slope = p['bank_slope'] * BASE_M                       # rise per cell
    carve = np.zeros_like(H)
    if np.isfinite(bed).any():
        dist, (ii, jj) = ndimage.distance_transform_edt(~np.isfinite(bed), return_indices=True)
        b = bed[ii, jj]
        cut = np.maximum(H[ii, jj] - b, 0.0)
        carve = np.maximum(np.minimum(H - b - slope * dist, cut * np.clip(1 - dist / radius, 0, 1)), 0.0)
    fade = lambda d: np.clip((d - .5) / strip, 0, 1)
    i, j = np.arange(R.rows) + .5, np.arange(R.cols) + .5
    carve *= fade(np.minimum(np.minimum(i, R.rows - i)[:, None], np.minimum(j, R.cols - j)[None, :]))
    prow, pcol = (R.r1 - R.r0) * BLOCK, (R.c1 - R.c0) * BLOCK
    perp_rows = fade(np.minimum(i % prow, prow - i % prow))[:, None]     # to horizontal piece lines
    perp_cols = fade(np.minimum(j % pcol, pcol - j % pcol))[None, :]     # to vertical piece lines
    reach = int(math.ceil(radius)) + 1
    for rec in shared:
        width, depth = _bankfull(rec['flow_m3s'])
        depth = float(depth * min(1.0, width / BASE_M))
        x, y = rec['xy']
        ci, cj = int(math.floor(y / BASE_M)) - R.g0[0], int(math.floor(x / BASE_M)) - R.g0[1]
        rs = slice(max(0, ci - reach), min(R.rows, ci + reach + 1))
        cs = slice(max(0, cj - reach), min(R.cols, cj + reach + 1))
        if rs.start >= rs.stop or cs.start >= cs.stop:
            continue
        d = np.hypot(R.cx[None, cs] - x, R.cy[rs, None] - y) / BASE_M
        g = np.maximum(np.minimum(H[rs, cs] - (rec['level_m'] - depth) - slope * d,
                                  depth * np.clip(1 - d / radius, 0, 1)), 0.0)
        g = g * (perp_rows[rs] if rec['axis'] == 'x' else perp_cols[:, cs] if rec['axis'] == 'y'
                 else perp_rows[rs] * perp_cols[:, cs])
        carve[rs, cs] = np.maximum(carve[rs, cs], g)
    return np.where(carve > 0, H - carve, np.inf)


def _local_lakes(R, H, filled, seed, cls, trunk_of, flood, *, drain=None, parent=None):
    """Flow-through local lakes: equal-level filled components fully inside the core.

    Components are labelled per flood key: the keyed Priority-Flood gives one
    level only inside a key. The rim surface is seed water or bare ground
    (neighbouring floods are not assumed drawn). A component standing more than
    ``min_lake_depth_m`` above its rim is impounded by the unit-block ownership
    partition (a key wall) or by a lower river. Drawn channels breach its sill
    (``_channel_water``) down to the lowest water its routed outlet reaches
    (``drain = (receivers, rank, floor)``; otherwise its rim level). It is drawn
    only as the ponds holding water at that level within their own rims;
    ``routed_level_m`` keeps the routing level. Routing is unchanged.
    Continuations of a parent lake (``parent`` >= 0) keep the fixed parent level.
    """
    p = R.p
    depth = p['min_lake_depth_m']
    wet = ~seed & (cls == NETWORK) & (filled - H >= depth)
    lake_of = np.full(H.size, -1, np.int64)
    lakes, walls = [], dict(count=0, area_m2=0.0, max_excess_m=0.0, undrawn=0)
    flat = np.zeros(H.size, np.int64)
    count = 0
    for k in np.unique(flood[wet]):
        lab, n = ndimage.label((wet & (flood == k)).reshape(R.rows, R.cols), np.ones((3, 3), bool))
        lab = lab.ravel()
        flat[lab > 0] = lab[lab > 0] + count
        count += n
    if not count:
        return lakes, lake_of, walls
    order = np.argsort(flat, kind='stable')
    cuts = np.searchsorted(flat[order], np.arange(count + 2))
    trunk = (trunk_of >= 0).reshape(R.rows, R.cols)
    near_trunk = ndimage.binary_dilation(trunk, np.ones((3, 3), bool)).ravel()
    # Padded by one cell: beyond the core the rim is unknown (never holds water).
    surface = np.pad(np.where(seed, filled, H).reshape(R.rows, R.cols), 1, constant_values=-np.inf)
    ground = np.pad(H.reshape(R.rows, R.cols), 1, constant_values=np.inf)

    def window(cells):
        r, c = np.divmod(cells, R.cols)
        box = (slice(r.min(), r.max() + 3), slice(c.min(), c.max() + 3))
        mask = np.zeros((r.max() - r.min() + 3, c.max() - c.min() + 3), bool)
        mask[r - r.min() + 1, c - c.min() + 1] = True
        return box, mask, ndimage.binary_dilation(mask, np.ones((3, 3), bool)) & ~mask

    for comp in range(1, count + 1):
        cells = np.sort(order[cuts[comp]:cuts[comp + 1]])
        if cells.size < p['min_lake_cells']:
            continue
        routed = float(filled[cells].max())
        box, inside, rim = window(cells)
        spill = float(surface[box][rim].min())
        if spill == -np.inf:
            continue          # touches an unknown boundary: not an owned closed shore
        level = routed
        if parent is None or not (parent[cells] >= 0).any():
            if routed - spill > depth:
                walls['count'] += 1
                walls['area_m2'] += float(R.cell_area.ravel()[cells].sum())
                walls['max_excess_m'] = max(walls['max_excess_m'], routed - spill)
                level = spill
                if drain is not None:          # drawn channels breach its sill down to this
                    receivers, rank, floor = drain
                    below = receivers[cells[np.argmin(rank[cells])]]
                    level = float(floor[below]) if below >= 0 else level
            # An inlet ceiling path through the pond fixes it at that lower level.
            level = min(level, float(filled[cells].min()))
        parts = [(cells, routed)]
        if level < routed:
            # Component cells left dry are >= level - depth; outer rims are checked.
            lab, n = ndimage.label(inside & (ground[box] <= level - depth), np.ones((3, 3), bool))
            r0, c0 = box[0].start - 1, box[1].start - 1
            parts = []
            for q in range(1, n + 1):
                i, j = np.nonzero(lab == q)
                parts.append(((i + r0) * R.cols + j + c0, level))
        for cells, level in parts:
            if cells.size < p['min_lake_cells']:
                continue
            cells = np.sort(cells)
            if level >= spill - depth and level < routed:
                pbox, _, ring = window(cells)
                if surface[pbox][ring].min() < level - depth:
                    walls['undrawn'] += 1      # this pond would spill over a lower or unknown rim
                    continue
            lowest = int(cells[np.lexsort((cells, H[cells]))[0]])
            lid = len(lakes)
            lake_of[cells] = lid
            lakes.append(dict(id=f'L{R.gid(lowest)[1:]}', level_m=level, closed=False, flow_through=True,
                              area_m2=float(R.cell_area.ravel()[cells].sum()), cells=cells,
                              max_depth_m=float(level - H[cells].min()),
                              level_spread_m=float(np.maximum(level - filled[cells], 0).max()),
                              routed_level_m=routed, on_trunk=bool(near_trunk[cells].any()),
                              basin_id=int(R.basin.flat[lowest]), unit_key=int(R.key.flat[lowest]),
                              xy=R.xy_of(cells).mean(0), entry_reaches=[], exit_reach=None,
                              lowest_cell=lowest, outlet_cell=-1))
    return lakes, lake_of, walls


def _tributaries(R, channel, rcv, lake_of, lakes, seed, inlet, order):
    """Trace channel cells into reaches between heads, confluences and lake exits."""
    chan = np.flatnonzero(channel)
    ndon = np.zeros(rcv.size, np.int64)
    targets = rcv[chan]
    ok = targets >= 0
    np.add.at(ndon, targets[ok], 1)
    from_lake = np.zeros(rcv.size, bool)
    rank = np.full(rcv.size, -1, np.int64)
    rank[order] = np.arange(len(order))
    for lid, lake in enumerate(lakes):
        # A morphological water component can be left and re-entered along
        # an acyclic receiver chain. Starting at its deepest cell and stopping
        # at the first channel can therefore pick an INLET as its outlet.
        # Exit from the component's downstream-most receiver rank instead:
        # every later receiver has lower rank than every cell in this lake,
        # so condensed lake/reach links cannot return to it, even on flats.
        outlet = int(lake['cells'][np.argmin(rank[lake['cells']])])
        lake['outlet_from_cell'] = outlet
        c = int(rcv[outlet])
        while c >= 0 and not (seed[c] or channel[c] or lake_of[c] not in (-1, lid)):
            c = int(rcv[c])
        lake['outlet_cell'] = c
        if c >= 0:
            from_lake[c] = True
    is_start = channel & ((ndon != 1) | from_lake | inlet)
    starts = np.flatnonzero(is_start)
    tribs = []
    for s in starts:
        cells = [int(s)]
        c = int(s)
        while True:
            r = int(rcv[c])
            if r < 0 or seed[r] or lake_of[r] >= 0 or is_start[r] or not channel[r]:
                break
            cells.append(r)
            c = r
        tribs.append(dict(start=int(s), cells=np.array(cells, np.int64), end=int(rcv[c])))
    return tribs


def _trunk_records(R, spans, tribs, flow, flow_into, trunk_of, trunk_pos):
    """Trunk segments split at fine confluences; flows partition the parent increment."""
    joins = {}
    for t in tribs:
        e = t['end']
        if e >= 0 and trunk_of[e] >= 0:
            joins.setdefault(int(trunk_of[e]), set()).add(int(trunk_pos[e]))
    records, span_flow = [], []
    yield_ = R.yield_m3s.ravel()
    for k, span in enumerate(spans):
        reach, path, water = span['reach'], span['path'], span['water']
        own = trunk_of[path] == k
        lateral = np.where(own, yield_[path] + flow_into[path], 0.0)
        total = lateral.sum()
        frac = (np.cumsum(lateral) / total if total > 0
                else np.linspace(0, 1, len(path)) if len(path) > 1 else np.ones(1))
        q0, q1 = span['start']['flow'], span['end']['flow']
        q = q0 + (q1 - q0) * frac
        q[0], q[-1] = q0, q1
        span_flow.append(q)
        span['diagnostics'].update(parent_increment_m3s=float(q1 - q0), fine_lateral_m3s=float(total))
        cuts = sorted({0, *(j for j in joins.get(k, ()) if 0 < j < len(path) - 1)})
        keys = [span['start']['key'] if j == 0 else R.gid(path[j]) for j in cuts]
        rid = reach['id']
        span['segments'] = (cuts, [f'{rid}#{key}' for key in keys])
        for i, j in enumerate(cuts):
            stop = cuts[i + 1] if i + 1 < len(cuts) else len(path) - 1
            seg = slice(j, stop + 1)
            if i + 1 < len(cuts):
                down = f'{rid}#{keys[i + 1]}'
            elif span['end']['kind'] == 'portal' and not span['end']['portal']['downstream_sea']:
                down = f"{rid}#p{span['end']['portal']['id']}"
            elif reach.get('downstream_id') is not None:
                down = f"{reach['downstream_id']}#src"
            else:
                down = None
            cells = path[seg]
            xy_seg, w_seg, q_seg = R.xy_of(cells), water[seg].copy(), q[seg].copy()
            # Polylines meet exactly on the shared portal point and level.
            for end_, at in (('start', 0), ('end', -1)):
                if span[end_]['kind'] == 'portal' and (i == 0 if at == 0 else i + 1 == len(cuts)):
                    portal = span[end_]['portal']
                    xy_seg = np.insert(xy_seg, len(xy_seg) if at else 0, portal['xy'], axis=0)
                    w_seg = np.insert(w_seg, len(w_seg) if at else 0, portal['level_m'])
                    q_seg = np.insert(q_seg, len(q_seg) if at else 0, portal['flow_m3s'])
            records.append(dict(
                id=f'{rid}#{keys[i]}', parent_id=rid, kind='trunk', level=240,
                source=keys[i], target=(keys[i + 1] if i + 1 < len(cuts) else
                                        (f"p{span['end']['portal']['id']}" if span['end']['kind'] == 'portal'
                                         else span['end'].get('node_kind'))),
                basin_id=reach['basin_id'], xy=xy_seg, water_m=w_seg,
                flow_m3s=float(q_seg[-1]), flow_profile_m3s=q_seg,
                area_m2=float(reach['area_m2']), downstream_id=down, cells=cells,
                start_portal=span['start'].get('portal', {}).get('id') if i == 0 else None,
                end_portal=(span['end']['portal']['id'] if span['end']['kind'] == 'portal'
                            and i + 1 == len(cuts) else None),
                start_tangent=(span['start']['portal']['tangent'] if i == 0 and span['start']['kind'] == 'portal'
                               else None),
                end_tangent=(span['end']['portal']['tangent'] if i + 1 == len(cuts)
                             and span['end']['kind'] == 'portal' else None),
                pins=[0, len(xy_seg) - 1], diagnostics=span['diagnostics']))
    return records, span_flow


def _finish_tributaries(R, tribs, trunk_reaches, spans, trunk_of, trunk_pos, lakes, lake_of,
                        filled, flow, area, rcv, seed_kind, links, link_of, inlet_of):
    world = R.world
    by_start = {t['start']: f'{_owner_name(R, t["start"])}~{R.gid(t["start"])}' for t in tribs}

    def trunk_segment(cell):
        cuts, ids = spans[int(trunk_of[cell])]['segments']
        return ids[bisect.bisect_right(cuts, int(trunk_pos[cell])) - 1]

    def destination(cell):
        """First network element downstream of ``cell`` (walks the receiver chain)."""
        while True:
            if cell < 0:
                return None, 'edge' if cell == -3 else 'terminal'
            if trunk_of[cell] >= 0:
                return trunk_segment(cell), 'trunk'
            if lake_of[cell] >= 0:
                lake = lakes[lake_of[cell]]
                return destination(lake['outlet_cell'])[0], f"lake:{lake['id']}"
            if cell in by_start:
                return by_start[cell], 'confluence'
            kind = seed_kind[cell]
            if kind == PARENT_LAKE_SEED:
                parent = world.lakes[int(R.lake.flat[cell])]
                exit_reach = parent.get('exit_reach')
                return (f'{exit_reach}#src' if exit_reach is not None and not parent['closed'] else None,
                        f"parent_lake:{parent['id']}")
            if kind == SEA_SEED:
                return None, 'sea'
            if kind == LINK_SEED:
                link = links[link_of[cell]]
                return link['downstream_tributary'], 'sea' if link['downstream_sea'] else f"link:{link['id']}"
            if kind in (EDGE_SEED, TERMINAL_SEED):
                return None, 'edge' if kind == EDGE_SEED else 'terminal'
            cell = int(rcv[cell])

    records = []
    for t in tribs:
        cells = t['cells']
        end = t['end']
        down, target = destination(end)
        xy = R.xy_of(cells)
        water = filled[cells]
        end_link = start_link = None
        if end >= 0:
            end_xy = R.xy_of([end])[0]
            if target.startswith(('lake', 'parent_lake', 'sea')) and end not in link_of:
                end_xy = .5 * (xy[-1] + end_xy)
            xy = np.vstack((xy, end_xy))
            water = np.append(water, filled[end])
            if end in link_of:                  # ends exactly on the shared link point
                end_link = links[link_of[end]]
                xy = np.vstack((xy, end_link['xy']))
                water = np.append(water, end_link['level_m'])
        if t['start'] in inlet_of:              # starts exactly on the shared link point
            start_link = links[inlet_of[t['start']]]
            xy = np.vstack((start_link['xy'], xy))
            water = np.r_[start_link['level_m'], water]
        rid = by_start[t['start']]
        records.append(dict(
            id=rid, parent_id=_owner_parent(R, t['start']), kind='tributary', level=240,
            source=R.gid(t['start']), target=target, basin_id=int(R.basin.flat[t['start']]),
            xy=xy, water_m=water, flow_m3s=float(max(flow[cells[-1]], 0.0)),
            area_m2=float(area[cells[-1]]), downstream_id=down, cells=cells, pins=[0, len(xy) - 1],
            start_link=start_link['id'] if start_link else None, end_link=end_link['id'] if end_link else None))
        if target.startswith('lake:'):
            lid = lake_of[end]
            lakes[lid]['entry_reaches'].append(rid)
    for lake in lakes:
        o = lake['outlet_cell']
        lake['exit_reach'], lake['exit_kind'] = destination(o)
    records.sort(key=lambda r: r['id'])
    return records


def _owner_name(R, cell):
    return _key_name(R.world, int(R.key.flat[cell]))


def _owner_parent(R, cell):
    k = int(R.key.flat[cell])
    return k if 0 <= k < len(R.world.reaches) else None


def _quality(R, spans, key, basin, seed, rcv, cls, flow_into, seed_kind, conditioned,
             terminals, phase_a, source, stranded, links, link_of, link_diag, extension, below_parent):
    """Diagnostics of parent preservation, ownership and water accounting."""
    H = R.H
    means = H.reshape(R.r1 - R.r0, BLOCK, R.c1 - R.c0, BLOCK).mean(axis=(1, 3))
    # Unaligned (Base-only) routing has no coarse target: its error is reported as NaN.
    coarse_core = (_crop(R.coarse, R.co, R.r0, R.c0, R.r1, R.c1, 'coarse') if R.coarse is not None
                   else np.full_like(means, np.nan))
    base_core = _crop(R.base, R.bo, R.g0[0], R.g0[1], R.g0[0] + R.rows, R.g0[1] + R.cols, 'base')
    routed = np.flatnonzero(~seed)
    r = rcv[routed]
    valid = r >= 0
    cross = (basin[routed[valid]] != basin[r[valid]]) & ~(seed_kind[r[valid]] == SEA_SEED)
    # Divide disagreement: label-boundary land cells whose steepest D8 descent crosses the label.
    pad = np.pad(H, 1, constant_values=np.inf)
    kpad = np.pad(key.reshape(H.shape), 1, constant_values=-2)
    best = np.zeros(H.shape)
    target = np.full(H.shape, -2, np.int64)
    boundary = np.zeros(H.shape, bool)
    for di, dj in zip(DI, DJ):
        nb = pad[1 + di:1 + di + H.shape[0], 1 + dj:1 + dj + H.shape[1]]
        nk = kpad[1 + di:1 + di + H.shape[0], 1 + dj:1 + dj + H.shape[1]]
        boundary |= (nk != key.reshape(H.shape)) & (nk >= 0)
        drop = (H - nb) / math.hypot(di, dj)
        better = drop > best
        best = np.where(better, drop, best)
        target = np.where(better, nk, target)
    land = (key.reshape(H.shape) >= 0) & ~seed.reshape(H.shape) & boundary
    disagree = land & (target >= 0) & (target != key.reshape(H.shape))
    into = {name: float(flow_into[seed_kind == kind].sum() + source[seed_kind == kind].sum())
            for name, kind in (('trunk', TRUNK_SEED), ('parent_lakes', PARENT_LAKE_SEED),
                               ('sea', SEA_SEED), ('link_exits', LINK_SEED), ('local_pits', TERMINAL_SEED))}
    total = float(R.yield_m3s.sum())
    inside = lambda b: R.r0 <= b[0] < R.r1 and R.c0 <= b[1] < R.c1
    inlets = [r for r in links if inside(r['downstream_block'])]
    inflow = float(sum(r['flow_m3s'] for r in inlets))
    exits = [(links[i], float(flow_into[c] + source[c])) for c, i in link_of.items()]
    cell_area = R.cell_area.ravel()
    return dict(
        block_mean_error_max_m=float(np.max(np.abs(means - coarse_core))),
        block_mean_error_rms_m=float(np.sqrt(np.mean((means - coarse_core) ** 2))),
        coarse_base_offset_rms_m=float(np.sqrt(np.mean((H - base_core) ** 2))),
        cross_basin_links=int(np.count_nonzero(cross)),
        divide_disagreement=float(disagree.sum() / max(land.sum(), 1)),
        orphan_area_m2=float(cell_area[stranded].sum()),
        local_pit_area_m2=float(cell_area[cls == TERMINAL].sum()), local_pits=int(terminals),
        fragment_blocks=link_diag, links_out=len(exits), links_in=len(inlets),
        link_exit_area_m2=float(sum(r['area_m2'] for r, _ in exits)),
        link_flow_mismatch_m3s=float(sum(abs(r['flow_m3s'] - q) for r, q in exits)),
        parent_lake_extension_area_m2=float(cell_area[extension].sum()),
        # Dry land filled to a parent lake level that touches the core edge (not drawn as water).
        parent_lake_conflict_area_m2=float(cell_area[below_parent].sum()),
        yield_m3s=total, link_inflow_m3s=inflow, yield_destination_m3s=into,
        yield_residual_m3s=float(total + inflow - sum(into.values())),
        max_conditioning_m=float(np.max(R.H - conditioned, initial=0.0)),
        spans=[dict(reach_id=s['reach']['id'], **s['diagnostics']) for s in spans],
        strong_corrections=int(sum(s['diagnostics']['strong_correction'] for s in spans)),
        network_cells_phase_a=int(phase_a))


# ----------------------------------------------------------------------------
# 30 m: aligned composition and geometry refit on the fixed 240 m graph

def compose_decoder(plan, decoder_height, decoder_bounds, rect_bounds, *, steps=1, omega=1.6):
    """Decoder detail aligned on the conditioned base, then the channel envelope re-applied.

    ``rect_bounds`` (multiples of 240 m) is the output window; the decoder input
    must cover it plus ``1 + steps`` base cells and that grown window must lie
    inside the plan core. Returns (height_30m, info).
    """
    origin = _grid_origin(decoder_height, decoder_bounds, DECODER_M, 'decoder')
    x0, y0, x1, y1 = (int(round(v / BASE_M)) for v in rect_bounds)
    rect = (y0, x0, y1, x1)
    out = aligned_detail(plan.conditioned_base_m, plan.origin, decoder_height, origin, 8, rect,
                         steps=steps, omega=omega)
    upper = _crop(plan.upper_m, plan.origin, y0 - 1, x0 - 1, y1 + 1, x1 + 1, 'plan upper')
    finite = np.isfinite(upper)
    envelope = _bilinear_up(np.where(finite, upper, 1e12), 8) if finite.any() else None
    if envelope is not None:
        out = np.minimum(out, envelope)
    parent = _crop(plan.conditioned_base_m, plan.origin, y0, x0, y1, x1, 'plan')
    means = out.reshape(parent.shape[0], 8, parent.shape[1], 8).mean(axis=(1, 3))
    return out, dict(block_mean_error_max_m=float(np.max(np.abs(means - parent))))


def refit_reach_30m(reach, decoder_height, decoder_bounds, *, max_shift_m=105.0, smooth_m=4.0):
    """Shift a 240 m reach by at most ``max_shift_m`` towards the 30 m valley floor.

    Topology, ids, endpoints (and any ``pins``), downstream links and the
    water profile's pinned levels are unchanged; only intermediate vertices
    move perpendicular to the course. Viterbi over lateral offsets with unit
    steps; ``smooth_m`` is the height-equivalent cost of one lateral step.
    """
    from scipy.ndimage import map_coordinates
    xy = np.asarray(reach['xy'], np.float64)
    if len(xy) < 3:
        return dict(reach)
    seg = np.hypot(*np.diff(xy, axis=0).T)
    s = np.r_[0, np.cumsum(seg)]
    st = np.unique(np.r_[np.arange(0, s[-1], DECODER_M), s])
    axis = np.stack([np.interp(st, s, xy[:, 0]), np.interp(st, s, xy[:, 1])], 1)
    tangent = np.gradient(axis, axis=0)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], 1)
    normal /= np.maximum(np.hypot(*normal.T), 1e-12)[:, None]
    k = int(max_shift_m // DECODER_M)
    offs = np.arange(-k, k + 1) * DECODER_M
    px = axis[:, None, 0] + normal[:, None, 0] * offs
    py = axis[:, None, 1] + normal[:, None, 1] * offs
    x0, y0 = float(decoder_bounds[0]), float(decoder_bounds[1])
    z = map_coordinates(np.asarray(decoder_height, np.float64),
                        [(py - y0) / DECODER_M - .5, (px - x0) / DECODER_M - .5], order=1, mode='nearest')
    cost_node = z - z.min(1, keepdims=True)
    pins = np.searchsorted(st, s[[0, -1, *[i for i in reach.get('pins', ()) if 0 < i < len(xy) - 1]]])
    cost_node[pins] = np.inf
    cost_node[pins, k] = 0.0
    step = np.abs(np.arange(2 * k + 1)[:, None] - np.arange(2 * k + 1)[None, :])
    trans = np.where(step <= 1, smooth_m * step, np.inf)
    cost = cost_node[0].copy()
    back = np.zeros(cost_node.shape, np.int64)
    for i in range(1, len(st)):
        total = cost[:, None] + trans
        back[i] = np.argmin(total, 0)
        cost = total[back[i], np.arange(2 * k + 1)] + cost_node[i]
    choice = np.empty(len(st), np.int64)
    choice[-1] = int(np.argmin(cost))
    for i in range(len(st) - 1, 0, -1):
        choice[i - 1] = back[i, choice[i]]
    shift = offs[choice]
    out = dict(reach)
    out['xy'] = axis + normal * shift[:, None]
    out['water_m'] = np.minimum.accumulate(np.interp(st, s, np.asarray(reach['water_m'], np.float64)))
    out['station_m'] = st
    out['pins'] = sorted(set(int(i) for i in pins))
    out['refit_max_shift_m'] = float(np.max(np.abs(shift)))
    return out

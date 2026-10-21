"""Global hydrology authority on one fixed source grid.

``build_hydrology`` turns a final physical height raster into the macro
drainage skeleton that finer levels inherit: open sea versus inland
depressions, a depression forest with spill levels, annual open/closed lake
equilibrium, receivers on the routing surface, physical contributing area and
mean annual discharge, basins, reaches between network nodes and incremental
ownership of every cell. Finer levels query this record (``sample``,
``region``, ``catchment_cells``); they do not route their own basins.

Conventions: map metres with y increasing southward, row 0 at the northern
edge; flat cell index ``row * width + col``; heights and levels in metres;
areas in m^2; discharge in m^3/s (annual mean). ``receivers`` holds a flat
index or -1 for a terminal cell (sea, closed sink, declared plane edge).
``order`` lists every cell with each receiver before its donors; iterate it in
reverse to accumulate downstream. Method note: docs/methods/hydrology-routing.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import time

import numpy as np

import terrain_hydrology_routing as hr

ALGORITHM_VERSION = 'hydro-core-v2'  # v2: tied sills and dry cells follow the cascade
SECONDS_PER_YEAR = 365 * 86400  # the same synthetic climatological year as the monthly balance
DEFAULT_RUNOFF_MM_YEAR = 300.
DEFAULT_EVAPORATION_MM_YEAR = 800.
NODE_KINDS = ('source', 'confluence', 'lake_exit', 'lake', 'mouth', 'sink', 'edge', 'dry_out')
BASIN_KINDS = ('sea', 'closed', 'edge', 'unresolved')


def _mix64(x):
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return x ^ (x >> np.uint64(31))


def stable_keys(kind, seed, *values):
    """Positive 63-bit identities from rounded map metres (splitmix64 chain).

    Keys depend on the record kind, world seed and geographic anchors only,
    never on request order or array position. Combine with ``hydro_id`` for
    cache identity across terrain revisions.
    """
    salt = hashlib.blake2b(f'{ALGORITHM_VERSION}:{kind}:{int(seed)}'.encode(), digest_size=8).digest()
    out = np.full(np.shape(values[0]), int.from_bytes(salt, 'little'), np.uint64)
    with np.errstate(over='ignore'):
        for v in values:
            out = _mix64(out ^ np.round(np.asarray(v, np.float64)).astype(np.int64).astype(np.uint64))
    return (out & np.uint64(2**63 - 1)).astype(np.int64)


def _describe(value, default):
    if value is None:
        return float(default)
    return 'array' if np.ndim(value) else float(value)


def _cell_field(value, default, shape, name):
    if value is None:
        value = default
    arr = np.asarray(value, np.float64)
    if arr.ndim == 0:
        arr = np.full(shape, float(arr))
    if arr.shape != shape:
        raise ValueError(f'{name} must be a scalar or have shape {shape}')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} must be finite')
    negative = int(np.count_nonzero(arr < 0))
    return np.maximum(arr, 0).ravel(), negative


@dataclass(eq=False, repr=False)
class HydroWorld:
    height_m: np.ndarray
    filled_m: np.ndarray
    receivers: np.ndarray
    order: np.ndarray
    area_m2: np.ndarray
    discharge_m3s: np.ndarray
    basin_ids: np.ndarray
    lake_ids: np.ndarray
    ocean: np.ndarray
    reaches: list
    lakes: list
    bounds: tuple
    metadata: dict
    row_area_m2: np.ndarray = None
    reach_ids: np.ndarray = None
    owner_reach: np.ndarray = None
    owner_lake: np.ndarray = None
    terminal_kind: np.ndarray = None
    step_m: np.ndarray = None
    nodes: list = field(default_factory=list)
    basins: dict = field(default_factory=dict)
    hierarchy: dict = field(default_factory=dict)
    _donor_offsets: np.ndarray = None
    _donors: np.ndarray = None
    _owner_index: dict = None

    def __repr__(self):
        h, w = self.shape
        return (f'HydroWorld({h}x{w} {self.topology}, {len(self.reaches)} reaches, '
                f'{len(self.lakes)} lakes, id={self.metadata["hydro_id"][:12]})')

    @property
    def shape(self):
        return self.height_m.shape

    @property
    def topology(self):
        return self.metadata['topology']

    @property
    def cell_area_m2(self):
        """Physical cell areas (read-only broadcast of the per-row areas)."""
        return np.broadcast_to(self.row_area_m2[:, None], self.shape)

    def _spacing(self):
        h, w = self.shape
        x0, y0, x1, y1 = self.bounds
        return (x1 - x0) / w, (y1 - y0) / h

    def cell_xy(self, cells):
        """Cell centres, shape (N, 2), geographic map metres."""
        cells = np.asarray(cells, np.int64)
        h, w = self.shape
        dx, dy = self._spacing()
        rows, cols = np.divmod(cells, w)
        return np.stack((self.bounds[0] + (cols + .5) * dx, self.bounds[1] + (rows + .5) * dy), -1)

    def cell_at(self, x, y):
        """Flat source cell containing each point; longitude wraps on a sphere."""
        return _cell_at(self.shape, self.bounds, self.topology, x, y)

    def sample(self, x, y):
        """Authoritative parent record for fine samples at (x, y)."""
        cells = self.cell_at(x, y)
        return {'cell': cells,
                'receiver': self.receivers.ravel()[cells],
                'basin_id': self.basin_ids.ravel()[cells],
                'owner_reach': self.owner_reach.ravel()[cells],
                'owner_lake': self.owner_lake.ravel()[cells],
                'lake_id': self.lake_ids.ravel()[cells],
                'reach_id': self.reach_ids.ravel()[cells],
                'ocean': self.ocean.ravel()[cells],
                'filled_m': self.filled_m.ravel()[cells],
                'area_m2': self.area_m2.ravel()[cells],
                'discharge_m3s': self.discharge_m3s.ravel()[cells]}

    def link_points(self, a, b):
        """Points between neighbouring cells a and b, (N, 2); the pole for cross-pole links."""
        a, b = np.asarray(a, np.int64), np.asarray(b, np.int64)
        h, w = self.shape
        pa, pb = self.cell_xy(a), self.cell_xy(b)
        if self.topology != 'sphere':
            return .5 * (pa + pb)
        width = self.bounds[2] - self.bounds[0]
        pb[:, 0] += width * np.round((pa[:, 0] - pb[:, 0]) / width)
        mid = .5 * (pa + pb)
        ra, rb = a // w, b // w
        pole = (ra == rb) & ((ra == 0) | (ra == h - 1)) & ~np.isin(np.abs(a % w - b % w), (0, 1, w - 1))
        mid[pole, 0] = pa[pole, 0]
        mid[pole, 1] = np.where(ra[pole] == 0, self.bounds[1], self.bounds[3])
        return mid

    def region(self, bounds):
        """Fixed region contract: inherited labels and every crossing flow.

        Returns cell indices (rows x cols), the authoritative per-cell labels,
        all reaches/lakes/basins present and ``portals`` for every receiver
        edge crossing the region boundary (``direction`` +1 leaves, -1 enters),
        with the parent area/discharge/reach carried across it.
        """
        h, w = self.shape
        dx, dy = self._spacing()
        rx0, ry0, rx1, ry1 = (float(v) for v in bounds)
        j0 = math.floor((rx0 - self.bounds[0]) / dx)
        j1 = math.ceil((rx1 - self.bounds[0]) / dx) - 1
        i0 = max(0, math.floor((ry0 - self.bounds[1]) / dy))
        i1 = min(h - 1, math.ceil((ry1 - self.bounds[1]) / dy) - 1)
        if self.topology == 'sphere':
            cols = np.arange(j0, j0 + min(w, j1 - j0 + 1)) % w
        else:
            cols = np.arange(max(0, j0), min(w - 1, j1) + 1)
        rows = np.arange(i0, i1 + 1)
        cells = rows[:, None] * w + cols[None, :]
        flat = cells.ravel()
        inside = np.zeros(h * w, bool)
        inside[flat] = True
        rcv = self.receivers.ravel()
        out_from = flat[(rcv[flat] >= 0) & ~inside[np.maximum(rcv[flat], 0)]]
        lo, hi = self._donor_offsets[flat], self._donor_offsets[flat + 1]
        counts = hi - lo
        donor_pos = np.repeat(lo - np.cumsum(counts) + counts, counts) + np.arange(counts.sum())
        donors = self._donors[donor_pos]
        in_from = donors[~inside[donors]]
        src = np.concatenate((out_from, in_from))
        portals = {'from_cell': src, 'to_cell': rcv[src],
                   'direction': np.concatenate((np.ones(out_from.size, np.int8),
                                                -np.ones(in_from.size, np.int8))),
                   'xy': self.link_points(src, rcv[src]),
                   'area_m2': self.area_m2.ravel()[src],
                   'discharge_m3s': self.discharge_m3s.ravel()[src],
                   'reach_id': self.owner_reach.ravel()[src],
                   'lake_id': self.owner_lake.ravel()[src],
                   'basin_id': self.basin_ids.ravel()[src]}
        take = lambda a: a.ravel()[cells]
        return {'hydro_id': self.metadata['hydro_id'], 'bounds': tuple(bounds),
                'cells': cells, 'receivers': take(self.receivers), 'basin_ids': take(self.basin_ids),
                'owner_reach': take(self.owner_reach), 'owner_lake': take(self.owner_lake),
                'lake_ids': take(self.lake_ids), 'reach_ids': take(self.reach_ids),
                'filled_m': take(self.filled_m), 'area_m2': take(self.area_m2),
                'discharge_m3s': take(self.discharge_m3s), 'ocean': take(self.ocean),
                'reach_list': np.unique(take(self.reach_ids)[take(self.reach_ids) >= 0]),
                'lake_list': np.unique(take(self.lake_ids)[take(self.lake_ids) >= 0]),
                'basin_list': np.unique(take(self.basin_ids)[take(self.basin_ids) >= 0]),
                'portals': portals}

    def _owners(self, kind):
        if self._owner_index is None:
            self._owner_index = {}
        if kind not in self._owner_index:
            owner = (self.owner_reach if kind == 'reach' else self.owner_lake).ravel()
            idx = np.argsort(owner, kind='stable')
            bounds = np.searchsorted(owner[idx], np.arange(owner.max() + 2), 'left')
            self._owner_index[kind] = (idx, bounds)
        return self._owner_index[kind]

    def catchment_cells(self, reach_id):
        """Incremental catchment of a reach: cells that first join the network there."""
        idx, b = self._owners('reach')
        if reach_id < 0 or reach_id + 1 >= b.size:
            return np.zeros(0, np.int64)
        return idx[b[reach_id]:b[reach_id + 1]].astype(np.int64)

    def lake_catchment_cells(self, lake_id):
        """Lake cells plus cells that drain directly into the lake."""
        idx, b = self._owners('lake')
        if lake_id < 0 or lake_id + 1 >= b.size:
            return np.zeros(0, np.int64)
        return idx[b[lake_id]:b[lake_id + 1]].astype(np.int64)

    def corridor_cells(self, reach_id, radius_cells=2):
        """Inherited corridor of a reach: cells within ``radius_cells`` steps of
        its course, restricted to its parent basin (finer levels may not leave it).
        Returns (cells, steps)."""
        reach = self.reaches[reach_id]
        h, w = self.shape
        allowed = (self.basin_ids.ravel() == reach['basin_id']) & ~self.ocean.ravel()
        return hr.dilate(reach['cells'].astype(np.int64), int(radius_cells), allowed, h, w,
                         self.topology == 'sphere')

    def upstream_cells(self, cells, limit=None):
        seeds = np.atleast_1d(np.asarray(cells, np.int64))
        out, complete = hr.collect_upstream(self._donor_offsets, self._donors, seeds,
                                            int(limit or self.receivers.size))
        if not complete:
            raise ValueError('upstream query exceeded its cell limit')
        return out

    def downstream_path(self, cell):
        rcv = self.receivers.ravel()
        path = [int(cell)]
        while rcv[path[-1]] >= 0:
            path.append(int(rcv[path[-1]]))
        return np.array(path, np.int64)


def _choose_ocean(z, shape, area, sphere, sea_level_m, ocean_mask, ocean_seeds,
                  min_ocean_area_m2, bounds, topology):
    h, w = shape
    if ocean_mask is not None:
        mask = np.asarray(ocean_mask, bool)
        if mask.shape != shape:
            raise ValueError(f'ocean_mask must have shape {shape}')
        return mask.ravel().copy(), {'rule': 'explicit_mask'}
    below = z < sea_level_m
    labels, count = hr.label_components(below, h, w, sphere)
    if not count:
        return np.zeros(z.size, bool), {'rule': 'none_below_sea_level', 'components': 0}
    comp_area = np.bincount(labels[below], area[below], count)
    if ocean_seeds is not None:
        seeds = np.atleast_2d(np.asarray(ocean_seeds, np.float64))
        probe = _cell_at(shape, bounds, topology, seeds[:, 0], seeds[:, 1])
        chosen = np.unique(labels[probe][labels[probe] >= 0])
        rule = {'rule': 'seeded_components', 'seeds': int(len(seeds))}
    else:
        threshold = (min(1e12, .02 * float(area.sum())) if min_ocean_area_m2 is None
                     else float(min_ocean_area_m2))
        chosen = np.flatnonzero(comp_area >= threshold)
        rule = {'rule': 'component_area', 'min_area_m2': threshold}
    keep = np.zeros(count, bool)
    keep[chosen] = True
    rule.update(components=int(count), sea_components=int(keep.sum()),
                inland_below_sea_components=int(count - keep.sum()))
    return below & keep[np.maximum(labels, 0)], rule


def _cell_at(shape, bounds, topology, x, y):
    h, w = shape
    dx, dy = (bounds[2] - bounds[0]) / w, (bounds[3] - bounds[1]) / h
    cols = np.floor((np.asarray(x, np.float64) - bounds[0]) / dx).astype(np.int64)
    rows = np.floor((np.asarray(y, np.float64) - bounds[1]) / dy).astype(np.int64)
    cols = cols % w if topology == 'sphere' else np.clip(cols, 0, w - 1)
    return np.clip(rows, 0, h - 1) * w + cols


def build_hydrology(height_m, *, bounds, seed=0, topology='sphere', runoff_mm_year=None,
                    evaporation_mm_year=None, precipitation_mm_year=None, ocean_mask=None,
                    ocean_seeds=None, sea_level_m=0., min_ocean_area_m2=None,
                    plane_edges='outlet', channel_min_area_m2=2e9,
                    channel_min_discharge_m3s=1., min_lake_depth_m=2., min_lake_area_m2=0.):
    """Build the authoritative macro hydrology of one final source raster.

    ``runoff_mm_year`` is net land yield (precipitation minus land
    evapotranspiration) and ``evaporation_mm_year`` open-water evaporation;
    both accept scalars or full arrays (defaults 300 and 800 mm/year). If
    ``precipitation_mm_year`` is given, lake surfaces receive it instead of the
    runoff yield: a lake cell nets P - E. This needs E >= P - R on every cell
    (true when R = P - AET with AET <= E); where E < P - R the net loss is
    clamped to zero and the lake keeps the land yield R. The sea is the below-``sea_level_m`` components chosen by
    ``ocean_mask``, ``ocean_seeds`` (map points) or ``min_ocean_area_m2``
    (default: min(1e12 m^2, 2 % of the domain)); other depressions are lakes
    or dry closed basins. ``plane_edges`` is 'outlet' or 'closed' and is
    ignored on a sphere. ``seed`` enters identities only: there are no draws.
    """
    timings = {}
    t0 = time.perf_counter()
    z2 = np.asarray(height_m)
    if z2.ndim != 2:
        raise ValueError('height_m must be a 2-D raster')
    if not np.all(np.isfinite(z2)):
        raise ValueError('height_m must be finite')
    shape = z2.shape
    h, w = shape
    bounds = tuple(float(v) for v in bounds)
    if topology not in ('sphere', 'plane'):
        raise ValueError("topology must be 'sphere' or 'plane'")
    if plane_edges not in ('outlet', 'closed'):
        raise ValueError("plane_edges must be 'outlet' or 'closed'")
    sphere = topology == 'sphere'
    edge_outlet = (not sphere) and plane_edges == 'outlet'
    z = z2.astype(np.float64).ravel()
    row_area, dist, radius = hr.grid_geometry(shape, bounds, topology)
    area = np.repeat(row_area, w)
    runoff, neg_r = _cell_field(runoff_mm_year, DEFAULT_RUNOFF_MM_YEAR, shape, 'runoff_mm_year')
    evap, neg_e = _cell_field(evaporation_mm_year, DEFAULT_EVAPORATION_MM_YEAR, shape,
                              'evaporation_mm_year')
    precip = None
    if precipitation_mm_year is not None:
        precip, _ = _cell_field(precipitation_mm_year, 0., shape, 'precipitation_mm_year')
    ocean, ocean_rule = _choose_ocean(z, shape, area, sphere, float(sea_level_m), ocean_mask,
                                      ocean_seeds, min_ocean_area_m2, bounds, topology)
    params = {'topology': topology, 'plane_edges': plane_edges if not sphere else None,
              'sea_level_m': float(sea_level_m), 'min_ocean_area_m2': min_ocean_area_m2,
              'channel_min_area_m2': float(channel_min_area_m2),
              'channel_min_discharge_m3s': float(channel_min_discharge_m3s),
              'min_lake_depth_m': float(min_lake_depth_m), 'min_lake_area_m2': float(min_lake_area_m2),
              'seed': int(seed)}
    digest = hashlib.blake2b(digest_size=16)
    digest.update(json.dumps({'version': ALGORITHM_VERSION, 'shape': shape, 'bounds': bounds,
                              'params': params, 'ocean_rule': ocean_rule}, sort_keys=True,
                             default=str).encode())
    for arr in (z, runoff, evap, ocean.view(np.uint8)) + ((precip,) if precip is not None else ()):
        digest.update(np.ascontiguousarray(arr).tobytes())
    hydro_id = digest.hexdigest()
    timings['inputs'] = time.perf_counter() - t0

    t = time.perf_counter()
    land = np.flatnonzero(~ocean)
    sweep_order = land[np.argsort(z[land], kind='stable')]
    (label, down, attach0, nn, parent, child0, child1, spill, base, to_outlet, pass_cell,
     inside_cell, outside_cell, exit_to, odep, pit_cell, ev_node, ev_kind) = hr.depression_sweep(
        z, sweep_order, ocean, h, w, sphere, edge_outlet, dist, float(sea_level_m))
    del sweep_order
    runoff_vol = np.where(ocean, 0., runoff / 1000. * area)
    rate = evap if precip is None else np.maximum(evap - (precip - runoff), 0.)
    loss_vol = rate / 1000. * area
    lake_precip = precip is not None
    del runoff, evap, rate, precip
    attach, gross0, loss, flood_area_spill, volume_spill, lowpit = hr.depression_budget(
        z, label, attach0, ocean, runoff_vol, loss_vol, area, nn, parent, child0, child1,
        spill, pit_cell)
    del attach0
    gross, full, fsm_to_sea = hr.fill_spill_cascade(nn, parent, child0, child1, odep, gross0, loss,
                                                    ev_node, ev_kind)
    cand = np.flatnonzero(attach > 0)
    cand = cand[np.lexsort((z[cand], attach[cand]))]
    keys = attach[cand]
    nodes_range = np.arange(nn)
    group_start = np.searchsorted(keys, nodes_range, 'left').astype(np.int64)
    group_end = np.searchsorted(keys, nodes_range, 'right').astype(np.int64)
    (final, level, flood, wl, connector, sink, ncell, farea, fvol, zmin, evap_vol,
     residual_vol) = hr.final_lakes(z, loss_vol, area, attach, cand, group_start, group_end, nn,
                                    parent, child0, child1, spill, base, full, gross, loss, lowpit,
                                    pass_cell, inside_cell, outside_cell)
    del cand, keys, group_start, group_end
    timings['depressions'] = time.perf_counter() - t

    t = time.perf_counter()
    finals = np.flatnonzero(final)
    spilling = finals[full[finals]]
    # Several lakes can share one pass cell; the lake that contains it pins it.
    spilling = spilling[np.argsort(inside_cell[spilling] != pass_cell[spilling], kind='stable')]
    seed_cell, seed_rcv = hr.lake_seeds(finals, full, ncell, pass_cell, sink, flood, z, h, w, sphere)
    filled, rcv, term, step, unresolved = hr.route_cells(
        z, ocean, flood, wl, down, seed_cell, seed_rcv,
        pass_cell[spilling], exit_to[spilling], h, w, sphere, edge_outlet, dist,
        float(sea_level_m))
    del down, wl
    order, offsets, donors = hr.drainage_order(rcv)
    if order.size != z.size:
        raise RuntimeError(f'receiver graph has a cycle ({z.size - order.size} cells unreachable)')
    rcv, order, donors, offsets, step = (rcv.astype(np.int32), order.astype(np.int32),
                                         donors.astype(np.int32), offsets.astype(np.int32),
                                         step.astype(np.float32))
    timings['routing'] = time.perf_counter() - t

    t = time.perf_counter()
    flooded = flood >= 0
    evaporating = flooded & ~connector
    yield_vol = runoff_vol - np.where(evaporating, loss_vol, 0.)
    signed = hr.accumulate(order, rcv, yield_vol)
    acc_area = hr.accumulate(order, rcv, np.where(ocean, 0., area))
    discharge = (np.maximum(signed, 0.) / SECONDS_PER_YEAR).astype(np.float32)
    budget = {'runoff_m3s': float(runoff_vol.sum()) / SECONDS_PER_YEAR,
              'open_water_evaporation_m3s': float(loss_vol[evaporating].sum()) / SECONDS_PER_YEAR}
    del yield_vol, runoff_vol, loss_vol, area, evaporating

    # Lakes: significant final flooded components, largest first.
    depth = level[finals] - zmin[finals]
    sig = finals[(ncell[finals] > 0) & (depth >= min_lake_depth_m) & (farea[finals] >= min_lake_area_m2)]
    sig = sig[np.lexsort((sink[sig], -farea[sig]))]
    lake_of_node = np.full(nn, -1, np.int32)
    lake_of_node[sig] = np.arange(sig.size, dtype=np.int32)
    lake_ids = np.where(flooded, lake_of_node[np.maximum(flood, 0)], -1).astype(np.int32)

    roots = hr.basin_roots(order, rcv, ocean)
    land_roots = np.unique(roots[roots >= 0])
    land_roots = land_roots[np.lexsort((land_roots, -acc_area[land_roots]))]
    basin_of_root = np.full(z.size, -1, np.int32)
    basin_of_root[land_roots] = np.arange(land_roots.size, dtype=np.int32)
    basin_ids = np.where(roots >= 0, basin_of_root[np.maximum(roots, 0)], -1).astype(np.int32)
    timings['accumulation'] = time.perf_counter() - t

    t = time.perf_counter()
    is_chan = (~ocean & (lake_ids < 0) & (acc_area >= channel_min_area_m2)
               & (discharge >= channel_min_discharge_m3s))
    acc_area = acc_area.astype(np.float32)
    start, rcells, end_kind, end_next, reach_of, up, lake_up = hr.extract_reaches(
        rcv, is_chan, lake_ids, ocean)
    owner_reach, owner_lake = hr.propagate_owners(order, rcv, ocean, reach_of, lake_ids)

    world = HydroWorld(
        height_m=z2, filled_m=filled.reshape(shape), receivers=rcv.reshape(shape), order=order,
        area_m2=acc_area.reshape(shape), discharge_m3s=discharge.reshape(shape),
        basin_ids=basin_ids.reshape(shape), lake_ids=lake_ids.reshape(shape),
        ocean=ocean.reshape(shape), reaches=[], lakes=[], bounds=bounds, metadata={},
        row_area_m2=row_area, reach_ids=reach_of.reshape(shape),
        owner_reach=owner_reach.reshape(shape), owner_lake=owner_lake.reshape(shape),
        terminal_kind=term.reshape(shape), step_m=step.reshape(shape),
        _donor_offsets=offsets, _donors=donors)
    world.metadata = {'topology': topology, 'hydro_id': hydro_id}

    lakes = _lake_records(world, sig, level, full, spill, pass_cell, sink, farea, fvol, zmin,
                          gross, evap_vol, residual_vol, signed, seed)
    reaches, nodes = _reach_records(world, start, rcells, end_kind, end_next, up, lake_up,
                                    lakes, seed, float(sea_level_m))
    world.reaches, world.lakes, world.nodes = reaches, lakes, nodes
    kinds = np.full(land_roots.size, 0, np.int8)
    kinds[term[land_roots] == hr.TERM_SINK] = 1
    kinds[term[land_roots] == hr.TERM_EDGE] = 2
    kinds[term[land_roots] == hr.TERM_UNRESOLVED] = 3
    root_xy = world.cell_xy(land_roots)
    world.basins = {'root_cell': land_roots, 'kind': kinds, 'kind_names': BASIN_KINDS,
                    'area_m2': acc_area[land_roots], 'outflow_m3s': signed[land_roots] / SECONDS_PER_YEAR,
                    'lake_id': lake_ids[land_roots],
                    'key': stable_keys('basin', seed, root_xy[:, 0], root_xy[:, 1])}
    world.hierarchy = {'parent': parent, 'child0': child0, 'child1': child1, 'spill_m': spill,
                       'base_m': base, 'to_outlet': to_outlet, 'pass_cell': pass_cell,
                       'inside_cell': inside_cell, 'outside_cell': outside_cell,
                       'exit_to': exit_to,
                       'pit_cell': pit_cell, 'full': full, 'final': final,
                       'level_m': level, 'inflow_m3s': gross / SECONDS_PER_YEAR,
                       'loss_at_spill_m3s': loss / SECONDS_PER_YEAR,
                       'area_at_spill_m2': flood_area_spill, 'volume_at_spill_m3': volume_spill,
                       'lake_id': lake_of_node, 'flood_node': flood.reshape(shape),
                       'leaf': label.reshape(shape)}
    timings['records'] = time.perf_counter() - t

    to_sea = float(signed[ocean].sum())
    budget.update({'to_sea_m3s': to_sea / SECONDS_PER_YEAR,
              'to_plane_edge_m3s': float(signed[term == hr.TERM_EDGE].sum()) / SECONDS_PER_YEAR,
              'closed_sink_residual_m3s': float(signed[term == hr.TERM_SINK].sum()) / SECONDS_PER_YEAR,
              'unresolved_m3s': float(signed[term == hr.TERM_UNRESOLVED].sum()) / SECONDS_PER_YEAR})
    budget['conservation_error_m3s'] = (budget['runoff_m3s'] - budget['open_water_evaporation_m3s']
                                        - budget['to_sea_m3s'] - budget['to_plane_edge_m3s']
                                        - budget['closed_sink_residual_m3s'] - budget['unresolved_m3s'])
    closed = [lk for lk in lakes if lk['closed']]
    world.metadata = {
        'algorithm': ALGORITHM_VERSION, 'hydro_id': hydro_id, 'topology': topology,
        'shape': shape, 'bounds': bounds, 'planet_radius_m': radius,
        'parameters': params, 'ocean': ocean_rule,
        'inputs': {'runoff': _describe(runoff_mm_year, DEFAULT_RUNOFF_MM_YEAR),
                   'evaporation': _describe(evaporation_mm_year, DEFAULT_EVAPORATION_MM_YEAR),
                   'precipitation_on_lakes': lake_precip,
                   'negative_runoff_cells_clamped': neg_r, 'negative_evaporation_cells_clamped': neg_e},
        'counts': {'cells': int(z.size), 'sea_cells': int(ocean.sum()),
                   'depression_nodes': int(nn - 1), 'pits': int(np.count_nonzero(child0[1:] < 0)),
                   'final_water_bodies': int(finals.size), 'lakes': len(lakes),
                   'closed_lakes': len(closed), 'reaches': len(reaches), 'nodes': len(nodes),
                   'basins': int(land_roots.size), 'unresolved_flat_cells': int(unresolved)},
        'budget': budget,
        'closed_lake_routing_mismatch_max_rel': max(
            (abs(lk['routed_inflow_m3s'] - lk['inflow_m3s']) / max(lk['inflow_m3s'], 1e-12)
             for lk in closed), default=0.),
        'conventions': {'order': 'receivers before donors (Braun-Willett stack)',
                        'receivers': 'flat index, -1 terminal', 'y': 'south', 'units': 'm, m2, m3/s'},
        'timings_s': timings}
    timings['total'] = time.perf_counter() - t0
    return world


def _lake_records(world, sig, level, full, spill, pass_cell, sink, farea, fvol, zmin,
                  gross, evap_vol, residual_vol, signed, seed):
    lake_ids = world.lake_ids.ravel()
    n = sig.size
    idx = np.flatnonzero(lake_ids >= 0)
    idx = idx[np.argsort(lake_ids[idx], kind='stable')]
    cuts = np.searchsorted(lake_ids[idx], np.arange(n + 1), 'left')
    width = world.bounds[2] - world.bounds[0]
    xy = world.cell_xy(idx)
    lid = lake_ids[idx]
    count = np.maximum(np.diff(cuts), 1)
    cy = np.bincount(lid, xy[:, 1], n) / count
    if world.topology == 'sphere':
        ang = (xy[:, 0] - world.bounds[0]) / width * 2 * np.pi
        mean = np.arctan2(np.bincount(lid, np.sin(ang), n), np.bincount(lid, np.cos(ang), n))
        cx = world.bounds[0] + (mean % (2 * np.pi)) / (2 * np.pi) * width
    else:
        cx = np.bincount(lid, xy[:, 0], n) / count
    closed = ~full[sig]
    anchor_cell = np.where(closed, sink[sig], pass_cell[sig])
    anchor = world.cell_xy(anchor_cell)
    keys = stable_keys('lake', seed, anchor[:, 0], anchor[:, 1])
    inflow = gross[sig] / SECONDS_PER_YEAR
    evap = evap_vol[sig] / SECONDS_PER_YEAR
    routed = signed[anchor_cell] / SECONDS_PER_YEAR + evap
    basin = world.basin_ids.ravel()[anchor_cell]
    out = []
    for lake in range(n):
        d = sig[lake]
        c = bool(closed[lake])
        out.append({'id': lake, 'key': int(keys[lake]),
                    'level_m': float(level[d]), 'closed': c, 'area_m2': float(farea[d]),
                    'cells': idx[cuts[lake]:cuts[lake + 1]], 'volume_m3': float(fvol[d]),
                    'max_depth_m': float(level[d] - zmin[d]),
                    'spill_level_m': float(spill[d]) if np.isfinite(spill[d]) else None,
                    'outlet_cell': -1 if c else int(pass_cell[d]),
                    'sink_cell': int(sink[d]) if c else -1,
                    'xy': np.array((cx[lake], cy[lake])),
                    'inflow_m3s': float(inflow[lake]), 'evaporation_m3s': float(evap[lake]),
                    'outflow_m3s': 0. if c else float(inflow[lake] - evap[lake]),
                    'residual_m3s': float(residual_vol[d]) / SECONDS_PER_YEAR if c else 0.,
                    'routed_inflow_m3s': float(routed[lake]) if c else None,
                    'basin_id': int(basin[lake]), 'depression_node': int(d), 'node_id': None,
                    'entry_reaches': [], 'exit_reach': None})
    return out


def _reach_records(world, start, rcells, end_kind, end_next, up, lake_up, lakes, seed, sea_level):
    """Reach and node records, built with array operations over all reaches."""
    h, w = world.shape
    filled = world.filled_m.ravel()
    step = world.step_m.ravel()
    term = world.terminal_kind.ravel()
    lake_ids = world.lake_ids.ravel()
    basin = world.basin_ids.ravel()
    reach_of = world.reach_ids.ravel()
    width = world.bounds[2] - world.bounds[0]
    sphere = world.topology == 'sphere'
    offsets, donors = world._donor_offsets, world._donors
    nreach = start.size - 1
    level = np.array([lk['level_m'] for lk in lakes] + [np.nan])
    cnt = np.diff(start)
    heads = rcells[start[:-1]]
    lasts = rcells[start[1:] - 1]
    ek, nxt = end_kind.astype(np.int64), end_next
    has_pre = lake_up[heads] >= 0
    has_suf = (ek == hr.END_JOIN) | (ek == hr.END_LAKE) | (ek == hr.END_MOUTH)
    npts = cnt + has_pre + has_suf
    pstart = np.concatenate(([0], np.cumsum(npts)))
    xy = np.empty((int(pstart[-1]), 2))
    wat = np.empty(int(pstart[-1]))
    pos = np.repeat(pstart[:-1] + has_pre - start[:-1], cnt) + np.arange(rcells.size)
    xy[pos] = world.cell_xy(rcells)
    wat[pos] = filled[rcells]
    length = (np.add.reduceat(step[rcells], start[:-1]) - step[lasts]) if nreach else np.zeros(0)
    # Lake exits start on the shore between the lake drain cell and the sill.
    pre = np.flatnonzero(has_pre)
    drains = np.empty(pre.size, np.int64)
    for i, r in enumerate(pre):
        dn = donors[offsets[heads[r]]:offsets[heads[r] + 1]]
        drains[i] = dn[lake_ids[dn] == lake_up[heads[r]]][0]
    if pre.size:
        xy[pstart[pre]] = world.link_points(heads[pre], drains)
        wat[pstart[pre]] = level[lake_up[heads[pre]]]
        length[pre] += .5 * step[drains]
    # Ends: shared confluence vertex, lake shore or coast.
    for kind, part in ((hr.END_JOIN, 1.), (hr.END_LAKE, .5), (hr.END_MOUTH, .5)):
        sel = np.flatnonzero(ek == kind)
        if not sel.size:
            continue
        at = pstart[sel + 1] - 1
        if kind == hr.END_JOIN:
            xy[at] = world.cell_xy(nxt[sel])
            wat[at] = filled[nxt[sel]]
        else:
            xy[at] = world.link_points(lasts[sel], nxt[sel])
            wat[at] = (level[lake_ids[nxt[sel]]] if kind == hr.END_LAKE
                       else np.minimum(sea_level, filled[lasts[sel]]))
        length[sel] += part * step[lasts[sel]]
    if sphere and xy.size:
        jumps = np.round(np.diff(xy[:, 0]) / width)
        jumps[pstart[1:-1] - 1] = 0
        cum = np.concatenate(([0.], np.cumsum(jumps)))
        xy[:, 0] -= (cum - np.repeat(cum[pstart[:-1]], npts)) * width
    # Network nodes: reach heads and terminal ends sorted by (cell, role), then lakes.
    up_h = up[heads]
    head_kind = np.where(has_pre, np.where(up_h >= 1, 1, 2), np.where(up_h == 0, 0, 1))
    end_sel = np.flatnonzero((ek == hr.END_MOUTH) | (ek == hr.END_TERMINAL) | (ek == hr.END_FADE))
    end_code = np.where(ek[end_sel] == hr.END_MOUTH, 4,
                        np.where(ek[end_sel] == hr.END_FADE, 7,
                                 np.where(term[lasts[end_sel]] == hr.TERM_EDGE, 6, 5)))
    n_cells = np.concatenate((heads, lasts[end_sel]))
    n_role = np.concatenate((np.zeros(nreach, np.int8), np.ones(end_sel.size, np.int8)))
    n_kind = np.concatenate((head_kind, end_code)).astype(np.int64)
    n_xy = np.concatenate((world.cell_xy(heads), xy[pstart[end_sel + 1] - 1]))
    n_lake = np.concatenate((np.where(has_pre, lake_up[heads], -1), np.full(end_sel.size, -1)))
    perm = np.lexsort((n_role, n_cells))
    rank = np.empty(perm.size, np.int64)
    rank[perm] = np.arange(perm.size)
    head_node = rank[:nreach]
    target = np.full(nreach, -1, np.int64)
    target[end_sel] = rank[nreach:]
    downstream = np.full(nreach, -1, np.int64)
    joins = np.flatnonzero(ek == hr.END_JOIN)
    downstream[joins] = reach_of[nxt[joins]]
    target[joins] = head_node[downstream[joins]]
    lake_base = perm.size
    entering = np.flatnonzero(ek == hr.END_LAKE)
    entered = lake_ids[nxt[entering]]
    target[entering] = lake_base + entered
    for lake in lakes:
        lake['node_id'] = lake_base + lake['id']
    for r in pre:
        lakes[int(lake_up[heads[r]])]['exit_reach'] = int(r)
    for r, lake in zip(entering, entered):
        lakes[int(lake)]['entry_reaches'].append(int(r))
        ex = lakes[int(lake)]['exit_reach']
        downstream[r] = -1 if ex is None else ex
    node_keys = stable_keys('node', seed, n_kind, n_xy[:, 0], n_xy[:, 1])
    nodes = []
    for i in perm:
        nodes.append({'id': len(nodes), 'kind': NODE_KINDS[n_kind[i]], 'cell': int(n_cells[i]),
                      'xy': n_xy[i].copy(), 'lake_id': int(n_lake[i]) if n_lake[i] >= 0 else None,
                      'basin_id': int(basin[n_cells[i]]), 'key': int(node_keys[i])})
    for lake in lakes:
        nodes.append({'id': lake['node_id'], 'kind': 'lake',
                      'cell': lake['sink_cell'] if lake['closed'] else lake['outlet_cell'],
                      'xy': lake['xy'], 'lake_id': lake['id'], 'basin_id': lake['basin_id'],
                      'key': lake['key']})
    first, final = xy[pstart[:-1]], xy[pstart[1:] - 1]
    reach_keys = stable_keys('reach', seed, first[:, 0], first[:, 1], final[:, 0], final[:, 1])
    flow = world.discharge_m3s.ravel()[lasts]
    area = world.area_m2.ravel()[lasts]
    ends = ('join', 'lake', 'mouth', 'terminal', 'dry_out')
    reaches = []
    for r in range(nreach):
        a, b = pstart[r], pstart[r + 1]
        ln = float(length[r])
        reaches.append({'id': r, 'key': int(reach_keys[r]), 'source': int(head_node[r]),
                        'target': int(target[r]), 'basin_id': int(basin[heads[r]]),
                        'xy': xy[a:b], 'water_m': wat[a:b], 'flow_m3s': float(flow[r]),
                        'area_m2': float(area[r]),
                        'downstream_id': int(downstream[r]) if downstream[r] >= 0 else None,
                        'cells': rcells[start[r]:start[r + 1]], 'length_m': ln,
                        'slope': float(wat[a] - wat[b - 1]) / ln if ln > 0 else 0.,
                        'enters_lake': int(lake_ids[nxt[r]]) if ek[r] == hr.END_LAKE else None,
                        'leaves_lake': int(lake_up[heads[r]]) if has_pre[r] else None,
                        'end': ends[ek[r]], 'crosses_pole': False})
    if sphere and rcells.size > 1:
        rows, cols = np.divmod(rcells, w)
        pole = ((rows[:-1] == rows[1:]) & np.isin(rows[:-1], (0, h - 1))
                & ~np.isin(np.abs(cols[:-1] - cols[1:]), (0, 1, w - 1)))
        pole[start[1:-1] - 1] = False
        for r in np.unique(np.searchsorted(start, np.flatnonzero(pole), 'right') - 1):
            _insert_pole_vertices(world, reaches[r])
    _strahler(reaches)
    return reaches, nodes


def _insert_pole_vertices(world, reach):
    """Replace each cross-pole D8 link by two vertices on the pole line."""
    h, w = world.shape
    width = world.bounds[2] - world.bounds[0]
    y_mid = .5 * (world.bounds[1] + world.bounds[3])
    xy, wat = reach['xy'], reach['water_m']
    pts, ws = [xy[0]], [wat[0]]
    for i in range(1, len(xy)):
        dx = abs(xy[i, 0] - xy[i - 1, 0])
        if abs(xy[i, 1] - xy[i - 1, 1]) < 1e-6 and abs(dx - width / 2) <= 1.01 * width / w:
            y_pole = world.bounds[1] if xy[i, 1] < y_mid else world.bounds[3]
            pts += [np.array((xy[i - 1, 0], y_pole)), np.array((xy[i, 0], y_pole))]
            ws += [wat[i - 1], wat[i]]
        pts.append(xy[i])
        ws.append(wat[i])
    reach['xy'] = np.array(pts)
    reach['water_m'] = np.array(ws)
    reach['crosses_pole'] = True


def _strahler(reaches):
    """Strahler order over the reach graph (Kahn order, ties by reach id)."""
    n = len(reaches)
    upstream = [[] for _ in range(n)]
    for reach in reaches:
        if reach['downstream_id'] is not None:
            upstream[reach['downstream_id']].append(reach['id'])
    pending = [len(u) for u in upstream]
    ready = [r for r in range(n) if not pending[r]][::-1]
    done = 0
    while ready:
        r = ready.pop()
        done += 1
        orders = sorted((reaches[u]['strahler'] for u in upstream[r]), reverse=True)
        if not orders:
            s = 1
        elif len(orders) > 1 and orders[0] == orders[1]:
            s = orders[0] + 1
        else:
            s = orders[0]
        reaches[r]['strahler'] = s
        reaches[r]['upstream_ids'] = upstream[r]
        d = reaches[r]['downstream_id']
        if d is not None:
            pending[d] -= 1
            if not pending[d]:
                ready.append(d)
    if done != n:
        raise RuntimeError(f'reach graph has a cycle ({n - done} reaches)')

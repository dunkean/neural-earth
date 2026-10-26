"""Lazy geographic hydrology delivery; raw neural caches remain independent.

Source acquisition is supplied by the server. CPU routing/balance and disk IO
run on one background worker. Display requests never route an isolated tile.

Default delivery combines the global source network and local vector routing
on already-generated Base, without new neural work. ``detail=True`` adds
physical profiles and carving; ``cosmetics`` are optional (all off by default).
"""
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from pathlib import Path
from dataclasses import fields
import hashlib
import inspect
import json
import math
import os
import tempfile
import threading
import time
import zipfile

import numpy as np

from terrain_paths import RUNTIME_ROOT

VERSION = 'hydrology-source-regions-v1'
YEAR_SECONDS = 365 * 86400.0
# Bound total generated centreline displacement in map metres; width/bank
# support is added separately in physical metres when acquiring neighbours.
REGIONAL_RIVER_SUPPORT_M = 3840.
REGIONAL_VALLEY_CUT_M = 60.  # conflict allowance shared by profiles, routing and decoder
# Decoder relief above its routing parent that a resolved channel may remove
# beyond the parent's share (tile -1871,53: p99 18.8 m, max 29.5 m).
DECODER_RESIDUAL_CUT_M = 30.
# Ready regional packages kept in memory (~7 MB each, 19 MB measured maximum).
# One tile's support needs at most ~30. Packages requested within 30 s may
# overflow the soft size up to the hard ceiling.
REGION_CACHE, REGION_CEILING = 64, 96
# Default vector packages: Base-only regional routing on already generated
# neural Base (cache-only callback). CPU only, bounded, on the shared worker.
VECTOR_CACHE, VECTOR_BUILDS = 32, 3
VECTOR_CORE_M = 4*7680.  # small enough to reuse Base around an initial fine view
# Lake routing remains active; detailed lake rendering is deferred.
DISPLAY_LAKES = False
# Cache identities. The persisted world depends only on routing/balance code
# (and the stored seasonal properties); regional plans add their own modules.
# Geometry and delivery code change the display identity only.
WORLD_FILES = ('terrain_hydrology.py', 'terrain_hydrology_routing.py', 'terrain_hydrology_balance.py')
REGION_FILES = ('terrain_hydrology_regions.py', 'terrain_hydrology_profiles.py', 'terrain_hydrology_sampling.py')
DISPLAY_FILES = ('terrain_hydrology_geometry.py', 'terrain_hydrology_connectors.py', 'terrain_hydrology_lakes.py',
                 'terrain_hydrology_service.py')
# Source reach ``end`` values as display end kinds (local records use their target kind).
SOURCE_END_KINDS = dict(join='confluence', lake='lake', mouth='mouth', terminal='terminal', dry_out='dry_out')
DRY_FLOW_M3S = 0.01  # near-dry month, as terrain_hydrology_balance.flow_statistics
TRUNK_END_KINDS = ('mouth', 'confluence', 'lake', 'lake_exit', 'sink', 'edge', 'dry_out')
STORE_VERSION = 'store-v1'  # bump when _properties or stored plan corrections change


def _json(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _bbox(xy):
    points = np.asarray(xy)
    return np.array([points[:, 0].min(), points[:, 1].min(),
                     points[:, 0].max(), points[:, 1].max()])


def _intersects(a, b):
    return a[0] <= b[2] and a[2] >= b[0] and a[1] <= b[3] and a[3] >= b[1]


def _world_intersects(world, box, query):
    period = world.bounds[2]-world.bounds[0]
    return any(_intersects((box[0]+s, box[1], box[2]+s, box[3]), query)
               for s in ((-period, 0., period) if world.topology == 'sphere' else (0.,)))


def _outside_pieces(xy, water, boxes):
    """Split an overview curve at ready regional cores, keeping its remainder.

    Intersections are inserted before masking, so even a segment crossing a
    whole core without an interior vertex is clipped at the exact boundary.
    This affects delivery only, never the stored physical curve or its seed.
    """
    xy, water = np.asarray(xy), np.asarray(water)
    if not boxes:
        return [(xy, water)]
    stations = np.arange(len(xy), dtype=float)
    cuts = [stations]
    delta = np.diff(xy, axis=0)
    for box in boxes:
        for axis, edge in ((0, box[0]), (0, box[2]), (1, box[1]), (1, box[3])):
            moving = delta[:, axis] != 0
            indices = np.flatnonzero(moving)
            u = (edge-xy[:-1, axis][moving])/delta[moving, axis]
            other = xy[:-1, 1-axis][moving]+u*delta[moving, 1-axis]
            inside = (u > 0)&(u < 1)&(other >= box[1-axis])&(other <= box[3-axis])
            cuts.append(indices[inside]+u[inside])
    t = np.unique(np.concatenate(cuts))
    mid = (t[:-1]+t[1:])*.5
    mx, my = (np.interp(mid, stations, xy[:, a]) for a in (0, 1))
    keep = np.ones(len(mid), bool)
    for box in boxes:
        keep &= ~((mx >= box[0])&(mx <= box[2])&(my >= box[1])&(my <= box[3]))
    starts = np.flatnonzero(keep & ~np.r_[False, keep[:-1]])
    ends = np.flatnonzero(keep & ~np.r_[keep[1:], False])+1
    return [(np.column_stack([np.interp(t[a:b+1], stations, xy[:, k]) for k in (0, 1)]),
             np.interp(t[a:b+1], stations, water)) for a, b in zip(starts, ends)]


def _land_course(xy, water, terrain, bounds):
    """Stop a vector at its first Base sea crossing; retain the exact shore.

    Sampling at half a Base cell also catches water crossed by a long straight
    segment. This clips display geometry, never the routed graph or budgets.
    """
    from terrain_hydrology_geometry import sample_bilinear
    xy, water = np.asarray(xy), np.asarray(water)
    if len(xy) < 2:
        return xy, water
    station = np.r_[0., np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    samples = np.unique(np.r_[station, np.arange(0., station[-1], 120.)])
    points = np.column_stack([np.interp(samples, station, xy[:, axis]) for axis in (0, 1)])
    height = sample_bilinear(terrain, bounds, points[:, 0], points[:, 1])
    sea = np.flatnonzero(height < 0.)
    if not len(sea):
        return xy, water
    k = int(sea[0])
    if k == 0:
        return xy[:0], water[:0]
    fraction = height[k-1]/(height[k-1]-height[k])
    end = samples[k-1]+fraction*(samples[k]-samples[k-1])
    keep = station < end
    shore = np.array([np.interp(end, station, xy[:, axis]) for axis in (0, 1)])
    return np.vstack([xy[keep], shore]), np.r_[water[keep], np.interp(end, station, water)]


def _plain(value):
    """Portable JSON: no typed arrays, unsafe 64-bit keys or NaN tokens."""
    kind = type(value)
    if kind is float:
        return value if math.isfinite(value) else None
    if kind in (int, str, bool) or value is None:
        return value
    if kind is dict:
        return {str(k): str(v) if k == 'key' else _plain(v) for k, v in value.items()}
    if kind in (list, tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        # Vertex arrays dominate the payload: convert numeric arrays at once.
        if value.dtype.kind == 'f':
            finite = np.isfinite(value)
            if finite.all():
                return value.tolist()
            value = value.astype(object)
            value[~finite] = None
            return value.tolist()
        if value.dtype.kind in 'biu':
            return value.tolist()
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, dict):
        return {str(k): str(v) if k == 'key' else _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sample(field, bounds, xy, wrap_x=False):
    h, w = field.shape
    points = np.asarray(xy)
    col = np.floor((points[:, 0]-bounds[0])/(bounds[2]-bounds[0])*w).astype(int)
    col = col % w if wrap_x else np.clip(col, 0, w-1)
    row = np.clip(((points[:, 1]-bounds[1])/(bounds[3]-bounds[1])*h).astype(int), 0, h-1)
    return field[row, col]


class HydrologyService:
    def __init__(self, read_source, *, read_detail=None, read_coarse_points=None, read_cached_base=None,
                 cache_root=None, max_worlds=2):
        # read_cached_base(seed, profile, bounds) -> dict(base_height, base_bounds)
        # from already processed ordinary Base windows only, or None.
        self.read_cached_base = read_cached_base
        self.read_source = read_source
        self.read_detail = read_detail
        self.read_coarse_points = read_coarse_points
        self.root = Path(cache_root or RUNTIME_ROOT / 'hydrology')
        self.max_worlds = max_worlds
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='hydrology')
        self.lock = threading.RLock()
        self.entries = OrderedDict()
        digest = hashlib.sha256('/'.join((VERSION, STORE_VERSION,
                                          os.environ.get('TERRAIN_HYDRO_WIDTH', '2048'))).encode())
        identities = []
        for names in (WORLD_FILES, REGION_FILES, DISPLAY_FILES):
            for name in names:
                path = Path(__file__).with_name(name)
                if path.is_file():
                    digest.update(path.read_bytes())
            identities.append(digest.copy().hexdigest()[:20])
        self.world_implementation, self.region_implementation, self.implementation = identities

    @staticmethod
    def parse_cosmetics(value=()):
        """Enabled cosmetic names from a comma string or iterable, as a frozenset."""
        from terrain_hydrology_geometry import COSMETICS
        names = value.split(',') if isinstance(value, str) else (value or ())
        names = frozenset(n.strip() for n in names if n and n.strip())
        unknown = names - set(COSMETICS)
        if unknown:
            raise ValueError(f'Unknown hydrology cosmetics: {", ".join(sorted(unknown))}')
        return names

    def ensure(self, seed, profile, world_identity):
        key = hashlib.sha256(f'{world_identity}/{self.implementation}'.encode()).hexdigest()
        with self.lock:
            entry = self.entries.get(key)
            if entry is None:
                stored = lambda impl: self.root/hashlib.sha256(f'{world_identity}/{impl}'.encode()).hexdigest()
                entry = dict(key=key, seed=int(seed), profile=profile, status='building',
                             phase='source', started=time.perf_counter(), error=None,
                             world_path=stored(self.world_implementation)/'world.npz',
                             region_dir=stored(self.region_implementation)/'regions',
                             vector_dir=stored(self.region_implementation)/'vector-regions')
                self.entries[key] = entry
                entry['future'] = self.pool.submit(self._build, entry)
            self.entries.move_to_end(key)
            for old in list(self.entries):
                if len(self.entries) <= self.max_worlds:
                    break
                if old != key:
                    previous = self.entries[old]
                    if previous['status'] != 'building' or previous['future'].cancel():
                        del self.entries[old]
            return entry

    @staticmethod
    def state(entry):
        result = dict(status=entry['status'], hydro_id=entry['key'],
                      phase=entry.get('phase'), seconds=round(time.perf_counter()-entry['started'], 3))
        if entry['status'] == 'error':
            result['error'] = entry['error']
        if entry['status'] == 'ready':
            world = entry['world']
            result.update(world_bounds=list(world.bounds), reaches=len(world.reaches),
                          lakes=len(world.lakes), timings=entry.get('timings', {}),
                          metadata=world.metadata)
        return result

    def _build(self, entry):
        begun = time.perf_counter()
        try:
            source = self.read_source(entry['seed'], entry['profile'])
            entry['source'] = source
            source_seconds = time.perf_counter()-begun
            cache = entry.get('world_path') or self.root/entry['key']/'world.npz'
            world = self._load(cache)
            routing_started = time.perf_counter()
            # A stored world already carries its seasonal properties.
            seasonal = world is None or 'seasonality' not in world.metadata
            if world is None:
                entry['phase'] = 'water balance'
                from terrain_hydrology_balance import annual_water_balance
                climate = source['climate']
                h, w = source['height_m'].shape
                latitude = source.get('latitude_deg', np.zeros((h, 1)))
                water = annual_water_balance(climate[2], climate[0], latitude_deg=latitude,
                                             bio4=climate[1], bio15=climate[3],
                                             **source.get('seasonal', {}))
                entry['phase'] = 'drainage and lakes'
                from terrain_hydrology import build_hydrology
                dx = (source['bounds'][2]-source['bounds'][0])/w
                dy = (source['bounds'][3]-source['bounds'][1])/h
                world = build_hydrology(source['height_m'], bounds=source['bounds'],
                    seed=entry['seed'], topology=source['topology'],
                    runoff_mm_year=water['runoff_mm_year'],
                    evaporation_mm_year=water['potential_evaporation_mm_year'],
                    precipitation_mm_year=water['precipitation_mm_year'],
                    channel_min_area_m2=max(1e7, abs(dx*dy)*8))
                world.metadata.update(climate_balance=water['metadata'],
                    balance_max_residual_mm=float(np.max(np.abs(water['residual_mm']))),
                    balance_converged_fraction=float(np.mean(water['converged'])))
            entry['world'] = world
            entry['geometry'] = OrderedDict()
            entry['geometry_bytes'] = 0
            entry['lake_geometry'] = {}
            entry['catchments'] = OrderedDict()
            entry['regions'] = OrderedDict()
            entry['vector_regions'] = OrderedDict()
            entry['profiles'] = {}
            entry['water_grid'] = None
            from terrain_hydrology_geometry import _node_tangents
            entry['tangents'] = _node_tangents(world.reaches, self.radius(world))
            entry['reach_bounds'] = np.array([_bbox(r['xy']) for r in world.reaches]).reshape(-1, 4)
            entry['height64'] = np.asarray(world.height_m, np.float64)
            entry['river_max_flow_m3s'] = max(0., float(np.max(world.discharge_m3s, initial=0.)),
                max((r['flow_m3s'] for r in world.reaches), default=0.))
            if seasonal:
                entry['phase'] = 'seasonal river properties'
                self._properties(world, source)
            from terrain_hydrology_connectors import display_connectors
            entry['connectors'] = display_connectors(world)
            entry['source_lake_display'] = OrderedDict()
            lake_ids = np.asarray(world.lake_ids).ravel()
            for link in entry['connectors'].values():
                ids = lake_ids[np.asarray(link.get('cells', ()), np.int64)] if link['kind'] == 'lake' else ()
                ids = np.asarray(ids)[np.asarray(ids) >= 0] if len(ids) else ids
                link['lake_id'] = int(np.bincount(ids).argmax()) if len(ids) else None
            for rid, link in entry['connectors'].items():
                # A hidden-lake course may cross the view while both shores
                # lie outside it. Include it in the parent's display extent.
                box = _bbox(link['xy'])
                entry['reach_bounds'][rid, :2] = np.minimum(entry['reach_bounds'][rid, :2], box[:2])
                entry['reach_bounds'][rid, 2:] = np.maximum(entry['reach_bounds'][rid, 2:], box[2:])
            entry['timings'] = dict(source_seconds=round(source_seconds, 3),
                                    routing_and_balance_seconds=round(time.perf_counter()-routing_started, 3))
            entry['phase'] = 'ready'
            entry['status'] = 'ready'
            if not cache.exists():
                try:
                    self._save(cache, world)
                except OSError as exc:
                    entry['cache_warning'] = str(exc)
        except Exception as exc:
            entry.update(status='error', phase='failed', error=f'{type(exc).__name__}: {exc}')

    @staticmethod
    def _properties(world, source):
        """Downscale fixed annual lateral budgets into a synthetic seasonal year.

        Reach forcing is sampled from its incremental unit's representative
        station. No global 12-month cell tensor or visit-dependent rainfall.
        """
        if not world.reaches:
            return
        from terrain_hydrology_balance import (route_monthly,
            channel_hydraulics, navigability, MONTH_SECONDS)
        reaches = world.reaches
        points = np.array([r['xy'][len(r['xy'])//2] for r in reaches])
        wrap = source['topology'] == 'sphere'
        budget = HydrologyService._monthly_budget(world, source, points)
        q = np.array([max(0., r['flow_m3s']) for r in reaches])
        lookup = {r['id']: i for i, r in enumerate(reaches)}
        downstream = np.array([lookup.get(r.get('downstream_id'), -1) for r in reaches])
        # A visually faded channel can join another channel after crossing
        # unrendered cells. Its water still has that hydraulic continuation.
        for i, reach in enumerate(reaches):
            if downstream[i] < 0 and len(reach.get('cells', ())):
                cell = world.receivers.ravel()[np.asarray(reach['cells'])[-1]]
                owner = int(world.owner_reach.ravel()[cell]) if cell >= 0 else -1
                if owner >= 0 and owner != reach['id']:
                    downstream[i] = lookup.get(owner, -1)
                    reach['hydraulic_downstream_id'] = owner
        incoming = np.zeros(len(reaches))
        has = downstream >= 0
        np.add.at(incoming, downstream[has], q[has])
        lateral = q-incoming
        weights = np.asarray(budget['runoff_mm']).T
        sums = weights.sum(1, keepdims=True)
        weights = np.divide(weights, sums, out=np.broadcast_to(MONTH_SECONDS/YEAR_SECONDS,
                            weights.shape).copy(), where=sums>0)
        loss_weights = np.asarray(budget['potential_evaporation_mm']).T
        sums = loss_weights.sum(1, keepdims=True)
        loss_weights = np.divide(loss_weights, sums,
            out=np.broadcast_to(MONTH_SECONDS/YEAR_SECONDS, loss_weights.shape).copy(), where=sums>0)
        routed = route_monthly(downstream, np.maximum(lateral, 0)[:, None]*YEAR_SECONDS*weights,
                              loss_m3=np.maximum(-lateral, 0)[:, None]*YEAR_SECONDS*loss_weights,
                              conserve_annual_loss=True)
        monthly = routed['outflow_m3']/MONTH_SECONDS
        lengths, slopes = [], []
        for r in reaches:
            xy = np.asarray(r['xy'])
            delta = np.diff(xy, axis=0)
            if source['topology'] == 'sphere':
                phi = np.radians(90-(xy[:-1, 1]+xy[1:, 1]-2*world.bounds[1])*.5/
                                 (world.bounds[3]-world.bounds[1])*180)
                span = world.bounds[2]-world.bounds[0]
                delta[:, 0] = (delta[:, 0]+span*.5)%span-span*.5
                delta[:, 0] *= np.cos(phi)
            length = float(np.linalg.norm(delta, axis=1).sum())
            level = np.asarray(r['water_m'])
            lengths.append(length)
            slopes.append(max(0., float(level[0]-level[-1]))/max(length, 1))
        widths, depths, speeds = channel_hydraulics(monthly, np.asarray(slopes))
        # A terminal closed basin is not a sea/port. Ocean mouths explicitly
        # marked by the core are the initial downstream access targets.
        nodes = {n['id']: n for n in world.nodes}
        outlet = np.array([nodes.get(r.get('target'), {}).get('kind') == 'mouth' for r in reaches])
        if not outlet.any():
            for i, r in enumerate(reaches):
                xy = np.asarray(r['xy'])[-1:]
                outlet[i] = downstream[i] < 0 and bool(_sample(world.ocean, world.bounds, xy, wrap)[0])
        navigation = navigability(downstream, np.asarray(lengths), np.asarray(slopes), monthly,
                                  outlet_navigable=outlet)
        for i, r in enumerate(reaches):
            r.update(monthly_flow_m3s=monthly[i].tolist(),
                     unit_monthly_fraction=weights[i].tolist(),
                     length_m=lengths[i], slope=slopes[i],
                     width_m=max(.5, float(np.mean(widths[i]))),
                     depth_m=float(np.mean(depths[i])), velocity_m_s=float(np.mean(speeds[i])),
                     min_month_flow_m3s=float(monthly[i].min()))
            # Always expose per-vessel/month flags; UI wording stays a clearly
            # approximate navigation class instead of a measured certification.
            r['navigation_months'] = navigation['navigable'][:, i].astype(int).tolist()
            r['navigation'] = dict(vessels=list(navigation['vessels']),
                                  year_round_class=int(navigation['year_round_class'][i]),
                                  seasonal_class=int(navigation['seasonal_class'][i]))
        world.metadata['seasonality'] = 'synthetic monthly unit forcing; fixed annual lateral budgets'
        world.metadata['monthly_routing_residual_m3'] = np.asarray(routed['residual_m3']).tolist()
        world.metadata['monthly_mean_discharge_error_m3s'] = float(np.max(np.abs(
            (monthly*MONTH_SECONDS).sum(1)/YEAR_SECONDS-q)))

    @staticmethod
    def _monthly_budget(world, source, points):
        """Synthetic monthly balance (mm per month, shaped (12, n)) at source-cell
        samples of ``points``: the forcing of ``_properties`` reach stations."""
        from terrain_hydrology_balance import monthly_water_balance
        points = np.asarray(points, np.float64).reshape(-1, 2)
        wrap = source['topology'] == 'sphere'
        climate = [_sample(f, world.bounds, points, wrap) for f in source['climate']]
        lat = (90-(points[:, 1]-world.bounds[1])/(world.bounds[3]-world.bounds[1])*180
               if source['topology'] == 'sphere' else np.zeros(len(points)))
        seasonal = {k: _sample(v, world.bounds, points, wrap) for k, v in source.get('seasonal', {}).items()}
        return monthly_water_balance(climate[2], climate[0], latitude_deg=lat,
                                     bio4=climate[1], bio15=climate[3], **seasonal)

    def _station_climate(self, entry, points):
        """Monthly (runoff, potential evaporation, precipitation) in mm, shaped
        (n, 3, 12), at the source cells containing ``points``; cached per cell."""
        world = entry['world']
        points = np.asarray(points, np.float64).reshape(-1, 2)
        cells = np.asarray(world.cell_at(points[:, 0], points[:, 1]), np.int64).ravel()
        cache = entry.setdefault('cell_climate', {})
        missing = sorted({int(c) for c in cells} - set(cache))
        if missing:
            budget = self._monthly_budget(world, entry['source'], world.cell_xy(missing))
            values = np.stack([np.asarray(budget[k], np.float64).reshape(12, -1) for k in
                               ('runoff_mm', 'potential_evaporation_mm', 'precipitation_mm')], 1)
            for k, c in enumerate(missing):
                cache[c] = values[:, :, k].T
        return np.stack([cache[int(c)] for c in cells]) if len(cells) else np.zeros((0, 3, 12))

    def _annual_lake_climate(self, entry):
        """``climate(xy) -> (precip, pet, runoff)`` mm/year for lake budgets."""
        def climate(points):
            values = self._station_climate(entry, points).sum(-1)
            return values[:, 2], values[:, 1], values[:, 0]
        return climate

    def _local_runoff(self, entry, xy):
        """Net land yield (mm/year) of the source cell at each point; lakes and sea are 0."""
        world = entry['world']
        grid = entry.get('runoff_mm_year')
        if grid is None:
            q = np.asarray(world.discharge_m3s, np.float64).ravel()
            rcv = np.asarray(world.receivers, np.int64).ravel()
            incoming = np.zeros_like(q)
            np.add.at(incoming, rcv[rcv >= 0], q[rcv >= 0])
            local = (q-incoming).reshape(world.shape)*YEAR_SECONDS/np.asarray(world.cell_area_m2)*1000.
            grid = entry['runoff_mm_year'] = np.where(world.ocean | (world.lake_ids >= 0), 0., np.maximum(local, 0.))
        xy = np.asarray(xy, np.float64).reshape(-1, 2)
        return grid.ravel()[np.asarray(world.cell_at(xy[:, 0], xy[:, 1]), np.int64).ravel()]

    @staticmethod
    def _save(path, world):
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        def pack(value):
            if isinstance(value, np.ndarray):
                # Reach records contain tens of thousands of tiny array views.
                # One ZIP member per view made saving a world block the worker.
                if value.nbytes <= 2048 and value.dtype.kind in 'biuf':
                    return dict(_inline_array=value.tolist(), dtype=value.dtype.str, shape=value.shape)
                key = f'a{len(arrays)}'
                arrays[key] = value
                return {'_array': key}
            if isinstance(value, dict):
                return {str(k): pack(v) for k, v in value.items()}
            if isinstance(value, (tuple, list)):
                return [pack(v) for v in value]
            return value.item() if isinstance(value, np.generic) else value
        meta = json.dumps({f.name: pack(getattr(world, f.name)) for f in fields(world)
                           if f.name != '_owner_index'}, allow_nan=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.npz', delete=False) as f:
                temporary = f.name
                # UTF-8 avoids a four-byte Unicode element for every JSON
                # character. Fast DEFLATE keeps planetary packages compact
                # without holding up the single regional worker for a minute.
                payload = dict(metadata=np.frombuffer(meta.encode('utf-8'), np.uint8), **arrays)
                with zipfile.ZipFile(f, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
                    for name, value in payload.items():
                        with archive.open(name+'.npy', 'w', force_zip64=True) as member:
                            np.lib.format.write_array(member, np.asanyarray(value), allow_pickle=False)
            os.replace(temporary, path)
            temporary = None
        finally:
            if temporary:
                Path(temporary).unlink(missing_ok=True)

    @staticmethod
    def _load(path, record_class=None):
        if not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as data:
                def unpack(value):
                    if isinstance(value, dict):
                        if set(value) == {'_array'}:
                            return data[value['_array']]
                        if '_inline_array' in value:
                            return np.asarray(value['_inline_array'], dtype=value['dtype']).reshape(value['shape'])
                        return {k: unpack(v) for k, v in value.items()}
                    if isinstance(value, list):
                        return [unpack(v) for v in value]
                    return value
                encoded = data['metadata']
                metadata = encoded.tobytes().decode('utf-8') if encoded.dtype == np.uint8 else str(encoded)
                meta = unpack(json.loads(metadata))
            from terrain_hydrology import HydroWorld
            meta['bounds'] = tuple(meta['bounds'])
            return (record_class or HydroWorld)(**meta)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    @staticmethod
    def radius(world):
        return (world.bounds[2]-world.bounds[0])/(2*np.pi) if world.topology == 'sphere' else None

    def _chosen(self, entry, bounds, limit=512, lod=None):
        world = entry['world']
        pad = max(world._spacing())*2
        box = (bounds[0]-pad, bounds[1]-pad, bounds[2]+pad, bounds[3]+pad)
        period = world.bounds[2]-world.bounds[0]
        shifts = (-period, 0, period) if world.topology == 'sphere' else (0,)
        b = np.asarray(entry['reach_bounds']).reshape(-1, 4)
        hit = np.zeros(len(b), bool)
        for s in shifts:
            hit |= (b[:, 0]+s <= box[2]) & (b[:, 2]+s >= box[0]) & (b[:, 1] <= box[3]) & (b[:, 3] >= box[1])
        chosen = np.flatnonzero(hit)
        flow = np.array([world.reaches[i]['flow_m3s'] for i in chosen], np.float64)
        if lod is not None:
            keep = 7.2*np.sqrt(flow) >= .08*30*2**lod
            chosen, flow = chosen[keep], flow[keep]
        if len(chosen) > limit:
            chosen = chosen[np.argsort(-flow, kind='stable')[:limit]]
        return chosen.tolist()

    def _geometry(self, entry, index, cosmetics=frozenset()):
        """One physical curve per reach and option set, independent of
        viewport, month and LOD. Without cosmetics: the smoothed axis."""
        from terrain_hydrology_geometry import refine_reach, cosmetic_options
        world = entry['world']
        key = (index, cosmetics)
        with self.lock:
            geometry = entry['geometry'].get(key)
        if geometry is None:
            # Refined outside the shared lock: concurrent tiles would
            # otherwise serialise on one another's curves.
            raw = world.reaches[index]
            mid = np.asarray(raw['xy'])[len(raw['xy'])//2:len(raw['xy'])//2+1]
            climate = entry['source']['climate']
            tangent = entry['tangents'].get(index, {})
            geometry = dict(raw, **refine_reach(raw,
                seed=entry['seed'], terrain=entry['height64'], terrain_bounds=world.bounds,
                climate=dict(temperature_c=float(_sample(climate[0], world.bounds, mid, world.topology == 'sphere')[0]),
                             precip_mm_year=float(_sample(climate[2], world.bounds, mid, world.topology == 'sphere')[0])),
                sphere_radius_m=self.radius(world),
                wrap_period_m=world.bounds[2]-world.bounds[0] if self.radius(world) else None,
                start_tangent=tangent.get('start'), end_tangent=tangent.get('end'),
                **cosmetic_options(cosmetics)))
        with self.lock:
            if key not in entry['geometry']:
                entry['geometry'][key] = geometry
                entry['geometry_bytes'] += sum(v.nbytes for v in geometry.values() if isinstance(v, np.ndarray))
            entry['geometry'].move_to_end(key)
            while len(entry['geometry']) > 1 and (len(entry['geometry']) > 4096 or entry['geometry_bytes'] > 128*1024**2):
                _, evicted = entry['geometry'].popitem(last=False)
                entry['geometry_bytes'] -= sum(v.nbytes for v in evicted.values() if isinstance(v, np.ndarray))
            return entry['geometry'][key]

    def _chart_references(self):
        """Whether ``read_detail`` can return raw per-chart Base references."""
        try:
            return 'charts' in inspect.signature(self.read_detail).parameters
        except (TypeError, ValueError):
            return False

    def _regions(self, entry, bounds, lod, reference=None):
        """Fixed 8-coarse-cell packages. Requests subscribe; they never define
        the routing domain. Seam/pole guard strips keep the global authority.

        ``reference`` names the raw chart whose decoder a fallback composition
        will correct. A ready package lacking that chart's correction is
        completed lazily on the same worker and reported as pending meanwhile.

        Views coarser than the 240 m Base (``lod > 3``) never schedule Base,
        neural or reference work: they reuse packages that are already ready,
        so a generated course does not revert to the source preview.
        """
        if self.read_detail is None:
            return [], 0
        if lod > 3:
            return self._ready_regions(entry, bounds), 0
        if reference is not None and not self._chart_references():
            reference = None
        world, size = entry['world'], 8*7680.
        extent = world.bounds
        period = extent[2]-extent[0]
        if world.topology == 'sphere' and bounds[2]-bounds[0] < period and (bounds[0] < extent[0] or bounds[2] > extent[2]):
            start = (bounds[0]-extent[0]) % period+extent[0]
            stop = start+bounds[2]-bounds[0]
            boxes = [(start, bounds[1], min(stop, extent[2]), bounds[3])]
            if stop > extent[2]:
                boxes.append((extent[0], bounds[1], extent[0]+stop-extent[2], bounds[3]))
            parts, count = {}, 0
            for box in boxes:
                ready, pending = self._regions(entry, box, lod, reference)
                parts.update((p['bounds'], p) for p in ready)
                count += pending
            return list(parts.values()), count
        pieces = []
        for row in range(int(np.floor(bounds[1]/size)), int(np.ceil(bounds[3]/size))):
            for col in range(int(np.floor(bounds[0]/size)), int(np.ceil(bounds[2]/size))):
                core = (col*size, row*size, (col+1)*size, (row+1)*size)
                if world.topology == 'sphere' and (core[0] < extent[0] or core[2] > extent[2]
                        or max(abs(core[1]), abs(core[3])) > .47*(extent[3]-extent[1])):
                    continue
                # Do not request a neural base package for open ocean.
                dx, dy = world._spacing()
                c0, c1 = np.clip([int(np.floor((core[0]-extent[0])/dx)), int(np.ceil((core[2]-extent[0])/dx))], 0, world.shape[1])
                r0, r1 = np.clip([int(np.floor((core[1]-extent[1])/dy)), int(np.ceil((core[3]-extent[1])/dy))], 0, world.shape[0])
                if world.ocean[r0:r1, c0:c1].all():
                    continue
                pieces.append(core)
        # Closest pieces first. Every required piece is ready, building or
        # counted pending: a tile is never reported ready with parts omitted.
        # Builds stay bounded (four in flight). A view needing more packages
        # than the cache holds stays pending until a narrower view fits.
        center = np.array([(bounds[0]+bounds[2])*.5, (bounds[1]+bounds[3])*.5])
        pieces.sort(key=lambda b: np.sum((np.array([(b[0]+b[2])*.5, (b[1]+b[3])*.5])-center)**2))
        ready, pending = [], 0
        with self.lock:
            active = sum(p['status'] == 'building' for p in entry['regions'].values())
            for core in pieces[:REGION_CACHE]:
                part = entry['regions'].get(core)
                if part is None or part['status'] == 'error' and time.monotonic() >= part.get('retry_at', np.inf):
                    if active >= 4:
                        pending += 1
                        continue
                    part = dict(status='building', bounds=core, references={reference} - {None})
                    entry['regions'][core] = part
                    part['future'] = self.pool.submit(self._build_region, entry, part)
                    active += 1
                part['last_seen'] = time.monotonic()
                entry['regions'].move_to_end(core)
                if part['status'] == 'ready' and reference is not None and \
                        reference not in part['plan'].metadata.get('chart_correction_m', {}):
                    jobs = part.setdefault('reference_jobs', {})
                    job = jobs.get(reference)
                    if job is None or job.done() and time.monotonic() >= part.get('reference_retry_at', 0.):
                        jobs[reference] = self.pool.submit(self._add_reference, entry, part, reference)
                    pending += 1
                elif part['status'] == 'ready':
                    ready.append(part)
                elif part['status'] == 'building':
                    pending += 1
                elif part['status'] == 'error' and np.isfinite(part.get('retry_at', np.inf)):
                    pending += 1
            for core in pieces[REGION_CACHE:]:
                part = entry['regions'].get(core)
                if part is not None and part['status'] == 'ready' and (reference is None or
                        reference in part['plan'].metadata.get('chart_correction_m', {})):
                    part['last_seen'] = time.monotonic()
                    entry['regions'].move_to_end(core)
                    ready.append(part)
                else:
                    pending += 1
            # Evict least recently used packages that no tile has requested
            # within the 30 s supersede window. Concurrent tiles of one view
            # may overflow the soft size, never the hard ceiling.
            wanted, now = set(pieces), time.monotonic()
            while len(entry['regions']) > REGION_CACHE:
                idle = [k for k, v in entry['regions'].items() if v['status'] != 'building' and k not in wanted]
                old = next((k for k in idle if now-entry['regions'][k].get('last_seen', 0.) > 30.), None)
                if old is None and len(entry['regions']) > REGION_CEILING and idle:
                    old = idle[0]
                if old is None:
                    break
                del entry['regions'][old]
        return ready, pending

    def _ready_regions(self, entry, bounds):
        """Cached ready packages touching ``bounds`` (periodic), in core order."""
        world = entry['world']
        with self.lock:
            parts = list(entry['regions'].values())
        return sorted((p for p in parts if p['status'] == 'ready' and
                       _world_intersects(world, p['bounds'], bounds)), key=lambda p: p['bounds'])

    @staticmethod
    def _coarse_visible(world, reach, lod=None):
        """Coarse views show a regional course under the source network's
        own rules: its channel threshold and, for rasters, the LOD width cut."""
        parameters = getattr(world, 'metadata', {}).get('parameters', {})
        q = float(reach['flow_m3s'])
        if q < float(parameters.get('channel_min_discharge_m3s', 1.)):
            return False
        # Fine history must not reveal a rectangular patch of small tributaries
        # in a coarse view. Keep the source network's drainage-area threshold.
        if (reach.get('kind') == 'tributary' and
                float(reach.get('area_m2', 0.)) < float(parameters.get('channel_min_area_m2', 0.))):
            return False
        return lod is None or 7.2*np.sqrt(q) >= .08*30*2**lod

    def _river_query_bounds(self, world, bounds, cell_m, max_flow=None):
        """Conservative inverse support of a regional river raster.

        Generated channel/belt positions stay in core + 3840 map metres.
        Effective belt width is at most four bankfull widths and is only
        used below cell/2.75 bankfull width. Seasonality is at most 1.8;
        three representation widths cover wet pixels and bank searches.
        The extra target cell covers antialiasing at every target LOD.
        """
        from terrain_hydrology_geometry import bankfull_geometry, x_scale
        q = (max(float(np.max(getattr(world, 'discharge_m3s', np.zeros(0)), initial=0.)),
                 max((r['flow_m3s'] for r in world.reaches), default=0.))
             if max_flow is None else max_flow)
        bankfull = float(bankfull_geometry(q)[0])
        width = max(bankfull, 4*min(bankfull, cell_m/2.75))*1.8
        kernel = 3*width+cell_m
        yp = REGIONAL_RIVER_SUPPORT_M+kernel
        scale = float(np.min(x_scale(np.array([bounds[1]-yp, bounds[3]+yp]), self.radius(world))))
        xp = REGIONAL_RIVER_SUPPORT_M+kernel/scale
        return (bounds[0]-xp, bounds[1]-yp, bounds[2]+xp, bounds[3]+yp)

    def _build_region(self, entry, part):
        from terrain_hydrology_regions import refine_region, RegionPlan
        from terrain_hydrology_geometry import prepare_water_grid
        try:
            # A queued package superseded by a long camera move should not
            # keep generating neural terrain that nobody is inspecting.
            if time.monotonic()-part.get('last_seen', time.monotonic()) > 30.:
                part.update(status='error', retry_at=0., error='Superseded regional request')
                return
            core, world = part['bounds'], entry['world']
            tag = '_'.join(str(int(v/7680)) for v in core)
            path = self._region_dir(entry)/f'{tag}.npz'
            plan = self._load(path, RegionPlan)
            if plan is None:
                profiles = self._water_profiles(entry, core)
                raw = self.read_detail(entry['seed'], entry['profile'], core)
                plan = refine_region(world, **raw, core_bounds=core, seed=entry['seed'], water_profiles=profiles,
                                     strong_incision_m=REGIONAL_VALLEY_CUT_M)
                plan.metadata['base_correction_m'] = self._halo_correction(plan, raw['base_height'], raw['base_bounds'])
                if part.get('references'):
                    self._chart_correction(entry, plan, core, sorted(part['references']))
                try:
                    self._save(path, plan)
                except OSError:
                    pass
            elif set(part.get('references', ())) - set(plan.metadata.get('chart_correction_m', {})):
                self._chart_correction(entry, plan, core, sorted(part['references']))
                try:
                    self._save(path, plan)
                except OSError:
                    pass
            geometry, diagnostics = self._prepare_courses(entry, part, plan)
            # Global lakes retain global shores; this grid adds only wholly
            # owned, flow-through local depressions (never an edge-cut lake).
            wet_lakes = []
            for lake in plan.local_lakes:
                flows = np.asarray(plan.flow_m3s).ravel()[np.asarray(lake['cells'], np.int64)]
                through = float(np.max(flows[np.isfinite(flows)], initial=0.))
                if lake.get('parent_lake_id') is not None or through > 1e-6:
                    wet_lakes.append(lake)
            # Priority-Flood also identifies potential depressions in a dry
            # catchment. Without annual water they are not standing lakes.
            plan.metadata['quality']['dry_local_depressions'] = len(plan.local_lakes)-len(wet_lakes)
            display_lakes = wet_lakes if DISPLAY_LAKES else []
            local_grid = prepare_water_grid(plan.conditioned_base_m, plan.bounds, lakes=display_lakes)
            part.update(plan=plan, geometry=geometry, local_water_grid=local_grid, local_lakes=display_lakes,
                        geometry_diagnostics=diagnostics, status='ready')
        except Exception as exc:
            transient = type(exc).__name__ in ('QueueFull', 'TimeoutError', 'JobCancelled')
            part.update(status='error', error=f'{type(exc).__name__}: {exc}',
                        retry_at=time.monotonic()+10. if transient else np.inf)

    def _prepare_courses(self, entry, part, plan):
        """Display courses of a regional plan (detailed or vector): parent
        attributes, monthly/navigation properties, compact chains, axis geometry."""
        from terrain_hydrology_geometry import compact_display_reaches, bankfull_geometry, x_scale
        core, world = part['bounds'], entry['world']
        # Forcing and navigation remain attached to the inherited parent.
        climate = entry['source']['climate']
        for reach in plan.reaches:
            parent = reach.get('parent_id')
            # A trunk is its parent reach: it keeps the source-network order.
            # A local tributary gets its own order below, never its unit's.
            if reach.get('kind') == 'trunk' and parent is not None and 0 <= int(parent) < len(world.reaches):
                reach['strahler'] = world.reaches[int(parent)].get('strahler', 1)
                reach['strahler_scale'] = 'source'
            # A re-saved plan already holds endpoint tuples as targets.
            target = reach.get('target')
            reach['terminal_kind'] = target if isinstance(target, str) else reach.get('terminal_kind')
            # Shared geometry directions use actual endpoint positions.
            # Compact local names such as 'src' are not globally unique.
            reach['source'] = tuple(np.round(np.asarray(reach['xy'])[0], 3))
            reach['target'] = tuple(np.round(np.asarray(reach['xy'])[-1], 3))
        self._local_orders(plan.reaches)
        fractions = self._unit_fractions(entry, plan)
        self._regional_properties(world, plan, unit_fraction=lambda r: fractions.get(r['id']))
        # Annual-zero drainage remains in routing/budget arrays. It has no
        # water in any month, so do not tessellate or repair invisible bends.
        visible_reaches = compact_display_reaches([r for r in plan.reaches if r['flow_m3s'] > 1e-12])
        for reach in visible_reaches:
            reach['end_kind'] = self._end_kind(reach)
        plan.metadata['quality']['annual_dry_reaches'] = sum(r['flow_m3s'] <= 1e-12 for r in plan.reaches)
        support = (core[0]-REGIONAL_RIVER_SUPPORT_M, core[1]-REGIONAL_RIVER_SUPPORT_M,
                   core[2]+REGIONAL_RIVER_SUPPORT_M, core[3]+REGIONAL_RIVER_SUPPORT_M)
        for reach in visible_reaches:
            # A quarter of a native decoder cell avoids visible chords on
            # wide rivers, without coarsening small streams' finer tolerance.
            reach['tolerance_m'] = 7.5
            if len(reach['members']) < 2:
                continue
            # Long courses can reveal nested bends in an open floodplain.
            # Actual valley walls still limit motion, as does package support.
            xy = np.asarray(reach['xy'])
            cosine = x_scale(xy[:, 1], self.radius(world))
            room = min(np.min((xy[:, 0]-support[0])*cosine), np.min((support[2]-xy[:, 0])*cosine),
                       np.min(xy[:, 1]-support[1]), np.min(support[3]-xy[:, 1]))
            width = float(bankfull_geometry(reach['flow_m3s'])[0])
            delta = np.diff(xy, axis=0)
            spacing = np.median(np.hypot(delta[:, 0]*cosine[:-1], delta[:, 1]))
            reach['corridor_m'] = max(.75*spacing, 2.5*width, min(8*width, room))
        mid = np.array([r['xy'][len(r['xy'])//2] for r in visible_reaches])
        forcing = ({r['id']: dict(temperature_c=float(t), precip_mm_year=float(p))
                    for r, t, p in zip(visible_reaches, _sample(climate[0], world.bounds, mid, world.topology == 'sphere'),
                                       _sample(climate[2], world.bounds, mid, world.topology == 'sphere'))} if len(mid) else {})
        part.update(display_reaches=visible_reaches, forcing=forcing, support=support,
                    terrain=(plan.routing_height_m, plan.bounds))
        if part.get('mode') == 'vector':
            part['display_terrain'] = (plan.metadata.get('routing_halo_m', plan.routing_height_m),
                                       self._correction_bounds(plan))
        geometry, diagnostics = self._refine_part(entry, part, frozenset())
        self._prepare_lakes(entry, part, plan, geometry)
        return geometry, diagnostics

    def _prepare_lakes(self, entry, part, plan, geometry):
        """Vector lakes of one package, computed once on the worker: displayable
        local ponds, Base pieces of source lakes and routed lake connectors."""
        from terrain_hydrology_lakes import local_lakes, local_connectors, source_lake_pieces
        try:
            shown, hidden = local_lakes(plan, self._annual_lake_climate(entry))
            vector = part.get('mode') == 'vector'
            links = local_connectors(plan, geometry, display_key='display_xy' if vector else 'xy')
            for link in links.values():
                if vector:
                    link['xy'], link['water_m'] = _land_course(link['xy'], link['water_m'], *part['display_terrain'])
                link['lake_shown'] = link['lake_id'] in shown
            part.update(vector_lakes=shown, lake_hidden=hidden,
                        lake_connectors={k: v for k, v in links.items() if len(v['xy']) >= 2},
                        source_lake_pieces=source_lake_pieces(entry['world'], plan))
        except Exception as exc:     # rivers stay usable without their lakes
            part.update(vector_lakes={}, lake_hidden={}, lake_connectors={}, source_lake_pieces={},
                        lake_error=f'{type(exc).__name__}: {exc}')

    @staticmethod
    def _local_orders(reaches):
        """Strahler order of wet tributaries on the package's own 240 m network.

        Orders cross local lakes (``downstream_id`` follows the pond outlet). A
        record fed through a unit link from a neighbouring package only knows its
        own side: its order is a lower bound (``strahler_partial``)."""
        tribs = {r['id']: r for r in reaches if r.get('kind') == 'tributary' and r['flow_m3s'] > 1e-12}
        upstream = {}
        for r in tribs.values():
            if r.get('downstream_id') in tribs:
                upstream.setdefault(r['downstream_id'], []).append(r['id'])
        pending = {k: len(upstream.get(k, ())) for k in tribs}
        ready = [k for k, n in pending.items() if n == 0]
        order, partial = {}, {}
        while ready:
            k = ready.pop()
            children = upstream.get(k, ())
            values = [order[c] for c in children]
            top = max(values, default=1)
            order[k] = top+1 if values.count(top) >= 2 else top
            partial[k] = bool(tribs[k].get('start_link')) or any(partial[c] for c in children)
            d = tribs[k].get('downstream_id')
            if d in pending:
                pending[d] -= 1
                if not pending[d]:
                    ready.append(d)
        for k, r in tribs.items():   # a (never expected) cycle keeps an unknown order
            r.update(strahler=order.get(k, 1), strahler_scale='base-240m', strahler_partial=partial.get(k, True))

    @staticmethod
    def _unit_station(world, key, fallback):
        """Representative forcing station of an incremental unit without a reach."""
        nreach, nlake = len(world.reaches), len(world.lakes)
        if nreach <= key < nreach+nlake and world.lakes[key-nreach].get('xy') is not None:
            return np.asarray(world.lakes[key-nreach]['xy'], np.float64)
        roots = world.basins.get('root_cell', ()) if isinstance(world.basins, dict) else ()
        if 0 <= key-nreach-nlake < len(roots):
            return world.cell_xy([int(roots[key-nreach-nlake])])[0]
        return np.asarray(fallback, np.float64)

    def _unit_fractions(self, entry, plan):
        """Monthly runoff weights for records whose unit has no source reach."""
        world = entry['world']
        nreach = len(world.reaches)
        links = {link['id']: link for link in getattr(plan, 'links', ())}
        keys = np.asarray(plan.unit_key).ravel()
        stations = {}
        for r in plan.reaches:
            inlet = links.get(r.get('start_link'), {})
            parent = r.get('parent_id')
            if parent is None:
                parent = inlet.get('unit_key')
            if parent is not None and 0 <= int(parent) < nreach:
                continue
            key = inlet.get('unit_key')
            cells = r.get('cells')
            if key is None:
                key = int(keys[int(np.asarray(cells)[0])]) if cells is not None and len(cells) else -1
            stations[r['id']] = self._unit_station(world, int(key), np.asarray(r['xy'], np.float64)[0])
        if not stations:
            return {}
        values = self._station_climate(entry, np.array(list(stations.values())))
        return {rid: values[k, 0] for k, rid in enumerate(stations)}

    @staticmethod
    def _end_kind(reach):
        """Immediate downstream end of a regional display record."""
        kind = str(reach.get('terminal_kind') or '')
        if reach.get('kind') == 'trunk':
            if reach.get('end_portal') is not None:
                return 'portal'
            return kind if kind in TRUNK_END_KINDS else 'confluence'
        return kind.split(':', 1)[0] or 'terminal'

    def _cores(self, world, bounds):
        """Fixed vector land cores touching ``bounds`` (periodic), centre first."""
        size, extent = VECTOR_CORE_M, world.bounds
        period = extent[2]-extent[0]
        boxes = [tuple(bounds)]
        if world.topology == 'sphere' and bounds[2]-bounds[0] < period and (bounds[0] < extent[0] or bounds[2] > extent[2]):
            start = (bounds[0]-extent[0]) % period+extent[0]
            stop = start+bounds[2]-bounds[0]
            boxes = [(start, bounds[1], min(stop, extent[2]), bounds[3])]
            if stop > extent[2]:
                boxes.append((extent[0], bounds[1], extent[0]+stop-extent[2], bounds[3]))
        dx, dy = world._spacing()
        cores = []
        for box in boxes:
            for row in range(int(np.floor(box[1]/size)), int(np.ceil(box[3]/size))):
                for col in range(int(np.floor(box[0]/size)), int(np.ceil(box[2]/size))):
                    core = (col*size, row*size, (col+1)*size, (row+1)*size)
                    if world.topology == 'sphere' and (core[0] < extent[0] or core[2] > extent[2]
                            or max(abs(core[1]), abs(core[3])) > .47*(extent[3]-extent[1])):
                        continue
                    c0, c1 = np.clip([int(np.floor((core[0]-extent[0])/dx)), int(np.ceil((core[2]-extent[0])/dx))], 0, world.shape[1])
                    r0, r1 = np.clip([int(np.floor((core[1]-extent[1])/dy)), int(np.ceil((core[3]-extent[1])/dy))], 0, world.shape[0])
                    if core not in cores and not world.ocean[r0:r1, c0:c1].all():
                        cores.append(core)
        cx, cy = (bounds[0]+bounds[2])*.5, (bounds[1]+bounds[3])*.5
        def distance(core):
            ex = (core[0]+core[2])*.5-cx
            if world.topology == 'sphere':
                ex = (ex+period*.5) % period-period*.5
            return ex*ex+((core[1]+core[3])*.5-cy)**2
        return sorted(cores, key=distance)

    def _vector_regions(self, entry, bounds, lod):
        """Ready default vector packages touching ``bounds``.

        Returns ``(ready, cpu_pending, unavailable)``. Only views at Base scale
        (lod <= 3) subscribe new cores; a core is built only from Base that the
        displayed terrain already generated. A core without it stays on the
        source overview (``unavailable``), never a pending neural request."""
        store = entry.setdefault('vector_regions', OrderedDict())
        if self.read_cached_base is None:
            return [], 0, 0
        world = entry['world']
        if lod > 3:
            with self.lock:
                ready = [p for p in store.values() if p['status'] == 'ready'
                         and _world_intersects(world, p['bounds'], bounds)]
            return sorted(ready, key=lambda p: p['bounds']), 0, 0
        cores = self._cores(world, bounds)[:VECTOR_CACHE]
        ready, pending, unavailable, now = [], 0, 0, time.monotonic()
        with self.lock:
            active = sum(p['status'] == 'building' for p in store.values())
            for core in cores:
                part = store.get(core)
                if part is None or part['status'] == 'unavailable' or (
                        part['status'] == 'error' and now >= part['retry_at']):
                    if active >= VECTOR_BUILDS:
                        pending += 1
                        continue
                    tag = '_'.join(str(int(v/7680)) for v in core)
                    directory = entry.get('vector_dir') or self.root/entry['key']/'vector-regions'
                    raw = None
                    if not (directory/f'{tag}.npz').is_file():
                        # Probe without waiting for the GPU, before queueing CPU
                        # work. Otherwise missing cores continually resubmit
                        # probes and a wide view can remain "building" forever.
                        from terrain_hydrology_regions import piece_windows
                        try:
                            raw = self.read_cached_base(entry['seed'], entry['profile'],
                                piece_windows(core, align_coarse=False)[0])
                        except Exception as exc:
                            store[core] = dict(status='error', bounds=core, last_seen=now,
                                retry_at=now+30., error=f'{type(exc).__name__}: {exc}')
                            unavailable += 1
                            continue
                        if raw is None:
                            store[core] = dict(status='unavailable', bounds=core, last_seen=now)
                            unavailable += 1
                            continue
                    part = store[core] = dict(status='building', bounds=core, mode='vector',
                                               last_seen=now, cached_base=raw)
                    part['future'] = self.pool.submit(self._build_vector_region, entry, part)
                    active += 1
                part['last_seen'] = now
                store.move_to_end(core)
                if part['status'] == 'ready':
                    ready.append(part)
                elif part['status'] == 'building':
                    pending += 1
                else:
                    unavailable += 1
            wanted = set(cores)
            while len(store) > VECTOR_CACHE:
                idle = [k for k, v in store.items() if v['status'] != 'building' and k not in wanted]
                if not idle:
                    break
                del store[idle[0]]
        return ready, pending, unavailable

    def _build_vector_region(self, entry, part):
        """Vector-only package: Base-only routing with source water levels
        (no Coarse profiles, parent composition, carving or local water grid)."""
        from terrain_hydrology_regions import refine_region, RegionPlan, piece_windows
        try:
            if time.monotonic()-part['last_seen'] > 30.:
                part.update(status='unavailable', retry_at=0.)
                return
            core, world = part['bounds'], entry['world']
            tag = '_'.join(str(int(v/7680)) for v in core)
            directory = entry.get('vector_dir') or self.root/entry['key']/'vector-regions'
            plan = self._load(directory/f'{tag}.npz', RegionPlan)
            if plan is None:
                raw = part.pop('cached_base', None)
                if raw is None:
                    raw = self.read_cached_base(entry['seed'], entry['profile'], piece_windows(core, align_coarse=False)[0])
                if raw is None:
                    part.update(status='unavailable')
                    return
                plan = refine_region(world, None, None, raw['base_height'], raw['base_bounds'], core,
                                     seed=entry['seed'], align_coarse=False, strong_incision_m=REGIONAL_VALLEY_CUT_M)
                try:
                    self._save(directory/f'{tag}.npz', plan)
                except OSError:
                    pass
            geometry, diagnostics = self._prepare_courses(entry, part, plan)
            part.update(plan=plan, geometry=geometry, geometry_diagnostics=diagnostics,
                        local_lakes=[], status='ready')
        except Exception as exc:
            part.update(status='error', error=f'{type(exc).__name__}: {exc}', retry_at=time.monotonic()+30.)

    @staticmethod
    def _bridge(fragments, reach_id, frontier, tolerance=1e-3):
        """Join overview fragments cut at a ready core edge to that package's
        portal of the same parent reach on the same edge (valley portals may
        lie up to a corridor away from the parent crossing). Levels approximate."""
        out = []
        for xy, water in fragments:
            xy, water = np.asarray(xy, np.float64), np.asarray(water, np.float64)
            for at in (0, -1):
                point, best = xy[at], None
                for box, shift, plan in frontier:
                    for axis, k in ((0, 0), (0, 2), (1, 1), (1, 3)):
                        if abs(point[axis]-box[k]) > tolerance or not (
                                box[1-axis]-tolerance <= point[1-axis] <= box[3-axis]+tolerance):
                            continue
                        line = box[k]-(shift if axis == 0 else 0.)
                        for portal in getattr(plan, 'portals', ()):
                            if (portal.get('reach_id') is None or int(portal['reach_id']) != int(reach_id)
                                    or portal.get('axis') != 'xy'[axis] or abs(float(portal['line_m'])-line) > tolerance):
                                continue
                            p = np.asarray(portal['xy'], np.float64)+(shift, 0.)
                            d = float(np.hypot(*(p-point)))
                            if best is None or d < best[0]:
                                best = (d, p, float(portal['level_m']))
                if best is not None and best[0] > tolerance:
                    if at == 0:
                        xy, water = np.vstack([best[1], xy]), np.r_[best[2], water]
                    else:
                        xy, water = np.vstack([xy, best[1]]), np.r_[water, best[2]]
            out.append((xy, water))
        return out

    def _region_dir(self, entry):
        # Hand-built (offline/test) entries may lack the split cache layout.
        return entry.get('region_dir') or self.root/entry['key']/'regions'

    def _refine_part(self, entry, part, cosmetics):
        """Regional courses of one package for one cosmetic option set."""
        from terrain_hydrology_geometry import refine_network, cosmetic_options
        world, reaches = entry['world'], part['display_reaches']
        geometry, diagnostics = refine_network(reaches, seed=entry['seed'],
            climate=lambda reach: part['forcing'].get(reach['id'], {}),
            terrain=part['terrain'][0], terrain_bounds=part['terrain'][1],
            sample_spacing_m=240., sphere_radius_m=self.radius(world),
            wrap_period_m=world.bounds[2]-world.bounds[0] if self.radius(world) else None,
            fit_profile=False, support_bounds=part['support'], **cosmetic_options(cosmetics))
        geometry = [dict(raw, **g) for raw, g in zip(reaches, geometry)]
        if part.get('mode') == 'vector':
            for g in geometry:
                g['display_xy'], g['display_water_m'] = _land_course(
                    g['xy'], g['water_m'], *part['display_terrain'])
        return geometry, diagnostics

    def _part_geometry(self, entry, part, cosmetics):
        """Ready package courses; a cosmetic variant is refined once on demand
        (in memory only: stored plans never depend on display options)."""
        if not cosmetics:
            return part['geometry']
        variants = part.setdefault('geometry_variants', {})
        if cosmetics not in variants:
            variants[cosmetics] = self._refine_part(entry, part, cosmetics)[0]
        return variants[cosmetics]

    def _chart_correction(self, entry, plan, core, charts):
        """Per-chart ``H_routing - Base_chart`` on the plan's geographic grid.

        A display decoder belongs to one raw chart; in the canonical blend
        strips its own Base differs from the canonical Base. Correcting it
        against this raw reference replaces exactly that chart's parent and
        keeps its decoder residual. Stored in the package, never in the NN cache.
        """
        raw = self.read_detail(entry['seed'], entry['profile'], core, charts=tuple(charts))
        corrections = dict(plan.metadata.get('chart_correction_m', {}))
        for chart in charts:
            corrections[chart] = self._halo_correction(plan, raw['chart_base'][chart], raw['base_bounds'])
        plan.metadata['chart_correction_m'] = corrections

    @staticmethod
    def _correction_bounds(plan):
        return tuple(plan.metadata.get('routing_halo_bounds', plan.bounds))

    @staticmethod
    def _halo_correction(plan, base, base_bounds):
        """``H_routing - Base`` over core + one 240 m cell, so bilinear samples
        near a core edge use the true neighbouring centres, not a clamp."""
        box = HydrologyService._correction_bounds(plan)
        routing = plan.metadata.get('routing_halo_m', plan.routing_height_m)
        r0, c0 = (round((box[i]-base_bounds[i])/240) for i in (1, 0))
        crop = np.asarray(base)[r0:r0+routing.shape[0], c0:c0+routing.shape[1]]
        if crop.shape != routing.shape:
            raise ValueError('Base reference does not cover the correction extent')
        return (np.asarray(routing, np.float64)-crop).astype(np.float32)

    def _add_reference(self, entry, part, chart):
        try:
            plan = part['plan']
            self._chart_correction(entry, plan, part['bounds'], [chart])
            tag = '_'.join(str(int(v/7680)) for v in part['bounds'])
            try:
                self._save(self._region_dir(entry)/f'{tag}.npz', plan)
            except OSError:
                pass
        except Exception as exc:
            part.update(reference_error=f'{type(exc).__name__}: {exc}', reference_retry_at=time.monotonic()+10.)

    def _water_profiles(self, entry, core):
        if self.read_coarse_points is None:
            return None
        from terrain_hydrology_profiles import build_profiles
        from terrain_hydrology_regions import required_profile_reaches
        world = entry['world']
        requested = self._chosen(entry, core, limit=10000)
        xx, yy = np.meshgrid(np.linspace(core[0], core[2], 10), np.linspace(core[1], core[3], 10))
        owners = world.sample(xx, yy)['owner_reach']
        requested = sorted(set(requested) | set(int(v) for v in owners.ravel() if v >= 0)
                           | set(required_profile_reaches(world, core)))
        missing = [r for r in requested if r not in entry['profiles']]
        if missing:
            built = build_profiles(world, missing, lambda x, y:
                self.read_coarse_points(entry['seed'], entry['profile'], x, y), known=entry['profiles'],
                conflict_incision_m=REGIONAL_VALLEY_CUT_M)
            entry['profiles'].update(built.profiles)
            entry['profile_quality'] = built.metadata
        class Lookup:
            def level(self, reach_id, station):
                profile = entry['profiles'][reach_id]
                return np.interp(station, profile['station'], profile['water_m'])
        return Lookup()

    @staticmethod
    def _regional_properties(world, plan, unit_fraction=None):
        """Local section limits AND the seasonal inherited downstream route.
        A small tributary cannot inherit the ship access of its parent river.

        ``unit_fraction(record)`` gives monthly runoff weights of a record whose
        unit has no source reach (lake or coastal-fringe units): its unit's
        representative station. Without it such records get the old uniform
        regime."""
        if not plan.reaches:
            return
        from terrain_hydrology_balance import navigability, route_monthly, MONTH_SECONDS
        reaches, n = plan.reaches, len(plan.reaches)
        lookup = {r['id']: i for i, r in enumerate(reaches)}
        downstream = np.array([lookup.get(r.get('downstream_id'), -1) for r in reaches])
        q = np.array([max(0., r['flow_m3s']) for r in reaches])
        incoming = np.zeros(n)
        has = downstream >= 0
        np.add.at(incoming, downstream[has], q[has])
        links = {link['id']: link for link in getattr(plan, 'links', ())}
        external, inherited, local, parents = np.zeros(n), [], [], []
        def fraction(value):
            value = np.maximum(np.asarray(value, np.float64), 0.)
            return value/value.sum() if value.sum() > 0 else MONTH_SECONDS/YEAR_SECONDS
        for i, r in enumerate(reaches):
            inlet = links.get(r.get('start_link'), {})
            parent = r.get('parent_id')
            if parent is None:
                parent = inlet.get('unit_key')
            parent = int(parent) if parent is not None and 0 <= int(parent) < len(world.reaches) else None
            parents.append(parent)
            source = world.reaches[parent] if parent is not None else {}
            # A record's own earlier monthly output is never its forcing: a
            # re-saved detailed plan would otherwise keep a stale regime.
            own = unit_fraction(r) if parent is None and unit_fraction is not None else None
            shape = fraction(own if own is not None else np.asarray(source.get('monthly_flow_m3s', [1.]*12))*MONTH_SECONDS)
            unit = fraction(source.get('unit_monthly_fraction', shape))
            if r.get('kind') == 'trunk':
                external[i] = max(0., float(np.asarray(r.get('flow_profile_m3s', [q[i]]))[0])-incoming[i])
            elif r.get('start_link') and incoming[i] == 0:
                external[i] = max(0., inlet.get('flow_m3s', q[i]))
                # Unit links carry incremental drainage, not the main trunk's
                # upstream hydrograph. Both packages derive this same regime.
                key = int(inlet.get('unit_key', -1))
                if 0 <= key < len(world.reaches):
                    shape = fraction(world.reaches[key].get('unit_monthly_fraction', shape))
            inherited.append(shape)
            local.append(unit)
        lateral = q-incoming-external
        local, inherited = np.asarray(local), np.asarray(inherited)
        # Route inherited inflow and new local yield together. Merely assigning
        # one regime per record loses the inflow's season after a confluence.
        volumes = YEAR_SECONDS*(external[:, None]*inherited+np.maximum(lateral, 0)[:, None]*local)
        routed = route_monthly(downstream, volumes,
            loss_m3=YEAR_SECONDS*np.maximum(-lateral, 0)[:, None]*local, conserve_annual_loss=True)
        monthly = routed['outflow_m3']/MONTH_SECONDS
        for r, values in zip(reaches, monthly):
            r['monthly_flow_m3s'] = values.tolist()
        if hasattr(plan, 'metadata'):
            plan.metadata.setdefault('quality', {}).update(monthly_routing_residual_max_m3=float(np.max(np.abs(routed['residual_m3']))),
                monthly_mean_discharge_error_m3s=float(np.max(np.abs(monthly@MONTH_SECONDS/YEAR_SECONDS-q))))
        lengths, slopes, obstacle = [], [], []
        boundary = np.zeros((3, n, 12), bool)
        lakes = {str(lake['id']): lake for lake in getattr(plan, 'local_lakes', ())}
        for i, r in enumerate(reaches):
            xy, z = np.asarray(r['xy']), np.asarray(r['water_m'])
            delta = np.diff(xy, axis=0)
            if world.topology == 'sphere':
                period = world.bounds[2]-world.bounds[0]
                delta[:, 0] = (delta[:, 0]+period*.5)%period-period*.5
                delta[:, 0] *= np.cos((xy[:-1, 1]+xy[1:, 1])*.5/(period/(2*np.pi)))
            distance = np.linalg.norm(delta, axis=1)
            length = float(distance.sum())
            lengths.append(length)
            slopes.append(max(0., float(z[0]-z[-1]))/max(length, 1.))
            obstacle.append(bool(np.any((z[:-1]-z[1:] > 2.) & ((z[:-1]-z[1:])/np.maximum(distance, 1.) > .03))))
            if downstream[i] < 0:
                kind = r.get('terminal_kind') or ''
                parent = parents[i]
                seen = set()
                while kind.startswith('lake:'):
                    lid = kind[5:]
                    if lid in seen or lid not in lakes:
                        kind = 'terminal'
                        break
                    seen.add(lid)
                    lake = lakes[lid]
                    kind = lake.get('exit_kind') or ('terminal' if lake.get('outlet_cell', -1) == -1 else 'edge')
                if kind == 'sea':
                    boundary[:, i] = True
                elif kind == 'terminal':
                    continue
                elif kind.startswith('parent_lake:'):
                    lake = world.lakes[int(kind.split(':', 1)[1])]
                    parent = None if lake['closed'] else lake.get('exit_reach')
                    if parent is not None:
                        flags = world.reaches[int(parent)].get('navigation_months')
                        if flags:
                            boundary[:, i] = np.asarray(flags, bool)
                elif parent is not None:
                    flags = world.reaches[int(parent)].get('navigation_months')
                    if flags:
                        boundary[:, i] = np.asarray(flags, bool)
        # The existing solver accepts seasonal outlet flags. Run each vessel
        # with its corresponding parent-route condition (three small passes).
        from terrain_hydrology_balance import VESSELS
        flags = []
        for v, vessel in enumerate(VESSELS):
            nav = navigability(downstream, lengths, slopes, monthly,
                outlet_navigable=boundary[v], obstacle=obstacle, vessels=(vessel,))
            flags.append(nav['navigable'][0])
        flags = np.asarray(flags)
        for i, r in enumerate(reaches):
            count = flags[:, i].sum(1)
            r.update(length_m=lengths[i], slope=slopes[i],
                     min_month_flow_m3s=float(monthly[i].min()), navigation_months=flags[:, i].astype(int).tolist(),
                     navigation=dict(vessels=[v['name'] for v in VESSELS],
                        year_round_class=max((v for v in range(3) if count[v] == 12), default=-1),
                        seasonal_class=max((v for v in range(3) if count[v] > 0), default=-1)))

    @staticmethod
    def _aligned(shape, bounds, lod, coordinates=None):
        """Ordinary display grid made of complete global 240 m blocks."""
        spacing = (bounds[2]-bounds[0])/shape[1]
        factor = max(1, int(round(240./spacing)))
        return (coordinates is None and 0 <= lod <= 3 and abs(factor*spacing-240.) < 1e-6 and
                all(n % factor == 0 for n in shape) and
                max(abs(bounds[i]/240.-round(bounds[i]/240.)) for i in (0, 1)) < 1e-6)

    @staticmethod
    def _parent_surface(parts, elevation, bounds, lod, coordinates=None, world_bounds=None, reference=None):
        from terrain_hydrology_geometry import cell_centres, sample_bilinear, block_mean, mean_preserving_correction
        spacing = (bounds[2]-bounds[0])/elevation.shape[1]
        factor = max(1, int(round(240./spacing)))
        if HydrologyService._aligned(elevation.shape, bounds, lod, coordinates):
            target = block_mean(elevation, factor)
            x, y = np.meshgrid(*cell_centres(target.shape, bounds))
            if world_bounds:
                x = (x-world_bounds[0]) % (world_bounds[2]-world_bounds[0])+world_bounds[0]
            for part in parts:
                plan, box = part['plan'], part['bounds']
                take = (x >= box[0]) & (x < box[2]) & (y >= box[1]) & (y < box[3])
                if take.any():
                    target[take] = sample_bilinear(plan.routing_height_m, plan.bounds, x[take], y[take])
            # Three base-cell support fits the existing 24 native-sample halo.
            # No piecewise-constant exact remainder: that would draw 240 m
            # terraces. Channel carving is applied after this smooth bridge.
            out = elevation+mean_preserving_correction(target-block_mean(elevation, factor),
                                                       factor, iterations=2, exact=False)
            return out, dict(max_mean_error_m=float(np.max(np.abs(target-block_mean(out, factor)))))
        x, y = np.meshgrid(*cell_centres(elevation.shape, bounds))
        if coordinates:
            x, y = coordinates(x, y)
        if world_bounds:
            x = (x-world_bounds[0]) % (world_bounds[2]-world_bounds[0])+world_bounds[0]
        # The decoder's local mean follows its own raw chart's Base. Replace
        # that Base by the canonical routing parent; keep the decoder residual.
        out = np.asarray(elevation, np.float64).copy()
        legacy = False
        for part in parts:
            box, plan = part['bounds'], part['plan']
            take = (x >= box[0]) & (x < box[2]) & (y >= box[1]) & (y < box[3])
            delta = plan.metadata.get('chart_correction_m', {}).get(reference)
            if delta is None:
                # Callback without raw chart references: canonical Base is
                # only an approximation of this decoder's baseline.
                delta, legacy = plan.metadata.get('base_correction_m'), True
            if take.any() and delta is not None:
                out[take] += sample_bilinear(delta, HydrologyService._correction_bounds(plan), x[take], y[take])
        return out, dict(max_mean_error_m=None, composition=
            'canonical-base correction (no raw chart reference)' if legacy else
            f'{reference} chart Base replaced by parent; rotated or sub-30m grid')

    @staticmethod
    def _chart_geometry(reach, settings, center):
        from terrain_geometry import world_bounds
        from terrain_polar import to_chart
        period = world_bounds(settings)[2]-world_bounds(settings)[0]
        out = dict(reach)
        for key in ('xy', 'belt_xy'):
            if key not in reach:
                continue
            points = np.asarray(reach[key])
            x, y = to_chart(points[:, 0], points[:, 1], settings)
            x = np.unwrap(x, period=period)
            x += period*np.round((center-(x.min()+x.max())*.5)/period)
            out[key] = np.stack((x, y), -1)
        return out

    @staticmethod
    def _seasonal_geometry(geom, month):
        if geom['flow_m3s'] <= 1e-12:
            result = dict(geom)
            for key in ('width_m', 'belt_width_m', 'depth_m'):
                if key in geom:
                    result[key] = np.zeros_like(geom[key])
            return result
        if not month or len(geom.get('monthly_flow_m3s', ())) != 12:
            return geom
        result = dict(geom)
        q = max(0., geom['monthly_flow_m3s'][month-1])
        ratio = q/max(geom['flow_m3s'], 1e-12)
        for key in ('width_m', 'belt_width_m'):
            if key in geom:
                result[key] = np.asarray(geom[key])*min(1.8, ratio**.26)
        result['depth_m'] = np.asarray(geom['depth_m'])*ratio**.4
        result['flow_m3s'] = q
        return result

    def physical(self, entry, elevation, bounds, *, lod=0, month=0, chart=None, detail=False, cosmetics=(),
                 render_lakes=False, min_lake_area_m2=1e6):
        """Condition a display copy, never the raw neural store. Same world curve
        and still-water ownership feed CPU, WebGPU and terrain geometry.

        Default: source courses painted as coverage, terrain uncarved, no
        neural work. ``detail`` composes regional Base packages and carves
        their resolved channels (lod <= 3 schedules packages)."""
        if entry['status'] != 'ready':
            return elevation, None, dict(hydro_state=entry['status'])
        cosmetics = self.parse_cosmetics(cosmetics)
        render_lakes = bool(render_lakes) and not detail
        min_lake_area_m2 = float(min_lake_area_m2)
        if not np.isfinite(min_lake_area_m2) or min_lake_area_m2 < 0:
            raise ValueError('Invalid displayed lake minimum area')
        from terrain_hydrology_geometry import (prepare_water_grid, rasterize_water, condition_surface,
                                                cell_centres, sample_bilinear, surface_envelopes)
        world = entry['world']
        begun = time.perf_counter()
        settings = entry['source']['settings']
        coordinates, query_bounds, query_boxes = None, bounds, [bounds]
        if chart == 'polar':
            from terrain_polar import from_chart
            coordinates = lambda x, y: from_chart(x, y, settings)
            x, y = np.meshgrid(*cell_centres(elevation.shape, bounds))
            gx, gy = coordinates(x, y)
            query_bounds = (gx.min(), gy.min(), gx.max(), gy.max())
            query_boxes = [query_bounds]
            if np.ptp(gx) > (world.bounds[2]-world.bounds[0])*.5:
                middle = (world.bounds[0]+world.bounds[2])*.5
                west, east = gx < middle, gx >= middle
                query_boxes = [(world.bounds[0], gy.min(), gx[west].max(), gy.max()),
                               (gx[east].min(), gy.min(), world.bounds[2], gy.max())]
        selected, pending = {}, 0
        cell_m = max((bounds[2]-bounds[0])/elevation.shape[1],
                     (bounds[3]-bounds[1])/elevation.shape[0])
        # Coarser than the 240 m Base: ready regional courses replace the
        # source preview, but the display keeps its own (e.g. 7680 m) cells.
        # No parent composition and no carving at a scale it cannot resolve.
        coarse = lod > 3
        # Only the fallback composition needs the requested chart's raw Base.
        reference = None if coarse or self._aligned(elevation.shape, bounds, lod, coordinates) else \
            'polar' if chart == 'polar' else 'ordinary'
        for box in query_boxes if detail else ():
            support = self._river_query_bounds(world, box, cell_m, entry.get('river_max_flow_m3s'))
            ready, waiting = self._regions(entry, support, lod, reference)
            selected.update((p['bounds'], p) for p in ready)
            pending += waiting
        parts = [selected[key] for key in sorted(selected)]
        if parts and not coarse:
            elevation, parent_info = self._parent_surface(parts, elevation, bounds, lod, coordinates,
                world_bounds=world.bounds if world.topology == 'sphere' else None, reference=reference)
        else:
            parent_info = dict(composition='none (coarse view keeps its own cells)') if parts else {}
        with self.lock:
            if entry['water_grid'] is None:
                entry['water_grid'] = prepare_water_grid(world.height_m, world.bounds,
                    lakes=world.lakes if DISPLAY_LAKES else (), lake_ids=world.lake_ids, ocean=world.ocean,
                    wrap_x=world.topology == 'sphere')
            # Separate display-only authority. Never change the historical
            # grid or enable the deferred small Base depression raster.
            if render_lakes and entry.get('v2_lake_water_grid') is None:
                entry['v2_lake_water_grid'] = prepare_water_grid(world.height_m, world.bounds,
                    lakes=world.lakes, lake_ids=world.lake_ids, ocean=world.ocean,
                    wrap_x=world.topology == 'sphere')
            water_grid = entry['v2_lake_water_grid'] if render_lakes else entry['water_grid']
        # While the first base package is building, the viewer already draws
        # the connected source-vector preview. Do not rasterize those same
        # long curves independently in every fine display tile as well.
        vector_preview = lod <= 3 and pending > 0 and not parts
        chosen = set() if vector_preview else set(i for box in query_boxes for i in self._chosen(entry, box, lod=lod))
        reaches = [self._seasonal_geometry(self._geometry(entry, i, cosmetics), month) for i in sorted(chosen)]
        def raster(geometry, grid, lakes):
            if chart == 'polar':
                geometry = [self._chart_geometry(r, settings, (bounds[0]+bounds[2])*.5) for r in geometry]
            source_coordinates = coordinates
            if source_coordinates is None and world.topology == 'sphere' and (bounds[0] < world.bounds[0] or bounds[2] > world.bounds[2]):
                source_coordinates = lambda x, y: ((x-world.bounds[0]) % (world.bounds[2]-world.bounds[0])+world.bounds[0], y)
            return rasterize_water(geometry, lakes, elevation.shape, bounds,
                water_grid=grid, terrain=elevation, sphere_radius_m=self.radius(world),
                wrap_period_m=world.bounds[2]-world.bounds[0] if self.radius(world) else None,
                still_coordinates=source_coordinates)
        fields = raster(reaches, water_grid, world.lakes)
        if render_lakes:
            lake = fields['class'] == 2
            index = fields['lake_index']
            areas = np.asarray([max(0., float(l.get('area_m2', 0.))) for l in world.lakes])
            allowed = np.zeros(elevation.shape, bool)
            known = (index >= 0) & (index < len(areas))
            allowed[known] = areas[index[known]] >= min_lake_area_m2
            # Source ownership may identify a lake under a fine neural ridge.
            # Do not flood that ridge, even by a zero-depth blue overlay.
            allowed &= np.isfinite(fields['water_level_m']) & (elevation < fields['water_level_m'])
            rejected = lake & ~allowed
            fields['still_coverage'][rejected] = 0.
            fields['still_level_m'][rejected] = np.nan
            fields['lake_index'][rejected] = -1
            river = (fields['river_coverage'] > fields['still_coverage']) & (fields['river_coverage'] > 1e-4)
            fields['class'] = np.where(river, 3, np.where(rejected, 0, fields['class'])).astype(np.uint8)
            fields['coverage'] = np.where(fields['class'] > 0, np.maximum(fields['river_coverage'], fields['still_coverage']), 0.)
            fields['water_level_m'] = np.where(river, fields['river_level_m'],
                np.where(rejected, np.nan, fields['water_level_m']))
        # The source network is an overview/coverage authority, not permission
        # to excavate a 30 m decoder down to a 19.5 km source-cell elevation.
        fields['resolved'].fill(0.)
        if parts:
            x, y = np.meshgrid(*cell_centres(elevation.shape, bounds))
            if coordinates:
                x, y = coordinates(x, y)
            if world.topology == 'sphere':
                x = (x-world.bounds[0]) % (world.bounds[2]-world.bounds[0])+world.bounds[0]
            # Compose river support before applying core ownership. A curve
            # generated in the left package can meander into the right one.
            # Only parent terrain and local still water retain core masks.
            union = raster([], entry['water_grid'], world.lakes)
            still_kind = union['class'].copy()
            owned = np.zeros(elevation.shape, bool)
            parent = np.full(elevation.shape, np.nan)
            river_keys = ('river_coverage', 'distance_m', 'width_m', 'bankfull_depth_m',
                          'river_level_m', 'resolved', 'reach_index', 'flow_x', 'flow_y')
            for part in parts:
                regional = [self._seasonal_geometry(r, month) for r in self._part_geometry(entry, part, cosmetics)
                            if not coarse or self._coarse_visible(world, r, lod)]
                box = part['bounds']
                take = (x >= box[0]) & (x < box[2]) & (y >= box[1]) & (y < box[3])
                owned |= take
                fine = raster(regional, None, [])
                better = ((fine['river_coverage'] > union['river_coverage']) |
                    ((fine['river_coverage'] == union['river_coverage']) &
                     (fine['distance_m'] < union['distance_m'])))
                for key in river_keys:
                    union[key][better] = fine[key][better]
                if coarse:
                    continue
                plan = part['plan']
                if take.any():
                    # Canonical routing parent S, the surface the 240 m route
                    # and profile charged their shared allowance against.
                    halo = plan.metadata.get('routing_halo_m')
                    parent[take] = sample_bilinear(plan.routing_height_m if halo is None else halo,
                                                   self._correction_bounds(plan), x[take], y[take])
                local = raster([], part['local_water_grid'], part['local_lakes'])
                wet = take & (local['class'] > 0) & (local['coverage'] > union['still_coverage'])
                still_kind[wet] = local['class'][wet]
                for key, source in (('still_coverage', 'coverage'), ('still_level_m', 'water_level_m'),
                                    ('lake_index', 'lake_index')):
                    union[key][wet] = local[source][wet]
            river = (union['river_coverage'] > union['still_coverage']) & (union['river_coverage'] > 1e-4)
            still = ~river & (union['still_coverage'] > 1e-4)
            union['class'] = np.where(river, 3, np.where(still, still_kind, 0)).astype(np.uint8)
            union['coverage'] = np.where(union['class'] > 0,
                np.maximum(union['river_coverage'], union['still_coverage']), 0.)
            union['water_level_m'] = np.where(river, union['river_level_m'],
                np.where(still, union['still_level_m'], np.nan))
            # Outside ready cores retain the global preview except wherever
            # neighbouring regional channel/bank support actually contributes.
            replace = owned | (union['width_m'] > 0)
            for key in fields:
                if np.shape(fields[key]) == elevation.shape:
                    fields[key][replace] = union[key][replace]
            if coarse:
                fields['resolved'].fill(0.)
        # Match the profile's bounded conflict incision, then allow the bed
        # below that surface. A smaller decoder budget would hide valid cuts.
        # The shared allowance P is charged to the canonical routing parent S
        # (what S must lose to reach the channel envelope U, at most P), as in
        # the profile and regional route, not to decoder relief above S. The
        # decoder may remove at most DECODER_RESIDUAL_CUT_M more:
        # B = max(P, clip(S - U, 0, P) + R). Monotone (never below P); only
        # acts where U is finite. Larger residuals or parent excess stay dry
        # and reported.
        cut_budget = REGIONAL_VALLEY_CUT_M+np.minimum(fields['bankfull_depth_m'], 30.)
        channel = {}
        if parts and not coarse:
            known = np.isfinite(parent)
            _, envelope = surface_envelopes(np.where(known, parent, 0.), fields)
            with np.errstate(invalid='ignore'):
                excess = np.where(known & np.isfinite(envelope), parent-envelope, 0.)
            channel = dict(parent='routing parent', residual_allowance_m=DECODER_RESIDUAL_CUT_M,
                           parent_excess_cells=int(np.count_nonzero((excess > cut_budget) & (fields['resolved'] > .5))))
            cut_budget = np.where(known, np.maximum(cut_budget, np.clip(excess, 0., cut_budget)+DECODER_RESIDUAL_CUT_M),
                                  cut_budget)
        # A resolved section dry inside its bankfull half-width is a channel
        # blocker; antialiased edge pixels on the prescribed bank are not.
        bed = ((fields['class'] == 3) & (fields['resolved'] > .5) &
               (fields['distance_m'] < fields['width_m']*.5))
        if render_lakes:
            # Pure appearance: a lake may occupy existing low terrain only.
            # Both budgets and the desired extra bed depth are exactly zero.
            conditioned, fields, info = condition_surface(elevation, fields,
                max_cut_m=0., max_still_cut_m=0., lake_depth_m=0., max_bank_fill_m=0.)
        else:
            conditioned, fields, info = condition_surface(elevation, fields,
                max_cut_m=cut_budget, max_still_cut_m=2.)
        channel['blocked_bed_cells'] = int(np.count_nonzero(bed & (fields['class'] != 3)))
        rgba = np.empty((*elevation.shape, 4), np.float32)
        rgba[..., 0] = np.nan_to_num(fields['water_level_m'], nan=0.)
        rgba[..., 1], rgba[..., 2], rgba[..., 3] = fields['coverage'], fields['class'], 1.
        report = dict(hydro_state='regional-pending' if pending else 'ready', hydro_id=entry['key'], hydro_month=month,
                      hydro_detail=bool(detail), hydro_cosmetics=','.join(sorted(cosmetics)),
                      hydro_render_lakes=render_lakes, hydro_min_lake_area_m2=min_lake_area_m2 if render_lakes else None,
                      hydro_max_cut_m=round(info['max_cut_m'], 3),
                      hydro_unresolved_river_cells=info['unresolved_river_cells'],
                      # Source courses are an overview: never carved, so the
                      # viewer's vector network is their authoritative display.
                      # Their coverage is still rasterized for CPU tiles.
                      hydro_vector_preview=bool(vector_preview or not detail),
                      hydro_regions=len(parts), hydro_pending_regions=pending, hydro_parent=parent_info,
                      hydro_channel=channel,
                      hydro_seconds=round(time.perf_counter()-begun, 4))
        return conditioned.astype(np.float32), rgba, report

    def catchment(self, entry, *, reach_id=None, basin_id=None):
        """Full upstream watershed, or a complete terminal basin. Boundaries
        have the source-grid resolution; a fine bend does not invent a divide."""
        if entry['status'] != 'ready':
            return self.state(entry)
        from terrain_hydrology_geometry import lake_rings
        world = entry['world']
        if reach_id is not None:
            reach_id = int(reach_id)
            if not 0 <= reach_id < len(world.reaches):
                raise ValueError('Unknown river reach')
            key = ('reach', reach_id)
        else:
            basin_id = int(basin_id)
            if not 0 <= basin_id < len(world.basins['root_cell']):
                raise ValueError('Unknown drainage basin')
            key = ('basin', basin_id)
        with self.lock:
            if key in entry['catchments']:
                entry['catchments'].move_to_end(key)
                return entry['catchments'][key]
        if reach_id is not None:
            reach = world.reaches[reach_id]
            cells = world.upstream_cells(np.asarray(reach['cells'])[-1:])
            basin_id = reach['basin_id']
        else:
            cells = np.flatnonzero((world.basin_ids == basin_id)&~world.ocean)
            reach = None
        h, w = world.shape
        r, c = np.divmod(cells, w)
        rings = []
        if len(cells):
            r0, r1, c0, c1 = max(0, r.min()-1), min(h, r.max()+2), max(0, c.min()-1), min(w, c.max()+2)
            mask = np.zeros((r1-r0, c1-c0), bool)
            mask[r-r0, c-c0] = True
            dx, dy = world._spacing()
            bounds = (world.bounds[0]+c0*dx, world.bounds[1]+r0*dy,
                      world.bounds[0]+c1*dx, world.bounds[1]+r1*dy)
            rings = [ring['xy'] for ring in lake_rings(mask, bounds)]
        result = _plain(dict(status='ready', hydro_id=entry['key'], reach_id=reach_id,
            basin_id=basin_id, basin_key=str(world.basins['key'][basin_id]),
            area_m2=float(world.row_area_m2[r].sum()), source_cells=len(cells),
            resolution_m=list(world._spacing()), rings=rings,
            annual_flow_m3s=reach['flow_m3s'] if reach else world.basins['outflow_m3s'][basin_id],
            monthly_flow_m3s=reach.get('monthly_flow_m3s') if reach else None,
            outlet_kind=world.basins['kind_names'][world.basins['kind'][basin_id]]))
        with self.lock:
            entry['catchments'][key] = result
            while len(entry['catchments']) > 8:
                entry['catchments'].popitem(last=False)
        return result

    @staticmethod
    def _outlet_kind(world, basin_id):
        basins = getattr(world, 'basins', None)
        basins = basins if isinstance(basins, dict) else {}
        kinds, names = basins.get('kind'), basins.get('kind_names')
        if basin_id is None or kinds is None or names is None or not 0 <= int(basin_id) < len(kinds):
            return 'unresolved'
        return str(names[int(kinds[int(basin_id)])])

    def _hierarchy(self, entry, raw, source_reach):
        """Physical hierarchy and regime metadata of one delivered record.

        Only quantities the network already has: end and basin outlet kinds,
        catchment and local specific yield, and the synthetic monthly
        hydrograph. ``dry_months`` counts months below DRY_FLOW_M3S (the
        ``flow_statistics`` default); ``low_flow_months`` counts months below
        1 % of the annual mean, which a large river can have while flowing.
        Cached on the stored course (independent of view, LOD and month)."""
        cached = raw.get('_hierarchy')
        if cached is not None:
            return cached
        world = entry['world']
        q = max(float(raw.get('flow_m3s') or 0.), 0.)
        area = float(raw.get('area_m2') or 0.)
        monthly = np.asarray(raw.get('monthly_flow_m3s', ()), np.float64)
        if monthly.size == 12:
            dry = int(np.count_nonzero(monthly < DRY_FLOW_M3S))
            low = int(np.count_nonzero(monthly < .01*q)) if q > 0 else 12
        else:
            dry = 12 if q < DRY_FLOW_M3S else 0
            low = 0
        xy = np.asarray(raw['xy'])
        end = raw.get('end')
        cached = dict(strahler_scale=raw.get('strahler_scale', 'source' if source_reach else 'base-240m'),
                      end_kind=(SOURCE_END_KINDS.get(end, end) if source_reach else raw.get('end_kind', 'terminal')),
                      outlet_kind=self._outlet_kind(world, raw.get('basin_id')),
                      catchment_runoff_mm_year=q*YEAR_SECONDS*1000./area if area > 0 else None,
                      local_runoff_mm_year=(float(self._local_runoff(entry, xy[len(xy)//2])[0])
                                            if hasattr(world, 'discharge_m3s') else None),
                      dry_months=dry, low_flow_months=low,
                      regime='perennial' if dry == 0 else 'dry' if dry == 12 else 'intermittent')
        raw['_hierarchy'] = cached
        return cached

    def _source_lake_boxes(self, entry):
        """Map bounds of every source lake (cell edges plus one cell), cached."""
        boxes = entry.get('lake_boxes')
        if boxes is None:
            world = entry['world']
            h, w = world.shape
            dx, dy = world._spacing()
            boxes = np.zeros((len(world.lakes), 4))
            for k, lake in enumerate(world.lakes):
                cells = np.asarray(lake['cells'], np.int64)
                if not cells.size:
                    boxes[k] = np.inf
                    continue
                rows, cols = cells//w, cells % w
                boxes[k] = (world.bounds[0]+max(0, int(cols.min())-1)*dx, world.bounds[1]+max(0, int(rows.min())-1)*dy,
                            world.bounds[0]+min(w, int(cols.max())+2)*dx, world.bounds[1]+min(h, int(rows.max())+2)*dy)
            entry['lake_boxes'] = boxes
        return boxes

    def _vector_lakes(self, entry, bounds, parts, lod, tolerance):
        """Lake polygons of the view: local Base ponds of ready packages, and
        source lakes (Base pieces in ready cores, coarse shoreline elsewhere).

        Fixed geographic geometry, cached once per lake (and per ready-core set
        for source lakes) and simplified per LOD. Polygons smaller than one
        display pixel (bounding box) are left out; their connectors keep the rivers continuous."""
        from terrain_hydrology_lakes import (geometry_rings, physical_area, source_lake_geometry,
                                             combine_source_lake)
        world = entry['world']
        radius = self.radius(world)
        pixel = 30.*2**lod
        visible = lambda bb: (_world_intersects(world, bb, bounds) and (bb[2]-bb[0])*(bb[3]-bb[1]) >= pixel*pixel)
        lakes = []
        for part in parts:
            for lid, lake in part.get('vector_lakes', {}).items():
                bb = lake['geometry'].bounds
                if not visible(bb):
                    continue
                cache = lake.setdefault('rings', {})
                if tolerance not in cache:
                    cache[tolerance] = geometry_rings(lake['geometry'], tolerance)
                if not cache[tolerance]:
                    continue
                record = {k: v for k, v in lake.items() if k not in ('geometry', 'rings')}
                record.update(rings=cache[tolerance], bbox=list(bb), outlet_kind=self._outlet_kind(world, lake['basin_id']))
                lakes.append(record)
        boxes = self._source_lake_boxes(entry)
        period = world.bounds[2]-world.bounds[0]
        hit = np.zeros(len(boxes), bool)
        for s in ((-period, 0., period) if world.topology == 'sphere' else (0.,)):
            hit |= ((boxes[:, 0]+s <= bounds[2]) & (boxes[:, 2]+s >= bounds[0]) &
                    (boxes[:, 1] <= bounds[3]) & (boxes[:, 3] >= bounds[1]))
        store = entry.setdefault('source_lake_display', OrderedDict())
        for L in np.flatnonzero(hit):
            L = int(L)
            lake = world.lakes[L]
            if float(lake.get('area_m2', 0.)) <= 0 or float(lake.get('max_depth_m', 0.)) <= 0:
                continue
            cores = tuple(sorted(tuple(p['bounds']) for p in parts if _intersects(p['bounds'], boxes[L])))
            key = (L, cores)
            cached = store.get(key)
            if cached is None:
                if L not in entry['lake_geometry']:
                    entry['lake_geometry'][L] = source_lake_geometry(world, lake)
                pieces = [p['source_lake_pieces'][L] for p in parts
                          if tuple(p['bounds']) in cores and L in p.get('source_lake_pieces', {})]
                base, rest = combine_source_lake(entry['lake_geometry'][L], [x['geometry'] for x in pieces], cores)
                cached = store[key] = dict(base=base, rest=rest, rings={},
                                           depth=max((x['max_depth_m'] for x in pieces), default=0.))
                while len(store) > 512:
                    store.popitem(last=False)
            store.move_to_end(key)
            if tolerance not in cached['rings']:
                cached['rings'][tolerance] = (geometry_rings(cached['base'], tolerance),
                                              geometry_rings(cached['rest'], tolerance))
            for resolution, geom, rings, depth in (
                    ('base-240m', cached['base'], cached['rings'][tolerance][0], cached['depth']),
                    ('source', cached['rest'], cached['rings'][tolerance][1], float(lake['max_depth_m']))):
                if geom.is_empty or not rings or not visible(geom.bounds):
                    continue
                lakes.append(dict(id=L, kind='source', resolution=resolution, rings=rings,
                                  level_m=float(lake['level_m']), area_m2=physical_area(geom, radius),
                                  max_depth_m=depth, closed=bool(lake['closed']),
                                  inflow_m3s=lake.get('inflow_m3s'), outflow_m3s=lake.get('outflow_m3s'),
                                  evaporation_m3s=lake.get('evaporation_m3s'), basin_id=int(lake['basin_id']),
                                  outlet_kind=self._outlet_kind(world, lake['basin_id']),
                                  exit_reach=lake.get('exit_reach'), bbox=list(geom.bounds)))
        return lakes

    def features(self, entry, bounds, *, lod=4, month=0, max_reaches=512, detail=False, cosmetics=()):
        """Source overview and ready cached-Base courses, simplified per LOD.
        Default routing consumes existing Base, without new neural work."""
        result = self.state(entry)
        if result['status'] != 'ready':
            return result
        cosmetics = self.parse_cosmetics(cosmetics)
        world = entry['world']
        from terrain_hydrology_geometry import simplify_reach, _simplify
        if detail:
            (parts, pending), unavailable = self._regions(entry, bounds, lod), 0
        else:
            parts, pending, unavailable = self._vector_regions(entry, bounds, lod)
        chosen = self._chosen(entry, bounds, max_reaches)
        reaches = []
        # Coarse views simplify the same ready regional course; they add no
        # tributary finer than the source network's own channel threshold.
        candidates = [(g, False) for part in parts for g in self._part_geometry(entry, part, cosmetics)
                      if (lod <= 3 or self._coarse_visible(world, g))
                      and _world_intersects(world, _bbox(g['xy']), bounds)]
        candidates.extend((self._geometry(entry, i, cosmetics), True) for i in chosen
                          if world.reaches[i]['flow_m3s'] > 1e-12)
        period = world.bounds[2]-world.bounds[0]
        frontier = [((b[0]+s, b[1], b[2]+s, b[3]), s, part['plan']) for part in parts for b in [part['bounds']]
                    for s in ((-period, 0., period) if world.topology == 'sphere' else (0.,))]
        boxes = [box for box, _, _ in frontier]
        overview, source_records, regional_records = 0, {}, {}
        tolerance = max(1., 30*2**lod*.2)
        with self.lock:
            for raw, coarse in candidates:
                if not coarse and 'display_xy' in raw and len(raw['display_xy']) < 2:
                    continue
                geom = simplify_reach(raw, tolerance)
                fragments = _outside_pieces(geom['xy'], geom['water_m'], boxes) if coarse else [(geom['xy'], geom['water_m'])]
                if not coarse and 'display_xy' in raw:
                    xy = raw['display_xy']
                    pins = [p for p in raw.get('pins', ()) if p < len(xy)]
                    indices = _simplify(xy, tolerance, pins)
                    fragments = [(xy[indices], raw['display_water_m'][indices])]
                if coarse and boxes:
                    fragments = self._bridge(fragments, raw['id'], frontier)
                if not fragments and not coarse:
                    continue
                overview += int(coarse and bool(fragments))
                keep = ('id', 'key', 'source', 'target', 'basin_id', 'parent_id', 'kind', 'members',
                        'downstream_id', 'flow_m3s', 'area_m2', 'strahler', 'strahler_partial', 'length_m', 'slope',
                        'monthly_flow_m3s', 'min_month_flow_m3s', 'navigation_months', 'navigation')
                record = {k: raw[k] for k in keep if k in raw}
                record.update(xy=geom['xy'], water_m=geom['water_m'],
                              width_m=float(np.mean(geom['width_m'])),
                              depth_m=float(np.mean(geom['depth_m'])),
                              morphology=geom.get('morphology', 'river'))
                record.update(self._hierarchy(entry, raw, coarse))
                nav = record.get('navigation', {})
                capability = (max((v for v, flags in enumerate(record.get('navigation_months', ()))
                                   if flags[month-1]), default=-1) if month else nav.get('year_round_class', -1))
                names = nav.get('vessels', ())
                record['navigable'] = capability >= 0
                record['navigation'] = dict(nav, label=(names[capability] if capability >= 0 else 'no continuous route'))
                (source_records if coarse else regional_records)[raw['id']] = record
                for xy, water in fragments:
                    reaches.append(dict(record, xy=xy, water_m=water))
            lakes = self._vector_lakes(entry, bounds, parts, lod, tolerance)
            shown = {(lake['kind'], lake['id']) for lake in lakes}
            delivered = set()
            for rid, link in entry.get('connectors', {}).items():
                parent = source_records.get(rid)
                if parent is None or link['reach_target'] not in source_records:
                    continue
                if link['target_kind'] == 'connector' and link['target'] not in delivered:
                    continue
                pieces = (_outside_pieces(link['xy'], link['water_m'], boxes) if parts
                          else [(link['xy'], link['water_m'])])
                for xy, water in pieces:
                    reaches.append(dict(parent, xy=xy, water_m=water, parent_id=rid, continues_id=rid,
                                        display_downstream_id=(link['target'] if link['target_kind'] == 'connector'
                                                               else link['reach_target']),
                                        morphology='connector', connector=link['kind'], lake_id=link.get('lake_id'),
                                        lake_shown=('source', link.get('lake_id')) in shown))
                delivered.add(rid)
            # Local pond crossings, for the rivers delivered from each package.
            for part in parts:
                for rid, link in part.get('lake_connectors', {}).items():
                    parent = regional_records.get(rid)
                    if parent is None or not _world_intersects(world, _bbox(link['xy']), bounds):
                        continue
                    reaches.append(dict(parent, xy=link['xy'], water_m=link['water_m'], continues_id=rid,
                                        display_downstream_id=(link['target'] if link['target'] is not None
                                                               else parent.get('downstream_id')),
                                        morphology='connector', connector='lake', lake_id=link['lake_id'],
                                        lake_shown=('local', link['lake_id']) in shown))
        hidden = {}
        for part in parts:
            for reason in part.get('lake_hidden', {}).values():
                hidden[reason] = hidden.get(reason, 0)+1
        result.update(reaches=reaches, lakes=lakes, query_bounds=list(bounds),
                      detail=bool(detail), cosmetics=sorted(cosmetics),
                      resolution=('mixed' if overview else 'base-240m') if parts else 'source-network', month=month, lod=lod,
                      status='building' if pending else 'ready', phase='base detail' if pending else 'ready',
                      regional=dict(ready=len(parts), pending=pending,
                                    quality=[p['plan'].metadata.get('quality') for p in parts],
                                    cached_only=not detail, unavailable=unavailable,
                                    hidden_local_lakes=hidden,
                                    lake_errors=[p['lake_error'] for p in parts if p.get('lake_error')],
                                    errors=[p['error'] for p in list(entry['regions'].values())+
                                            list(entry.get('vector_regions', {}).values()) if p['status']=='error']))
        return _plain(result)

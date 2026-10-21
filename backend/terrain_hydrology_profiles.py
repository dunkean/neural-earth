"""Immutable coarse water profiles shared by every regional crop of a reach.

Coordinates and heights are metres; station is the fractional vertex index of
the original HydroWorld polyline. Sampling and caching belong to the caller.
The valley at a station is the lowest same-basin Coarse sample across the
regional trunk corridor, the band a 240 m trunk can actually follow.
"""
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

PROFILES_VERSION = 'hydro-profiles-v2'


@dataclass(frozen=True)
class CanonicalProfiles:
    profiles: object
    metadata: dict

    def level(self, reach_id, station):
        profile = self.profiles[int(reach_id)]
        return np.interp(station, profile['station'], profile['water_m'])


def _downstream(reach):
    target = reach.get('downstream_id')
    if target is None:
        target = reach.get('hydraulic_downstream_id')
    return None if target is None or target < 0 else int(target)


def corridor_width(world, spacing_m=7680.):
    """Half-width of the regional trunk corridor: 0.75 source cell, as in regions."""
    shape = getattr(world, 'shape', None)
    if shape is None:
        return .5 * spacing_m
    return .75 * max((world.bounds[2] - world.bounds[0]) / shape[1], (world.bounds[3] - world.bounds[1]) / shape[0])


def _stations(world, reach, spacing, half_width):
    xy = np.asarray(reach['xy'], np.float64)
    if len(xy) == 0:
        raise ValueError(f"reach {reach['id']} has no vertices")
    delta = np.diff(xy, axis=0)
    if world.topology == 'sphere':
        latitude = np.pi / 2 - (xy[:, 1] - world.bounds[1]) * np.pi / (world.bounds[3] - world.bounds[1])
        # Upper bound the longitude metric along a segment, including one
        # crossing the equator, so physical sample spacing never exceeds it.
        lat = np.minimum(np.abs(latitude[:-1]), np.abs(latitude[1:]))
        lat = np.where(latitude[:-1] * latitude[1:] <= 0, 0., lat)
        delta = delta.copy()
        delta[:, 0] *= np.maximum(np.cos(lat), 0.0)
    count = np.maximum(1, np.ceil(np.hypot(*delta.T) / spacing).astype(int))
    station = np.r_[np.concatenate([i + np.arange(n) / n for i, n in enumerate(count)])
                    if len(count) else np.zeros(0), float(len(xy) - 1)]
    points = np.column_stack([np.interp(station, np.arange(len(xy)), xy[:, axis]) for axis in (0, 1)])
    tangent = np.gradient(points, axis=0) if len(points) > 1 else np.array([[1., 0.]])
    scale = np.ones(len(points))
    if world.topology == 'sphere':
        lat = np.pi / 2 - (points[:, 1] - world.bounds[1]) * np.pi / (world.bounds[3] - world.bounds[1])
        scale = np.maximum(np.cos(lat), 1e-6)
    tangent[:, 0] *= scale
    normal = np.column_stack([-tangent[:, 1], tangent[:, 0]])
    normal /= np.maximum(np.hypot(*normal.T), 1e-12)[:, None]
    normal[:, 0] /= scale
    # A cross-section over the whole corridor, samples at most a quarter cell
    # apart: the 19.5 km source line can sit kilometres off the Coarse valley.
    offset = np.linspace(-1., 1., 2 * int(np.ceil(half_width / (.25 * spacing))) + 1) * half_width
    candidates = points[:, None, :] + normal[:, None, :] * offset[None, :, None]
    basin = world.sample(candidates[..., 0], candidates[..., 1])['basin_id']
    allowed = np.asarray(basin).reshape(len(points), len(offset)) == reach['basin_id']
    return station, candidates, allowed


def build_profiles(world, reach_ids, sample_height, *, incision_m=30., conflict_incision_m=60., fill_m=2.,
                   fine_valley_m=15., spacing_m=7680., corridor_m=None, known=None):
    """Build requested reaches and their downstream closure with one sampler call.

    The preferred water is a suffix maximum of the corridor valley minus
    ``incision_m``, capped by the descending source profile. Where that would
    stand more than ``fill_m`` above the expected 240 m valley upstream (the
    Coarse corridor minimum less ``fine_valley_m``), a barrier the preferred
    incision cannot cross, the water cuts up to ``conflict_incision_m`` rather
    than damming the whole upstream network. Fixed downstream and lake levels
    take precedence; remaining conflicts stay explicit in diagnostics.
    Profiles in ``known`` (reach id -> profile) are reused as downstream
    levels, never resampled and not returned.
    """
    values = (incision_m, conflict_incision_m, fill_m, fine_valley_m, spacing_m)
    if (not all(np.isfinite(v) for v in values) or incision_m < 0 or conflict_incision_m < incision_m
            or fill_m < 0 or fine_valley_m < 0 or spacing_m <= 0):
        raise ValueError('need finite 0 <= incision_m <= conflict_incision_m, fill_m, fine_valley_m >= 0, spacing_m > 0')
    known = {} if known is None else known
    half = float(corridor_m) if corridor_m is not None else corridor_width(world, spacing_m)
    order, done = [], set()
    for requested in sorted(set(int(v) for v in reach_ids)):
        chain, active = [], set()
        rid = requested
        while rid is not None and rid not in done and rid not in known:
            if rid < 0 or rid >= len(world.reaches):
                raise ValueError(f'unknown reach {rid}')
            if rid in active:
                raise ValueError(f'cyclic downstream graph at reach {rid}')
            active.add(rid)
            chain.append(rid)
            rid = _downstream(world.reaches[rid])
        order.extend(reversed(chain))
        done.update(chain)
    # Water descends from every direct tributary into a reach, so a low
    # tributary valley also bounds its fill. Lake entries keep the lake level.
    upstream = {}
    for reach in world.reaches:
        down = _downstream(reach)
        lake = reach.get('enters_lake')
        if down in done and (lake is None or lake < 0):
            upstream.setdefault(down, []).append(int(reach['id']))
    samples, clouds, offset = {}, [], 0
    for rid in sorted(done | {u for ups in upstream.values() for u in ups}):
        station, points, allowed = _stations(world, world.reaches[rid], spacing_m, half)
        samples[rid] = (station, allowed, offset, points.size // 2)
        clouds.append(points.reshape(-1, 2))
        offset += points.size // 2
    points = np.vstack(clouds) if clouds else np.empty((0, 2))
    heights = np.asarray(sample_height(points[:, 0], points[:, 1]), np.float64).reshape(-1) if len(points) else np.empty(0)
    if heights.size != len(points) or not np.isfinite(heights).all():
        raise ValueError('sample_height must return one finite physical height per point')
    suffix_max = lambda a: np.maximum.accumulate(a[::-1])[::-1]

    def valley_of(rid):
        reach = world.reaches[rid]
        station, allowed, offset, size = samples[rid]
        ground = heights[offset:offset + size].reshape(len(station), -1)
        valley = np.min(np.where(allowed, ground, np.inf), axis=1)
        raw = np.asarray(reach['water_m'], np.float64)
        if len(raw) != len(reach['xy']) or not np.isfinite(raw).all():
            raise ValueError(f'reach {rid} needs one finite water level per original vertex')
        source = np.minimum.accumulate(np.interp(station, np.arange(len(raw)), raw))
        # A shore/pole point can have no same-basin candidate. Retain the
        # source reference there, never borrow a neighbouring basin's valley.
        return station, allowed, np.where(np.isfinite(valley), valley, source), source

    profiles, diagnostics = {}, {}
    for rid in order:
        reach = world.reaches[rid]
        station, allowed, valley, source = valley_of(rid)
        inflow = min((float(valley_of(u)[2].min()) for u in upstream.get(rid, ())), default=np.inf)
        downstream = _downstream(reach)
        lake = reach.get('enters_lake')
        below = profiles.get(downstream, known.get(downstream))
        target = (float(world.lakes[int(lake)]['level_m']) if lake is not None and lake >= 0 else
                  float(below['water_m'][0]) if downstream is not None else float(source[-1]))
        # Every envelope descends, so the result descends. Unless a downstream
        # valley rises more than incision - fine_valley + fill above one
        # upstream, ``preferred`` already fits and is used as is.
        preferred = suffix_max(valley - incision_m)
        fits = np.minimum.accumulate(np.minimum(valley, inflow) - fine_valley_m + fill_m)
        bounded = np.maximum(suffix_max(valley - conflict_incision_m), np.minimum(preferred, fits))
        water = np.maximum(np.minimum(bounded, source), target)
        water[-1] = target
        lake = reach.get('leaves_lake')
        if lake is not None and lake >= 0:
            fixed = float(world.lakes[int(lake)]['level_m'])
            if fixed < target:
                raise ValueError(f'reach {rid} fixed lake source {fixed} below downstream level {target}')
            water = np.minimum(water, fixed)
            water[0] = fixed
        if len(water) == 1 and lake is not None and lake >= 0 and water[0] != target:
            raise ValueError(f'reach {rid} has incompatible levels at one vertex')
        dam = np.maximum(np.minimum(preferred, source), target)
        diagnostics[rid] = dict(max_incision_m=float(np.maximum(valley - water, 0).max(initial=0)),
                                max_fill_m=float(np.maximum(water - valley, 0).max(initial=0)),
                                conflict_lowering_m=float(np.maximum(dam - water, 0).max(initial=0)),
                                source_cap_excess_m=float(np.maximum(water - source, 0).max(initial=0)),
                                basin_fallback_samples=int(np.count_nonzero(~allowed.any(axis=1))))
        for value in (station, water):
            value.flags.writeable = False
        profiles[rid] = MappingProxyType(dict(station=station, water_m=water))
    metadata = dict(version=PROFILES_VERSION, spacing_m=float(spacing_m), incision_m=float(incision_m),
                    conflict_incision_m=float(conflict_incision_m), fill_m=float(fill_m),
                    fine_valley_m=float(fine_valley_m), corridor_m=half,
                    sampled_points=int(len(points)), reused_known=len(set(known) & set(map(int, reach_ids))),
                    reaches=diagnostics,
                    max_incision_m=max((d['max_incision_m'] for d in diagnostics.values()), default=0.),
                    max_fill_m=max((d['max_fill_m'] for d in diagnostics.values()), default=0.))
    return CanonicalProfiles(MappingProxyType(profiles), metadata)

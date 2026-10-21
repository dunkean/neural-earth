"""Vector lakes and lake connectors for display; routing is never changed.

Two lake sources feed ``features()``:

* **Local Base lakes**: the flow-through ponds that a ready regional package
  already found (``plan.local_lakes``). Each one lies wholly inside its core, so it
  cannot be cut at a core edge. A pond is shown only if it is wet and its annual
  inflow covers its net open-water loss.
* **Source lakes**: the global lake records. Inside a ready core the 19.5 km
  source shape is replaced by the connected Base surface below the lake level
  around the lake's own blocks. Elsewhere the coarse source shoreline remains
  the honest fallback.

Connectors draw the routed receiver path across a local lake, so a river stays
continuous whether or not the client shows the lake. Method note:
docs/methods/hydrology-hierarchy-climate-lakes.md.
"""
import numpy as np
import shapely
from shapely.geometry import Polygon, box as _box
from scipy import ndimage

from terrain_hydrology_geometry import lake_rings, lake_shore_rings, x_scale

YEAR_SECONDS = 365*86400.
BLOCK, BASE_M, COARSE_M = 32, 240., 7680.
PRECISION_M = 1e-3     # snaps pieces of adjacent cores onto one shared cut line
WET_FLOW_M3S = 1e-6    # same through-flow rule as the detailed local water grid


def rings_geometry(rings):
    """``lake_rings`` output -> one valid shapely geometry (outer rings minus holes)."""
    outer = [shapely.make_valid(Polygon(r['xy'])) for r in rings if not r['hole'] and len(r['xy']) >= 3]
    holes = [shapely.make_valid(Polygon(r['xy'])) for r in rings if r['hole'] and len(r['xy']) >= 3]
    geom = shapely.union_all(outer) if outer else Polygon()
    if holes:
        geom = geom.difference(shapely.union_all(holes))
    return _polygons(geom)


def _polygons(geom):
    parts = [g for g in shapely.get_parts(geom) if g.geom_type == 'Polygon' and not g.is_empty]
    return shapely.union_all(parts) if parts else Polygon()


def geometry_rings(geom, tolerance_m=0.):
    """Closed rings (first point not repeated). Exterior rings have a positive
    (x, y-south) shoelace area and holes a negative one, as in ``lake_rings``."""
    if geom.is_empty:
        return []
    if tolerance_m > 0:
        geom = _polygons(shapely.simplify(geom, tolerance_m, preserve_topology=True))
    out = []
    for poly in shapely.get_parts(geom):
        if poly.geom_type != 'Polygon' or poly.is_empty:
            continue
        poly = shapely.geometry.polygon.orient(poly, 1.0)
        for ring in (poly.exterior, *poly.interiors):
            xy = np.asarray(ring.coords)[:-1]
            if len(xy) >= 3:
                out.append(xy)
    return out


def physical_area(geom, sphere_radius_m=None):
    """Polygon area in m^2, map area scaled by cos(latitude) at its centroid."""
    if geom.is_empty:
        return 0.
    return float(geom.area*x_scale(np.array([geom.centroid.y]), sphere_radius_m)[0])


# ----------------------------------------------------------------------------
# Local Base lakes of one package

def local_lakes(plan, climate=None):
    """Displayable local ponds of a package, as ``{id: record}``; plus every pond id.

    ``climate(xy) -> (precip, pet, runoff)`` in mm/year per point gives the
    open-water loss ``max(E - (P - R), 0)``, the same rule the source lake
    balance uses. A pond is shown when it is wet (through-flow > 1e-6 m^3/s)
    and its routed through-flow covers that loss over its area. Ponds that
    continue a parent lake keep the parent's budget. Every other pond stays
    routing topology only (the returned ``hidden`` reasons).
    """
    flows = np.asarray(plan.flow_m3s, np.float64).ravel()
    candidates = [lake for lake in plan.local_lakes if not str(lake['id']).startswith('P')]
    shown, hidden = {}, {}
    if not candidates:
        return shown, hidden
    through = []
    for lake in candidates:
        f = flows[np.asarray(lake['cells'], np.int64)]
        through.append(float(np.max(f[np.isfinite(f)], initial=0.)))
    loss = np.zeros(len(candidates))
    if climate is not None:
        precip, pet, runoff = climate(np.array([np.asarray(lake['xy'], np.float64) for lake in candidates]))
        rate = np.maximum(np.asarray(pet)-(np.asarray(precip)-np.asarray(runoff)), 0.)
        loss = rate/1000.*np.array([float(lake['area_m2']) for lake in candidates])/YEAR_SECONDS
    height = np.asarray(plan.routing_height_m, np.float64)
    for lake, q, lost in zip(candidates, through, loss):
        parent = lake.get('parent_lake_id') is not None
        if not parent and q <= WET_FLOW_M3S:
            hidden[lake['id']] = 'dry'
            continue
        if not parent and q < lost:
            hidden[lake['id']] = 'evaporative'
            continue
        rings = lake_shore_rings(lake, dict(height_m=height, bounds=plan.bounds))
        geom = rings_geometry(rings)
        if geom.is_empty:
            hidden[lake['id']] = 'empty'
            continue
        shown[lake['id']] = dict(
            id=lake['id'], kind='local', resolution='base-240m', geometry=geom,
            level_m=float(lake['level_m']), area_m2=float(lake['area_m2']),
            max_depth_m=float(lake['max_depth_m']), closed=False,
            inflow_m3s=q, outflow_m3s=max(q-float(lost), 0.), evaporation_m3s=float(lost),
            basin_id=int(lake['basin_id']), parent_lake_id=lake.get('parent_lake_id'),
            on_trunk=bool(lake.get('on_trunk', False)))
    return shown, hidden


def _nearest_on(xy, point):
    """Closest point of a polyline to ``point``."""
    a, b = xy[:-1], xy[1:]
    d = b-a
    t = np.clip(np.einsum('ij,ij->i', point-a, d)/np.maximum(np.einsum('ij,ij->i', d, d), 1e-12), 0, 1)
    p = a+t[:, None]*d
    return p[int(np.argmin(np.hypot(*(p-point).T)))]


def local_connectors(plan, geometry, *, display_key='xy', limit=4096):
    """Routed courses across every local pond that a displayed river enters.

    From the receiver of the entering record's last cell, follow
    ``plan.receivers`` to the pond's ``outlet_cell``. The course joins:
    * the first vertex of the record that starts at the outlet;
    * otherwise the nearest point of the trunk that owns the outlet cell;
    * otherwise the outlet cell centre.
    Larger inflows are routed first. A smaller one stops on the first cell of
    an earlier course; that cell centre is an exact vertex of the earlier course.
    Flow, ids and budgets are unchanged; a failed walk adds no course.
    Returns ``{entering record id: link}``.
    """
    lakes = {str(lake['id']): lake for lake in plan.local_lakes if not str(lake['id']).startswith('P')}
    if not lakes:
        return {}
    rows, cols = plan.shape
    x0, y0 = plan.bounds[0], plan.bounds[1]
    centre = lambda c: np.array([x0+(c % cols+.5)*BASE_M, y0+(c//cols+.5)*BASE_M])
    receivers = np.asarray(plan.receivers, np.int64).ravel()
    lake_of = {}
    for lid, lake in lakes.items():
        for c in np.asarray(lake['cells'], np.int64):
            lake_of[int(c)] = lid
    starts, trunks = {}, {}
    for g in geometry:
        xy = np.asarray(g.get(display_key, g['xy']))
        cells = g.get('cells')
        if cells is None or len(xy) < 2 or not len(cells):
            continue
        if g.get('kind') == 'trunk':
            for c in np.asarray(cells, np.int64):
                trunks[int(c)] = xy
        else:
            starts[int(np.asarray(cells)[0])] = xy[0]
    entering = []
    for g in geometry:
        kind = str(g.get('terminal_kind') or '')
        if not kind.startswith('lake:') or kind[5:] not in lakes or g.get('cells') is None:
            continue
        xy = np.asarray(g.get(display_key, g['xy']))
        # A course already clipped at a Base sea crossing does not reach the shore.
        if len(xy) < 2 or np.hypot(*(xy[-1]-np.asarray(g['xy'])[-1])) > 1e-6:
            continue
        entering.append(g)
    entering.sort(key=lambda g: (-float(g['flow_m3s']), str(g['id'])))
    owned, links = {}, {}
    for g in entering:
        lid = g['terminal_kind'][5:]
        lake = lakes[lid]
        outlet = int(lake.get('outlet_cell', -1))
        level = float(lake['level_m'])
        c = int(receivers[int(np.asarray(g['cells'])[-1])])
        path, join, target = [], None, None
        for _ in range(limit):
            if c < 0 or c in path:
                break
            if c in owned:
                join, target = centre(c), owned[c]
                break
            path.append(c)
            if c == outlet:
                break
            c = int(receivers[c])
        else:
            continue
        if join is None:
            if not path or path[-1] != outlet or outlet < 0:
                continue                 # closed pond or unverified walk: keep the end
            if outlet in starts:
                join = starts[outlet]
                path = path[:-1]
            elif outlet in trunks:
                join = _nearest_on(trunks[outlet], centre(outlet))
                path = path[:-1]
            else:
                join = centre(outlet)
                path = path[:-1]
        xy = np.vstack([np.asarray(g.get(display_key, g['xy']))[-1:],
                        *(centre(c)[None] for c in path), np.asarray(join)[None]])
        if np.any(np.hypot(*np.diff(xy, axis=0).T) > 3*BASE_M):
            continue                     # never draw a jump that the receivers did not make
        end_level = float(np.asarray(g['water_m'])[-1])
        water = np.minimum.accumulate(np.r_[end_level, np.full(len(path), level), level])
        for c in path:
            owned.setdefault(c, g['id'])
        links[g['id']] = dict(xy=xy, water_m=water, lake_id=lid, target=target)
    return links


# ----------------------------------------------------------------------------
# Source lakes constrained by a ready package's Base

def _block_lakes(world, i0, j0, i1, j1):
    from terrain_hydrology_regions import _sample_blocks
    gi, gj = np.meshgrid(np.arange(i0, i1), np.arange(j0, j1), indexing='ij')
    return _sample_blocks(world, gi, gj)


def source_lake_pieces(world, plan):
    """Base water of source lakes inside one core, ``{lake id: piece}``.

    The halo grid (core + one 240 m cell, the piece-independent routing
    surface) is wet where Base lies below the lake level, inside the lake's own
    7680 m blocks or the land blocks 8-adjacent to them in its basin. Kept
    components contain such a lake-block cell below the level, or touch the
    halo edge (their anchor may lie in the neighbour). Both cores of a cut see
    the same values on the two shared cell columns. Clipping at the exact core
    line therefore lets neighbouring pieces abut.
    """
    halo = np.asarray(plan.metadata.get('routing_halo_m', plan.routing_height_m), np.float64)
    hb = tuple(plan.metadata.get('routing_halo_bounds', plan.bounds))
    core = tuple(float(v) for v in plan.bounds)
    rows, cols = halo.shape
    g_r0, g_c0 = int(round(hb[1]/BASE_M)), int(round(hb[0]/BASE_M))
    bi = (g_r0+np.arange(rows))//BLOCK
    bj = (g_c0+np.arange(cols))//BLOCK
    i0, i1, j0, j1 = int(bi.min())-1, int(bi.max())+2, int(bj.min())-1, int(bj.max())+2
    blocks = _block_lakes(world, i0, j0, i1, j1)
    lake_b, basin_b, ocean_b = blocks['lake'], blocks['basin'], blocks['ocean']
    ids = sorted(int(v) for v in np.unique(lake_b) if v >= 0)
    out = {}
    if not ids:
        return out
    ri, rj = bi-i0, bj-j0
    struct = np.ones((3, 3), bool)
    cx = hb[0]+(np.arange(cols)+.5)*BASE_M
    cy = hb[1]+(np.arange(rows)+.5)*BASE_M
    inside = ((cy > core[1]) & (cy < core[3]))[:, None] & ((cx > core[0]) & (cx < core[2]))[None, :]
    for L in ids:
        lake = world.lakes[L]
        level = float(lake['level_m'])
        own_b = lake_b == L
        near = ndimage.binary_dilation(own_b, struct) & ~own_b & (lake_b < 0) & ~ocean_b & \
            (basin_b == int(lake['basin_id']))
        own = own_b[ri[:, None], rj[None, :]]
        allowed = own | near[ri[:, None], rj[None, :]]
        wet = allowed & (halo < level)
        if not wet.any():
            continue
        labels, n = ndimage.label(wet, struct)
        anchor = np.unique(labels[wet & own])
        edge = np.unique(np.r_[labels[0], labels[-1], labels[:, 0], labels[:, -1]])
        keep = np.isin(labels, np.setdiff1d(np.union1d(anchor, edge), [0]))
        if not keep.any():
            continue
        geom = rings_geometry(lake_rings(halo, hb, level=level, region=keep))
        geom = _polygons(shapely.set_precision(geom.intersection(_box(*core)), PRECISION_M))
        if geom.is_empty:
            continue
        bottom = halo[keep & inside]
        out[L] = dict(geometry=geom, max_depth_m=float(level-bottom.min()) if bottom.size else 0.)
    return out


def source_lake_geometry(world, lake):
    """Coarse 19.5 km shoreline of a source lake (the fallback), cached by the caller."""
    rings = lake_shore_rings(lake, dict(height_m=world.height_m, lake_ids=world.lake_ids, bounds=world.bounds))
    return rings_geometry(rings)


def combine_source_lake(coarse, pieces, cores):
    """``(base, remainder)``: the union of the Base pieces, and the coarse shape
    outside every ready core. An unfinished fine shore keeps the whole coarse
    fallback: a ready-core edge inside a lake is not a physical shoreline.
    """
    base = _polygons(shapely.union_all([p for p in pieces])) if pieces else Polygon()
    rest = coarse
    if cores and not coarse.is_empty:
        coverage = shapely.union_all([_box(*c) for c in cores])
        rest = _polygons(shapely.set_precision(
            coarse.difference(coverage), PRECISION_M))
        if not base.is_empty and not rest.is_empty:
            # Ignore the parent's actual shore and point tangencies. Only a
            # water edge through the lake's interior signals missing Base.
            frontier = coverage.boundary.intersection(coarse.buffer(-2*PRECISION_M))
            if base.intersection(frontier).length > PRECISION_M:
                return Polygon(), coarse
    return base, rest

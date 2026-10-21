"""Display connectors across hidden lakes and unrendered hydraulic links.

A reach that enters a lake ends on the shore; the lake's exit reach starts on
the opposite shore. With lakes hidden, the water between them is already in
the routing graph (``downstream_id``) and in ``world.receivers``, but no
course is drawn. Likewise a faded channel whose ``hydraulic_downstream_id``
was synthesized by ``HydrologyService._properties`` has a receiver path
through unrendered cells to a wet channel.

:func:`display_connectors` walks those receiver paths once per world and
returns display-only polylines. Nothing in the world is modified: reach ids,
``downstream_id``, flows, budgets, lakes and basins stay as routed.
See ``docs/methods/hydrology-connectors.md``.
"""
import numpy as np

CHAIKIN_ROUNDS = 2
MAX_CELLS = 512          # global maximum on the seed-42 world is 133 cells
WET_M3S = 1e-9


def _cell_xy(shape, bounds, cells):
    """Cell centres, identical to ``HydroWorld.cell_xy``."""
    h, w = shape
    dx, dy = (bounds[2]-bounds[0])/w, (bounds[3]-bounds[1])/h
    rows, cols = np.divmod(np.asarray(cells, np.int64), w)
    return np.stack((bounds[0]+(cols+.5)*dx, bounds[1]+(rows+.5)*dy), -1)


def _unwrap(xy, period):
    """Longitudes continued from the first point (shortest step each time)."""
    xy = np.array(xy, np.float64)
    if period and len(xy) > 1:
        xy[1:, 0] -= np.cumsum(np.round(np.diff(xy[:, 0])/period))*period
    return xy


def _chaikin(xy, rounds=CHAIKIN_ROUNDS):
    """Corner cutting with pinned ends: each segment (P, Q) becomes
    (3P+Q)/4, (P+3Q)/4. The curve stays inside the original polyline's hull."""
    for _ in range(rounds):
        if len(xy) < 3:
            break
        a, b = xy[:-1], xy[1:]
        cut = np.stack((.75*a+.25*b, .25*a+.75*b), 1).reshape(-1, 2)
        xy = np.vstack((xy[:1], cut[1:-1], xy[-1:]))
    return xy


def _insert_on(link, point, period):
    """Insert ``point`` (projected) as a vertex of ``link`` and return its
    coordinates in that link's frame. The join is then a shared vertex."""
    xy = link['xy']
    p = np.asarray(point, np.float64).copy()
    if period:
        p[0] += np.round((xy[0, 0]-p[0])/period)*period
    a, d = xy[:-1], np.diff(xy, axis=0)
    t = np.clip(np.einsum('ij,ij->i', p-a, d)/np.maximum(np.einsum('ij,ij->i', d, d), 1e-18), 0., 1.)
    near = a+d*t[:, None]
    k = int(np.argmin(np.hypot(*(near-p).T)))
    q, tk = near[k], t[k]
    level = link['water_m'][k]+(link['water_m'][k+1]-link['water_m'][k])*tk
    if tk <= 1e-9:
        index = k
    elif tk >= 1.-1e-9:
        index = k+1
    else:
        index = k+1
        link['xy'] = np.insert(xy, index, q, axis=0)
        link['water_m'] = np.insert(link['water_m'], index, level)
        link['pins'] = [i+(i >= index) for i in link['pins']]
    link['pins'] = sorted(set(link['pins']) | {index})
    return link['xy'][index].copy(), float(link['water_m'][index])


def _water(xy, start, end):
    """Non-increasing level from the parent end to the join, linear in length."""
    s = np.r_[0., np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    end = min(end, start)
    return np.minimum.accumulate(start+(end-start)*(s/s[-1] if s[-1] > 0 else s))


def display_connectors(world, max_cells=MAX_CELLS):
    """Display connectors ``{parent reach id: link}``.

    Candidates (flow above ``WET_M3S`` only):

    * ``end == 'lake'`` with a ``downstream_id`` (open lake with outflow): the
      receiver path must cross only cells of the entered lake and stop on the
      exit reach's head cell;
    * ``hydraulic_downstream_id``: the path must cross only unchannelled cells,
      each with routed discharge, and stop on a cell owned by that reach.

    Larger flows are walked first and
    own their cells; a smaller link stops on the first owned cell and joins
    that connector at an inserted shared vertex (one trunk per lake), after
    checking that the owner ends on the same downstream reach. Closed lakes,
    dry ends, unverified or unsafe paths get no link: the gap stays honest.

    ``link``: ``xy`` (N, 2) display vertices in the parent's longitude frame,
    ``water_m`` (N,), ``target`` (a reach id, or the parent id of a larger
    connector when ``target_kind == 'connector'``), ``reach_target`` (the real
    downstream reach), ``kind`` ('lake' | 'hydraulic'), ``cells`` (receiver
    path) and ``pins`` (indices of branch joins on this connector).
    """
    reaches = world.reaches
    rcv = np.asarray(world.receivers).ravel()
    reach_of = np.asarray(world.reach_ids).ravel()
    discharge = np.asarray(world.discharge_m3s).ravel()
    lake_ids = getattr(world, 'lake_ids', None)
    lake_ids = None if lake_ids is None else np.asarray(lake_ids).ravel()
    lakes = getattr(world, 'lakes', None) or []
    shape, bounds = tuple(world.shape), tuple(world.bounds)
    period = bounds[2]-bounds[0] if world.topology == 'sphere' else None
    h, w = shape
    step_max = 2.*np.hypot((bounds[2]-bounds[0])/w, (bounds[3]-bounds[1])/h)
    index = {r['id']: i for i, r in enumerate(reaches)}

    def flow(r):
        return float(r.get('flow_m3s') or 0.)

    candidates = []
    for r in reaches:
        cells = np.asarray(r.get('cells', ()), np.int64)
        if flow(r) <= WET_M3S or not cells.size or len(r.get('xy', ())) < 1:
            continue
        if r.get('end') == 'lake' and r.get('downstream_id') is not None:
            candidates.append((r, int(r['downstream_id']), 'lake'))
        elif r.get('hydraulic_downstream_id') is not None:
            candidates.append((r, int(r['hydraulic_downstream_id']), 'hydraulic'))

    owner, links = {}, {}
    for r, target, kind in sorted(candidates, key=lambda c: (-flow(c[0]), c[0]['id'])):
        t = index.get(target)
        if t is None or flow(reaches[t]) <= WET_M3S or target == r['id']:
            continue
        lake = r.get('enters_lake') if kind == 'lake' else None
        c, path, seen = int(rcv[int(np.asarray(r['cells'])[-1])]), [], set()
        while 0 <= c < rcv.size and reach_of[c] < 0 and c not in owner and c not in seen and len(path) < max_cells:
            seen.add(c)
            path.append(c)
            c = int(rcv[c])
        if not 0 <= c < rcv.size or c in seen or len(path) >= max_cells:
            continue
        if kind == 'hydraulic' and path and not np.all(discharge[path] > WET_M3S):
            continue                                   # water lost on the way: truly dry
        if kind == 'lake':
            # Inside a lake the routed cell discharge is not the flow measure
            # (the lake balance is): require an open, discharging lake.
            if lake is None or (lake_ids is not None and path and not np.all(lake_ids[path] == int(lake))):
                continue                               # left the entered lake
            record = lakes[int(lake)] if 0 <= int(lake) < len(lakes) else {}
            if record.get('closed') or float(record.get('outflow_m3s', 1.)) <= WET_M3S:
                continue
        merged = c in owner
        if merged:
            host = links[owner[c]]
            if host['reach_target'] != target:
                continue
        elif int(reach_of[c]) != target:
            continue
        elif kind == 'lake' and c != int(np.asarray(reaches[t]['cells'])[0]):
            continue                                   # must reach the exit head
        start = np.asarray(r['xy'][-1], np.float64)
        level0 = float(np.asarray(r.get('water_m', [np.nan]))[-1])
        if merged:
            end, level1 = _insert_on(host, _cell_xy(shape, bounds, [c])[0], period)
        elif kind == 'lake':
            end = np.asarray(reaches[t]['xy'][0], np.float64)
            level1 = float(np.asarray(reaches[t].get('water_m', [level0]))[0])
        else:
            end = _cell_xy(shape, bounds, [c])[0]      # a vertex of the target's course
            cells = np.asarray(reaches[t]['cells'], np.int64)
            k = int(np.flatnonzero(cells == c)[0]) + (reaches[t].get('leaves_lake') is not None)
            water = np.asarray(reaches[t].get('water_m', [level0]))
            level1 = float(water[min(k, len(water)-1)])
        raw = _unwrap(np.vstack((start[None], _cell_xy(shape, bounds, path).reshape(-1, 2), end[None])), period)
        # The join is the target's own point; on a sphere it is continued by
        # whole periods only, so it stays the same place on the planet.
        raw[-1] = end
        if period:
            raw[-1, 0] += np.round((raw[-2, 0]-end[0])/period)*period
        if not np.isfinite(raw).all() or np.max(np.hypot(*np.diff(raw, axis=0).T), initial=0.) > step_max:
            continue                                   # pole crossing or corrupt geometry
        xy = _chaikin(raw)
        if not np.isfinite(level0):
            level0 = level1
        links[r['id']] = dict(xy=xy, water_m=_water(xy, level0, level1), kind=kind,
                              target=owner[c] if merged else target,
                              target_kind='connector' if merged else 'reach',
                              reach_target=target, cells=np.asarray(path, np.int64), pins=[])
        for p in path:
            owner[p] = r['id']
    return links

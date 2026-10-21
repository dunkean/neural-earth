"""Deterministic array kernels for the global hydrology authority.

Everything here works on flat row-major cell indices of one fixed source grid
(row 0 is the northern edge, y increases southward). There is no randomness:
ties are broken by height, then flat index, then a fixed neighbour order.

Topology:
- ``sphere``: longitude wraps; a cell of the first (last) row has its three
  northern (southern) neighbours across the pole, in the same row half a turn
  away. The pole is therefore a shared node between rows, never an outlet.
- ``plane``: finite rectangle; optional explicit edge outlets.

Kernels:
- ``depression_sweep``: one pass in ascending height order with union-find.
  Pits create leaf depressions; the first contact between two drainage sets
  is their pass (the events arrive sorted by elevation, so this is Kruskal's
  minimum spanning forest of Barnes et al. 2020 without an edge list).
- ``depression_budget`` / ``fill_spill_cascade`` / ``final_lakes``: an annual
  steady-state Fill-Spill-Merge on that hierarchy.
- ``route_cells``: receivers on max(height, lake level): one BFS tree per lake
  to its single outlet or sink, sills pinned to the side the cascade credited,
  the sweep's raw steepest neighbour (so routed catchments equal the
  cascade's), steepest geodesic descent, then the Barnes, Lehman and Mulla
  (2014) combined gradient on remaining flats.
- ``drainage_order``, ``accumulate``, ``basin_roots``, ``propagate_owners``,
  ``extract_reaches``: stack order, sums, basins, incremental ownership and
  reach segmentation.
"""
from __future__ import annotations

import math

import numpy as np
from numba import njit

OUTLET = 0  # node 0 of the depression forest: open sea or declared world edge

# Neighbour order is part of the deterministic identity.
DI = np.array((-1, -1, -1, 0, 0, 1, 1, 1), np.int64)
DJ = np.array((-1, 0, 1, -1, 1, -1, 0, 1), np.int64)

UNASSIGNED = -2

TERM_NONE, TERM_OCEAN, TERM_SINK, TERM_EDGE, TERM_UNRESOLVED = 0, 1, 2, 3, 4
END_JOIN, END_LAKE, END_MOUTH, END_TERMINAL, END_FADE = 0, 1, 2, 3, 4
EV_LEAF, EV_MERGE, EV_OUTLET = 0, 1, 2


def grid_geometry(shape, bounds, topology):
    """Cell areas per row (m^2), neighbour distances per row/direction (m)."""
    h, w = (int(v) for v in shape)
    x0, y0, x1, y1 = (float(v) for v in bounds)
    if h < 2 or w < 2:
        raise ValueError('Hydrology needs at least a 2 by 2 grid')
    if not (x1 > x0 and y1 > y0):
        raise ValueError('bounds must be (x0, y0_north, x1, y1_south) with x1 > x0 and y1 > y0')
    dist = np.empty((h, 8), np.float64)
    if topology == 'sphere':
        if abs((y1 - y0) * 2 - (x1 - x0)) > 1e-6 * (x1 - x0):
            raise ValueError('Sphere hydrology needs a full equirectangular world (height = width / 2)')
        if w % 2:
            raise ValueError('Sphere hydrology needs an even column count for cross-pole neighbours')
        radius = (x1 - x0) / (2 * math.pi)
        dlon = 2 * math.pi / w
        edges = np.radians(90 - np.arange(h + 1) * 180 / h)
        row_area = radius ** 2 * dlon * (np.sin(edges[:-1]) - np.sin(edges[1:]))
        lat = np.radians(90 - (np.arange(h) + .5) * 180 / h)
        rows = np.arange(h)
        for k in range(8):
            ni = rows + DI[k]
            dl = np.full(h, DJ[k] * dlon)
            cross = (ni < 0) | (ni >= h)
            ni = np.where(cross, rows, ni)
            dl[cross] = math.pi + DJ[k] * dlon
            p1, p2 = lat, lat[ni]
            s = (np.sin((p2 - p1) / 2) ** 2
                 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2)
            dist[:, k] = 2 * radius * np.arcsin(np.sqrt(np.clip(s, 0, 1)))
    elif topology == 'plane':
        radius = None
        dx, dy = (x1 - x0) / w, (y1 - y0) / h
        row_area = np.full(h, dx * dy)
        for k in range(8):
            dist[:, k] = math.hypot(DI[k] * dy, DJ[k] * dx)
    else:
        raise ValueError("topology must be 'sphere' or 'plane'")
    if not np.all(dist > 0):
        raise ValueError('Degenerate neighbour distance')
    return row_area, dist, radius


@njit(inline='always')
def _neighbor(c, k, h, w, sphere):
    i = c // w
    j = c - i * w
    ni = i + DI[k]
    nj = j + DJ[k]
    if sphere:
        if ni < 0:
            ni = 0
            nj = j + w // 2 + DJ[k]
        elif ni >= h:
            ni = h - 1
            nj = j + w // 2 + DJ[k]
        nj = nj % w
    elif ni < 0 or ni >= h or nj < 0 or nj >= w:
        return -1
    return ni * w + nj


@njit(inline='always')
def _step_to(c, r, h, w, sphere, dist):
    for k in range(8):
        if _neighbor(c, k, h, w, sphere) == r:
            return dist[c // w, k]
    return 0.0


@njit(inline='always')
def _is_edge(c, h, w, sphere):
    if sphere:
        return False
    i = c // w
    j = c - i * w
    return i == 0 or i == h - 1 or j == 0 or j == w - 1


@njit(cache=True)
def label_components(mask, h, w, sphere):
    """Connected components of ``mask`` (8-neighbour, topology aware)."""
    n = h * w
    labels = np.full(n, -1, np.int32)
    queue = np.empty(n, np.int64)
    count = 0
    for s in range(n):
        if not mask[s] or labels[s] >= 0:
            continue
        labels[s] = count
        head = 0
        tail = 1
        queue[0] = s
        while head < tail:
            c = queue[head]
            head += 1
            for k in range(8):
                nb = _neighbor(c, k, h, w, sphere)
                if nb >= 0 and mask[nb] and labels[nb] < 0:
                    labels[nb] = count
                    queue[tail] = nb
                    tail += 1
        count += 1
    return labels, count


@njit(inline='always')
def _find(uf, a):
    root = a
    while uf[root] != root:
        root = uf[root]
    while uf[a] != root:
        nxt = uf[a]
        uf[a] = root
        a = nxt
    return root


@njit(cache=True)
def depression_sweep(z, order, ocean, h, w, sphere, edge_outlet, dist, sea_level):
    """Pit labels and the depression forest from one ascending-height sweep.

    ``order`` lists every non-ocean cell sorted by (height, index). Each cell
    takes the drainage label of its steepest strictly lower neighbour (ocean
    counts as lower); a cell with none is a pit and a new leaf. Every already
    swept neighbour in another set is a contact at this cell's height, which
    is the lowest pass between the two sets. Two depressions merge into a
    parent node; a depression meeting node 0 (sea/edge) drains out there.
    """
    n = h * w
    npit = 0
    for c in range(n):
        if ocean[c]:
            continue
        zc = z[c]
        lower = False
        for k in range(8):
            nb = _neighbor(c, k, h, w, sphere)
            if nb >= 0 and (ocean[nb] or z[nb] < zc):
                lower = True
                break
        if not lower and not (edge_outlet and _is_edge(c, h, w, sphere)):
            npit += 1
    m = 2 * npit + 2
    uf = np.arange(m).astype(np.int32)
    parent = np.full(m, -1, np.int32)
    child0 = np.full(m, -1, np.int32)
    child1 = np.full(m, -1, np.int32)
    spill = np.full(m, np.inf)
    base = np.full(m, -np.inf)
    to_outlet = np.zeros(m, np.bool_)
    pass_cell = np.full(m, -1, np.int64)
    inside_cell = np.full(m, -1, np.int64)
    outside_cell = np.full(m, -1, np.int64)
    exit_to = np.full(m, -1, np.int64)
    odep = np.full(m, -1, np.int32)
    pit_cell = np.full(m, -1, np.int64)
    ev_node = np.empty(2 * m, np.int32)
    ev_kind = np.empty(2 * m, np.int8)
    ne = 0
    label = np.zeros(n, np.int32)
    down = np.full(n, -1, np.int64)
    attach = np.full(n, -1, np.int32)
    popped = ocean.copy()
    nn = 1
    for idx in range(order.size):
        c = order[idx]
        zc = z[c]
        row = c // w
        best = -1
        best_slope = -1.0
        for k in range(8):
            nb = _neighbor(c, k, h, w, sphere)
            if nb < 0:
                continue
            if ocean[nb]:
                slope = (zc - min(sea_level, zc)) / dist[row, k]
            elif z[nb] < zc:
                slope = (zc - z[nb]) / dist[row, k]
            else:
                continue
            if slope > best_slope:
                best_slope = slope
                best = nb
        edge = edge_outlet and _is_edge(c, h, w, sphere)
        down[c] = best
        if best >= 0:
            lab = label[best]
        elif edge:
            lab = OUTLET
        else:
            lab = nn
            pit_cell[nn] = c
            base[nn] = zc
            ev_node[ne] = nn
            ev_kind[ne] = EV_LEAF
            ne += 1
            nn += 1
        label[c] = lab
        popped[c] = True
        rc = _find(uf, lab)
        attach[c] = rc
        if edge and rc != OUTLET:
            to_outlet[rc] = True
            spill[rc] = zc
            pass_cell[rc] = c
            inside_cell[rc] = c
            odep[rc] = OUTLET
            ev_node[ne] = rc
            ev_kind[ne] = EV_OUTLET
            ne += 1
            uf[rc] = OUTLET
            rc = OUTLET
        for k in range(8):
            nb = _neighbor(c, k, h, w, sphere)
            if nb < 0 or nb == c or not popped[nb]:
                continue
            rn = OUTLET if ocean[nb] else _find(uf, label[nb])
            if rn == rc:
                continue
            if rc == OUTLET or rn == OUTLET:
                r = rc if rn == OUTLET else rn
                to_outlet[r] = True
                spill[r] = zc
                pass_cell[r] = c
                if r == rc:
                    inside_cell[r] = c
                    outside_cell[r] = nb
                    exit_to[r] = nb
                    odep[r] = label[nb]
                else:
                    inside_cell[r] = nb
                    outside_cell[r] = c
                    exit_to[r] = best
                    odep[r] = label[c]
                ev_node[ne] = r
                ev_kind[ne] = EV_OUTLET
                ne += 1
                uf[r] = OUTLET
                rc = OUTLET
            else:
                mm = nn
                nn += 1
                child0[mm] = rc
                child1[mm] = rn
                parent[rc] = mm
                parent[rn] = mm
                spill[rc] = zc
                spill[rn] = zc
                base[mm] = zc
                pass_cell[rc] = c
                pass_cell[rn] = c
                inside_cell[rc] = c
                outside_cell[rc] = nb
                inside_cell[rn] = nb
                outside_cell[rn] = c
                exit_to[rc] = nb
                exit_to[rn] = best
                odep[rc] = label[nb]
                odep[rn] = label[c]
                uf[rc] = mm
                uf[rn] = mm
                ev_node[ne] = mm
                ev_kind[ne] = EV_MERGE
                ne += 1
                rc = mm
    for c in range(n):
        if ocean[c]:
            label[c] = OUTLET
    return (label, down, attach, nn, parent[:nn].copy(), child0[:nn].copy(), child1[:nn].copy(),
            spill[:nn].copy(), base[:nn].copy(), to_outlet[:nn].copy(), pass_cell[:nn].copy(),
            inside_cell[:nn].copy(), outside_cell[:nn].copy(), exit_to[:nn].copy(),
            odep[:nn].copy(), pit_cell[:nn].copy(), ev_node[:ne].copy(), ev_kind[:ne].copy())


@njit(cache=True)
def depression_budget(z, label, attach0, ocean, runoff_vol, loss_vol, area, nn,
                      parent, child0, child1, spill, pit_cell):
    """Attach cells to their flooding node; subtree losses/areas at spill.

    A cell floods in node d when its height is below d's spill and it is in
    d's subtree; it is attached to the lowest such node. Units: m^3/year
    for water, m^2 for area, m^3 for volumes.
    """
    n = z.size
    attach = np.full(n, -1, np.int32)
    for c in range(n):
        t = attach0[c]
        if ocean[c] or t <= 0:
            continue
        while t > 0 and spill[t] <= z[c]:
            t = parent[t]
            if t < 0:
                break
        attach[c] = t if t > 0 else -1
    gross0 = np.zeros(nn)
    loss = np.zeros(nn)
    flood_area = np.zeros(nn)
    flood_za = np.zeros(nn)
    for c in range(n):
        lb = label[c]
        if lb > 0:
            gross0[lb] += runoff_vol[c]
        t = attach[c]
        if t > 0:
            loss[t] += loss_vol[c]
            flood_area[t] += area[c]
            flood_za[t] += z[c] * area[c]
    lowpit = pit_cell.copy()
    for d in range(1, nn):
        a = child0[d]
        if a >= 0:
            b = child1[d]
            loss[d] += loss[a] + loss[b]
            flood_area[d] += flood_area[a] + flood_area[b]
            flood_za[d] += flood_za[a] + flood_za[b]
            pa = lowpit[a]
            pb = lowpit[b]
            if z[pb] < z[pa] or (z[pb] == z[pa] and pb < pa):
                lowpit[d] = pb
            else:
                lowpit[d] = pa
    volume = np.zeros(nn)
    for d in range(1, nn):
        if np.isfinite(spill[d]):
            volume[d] = spill[d] * flood_area[d] - flood_za[d]
        else:
            loss[d] = np.inf
            volume[d] = np.inf
    return attach, gross0, loss, flood_area, volume, lowpit


@njit(inline='always')
def _sibling(d, parent, child0, child1):
    p = parent[d]
    return child1[p] if child0[p] == d else child0[p]


@njit(cache=True)
def _deliver(x, q, top, gross, full, loss, parent, child0, child1, odep, fd, ft, fq, fw):
    """Water q (m^3/yr) from outside subtree(top) arrives at leaf x."""
    d = x
    while True:
        gross[d] += q
        if d == top:
            break
        d = parent[d]
    sp = 0
    fd[0] = x
    ft[0] = top
    fq[0] = q
    fw[0] = False
    while sp >= 0:
        d = fd[sp]
        top = ft[sp]
        if fw[sp]:
            fw[sp] = False
            s = _sibling(d, parent, child0, child1)
            if full[s]:
                d = parent[d]
                fd[sp] = d
            else:
                sp -= 1
                continue
        q = fq[sp]
        if not full[d]:
            if gross[d] >= loss[d]:
                full[d] = True
                q = gross[d] - loss[d]
            else:
                sp -= 1
                continue
        if d == top:
            sp -= 1
            continue
        s = _sibling(d, parent, child0, child1)
        if full[s]:
            fd[sp] = parent[d]
            fq[sp] = q
            continue
        fw[sp] = True
        x2 = odep[d]
        e = x2
        while True:
            gross[e] += q
            if e == s:
                break
            e = parent[e]
        sp += 1
        fd[sp] = x2
        ft[sp] = s
        fq[sp] = q
        fw[sp] = False


@njit(cache=True)
def fill_spill_cascade(nn, parent, child0, child1, odep, gross0, loss, ev_node, ev_kind):
    """Annual steady-state Fill-Spill-Merge on the depression forest.

    Events are replayed in sweep order. A node is full when the water
    reaching its lake covers net open-water loss at its spill extent. A full
    child spills its surplus into its sibling at the pass (``odep`` leaf); when
    both are full, their parent becomes the active lake. A full depression
    that reaches the sea-connected set spills into the depression on the
    other side of its pass (``odep``; 0 = sea or declared edge), which may be
    an earlier closed basin; that link chain is followed until the water is
    absorbed or reaches the sea. Water entering a subtree later cascades up
    through full nodes. Each rule only adds water, so the fixed point does not
    depend on anything but the event order, which is the sweep order.
    """
    gross = gross0.copy()
    full = np.zeros(nn, np.bool_)
    fd = np.empty(nn + 1, np.int32)
    ft = np.empty(nn + 1, np.int32)
    fq = np.empty(nn + 1)
    fw = np.zeros(nn + 1, np.bool_)
    to_sea = 0.0
    for e in range(ev_node.size):
        d = ev_node[e]
        kind = ev_kind[e]
        if kind == EV_LEAF:
            full[d] = gross[d] >= loss[d]
        elif kind == EV_MERGE:
            a = child0[d]
            b = child1[d]
            gross[d] = gross[a] + gross[b]
            if full[a] and not full[b]:
                _deliver(odep[a], gross[a] - loss[a], b, gross, full, loss,
                         parent, child0, child1, odep, fd, ft, fq, fw)
            elif full[b] and not full[a]:
                _deliver(odep[b], gross[b] - loss[b], a, gross, full, loss,
                         parent, child0, child1, odep, fd, ft, fq, fw)
            if full[a] and full[b]:
                full[d] = gross[d] >= loss[d]
        elif full[d]:
            q = gross[d] - loss[d]
            x = odep[d]
            while x > 0 and q > 0:
                top = x
                while parent[top] >= 0:
                    top = parent[top]
                was_full = full[top]
                _deliver(x, q, top, gross, full, loss,
                         parent, child0, child1, odep, fd, ft, fq, fw)
                if not full[top]:
                    q = 0.0
                    break
                if not was_full:
                    q = gross[top] - loss[top]
                x = odep[top]
            to_sea += q
    return gross, full, to_sea


@njit(cache=True)
def final_lakes(z, rate_vol, area, attach, sorted_cells, group_start, group_end, nn,
                parent, child0, child1, spill, base, full, gross, loss, lowpit,
                pass_cell, inside_cell, outside_cell):
    """Final lake nodes, their levels and the per-cell water surface.

    A final lake is an active node (leaf, or both children full) that is not
    merged into its parent. Full lakes stand at their spill. A closed lake
    floods its subtree cells in ascending height until the next cell would
    make net evaporation exceed the inflow; its level is the midpoint of that
    height step. ``sorted_cells`` groups attached cells by node then height.
    """
    n = z.size
    final = np.zeros(nn, np.bool_)
    for d in range(1, nn):
        a = child0[d]
        active = a < 0 or (full[a] and full[child1[d]])
        if not active:
            continue
        p = parent[d]
        if full[d] and p > 0 and full[_sibling(d, parent, child0, child1)]:
            continue
        final[d] = True
    own = np.full(nn, -1, np.int32)
    for d in range(nn - 1, 0, -1):
        if final[d]:
            own[d] = d
        else:
            p = parent[d]
            if p > 0 and own[p] >= 0:
                own[d] = own[p]
    level = np.full(nn, np.nan)
    count = np.zeros(nn, np.int64)
    evap = np.zeros(nn)
    residual = np.zeros(nn)
    for d in range(1, nn):
        if not final[d]:
            continue
        if full[d]:
            level[d] = spill[d]
            count[d] = group_end[d] - group_start[d]
            evap[d] = loss[d]
            continue
        cum = 0.0
        if child0[d] >= 0:
            cum = loss[child0[d]] + loss[child1[d]]
        k = group_start[d]
        while k < group_end[d] and cum + rate_vol[sorted_cells[k]] <= gross[d]:
            cum += rate_vol[sorted_cells[k]]
            k += 1
        count[d] = k - group_start[d]
        evap[d] = cum
        residual[d] = gross[d] - cum
        lower = base[d]
        if count[d] > 0:
            lower = max(lower, z[sorted_cells[k - 1]])
        if k < group_end[d]:
            level[d] = .5 * (lower + z[sorted_cells[k]])
        elif np.isfinite(spill[d]):
            level[d] = spill[d]
        else:
            level[d] = lower + 1.0
    flood = np.full(n, -1, np.int32)
    wl = np.full(n, -np.inf)
    for t in range(1, nn):
        lake = own[t]
        if lake < 0:
            continue
        stop = group_end[t]
        if t == lake and not full[t]:
            stop = group_start[t] + count[t]
        for k in range(group_start[t], stop):
            c = sorted_cells[k]
            flood[c] = lake
            wl[c] = level[lake]
    # A lake standing exactly at an internal sill would split at the sill
    # pair; flood every submerged merge pass so each lake is one component.
    # These zero-depth connectors carry topology, not open-water loss.
    connector = np.zeros(n, np.bool_)
    stack = np.empty(nn, np.int32)
    for d in range(1, nn):
        if not final[d]:
            continue
        exit_cell = pass_cell[d] if full[d] else -1
        # A sill on the far side of the pass is reached through the inside
        # cell; at a tied height that cell is dry, so it joins the lake.
        c = inside_cell[d]
        if exit_cell >= 0 and c >= 0 and c != exit_cell and flood[c] < 0 and z[c] <= level[d]:
            flood[c] = d
            wl[c] = level[d]
            connector[c] = True
        if child0[d] < 0:
            continue
        sp = 0
        stack[0] = d
        while sp >= 0:
            m = stack[sp]
            sp -= 1
            a = child0[m]
            if a < 0:
                continue
            for c in (inside_cell[a], outside_cell[a]):
                if c >= 0 and c != exit_cell and flood[c] < 0 and z[c] <= level[d]:
                    flood[c] = d
                    wl[c] = level[d]
                    connector[c] = True
            sp += 1
            stack[sp] = a
            sp += 1
            stack[sp] = child1[m]
    sink = np.full(nn, -1, np.int64)
    ncell = np.zeros(nn, np.int64)
    farea = np.zeros(nn)
    fvol = np.zeros(nn)
    zmin = np.full(nn, np.inf)
    for c in range(n):
        d = flood[c]
        if d < 0:
            continue
        ncell[d] += 1
        if z[c] < zmin[d] or (z[c] == zmin[d] and c < sink[d]):
            zmin[d] = z[c]
            sink[d] = c
        if not connector[c]:
            farea[d] += area[c]
            fvol[d] += (wl[c] - z[c]) * area[c]
    for d in range(1, nn):
        if final[d] and ncell[d] == 0:
            sink[d] = lowpit[d]
            zmin[d] = z[lowpit[d]]
    return final, level, flood, wl, connector, sink, ncell, farea, fvol, zmin, evap, residual


@njit(cache=True)
def lake_seeds(finals, full, ncell, pass_cell, sink, flood, z, h, w, sphere):
    """BFS roots: a closed lake's sink, or the lake cell beside a spilling sill."""
    cells = np.full(finals.size, -1, np.int64)
    rcv = np.full(finals.size, -1, np.int64)
    for i in range(finals.size):
        d = finals[i]
        if not full[d]:
            cells[i] = sink[d]
        elif ncell[d] > 0:
            p = pass_cell[d]
            best = -1
            for k in range(8):
                nb = _neighbor(p, k, h, w, sphere)
                if nb >= 0 and flood[nb] == d and (best < 0 or z[nb] < z[best]):
                    best = nb
            cells[i] = best
            rcv[i] = p
    return cells, rcv


@njit(cache=True)
def route_cells(z, ocean, flood, wl, down, seed_cell, seed_rcv, exit_from, exit_to, h, w,
                sphere, edge_outlet, dist, sea_level):
    """Receivers (-1 terminal), terminal kinds, routing surface and step lengths.

    ``exit_from``/``exit_to`` pin each spilling lake's sill to the side the
    cascade credited, also at an equal level (-1 = declared plane edge); the
    first entry for a shared sill wins, so callers list the lake that contains
    it first. Other dry cells keep the raw steepest neighbour ``down`` that
    defined their depression label unless it is higher on the routing surface
    (a dry cell is never below its own lake, so this holds at ties too), so
    routed catchments equal the cascade's; then steepest descent, then flats.
    Both rules follow the cascade's acyclic lake-to-lake order.
    """
    n = h * w
    filled = z.copy()
    for c in range(n):
        if ocean[c]:
            filled[c] = sea_level
        elif flood[c] >= 0 and wl[c] > filled[c]:
            filled[c] = wl[c]
    rcv = np.full(n, UNASSIGNED, np.int64)
    term = np.zeros(n, np.int8)
    step = np.zeros(n)
    queue = np.empty(n, np.int64)
    tail = 0
    for c in range(n):
        if ocean[c]:
            rcv[c] = -1
            term[c] = TERM_OCEAN
    # One BFS tree per flooded component, rooted at its sink or outlet. A lake
    # standing exactly at a sill that is also an internal pass touches it from
    # several sides; the second round roots each remaining part at that sill.
    head = 0
    for again in range(2):
        for s in range(seed_cell.size):
            c = seed_cell[s]
            r = seed_rcv[s]
            if c < 0:
                continue
            if again:
                if r < 0:
                    continue
                for k in range(8):
                    nb = _neighbor(r, k, h, w, sphere)
                    if nb >= 0 and rcv[nb] == UNASSIGNED and flood[nb] == flood[c]:
                        rcv[nb] = r
                        step[nb] = _step_to(nb, r, h, w, sphere, dist)
                        queue[tail] = nb
                        tail += 1
                continue
            if rcv[c] != UNASSIGNED:
                continue
            rcv[c] = r
            if r < 0:
                term[c] = TERM_SINK
            else:
                step[c] = _step_to(c, r, h, w, sphere, dist)
            if flood[c] >= 0:
                queue[tail] = c
                tail += 1
        while head < tail:
            c = queue[head]
            head += 1
            for k in range(8):
                nb = _neighbor(c, k, h, w, sphere)
                if nb >= 0 and rcv[nb] == UNASSIGNED and flood[nb] == flood[c]:
                    rcv[nb] = c
                    step[nb] = _step_to(nb, c, h, w, sphere, dist)
                    queue[tail] = nb
                    tail += 1
    # Dry closed sinks (no flooded cell) are terminal seeds without a tree.
    for s in range(exit_from.size):
        c = exit_from[s]
        t = exit_to[s]
        if c < 0 or rcv[c] != UNASSIGNED:
            continue
        if t < 0:
            if edge_outlet and _is_edge(c, h, w, sphere):
                rcv[c] = -1
                term[c] = TERM_EDGE
        elif ocean[t] or filled[t] <= filled[c]:
            rcv[c] = t
            step[c] = _step_to(c, t, h, w, sphere, dist)
    for c in range(n):
        b = down[c]
        if rcv[c] == UNASSIGNED and flood[c] < 0 and b >= 0 and (ocean[b] or filled[b] <= filled[c]):
            rcv[c] = b
            step[c] = _step_to(c, b, h, w, sphere, dist)
    # Steepest geodesic descent on the routing surface.
    for c in range(n):
        if rcv[c] != UNASSIGNED:
            continue
        fc = filled[c]
        row = c // w
        best = -1
        best_k = -1
        best_slope = -1.0
        for k in range(8):
            nb = _neighbor(c, k, h, w, sphere)
            if nb < 0:
                continue
            if ocean[nb]:
                slope = (fc - min(sea_level, fc)) / dist[row, k]
            elif filled[nb] < fc:
                slope = (fc - filled[nb]) / dist[row, k]
            else:
                continue
            if slope > best_slope:
                best_slope = slope
                best = nb
                best_k = k
        if best >= 0:
            rcv[c] = best
            step[c] = dist[row, best_k]
        elif edge_outlet and _is_edge(c, h, w, sphere):
            rcv[c] = -1
            term[c] = TERM_EDGE
    # Remaining cells are flats of the routing surface.
    flat = np.full(n, -1, np.int32)
    nflat = 0
    for s in range(n):
        if rcv[s] != UNASSIGNED or flat[s] >= 0:
            continue
        flat[s] = nflat
        head = 0
        tail = 1
        queue[0] = s
        while head < tail:
            c = queue[head]
            head += 1
            for k in range(8):
                nb = _neighbor(c, k, h, w, sphere)
                if (nb >= 0 and rcv[nb] == UNASSIGNED and flat[nb] < 0
                        and filled[nb] == filled[c]):
                    flat[nb] = nflat
                    queue[tail] = nb
                    tail += 1
        nflat += 1
    unresolved = 0
    if nflat:
        outer = np.zeros(nflat, np.bool_)
        for c in range(n):
            lab = flat[c]
            if lab < 0:
                continue
            for k in range(8):
                nb = _neighbor(c, k, h, w, sphere)
                if (nb >= 0 and flat[nb] < 0 and not ocean[nb] and rcv[nb] != UNASSIGNED
                        and filled[nb] == filled[c] and flood[nb] < 0):
                    outer[lab] = True
        towards = np.zeros(n, np.int64)
        away = np.zeros(n, np.int64)
        tail = 0
        for c in range(n):
            lab = flat[c]
            if lab < 0:
                continue
            for k in range(8):
                nb = _neighbor(c, k, h, w, sphere)
                if (nb >= 0 and flat[nb] < 0 and not ocean[nb] and rcv[nb] != UNASSIGNED
                        and filled[nb] == filled[c] and (flood[nb] < 0 or not outer[lab])):
                    towards[c] = 1
                    queue[tail] = c
                    tail += 1
                    break
        head = 0
        while head < tail:
            c = queue[head]
            head += 1
            for k in range(8):
                nb = _neighbor(c, k, h, w, sphere)
                if nb >= 0 and flat[nb] == flat[c] and towards[nb] == 0:
                    towards[nb] = towards[c] + 1
                    queue[tail] = nb
                    tail += 1
        tail = 0
        for c in range(n):
            if flat[c] < 0:
                continue
            for k in range(8):
                nb = _neighbor(c, k, h, w, sphere)
                if nb >= 0 and not ocean[nb] and filled[nb] > filled[c]:
                    away[c] = 1
                    queue[tail] = c
                    tail += 1
                    break
        head = 0
        fh = np.zeros(nflat, np.int64)
        while head < tail:
            c = queue[head]
            head += 1
            if away[c] > fh[flat[c]]:
                fh[flat[c]] = away[c]
            for k in range(8):
                nb = _neighbor(c, k, h, w, sphere)
                if nb >= 0 and flat[nb] == flat[c] and away[nb] == 0:
                    away[nb] = away[c] + 1
                    queue[tail] = nb
                    tail += 1
        mask = np.zeros(n, np.int64)
        for c in range(n):
            if flat[c] >= 0:
                mask[c] = 2 * towards[c]
                if away[c] > 0:
                    mask[c] += fh[flat[c]] - away[c]
        for c in range(n):
            lab = flat[c]
            if lab < 0:
                continue
            row = c // w
            if towards[c] == 0:
                rcv[c] = -1
                term[c] = TERM_UNRESOLVED
                unresolved += 1
                continue
            best = -1
            best_k = -1
            best_score = mask[c]
            best_d = np.inf
            for k in range(8):
                nb = _neighbor(c, k, h, w, sphere)
                if nb < 0 or ocean[nb]:
                    continue
                if flat[nb] == lab:
                    score = mask[nb]
                elif (flat[nb] < 0 and rcv[nb] != UNASSIGNED and filled[nb] == filled[c]
                      and (flood[nb] < 0 or not outer[lab])):
                    score = -1
                else:
                    continue
                d = dist[row, k]
                if score < best_score or (score == best_score and best >= 0 and d < best_d):
                    best = nb
                    best_k = k
                    best_score = score
                    best_d = d
            if best < 0:
                rcv[c] = -1
                term[c] = TERM_UNRESOLVED
                unresolved += 1
            else:
                rcv[c] = best
                step[c] = dist[row, best_k]
    return filled, rcv, term, step, unresolved


@njit(cache=True)
def drainage_order(rcv):
    """Donor CSR and a stack order in which every receiver precedes donors."""
    n = rcv.size
    offsets = np.zeros(n + 1, np.int64)
    for c in range(n):
        r = rcv[c]
        if r >= 0:
            offsets[r + 1] += 1
    for c in range(n):
        offsets[c + 1] += offsets[c]
    fill = offsets[:-1].copy()
    donors = np.empty(offsets[n], np.int64)
    for c in range(n):
        r = rcv[c]
        if r >= 0:
            donors[fill[r]] = c
            fill[r] += 1
    order = np.empty(n, np.int64)
    tail = 0
    for c in range(n):
        if rcv[c] < 0:
            order[tail] = c
            tail += 1
    head = 0
    while head < tail:
        c = order[head]
        head += 1
        for t in range(offsets[c], offsets[c + 1]):
            order[tail] = donors[t]
            tail += 1
    return order[:tail].copy(), offsets, donors


@njit(cache=True)
def accumulate(order, rcv, values):
    acc = values.copy()
    for t in range(order.size - 1, -1, -1):
        c = order[t]
        r = rcv[c]
        if r >= 0:
            acc[r] += acc[c]
    return acc


@njit(cache=True)
def basin_roots(order, rcv, ocean):
    """Last land cell of each flow path (mouth, sink or edge outlet)."""
    root = np.full(rcv.size, -1, np.int64)
    for t in range(order.size):
        c = order[t]
        if ocean[c]:
            continue
        r = rcv[c]
        if r < 0 or ocean[r]:
            root[c] = c
        else:
            root[c] = root[r]
    return root


@njit(cache=True)
def propagate_owners(order, rcv, ocean, reach_of, lake_of):
    """Incremental ownership: first network reach or lake each cell joins."""
    n = rcv.size
    owner_reach = np.full(n, -1, np.int32)
    owner_lake = np.full(n, -1, np.int32)
    for t in range(order.size):
        c = order[t]
        if ocean[c]:
            continue
        if reach_of[c] >= 0:
            owner_reach[c] = reach_of[c]
        elif lake_of[c] >= 0:
            owner_lake[c] = lake_of[c]
        else:
            r = rcv[c]
            if r >= 0 and not ocean[r]:
                owner_reach[c] = owner_reach[r]
                owner_lake[c] = owner_lake[r]
    return owner_reach, owner_lake


@njit(cache=True)
def extract_reaches(rcv, is_chan, lake_of, ocean):
    """Split the channel tree into reaches between network nodes.

    Heads are sources (no channel donor), confluences (several channel donors)
    and lake exits (a lake cell drains into the cell). Reach r owns
    cells[start[r]:start[r + 1]]; it ends on a confluence head, a lake, the sea,
    a terminal cell or where the channel criterion lapses.
    """
    n = rcv.size
    up = np.zeros(n, np.int32)
    lake_up = np.full(n, -1, np.int32)
    for c in range(n):
        r = rcv[c]
        if r < 0:
            continue
        if is_chan[c]:
            up[r] += 1
        elif lake_of[c] >= 0 and lake_of[r] < 0:
            lake_up[r] = lake_of[c]
    heads = np.zeros(n, np.bool_)
    nhead = 0
    nchan = 0
    for c in range(n):
        if is_chan[c]:
            nchan += 1
            if up[c] != 1 or lake_up[c] >= 0:
                heads[c] = True
                nhead += 1
    start = np.empty(nhead + 1, np.int64)
    cells = np.empty(nchan, np.int64)
    end_kind = np.empty(nhead, np.int8)
    end_next = np.full(nhead, -1, np.int64)
    reach_of = np.full(n, -1, np.int32)
    rid = 0
    pos = 0
    for hcell in range(n):
        if not heads[hcell]:
            continue
        start[rid] = pos
        c = hcell
        while True:
            reach_of[c] = rid
            cells[pos] = c
            pos += 1
            r = rcv[c]
            if r < 0:
                end_kind[rid] = END_TERMINAL
                break
            if ocean[r]:
                end_kind[rid] = END_MOUTH
                end_next[rid] = r
                break
            if lake_of[r] >= 0:
                end_kind[rid] = END_LAKE
                end_next[rid] = r
                break
            if not is_chan[r]:
                end_kind[rid] = END_FADE
                break
            if heads[r]:
                end_kind[rid] = END_JOIN
                end_next[rid] = r
                break
            c = r
        rid += 1
    start[rid] = pos
    return start, cells[:pos].copy(), end_kind, end_next, reach_of, up, lake_up


@njit(cache=True)
def collect_upstream(offsets, donors, seeds, limit):
    """All cells draining through ``seeds`` (inclusive), breadth first."""
    out = np.empty(min(limit, offsets.size - 1), np.int64)
    tail = 0
    for s in range(seeds.size):
        if tail < out.size:
            out[tail] = seeds[s]
            tail += 1
    head = 0
    while head < tail:
        c = out[head]
        head += 1
        for t in range(offsets[c], offsets[c + 1]):
            if tail >= out.size:
                return out[:tail].copy(), False
            out[tail] = donors[t]
            tail += 1
    return out[:tail].copy(), True


@njit(cache=True)
def dilate(seeds, radius, allowed, h, w, sphere):
    """Cells within ``radius`` neighbour steps of ``seeds`` through ``allowed`` cells."""
    n = h * w
    steps = np.full(n, -1, np.int32)
    out = np.empty(n, np.int64)
    tail = 0
    for s in range(seeds.size):
        c = seeds[s]
        if steps[c] < 0:
            steps[c] = 0
            out[tail] = c
            tail += 1
    head = 0
    while head < tail:
        c = out[head]
        head += 1
        if steps[c] >= radius:
            continue
        for k in range(8):
            nb = _neighbor(c, k, h, w, sphere)
            if nb >= 0 and steps[nb] < 0 and allowed[nb]:
                steps[nb] = steps[c] + 1
                out[tail] = nb
                tail += 1
    cells = out[:tail].copy()
    dist = np.empty(tail, np.int32)
    for t in range(tail):
        dist[t] = steps[cells[t]]
    return cells, dist

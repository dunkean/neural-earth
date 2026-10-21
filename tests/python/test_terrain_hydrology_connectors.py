"""Display connectors across hidden lakes: exact receiver paths, one trunk per
lake, no link for closed lakes or dry ends, sphere seams, untouched graph."""
import copy
import unittest

from tools._bootstrap import activate
activate()
import numpy as np
from terrain_hydrology_connectors import display_connectors, _cell_xy


class Grid:
    """Minimal routed world: receivers, channel ownership, lakes, reaches."""

    def __init__(self, h, w, topology='plane', cell=1000.):
        self.shape, self.topology = (h, w), topology
        self.bounds = (0., 0., w*cell, h*cell)
        self.receivers = np.full(h*w, -1, np.int64)
        self.reach_ids = np.full(h*w, -1, np.int32)
        self.lake_ids = np.full(h*w, -1, np.int32)
        self.discharge_m3s = np.zeros(h*w)
        self.reaches, self.lakes = [], []

    def cell(self, row, col):
        return row*self.shape[1]+col

    def route(self, cells, then=-1, q=None):
        cells = [self.cell(*c) for c in cells]
        for a, b in zip(cells, cells[1:]+[then]):
            self.receivers[a] = b
        if q is not None:
            self.discharge_m3s[cells] = q
        return cells

    def lake(self, cells, then, outflow, q=1.):
        lake_id = len(self.lakes)
        cells = self.route(cells, then, q)
        self.lake_ids[cells] = lake_id
        self.lakes.append(dict(id=lake_id, closed=then < 0, outflow_m3s=outflow))
        return lake_id

    def reach(self, cells, flow, end, then=-1, drain=None, **extra):
        rid = len(self.reaches)
        cells = self.route(cells, then, flow)
        self.reach_ids[cells] = rid
        centres = [_cell_xy(self.shape, self.bounds, [c])[0] for c in cells]
        if drain is not None:
            centres.insert(0, None)
        if end == 'lake':
            centres.append(None)
        xy = np.array([c if c is not None else [np.nan, np.nan] for c in centres])
        period = self.bounds[2] if self.topology == 'sphere' else None

        def link(a, b):
            pa, pb = _cell_xy(self.shape, self.bounds, [a, b])
            if period:
                pb[0] += np.round((pa[0]-pb[0])/period)*period
            return .5*(pa+pb)
        if drain is not None:
            xy[0] = link(cells[0], drain)
        if end == 'lake':
            xy[-1] = link(cells[-1], then)
        if period:
            xy[1:, 0] -= np.cumsum(np.round(np.diff(xy[:, 0])/period))*period
        self.reaches.append(dict(id=rid, xy=xy, water_m=np.linspace(100.-rid, 99.-rid, len(xy)),
                                 flow_m3s=float(flow), cells=np.asarray(cells, np.int64), end=end,
                                 downstream_id=None, **extra))
        return rid


def lake_fixture():
    """Two inflows into one open lake, a closed lake, a dry and a wet hydraulic link."""
    g = Grid(5, 9)
    head = g.cell(2, 6)
    open_lake = g.lake([(2, 3), (2, 4), (2, 5)], head, outflow=30.)
    g.discharge_m3s[g.cell(2, 4)] = 0.     # lake-interior cell discharge is not the lake's flow
    big = g.reach([(2, 0), (2, 1), (2, 2)], 20., 'lake', g.cell(2, 3), enters_lake=open_lake)
    small = g.reach([(4, 4), (3, 4)], 10., 'lake', g.cell(2, 4), enters_lake=open_lake)
    exit_ = g.reach([(2, 6), (2, 7), (2, 8)], 30., 'mouth', drain=g.cell(2, 5), leaves_lake=open_lake)
    for r in (big, small):
        g.reaches[r]['downstream_id'] = exit_
    closed_lake = g.lake([(0, 2)], -1, outflow=0.)
    closed = g.reach([(0, 0), (0, 1)], 5., 'lake', g.cell(0, 2), enters_lake=closed_lake)
    g.route([(4, 2), (3, 2)], g.cell(2, 2), q=0.)                     # water lost on the way
    dry = g.reach([(4, 0), (4, 1)], 3., 'dry_out', g.cell(4, 2), hydraulic_downstream_id=big)
    g.route([(1, 7)], g.cell(2, 7), q=2.)
    wet = g.reach([(0, 6), (0, 7)], 2., 'dry_out', g.cell(1, 7), hydraulic_downstream_id=exit_)
    terminal = g.reach([(4, 6), (4, 7)], 1., 'terminal')
    return g, dict(big=big, small=small, exit=exit_, closed=closed, dry=dry, wet=wet, terminal=terminal)


class ConnectorTests(unittest.TestCase):
    def test_two_entries_make_one_trunk_and_an_exact_branch(self):
        g, ids = lake_fixture()
        links = display_connectors(g)
        trunk, branch = links[ids['big']], links[ids['small']]
        exit_xy = g.reaches[ids['exit']]['xy']
        self.assertEqual((trunk['target'], trunk['target_kind'], trunk['kind']), (ids['exit'], 'reach', 'lake'))
        self.assertEqual(trunk['cells'].tolist(), [g.cell(2, 3), g.cell(2, 4), g.cell(2, 5)])
        np.testing.assert_array_equal(trunk['xy'][0], g.reaches[ids['big']]['xy'][-1])
        np.testing.assert_array_equal(trunk['xy'][-1], exit_xy[0])         # exit head, exactly
        # The smaller inflow ends on a shared, pinned vertex of the smoothed trunk.
        self.assertEqual((branch['target'], branch['target_kind'], branch['reach_target']),
                         (ids['big'], 'connector', ids['exit']))
        self.assertEqual(branch['cells'].size, 0)
        np.testing.assert_array_equal(branch['xy'][0], g.reaches[ids['small']]['xy'][-1])
        joined = np.flatnonzero(np.all(trunk['xy'] == branch['xy'][-1], axis=1))
        self.assertEqual(joined.tolist(), trunk['pins'])
        self.assertTrue(np.isfinite(trunk['xy']).all() and np.isfinite(branch['xy']).all())
        self.assertTrue(np.all(np.diff(trunk['water_m']) <= 0))
        # The trunk is drawn once: no other link covers its cells.
        self.assertFalse(set(trunk['cells'].tolist()) & set(branch['cells'].tolist()))

    def test_closed_lakes_and_dry_ends_get_no_link(self):
        g, ids = lake_fixture()
        links = display_connectors(g)
        for name in ('closed', 'dry', 'terminal', 'exit'):
            self.assertNotIn(ids[name], links, name)
        wet = links[ids['wet']]
        self.assertEqual((wet['kind'], wet['target'], wet['cells'].tolist()),
                         ('hydraulic', ids['exit'], [g.cell(1, 7)]))
        # Joins the target's own course at the cell it reaches (a vertex of it).
        self.assertTrue(np.any(np.all(g.reaches[ids['exit']]['xy'] == wet['xy'][-1], axis=1)))
        # An open lake without outflow is closed for display as well.
        g.lakes[0]['outflow_m3s'] = 0.
        self.assertNotIn(ids['big'], display_connectors(g))

    def test_sphere_seam_stays_unwrapped_and_short(self):
        g = Grid(3, 8, topology='sphere')
        lake = g.lake([(1, 0), (1, 1)], g.cell(1, 2), outflow=4.)
        entry = g.reach([(1, 5), (1, 6), (1, 7)], 4., 'lake', g.cell(1, 0), enters_lake=lake)
        exit_ = g.reach([(1, 2), (1, 3)], 4., 'mouth', drain=g.cell(1, 1), leaves_lake=lake)
        g.reaches[entry]['downstream_id'] = exit_
        xy = display_connectors(g)[entry]['xy']
        self.assertEqual(xy[0, 0], 8000.)
        self.assertEqual(xy[-1, 0], g.reaches[exit_]['xy'][0, 0]+8000.)  # same place, one period on
        self.assertTrue(np.all(np.diff(xy[:, 0]) > 0))
        self.assertLessEqual(np.max(np.hypot(*np.diff(xy, axis=0).T)), 1000.+1e-9)

    def test_world_graph_is_not_modified(self):
        g, _ = lake_fixture()
        before = copy.deepcopy(g.reaches), copy.deepcopy(g.lakes)
        arrays = [a.copy() for a in (g.receivers, g.reach_ids, g.lake_ids, g.discharge_m3s)]
        display_connectors(g)
        for a, b in zip(before[0], g.reaches):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                np.testing.assert_array_equal(a[key], b[key])
        self.assertEqual(before[1], g.lakes)
        for a, b in zip(arrays, (g.receivers, g.reach_ids, g.lake_ids, g.discharge_m3s)):
            np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()

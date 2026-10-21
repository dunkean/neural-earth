"""Vector lakes, lake connectors, local orders and unit regimes (display only)."""
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'backend'))

import numpy as np

from terrain_hydrology_lakes import (local_connectors, local_lakes, source_lake_pieces,
                                     combine_source_lake, geometry_rings)
from terrain_hydrology_service import HydrologyService
from terrain_hydrology_balance import MONTH_SECONDS

B = 240.


def centre(c, cols=6):
    return np.array([(c % cols+.5)*B, (c//cols+.5)*B])


def pond_plan():
    """3 x 6 cells: inflow A (6, 7) and inflow B (3) enter pond {8, 9}; outlet 10."""
    receivers = np.full(18, -1)
    receivers[[6, 7, 8, 9, 10, 3]] = [7, 8, 9, 10, 11, 9]
    height = np.full((3, 6), 20.)
    height.ravel()[[8, 9]] = 5.
    flow = np.zeros(18)
    flow[[8, 9]] = 3.
    lake = dict(id='L1', cells=np.array([8, 9]), outlet_cell=10, level_m=8., area_m2=2*B*B,
                max_depth_m=3., basin_id=0, xy=centre(8)*.5+centre(9)*.5)
    return SimpleNamespace(shape=(3, 6), bounds=(0., 0., 6*B, 3*B), receivers=receivers.reshape(3, 6),
                           local_lakes=[lake], routing_height_m=height, flow_m3s=flow.reshape(3, 6))


def course(rid, cells, q, end=None):
    xy = np.array([centre(c) for c in cells])
    if end is not None:
        xy = np.vstack([xy, .5*(xy[-1]+centre(end))])
    return dict(id=rid, kind='tributary', cells=np.array(cells), xy=xy, water_m=np.linspace(12, 9, len(xy)),
                flow_m3s=q, terminal_kind='lake:L1' if end is not None else 'confluence')


class LakeConnectorTests(unittest.TestCase):
    def test_merged_inflows_follow_receivers_and_join_the_larger_course(self):
        geometry = [course('A', [6, 7], 2., end=8), course('B', [3], .5, end=9), course('out', [10, 11], 2.6)]
        links = local_connectors(pond_plan(), geometry)
        a, b = links['A'], links['B']
        # A: its shore point, the receiver path through the pond, then the outlet record's first vertex.
        np.testing.assert_allclose(a['xy'][1:-1], [centre(8), centre(9)])
        np.testing.assert_allclose(a['xy'][-1], geometry[2]['xy'][0])
        self.assertIsNone(a['target'])
        # B stops on A's first owned cell; that point is an exact vertex of A.
        self.assertEqual(b['target'], 'A')
        self.assertTrue(np.any(np.all(np.isclose(a['xy'], b['xy'][-1]), axis=1)))
        self.assertTrue(np.all(np.diff(a['water_m']) <= 0))

    def test_hidden_dry_and_evaporative_ponds_still_keep_their_connector(self):
        plan = pond_plan()
        wet, extra = dict(plan.local_lakes[0]), dict(plan.local_lakes[0], id='L2')
        dry = dict(plan.local_lakes[0], id='L3', cells=np.array([14, 15]))
        plan.local_lakes = [wet, extra, dry]
        humid = lambda xy: (np.full(len(xy), 900.), np.full(len(xy), 600.), np.full(len(xy), 300.))
        shown, hidden = local_lakes(plan, humid)
        self.assertIn('L1', shown)
        self.assertEqual(hidden['L3'], 'dry')
        self.assertGreater(len(geometry_rings(shown['L1']['geometry'])), 0)
        # Desert: E - (P - R) = 3000 mm over 0.115 km^2 loses ~0.011 m^3/s; 3 m^3/s still overflows.
        arid = lambda xy: (np.zeros(len(xy)), np.full(len(xy), 3000.), np.zeros(len(xy)))
        self.assertIn('L1', local_lakes(plan, arid)[0])
        plan.flow_m3s = np.where(plan.flow_m3s > 0, 1e-3, 0.)
        shown, hidden = local_lakes(plan, arid)
        self.assertEqual(hidden['L1'], 'evaporative')
        links = local_connectors(plan, [course('A', [6, 7], 1e-3, end=8), course('out', [10, 11], 1e-3)])
        self.assertIn('A', links)   # the river stays continuous across the hidden pond


class SourceLakeTests(unittest.TestCase):
    def world(self):
        """Lake 0 owns blocks with x < 7680 (level 10 m) in basin 0; nothing else."""
        def sample(x, y):
            x = np.asarray(x, float).ravel()
            n = x.size
            lake = np.where(x < 7680., 0, -1)
            neg = np.full(n, -1)
            return dict(cell=np.arange(n), basin_id=np.zeros(n, int), ocean=np.zeros(n, bool), lake_id=lake,
                        reach_id=neg, filled_m=np.zeros(n), owner_reach=neg, owner_lake=neg)
        return SimpleNamespace(sample=sample, reaches=[], lakes=[dict(level_m=10., basin_id=0)],
                               basins=dict(kind=np.zeros(1, np.int8)))

    def plan(self, core, height):
        halo_bounds = (core[0]-B, core[1]-B, core[2]+B, core[3]+B)
        return SimpleNamespace(bounds=core, routing_height_m=height[1:-1, 1:-1],
                               metadata=dict(routing_halo_m=height, routing_halo_bounds=halo_bounds))

    def test_base_constrains_source_lake_and_neighbouring_pieces_union(self):
        world = self.world()
        n = 32
        # One global surface: Base 0 m below the 10 m level west of x = 6 km, 50 m east of it,
        # with a 30 m island at x 2-3 km and an unconnected 0 m pit far east (x 13-14 km).
        x = (np.arange(-1, 2*n+1)+.5)*B
        y = (np.arange(-1, n+1)+.5)*B
        X, Y = np.meshgrid(x, y)
        H = np.where(X < 6000., 0., 50.)
        H[(X > 2000.) & (X < 3000.) & (Y > 2000.) & (Y < 3000.)] = 30.
        H[(X > 13000.) & (X < 14000.) & (Y > 2000.) & (Y < 3000.)] = 0.
        west, east = (0., 0., n*B, n*B), (n*B, 0., 2*n*B, n*B)
        pieces = [source_lake_pieces(world, self.plan(core, H[:, k*n:k*n+n+2]))
                  for k, core in enumerate((west, east))]
        self.assertIn(0, pieces[0])
        self.assertNotIn(0, pieces[1])             # only an unconnected pit and land there
        geom = pieces[0][0]['geometry']
        self.assertLess(geom.bounds[2], 6100.)       # stops at the Base shore, not at the block edge
        self.assertEqual(len(geom.interiors) if geom.geom_type == 'Polygon' else 1, 1)   # the island
        from shapely.geometry import box
        coarse = box(0., 0., 12000., n*B)            # a coarse 19.5 km shape spilling over Base land
        base, rest = combine_source_lake(coarse, [geom], [west, east])
        self.assertTrue(rest.is_empty)               # both cores are ready: Base is authoritative
        self.assertAlmostEqual(base.area, geom.area, places=3)
        base, rest = combine_source_lake(coarse, [geom], [west])
        self.assertAlmostEqual(rest.bounds[0], n*B, places=2)   # the coarse remainder lies outside ready cores

    def test_pieces_of_adjacent_cores_union_without_a_seam(self):
        world = self.world()
        world.sample = (lambda base: (lambda x, y: dict(base(np.zeros_like(np.asarray(x, float)), y))))(world.sample)
        n = 32
        H = np.zeros((n+2, 2*n+2))
        west, east = (0., 0., n*B, n*B), (n*B, 0., 2*n*B, n*B)
        parts = [source_lake_pieces(world, self.plan(core, H[:, k*n:k*n+n+2]))[0]['geometry']
                 for k, core in enumerate((west, east))]
        base, _ = combine_source_lake(parts[0], parts, [])
        self.assertEqual(base.geom_type, 'Polygon')
        self.assertAlmostEqual(base.area, sum(p.area for p in parts), delta=1.)

    def test_unfinished_source_shore_keeps_the_coarse_fallback(self):
        from shapely.geometry import box
        coarse, core = box(0., 0., 20., 10.), (0., 0., 10., 10.)
        partial = box(2., 2., 10., 8.)
        base, rest = combine_source_lake(coarse, [partial], [core])
        self.assertTrue(base.is_empty, 'The calculation edge must not become a rectangular shoreline')
        self.assertTrue(rest.equals(coarse))
        closed = box(2., 2., 8., 8.)
        base, rest = combine_source_lake(coarse, [closed], [core])
        self.assertTrue(base.equals(closed))
        self.assertAlmostEqual(rest.bounds[0], 10., places=3)


class HierarchyTests(unittest.TestCase):
    def test_local_strahler_is_the_package_network_order(self):
        reaches = [dict(id=k, kind='tributary', flow_m3s=1., downstream_id=d, start_link=link)
                   for k, d, link in (('a', 'c', None), ('b', 'c', None), ('c', 'e', None),
                                      ('d', 'e', 'L7'), ('e', None, None))]
        reaches.append(dict(id='t', kind='trunk', flow_m3s=50., downstream_id=None, strahler=5))
        HydrologyService._local_orders(reaches)
        order = {r['id']: r.get('strahler') for r in reaches}
        self.assertEqual(order, dict(a=1, b=1, c=2, d=1, e=2, t=5))
        self.assertTrue(reaches[4]['strahler_partial'])      # fed through a unit link
        self.assertFalse(reaches[2]['strahler_partial'])

    def test_unit_without_source_reach_gets_its_station_regime_and_keeps_its_budget(self):
        world = SimpleNamespace(reaches=[], lakes=[], topology='plane', bounds=(0., 0., 1e5, 1e5))
        def plan():
            r = dict(id='t', kind='tributary', parent_id=None, flow_m3s=2., downstream_id=None,
                     xy=np.array([[0., 0.], [500., 0.]]), water_m=np.array([10., 9.]), terminal_kind='sea')
            return SimpleNamespace(reaches=[r], links=[], local_lakes=[], metadata={})
        flat, seasonal = plan(), plan()
        HydrologyService._regional_properties(world, flat)
        weights = np.r_[np.full(6, 10.), np.full(6, 90.)]
        HydrologyService._regional_properties(world, seasonal, unit_fraction=lambda r: weights)
        q_flat = np.asarray(flat.reaches[0]['monthly_flow_m3s'])
        q = np.asarray(seasonal.reaches[0]['monthly_flow_m3s'])
        self.assertLess(np.ptp(q_flat), 1e-9)
        self.assertGreater(q[6:].min(), 3*q[:6].max())
        self.assertAlmostEqual(float(q@MONTH_SECONDS/MONTH_SECONDS.sum()), 2., places=9)   # annual budget unchanged

    def test_low_flow_months_are_not_dry_months(self):
        service = HydrologyService(lambda *_: None, cache_root=Path(os.environ.get('TEMP', '.')))
        entry = dict(world=SimpleNamespace(basins=dict(kind=np.zeros(1, np.int8), kind_names=['sea'])))
        big = dict(flow_m3s=642., area_m2=1e11, basin_id=0, xy=np.zeros((2, 2)),
                   monthly_flow_m3s=[700.]*11+[5.])
        meta = service._hierarchy(entry, big, False)
        self.assertEqual((meta['low_flow_months'], meta['dry_months'], meta['regime']), (1, 0, 'perennial'))
        wadi = dict(flow_m3s=.02, area_m2=1e8, basin_id=0, xy=np.zeros((2, 2)), monthly_flow_m3s=[.05]*4+[.001]*8)
        meta = service._hierarchy(entry, wadi, False)
        self.assertEqual((meta['dry_months'], meta['regime'], meta['outlet_kind']), (8, 'intermittent', 'sea'))


if __name__ == '__main__':
    unittest.main()

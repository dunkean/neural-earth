"""Global hydrology authority: routing, depressions, lakes, reaches, topology."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import math
import unittest

import numpy as np

from terrain_hydrology import SECONDS_PER_YEAR, build_hydrology
import terrain_hydrology_routing as hr

WORLD = (-20e6, -10e6, 20e6, 10e6)


def plane_bounds(h, w, cell=1000.):
    return (0., 0., w * cell, h * cell)


def check_invariants(test, hw):
    """Shared structural checks: complete stack, closed budget, lake topology."""
    n = hw.receivers.size
    rcv = hw.receivers.ravel()
    test.assertEqual(hw.order.size, n)
    rank = np.empty(n, np.int64)
    rank[hw.order] = np.arange(n)
    has = rcv >= 0
    test.assertTrue(np.all(rank[rcv[has]] < rank[has]), 'receiver after donor')
    test.assertEqual(hw.metadata['counts']['unresolved_flat_cells'], 0)
    budget = hw.metadata['budget']
    test.assertLess(abs(budget['conservation_error_m3s']), 1e-9 * max(budget['runoff_m3s'], 1.))
    test.assertLess(hw.metadata['closed_lake_routing_mismatch_max_rel'], 1e-9)
    lake = hw.lake_ids.ravel()
    leaving = np.flatnonzero((lake >= 0) & has)
    leaving = leaving[lake[rcv[leaving]] != lake[leaving]]
    outlets = np.array([hw.lakes[i]['outlet_cell'] for i in lake[leaving]], np.int64)
    test.assertTrue(np.array_equal(rcv[leaving], outlets), 'lake water leaves only at its sill')
    for reach in hw.reaches:
        test.assertEqual(reach['xy'].shape, (reach['water_m'].size, 2))
        test.assertTrue(np.all(np.diff(reach['water_m']) <= 1e-9), 'water profile rises')
        test.assertTrue(np.all(lake[reach['cells']] < 0), 'reach ribbon inside a lake')


class HydrologyTests(unittest.TestCase):
    def test_ranked_flats_acyclic_and_deterministic_replay(self):
        h, w = 20, 30
        z = np.full((h, w), 10.)
        z[0], z[-1] = 20., 20.
        z[:, 0] = -5.
        hw = build_hydrology(z, bounds=plane_bounds(h, w), topology='plane', plane_edges='closed',
                             channel_min_area_m2=4e6, channel_min_discharge_m3s=0.)
        check_invariants(self, hw)
        rcv = hw.receivers
        rows, cols = np.mgrid[1:h - 1, 2:w]
        # Flat cells drain one ring closer to the coast (towards-lower rank) and
        # away from the raised rows (away-from-higher rank).
        self.assertTrue(np.all(rcv[rows, cols] % w == cols - 1))
        self.assertTrue(np.all(rcv[1, 2:] // w >= 1) and np.all(rcv[-2, 2:] // w <= h - 2))
        self.assertTrue(np.all(hw.terminal_kind[:, 0] == hr.TERM_OCEAN))

        rng = np.random.default_rng(7)
        rough = rng.normal(0, 1, (48, 96)).cumsum(0).cumsum(1)
        rough = (rough - rough.mean()) / rough.std() * 1500
        kw = dict(bounds=WORLD, channel_min_area_m2=2e10, channel_min_discharge_m3s=1.)
        a, b = build_hydrology(rough, **kw), build_hydrology(rough.copy(), **kw)
        check_invariants(self, a)
        self.assertGreater(a.metadata['counts']['closed_lakes'], 0)
        self.assertEqual(a.metadata['hydro_id'], b.metadata['hydro_id'])
        for name in ('receivers', 'order', 'filled_m', 'basin_ids', 'lake_ids', 'owner_reach'):
            self.assertTrue(np.array_equal(getattr(a, name), getattr(b, name)), name)
        self.assertEqual([r['key'] for r in a.reaches], [r['key'] for r in b.reaches])
        c = build_hydrology(rough, seed=1, **kw)
        self.assertTrue(np.array_equal(a.receivers, c.receivers))
        self.assertNotEqual(a.metadata['hydro_id'], c.metadata['hydro_id'])
        self.assertNotEqual(a.reaches[0]['key'], c.reaches[0]['key'])

    def test_tied_sills_follow_the_cascade(self):
        # Integer rasters put lake levels, sills and dry cells at equal heights.
        closed = dict(topology='plane', plane_edges='closed', channel_min_area_m2=1,
                      channel_min_discharge_m3s=0.)
        # A spilling lake whose sill drains into a lake at the same level
        # (review fixture: formerly a receiver cycle at evaporation 3000).
        z = np.array([[2, 3, 0, 0, 3, 2, 1], [1, 3, 3, 3, 1, 1, 1],
                      [3, 1, 2, 0, 3, 0, 0], [2, 1, 1, 2, 2, 3, 2]], float)
        for evap in (0., 300., 3000.):
            hw = build_hydrology(z, bounds=plane_bounds(4, 7), evaporation_mm_year=evap,
                                 min_ocean_area_m2=1e30, **closed)
            check_invariants(self, hw)
        # One lake meets itself only at its own sill; both parts drain there.
        z = np.array([[3, 3, 0, 3, 3], [2, 3, 2, 2, 1], [3, 0, 3, 0, 2], [2, 2, 3, 3, 2]], float)
        hw = build_hydrology(z, bounds=plane_bounds(4, 5), evaporation_mm_year=3000.,
                             runoff_mm_year=550., min_ocean_area_m2=1e30, **closed)
        check_invariants(self, hw)
        lake = hw.lakes[0]
        self.assertEqual(set(hw.receivers.ravel()[[2, 11]]), {lake['outlet_cell']})
        # Two open lakes share one sill; the lake that contains it decides its
        # receiver, so the other lake's water cannot loop back through it.
        z = np.array([[0, 0, 2, 2, 3, 1], [2, 3, 3, 0, 3, 2], [2, 0, 0, 1, 2, 2],
                      [0, 0, 0, 2, 3, 0], [3, 3, 0, 0, 2, 2]], float)
        evap = np.zeros(z.shape)
        evap[0, 5] = 5000.
        hw = build_hydrology(z, bounds=plane_bounds(5, 6), runoff_mm_year=100., evaporation_mm_year=evap,
                             min_ocean_area_m2=1e30, **closed)
        check_invariants(self, hw)
        self.assertEqual([lk['outlet_cell'] for lk in hw.lakes], [16, 16])
        self.assertEqual(hw.downstream_path(16)[-1], 5)
        # Dry cells at the lake level drain into the lake that the cascade
        # credited, so its outflow reaches the sill instead of bypassing it.
        z = np.array([[0, 2, 1, -1, 3], [3, 2, 3, 3, 3], [2, 0, 0, 2, -1]], float)
        hw = build_hydrology(z, bounds=plane_bounds(3, 5), min_ocean_area_m2=1., **closed)
        check_invariants(self, hw)
        q = hw.discharge_m3s.ravel()
        for lk in hw.lakes:
            self.assertFalse(lk['closed'])
            self.assertAlmostEqual(q[lk['outlet_cell']], lk['outflow_m3s'], places=6)
        # Quantized smooth terrain: plateaus, tied sills and clamped sea level.
        from scipy.ndimage import gaussian_filter
        for trial in range(6):
            rng = np.random.default_rng(trial)
            rough = gaussian_filter(rng.normal(size=(32, 64)), 2, mode='wrap') * 800 + 100
            hw = build_hydrology(np.maximum(np.round(rough / 10) * 10, 0.), bounds=WORLD,
                                 runoff_mm_year=rng.uniform(0, 600, (32, 64)),
                                 evaporation_mm_year=1000., channel_min_area_m2=1,
                                 channel_min_discharge_m3s=0.)
            check_invariants(self, hw)

    def test_spherical_area_and_runoff_discharge(self):
        h, w = 32, 64
        sphere = build_hydrology(np.full((h, w), 100.), bounds=WORLD)
        radius = sphere.metadata['planet_radius_m']
        self.assertAlmostEqual(radius, 40e6 / (2 * math.pi))
        self.assertAlmostEqual(sphere.cell_area_m2.sum() / (4 * math.pi * radius**2), 1., places=12)
        self.assertGreater(sphere.cell_area_m2[h // 2, 0], 20 * sphere.cell_area_m2[0, 0])

        y, x = np.mgrid[0:h, 0:w]
        z = 2. * x + .3 * np.abs(y - h / 2)
        z[:, :2] = -30.
        runoff = 100. + 10. * x
        hw = build_hydrology(z, bounds=plane_bounds(h, w), topology='plane', plane_edges='closed',
                             runoff_mm_year=runoff, channel_min_area_m2=1e7, channel_min_discharge_m3s=0.)
        check_invariants(self, hw)
        land = ~hw.ocean
        yield_m3s = (runoff / 1000. * hw.cell_area_m2 / SECONDS_PER_YEAR)[land]
        basins = hw.basin_ids[land]
        roots = hw.basins['root_cell']
        expect_q = np.bincount(basins, yield_m3s, roots.size)
        expect_a = np.bincount(basins, hw.cell_area_m2[land], roots.size)
        np.testing.assert_allclose(hw.discharge_m3s.ravel()[roots], expect_q, rtol=1e-6)
        np.testing.assert_allclose(hw.area_m2.ravel()[roots], expect_a, rtol=1e-6)
        self.assertAlmostEqual(hw.metadata['budget']['to_sea_m3s'], yield_m3s.sum(), places=6)
        self.assertAlmostEqual(hw.area_m2[hw.ocean].sum(), hw.cell_area_m2[land].sum(), delta=1.)

    def test_dry_closed_basin_versus_sea_and_humid_spill(self):
        h, w = 40, 60
        y, x = np.mgrid[0:h, 0:w]
        z = 50. + .5 * x - 80. * np.exp(-((x - 40.)**2 + (y - 20.)**2) / 30.)
        z -= 30. * np.exp(-((x - 56.)**2 + (y - 20.)**2) / 8.)
        z[:, :6] = -50.
        bowl, upper = 20 * w + 40, 20 * w + 56
        self.assertLess(z.flat[bowl], 0)
        kw = dict(bounds=plane_bounds(h, w), topology='plane', plane_edges='closed',
                  channel_min_area_m2=3e6, channel_min_discharge_m3s=0., min_lake_depth_m=1.)
        # The upper lake does not evaporate, fills and spills into the arid bowl,
        # whose basin touched the sea-connected set at a lower pass.
        arid = build_hydrology(z, runoff_mm_year=20., evaporation_mm_year=np.where(x > 50, 0., 2500.), **kw)
        humid = build_hydrology(z, runoff_mm_year=1500., evaporation_mm_year=300., **kw)
        for hw in (arid, humid):
            check_invariants(self, hw)
            self.assertFalse(hw.ocean.flat[bowl], 'inland depression became sea')
            self.assertEqual(hw.metadata['ocean']['inland_below_sea_components'], 1)
        path = arid.downstream_path(bowl)
        self.assertEqual(arid.terminal_kind.flat[path[-1]], hr.TERM_SINK)
        self.assertEqual(arid.downstream_path(upper)[-1], path[-1])
        basin = arid.basin_ids.flat[bowl]
        self.assertEqual(arid.basins['kind'][basin], 1)
        leaf = arid.hierarchy['leaf'].ravel()[upper]
        self.assertTrue(arid.hierarchy['to_outlet'][leaf], 'fixture must link via the sea set')
        spill = arid.lakes[arid.lake_ids.flat[upper]]
        self.assertFalse(spill['closed'])
        self.assertGreater(spill['outflow_m3s'], 0.)
        for lk in arid.lakes:
            if lk['closed']:
                self.assertLess(lk['level_m'], lk['spill_level_m'])
                self.assertAlmostEqual(lk['inflow_m3s'], lk['evaporation_m3s'] + lk['residual_m3s'])
        self.assertTrue(arid.lakes[arid.lake_ids.flat[bowl]]['closed'])
        lake = humid.lakes[humid.lake_ids.flat[bowl]]
        self.assertFalse(lake['closed'])
        self.assertEqual(lake['level_m'], lake['spill_level_m'])
        self.assertAlmostEqual(lake['outflow_m3s'], lake['inflow_m3s'] - lake['evaporation_m3s'])
        self.assertTrue(humid.ocean.flat[humid.downstream_path(bowl)[-1]])
        self.assertAlmostEqual(humid.discharge_m3s.flat[lake['outlet_cell']], lake['outflow_m3s'], places=4)

        dry = build_hydrology(z + 5000., **kw)
        check_invariants(self, dry)
        self.assertEqual(dry.metadata['counts']['sea_cells'], 0)
        self.assertEqual(dry.metadata['budget']['to_sea_m3s'], 0.)
        self.assertTrue(np.all(np.isin(dry.terminal_kind[dry.receivers < 0], (hr.TERM_SINK,))))

    def test_lake_node_confluence_and_incremental_ownership(self):
        h, w = 61, 81
        y, x = np.mgrid[0:h, 0:w].astype(float)
        main = np.abs(y - 30)
        t = np.clip(((x - 55) * 25 - (y - 30) * 30) / (25**2 + 30**2), 0, 1)
        trib = np.hypot(x - (55 + 25 * t), y - (30 - 30 * t))
        trib[x < 55] = np.inf
        z = 5. + 2. * x + 4. * np.minimum(main, trib)
        z -= 25. * np.exp(-((x - 30)**2 / 40. + (y - 30)**2 / 10.))
        z[:, :3] = -20.
        hw = build_hydrology(z, bounds=plane_bounds(h, w), topology='plane', plane_edges='closed',
                             runoff_mm_year=600., evaporation_mm_year=800.,
                             channel_min_area_m2=4e7, channel_min_discharge_m3s=0.)
        check_invariants(self, hw)
        self.assertEqual(len(hw.lakes), 1)
        lake = hw.lakes[0]
        self.assertFalse(lake['closed'])
        mask = hw.lake_ids.ravel() == 0
        self.assertEqual(hr.label_components(mask, h, w, False)[1], 1)
        exit_reach = hw.reaches[lake['exit_reach']]
        self.assertEqual(exit_reach['leaves_lake'], 0)
        self.assertEqual(exit_reach['water_m'][0], lake['level_m'])
        self.assertTrue(lake['entry_reaches'])
        for r in lake['entry_reaches']:
            self.assertEqual(hw.reaches[r]['target'], lake['node_id'])
            self.assertEqual(hw.reaches[r]['downstream_id'], lake['exit_reach'])
        joins = [r for r in hw.reaches if hw.nodes[r['source']]['kind'] == 'confluence']
        conf = max(joins, key=lambda r: r['area_m2'])
        self.assertEqual(len(conf['upstream_ids']), 2)
        self.assertEqual(conf['enters_lake'], 0)
        for u in conf['upstream_ids']:
            self.assertEqual(hw.reaches[u]['target'], conf['source'])
            np.testing.assert_allclose(hw.reaches[u]['xy'][-1], conf['xy'][0])
        self.assertGreaterEqual(conf['strahler'], 2)

        mouth = max((r for r in hw.reaches if r['end'] == 'mouth'), key=lambda r: r['area_m2'])
        units, stack, lakes = set(), [mouth['id']], set()
        while stack:
            r = stack.pop()
            units.add(r)
            if hw.reaches[r]['leaves_lake'] is not None:
                lakes.add(hw.reaches[r]['leaves_lake'])
            stack += hw.reaches[r]['upstream_ids']
        area = hw.cell_area_m2.ravel()
        owned = sum(area[hw.catchment_cells(r)].sum() for r in units)
        owned += sum(area[hw.lake_catchment_cells(k)].sum() for k in lakes)
        self.assertAlmostEqual(owned / mouth['area_m2'], 1., places=6)
        both = (hw.owner_reach >= 0) & (hw.owner_lake >= 0)
        self.assertFalse(both.any())
        cells, steps = hw.corridor_cells(conf['id'], 2)
        self.assertTrue(np.all(hw.basin_ids.ravel()[cells] == conf['basin_id']))
        self.assertEqual(steps.max(), 2)

    def test_sphere_seam_pole_region_and_plane_edges(self):
        h, w = 32, 64
        rows, cols = np.mgrid[0:h, 0:w]
        z = 10. + 20. * ((cols - 50) % w) + 30. * np.abs(rows - 15)
        z[(rows < 8) | (rows > 22)] = -100.
        z[:, 50] = -100.
        hw = build_hydrology(z, bounds=WORLD, channel_min_area_m2=1e11, channel_min_discharge_m3s=0.)
        check_invariants(self, hw)
        dx = 40e6 / w
        seam = [r for r in hw.reaches if np.isin(r['cells'] % w, (0,)).any()
                and np.isin(r['cells'] % w, (w - 1,)).any()]
        self.assertTrue(seam, 'no reach crosses the antimeridian')
        for r in seam:
            self.assertLessEqual(np.abs(np.diff(r['xy'][:, 0])).max(), 1.01 * dx)
        region = hw.region((WORLD[2] - 3 * dx, WORLD[1] + 12 * 20e6 / h,
                            WORLD[2] + 3 * dx, WORLD[1] + 19 * 20e6 / h))
        self.assertEqual(sorted(set(region['cells'].ravel() % w)), [0, 1, 2, w - 3, w - 2, w - 1])
        portals = region['portals']
        main_in = portals['discharge_m3s'][portals['direction'] < 0].max()
        main_out = portals['discharge_m3s'][portals['direction'] > 0].max()
        self.assertGreater(main_out, main_in)
        inside = np.isin(portals['to_cell'], region['cells'])
        self.assertTrue(np.array_equal(inside, portals['direction'] < 0))
        self.assertTrue(np.array_equal(np.isin(portals['from_cell'], region['cells']), ~inside))

        # A tilted cap: the steepest route from 45N on lon 0 runs over the pole.
        h, w = 24, 48
        lat = np.radians(90 - (np.arange(h) + .5) * 180 / h)[:, None]
        lon = ((np.arange(w) + .5) / w * 2 * np.pi - np.pi)[None, :]
        z = 1000. + 800. * np.cos(lat) * np.cos(lon)
        sea = np.broadcast_to(lat < np.radians(10), z.shape)
        z = np.where(sea, -100., z)
        cap = build_hydrology(z, bounds=WORLD, ocean_mask=sea, channel_min_area_m2=1e12,
                              channel_min_discharge_m3s=0.)
        check_invariants(self, cap)
        self.assertFalse(np.any(cap.terminal_kind[0] != hr.TERM_NONE), 'pole row is an outlet')
        start = 3 * w + w // 2
        path = cap.downstream_path(start)
        prow, pcol = np.divmod(path, w)
        crossing = np.flatnonzero((prow[:-1] == 0) & (prow[1:] == 0)
                                  & (np.abs(pcol[1:] - pcol[:-1]) >= w // 2 - 1))
        self.assertTrue(crossing.size, 'route did not cross the pole')
        self.assertTrue(sea.ravel()[path[-1]])
        self.assertTrue(any(r['crosses_pole'] for r in cap.reaches))

        h, w = 12, 16
        tilt = np.add.outer(np.zeros(h), 5. * np.arange(w)) + 1.
        common = dict(bounds=plane_bounds(h, w), topology='plane', channel_min_area_m2=1e7,
                      channel_min_discharge_m3s=0.)
        open_edge = build_hydrology(tilt, plane_edges='outlet', **common)
        closed_edge = build_hydrology(tilt, plane_edges='closed', **common)
        check_invariants(self, open_edge)
        check_invariants(self, closed_edge)
        budget = open_edge.metadata['budget']
        self.assertAlmostEqual(budget['to_plane_edge_m3s'], budget['runoff_m3s'])
        self.assertTrue(np.all(open_edge.terminal_kind[:, 0] == hr.TERM_EDGE))
        self.assertEqual(closed_edge.metadata['budget']['to_plane_edge_m3s'], 0.)
        self.assertFalse(np.any(closed_edge.terminal_kind == hr.TERM_EDGE))


if __name__ == '__main__':
    unittest.main()

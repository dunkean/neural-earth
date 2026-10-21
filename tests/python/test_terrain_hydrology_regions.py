"""Regional 240 m refinement inside inherited global ownership."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'backend'))

from terrain_hydrology import build_hydrology  # noqa: E402
import terrain_hydrology_regions as hreg  # noqa: E402
from terrain_hydrology_profiles import build_profiles  # noqa: E402

SRC = 19200.0            # source cell: 2.5 coarse blocks, deliberately not aligned
B = hreg.COARSE_M
VALLEY_Y = 8.5 * SRC     # parent trunk row centre
SHIFT = 2400.0           # base valley offset from the parent/coarse valley


def macro(x, y, valley=VALLEY_Y):
    return 2000 - 0.002 * x + 0.01 * np.sqrt((y - valley) ** 2 + 1000.0 ** 2)


def make_world(z_fn=macro, shape=(16, 32)):
    h, w = shape
    x = (np.arange(w) + .5) * SRC
    y = (np.arange(h) + .5) * SRC
    gx, gy = np.meshgrid(x, y)
    return build_hydrology(z_fn(gx, gy), bounds=(0, 0, w * SRC, h * SRC), topology='plane',
                           channel_min_area_m2=2e9, channel_min_discharge_m3s=.1)


def grid(fn, bounds, cell):
    x0, y0, x1, y1 = bounds
    x = x0 + (np.arange(int(round((x1 - x0) / cell))) + .5) * cell
    y = y0 + (np.arange(int(round((y1 - y0) / cell))) + .5) * cell
    gx, gy = np.meshgrid(x, y)
    return fn(gx, gy)


def coarse_fn(x, y):
    return macro(x, y)


def base_fn(x, y):
    # Opposite macro tilt (coarse and base disagree strongly) plus a narrow
    # valley 2.4 km south of the coarse/parent valley and a closed bowl.
    detail = (-30 * np.exp(-((y - VALLEY_Y - SHIFT) / 700.0) ** 2)
              - 25 * np.exp(-((x - 337000) ** 2 + (y - VALLEY_Y + 6000) ** 2) / 900.0 ** 2))
    return 1500 + 0.003 * x + 0.004 * np.abs(y - VALLEY_Y) + detail


def core(col0, cols=8, row0=17, rows=8):
    return (col0 * B, row0 * B, (col0 + cols) * B, (row0 + rows) * B)


def refine(world, core_bounds, *, extra=0, coarse=coarse_fn, base=base_fn, **options):
    base_b, coarse_b = hreg.piece_windows(core_bounds)
    grow = lambda b: (b[0] - extra * B, b[1] - extra * B, b[2] + extra * B, b[3] + extra * B)
    base_b, coarse_b = grow(base_b), grow(coarse_b)
    return hreg.refine_region(world, grid(coarse, coarse_b, B), coarse_b,
                              grid(base, base_b, hreg.BASE_M), base_b, core_bounds, seed=7, **options)


def main_trunk(plan):
    """Trunk segments from the west-edge valley portal downstream through the core."""
    by_id = {r['id']: r for r in plan.reaches}
    west = next(p['id'] for p in plan.portals if p['axis'] == 'x' and p['line_m'] == plan.bounds[0])
    chain = [next(r for r in plan.reaches if r.get('start_portal') == west)]
    while chain[-1]['downstream_id'] in by_id:
        chain.append(by_id[chain[-1]['downstream_id']])
    return chain


class RegionRefinementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.world = make_world()
        cls.core_a, cls.core_b = core(40), core(48)
        cls.plan_a = refine(cls.world, cls.core_a)

    def test_shifted_base_valley_followed_inside_parent_corridor(self):
        plan, q = self.plan_a, self.plan_a.metadata['quality']
        chain = main_trunk(plan)
        xy = np.vstack([r['xy'] for r in chain])
        water = np.concatenate([r['water_m'] for r in chain])
        # Base valley is followed, not the parent line, not the opposite base macro slope.
        self.assertLess(abs(np.median(xy[:, 1]) - (VALLEY_Y + SHIFT)), hreg.BASE_M)
        self.assertLess(np.max(np.abs(xy[:, 1] - VALLEY_Y)), plan.metadata['corridor_m'])
        self.assertTrue(np.all(np.diff(water) <= 1e-9))
        self.assertGreater(water[0] - water[-1], 50.0)       # coarse eastward fall kept
        # Coarse block means survive the aligned residual composition.
        self.assertLess(q['block_mean_error_max_m'], 3.0)
        self.assertGreater(q['coarse_base_offset_rms_m'], 100.0)
        # Rough (white) 30 m deficit: smooth correction, bounded mean error, no block terraces.
        rng = np.random.default_rng(0)
        fine = rng.normal(0, 5, (640, 640))
        parent = fine.reshape(20, 32, 20, 32).mean(axis=(1, 3)) + rng.normal(0, 30, (20, 20))
        out = hreg.aligned_detail(parent, (0, 0), fine, (0, 0), 32, (4, 4, 16, 16))
        err = out.reshape(12, 32, 12, 32).mean(axis=(1, 3)) - parent[4:16, 4:16]
        self.assertLess(np.abs(err).max(), 4.0)
        jump = np.abs(np.diff(out - fine[128:-128, 128:-128], axis=1))
        self.assertLessEqual(jump[:, 31::32].max(), np.delete(jump, np.s_[31::32], axis=1).max())
        self.assertEqual(q['strong_corrections'], 0)
        self.assertTrue(all(r['diagnostics']['stage'] == 'corridor' for r in chain))

    def test_portals_shared_by_neighbours_independent_of_order_and_window(self):
        world = self.world
        b_first = refine(world, self.core_b)
        a_second = refine(world, self.core_a, extra=2)       # larger cache window
        a_ids = {p['id']: p for p in self.plan_a.portals}
        b_ids = {p['id']: p for p in b_first.portals}
        shared = sorted(set(a_ids) & set(b_ids))
        self.assertTrue(shared)
        for pid in shared:
            pa, pb = a_ids[pid], b_ids[pid]
            for k in ('xy', 'level_m', 'tangent', 'flow_m3s', 'cells', 'area_m2'):
                np.testing.assert_array_equal(np.asarray(pa[k]), np.asarray(pb[k]), err_msg=k)
        exit_a = main_trunk(self.plan_a)[-1]
        entry_b = main_trunk(b_first)[0]
        self.assertEqual(exit_a['downstream_id'], entry_b['id'])
        np.testing.assert_array_equal(exit_a['xy'][-1], entry_b['xy'][0])   # the portal point
        self.assertEqual(exit_a['water_m'][-1], entry_b['water_m'][0])
        self.assertEqual(exit_a['flow_m3s'], entry_b['flow_profile_m3s'][0])
        # Same core from a larger window: identical package.
        np.testing.assert_array_equal(self.plan_a.conditioned_base_m, a_second.conditioned_base_m)
        self.assertEqual([r['id'] for r in self.plan_a.reaches], [r['id'] for r in a_second.reaches])
        self.assertEqual(self.plan_a.metadata['region_id'], a_second.metadata['region_id'])

    def test_basin_labels_not_stolen_by_conflicting_fine_relief(self):
        north, south = 4.5 * SRC, 11.5 * SRC
        two = lambda x, y: 2000 - 0.002 * x + 0.01 * np.minimum(np.abs(y - north), np.abs(y - south))
        world = make_world(two)
        divide = 8 * SRC
        # Coarse divide sits 12 km further north than the parent divide, and the
        # base adds a valley that cuts straight across it.
        coarse = lambda x, y: two(x, y + 12000 * (np.abs(y - divide) < 30000))
        base = lambda x, y: coarse(x, y) - 40 * np.exp(-((x - 330000) / 900.0) ** 2) * (np.abs(y - divide) < 40000)
        plan = refine(world, core(40, rows=8, row0=16), coarse=coarse, base=base)
        q = plan.metadata['quality']
        self.assertEqual(q['cross_basin_links'], 0)
        self.assertGreater(q['divide_disagreement'], 0.0)     # the conflict is measured, not hidden
        rcv = plan.receivers.ravel()
        basin = plan.basin_ids.ravel()
        net = np.flatnonzero((plan.drain_class.ravel() == hreg.NETWORK) & (rcv >= 0))
        self.assertTrue(np.all(basin[net] == basin[rcv[net]]))
        for r in plan.reaches:
            self.assertTrue(np.all(basin[r['cells']] == r['basin_id']), r['id'])
        self.assertTrue(all(s['foreign_basin_cells'] == 0 for s in q['spans']))

    def test_flow_partition_conserves_parent_and_has_no_tile_edge_outlets(self):
        plan, q = self.plan_a, self.plan_a.metadata['quality']
        self.assertLess(abs(q['yield_residual_m3s']), 1e-9 * max(q['yield_m3s'], 1))
        chain = main_trunk(plan)
        entry = next(p for p in plan.portals if p['id'] == chain[0]['start_portal'])
        exit_ = next(p for p in plan.portals if p['id'] == chain[-1]['end_portal'])
        flows = np.concatenate([r['flow_profile_m3s'] for r in chain])
        self.assertEqual(flows[0], entry['flow_m3s'])
        self.assertEqual(flows[-1], exit_['flow_m3s'])
        self.assertTrue(np.all(np.diff(flows) >= -1e-9))
        ids = {r['id'] for r in plan.reaches}
        tribs = [r for r in plan.reaches if r['kind'] == 'tributary']
        self.assertTrue(tribs)
        rows, cols = plan.shape
        for r in tribs:
            self.assertLessEqual(r['flow_m3s'], flows[-1])
            self.assertTrue(np.all(np.diff(r['water_m']) <= 1e-9))
            if r['downstream_id'] is None:
                self.assertTrue(r['target'] in ('sea', 'terminal') or r['target'].startswith('parent_lake'),
                                r['target'])
            else:
                self.assertTrue(r['downstream_id'] in ids or r['downstream_id'].endswith('#src')
                                or r['end_link'] is not None)
        # Local tributaries never leave through the core edge.
        self.assertFalse(any(r['target'] == 'edge' for r in tribs))
        # Flow-through local lake in the closed bowl, linked into the network.
        self.assertTrue(plan.local_lakes)
        lake = max(plan.local_lakes, key=lambda lk: lk['area_m2'])
        self.assertFalse(lake['closed'])
        self.assertIsNotNone(lake['exit_reach'])
        self.assertIn(lake['exit_reach'], ids)
        self.assertGreater(lake['max_depth_m'], 5.0)
        # Conditioned surface: never raised, idempotent projection.
        self.assertTrue(np.all(plan.conditioned_base_m <= plan.routing_height_m))
        np.testing.assert_array_equal(np.minimum(plan.conditioned_base_m, plan.upper_m),
                                      plan.conditioned_base_m)

    def test_unit_fragment_crossing_cut_links_into_neighbour_reach(self):
        # Source column 19 straddles the cut x = 48 blocks; its north and south
        # tributary reaches (19, 83) lie wholly east of the cut, in piece B.
        world, plan_a = self.world, self.plan_a
        self.assertTrue(all(np.all(world.reaches[r]['xy'][:, 0] > self.core_b[0]) for r in (19, 83)))
        plan_b = refine(world, self.core_b)
        a_first = refine(world, self.core_a, extra=1)          # other order and window
        for plan in (plan_a, plan_b):
            q = plan.metadata['quality']
            self.assertEqual(q['orphan_area_m2'], 0.0)          # no undrawn fragment
            self.assertLess(abs(q['yield_residual_m3s']), 1e-9)
            self.assertLess(q['link_flow_mismatch_m3s'], 1e-9)
        links_a = {r['id']: r for r in plan_a.links}
        links_b = {r['id']: r for r in plan_b.links}
        self.assertEqual(sorted(links_a), sorted(r['id'] for r in a_first.links))
        crossing = [r for r in plan_a.links if r['downstream_block'][1] == 48]
        self.assertEqual(sorted(r['unit_key'] for r in crossing), [19, 83])
        by_id = {r['id']: r for r in plan_b.reaches}
        for link in crossing:
            other = links_b[link['id']]
            for k in ('xy', 'level_m', 'flow_m3s', 'area_m2', 'upstream_cell', 'downstream_cell',
                      'downstream_tributary'):
                np.testing.assert_array_equal(np.asarray(link[k]), np.asarray(other[k]), err_msg=k)
            # The tributary continues: same point, downstream id, flow carried exactly once.
            ups = [r for r in plan_a.reaches if r.get('end_link') == link['id']]
            down = by_id[link['downstream_tributary']]
            self.assertTrue(ups)
            self.assertEqual(down['start_link'], link['id'])
            for up in ups:
                self.assertEqual(up['downstream_id'], down['id'])
                np.testing.assert_array_equal(up['xy'][-1], down['xy'][0])
                self.assertEqual(up['water_m'][-1], link['level_m'])
                self.assertEqual(up['water_m'][-1], down['water_m'][0])
                self.assertLessEqual(up['flow_m3s'], link['flow_m3s'] + 1e-9)
            self.assertGreaterEqual(down['flow_m3s'], link['flow_m3s'] - 1e-9)
            self.assertEqual(down['parent_id'], link['unit_key'])
        self.assertGreaterEqual(plan_b.metadata['quality']['link_inflow_m3s'],
                                sum(r['flow_m3s'] for r in crossing) - 1e-9)
        # Shared envelope contract: no conditioning step across the cut.
        step = ((plan_a.conditioned_base_m[:, -1] - plan_b.conditioned_base_m[:, 0])
                - (plan_a.routing_height_m[:, -1] - plan_b.routing_height_m[:, 0]))
        self.assertLess(np.abs(step).max(), 1e-9)

    def test_canonical_profiles_keep_bumpy_neighbours_and_link_joins_descending(self):
        world = self.world
        # A strong downstream neural bump makes independent node/portal ground
        # heights rise; the coarse/global constraint must instead report incision.
        bump = lambda x, y: coarse_fn(x, y) + 500 * np.exp(-((x - 49 * B) / (5 * B)) ** 2)
        required = set(hreg.required_profile_reaches(world, self.core_a))
        required.update(hreg.required_profile_reaches(world, self.core_b))
        profiles = build_profiles(world, range(len(world.reaches)), bump)
        self.assertTrue(required.issubset(profiles.profiles))
        b = refine(world, self.core_b, coarse=bump, water_profiles=profiles)
        a = refine(world, self.core_a, coarse=bump, water_profiles=profiles)
        self.assertGreater(profiles.metadata['max_incision_m'], 30.)
        for plan in (a, b):
            by_id = {r['id']: r for r in plan.reaches}
            for link in plan.links:
                if (plan.bounds[0] <= link['downstream_cell'][1] * hreg.BASE_M < plan.bounds[2]
                        and plan.bounds[1] <= link['downstream_cell'][0] * hreg.BASE_M < plan.bounds[3]
                        and link['channel'] and not link['downstream_sea']):
                    self.assertIn(link['downstream_tributary'], by_id)
            for r in plan.reaches:
                self.assertTrue(np.all(np.diff(r['water_m']) <= 1e-9), r['id'])
                down = by_id.get(r['downstream_id'])
                if down is not None and r['kind'] == 'tributary' and r['target'] in ('trunk', 'confluence'):
                    at = np.flatnonzero(np.all(down['xy'] == r['xy'][-1], axis=1))
                    self.assertTrue(len(at), r['id'])
                    self.assertEqual(r['water_m'][-1], down['water_m'][at[0]])
            for portal in plan.portals:
                self.assertEqual(portal['level_m'], profiles.level(portal['reach_id'], portal['station']))
        by_portal = {p['id']: p for p in b.portals}
        shared = [p for p in a.portals if p['id'] in by_portal]
        self.assertTrue(shared)
        for p in shared:
            self.assertEqual(p['level_m'], by_portal[p['id']]['level_m'])
        for link in a.links:
            if link['id'] not in {x['id'] for x in b.links} or not link['channel']:
                continue
            ups = [r for r in a.reaches if r.get('end_link') == link['id']]
            downs = [r for r in b.reaches if r.get('start_link') == link['id']]
            for up in ups:
                for down in downs:
                    self.assertEqual(up['water_m'][-1], down['water_m'][0])
                    self.assertEqual(up['water_m'][-1], link['level_m'])

    def test_high_upstream_absorber_cannot_raise_own_unit_links(self):
        # The real seed42 failure had an own receiver near110m but a remote
        # upstream absorber near595m. The latter must become linked instead.
        def region(bounds, high):
            bb, cb = hreg.piece_windows(bounds)
            profiles = SimpleNamespace(level=lambda rid, station: high if rid == 50 else 100.)
            return hreg._Region(self.world, grid(coarse_fn, cb, B),
                (round(cb[1] / B), round(cb[0] / B)), grid(base_fn, bb, 240.),
                (round(bb[1] / 240.), round(bb[0] / 240.)), bounds,
                dict(hreg.DEFAULTS), profiles)

        low_override, _, low_diag = hreg._unit_links(region(self.core_a, 90.))
        self.assertIn(50, low_override.values())   # compatible absorption is retained
        override, links, diag = hreg._unit_links(region(self.core_a, 600.))
        self.assertNotIn(50, override.values())
        self.assertGreater(diag['rejected_high_absorber_candidates'], 0)
        self.assertEqual(diag['unresolved_authority_blocks'], 0)
        self.assertEqual(diag['stranded'], 0)
        ours = [r for r in links if r['unit_key'] == 51]
        self.assertTrue(ours)
        other = {r['id']: r for r in hreg._unit_links(region(self.core_b, 600.))[1]}
        for link in ours:
            self.assertEqual(link['level_m'], 100.)
            for field in ('level_m', 'flow_m3s', 'area_m2', 'xy', 'upstream_cell', 'downstream_cell'):
                np.testing.assert_array_equal(link[field], other[link['id']][field])

        # A disconnected ownership component with no own or acceptable sink
        # emits no invented links and exposes its unresolved authority.
        original = hreg._reach_blocks
        original_ids = hreg._unit_profile_ids
        with patch.object(hreg, '_reach_blocks', side_effect=lambda world, rid:
                          set() if rid == 51 else original(world, rid)), \
             patch.object(hreg, '_unit_profile_ids', side_effect=lambda world, key, blocks:
                          [51, 50] if key == 51 else original_ids(world, key, blocks)):
            _, missing, quality = hreg._unit_links(region(self.core_a, 600.))
        self.assertFalse(any(r['unit_key'] == 51 for r in missing))
        self.assertGreater(quality['unresolved_authority_blocks'], 0)
        self.assertGreater(quality['stranded'], 0)

    def test_inlet_ceiling_propagates_through_downstream_obstacles(self):
        # Original flood 130 -> 120 -> 100; inlet 110 requires a 20 m
        # lowering at its first cell and 10 m downstream, without moving sink.
        receiver = np.array([1, 2, -1], np.int64)
        order = np.array([1, 0], np.int64)
        ceiling = np.array([110., np.inf, np.inf])
        hreg._propagate_ceiling(order, receiver, ceiling)
        water = np.minimum([130., 120., 100.], ceiling)
        np.testing.assert_array_equal(water, [110., 110., 100.])
        self.assertTrue(np.all(np.diff(water) <= 0))

    def test_real_neural_lake_reentry_cannot_choose_its_inlet_as_outlet(self):
        # Exact receiver subgraph from seed42 cached region
        # ec9a9f13.../-1872_48_-1864_56.npz, lake L1717_-59902.
        # Old output 5860~g1712_-59899 pointed through this lake to itself:
        # deepest46338 -> ... -> channel45061 -> lake44806 -> channel44551.
        # Crop the actual downstream continuation at44296 for this fixture.
        cells = np.array([44294, 44550, 44806, 45062, 45314, 45316, 45318, 45571, 45572, 45573,
                          45826, 45828, 45829, 45830, 46082, 46086, 46087, 46338, 46340, 46341])
        receivers = [44551, 44551, 44551, 44806, 45315, 45061, 45061, 45316, 45316, 45316,
                     45571, 45571, 45572, 45573, 45826, 45829, 45830, 46082, 46341, 46086]
        depth = [41, 41, 41, 42, 45, 43, 43, 44, 44, 44, 45, 45, 45, 45, 46, 46, 46, 47, 48, 47]
        n = 46342
        rcv = np.full(n, -1, np.int64)
        rcv[cells] = receivers
        rcv[[45061, 45315, 44551]] = [44806, 45316, 44296]
        order = np.r_[cells, [45061, 45315, 44551]]
        order = order[np.argsort(np.r_[depth, [42, 44, 40]], kind='stable')]
        lake_of = np.full(n, -1, np.int64)
        lake_of[cells] = 0
        seed = rcv < 0
        channel = np.zeros(n, bool)
        channel[[45061, 44551]] = True
        lakes = [dict(id='L1717_-59902', cells=cells, lowest_cell=46338, entry_reaches=[])]
        tribs = hreg._tributaries(None, channel, rcv, lake_of, lakes, seed, np.zeros(n, bool), order)
        self.assertEqual(lakes[0]['outlet_cell'], 44551)
        self.assertNotEqual(lakes[0]['outlet_cell'], 45061)
        starts = {t['start']: t for t in tribs}
        self.assertEqual(starts[45061]['end'], 44806)
        self.assertEqual(starts[44551]['end'], 44296)

        R = SimpleNamespace(world=SimpleNamespace(reaches=[None] * 6000, lakes=[]),
                            key=np.full((1, n), 5860), basin=np.zeros((1, n), int),
                            gid=lambda c: f'g{1536 + int(c) // 256}_{-59904 + int(c) % 256}',
                            xy_of=lambda c: np.column_stack((np.asarray(c) % 256, np.asarray(c) // 256)))
        seed_kind = np.where(seed, hreg.TERMINAL_SEED, 0)
        records = hreg._finish_tributaries(R, tribs, [], [], np.full(n, -1), np.full(n, -1), lakes,
                                          lake_of, np.full(n, 109.87055969238281), np.ones(n), np.ones(n),
                                          rcv, seed_kind, [], {}, {})
        by_id = {r['id']: r for r in records}
        old_cycle = by_id['5860~g1712_-59899']
        self.assertEqual(old_cycle['downstream_id'], f'5860~{R.gid(44551)}')
        self.assertIsNone(by_id[old_cycle['downstream_id']]['downstream_id'])

    def test_key_wall_pond_is_drawn_only_at_its_rim_level(self):
        # Real seed42: per-key Priority-Flood ponds stood up to 11.5 m above
        # the lower ground across a unit-block edge, merged at the max level.
        rows, cols = 7, 10
        H = np.full((rows, cols), 20.)
        filled = H.copy()
        H[2:5, 2:5], filled[2:5, 2:5] = 5., 12.          # key 1 pond, in-key spill 12
        H[2:5, 7:9], filled[2:5, 7:9] = 15., 18.         # key 2 pond, contained
        flood = np.where(np.arange(cols) < 5, 1, 2)[None, :].repeat(rows, 0)
        R = SimpleNamespace(p=dict(min_lake_depth_m=1., min_lake_cells=4), rows=rows, cols=cols,
                            cell_area=np.ones((rows, cols)), basin=np.zeros((rows, cols), int),
                            key=flood, gid=lambda c: f'g0_{c}', xy_of=lambda c: np.zeros((len(c), 2)))
        n = rows * cols
        args = (np.zeros(n, bool), np.full(n, hreg.NETWORK), np.full(n, -1), flood.ravel())
        for wall, drawn in ((8., 8.), (2., None)):
            H[2:5, 5] = filled[2:5, 5] = wall            # dry key 2 bank across the cut
            lakes, lake_of, walls = hreg._local_lakes(R, H.ravel(), filled.ravel(), *args)
            levels = sorted(lk['level_m'] for lk in lakes)
            self.assertEqual(walls['count'], 1)
            self.assertEqual(walls['max_excess_m'], 12. - wall)
            self.assertEqual(levels, [18.] if drawn is None else [drawn, 18.])
            for lake in lakes:                            # drawn ponds never exceed their rim
                self.assertEqual(lake['level_spread_m'], 0.)
                ring = hreg.ndimage.binary_dilation(lake_of.reshape(rows, cols) == lakes.index(lake),
                                                    np.ones((3, 3), bool)) & (lake_of.reshape(rows, cols) < 0)
                self.assertGreaterEqual(H[ring].min(), lake['level_m'] - 1.)

    def test_channel_water_cuts_hidden_sills_and_descends_on_the_dag(self):
        # Trunk seed 0 at 50 m; a flood at 60 m behind a 60 m sill (cell 1)
        # over a 30 m pit (cell 2); a second branch joins at the pit.
        receivers = np.array([-1, 0, 1, 2, 3, 4, 2], np.int64)
        ground = np.array([40., 60., 30., 35., 40., 75., 55.])
        filled = np.array([50., 60., 60., 60., 60., 75., 60.])
        order = np.array([1, 2, 3, 6, 4, 5], np.int64)
        fixed = np.array([50.] + [np.nan] * 6)
        out, floor = hreg._channel_water(order, receivers, fixed, filled, ground, 2., 60.)
        self.assertEqual(out[0], 50.)
        for c in order:
            self.assertLessEqual(out[receivers[c]], out[c])
            self.assertLessEqual(out[c], filled[c])
            self.assertGreaterEqual(out[c], 50.)
        self.assertEqual(out[1], 50.)                     # sill cut down to the trunk water
        self.assertEqual(out[5], 75.)                     # an unflooded head stays on its ground
        self.assertTrue(np.all(floor[order] == 50.))
        # A >60 m sill is not cut deeper than the bound: the water stays held up.
        ground[1] = filled[1:5] = 130.
        out, _ = hreg._channel_water(order, receivers, fixed, filled, ground, 2., 60.)
        self.assertEqual(out[1], 70.)

    def test_parent_lake_shore_keeps_parent_level(self):
        # A parent lake whose neural floor lies far below its level, with a
        # wider neural bowl than the parent lake cells.
        cx, cy = 52 * B, VALLEY_Y
        bowl = lambda x, y: macro(x, y) - 400 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 40000.0 ** 2)
        world = build_hydrology(grid(bowl, (0, 0, 32 * SRC, 16 * SRC), SRC), bounds=(0, 0, 32 * SRC, 16 * SRC),
                                topology='plane', runoff_mm_year=600., evaporation_mm_year=100.,
                                channel_min_area_m2=2e9, channel_min_discharge_m3s=.1)
        level = world.lakes[0]['level_m']
        base = lambda x, y: (bowl(x, y) + 120 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 30000.0 ** 2)
                             + 3 * np.sin(x / 700.) * np.cos(y / 900.))
        plan = refine(world, (48 * B, 16 * B, 56 * B, 24 * B), coarse=bowl, base=base)
        q = plan.metadata['quality']
        self.assertGreater(level - plan.routing_height_m.min(), 100.0)
        self.assertGreater(q['parent_lake_extension_area_m2'], 0.0)
        for lake in plan.local_lakes:
            if lake.get('parent_lake_id') is None:      # owned ponds never sit below the parent level
                self.assertGreater(lake['level_m'], level - 1.0, lake['id'])
            else:
                self.assertEqual(lake['level_m'], level)
        shore = next(lk for lk in plan.local_lakes if lk['id'].startswith('P0@'))
        np.testing.assert_array_equal(plan.water_m.ravel()[shore['cells']], level)
        self.assertEqual(q['orphan_area_m2'], 0.0)
        self.assertLess(abs(q['yield_residual_m3s']), 1e-9 * max(q['yield_m3s'], 1))

    def test_decoder_refines_geometry_without_changing_graph(self):
        plan = self.plan_a
        reach = max(main_trunk(plan)[1:-1], key=lambda r: len(r['xy']))
        sub = dict(reach, xy=reach['xy'][:30], water_m=reach['water_m'][:30])
        lo = np.floor((sub['xy'].min(0) - 1200) / hreg.BASE_M) * hreg.BASE_M
        hi = np.ceil((sub['xy'].max(0) + 1200) / hreg.BASE_M) * hreg.BASE_M
        bounds = (lo[0], lo[1], hi[0], hi[1])
        groove_y = float(np.median(sub['xy'][:, 1])) + 90.0
        decoder = grid(lambda x, y: base_fn(x, y) - 6 * np.exp(-((y - groove_y) / 25.0) ** 2),
                       bounds, hreg.DECODER_M)
        refit = hreg.refit_reach_30m(sub, decoder, bounds)
        for k in ('id', 'parent_id', 'downstream_id', 'basin_id', 'source', 'target'):
            self.assertEqual(refit[k], sub[k])
        np.testing.assert_array_equal(refit['xy'][[0, -1]], sub['xy'][[0, -1]])
        self.assertLessEqual(refit['refit_max_shift_m'], 105.0)
        middle = refit['xy'][len(refit['xy']) // 3: 2 * len(refit['xy']) // 3]
        self.assertLess(np.median(np.abs(middle[:, 1] - groove_y)), 31.0)
        self.assertTrue(np.all(np.diff(refit['water_m']) <= 1e-9))
        # Decoder detail aligned on the conditioned base keeps its 240 m means.
        rect = (lo[0] + 480, lo[1] + 480, hi[0] - 480, hi[1] - 480)
        h30, info = hreg.compose_decoder(plan, decoder, bounds, rect)
        self.assertEqual(h30.shape, (int((rect[3] - rect[1]) / 30), int((rect[2] - rect[0]) / 30)))
        self.assertLess(info['block_mean_error_max_m'], 3.0)


if __name__ == '__main__':
    unittest.main()

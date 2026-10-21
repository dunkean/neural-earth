"""The delivery boundary: persisted ownership, crop agreement, and real
regional composition through the same service used by the tile endpoints."""
from pathlib import Path
import tempfile
import unittest
import unittest.mock
from unittest.mock import patch
from types import SimpleNamespace

from tools._bootstrap import activate
activate()
import numpy as np
from terrain_hydrology_service import HydrologyService, _outside_pieces, _land_course
from terrain_hydrology_sampling import canonical_field, sample_points
from terrain_hydrology_geometry import (sample_bilinear, block_mean, prepare_water_grid,
    rasterize_water, refine_reach, condition_surface)


class HydrologyDeliveryTests(unittest.TestCase):
    def test_display_stops_at_first_sea_crossing(self):
        # Both endpoints are on land: the display must still stop at the inlet.
        terrain = np.tile([10., 10., -10., -10., 10., 10.], (2, 1))
        xy = np.array([[60., 120.], [660., 120.]])
        water = np.array([30., 0.])
        result, level = _land_course(xy, water, terrain, (0., 0., 720., 240.))
        np.testing.assert_allclose(result, [[60., 120.], [240., 120.]])
        np.testing.assert_allclose(level, [30., 21.])
        np.testing.assert_array_equal(xy, [[60., 120.], [660., 120.]])
        empty, _ = _land_course(np.array([[300., 120.], [660., 120.]]),
                                 water, terrain, (0., 0., 720., 240.))
        self.assertEqual(len(empty), 0)

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        bounds = (0., 0., 614400., 307200.)
        x, y = np.meshgrid((np.arange(64)+.5)*9600., (np.arange(32)+.5)*9600.)
        z = 1600.-x*.004+np.abs(y-153600.)*.002
        climate = np.stack([np.full_like(z, 15), np.full_like(z, 5),
                            np.full_like(z, 1800), np.full_like(z, 30)])
        cls.source = dict(height_m=z, climate=climate, bounds=bounds,
                          topology='plane', settings=dict(world_topology='plane'),
                          latitude_deg=np.zeros((32, 1)))
        cls.service = HydrologyService(lambda seed, profile: cls.source,
                                       cache_root=Path(cls.tmp.name))
        cls.entry = cls.service.ensure(42, 'fixture', 'fixture-world')
        cls.entry['future'].result(timeout=60)
        if cls.entry['status'] != 'ready':
            raise AssertionError(cls.entry.get('error'))

    @classmethod
    def tearDownClass(cls):
        cls.service.pool.shutdown(wait=True)
        cls.tmp.cleanup()

    def test_persistence_keeps_world_ownership_queries(self):
        world = self.entry['world']
        path = Path(self.tmp.name)/'roundtrip.npz'
        self.service._save(path, world)
        restored = self.service._load(path)
        self.assertIsNotNone(restored)
        np.testing.assert_array_equal(restored.receivers, world.receivers)
        for rid in (0, len(world.reaches)-1):
            for key in ('xy', 'water_m', 'cells'):
                np.testing.assert_array_equal(restored.reaches[rid][key], world.reaches[rid][key])
                self.assertEqual(restored.reaches[rid][key].dtype, world.reaches[rid][key].dtype)
        for key, value in world.sample([200000., 300000.], [100000., 130000.]).items():
            np.testing.assert_array_equal(restored.sample([200000., 300000.], [100000., 130000.])[key], value)
        box = (120000., 80000., 350000., 200000.)
        np.testing.assert_array_equal(restored.region(box)['portals']['from_cell'],
                                      world.region(box)['portals']['from_cell'])
        reach = max(world.reaches, key=lambda r: r['area_m2'])
        basin = self.service.catchment(self.entry, reach_id=reach['id'])
        self.assertTrue(basin['rings'])
        self.assertAlmostEqual(basin['area_m2']/reach['area_m2'], 1., places=5)

    def test_physical_crop_and_finite_transport(self):
        world = self.entry['world']
        box, shape = (220000., 120000., 296800., 158400.), (64, 128)
        x, y = np.meshgrid(box[0]+(np.arange(shape[1])+.5)*600.,
                           box[1]+(np.arange(shape[0])+.5)*600.)
        raw = sample_bilinear(world.height_m, world.bounds, x, y)
        height, water, report = self.service.physical(self.entry, raw, box, lod=4)
        crop_box = (box[0]+32*600., box[1], box[2]-32*600., box[3])
        cropped, water_crop, _ = self.service.physical(self.entry, raw[:, 32:96], crop_box, lod=4)
        np.testing.assert_allclose(cropped, height[:, 32:96], atol=1e-4)
        np.testing.assert_allclose(water_crop, water[:, 32:96], atol=1e-4)
        self.assertTrue(np.isfinite(water).all())
        self.assertEqual(report['hydro_state'], 'ready')
        self.assertTrue(np.all((water[..., 1] >= 0)&(water[..., 1] <= 1)))

    def test_regional_delivery_preserves_base_in_decoder(self):
        from terrain_hydrology_regions import piece_windows
        def detail(seed, profile, core):
            bb, cb = piece_windows(core)
            def grid(box, spacing):
                x, y = np.meshgrid(np.arange(box[0]+spacing*.5, box[2], spacing),
                                   np.arange(box[1]+spacing*.5, box[3], spacing))
                return 1600.-x*.004+np.abs(y-153600.)*.002
            base = grid(bb, 240.)
            # Strong disagreement of the base's low frequencies with coarse.
            return dict(coarse_height=grid(cb, 7680.), coarse_bounds=cb,
                        base_height=base+300., base_bounds=bb)
        self.service.read_detail = detail
        box = (245760., 122880., 307200., 184320.)
        self.service._regions(self.entry, box, 0)
        for part in list(self.entry['regions'].values()):
            part['future'].result(timeout=60)
            self.assertEqual(part['status'], 'ready', part.get('error'))
        plan = self.entry['regions'][box]['plan']
        raw = np.full((256, 256), 20.)
        small_box = (box[0]+2400., box[1]+2400., box[0]+2400.+7680., box[1]+2400.+7680.)
        composed, info = self.service._parent_surface([self.entry['regions'][box]], raw, small_box, 0)
        x, y = np.meshgrid(np.arange(small_box[0]+120., small_box[2], 240.),
                           np.arange(small_box[1]+120., small_box[3], 240.))
        expected = sample_bilinear(plan.routing_height_m, plan.bounds, x, y)
        np.testing.assert_allclose(block_mean(composed, 8)[3:-3, 3:-3], expected[3:-3, 3:-3], atol=.05)
        self.assertLess(info['max_mean_error_m'], 5.)
        self.assertLess(np.max(np.abs(np.diff(composed[24:-24, 24:-24], axis=1))), 2.,
                        'The hierarchy bridge must not introduce 240 m terraces')
        # A display LOD-1 preview still transports a 240 m base grid.
        # Use physical spacing, not the display LOD, to select its parent.
        preview, _ = self.service._parent_surface([self.entry['regions'][box]],
                                                  np.full((32, 32), 20.), small_box, 1)
        np.testing.assert_allclose(preview, expected, atol=1e-8)

    def test_aligned_decoder_tiles_agree_inside_their_native_halo(self):
        # The real endpoint supplies 24 native samples of halo. The two-step
        # smooth parent correction must agree in the displayed tile interiors.
        parent_bounds = (-7680., -7680., 23040., 15360.)
        x, y = np.meshgrid(np.arange(-7560., 23040., 240.), np.arange(-7560., 15360., 240.))
        parent = 300.+40.*np.sin(x/900.)+30.*np.cos(y/1100.)
        plan = SimpleNamespace(bounds=parent_bounds, routing_height_m=parent)
        parts = [dict(bounds=parent_bounds, plan=plan)]
        bounds = (-720., -720., 16080., 8400.)
        x, y = np.meshgrid(np.arange(-705., 16080., 30.), np.arange(-705., 8400., 30.))
        raw = 900.+80.*np.sin(x/700.)+7.*np.sin(y/45.)
        whole, _ = self.service._parent_surface(parts, raw, bounds, 0)
        for start in (0, 256):
            box = (bounds[0]+start*30., bounds[1], bounds[0]+(start+304)*30., bounds[3])
            tile, _ = self.service._parent_surface(parts, raw[:, start:start+304], box, 0)
            np.testing.assert_allclose(tile[24:-24, 24:-24],
                                       whole[24:-24, start+24:start+280], atol=1e-9)

    def test_fallback_replaces_the_requested_charts_own_base(self):
        # Blend strip: ordinary Base 500 m, rotated Base 100 m, canonical
        # Base 300 m, routing parent 200 m. Each raw decoder must reach 200 m.
        box = (0., 0., 7680., 7680.)
        def part(h, ordinary, polar, canonical):
            plan = SimpleNamespace(bounds=box, routing_height_m=np.full((32, 32), h), metadata=dict(
                base_correction_m=np.full((32, 32), h-canonical),
                chart_correction_m=dict(ordinary=np.full((32, 32), h-ordinary),
                                        polar=np.full((32, 32), h-polar))))
            return dict(bounds=box, plan=plan)
        rotated = lambda x, y: (x, y)
        residual = np.sin(np.arange(512)/7.)[None, :]*3.
        for h, ordinary, polar, canonical in ((200., 500., 100., 300.), (-120., -40., -400., -220.)):
            parts = [part(h, ordinary, polar, canonical)]
            out, info = HydrologyService._parent_surface(parts, np.full((256, 256), polar), box, 0,
                                                         coordinates=rotated, reference='polar')
            np.testing.assert_allclose(out, h, atol=1e-9)
            self.assertIn('polar chart Base', info['composition'])
            # Experimental 15 m (LOD -1) ordinary grid: same rule, residual retained.
            out, _ = HydrologyService._parent_surface(parts, ordinary+residual+np.zeros((512, 1)),
                                                      box, -1, reference='ordinary')
            np.testing.assert_allclose(out, h+residual+np.zeros((512, 1)), atol=1e-9)
        # A callback without raw references keeps the old approximation, labelled.
        del parts[0]['plan'].metadata['chart_correction_m']
        _, info = HydrologyService._parent_surface(parts, np.full((256, 256), 0.), box, 0,
                                                   coordinates=rotated, reference='polar')
        self.assertIn('no raw chart reference', info['composition'])

    @staticmethod
    def _reference_detail(calls):
        """Server simulator: canonical parent windows, or raw chart Base on core + 240 m."""
        from terrain_hydrology_regions import piece_windows
        canonical = lambda x, y: 1600.-x*.004+np.abs(y-153600.)*.002
        # The rotated realization differs in slope, so clamping at a core edge would show.
        bases = dict(polar=lambda x, y: canonical(x, y)+100.+.02*x-.01*y,
                     ordinary=lambda x, y: canonical(x, y)+500.)
        def centres(box, spacing):
            return np.meshgrid(np.arange(box[0]+spacing*.5, box[2], spacing),
                               np.arange(box[1]+spacing*.5, box[3], spacing))
        def detail(seed, profile, core, charts=None):
            calls.append(charts)
            if charts:
                box = (core[0]-240., core[1]-240., core[2]+240., core[3]+240.)
                return dict(chart_base={c: bases[c](*centres(box, 240.)) for c in charts}, base_bounds=box)
            bb, cb = piece_windows(core)
            return dict(coarse_height=canonical(*centres(cb, 7680.)), coarse_bounds=cb,
                        base_height=canonical(*centres(bb, 240.))+300., base_bounds=bb)
        return detail, bases, centres

    def test_chart_references_are_acquired_lazily_per_package(self):
        from terrain_hydrology_regions import RegionPlan
        calls = []
        detail, bases, centres = self._reference_detail(calls)
        previous, self.service.read_detail = self.service.read_detail, detail
        try:
            box = (184320., 122880., 245760., 184320.)
            self.assertEqual(self.service._regions(self.entry, box, 0, 'polar'), ([], 1))
            part = self.entry['regions'][box]
            part['future'].result(timeout=60)
            self.assertEqual(part['status'], 'ready', part.get('error'))
            ready, _ = self.service._regions(self.entry, box, 0, 'polar')
            self.assertEqual(ready, [part])
            # Aligned ordinary grids need no raw reference; another chart waits.
            self.assertEqual(self.service._regions(self.entry, box, 0)[0], [part])
            self.assertEqual(self.service._regions(self.entry, box, 0, 'ordinary'), ([], 1))
            part['reference_jobs']['ordinary'].result(timeout=60)
            self.assertEqual(self.service._regions(self.entry, box, 0, 'ordinary')[0], [part])
            self.assertEqual(calls, [None, ('polar',), ('ordinary',)])
            plan = part['plan']
            halo = (box[0]-240., box[1]-240., box[2]+240., box[3]+240.)
            self.assertEqual(plan.bounds, box)
            np.testing.assert_allclose(plan.metadata['routing_halo_m'][1:-1, 1:-1], plan.routing_height_m, atol=1e-3)
            tag = '_'.join(str(int(v/7680)) for v in box)
            stored = self.service._load(self.entry['region_dir']/f'{tag}.npz', RegionPlan)
            self.assertEqual(tuple(stored.metadata['routing_halo_bounds']), halo)
            for chart in ('polar', 'ordinary'):
                np.testing.assert_allclose(plan.metadata['chart_correction_m'][chart],
                    plan.metadata['routing_halo_m']-bases[chart](*centres(halo, 240.)), atol=1e-2)
                np.testing.assert_array_equal(stored.metadata['chart_correction_m'][chart],
                                              plan.metadata['chart_correction_m'][chart])
            # A rotated decoder without residual composes to the parent.
            small = (box[0]+3000., box[1]+3000., box[0]+3000.+7680., box[1]+3000.+7680.)
            x, y = centres(small, 30.)
            out, _ = self.service._parent_surface([part], bases['polar'](x, y), small, 0,
                                                  coordinates=lambda x, y: (x, y), reference='polar')
            np.testing.assert_allclose(out, sample_bilinear(plan.routing_height_m, plan.bounds, x, y), atol=1e-2)
        finally:
            self.service.read_detail = previous

    def test_fallback_is_continuous_across_producer_packages(self):
        # Two adjacent fixed cores from the real producer path (plan.bounds is
        # the core). A rotated display straddles their edge at x = 122880 m.
        detail, bases, centres = self._reference_detail([])
        previous, self.service.read_detail = self.service.read_detail, detail
        try:
            cores = [(61440., 122880., 122880., 184320.), (122880., 122880., 184320., 184320.)]
            parts = []
            for core in cores:
                self.service._regions(self.entry, core, 0, 'polar')
                part = self.entry['regions'][core]
                part['future'].result(timeout=120)
                for job in list(part.get('reference_jobs', {}).values()):
                    job.result(timeout=60)
                self.assertEqual(self.service._regions(self.entry, core, 0, 'polar')[0], [part], part.get('error'))
                parts.append(part)
        finally:
            self.service.read_detail = previous
        west, east = (p['plan'] for p in parts)
        # The producer halo is the neighbour's own routing terrain.
        np.testing.assert_allclose(west.metadata['routing_halo_m'][1:-1, -1], east.routing_height_m[:, 0], atol=1e-3)
        np.testing.assert_allclose(east.metadata['routing_halo_m'][1:-1, 0], west.routing_height_m[:, -1], atol=1e-3)
        union = (cores[0][0], cores[0][1], cores[1][2], cores[1][3])
        truth = np.hstack([west.routing_height_m, east.routing_height_m])-bases['polar'](*centres(union, 240.))
        chart = lambda x, y: (y+118880., x+128000.)   # display chart -> geographic
        box = (0., 0., 8000., 8000.)
        cx, cy = centres(box, 20.)
        gx, gy = chart(cx, cy)
        residual = 4.*np.sin(cx/90.)*np.cos(cy/70.)
        decoder = bases['polar'](gx, gy)+residual
        out, info = self.service._parent_surface(parts, decoder, box, 0, coordinates=chart, reference='polar')
        self.assertIn('polar chart Base', info['composition'])
        expected = decoder+sample_bilinear(truth, union, gx, gy)
        np.testing.assert_allclose(out, expected, atol=1e-2)
        edge = np.abs(gx-122880.) < 120.
        self.assertTrue((gx[edge] < 122880.).any() and (gx[edge] > 122880.).any())
        # Core-only corrections would clamp here; the fixture is sensitive to it.
        clamped = decoder.copy()
        for part in parts:
            plan, core = part['plan'], part['bounds']
            take = (gx >= core[0]) & (gx < core[2])
            core_only = plan.routing_height_m-bases['polar'](*centres(core, 240.))
            clamped[take] += sample_bilinear(core_only, core, gx[take], gy[take])
        self.assertGreater(float(np.max(np.abs(clamped-expected)[edge])), .5)
        sub = (slice(50, -100), slice(100, -150))
        crop = (box[0]+100*20., box[1]+50*20., box[2]-150*20., box[3]-100*20.)
        cropped, _ = self.service._parent_surface(parts, decoder[sub], crop, 0, coordinates=chart, reference='polar')
        np.testing.assert_array_equal(cropped, out[sub])

    def test_canonical_charts_and_bounded_sampling(self):
        settings = dict(world_topology='sphere')
        charts = []
        def chart(name, x, y):
            charts.append(name)
            return np.full(len(x), 100. if name else 500.)
        polar = canonical_field([-20000000., 20000000.], [-9500000., 9500000.], settings, chart)
        np.testing.assert_array_equal(polar, 100.)
        self.assertEqual(set(charts), {'polar'})
        ordinary = canonical_field([-1000., 1000.], [-1000., 1000.], settings, chart)
        np.testing.assert_array_equal(ordinary, 500.)
        # The rotated chart's own seam crosses the geographic equator at
        # the antimeridian. A deliberately nonperiodic chart still has one
        # canonical value there, approached from either hemisphere.
        jump = canonical_field([19999000.], [-.1, .1], settings,
                               lambda name, x, y: np.asarray(x)/40000000.*10000.)
        self.assertLess(abs(float(jump[0, 0]-jump[1, 0])), .001)
        windows = []
        def block(world, source, x0, y0, x1, y1):
            windows.append((x1-x0, y1-y0))
            x, y = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
            return (2*x+3*y+4).astype(float)
        x, y = np.array([-1000000., -1000., 2000000.]), np.array([1000., 0., -2000000.])
        values = sample_points(None, x, y, 'latent', block)
        np.testing.assert_allclose(values, 2*(x/240.-.5)+3*(y/240.-.5)+4)
        self.assertLessEqual(max(max(s) for s in windows), 132)

    def test_source_preview_never_excavates_a_neural_mountain(self):
        box = (220000., 120000., 296800., 158400.)
        raw = np.full((64, 128), 4000.)
        conditioned, water, report = self.service.physical(self.entry, raw, box, lod=4)
        self.assertGreaterEqual(float(np.min(conditioned-raw)), -2.)
        self.assertLessEqual(report['hydro_max_cut_m'], 2.)

    def test_pending_base_leaves_source_rivers_to_vector_preview(self):
        box = (220000., 120000., 227680., 127680.)
        with patch.object(self.service, '_regions', return_value=([], 1)), \
             patch.object(self.service, '_geometry', side_effect=AssertionError('Per-tile overview work')):
            _, water, report = self.service.physical(self.entry, np.full((256, 256), 4000.), box, lod=0, detail=True)
        self.assertEqual(report['hydro_state'], 'regional-pending')
        self.assertTrue(report['hydro_vector_preview'])
        self.assertFalse(np.any(water[..., 2] == 3))

    def test_overview_remainder_survives_partial_regional_delivery(self):
        # A source segment crosses a ready core without any vertex inside.
        # It remains visible on both sides, including failed/guarded pieces.
        xy = np.array([[0., 10.], [300., 10.]])
        water = np.array([30., 0.])
        fragments = _outside_pieces(xy, water, [(100., 0., 200., 20.)])
        self.assertEqual(len(fragments), 2)
        np.testing.assert_allclose(fragments[0][0], [[0., 10.], [100., 10.]])
        np.testing.assert_allclose(fragments[1][0], [[200., 10.], [300., 10.]])
        np.testing.assert_allclose(fragments[0][1], [30., 20.])
        np.testing.assert_allclose(fragments[1][1], [10., 0.])

    def test_neighbour_river_overrun_is_acquired_and_composed_outside_its_core(self):
        world = SimpleNamespace(bounds=(0., 0., 122880., 61440.), topology='plane',
            height_m=np.full((8, 16), 4.), ocean=np.zeros((8, 16), bool),
            lake_ids=np.full((8, 16), -1), lakes=[], reaches=[dict(flow_m3s=100.)])
        geometry = dict(id='left-river', flow_m3s=100.,
            xy=np.array([[61200., 30000.], [62040., 30000.]]),
            water_m=np.array([5., 5.]), width_m=np.array([120., 120.]), depth_m=np.array([1., 1.]))
        parts = []
        for box in ((0., 0., 61440., 61440.), (61440., 0., 122880., 61440.)):
            plan = SimpleNamespace(bounds=box, routing_height_m=np.full((2, 2), 4.),
                                   metadata=dict(base_correction_m=np.zeros((2, 2))))
            parts.append(dict(bounds=box, plan=plan, geometry=[geometry] if box[0] == 0 else [],
                              local_water_grid=None, local_lakes=[]))
        # A neighbour's still-water sampler clamps outside its field. Its
        # wet border must not leak into a tile owned by the other package.
        local_lake = dict(id=0, level_m=6., cells=np.arange(4))
        parts[0]['local_lakes'] = [local_lake]
        parts[0]['local_water_grid'] = prepare_water_grid(np.zeros((2, 2)), parts[0]['bounds'],
                                                         lakes=[local_lake])
        entry = dict(status='ready', key='overrun-fixture', world=world,
            source=dict(settings=dict(world_topology='plane')), water_grid=prepare_water_grid(
                world.height_m, world.bounds, ocean=world.ocean, lakes=[], lake_ids=world.lake_ids))
        calls = []
        def ready(_, box, lod, reference=None):
            calls.append(box)
            return [p for p in parts if p['bounds'][0] < box[2] and p['bounds'][2] > box[0]], 0
        # Every target sample is inside the right core. The left package is
        # needed only because its generated river extends into that core.
        box, raw = (61500., 29520., 62460., 30480.), np.full((32, 32), 4.)
        reference = rasterize_water([geometry], [], raw.shape, box)
        _, reference, _ = condition_surface(raw, reference,
            max_cut_m=30.+np.minimum(reference['bankfull_depth_m'], 30.), max_still_cut_m=2.)
        expected = reference['river_coverage'] > .5
        self.assertGreater(np.count_nonzero(expected), 0)
        with patch.object(self.service, '_regions', side_effect=ready), \
             patch.object(self.service, '_chosen', return_value=[]):
            height, water, report = self.service.physical(entry, raw, box, lod=0, detail=True)
            self.assertEqual(report['hydro_regions'], 2)
            self.assertLess(calls[0][0], 61440.)
            self.assertFalse(np.any(water[..., 2] == 2))
            np.testing.assert_array_equal(water[..., 2] == 3, reference['class'] == 3)
            np.testing.assert_allclose(water[..., 1], reference['coverage'], atol=1e-6)
            # Arrival order does not pick the winner; overlapping crops agree.
            parts.reverse()
            cropped, crop_water, _ = self.service.physical(entry, raw[:, 4:28],
                (box[0]+120., box[1], box[2]-120., box[3]), lod=0, detail=True)
            np.testing.assert_allclose(cropped, height[:, 4:28], atol=1e-5)
            np.testing.assert_allclose(crop_water, water[:, 4:28], atol=1e-5)
        # A missing right package keeps the source preview away from the
        # left river's overrun; completing one neighbour must not erase it.
        parts[:] = [p for p in parts if p['bounds'][0] == 0.]
        preview = dict(geometry, id='overview',
            xy=np.array([[62280., 29500.], [62280., 30500.]]), width_m=np.array([60., 60.]))
        with patch.object(self.service, '_regions', side_effect=ready), \
             patch.object(self.service, '_chosen', return_value=[0]), \
             patch.object(self.service, '_geometry', return_value=preview):
            _, water, _ = self.service.physical(entry, raw, box, lod=0, detail=True)
            self.assertEqual(water[16, 26, 2], 3)
            self.assertEqual(water[16, 10, 2], 3)
            self.assertFalse(np.any(water[..., 2] == 2))

    def test_total_geometry_support_and_latitude_aware_query(self):
        raw = np.array([[60000., 10000.], [61320., 20000.],
                        [61320., 40000.], [60000., 50000.]])
        reach = dict(id=8, source='a', target='b', flow_m3s=1000., pins=[0, 1, 2, 3],
                     xy=raw, water_m=np.array([10., 9., 7., 6.]))
        # Deliberately tight envelope exercises saturation of the complete
        # Hermite/valley/meander displacement, not just the final bend offset.
        envelope = (59000., 9000., 61500., 51000.)
        unbounded = refine_reach(reach, seed=0, fit_profile=False)
        self.assertGreater(float(np.max(unbounded['xy'][:, 0])), envelope[2])
        g = refine_reach(reach, seed=0, support_bounds=envelope, fit_profile=False)
        for key in ('xy', 'belt_xy'):
            self.assertTrue(np.all(g[key] >= envelope[:2]))
            self.assertTrue(np.all(g[key] <= envelope[2:]))
        np.testing.assert_allclose(g['xy'][g['pins']], raw, atol=1e-8)
        self.assertLessEqual(float(np.max(g['belt_width_m'])), 4*g['bankfull_width_m'])
        polar_reach = dict(reach, xy=raw+np.array([0., 8800000.]))
        polar_envelope = np.array(envelope)+np.array([0., 8800000., 0., 8800000.])
        g = refine_reach(polar_reach, seed=0, support_bounds=polar_envelope,
                         sphere_radius_m=40000000./(2*np.pi), fit_profile=False)
        for key in ('xy', 'belt_xy'):
            self.assertTrue(np.all(g[key] >= polar_envelope[:2]))
            self.assertTrue(np.all(g[key] <= polar_envelope[2:]))
        np.testing.assert_allclose(g['xy'][g['pins']], polar_reach['xy'], atol=1e-8)
        sphere = SimpleNamespace(bounds=(-20000000., -10000000., 20000000., 10000000.),
                                 topology='sphere', reaches=[dict(flow_m3s=100.)])
        equator = self.service._river_query_bounds(sphere, (0., 0., 9000., 9000.), 30.)
        polar = self.service._river_query_bounds(sphere, (0., 8800000., 9000., 8809000.), 30.)
        self.assertGreater(-polar[0]-3840., (-equator[0]-3840.)*3)
        plane = SimpleNamespace(topology='plane', reaches=[dict(flow_m3s=100.)])
        interior = self.service._river_query_bounds(plane, (28000., 28000., 37000., 37000.), 30.)
        self.assertGreater(interior[0], 0.)
        self.assertLess(interior[2], 61440.)

    def test_regional_navigation_respects_terminal_water_bodies(self):
        flags = [[1]*12, [1]*12, [0]*12]
        world = SimpleNamespace(topology='plane', reaches=[
            dict(flow_m3s=500., navigation_months=flags),
            dict(flow_m3s=500., navigation_months=[[0]*12]*3)],
            lakes=[dict(closed=True, exit_reach=None), dict(closed=False, exit_reach=1)])
        xy = np.column_stack((np.linspace(0, 5000, 20), np.zeros(20)))
        def reach(name, parent, terminal):
            return dict(id=name, parent_id=parent, kind='tributary', downstream_id=None,
                xy=xy, water_m=np.linspace(10., 9.99, 20), flow_m3s=40., terminal_kind=terminal)
        plan = SimpleNamespace(reaches=[reach('pit', 0, 'terminal'), reach('lake', 0, 'lake:3'),
            reach('coast', 1, 'sea'), reach('closed', 0, 'parent_lake:0'),
            reach('open', 0, 'parent_lake:1'), reach('lake_to_sea', 1, 'lake:4'),
            reach('lake_to_pit', 0, 'lake:5'), reach('lake_to_closed', 0, 'lake:6'),
            reach('lake_to_edge', 0, 'lake:7')], local_lakes=[dict(id=3, outlet_cell=-1),
                dict(id=4, outlet_cell=1, exit_kind='sea'), dict(id=5, outlet_cell=1, exit_kind='terminal'),
                dict(id=6, outlet_cell=1, exit_kind='parent_lake:0'), dict(id=7, outlet_cell=-3, exit_kind='edge')])
        self.service._regional_properties(world, plan)
        self.assertEqual([sum(r['navigation_months'][0]) for r in plan.reaches], [0, 0, 12, 0, 0, 12, 0, 0, 12])

    def test_new_tributaries_keep_local_seasons_and_annual_flow(self):
        from terrain_hydrology_balance import MONTH_SECONDS
        local, upstream = np.zeros(12), np.zeros(12)
        local[0], upstream[6] = 1., 1.
        world = SimpleNamespace(topology='plane', reaches=[dict(flow_m3s=500.,
            monthly_flow_m3s=(500.*MONTH_SECONDS.sum()*upstream/MONTH_SECONDS).tolist(),
            unit_monthly_fraction=local.tolist())])
        xy = np.array([[0., 0.], [5000., 0.]])
        records = [dict(id=str(i), parent_id=0, kind=kind, start_link=link,
            downstream_id=None, xy=xy, water_m=np.array([10., 9.99]),
            flow_m3s=40., terminal_kind='sea')
            for i, (kind, link) in enumerate((('tributary', None), ('trunk', None), ('tributary', 'inlet')))]
        self.service._regional_properties(world, SimpleNamespace(reaches=records))
        monthly = np.array([r['monthly_flow_m3s'] for r in records])
        np.testing.assert_array_equal(np.argmax(monthly, axis=1), [0, 6, 6])
        np.testing.assert_allclose(monthly@MONTH_SECONDS/MONTH_SECONDS.sum(), 40., atol=1e-12)
        # A confluence retains inherited July water while adding January yield.
        below = dict(records[0], id='below', flow_m3s=81.)
        records[0]['downstream_id'] = records[1]['downstream_id'] = 'below'
        records.append(below)
        self.service._regional_properties(world, SimpleNamespace(reaches=records))
        monthly = np.array([r['monthly_flow_m3s'] for r in records])
        increment = monthly[3]-monthly[0]-monthly[1]
        self.assertGreater(increment[0], 0.)
        np.testing.assert_allclose(increment[1:], 0., atol=1e-12)
        np.testing.assert_allclose(monthly@MONTH_SECONDS/MONTH_SECONDS.sum(), [40., 40., 40., 81.], atol=1e-12)
        # A real unit link carries its unit's local regime on both sides of a cut.
        head = dict(records[0], id='head', downstream_id=None, flow_m3s=40.)
        linked = dict(head, id='linked', start_link='cut')
        self.service._regional_properties(world, SimpleNamespace(reaches=[head]))
        self.service._regional_properties(world, SimpleNamespace(reaches=[linked],
            links=[dict(id='cut', unit_key=0, flow_m3s=40.)]))
        np.testing.assert_array_equal(head['monthly_flow_m3s'], linked['monthly_flow_m3s'])

    def test_coarse_views_reuse_ready_courses_without_work_or_parent_composition(self):
        from collections import OrderedDict
        size = 61440.
        world = SimpleNamespace(bounds=(0., 0., 5*size, 4*size), topology='plane', shape=(32, 40),
            height_m=np.full((32, 40), 300.), ocean=np.zeros((32, 40), bool),
            lake_ids=np.full((32, 40), -1), lakes=[], reaches=[dict(flow_m3s=100.)],
            metadata=dict(parameters=dict(channel_min_discharge_m3s=1.)), _spacing=lambda: (7680., 7680.))
        def line(name, y, x0, x1, q):
            return dict(id=name, flow_m3s=q, kind='trunk', xy=np.array([[x0, y], [x1, y]]), pins=[0, 1],
                        station_m=np.array([0., x1-x0]), axis_station_m=np.array([0., x1-x0]),
                        tangent=np.array([[1., 0.], [1., 0.]]), water_m=np.array([300., 299.]),
                        width_m=np.array([120., 120.]), depth_m=np.array([2., 2.]))
        # Old source preview at y = 30 km; the ready Base course runs 10 km away.
        source = line(0, 30000., 0., 2*size, 100.)
        course, rill = line('0#src', 40000., 0., size, 100.), dict(line('0~rill', 50000., 0., 9000., .2), kind='tributary')
        def part(core, geometry):
            # A 240 m decoder correction (-900 m) that must never reach a coarse display.
            plan = SimpleNamespace(bounds=core, routing_height_m=np.full((256, 256), -600.), metadata=dict(
                base_correction_m=np.full((2, 2), -900.), chart_correction_m=dict(ordinary=np.full((2, 2), -900.)),
                quality={}))
            return dict(status='ready', bounds=core, plan=plan, geometry=geometry, local_water_grid=None, local_lakes=[])
        def entry(cores):
            return dict(status='ready', key='coarse-fixture', started=0., world=world, timings={}, lake_geometry={},
                source=dict(settings=dict(world_topology='plane')), river_max_flow_m3s=100.,
                regions=OrderedDict((c, part(c, [course, rill] if c[0] == 0. and c[1] == 0. else [])) for c in cores),
                water_grid=prepare_water_grid(world.height_m, world.bounds, ocean=world.ocean, lakes=[],
                                              lake_ids=world.lake_ids))
        service = HydrologyService(lambda *_: None, cache_root=Path(self.tmp.name)/'coarse',
            read_detail=lambda *_, **__: self.fail('coarse view requested Base/neural detail'))
        one = entry([(0., 0., size, size)])
        with patch.object(service.pool, 'submit', side_effect=AssertionError('coarse view scheduled work')), \
             patch.object(service, '_chosen', return_value=[0]), \
             patch.object(service, '_geometry', return_value=source):
            data = service.features(one, (0., 0., 2*size, size), lod=4, detail=True)
            # A 480 m display of a genuine coarse terrain covering the ready core.
            raw = (300.+np.arange(128)[:, None]*.5+np.zeros((1, 128))).astype(np.float32)
            height, water, report = service.physical(one, raw, (0., 0., size, size), lod=4, detail=True)
            fine_pending = service.physical(one, raw, (0., 0., size, size), lod=7, detail=True)[2]
        records = {r['id']: r for r in data['reaches']}
        self.assertEqual(data['resolution'], 'mixed')
        self.assertNotIn('0~rill', records, 'coarse views keep the source channel threshold')
        np.testing.assert_allclose(np.asarray(records['0#src']['xy'])[:, 1], 40000.)
        self.assertGreaterEqual(float(np.min(np.asarray(records[0]['xy'])[:, 0])), size, 'preview remains only outside the core')
        self.assertEqual(list(one['regions']), [(0., 0., size, size)])
        # Same course in the raster; the coarse cells keep their own heights.
        np.testing.assert_array_equal(height, raw)
        self.assertEqual(report['hydro_regions'], 1)
        self.assertIn('none', report['hydro_parent']['composition'])
        rows = np.flatnonzero((water[..., 2] == 3).any(1))
        np.testing.assert_array_equal(rows, [83])
        self.assertEqual(fine_pending['hydro_regions'], 1)
        # Fine views deliver every cached ready piece, however far from the view centre.
        cores = [(c*size, r*size, (c+1)*size, (r+1)*size) for r in range(4) for c in range(5)]
        many = entry(cores)
        with patch.object(service.pool, 'submit', side_effect=AssertionError('cached packages rescheduled')):
            ready, pending = service._regions(many, world.bounds, 0)
        self.assertEqual((len(ready), pending), (20, 0))
        self.assertEqual(sorted(p['bounds'] for p in ready), sorted(cores))
        # Six missing pieces lie beyond the closest 16: they are built (four in
        # flight) or reported pending, never silently omitted from a ready tile.
        centre = np.array([2.5*size, 2.*size])
        far = sorted(cores, key=lambda c: -np.hypot((c[0]+c[2])*.5-centre[0], (c[1]+c[3])*.5-centre[1]))[:6]
        for core in far:
            del many['regions'][core]
        submitted = []
        with patch.object(service.pool, 'submit', side_effect=lambda fn, entry, part: submitted.append(part['bounds'])):
            ready, pending = service._regions(many, world.bounds, 0)
        self.assertEqual((len(ready), pending, len(submitted)), (14, 6, 4))
        self.assertTrue(set(submitted) <= set(far))
        service.pool.shutdown(wait=True)

    def test_decoder_relief_is_not_charged_to_the_parent_allowance(self):
        # Routing parent 55 m above a 0 m river (5 m deep): within the shared
        # allowance P = 65 m. Decoder relief of +-20 m made the old budget,
        # measured from the decoder surface, leave the bed dry.
        core, box = (0., 0., 61440., 61440.), (24000., 24000., 33600., 33600.)
        world = SimpleNamespace(bounds=core, topology='plane', height_m=np.full((8, 8), 55.),
            ocean=np.zeros((8, 8), bool), lake_ids=np.full((8, 8), -1), lakes=[], reaches=[dict(flow_m3s=500.)])
        river = dict(id='r', flow_m3s=500., xy=np.array([[0., 30000.], [61440., 30000.]]),
                     water_m=np.array([0., 0.]), width_m=np.array([240., 240.]), depth_m=np.array([5., 5.]))
        x, y = np.meshgrid(np.arange(box[0]+15., box[2], 30.), np.arange(box[1]+15., box[3], 30.))
        relief = 20.*np.sin(x/150.)*np.cos(y/170.)
        entry = dict(status='ready', key='decoder-fixture', world=world,
            source=dict(settings=dict(world_topology='plane')), water_grid=prepare_water_grid(
                world.height_m, world.bounds, ocean=world.ocean, lakes=[], lake_ids=world.lake_ids))
        def run(parent, raw):
            plan = SimpleNamespace(bounds=core, routing_height_m=np.full((256, 256), parent),
                                   metadata=dict(base_correction_m=np.zeros((2, 2))))
            parts = [dict(bounds=core, plan=plan, geometry=[river], local_water_grid=None, local_lakes=[])]
            with patch.object(self.service, '_regions', return_value=(parts, 0)), \
                 patch.object(self.service, '_chosen', return_value=[]):
                composed, _ = self.service._parent_surface(parts, raw, box, 0)
                return composed, *self.service.physical(entry, raw, box, lod=0, detail=True)
        composed, height, water, report = run(55., 55.+relief)
        axis = np.abs(y[:, 0]-30000.) < 60.
        self.assertTrue(np.all(water[axis, :, 2] == 3), 'channel bed must be wet')
        self.assertEqual(report['hydro_channel']['blocked_bed_cells'], 0)
        self.assertTrue(np.all(height[axis] < 0.))
        # Outside finite river support, decoder relief and parent cells are untouched.
        far = np.abs(y[:, 0]-30000.) > 1500.
        np.testing.assert_array_equal(height[far], composed[far].astype(np.float32))
        # Decoder relief is bounded: a 45 m hill in the bed stays a reported, dry conflict.
        hill = np.hypot(x-28800., y-30000.) < 90.
        _, height, water, report = run(55., 55.+relief+np.where(hill, 45., 0.))
        self.assertGreater(report['hydro_channel']['blocked_bed_cells'], 0)
        self.assertFalse(np.any(water[hill & axis[:, None], 2] == 3))
        self.assertTrue(np.all(water[axis[:, None] & ~(np.hypot(x-28800., y-30000.) < 400.), 2] == 3))
        # A parent 105 m above its envelope exceeds P + R: explicit conflict, no blue on dry ground.
        _, height, water, report = run(100., 100.+relief)
        self.assertGreater(report['hydro_channel']['parent_excess_cells'], 0)
        self.assertGreater(report['hydro_channel']['blocked_bed_cells'], 0)
        self.assertFalse(np.any((water[..., 2] == 3) & (height >= 0.)))

    def test_default_is_the_source_network_without_neural_or_cosmetic_work(self):
        import terrain_hydrology_geometry as G
        world, service, entry = self.entry['world'], self.service, self.entry
        heavy = {name: AssertionError(f'{name} ran with cosmetics off') for name in
                 ('_valley_fit', '_channel_offsets', '_limit_axis_radius', '_spline_noise', 'refine_network')}
        def refuse(*_, **__):
            raise AssertionError('default view requested neural detail')
        previous = service.read_detail, service.read_coarse_points
        service.read_detail = service.read_coarse_points = refuse
        entry['geometry'].clear()
        try:
            with patch.object(service.pool, 'submit', side_effect=AssertionError('default view scheduled work')), \
                 patch.multiple(G, **{k: unittest.mock.DEFAULT for k in heavy}) as mocks:
                for name, mock in mocks.items():
                    mock.side_effect = heavy[name]
                views = {lod: service.features(entry, world.bounds, lod=lod) for lod in (2, 5, 7)}
                box = (220000., 120000., 296800., 158400.)
                raw = np.full((64, 128), 4000.)
                _, water, report = service.physical(entry, raw, box, lod=4)
        finally:
            service.read_detail, service.read_coarse_points = previous
        self.assertEqual((report['hydro_regions'], report['hydro_pending_regions']), (0, 0))
        self.assertTrue(report['hydro_vector_preview'])
        self.assertEqual(report['hydro_max_cut_m'] <= 2., True)
        self.assertTrue(np.isfinite(water).all())
        for data in views.values():
            self.assertEqual((data['resolution'], data['status'], data['detail']), ('source-network', 'ready', False))
        # One axis at every zoom: each LOD's vertices lie on the same stored
        # curve, which keeps the source endpoints (confluences) and budgets.
        self.assertTrue(views[7]['reaches'])
        for record in views[7]['reaches']:
            curve = entry['geometry'][(record['id'], frozenset())]
            raw_reach = world.reaches[record['id']]
            np.testing.assert_allclose(curve['xy'][[0, -1]], np.asarray(raw_reach['xy'])[[0, -1]], atol=1e-6)
            self.assertEqual(curve['flow_m3s'], raw_reach['flow_m3s'])
            self.assertEqual(curve['morphology'], 'axis')
            for lod in (2, 5, 7):
                for other in (r for r in views[lod]['reaches'] if r['id'] == record['id']):
                    gap = np.min(np.hypot(*(np.asarray(other['xy'])[:, None, :]-curve['xy'][None]).transpose(2, 0, 1)), 1)
                    self.assertLess(float(gap.max()), 1e-6)

    def test_cosmetics_are_explicit_and_independent(self):
        from terrain_hydrology_geometry import cosmetic_options
        parse = HydrologyService.parse_cosmetics
        self.assertEqual(parse(''), frozenset())
        self.assertEqual(parse(' meanders,valleyFit,meanders'), frozenset({'meanders', 'valleyFit'}))
        with self.assertRaises(ValueError):
            parse('lakes')
        world = self.entry['world']
        reach = max(world.reaches, key=lambda r: len(r['xy']))
        common = dict(seed=42, terrain=self.entry['height64'], terrain_bounds=world.bounds,
                      climate=dict(temperature_c=27., precip_mm_year=2600.))
        axis = refine_reach(reach, **common, **cosmetic_options(()))
        nested_only = refine_reach(reach, **common, **cosmetic_options({'nestedMeanders'}))
        np.testing.assert_array_equal(nested_only['xy'], axis['xy'])
        bends = refine_reach(reach, **common, **cosmetic_options({'meanders'}))
        self.assertGreater(len(bends['xy']), len(axis['xy']))
        # Display variants are memory-only and never change the world store.
        self.assertNotEqual(self.entry['world_path'].parent.name, self.entry['key'])
        self.assertTrue(self.entry['world_path'].is_file())


    def test_default_vectors_route_cached_base_only(self):
        from terrain_hydrology_regions import refine_region, piece_windows, RegionPlan
        from terrain_hydrology_service import VECTOR_CORE_M
        service, entry, world = self.service, self.entry, self.entry['world']
        size, y0 = VECTOR_CORE_M, 122880.
        cores = [(c*size, y0, (c+1)*size, y0+size) for c in (6, 7, 8)]
        relief = lambda x, y: 1600.-x*.004+np.abs(y-153600.)*.002+25.*np.sin(x/2900.)*np.cos(y/3700.)
        calls = []
        def cached(seed, profile, box):
            calls.append(tuple(box))
            if box[0] >= cores[2][0]-7680.-1:
                return None                       # this core's Base was never generated
            x, y = np.meshgrid(np.arange(box[0]+120., box[2], 240.), np.arange(box[1]+120., box[3], 240.))
            return dict(base_height=relief(x, y), base_bounds=tuple(box))
        def refuse(*_, **__):
            raise AssertionError('default vectors requested neural detail or coarse profiles')
        previous = service.read_detail, service.read_coarse_points, service.read_cached_base
        service.read_detail = service.read_coarse_points = refuse
        service.read_cached_base = cached
        entry['vector_regions'].clear()
        view = (cores[0][0]+1., y0+1., cores[2][2]-1., y0+size-1.)
        try:
            first = service.features(entry, view, lod=2)
            self.assertEqual((first['status'], first['regional']['pending']), ('building', 2))
            for part in list(entry['vector_regions'].values()):
                if 'future' in part:
                    part['future'].result(timeout=120)
            with patch.object(service.pool, 'submit', side_effect=AssertionError('missing Base requeued')):
                data = service.features(entry, view, lod=2)
                wide = service.features(entry, view, lod=5)
                raw = np.full((64, 128), 4000.)
                height, _, report = service.physical(entry, raw, (220000., 130000., 296800., 168400.), lod=0)
        finally:
            service.read_detail, service.read_coarse_points, service.read_cached_base = previous
        # Exactly core + one 7680 m block of Base per core; no Coarse window.
        self.assertEqual(set(calls), {piece_windows(c, align_coarse=False)[0] for c in cores})
        self.assertEqual(piece_windows(cores[0], align_coarse=False)[0],
                         (cores[0][0]-7680., cores[0][1]-7680., cores[0][2]+7680., cores[0][3]+7680.))
        regional = data['regional']
        self.assertEqual((data['status'], regional['ready'], regional['pending'], regional['unavailable']),
                         ('ready', 2, 0, 1))
        self.assertTrue(regional['cached_only'])
        self.assertEqual(wide['regional']['ready'], 2, 'ready vectors persist at wide LODs')
        source_area = world.metadata['parameters']['channel_min_area_m2']
        self.assertFalse(any(r.get('kind') == 'tributary' and r.get('area_m2', 0.) < source_area
                             for r in wide['reaches']))
        self.assertGreater(sum(r.get('kind') == 'tributary' for r in data['reaches']), 0)
        # The raw default terrain is untouched; vectors never carve.
        np.testing.assert_array_equal(height, raw)
        self.assertEqual(report['hydro_regions'], 0)
        # Neighbours from cached Base share their portals exactly (position, level, flow).
        west, east = (entry['vector_regions'][c]['plan'] for c in cores[:2])
        edge = lambda plan: {p['id']: (tuple(np.asarray(p['xy'])), p['level_m'], p['flow_m3s'])
                             for p in plan.portals if p['axis'] == 'x' and abs(p['line_m']-cores[1][0]) < 1e-6}
        self.assertTrue(edge(west))
        self.assertEqual(edge(west), edge(east))
        self.assertTrue(np.isnan(west.metadata['quality']['block_mean_error_max_m']))
        # Stored under the vector store only; detailed packages never read it.
        tag = '_'.join(str(int(v/7680)) for v in cores[0])
        stored = service._load(entry['vector_dir']/f'{tag}.npz', RegionPlan)
        self.assertFalse(stored.metadata['options']['align_coarse'])
        detailed = service._load(entry['region_dir']/f'{tag}.npz', RegionPlan)
        self.assertTrue(detailed is None or detailed.metadata['options']['align_coarse'])

    def test_overview_bridges_to_its_own_portal_at_a_ready_edge(self):
        plan = SimpleNamespace(portals=[
            dict(reach_id=7, axis='x', line_m=100., xy=np.array([100., 14.]), level_m=3.),
            dict(reach_id=8, axis='x', line_m=100., xy=np.array([100., 10.5]), level_m=9.),   # another reach
            dict(reach_id=7, axis='y', line_m=0., xy=np.array([150., 0.]), level_m=9.)])     # another edge
        fragments = _outside_pieces(np.array([[0., 10.], [300., 10.]]), np.array([30., 0.]), [(100., 0., 200., 20.)])
        bridged = HydrologyService._bridge(fragments, 7, [((100., 0., 200., 20.), 0., plan)])
        np.testing.assert_allclose(bridged[0][0], [[0., 10.], [100., 10.], [100., 14.]])
        np.testing.assert_allclose(bridged[0][1], [30., 20., 3.])
        # The downstream fragment starts on the far edge, which has no portal of reach 7.
        np.testing.assert_allclose(bridged[1][0], fragments[1][0])

    def test_uncached_wide_view_stays_ready_without_probe_jobs(self):
        service, entry = self.service, self.entry
        previous, previous_dir = service.read_cached_base, entry['vector_dir']
        service.read_cached_base = lambda *a: None
        entry['vector_dir'] = Path(self.tmp.name)/'no-vector-plans'
        entry['vector_regions'].clear()
        try:
            with patch.object(service.pool, 'submit', side_effect=AssertionError('missing Base queued work')):
                for _ in range(2):
                    data = service.features(entry, entry['world'].bounds, lod=3)
                    self.assertEqual(data['status'], 'ready')
                    self.assertEqual(data['regional']['pending'], 0)
                    self.assertGreater(data['regional']['unavailable'], 0)
        finally:
            service.read_cached_base = previous
            entry['vector_dir'] = previous_dir


if __name__ == '__main__':
    unittest.main()

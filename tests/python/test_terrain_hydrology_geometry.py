import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'backend'))

import terrain_hydrology_geometry as geo  # noqa: E402


def staircase(count=90, spacing=240.0, seed=1):
    """D8-like parent course at 240 m with diagonal stairs."""
    rng = np.random.default_rng(seed)
    points, x, y = [], 0.0, 0.0
    for i in range(count):
        points.append((x, y))
        x += spacing
        if i % 3:
            y += spacing * int(rng.integers(-1, 2))
    return np.asarray(points)


def reach(xy, *, flow=300.0, water=None, pins=(), rid=7):
    water = np.linspace(80.0, 60.0, len(xy)) if water is None else water
    return dict(id=rid, source=rid * 10, target=rid * 10 + 1, basin_id=1, xy=xy, water_m=water,
                flow_m3s=flow, area_m2=1e9, downstream_id=None, pins=list(pins))


def polyline_distance(points, line):
    a, b = line[:-1], line[1:]
    ab = b - a
    out = np.empty(len(points))
    for i, p in enumerate(points):
        t = np.clip(np.sum((p - a) * ab, 1) / np.maximum(np.sum(ab * ab, 1), 1e-12), 0, 1)
        out[i] = np.min(np.hypot(*(p - a - t[:, None] * ab).T))
    return out


class ReachGeometryTest(unittest.TestCase):
    def test_crossings_with_mixed_segment_scales_and_shared_nodes(self):
        # Fine tributaries must not make two long diagonal segments occupy
        # hundreds of thousands of tiny spatial bins. The broad phase must
        # still retain nonadjacent self-intersections and ignore bank joins.
        lines = [
            [[-1000., -1000.], [1000., 1000.]],
            [[-1000., 1000.], [1000., -1000.]],
            np.column_stack([10000.+np.arange(200.), np.full(200, -10000.)]),
            [[30000., 0.], [30020., 20.], [30000., 20.], [30020., 0.]],
            [[40000., 0.], [40010., 5.], [40020., 0.]],
            [[40000., 0.], [40010., 0.], [40020., 20.]],
        ]
        records = [dict(xy=np.asarray(xy), bankfull_width_m=40. if i >= 4 else 0.)
                   for i, xy in enumerate(lines)]
        expected = {(0, 1), (3, 3)}
        self.assertEqual(geo.find_crossings(records), expected)
        self.assertEqual(geo.find_crossings(records[::-1]),
            {(len(records)-1-b, len(records)-1-a) for a, b in expected})
        # Transposed boxes are spread along y, so the sweep runs on the other axis.
        transposed = [dict(r, xy=r['xy'][:, ::-1]) for r in records]
        self.assertEqual(geo.find_crossings(transposed), expected)
        # A tributary draining into a receiver that starts at an inserted
        # portal point meets it at its own mouth, not at a shared end.
        receiver = dict(id='m', xy=np.array([[0., 0.], [120., 0.], [120., 3000.]]), bankfull_width_m=700.)
        mouth = dict(id='t', downstream_id='m', xy=np.array([[60., -500.], [100., 60.]]), bankfull_width_m=5.)
        self.assertEqual(geo.find_crossings([receiver, mouth]), set())
        self.assertEqual(geo.find_crossings([receiver, dict(mouth, downstream_id=None)]), {(0, 1)})

    def test_display_compaction_joins_dry_cuts_and_keeps_wet_confluences(self):
        def rec(rid, x0, x1, down, q=100., kind='trunk', y=0.):
            xs = np.arange(x0, x1 + 1., 240.)
            return dict(id=rid, parent_id=7, kind=kind, xy=np.column_stack([xs, np.full(len(xs), y)]),
                        water_m=np.linspace(10. - x0 / 1e4, 10. - x1 / 1e4, len(xs)), flow_m3s=q,
                        monthly_flow_m3s=[q] * 12, navigation_months=[[1] * 12, [0] * 12], downstream_id=down,
                        pins=[0, len(xs) - 1], length_m=x1 - x0, slope=0.5, area_m2=1e9 + x1)
        records = [rec('a', 0., 2400., 'b'), rec('b', 2400., 4800., 'c'), rec('c', 4800., 7200., 'd'),
                   rec('d', 7200., 9600., 'e', q=120.), rec('e', 9600., 12000., 'f', q=120.),
                   dict(rec('t', 4800., 4800., 'c', q=5., kind='tributary'),
                        xy=np.array([[4800., 2400.], [4800., 0.]]), water_m=np.array([12., 10. - .48])),
                   rec('f', 12000., 14400., 'g', q=120.), rec('g', 14400., 16800., None, q=120.)]
        # Equal flags as list or ndarray still join; a flag change (f|g) keeps the cut.
        records[1]['navigation_months'] = np.array(records[1]['navigation_months'])
        records[7]['navigation_months'] = [[1] * 12, [1] * 6 + [0] * 6]
        out = geo.compact_display_reaches(records)
        by_id = {r['id']: r for r in out}
        # Dry cut a|b and the e|f link join; the wet confluence at c, the flow
        # step at d and the navigation change at g stay.
        self.assertEqual([r['members'] for r in out], [['a', 'b'], ['c'], ['d', 'e', 'f'], ['t'], ['g']])
        self.assertEqual(by_id['g']['navigation_months'], records[7]['navigation_months'])
        # Whole-course length, end-to-end slope and downstream-end area.
        self.assertEqual(by_id['a']['length_m'], 4800.)
        self.assertAlmostEqual(by_id['a']['slope'], 0.48 / 4800.)
        self.assertEqual((by_id['d']['length_m'], by_id['d']['area_m2']), (7200., 1e9 + 14400.))
        self.assertEqual(by_id['c']['slope'], 0.5)      # unmerged records are untouched
        ab = by_id['a']
        self.assertEqual(len(ab['xy']), 21)
        self.assertEqual(list(ab['pins']), [0, 20])
        np.testing.assert_array_equal(ab['xy'][-1], by_id['c']['xy'][0])
        self.assertTrue(np.all(np.diff(ab['water_m']) < 0))
        self.assertEqual((ab['downstream_id'], by_id['t']['downstream_id'], by_id['c']['downstream_id']), ('c', 'c', 'd'))
        # No displayed endpoint dangles and every downstream id is an output id or outside the set.
        for r in out:
            down = r['downstream_id']
            self.assertTrue(down is None or down in by_id)
            if down is not None:
                np.testing.assert_allclose(r['xy'][-1], by_id[down]['xy'][0])
        self.assertEqual(records[0]['downstream_id'], 'b')     # inputs are untouched

    def test_wide_channel_axis_has_no_corner_tighter_than_its_width(self):
        # A winding valley of 2.6 km chords: the Hermite axis alone turns at
        # radius ~900 m (1.25 W) at each control. The axis limiter rounds it
        # to >= 1.5 W (measured on the curve, not its tessellation) while
        # pins and node tangents stay exact.
        corners = np.column_stack([np.arange(0., 24001., 2400.), np.tile([0., 960.], 6)[:11]])
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(corners, axis=0).T))]
        t = np.arange(0., s[-1] + 1., 240.)
        xy = np.column_stack([np.interp(t, s, corners[:, 0]), np.interp(t, s, corners[:, 1])])
        record = reach(xy, flow=10000., water=np.linspace(10., 0., len(xy)))
        tangent = np.array([1., 0.])
        out = geo.refine_reach(record, seed=3, meander_scale=0.0, start_tangent=tangent, tolerance_m=0.5)
        width = out['bankfull_width_m']
        p = out['xy']
        s = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
        t = np.arange(4 * width, s[-1] - 4 * width, 0.25 * width)
        q = np.stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])], 1)
        a, b, c = q[:-2], q[1:-1], q[2:]
        cross = np.abs((b - a)[:, 0] * (c - a)[:, 1] - (b - a)[:, 1] * (c - a)[:, 0])
        radius = np.hypot(*(b - a).T) * np.hypot(*(c - b).T) * np.hypot(*(a - c).T) / np.maximum(2 * cross, 1e-9)
        self.assertGreater(radius.min(), 1.4 * width)
        np.testing.assert_array_equal(p[[0, -1]], xy[[0, -1]])
        np.testing.assert_allclose(out['tangent'][0], tangent, atol=1e-9)
        again = geo.refine_reach(record, seed=3, meander_scale=0.0, start_tangent=tangent, tolerance_m=50.)
        self.assertLess(polyline_distance(again['xy'], p).max(), 1e-6)

    def test_wide_channel_bends_inside_its_own_parent_trench(self):
        # A one-cell incised parent trench is narrower than the channel itself
        # and must not act as a valley wall; real walls further out still bound.
        bounds = (0., -12000., 60000., 12000.)
        terrain = np.zeros((100, 250))
        terrain[50] = -30.
        xy = np.column_stack([np.arange(120., 60000., 240.), np.full(250, 120.)])
        flat = reach(xy, flow=10000., water=np.linspace(1., 0., 250))
        free = geo.refine_reach(flat, seed=8, terrain=terrain, terrain_bounds=bounds, fit_profile=False)
        self.assertGreater(free['diagnostics']['max_offset_m'], 0.25 * free['bankfull_width_m'])
        walled = terrain.copy()
        walled[[48, 52]] = 200.
        walled[:48], walled[53:] = 200., 200.
        confined = geo.refine_reach(flat, seed=8, terrain=walled, terrain_bounds=bounds, fit_profile=False)
        self.assertLess(confined['diagnostics']['max_offset_m'], 0.5 * free['diagnostics']['max_offset_m'])

    def test_pinned_deterministic_and_lod_stable(self):
        xy = staircase()
        record = reach(xy, pins=[30, 61])
        tangent = np.array([0.8, 0.6])
        first = geo.refine_reach(record, seed=11, start_tangent=tangent)
        again = geo.refine_reach(dict(record), seed=11, start_tangent=tangent)
        for key in ('xy', 'water_m', 'station_m', 'belt_xy'):
            np.testing.assert_array_equal(first[key], again[key])
        # Parent nodes and portals are exact; start tangent is the shared one.
        np.testing.assert_array_equal(first['xy'][first['pins']], xy[[0, 30, 61, len(xy) - 1]])
        np.testing.assert_allclose(first['tangent'][0], tangent, atol=2e-2)
        # Bends exist, stay inside the inherited corridor, and depend on the seed.
        self.assertGreater(first['diagnostics']['sinuosity'], 1.03)
        self.assertLessEqual(polyline_distance(first['xy'], xy).max(), first['corridor_m'] + 1.0)
        other = geo.refine_reach(record, seed=12)
        self.assertGreater(np.abs(other['xy'][:, 1].mean() - first['xy'][:, 1].mean())
                           + abs(len(other['xy']) - len(first['xy'])), 1e-6)
        # Display tolerance changes vertex count, never the curve.
        coarse = geo.refine_reach(record, seed=11, start_tangent=tangent, tolerance_m=25.0)
        self.assertLess(len(coarse['xy']), len(first['xy']))
        self.assertLess(polyline_distance(coarse['xy'], first['xy']).max(), 1e-6)
        self.assertLess(polyline_distance(first['xy'], coarse['xy']).max(), 25.0 + 1e-6)
        shown = geo.simplify_reach(first, 40.0)
        np.testing.assert_array_equal(shown['xy'][shown['pins']], xy[[0, 30, 61, len(xy) - 1]])

    def test_descending_profile_and_corridor_valley_fit(self):
        # Parent course on y = 0; the finer valley floor is shifted 300 m south.
        xy = np.stack([np.arange(0.0, 24000.0, 240.0), np.zeros(100)], 1)
        rows, cols = 200, 820
        xs = (np.arange(cols) + 0.5) * 30.0 - 300.0
        ys = (np.arange(rows) + 0.5) * 30.0 - 3000.0
        X, Y = np.meshgrid(xs, ys)
        centre = 300.0 * np.clip(np.minimum(X, 24000.0 - X) / 3000.0, 0, 1)
        terrain = 70.0 - X * 0.0005 + 0.02 * np.abs(Y - centre) + 0.3 * np.sin(X / 97.0)
        bounds = (-300.0, -3000.0, -300.0 + cols * 30.0, -3000.0 + rows * 30.0)
        rng = np.random.default_rng(3)
        water = np.linspace(80.0, 55.0, len(xy)) + rng.normal(0.0, 2.0, len(xy))
        water[0], water[-1] = 81.0, 55.0
        out = geo.refine_reach(reach(xy, flow=40.0, water=water), seed=5, terrain=terrain,
                               terrain_bounds=bounds, corridor_m=600.0)
        level = out['water_m']
        self.assertTrue(np.all(np.diff(level) <= 1e-9))
        self.assertEqual(level[0], 81.0)
        self.assertEqual(level[-1], 55.0)
        middle = (out['xy'][:, 0] > 6000) & (out['xy'][:, 0] < 18000)
        self.assertLess(abs(np.median(out['xy'][middle, 1]) - 300.0), 120.0)
        self.assertLessEqual(np.abs(out['xy'][:, 1]).max(), 600.0 + 1e-6)
        self.assertNotIn('fit_rejected', out['diagnostics'])
        np.testing.assert_array_equal(out['xy'][[0, -1]], xy[[0, -1]])


class ParentPreservationTest(unittest.TestCase):
    def test_coarse_base_decoder_means_and_envelope(self):
        rng = np.random.default_rng(4)
        coarse = rng.uniform(0.0, 400.0, (3, 4))
        # Base reconstruction differs strongly and has a shifted valley.
        base_raw = geo._upsample(coarse, 32) + rng.normal(0.0, 15.0, (96, 128))
        base_raw[:, 40:52] -= 120.0
        lower, upper = geo.parent_envelope(coarse, 32, margin_m=20.0)
        base, info = geo.preserve_parent(base_raw, coarse, 32, lower=lower, upper=upper)
        self.assertTrue(np.all(base >= lower - 1e-9) and np.all(base <= upper + 1e-9))
        self.assertLess(info['max_mean_error_m'], 0.05)
        exact, _ = geo.preserve_parent(base_raw, coarse, 32)
        np.testing.assert_allclose(geo.block_mean(exact, 32), coarse, atol=1e-9)
        # Detail relative to the parent survives (it is not replaced).
        detail_in = base_raw - geo._upsample(geo.block_mean(base_raw, 32), 32)
        detail_out = exact - geo._upsample(coarse, 32)
        self.assertGreater(np.corrcoef(detail_in.ravel(), detail_out.ravel())[0, 1], 0.9)
        decoder_raw = geo._upsample(exact, 8) + rng.normal(0.0, 3.0, (768, 1024))
        decoder, _ = geo.preserve_parent(decoder_raw, exact, 8)
        np.testing.assert_allclose(geo.block_mean(decoder, 8), exact, atol=1e-9)
        np.testing.assert_allclose(geo.block_mean(decoder, 256), coarse, atol=1e-9)
        with self.assertRaises(ValueError):
            geo.preserve_parent(base_raw[:90], coarse, 32)
        self.assertEqual(geo.aligned_window(70, 10, 32, halo_blocks=1), (32, 64 + 32))


class WaterFieldTest(unittest.TestCase):
    def setUp(self):
        # Authoritative 240 m grid with a lake (and island) and an ocean edge.
        rows, cols = 60, 100
        self.bounds = (0.0, 0.0, cols * 240.0, rows * 240.0)
        xs, ys = geo.cell_centres((rows, cols), self.bounds)
        X, Y = np.meshgrid(xs, ys)
        height = 20.0 + 0.002 * X
        lake = np.hypot(X - 15000, Y - 7000) < 3000
        island = np.hypot(X - 15000, Y - 7000) < 700
        height[lake] = 8.0
        height[island] = 14.0
        height[:, :6] = -30.0
        self.lake_mask = lake & ~island
        self.lake = dict(id=3, level_m=10.0, closed=False, area_m2=float(self.lake_mask.sum()) * 240 ** 2,
                         cells=np.flatnonzero(self.lake_mask))
        ocean = np.zeros_like(lake)
        ocean[:, :6] = True
        self.water_grid = geo.prepare_water_grid(height, self.bounds, lakes=[self.lake], ocean=ocean)
        self.height = height
        xy = np.stack([np.linspace(23000, 18000, 40), np.linspace(1000, 6500, 40)], 1)
        self.reaches = [geo.refine_reach(reach(xy, flow=150.0, water=np.linspace(30, 10, 40)), seed=2)]

    def test_crop_overlap_rings_and_wrap(self):
        shape, bounds = (480, 800), self.bounds   # 30 m pixels
        full = geo.rasterize_water(self.reaches, [self.lake], shape, bounds, water_grid=self.water_grid)
        r0, c0, rows, cols = 100, 300, 180, 260
        crop_bounds = (c0 * 30.0, r0 * 30.0, (c0 + cols) * 30.0, (r0 + rows) * 30.0)
        crop = geo.rasterize_water(self.reaches, [self.lake], (rows, cols), crop_bounds,
                                   water_grid=self.water_grid)
        for key in ('class', 'coverage', 'water_level_m', 'depth_m', 'distance_m', 'flow_x'):
            np.testing.assert_allclose(crop[key], full[key][r0:r0 + rows, c0:c0 + cols],
                                       atol=1e-4, equal_nan=True, err_msg=key)
        classes = set(np.unique(full['class']))
        self.assertTrue({geo.DRY, geo.OCEAN, geo.LAKE, geo.RIVER} <= classes)
        island = full['class'][int(7000 / 30), int(15000 / 30)]
        self.assertEqual(island, geo.DRY)
        self.assertTrue(np.all(full['water_level_m'][full['class'] == geo.LAKE] == 10.0))
        rings = geo.lake_shore_rings(self.lake, dict(height=self.height, bounds=self.bounds))
        self.assertEqual([r['hole'] for r in rings], [False, True])
        self.assertAlmostEqual(rings[0]['area_m2'] + rings[1]['area_m2'], self.lake['area_m2'],
                               delta=0.15 * self.lake['area_m2'])
        # A reach crossing the periodic seam appears on both edges of the map.
        period = bounds[2]
        seam = dict(id=9, xy=np.array([[period - 900.0, 300.0], [period + 900.0, 400.0]]),
                    water_m=np.array([5.0, 4.0]), flow_m3s=200.0)
        left = geo.rasterize_water([seam], [], (40, 40), (0.0, 0.0, 1200.0, 1200.0), wrap_period_m=period)
        right = geo.rasterize_water([seam], [], (40, 40), (period - 1200.0, 0.0, period, 1200.0),
                                    wrap_period_m=period)
        self.assertGreater(left['river_coverage'][:, 0].max(), 0.5)
        self.assertGreater(right['river_coverage'][:, -1].max(), 0.5)

    def test_surface_projection_is_idempotent(self):
        shape = (480, 800)
        xs, ys = geo.cell_centres(shape, self.bounds)
        raw = geo.sample_bilinear(self.height, self.bounds, *np.meshgrid(xs, ys)) + 3.0
        fields = geo.rasterize_water(self.reaches, [self.lake], shape, self.bounds,
                                     water_grid=self.water_grid, terrain=raw)
        once, out, info = geo.condition_surface(raw, fields)
        lower, upper = info['envelopes']
        self.assertTrue(np.all(lower <= upper))
        twice, _, again = geo.condition_surface(once, fields, envelopes=info['envelopes'])
        np.testing.assert_array_equal(once, twice)
        self.assertEqual(again['changed_cells'], 0)
        rederived, _, _ = geo.condition_surface(once, fields)
        np.testing.assert_array_equal(once, rederived)
        lake = out['class'] == geo.LAKE
        self.assertTrue(np.all(once[lake] < 10.0))
        channel = (fields['resolved'] > 0.99) & (fields['distance_m'] < fields['width_m'] * 0.25)
        self.assertTrue(np.any(channel))
        self.assertTrue(np.all(once[channel] < fields['river_level_m'][channel]))
        self.assertTrue(np.all(out['depth_m'][channel] > 0))

    def test_bounded_channel_conflicts_are_dry_and_reported(self):
        fields = dict(width_m=np.full((1, 3), 120.), distance_m=np.zeros((1, 3)),
                      river_level_m=np.full((1, 3), 10.), bankfull_depth_m=np.full((1, 3), 2.),
                      resolved=np.array([[1., 1., 0.]]), still_level_m=np.full((1, 3), np.nan),
                      still_coverage=np.zeros((1, 3)), coverage=np.ones((1, 3)),
                      water_level_m=np.full((1, 3), 10.), **{'class': np.full((1, 3), geo.RIVER)})
        height, water, info = geo.condition_surface(np.full((1, 3), 100.), fields,
                                                   max_cut_m=np.array([[30., 92., 30.]]))
        self.assertEqual(info['unresolved_river_cells'], 1)
        self.assertEqual(water['class'][0, 0], geo.DRY)
        self.assertEqual(water['coverage'][0, 0], 0)
        self.assertEqual(water['depth_m'][0, 1], 2.)
        self.assertEqual(height[0, 0], 70.)
        # Unresolved source previews retain coverage without excavating.
        self.assertEqual(height[0, 2], 100.)
        self.assertEqual(water['class'][0, 2], geo.RIVER)


class NestedMorphologyTest(unittest.TestCase):
    def test_climate_gated_nested_belt_and_far_coverage(self):
        xy = np.stack([np.arange(0.0, 60000.0, 240.0), np.zeros(250)], 1)
        flat = reach(xy, flow=3000.0, water=np.linspace(1.0, 0.0, 250))
        humid = dict(precip_mm_year=2600.0, temperature_c=26.0)
        arid = dict(precip_mm_year=250.0, temperature_c=26.0)
        tropical = geo.refine_reach(flat, seed=8, climate=humid, corridor_m=8000.0)
        dry = geo.refine_reach(flat, seed=8, climate=arid, corridor_m=8000.0)
        self.assertEqual(tropical['morphology'], 'nested')
        self.assertNotEqual(dry['morphology'], 'nested')
        steep = geo.refine_reach(reach(xy, flow=3000.0, water=np.linspace(600.0, 0.0, 250)),
                                 seed=8, climate=humid, corridor_m=8000.0)
        self.assertNotEqual(steep['morphology'], 'nested')
        self.assertLess(steep['diagnostics']['sinuosity'], dry['diagnostics']['sinuosity'])
        # Short bends ride on the broad belt; the belt is a smooth far-view course.
        offset = np.interp(tropical['belt_station_m'], tropical['axis_station_m'], tropical['xy'][:, 1]) \
            - tropical['belt_xy'][:, 1]
        self.assertGreater(np.abs(tropical['belt_xy'][:, 1]).max(), 3 * tropical['bankfull_width_m'])
        self.assertLess(np.abs(offset).max(), np.abs(tropical['belt_xy'][:, 1]).max() + tropical['corridor_m'])
        # Wet area is conserved from resolved channel to far belt; far pixels are not flooded.
        ys = (-12000.0, 12000.0)
        near = geo.rasterize_water([tropical], [], (480, 1200), (0.0, ys[0], 60000.0, ys[1]))
        far = geo.rasterize_water([tropical], [], (12, 30), (0.0, ys[0], 60000.0, ys[1]))
        area_near = near['river_coverage'].sum() * 50.0 ** 2
        area_far = far['river_coverage'].sum() * 2000.0 ** 2
        self.assertLess(abs(area_far / area_near - 1.0), 0.35)
        self.assertLess(far['river_coverage'].max(), 0.95)
        again = geo.refine_reach(flat, seed=8, climate=humid, corridor_m=8000.0, tolerance_m=50.0)
        self.assertLess(polyline_distance(again['xy'], tropical['xy']).max(), 1e-6)


if __name__ == '__main__':
    unittest.main()

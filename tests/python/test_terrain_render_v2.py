"""Environmental/calendar/water contracts for the satellite material.

Fixtures are deterministic physical inputs; no server, inference or CUDA jobs.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools._bootstrap import activate
activate()

from types import SimpleNamespace
import json
import unittest

import numpy as np

from terrain_render_v2 import evaluate, colors, appearance, mineral_color, ground_tint, display_shoulder, water_color, colorize_surface, snow_cover, perennial_snow, clearing_fraction, clearing_target, stand_coverage
from terrain_soil import NAMES


class RenderV2ClearingProfileTests(unittest.TestCase):
    def test_optical_stand_density_keeps_dense_cover_without_growing_from_zero(self):
        cover = np.linspace(0., .98, 101)
        np.testing.assert_allclose(stand_coverage(cover, 0., 0., 0.), cover, atol=1e-12)
        for residual in [-3., 0., 3.]:
            actual = stand_coverage(cover, residual, .8, 0.)
            self.assertTrue(np.all(np.diff(actual) >= 0.))
            self.assertTrue(np.all((actual >= 0.) & (actual < 1.)))
            self.assertEqual(float(actual[0]), 0.)
        # Age/productivity fluctuations do not turn a dense stand into a large
        # mostly empty cell. Physical eligibility is multiplied afterwards.
        self.assertGreater(float(stand_coverage(.95, -2., 1., 0.)), .7)
        self.assertLess(float(stand_coverage(.2, -2., 1., 0.)), .2)

    def test_humid_woodland_uses_less_large_disturbance(self):
        humid = np.linspace(.1, 4., 100)
        fraction = clearing_target(1., humid, 0.)
        self.assertTrue(np.all(np.diff(fraction) <= 1e-12))
        self.assertAlmostEqual(float(fraction[0]), .4)
        self.assertAlmostEqual(float(fraction[-1]), .08)
        np.testing.assert_array_equal(clearing_target(0., humid, 1.), 0.)
        self.assertAlmostEqual(float(clearing_target(1., 4., 1.)), .012)

    def test_progressive_cover_has_no_hard_interior_threshold(self):
        score = np.linspace(-6, 6, 24001)
        opened = clearing_fraction(score, 1.)
        self.assertTrue(np.all(np.diff(opened) >= 0))
        self.assertTrue(np.all((opened > 0) & (opened <= .55)))
        self.assertLess(float(np.max(np.diff(opened)/np.diff(score))), .12)
        self.assertAlmostEqual(float(clearing_fraction(0., 1.)), 1/(1+np.exp(1.702/1.15)))

    def test_thinning_mean_and_bound_do_not_empty_a_stand(self):
        # The bounded policy replaces large empty cells, not physical rock,
        # bog, treeline or small gaps between individual candidate crowns.
        score = np.linspace(-9.2, 9.2, 120001)
        density = np.exp(-.5*(score/1.15)**2)
        density /= density.sum()
        for target in [.01, .027, .06, .09, .18, .27, .4]:
            threshold = 1.15*np.log((1-target)/target)/1.702
            actual = clearing_fraction(score, threshold)
            with self.subTest(target=target):
                self.assertTrue(np.all(actual <= .55))
                self.assertTrue(np.all(1-actual >= .45-1e-12))
                if target <= .275:
                    self.assertAlmostEqual(float(np.sum(actual*density)), target, places=10)
                else:
                    self.assertLess(float(np.sum(actual*density)), target)


class RenderV2EnvironmentTests(unittest.TestCase):
    def setUp(self):
        yy, xx = np.mgrid[:32, :48]
        self.point = np.stack((xx*5100.+1e6, yy*5100.-2e6, np.zeros_like(xx)), -1).astype(np.float32)
        self.height = np.full(xx.shape, 300., np.float32)
        self.gradient = np.stack((.25*np.sin(xx/8), .12*np.cos(yy/5)), -1).astype(np.float32)
        self.tpi = (.1*np.sin(xx/11)*np.cos(yy/9)).astype(np.float32)

    def environment(self, date=.5, *, ts=24., tw=-10., ps=600., pw=450., sand=.4):
        field = lambda v: np.full(self.height.shape, v, np.float32)
        return evaluate(self.point, self.height, self.gradient, self.tpi,
                        field(ts), field(tw), field(ps), field(pw), field(sand), date=date, seed=42)

    def test_repeated_queries_and_subregions_have_identical_properties(self):
        whole = self.environment(.27)
        again = self.environment(.27)
        region = (slice(4, 21), slice(8, 32))
        field = lambda v: np.full(self.height[region].shape, v, np.float32)
        cropped = evaluate(self.point[region], self.height[region], self.gradient[region], self.tpi[region],
                           field(24), field(-10), field(600), field(450), field(.4), date=.27, seed=42)
        for key, value in whole.items():
            with self.subTest(property=key):
                np.testing.assert_array_equal(value, again[key])
                np.testing.assert_array_equal(value[region], cropped[key])

    def test_shared_gpu_reference_fixture_matches_cpu(self):
        fixture=Path(__file__).resolve().parents[1]/'fixtures/render_v2_environment.json'
        for case in json.loads(fixture.read_text(encoding='utf-8'))['cases']:
            inputs={key:np.asarray(case[key],np.float32)
                    for key in ('point','height','gradient','tpi','ts','tw','ps','pw','sand','bank')}
            actual=evaluate(**inputs,date=case['date'],seed=case['seed'])
            for key, expected in case['expected'].items():
                with self.subTest(case=case['name'],property=key):
                    self.assertAlmostEqual(float(actual[key]),expected,delta=2e-5)

    def test_calendar_wraps_and_reverses_between_hemispheres(self):
        january = self.environment(0.)
        december_end = self.environment(1.)
        south = self.environment(.5, ts=-10., tw=24., ps=450., pw=600.)
        for key in january:
            np.testing.assert_allclose(january[key], december_end[key], atol=2e-6, rtol=0)
            np.testing.assert_allclose(january[key], south[key], atol=2e-6, rtol=0)

    def test_structural_stands_are_stable_while_leaves_and_snow_change(self):
        winter, summer = self.environment(0.), self.environment(.5)
        for key in ('mineral', 'sand', 'clay', 'organic', 'rock', 'woody', 'height', 'needle'):
            np.testing.assert_array_equal(winter[key], summer[key])
        self.assertGreater(float(summer['leaf'].mean()-winter['leaf'].mean()), .3)
        self.assertGreater(float(winter['snow'].mean()-summer['snow'].mean()), .3)

    def test_cover_and_substrate_fractions_remain_physical(self):
        for sand in (.05, .5, .95):
            e = self.environment(sand=sand)
            for key in ('sand', 'clay', 'organic', 'rock', 'grass', 'woody', 'needle', 'leaf', 'snow', 'ice'):
                self.assertTrue(np.isfinite(e[key]).all(), key)
                self.assertGreaterEqual(float(e[key].min()), 0., key)
                self.assertLessEqual(float(e[key].max()), 1., key)
            self.assertLessEqual(float((e['sand']+e['clay']+e['organic']).max()), 1.00001)

    def test_ground_enrichment_preserves_reference_and_overlay_limits(self):
        e = self.environment(.5, ts=35, tw=20, ps=10, pw=10)
        base = np.full(self.height.shape+(3,), .45, np.float32)
        e.update(rock=np.ones_like(self.height), snow=np.zeros_like(self.height))
        e['mineral'] = np.full_like(self.height, .5)
        np.testing.assert_array_equal(ground_tint(base, e), base)

        e['mineral'] = np.linspace(0, 1, self.height.size).reshape(self.height.shape).astype(np.float32)
        enriched = ground_tint(base, e)
        self.assertGreater(float(np.ptp(enriched)), .04)
        self.assertTrue(np.all(enriched >= base*.86-1e-6))
        self.assertTrue(np.all(enriched <= base*1.14+1e-6))
        np.testing.assert_array_equal(ground_tint(base, appearance(e, .15625)),
                                      ground_tint(base, appearance(e, 39.0625)))
        e['snow'] = np.ones_like(self.height)
        np.testing.assert_array_equal(ground_tint(base, e), base)
        e.update(snow=np.zeros_like(self.height), rock=np.zeros_like(self.height),
                 grass=np.ones_like(self.height))
        np.testing.assert_array_equal(ground_tint(base, e), base)

    def test_highlight_rolloff_preserves_midtones_hue_and_relief(self):
        ramp = np.linspace(.1, 2., 2000)
        rgb = ramp[:, None]*np.array([.94, 1., .98])
        mapped = display_shoulder(rgb)
        np.testing.assert_array_equal(mapped[ramp<=.8], rgb[ramp<=.8])
        self.assertTrue(np.all(np.diff(mapped[:, 1]) > 0))
        self.assertLess(float(mapped.max()), 1.)
        np.testing.assert_allclose(mapped[:, 0]/mapped[:, 1], .94, atol=1e-7)
        np.testing.assert_allclose(mapped[:, 2]/mapped[:, 1], .98, atol=1e-7)

    def test_drought_reduces_cover_and_mineral_provinces_are_not_one_colour(self):
        arid = self.environment(ts=35, tw=20, ps=10, pw=10)
        wet = self.environment(ts=28, tw=22, ps=1000, pw=1200)
        self.assertLess(float(arid['woody'].mean()), .03)
        self.assertGreater(float(wet['woody'].mean()), .6)
        substrate = colors(arid, 1)
        self.assertGreater(float(substrate[..., 0].std()), .015)
        self.assertGreater(float(np.ptp(substrate[..., 0])), .05)

    def test_water_freezes_by_type_and_thaws_in_summer(self):
        winter = self.environment(0., ts=22, tw=-6)
        summer = self.environment(.5, ts=22, tw=-6)
        shape = self.height.shape
        depth, cover = np.full(shape, 15.), np.ones(shape)
        lake = water_color(depth, np.full(shape, 2.), cover, winter, self.point, 1., 42)
        flowing = water_color(depth, np.full(shape, 3.), cover, winter, self.point, 1., 42)
        thawed = water_color(depth, np.full(shape, 2.), cover, summer, self.point, 1., 42)
        self.assertGreater(float(lake.mean()-flowing.mean()), .12)
        self.assertGreater(float(lake.mean()-thawed.mean()), .15)
        self.assertTrue(np.isfinite(lake).all())

    def test_marine_shelf_stays_distinct_from_abyss_without_tinting_deep_lakes(self):
        e = self.environment(.5, ts=30., tw=20.)
        shape = self.height.shape
        cover = np.ones(shape)
        def water(depth, kind):
            return water_color(np.full(shape, depth), np.full(shape, kind), cover,
                               e, self.point, 3152., 42)
        shelf, abyss = water(150., 1.), water(3000., 1.)
        self.assertGreater(float((shelf-abyss)[..., 1].mean()), .08)
        self.assertTrue(np.all(shelf > abyss))
        np.testing.assert_allclose(water(150., 2.), water(3000., 2.), atol=1e-7)

    def test_clear_marine_shallows_retain_bed_response_and_inland_optics_are_unchanged(self):
        e = self.environment(.5, ts=30., tw=20.)
        shape = self.height.shape
        cover = np.ones(shape)
        # Warm open water isolates absorption from all frozen-water overlays.
        shallow = water_color(np.zeros(shape), np.ones(shape), cover, e, self.point, 3152., 42)
        for depth in (20., 30.):
            marine = water_color(np.full(shape, depth), np.ones(shape), cover, e, self.point, 3152., 42)
            expected = shallow*np.exp(-depth*.035)+np.array([.045, .16, .27])*(1-np.exp(-depth*.035))
            np.testing.assert_allclose(marine, expected, atol=1e-7)
            self.assertGreater(np.exp(-depth*.035), .3)
            for kind in (2., 3.):
                inland = water_color(np.full(shape, depth), np.full(shape, kind), cover, e, self.point, 3152., 42)
                historical = shallow*np.exp(-depth*.15)+np.array([.025, .12, .21])*(1-np.exp(-depth*.15))
                np.testing.assert_allclose(inland, historical, atol=1e-7)

    def test_cold_flat_snow_fully_hides_substrate(self):
        field = lambda v: np.full(self.height.shape, v, np.float32)
        e = evaluate(self.point, self.height, self.gradient*0, self.tpi*0,
                     field(-20), field(-30), field(900), field(900), field(.3), seed=42)
        np.testing.assert_array_equal(e['snow'], 1)
        np.testing.assert_allclose(colors(e), np.broadcast_to([.88, .91, .91], (*self.height.shape, 3)), atol=1e-6)

    def test_partial_snow_is_climatic_not_stand_or_bank_patches(self):
        field = lambda v: np.full(self.height.shape, v, np.float32)
        args = (self.point, self.height, self.gradient*0, self.tpi*0,
                field(-1), field(-1), field(200), field(200), field(.3))
        reference = evaluate(*args, seed=42, footprint=.1)
        self.assertGreater(float(reference['snow'].min()), .1)
        self.assertLess(float(reference['snow'].max()), .9)
        for footprint in (.1, 41.9, 240., 7680.):
            actual = evaluate(*args, bank=field(1), seed=789, footprint=footprint,
                              land_point=self.point+120000)
            np.testing.assert_array_equal(actual['snow'], reference['snow'])
        np.testing.assert_array_equal(reference['snow'], np.maximum(
            snow_cover(field(-1), field(400/260), field(0)),
            perennial_snow(field(-1), field(400/260), field(0))))

    def test_annually_cold_wet_ground_retains_snow_through_warmest_season(self):
        field = lambda v: np.full(self.height.shape, v, np.float32)
        args = (self.point, self.height, self.gradient*0, self.tpi*0,
                field(-.874), field(-16.726), field(915), field(860), field(.3))
        annual_floor = perennial_snow(field(-.874), field(5), field(0))
        self.assertGreater(float(annual_floor.min()), .7)
        summer = evaluate(*args, date=.5, seed=42, footprint=.1)
        np.testing.assert_array_equal(summer['snow'], annual_floor)
        for date in (0., .125, .25, .375, .5, .75):
            actual = evaluate(*args, date=date, bank=field(1), seed=789,
                              footprint=7680., land_point=self.point+120000)
            self.assertTrue(np.all(actual['snow'] >= annual_floor))

    def test_annual_floor_keeps_supply_cliffs_and_warm_climate_contracts(self):
        warmest = np.linspace(-5., 5., 201, dtype=np.float32)
        actual = perennial_snow(warmest, np.ones_like(warmest), np.zeros_like(warmest))
        self.assertTrue((np.diff(actual) <= 0).all())
        self.assertLess(float(np.abs(np.diff(actual)).max()), .016)
        np.testing.assert_array_equal(perennial_snow(warmest, warmest*0, warmest*0), 0)
        np.testing.assert_array_equal(perennial_snow(warmest, warmest*0+1, warmest*0+1.5), 0)
        # DFC/arid climates with a genuinely warm season retain the previous
        # seasonal response exactly; the annual floor does not grow snow there.
        field = lambda v: np.full(self.height.shape, v, np.float32)
        for hot, cold, rain in ((14., -18., 700.), (30., 10., 10.)):
            for date in (0., .25, .5):
                e = evaluate(self.point, self.height, self.gradient*0, self.tpi*0,
                             field(hot), field(cold), field(rain), field(rain), field(.3), date=date)
                phase = .5-.5*np.cos(2*np.pi*date)
                supply = np.clip(2*rain/max(120., 20*(hot+cold)*.5+280), 0, 5)
                np.testing.assert_array_equal(e['snow'], snow_cover(field(cold+(hot-cold)*phase), field(supply), field(0)))

    def test_annual_floor_is_zero_above_warmest_exposure_bound(self):
        # Exposure spans [0, 1], shifting the warmest temperature by at most
        # 0.325 C. Thus even the coldest exposure cannot add annual snow once
        # the physical warmest season is 2.825 C or warmer.
        exposure = np.linspace(0., 1., 1001)
        for warmest in (2.825, 3., 10., 30.):
            np.testing.assert_array_equal(
                perennial_snow(warmest+.65*(exposure-.5), 5., 0.), 0.)

    def test_snow_retention_is_continuous_and_cliffs_expose_rock(self):
        slopes = np.linspace(0, 1.5, 301, dtype=np.float32)
        snow = snow_cover(np.full_like(slopes, -20), np.ones_like(slopes), slopes)
        np.testing.assert_array_equal(snow[slopes <= .45], 1)
        np.testing.assert_array_equal(snow[slopes >= 1.25], 0)
        self.assertTrue((np.diff(snow) <= 0).all())
        self.assertLess(float(np.abs(np.diff(snow)).max()), .01)
        self.assertEqual(float(snow_cover(np.float32(5), np.float32(1), np.float32(0))), 0)

    def test_appearance_reveal_preserves_ecological_fields(self):
        e = self.environment(ts=35, tw=20, ps=10, pw=10)
        original = {key: value.copy() for key, value in e.items()}
        shown = appearance(e, 3152)
        np.testing.assert_array_equal(shown['mineral'], .5)
        np.testing.assert_array_equal(shown['tone'], .5)
        for key in e:
            np.testing.assert_array_equal(e[key], original[key])
            if key not in ('mineral', 'tone'):
                np.testing.assert_array_equal(shown[key], e[key])

    def test_land_domain_does_not_change_canonical_province_or_planar_default(self):
        args = (self.point, self.height, self.gradient, self.tpi, 24., -10., 600., 450., .4)
        canonical = evaluate(*args, seed=42)
        same = evaluate(*args, seed=42, land_point=self.point)
        shifted = evaluate(*args, seed=42, land_point=self.point+np.array([3456, -789, 1200], np.float32))
        for key in canonical:
            np.testing.assert_array_equal(canonical[key], same[key])
        np.testing.assert_array_equal(canonical['mineral'], shifted['mineral'])
        self.assertGreater(float(np.abs(canonical['moisture']-shifted['moisture']).mean()), .001)

    def test_ferruginous_palette_keeps_muted_warm_stone(self):
        color = mineral_color({'mineral': np.array(1., np.float32)})
        self.assertGreater(float(color[0]-color[2]), .1)
        self.assertLess(float(color[0]-color[1]), .2)

    def test_riparian_bank_cannot_create_trees_below_thermal_limit(self):
        field = lambda v: np.full(self.height.shape, v, np.float32)
        e = evaluate(self.point, self.height, self.gradient*0, self.tpi*0,
                     field(-20), field(-30), field(900), field(900), field(.3),
                     bank=field(1), seed=42)
        np.testing.assert_array_equal(e['woody'], 0)
        np.testing.assert_array_equal(e['height'], 0)


class RenderV2SurfaceTests(unittest.TestCase):
    def setUp(self):
        self.n = 32
        self.xs = (np.arange(self.n)+.5)*30
        self.ys = (np.arange(self.n)+.5)*30
        shape = (16, 32)
        constant = lambda v: np.full(shape, v, np.float32)
        layers = dict(temperature_summer=constant(75/90), temperature_winter=constant(70/90),
                      precip_summer=constant(.08), precip_winter=constant(.1))
        values = (.48, .36, .23, .55, .40, .29, .72, .20, .08)
        layers.update({key: constant(v) for key, v in zip(NAMES, values)})
        self.world = SimpleNamespace(bounds=(0., 0., 960., 960.), width=32, height=16,
                                     height_m=constant(0), layers=layers, seed=42, periodic_longitude=False)
        self.elevation = np.full((self.n, self.n), -20, np.float32)

    def render(self, hydro, mode='render-v2', season=.5):
        return colorize_surface(self.world, self.xs, self.ys, self.elevation, 30.,
                                {'world_topology': 'plane'}, {'season': season},
                                mode=mode, hydrology=hydro)

    def water(self, kind=0., cover=0.):
        hydro = np.zeros((self.n, self.n, 4), np.float32)
        hydro[:] = (20., cover, kind, 1.)
        return hydro

    def test_authoritative_dry_basin_is_land_and_lake_depth_is_water(self):
        dry = self.render(self.water())
        lake = self.render(self.water(2., 1.))
        self.assertGreater(float(dry[..., 0].mean()-lake[..., 0].mean()), .1)
        self.assertGreater(float(lake[..., 2].mean()-lake[..., 0].mean()), .08)
        self.assertGreater(float(dry[..., 0].mean()-dry[..., 2].mean()), .05)

    def test_vegetation_diagnostic_excludes_water_and_render_is_finite(self):
        hydro = self.water(2., 1.)
        vegetation = self.render(hydro, 'render-v2-vegetation')
        np.testing.assert_array_equal(vegetation[..., :2], 0.)
        for mode in ('render-v2', 'render-v2-substrate', 'render-v2-moisture', 'render-v2-snow'):
            with self.subTest(mode=mode):
                rgb = self.render(hydro, mode)
                self.assertEqual(rgb.shape, (*self.elevation.shape, 3))
                self.assertTrue(np.isfinite(rgb).all())
                self.assertGreaterEqual(float(rgb.min()), 0.)
                self.assertLessEqual(float(rgb.max()), 1.)

    def test_zero_wet_fraction_does_not_paint_water(self):
        dry = self.render(self.water())
        empty_lake = self.render(self.water(2., 0.))
        self.assertLess(float(np.abs(dry-empty_lake).max()), .005)

    def test_nonfinite_hydrology_has_no_authority(self):
        baseline = self.render(None)
        for channel in range(4):
            hydro = self.water(2., 1.)
            hydro[..., channel] = np.nan
            with self.subTest(channel=channel):
                rgb = self.render(hydro)
                self.assertTrue(np.isfinite(rgb).all())
                np.testing.assert_array_equal(rgb, baseline)


if __name__ == '__main__':
    unittest.main()

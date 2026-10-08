"""CPU checks of physical units and globally consistent NN conditioning."""
import contextlib
import hashlib
import importlib
import io
from pathlib import Path
import sys
import tempfile
from unittest import mock
import unittest

import numpy as np
import torch

import terrain_conditioning as conditioning
import terrain_generation as generation


def source_fixture():
    probabilities = np.linspace(.0001, .9999, 65)
    latitudes = np.arange(0, 91, 5)
    climate = np.empty((4, len(latitudes), len(probabilities)), np.float32)
    for k, lat in enumerate(latitudes):
        climate[0, k] = 28 - .5 * lat + (probabilities - .5) * 6
        climate[1, k] = 100 + lat * 8 + probabilities * 300
        climate[2, k] = 100 + probabilities * 2400
        climate[3, k] = 10 + probabilities * 90
    return {"probabilities": probabilities, "latitudes": latitudes,
            "climate": climate,
            "noise_quantiles": np.tile(np.linspace(-.65, .65, 64), (5, 1))}


class HeightStub:
    def __init__(self, offset=0):
        self.offset = offset
        self.requests = []

    def sample_height_m(self, xs, ys):
        self.requests.append((np.asarray(xs).copy(), np.asarray(ys).copy()))
        x, y = np.meshgrid(xs, ys)
        return (self.offset + x / 100 - y / 200).astype(np.float32)


class ConstantHeightStub:
    def sample_height_m(self, xs, ys):
        return np.zeros((len(ys), len(xs)), np.float32)


class ConditioningTests(unittest.TestCase):
    def factory(self, seed=7, heightmap=None, stats=None):
        return conditioning.BootstrapConditioning(
            seed, heightmap=heightmap or HeightStub(),
            stats=source_fixture() if stats is None else stats)

    def test_physical_height_and_worldclim_units_survive_sampling(self):
        height = HeightStub(offset=-12000)
        factory = self.factory(heightmap=height)
        xs, ys = np.array([-100., 0., 100.]), np.array([-400., 0., 400.])
        fields = factory.sample(xs, ys)
        np.testing.assert_array_equal(fields[0], height.sample_height_m(xs, ys))
        self.assertEqual(fields.dtype, np.float32)
        self.assertEqual(fields.shape, (5, 3, 3))
        self.assertTrue(np.isfinite(fields).all())
        self.assertGreater(fields[2].min(), 100)  # BIO4 remains std *100
        self.assertGreater(fields[3].min(), 100)  # BIO12 remains mm/year
        self.assertGreater(fields[4].min(), 10)   # BIO15 remains percent CV
        self.assertLess(fields[0].min(), -10000)  # adapter does not clip height

    def test_lapse_applies_once_without_temperature_stretch_or_clamp(self):
        flat = self.factory(heightmap=HeightStub())
        high = self.factory(heightmap=HeightStub(offset=15000))
        xs, ys = np.array([0.]), np.array([0.])
        low_fields, high_fields = flat.sample(xs, ys), high.sample(xs, ys)
        beta = conditioning.lapse_rate(low_fields[3])
        np.testing.assert_allclose(high_fields[1] - low_fields[1], beta * 15000,
                                   atol=1e-5, rtol=1e-6)
        np.testing.assert_array_equal(high_fields[2:], low_fields[2:])
        self.assertLess(high_fields[1, 0, 0], -10)
        # Ocean depths are not interpreted as positive atmosphere altitude.
        ocean_fields = self.factory(heightmap=HeightStub(offset=-15000)).sample(xs, ys)
        np.testing.assert_array_equal(ocean_fields[1:], low_fields[1:])

    def test_nested_overlapping_windows_seed_and_request_order_are_invariant(self):
        factory = self.factory(seed=0)
        xs = (np.arange(-9, 11) + .5) * 7680
        ys = (np.arange(-7, 13) + .5) * 7680
        narrow = factory.sample(xs[3:10], ys[5:12])
        whole = factory.sample(xs, ys)
        np.testing.assert_array_equal(narrow, whole[:, 5:12, 3:10])
        np.testing.assert_array_equal(self.factory(seed=0).sample(xs, ys), whole)
        different = self.factory(seed=1234).sample(xs, ys)
        np.testing.assert_array_equal(different[0], whole[0])
        self.assertFalse(np.array_equal(different[1:], whole[1:]))

    def test_row_batching_and_traversal_preserve_the_same_world_samples(self):
        factory = self.factory()
        xs = np.linspace(-1e6, 1e6, 1024)
        ys = np.linspace(-1e6, 1e6, 129)  # crosses the 131072-point row batch
        whole = factory.sample(xs, ys)
        np.testing.assert_array_equal(factory.sample(xs, ys[-2:]), whole[:, -2:])
        np.testing.assert_array_equal(factory.sample(xs[::-1], ys[::-1]), whole[:, ::-1, ::-1])

    def test_terrestrial_climate_uses_high_seed_bits_on_constant_height(self):
        xs = (np.arange(-19, 21) + .5) * conditioning.COARSE_RESOLUTION
        ys = (np.arange(-17, 23) + .5) * conditioning.COARSE_RESOLUTION
        original = self.factory(seed=0, heightmap=ConstantHeightStub()).sample(xs, ys)
        for seed in (2**31, 2**40, 2**63, 2**64 - 1):
            with self.subTest(seed=seed):
                factory = self.factory(seed=seed, heightmap=ConstantHeightStub())
                different = factory.sample(xs, ys)
                np.testing.assert_array_equal(different[0], original[0])
                for channel in range(1, 5):
                    self.assertFalse(np.array_equal(different[channel], original[channel]))
                np.testing.assert_array_equal(factory.sample(xs[3:11], ys[5:13]),
                                              different[:, 5:13, 3:11])
                np.testing.assert_array_equal(self.factory(seed=seed, heightmap=ConstantHeightStub()).sample(xs, ys),
                                              different)

    def test_terrestrial_climate_channels_have_independent_seed_domains(self):
        factory = self.factory(seed=2**63, heightmap=ConstantHeightStub())
        self.assertEqual(len({noise.seed for noise in factory.noises}), 4)
        xs = (np.arange(-19, 21) + .5) * conditioning.COARSE_RESOLUTION
        ys = (np.arange(-17, 23) + .5) * conditioning.COARSE_RESOLUTION
        x, y = np.meshgrid(xs / conditioning.COARSE_RESOLUTION - .5,
                           ys / conditioning.COARSE_RESOLUTION - .5)
        coordinates = np.ascontiguousarray(np.stack((x.ravel(), y.ravel())), dtype=np.float32)
        noises = [noise.gen_from_coords(coordinates) for noise in factory.noises]
        for i in range(4):
            for j in range(i):
                self.assertFalse(np.array_equal(noises[i], noises[j]))
        original = factory.sample(xs, ys)
        # Changing BIO4's noise stream changes BIO4 alone: no shared generator
        # state leaks into the other three physical climate fields.
        factory.noises[1] = conditioning._noise(123456789, .006, 4)
        changed = factory.sample(xs, ys)
        self.assertFalse(np.array_equal(original[2], changed[2]))
        np.testing.assert_array_equal(original[[0, 1, 3, 4]], changed[[0, 1, 3, 4]])

    def test_halfcell_coordinates_and_signed_sqrt_exactly_once(self):
        height = HeightStub()
        factory = self.factory(heightmap=height)
        physical = factory.sample_raw(-3, -2, 4, 3)
        xs, ys = height.requests[-1]
        np.testing.assert_array_equal(xs, (np.arange(-3, 4) + .5) * 7680)
        np.testing.assert_array_equal(ys, (np.arange(-2, 3) + .5) * 7680)
        encoded = factory(-3, -2, 4, 3)
        self.assertEqual(encoded.dtype, torch.float32)
        decoded_height = np.sign(encoded[0].numpy()) * encoded[0].numpy() ** 2
        np.testing.assert_allclose(decoded_height, physical[0], rtol=2e-7, atol=2e-5)
        np.testing.assert_array_equal(encoded[1:].numpy(), physical[1:])
        copied = factory.finalize(physical)
        np.testing.assert_array_equal(copied, physical)
        self.assertFalse(np.shares_memory(copied, physical))

    def test_latitude_interpolation_uses_fixed_source_tables(self):
        factory = self.factory()
        ranks = np.array([[.25, .75], [.25, .75]])
        result = factory._quantile(0, np.array([[2.5], [12.5]]), ranks)
        expected = 28 - .5 * np.array([[2.5], [12.5]]) + (ranks - .5) * 6
        np.testing.assert_allclose(result, expected, atol=2e-6)
        self.assertAlmostEqual(float(conditioning.latitude([-1e7])[0]), 90.)
        self.assertAlmostEqual(float(conditioning.latitude([1e7])[0]), -90.)

    def test_empty_invalid_coordinates_and_invalid_provider(self):
        factory = self.factory()
        self.assertEqual(factory.sample([], [0.]).shape, (5, 1, 0))
        self.assertEqual(factory.sample([0.], []).shape, (5, 0, 1))
        for xs, ys in (([[0.]], [0.]), ([np.nan], [0.]), ([0.], [np.inf])):
            with self.assertRaises(ValueError):
                factory.sample(xs, ys)
        broken = mock.Mock()
        broken.sample_height_m.return_value = np.zeros((2, 2))
        with self.assertRaisesRegex(ValueError, "Bootstrap"):
            self.factory(heightmap=broken).sample([0.], [0.])
        stats = source_fixture()
        stats["climate"][0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "statistics"):
            self.factory(stats=stats)
        for invalid in (np.zeros(5), np.ones((5, 1)), np.tile(np.linspace(.65, -.65, 64), (5, 1))):
            stats = source_fixture()
            stats["noise_quantiles"] = invalid
            with self.assertRaisesRegex(ValueError, "statistics"):
                self.factory(stats=stats)

    def test_preview_transport_matches_factory_physical_fields(self):
        factory = self.factory()
        with mock.patch.object(conditioning, "make_conditioning_factory", return_value=factory):
            preview = conditioning.sample_conditioning_preview(7, "terrestrial-earthlike", [0., 500.], [0.])
        np.testing.assert_array_equal(preview["elev"], preview["fields"][0])
        np.testing.assert_array_equal(preview["climate"][:4], preview["fields"][1:])
        np.testing.assert_array_equal(preview["climate"][4], conditioning.lapse_rate(preview["fields"][3]).astype(np.float32))

    def test_cached_factory_profiles_and_lazy_native_provider(self):
        conditioning.make_conditioning_factory.cache_clear()
        with mock.patch.object(conditioning, "BootstrapConditioning") as bootstrap:
            for profile in (p for p in conditioning.WORLD_PROFILES[1:] if p != 'orogen'):
                first = conditioning.make_conditioning_factory(7, profile)
                self.assertIs(conditioning.make_conditioning_factory(7, profile), first)
                self.assertEqual(bootstrap.call_args.kwargs["style"], profile.removeprefix("terrestrial-"))
        with self.assertRaises(ValueError):
            conditioning.make_conditioning_factory(7, "A4")
        conditioning.make_conditioning_factory.cache_clear()
        provider = HeightStub()
        native = mock.Mock()
        native.get_heightmap.return_value = provider
        with mock.patch.dict(sys.modules, {"terrain_bootstrap": native}):
            factory = conditioning.BootstrapConditioning(7, "continents", stats=source_fixture())
        native.get_heightmap.assert_called_once_with(7, "continents")
        self.assertIs(factory.heightmap, provider)

    def test_natural_preserves_protected_reference_bytes(self):
        # Captured from the protected original class before its geometry module
        # was retired. The test has no dependency on that retired module.
        hashes = {
            0: "5e360c58f27b3704da2d4009acd492b6f173608376170d7ba0778a28792f63c4",
            7: "567eafc03e38addaf27bbf0ea269be3d2519d59c134a31ad254ca47ea3dee40c",
            2**40 + 123: "2e744f16f0b9bd8cccb65cdc63a4dbdbf1df0a0d083a9671d8ddf696a106041d",
        }
        for seed, expected in hashes.items():
            actual = conditioning.NaturalConditioning(seed)(-3, -2, 5, 6).numpy()
            self.assertEqual(hashlib.sha256(actual.tobytes()).hexdigest(), expected)

    def test_natural_matches_installed_upstream_factory_with_seed_zero_repair(self):
        package = str(Path(__file__).parent / "terrain-diffusion")
        with mock.patch.object(sys, "path", [package, *sys.path]):
            upstream = importlib.import_module("terrain_diffusion.inference.synthetic_map")
        with mock.patch.object(upstream, "STATS_CACHE_PATH", str(conditioning.STATS_PATH)), contextlib.redirect_stdout(io.StringIO()):
            for seed in (0, 7, 2**40 + 123):
                # Upstream uses `seed or random(...)`. 2**31 is nonzero while
                # its masked Perlin streams equal the deterministic seed-0 fix.
                reference = upstream.make_synthetic_map_factory(seed=seed or 2**31)
                actual = conditioning.NaturalConditioning(seed)
                for cj0, ci0, cj1, ci1 in ((-3, -2, 5, 6), (11, -12, 20, -3)):
                    observed = actual(cj0, ci0, cj1, ci1).numpy()
                    expected = reference(cj0, ci0, cj1, ci1).numpy()
                    np.testing.assert_array_equal(observed[[0, 1, 3, 4]], expected[[0, 1, 3, 4]])
                    # Protected BIO4 groups float32 additions differently from
                    # installed upstream; preserve that original rounding.
                    np.testing.assert_allclose(observed[2], expected[2], rtol=1e-6, atol=1e-4)
                    xs = (np.arange(cj0, cj1) + .5) * conditioning.COARSE_RESOLUTION
                    ys = (np.arange(ci0, ci1) + .5) * conditioning.COARSE_RESOLUTION
                    np.testing.assert_array_equal(actual.sample(xs, ys),
                                                  actual.finalize(actual.sample_raw(cj0, ci0, cj1, ci1)))


class AdjustableConditioningTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.registry = mock.patch.object(generation, "REGISTRY_ROOT", Path(self.directory.name))
        self.registry.start()
        conditioning.make_conditioning_factory.cache_clear()
        conditioning._MACRO_CACHE.clear()
        self.height = HeightStub(offset=1200)
        self.height.metadata = {"height_sha256": "height-fixture"}
        native = mock.Mock()
        native.get_heightmap.return_value = self.height
        self.previous_native = sys.modules.get("terrain_bootstrap")
        sys.modules["terrain_bootstrap"] = native
        self.stats = mock.patch.object(conditioning, "source_distributions", side_effect=source_fixture)
        self.stats.start()

    def tearDown(self):
        self.stats.stop()
        if self.previous_native is None:
            sys.modules.pop("terrain_bootstrap", None)
        else:
            sys.modules["terrain_bootstrap"] = self.previous_native
        self.registry.stop()
        self.directory.cleanup()
        conditioning.make_conditioning_factory.cache_clear()
        conditioning._MACRO_CACHE.clear()

    def factory(self, overrides, seed=7, base="natural"):
        token = generation.register_generation(base, overrides)
        return conditioning.make_conditioning_factory(seed, token)

    def test_cond_only_control_uses_exact_installed_upstream_fields(self):
        package = str(Path(__file__).parent / "terrain-diffusion")
        with mock.patch.object(sys, "path", [package, *sys.path]):
            upstream = importlib.import_module("terrain_diffusion.inference.synthetic_map")
        with mock.patch.object(upstream, "STATS_CACHE_PATH", str(conditioning.STATS_PATH)), contextlib.redirect_stdout(io.StringIO()):
            for seed in (0, 7, 2**40 + 123):
                f = self.factory({"cond_snr": [.3] * 5}, seed)
                expected = upstream.make_synthetic_map_factory(seed=seed or 2**31)(-3, -2, 5, 6).numpy()
                np.testing.assert_array_equal(f(-3, -2, 5, 6).numpy(), expected)

    def test_frequency_and_octave_controls_change_only_selected_raw_channel(self):
        baseline = self.factory({"cond_snr": [.3] * 5})
        for options in ({"frequency_mult": [2, 1, 1, 1, 1]}, {"frequency_mult": [0, 1, 1, 1, 1]},
                        {"frequency_mult": [-2.5, 1, 1, 1, 1]}, {"frequency_mult": [25.125, 1, 1, 1, 1]},
                        {"octaves": [12, 2, 4, 4, 4]}, {"octaves": [1, 2, 4, 4, 4]}):
            changed = self.factory(options)
            a, b = baseline.sample_raw(-9, -7, 11, 13), changed.sample_raw(-9, -7, 11, 13)
            self.assertFalse(np.array_equal(a[0], b[0]))
            np.testing.assert_array_equal(a[1:], b[1:])

    def test_mixed_native_height_natural_climate_replaces_before_finalize_once(self):
        f = self.factory({"height_source": "native"})
        raw = f.sample_raw(-3, -2, 5, 6)
        expected = f.natural.finalize(raw)
        xs, ys = (np.arange(-3, 5) + .5) * 7680, (np.arange(-2, 6) + .5) * 7680
        np.testing.assert_array_equal(raw[0], self.height.sample_height_m(xs, ys))
        np.testing.assert_array_equal(f.sample(xs, ys), expected)
        encoded = f(-3, -2, 5, 6).numpy()
        np.testing.assert_array_equal(encoded[1:], expected[1:])
        np.testing.assert_array_equal(encoded[0], np.sign(expected[0]) * np.sqrt(np.abs(expected[0])))
        self.assertIs(f.heightmap, self.height)

    def test_native_climate_uses_selected_natural_height_and_full_seed_bits(self):
        options = {"climate_source": "native"}
        a, b = self.factory(options, 0), self.factory(options, 2**31)
        xs, ys = np.linspace(-1e6, 1e6, 50), np.linspace(-1e6, 1e6, 30)
        fields = a.sample(xs, ys)
        np.testing.assert_array_equal(fields[0], a.natural.sample_raw_coordinates(xs, ys)[0])
        np.testing.assert_array_equal(fields[1:], a.native._climate(fields[0], xs, ys))
        self.assertFalse(np.array_equal(fields[1:], b.sample(xs, ys)[1:]))
        changed = self.factory({"climate_source": "native", "frequency_mult": [1, 2, 1, 1, 1]})
        before, after = self.factory(options).sample(xs, ys), changed.sample(xs, ys)
        np.testing.assert_array_equal(before[[0, 2, 3, 4]], after[[0, 2, 3, 4]])
        self.assertFalse(np.array_equal(before[1], after[1]))

    def test_continental_fixed_global_lowpass_preserves_regional_residual_and_overlaps(self):
        f = self.factory({"height_source": "natural-continental", "continental_strength": .8})
        xs, ys = np.linspace(-8e5, 8e5, 37), np.linspace(-5e5, 5e5, 29)
        first = f.sample(xs[5:13], ys[7:15])
        whole = f.sample(xs, ys)
        np.testing.assert_array_equal(first, whole[:, 7:15, 5:13])
        np.testing.assert_array_equal(f.sample(xs[::-1], ys[::-1]), whole[:, ::-1, ::-1])
        raw = f.natural.sample_raw_coordinates(xs, ys)[0]
        u = np.sign(raw) * np.sqrt(np.abs(raw))
        actual_u = np.sign(whole[0]) * np.sqrt(np.abs(whole[0]))
        expected_delta = .8 * conditioning._sample_fixed_raster(f._macro_delta(), xs, ys)
        np.testing.assert_allclose(actual_u - expected_delta, u, atol=3e-5, rtol=1e-6)
        self.assertEqual(f._macro_delta().shape, (512, 1024))
        self.assertFalse(f._macro_delta().flags.writeable)
        self.assertEqual(len(conditioning._MACRO_CACHE), 1)
        zero = self.factory({"height_source": "natural-continental", "continental_strength": 0})
        np.testing.assert_array_equal(zero.sample(xs, ys), zero.natural.finalize(zero.natural.sample_raw_coordinates(xs, ys)))

    def test_zero_continental_strength_is_exact_upstream_nn_input(self):
        package = str(Path(__file__).parent / "terrain-diffusion")
        with mock.patch.object(sys, "path", [package, *sys.path]):
            upstream = importlib.import_module("terrain_diffusion.inference.synthetic_map")
        with mock.patch.object(upstream, "STATS_CACHE_PATH", str(conditioning.STATS_PATH)), contextlib.redirect_stdout(io.StringIO()):
            for seed in (0, 7):
                f = self.factory({"height_source": "natural-continental", "continental_strength": 0}, seed)
                expected = upstream.make_synthetic_map_factory(seed=seed or 2**31)(-3, -2, 5, 6).numpy()
                np.testing.assert_array_equal(f(-3, -2, 5, 6).numpy(), expected)
        self.assertEqual(len(conditioning._MACRO_CACHE), 0)

    def test_native_defaults_are_identical_when_only_snr_changes(self):
        f = self.factory({"cond_snr": [.3] * 5}, base="terrestrial-earthlike")
        baseline = conditioning.BootstrapConditioning(7, heightmap=self.height, stats=source_fixture())
        np.testing.assert_array_equal(f(-3, -2, 5, 6).numpy(), baseline(-3, -2, 5, 6).numpy())

    def test_adjustable_rows_and_preview_nn_coordinates_agree(self):
        f = self.factory({"height_source": "native", "frequency_mult": [1, 2, 1, 1, 1]})
        xs, ys = np.linspace(-1e6, 1e6, 1024), np.linspace(-1e6, 1e6, 129)
        fields = f.sample(xs, ys)
        np.testing.assert_array_equal(f.sample(xs, ys[-2:]), fields[:, -2:])
        x = (np.arange(-3, 5) + .5) * 7680
        y = (np.arange(-2, 6) + .5) * 7680
        physical = f.sample(x, y)
        np.testing.assert_array_equal(f.finalize(f.sample_raw(-3, -2, 5, 6)), physical)
        preview = conditioning.sample_conditioning_preview(7, f._terrain_generation_profile, x, y)
        np.testing.assert_array_equal(preview["fields"], physical)
        self.assertEqual(f.sample([], [0.]).shape, (5, 1, 0))
        with self.assertRaises(ValueError):
            f.sample([np.nan], [0.])

    def test_macro_lru_is_bounded_readonly_and_rebuilding_is_deterministic(self):
        with mock.patch.object(conditioning, "_MACRO_SHAPE", (16, 32)):
            original = self.factory({"height_source": "natural-continental", "macro_scale_km": 600})
            before = original._macro_delta().copy()
            for scale in (700, 800, 900, 1000, 1100):
                self.factory({"height_source": "natural-continental", "macro_scale_km": scale})._macro_delta()
            self.assertEqual(len(conditioning._MACRO_CACHE), conditioning._MACRO_CACHE_SIZE)
            for value in conditioning._MACRO_CACHE.values():
                self.assertFalse(value.flags.writeable)
            np.testing.assert_array_equal(original._macro_delta(), before)

    def test_nondefault_factory_cache_still_checks_registry_tamper(self):
        f = self.factory({"cond_snr": [.3] * 5})
        profile = f._terrain_generation_profile
        self.assertEqual(f.generation_settings, generation.resolve_generation(profile).settings)
        path = Path(self.directory.name) / (profile + ".json")
        path.write_text('{}')
        with self.assertRaises(ValueError):
            conditioning.make_conditioning_factory(7, profile)

    def test_deterministic_water_weighting_excludes_invalid_and_land_only_endpoint(self):
        values = np.array([-100, -50, 1, 10, 100, np.nan, np.inf, -32768])
        p = np.linspace(.0001, .9999, 64)
        helper = conditioning._weighted_height_quantiles
        full, reduced, land = helper(values, p, 0), helper(values, p, .8), helper(values, p, 1)
        self.assertTrue(np.all(reduced >= full))
        self.assertTrue(np.all(land > 0))
        np.testing.assert_array_equal(helper(values[::-1], p, .8), reduced)
        with self.assertRaises(ValueError):
            helper([-32768, np.nan], p, 1)

    def test_nondefault_water_changes_height_but_preserves_raw_climate_and_default_table(self):
        baseline = self.factory({"cond_snr": [.3] * 5})
        with mock.patch.object(conditioning, "_drop_height_quantiles", return_value=np.linspace(-100, 4000, 64)) as quantiles:
            changed = self.factory({"drop_water_pct": .8})
        quantiles.assert_called_once()
        a, b = baseline.sample_raw(-9, -7, 11, 13), changed.sample_raw(-9, -7, 11, 13)
        self.assertFalse(np.array_equal(a[0], b[0]))
        np.testing.assert_array_equal(a[1:], b[1:])
        np.testing.assert_array_equal(baseline.natural.stats["data_quantile_tables"][0], conditioning._natural_stats()["data_quantile_tables"][0])


if __name__ == "__main__":
    unittest.main()

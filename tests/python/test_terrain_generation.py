"""CPU registry validation, replay and immutable identity checks."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import terrain_generation as generation


class GenerationRegistryTests(unittest.TestCase):
    def test_orogen_startup_defaults(self):
        settings = generation.generator_schema()['defaults_by_profile']['orogen']
        self.assertEqual(settings['height_source'], 'orogen')
        self.assertEqual(settings['relief_pipeline'], 'orogen')
        self.assertEqual(settings['continental_style'], 'continents')
        self.assertEqual(settings['orogen_continent_count'], 3)
        self.assertEqual(settings['orogen_detail'], 100000)
        self.assertEqual(settings['orogen_continent_variety'], .85)
        self.assertGreater(settings['orogen_hydraulic'], 0)
        self.assertFalse(any(settings[key] for key in generation.OROGEN_GPU_PARAMETERS))

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "registry"
        self.patch = mock.patch.object(generation, "REGISTRY_ROOT", self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def test_all_default_profiles_collapse_without_writing(self):
        schema = generation.generator_schema()
        for base in generation.BASE_PROFILES:
            for overrides in (None, {}, schema["defaults_by_profile"][base]):
                self.assertEqual(generation.register_generation(base, overrides), base)
            descriptor = generation.resolve_generation(base)
            self.assertTrue(descriptor.is_default)
            self.assertEqual(generation.base_profile(base), base)
        self.assertFalse(self.root.exists())

    def test_normalized_tokens_replay_and_settings_are_immutable_copies(self):
        token = generation.register_generation("natural", {"macro_scale_km": 900, "frequency_mult": (1, 2, 1, 1, 1)})
        same = generation.register_generation("natural", {"frequency_mult": [1., 2., 1., 1., 1.], "macro_scale_km": 900.})
        self.assertEqual(token, same)
        self.assertRegex(token, r"^natural--g[0-9a-f]{24}$")
        descriptor = generation.resolve_generation(token)
        settings = descriptor.settings
        settings["frequency_mult"][1] = 9
        settings["height_source"] = "native"
        self.assertEqual(descriptor.settings["frequency_mult"][1], 2.)
        self.assertEqual(descriptor.settings["height_source"], "natural")
        with self.assertRaises(AttributeError):
            descriptor.base_profile = "terrestrial-earthlike"
        self.assertEqual(generation.resolve_generation(token), descriptor)
        self.assertEqual(len(list(self.root.iterdir())), 1)
        self.assertEqual(json.loads(json.dumps(descriptor.settings)), descriptor.settings)

    def test_invalid_types_bounds_vectors_and_unknown_keys_are_rejected(self):
        invalid = [{"foo": 1}, {"height_source": "macro"}, {"cond_snr": [.5] * 4},
                   {"octaves": [4, True, 4, 4, 4]}, {"octaves": [4, 2.5, 4, 4, 4]},
                   {"frequency_mult": [1, 1, float("nan"), 1, 1]},
                   {"cond_snr": [.5, .5, .5, .5, float("inf")]},
                   {"drop_water_pct": True}, {"continental_strength": float("inf")},
                   {"macro_scale_km": -1}, {"octaves": [0, 2, 4, 4, 4]},
                   {"cond_snr": [-.1] * 5},
                   {"cond_snr": [0] * 5},
                   {"drop_water_pct": 1.01}, {"frequency_mult": "11111"}, {"macro_scale_km": 10**1000}]
        for settings in invalid:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                generation.register_generation("natural", settings)
        for profile in ("A0", "macro-earthlike", "../natural", None):
            with self.assertRaises(ValueError):
                generation.resolve_generation(profile)
        self.assertFalse(self.root.exists())

    def test_values_outside_old_ui_ranges_are_preserved_and_replayed(self):
        settings = {"frequency_mult": [0.000001, 25.125, 1e3, 0, -2.5],
                    "octaves": [12, 15, 1, 2, 4], "cond_snr": [.0001, 4.5, .005, 12, .5],
                    "continental_strength": 1.23456789, "macro_scale_km": 75.25}
        token = generation.register_generation("natural", settings)
        descriptor = generation.resolve_generation(token)
        for key, value in settings.items():
            self.assertEqual(descriptor.settings[key], value)
        for scale in (0., .001, 2500.125):
            token = generation.register_generation("natural", {"macro_scale_km": scale, "continental_strength": -.25})
            self.assertEqual(generation.resolve_generation(token).settings["macro_scale_km"], scale)
        properties = generation.generator_schema()["properties"]
        for key in ("frequency_mult", "octaves", "cond_snr", "macro_scale_km", "continental_strength"):
            self.assertNotIn("maximum", properties[key])

    def test_tampered_missing_or_incomplete_receipts_fail_and_are_not_repaired(self):
        token = generation.register_generation("natural", {"height_source": "native"})
        path = self.root / (token + ".json")
        payload = json.loads(path.read_text())
        payload["settings"]["cond_snr"][0] = .1
        tampered = json.dumps(payload)
        path.write_text(tampered)
        with self.assertRaisesRegex(ValueError, "canonical token"):
            generation.resolve_generation(token)
        with self.assertRaises(ValueError):
            generation.register_generation("natural", {"height_source": "native"})
        self.assertEqual(path.read_text(), tampered)
        path.write_text('{"version":')
        with self.assertRaises(ValueError):
            generation.resolve_generation(token)
        path.unlink()
        with self.assertRaises(ValueError):
            generation.resolve_generation(token)
        self.assertFalse(path.exists())

    def test_descriptor_bootstrap_requirements_and_semantic_identity(self):
        for height, climate, expected in (("natural", "natural", False), ("native", "natural", True),
                                         ("natural", "native", True), ("natural-continental", "natural", True)):
            token = generation.register_generation("natural", {"height_source": height, "climate_source": climate,
                                                                "continental_style": "archipelago"})
            d = generation.resolve_generation(token)
            self.assertEqual(d.needs_bootstrap, expected)
            self.assertEqual(d.bootstrap_style, "archipelago")
        a = generation.register_generation("natural", {"continental_strength": 0.})
        b = generation.register_generation("natural", {"continental_strength": -0.})
        self.assertEqual(a, b)

    def test_atomic_write_failure_leaves_no_receipt_or_temporary_file(self):
        with mock.patch.object(generation.os, "replace", side_effect=OSError("disk failure")), self.assertRaises(OSError):
            generation.register_generation("natural", {"macro_scale_km": 700})
        self.assertEqual(list(self.root.iterdir()), [])

    def test_orogen_climate_palette_and_thresholds_replay_and_route_to_generator(self):
        overrides={'orogen_temperature_equator':33., 'orogen_biome_tropical_snow':4.25,
                   'orogen_biome_color_af':[.125,.5,.875], 'orogen_precip_shadow_reach_km':3200.}
        token=generation.register_generation('orogen',overrides)
        descriptor=generation.resolve_generation(token)
        for key,value in overrides.items():
            self.assertEqual(descriptor.settings[key],value)
            self.assertEqual(descriptor.bootstrap_options[key.removeprefix('orogen_')],value)
        self.assertNotEqual(token,generation.register_generation('orogen',{'orogen_temperature_equator':32.}))
        for invalid in ([.1,.2], [.1,.2,2.], [.1,float('nan'),.2]):
            with self.assertRaises(ValueError):
                generation.register_generation('orogen',{'orogen_biome_color_af':invalid})

    def test_pre_climate_configuration_links_keep_their_original_identity(self):
        saved=generation._defaults('orogen')
        for key in generation.OROGEN_CLIMATE_PARAMETERS:
            del saved[key]
        saved['orogen_detail']=20000
        token=generation._token('orogen',saved)
        self.root.mkdir()
        (self.root/(token+'.json')).write_bytes(generation._bytes(generation._payload('orogen',saved)))
        descriptor=generation.resolve_generation(token)
        self.assertEqual(descriptor.profile,token)
        self.assertEqual(descriptor.settings['orogen_detail'],20000)
        self.assertEqual(descriptor.settings['orogen_temperature_equator'],28.)
        self.assertEqual(descriptor.settings['orogen_biome_tropical_snow'],5.5)

    def test_city_erosion_routes_every_source_and_replays_own_parameters(self):
        for source in ('orogen','natural','native','natural-continental'):
            overrides=dict(height_source=source,climate_source='natural',relief_pipeline='city-gpu',
                           city_erosion_strength=.75,city_erosion_iterations=8,
                           city_erosion_talus=.4,city_erosion_motif_km=125.)
            token=generation.register_generation('natural',overrides)
            descriptor=generation.resolve_generation(token)
            self.assertTrue(descriptor.needs_bootstrap)
            self.assertEqual(descriptor.bootstrap_generator,'orogen')
            for key,value in overrides.items():
                self.assertEqual(descriptor.settings[key],value)
            for key in generation.CITY_EROSION_PARAMETERS:
                self.assertEqual(descriptor.bootstrap_options[key],overrides[key])
        for overrides in ({'city_erosion_strength':-1},{'city_erosion_iterations':1.5},
                          {'city_erosion_iterations':65},{'city_erosion_motif_km':0},
                          {'city_erosion_talus':float('nan')}):
            with self.assertRaises(ValueError):generation.register_generation('orogen',overrides)

    def test_pre_city_links_keep_their_canonical_tokens(self):
        saved=generation._defaults('orogen')
        for key in generation.CITY_EROSION_PARAMETERS:del saved[key]
        saved['orogen_detail']=20000
        token=generation._token('orogen',saved)
        self.root.mkdir()
        (self.root/(token+'.json')).write_bytes(generation._bytes(generation._payload('orogen',saved)))
        descriptor=generation.resolve_generation(token)
        self.assertEqual(descriptor.profile,token)
        self.assertEqual(descriptor.settings['relief_pipeline'],'orogen')
        self.assertEqual(descriptor.settings['city_erosion_strength'],1.)
        self.assertEqual(descriptor.settings['city_erosion_iterations'],12)

    def test_gpu_stages_are_explicit_opt_in_and_old_links_remain_cpu(self):
        for profile in generation.BASE_PROFILES:
            defaults=generation.resolve_generation(profile).settings
            self.assertTrue(all(defaults[key] is False for key in generation.OROGEN_GPU_PARAMETERS))
        for key in generation.OROGEN_GPU_PARAMETERS:
            token=generation.register_generation('orogen',{key:True})
            descriptor=generation.resolve_generation(token)
            self.assertTrue(descriptor.settings[key])
            self.assertTrue(descriptor.bootstrap_options[key])
            self.assertEqual(descriptor.settings['relief_pipeline'],'orogen')
            with self.assertRaises(ValueError):generation.register_generation('orogen',{key:1})
        saved=generation._defaults('orogen')
        for key in generation.OROGEN_GPU_PARAMETERS:del saved[key]
        saved['orogen_detail']=20000
        token=generation._token('orogen',saved)
        self.root.mkdir(exist_ok=True)
        (self.root/(token+'.json')).write_bytes(generation._bytes(generation._payload('orogen',saved)))
        restored=generation.resolve_generation(token)
        self.assertEqual(restored.profile,token)
        self.assertTrue(all(restored.settings[key] is False for key in generation.OROGEN_GPU_PARAMETERS))


if __name__ == "__main__":
    unittest.main()

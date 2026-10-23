"""Polar material invariants on controlled world-space water; no CUDA/server."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools._bootstrap import activate
activate()

import numpy as np
from terrain_render import smooth
from terrain_render_v2 import ice_material


class RenderV2PolarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x, y = np.meshgrid(np.linspace(100000, 112000, 112, dtype=np.float32),
                           np.linspace(-376000, -364000, 112, dtype=np.float32))
        cls.point = np.stack((x, y, np.full_like(x, 12000)), -1)
        cls.one = np.ones_like(x)

    def material(self, temperature, *, source=0., kind=1., depth=90., footprint=2., point=None):
        p = self.point if point is None else point
        one = np.ones(p.shape[:-1], np.float32)
        env = dict(temperature=one*temperature, ice=one*smooth(-1.8, -8, temperature), glacier=one*source)
        return ice_material(one*depth, one*kind, env, p, footprint, 17331)

    def test_iceberg_provenance_is_required_and_inland_water_cannot_calve(self):
        clear = self.material(-1)
        ocean = self.material(-1, source=1)
        inland = self.material(-1, source=1, kind=2)
        np.testing.assert_array_equal(clear[..., 0], 0)
        np.testing.assert_array_equal(inland[..., 0], 0)
        self.assertGreater(float(ocean[..., 0].max()), .99)
        self.assertGreater(float(ocean[..., 0].mean()), .005)
        self.assertGreater(float(ocean[..., 2].max()), .4)
        self.assertGreater(float(ocean[..., 3].max()), .2)
        self.assertTrue(np.isfinite(ocean).all())

    def test_marginal_pack_selects_opaque_plates_and_open_channels(self):
        material = self.material(-5)
        cover = material[..., 0]
        self.assertGreater(float((cover > .99).mean()), .08)
        self.assertGreater(float((cover < .01).mean()), .3)
        # Most visible plate area is opaque; selection is not a gray wash.
        self.assertGreater(float(cover[cover > .99].sum()/cover.sum()), .75)
        # Unselected parents must not leave cyan ghost networks in open water.
        np.testing.assert_array_equal(material[..., 2][cover == 0], 0)

    def test_compact_pack_is_joint_with_thin_open_leads(self):
        cover = self.material(-18)[..., 0]
        self.assertGreater(float(cover.mean()), .9)
        self.assertGreater(float((cover > .98).mean()), .85)
        self.assertGreater(float((cover < .1).mean()), .001)
        self.assertLess(float((cover < .1).mean()), .08)

    def test_compact_leads_can_refreeze_without_translucent_floes(self):
        material = self.material(-18)
        cover, light = material[..., 0], material[..., 1]
        # Refrozen cracks stay opaque with a subdued gray response, rather
        # than a second dark network around all plates.
        self.assertGreater(float(((cover > .99) & (light < .86)).mean()), .001)
        self.assertGreater(float(light.min()), .79)
        self.assertGreater(float((cover < .1).mean()), .001)
        self.assertGreater(float((cover > .98).mean()), .9)

    def test_far_pack_preserves_statistical_area_of_resolved_fragments(self):
        rng = np.random.default_rng(248)
        point = rng.uniform(-120000, 120000, (16000, 3)).astype(np.float32)
        point += np.array([100000, -370000, 12000], np.float32)
        for temperature in (-3, -4, -5, -6, -7, -8, -18):
            with self.subTest(temperature=temperature):
                near = self.material(temperature, point=point, footprint=5)[..., 0].mean()
                far = self.material(temperature, point=point, footprint=500)[..., 0].mean()
                self.assertLess(float(abs(near-far)), .035)

    def test_world_queries_are_repeatable_and_crop_independent(self):
        whole = self.material(-5, source=1)
        crop = (slice(9, 45), slice(7, 53))
        np.testing.assert_array_equal(whole[crop], self.material(-5, source=1, point=self.point[crop]))
        np.testing.assert_array_equal(whole, self.material(-5, source=1))

    def test_unresolved_pack_integrates_without_spots_or_lod_material_jump(self):
        distant = self.material(-5, footprint=500)
        self.assertLess(float(distant[..., 0].std()), 1e-6)
        np.testing.assert_array_equal(distant[..., 2:], 0)
        below = self.material(-5, footprint=449.99)
        above = self.material(-5, footprint=450.01)
        self.assertLess(float(np.abs(below-above).max()), .0001)
        near_bergs = self.material(-1, source=1, footprint=289.99)
        far_bergs = self.material(-1, source=1, footprint=290.01)
        self.assertLess(float(np.abs(near_bergs-far_bergs).max()), .0001)

    def test_coastal_shallows_support_more_marginal_pack(self):
        near = self.material(-5, depth=15)[..., 0]
        offshore = self.material(-5, depth=1800)[..., 0]
        self.assertGreater(float(near.mean()-offshore.mean()), .005)


if __name__ == '__main__':
    unittest.main()

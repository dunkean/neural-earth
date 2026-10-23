"""V2 source-lake display isolation; no routing, inference or DEM excavation."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools._bootstrap import activate
activate()

import tempfile
import unittest
from types import SimpleNamespace
import numpy as np
from terrain_hydrology_service import HydrologyService, DISPLAY_LAKES


class V2LakeDisplayTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.service = HydrologyService(lambda *_: None, cache_root=self.tmp.name)
        source = np.full((8, 8), 20., np.float32)
        source[2:6, 2:6] = 1.
        cells = np.flatnonzero(source < 5)
        ids = np.full((8, 8), -1, int)
        ids.ravel()[cells] = 0
        world = SimpleNamespace(height_m=source, bounds=(0., 0., 800., 800.),
            shape=source.shape, topology='plane', ocean=np.zeros_like(source, bool), lake_ids=ids,
            lakes=[dict(id=0, level_m=5., cells=cells, area_m2=160000., max_depth_m=4.)], reaches=[],
            _spacing=lambda: (100., 100.))
        self.entry = dict(status='ready', key='fixture', world=world,
                          source=dict(settings={'world_topology': 'plane'}), water_grid=None,
                          reach_bounds=np.empty((0, 4)))
        self.raw = np.full((80, 80), 2., np.float32)
        self.raw[34:44, 34:44] = 8.  # emerged island inside the coarse lake

    def tearDown(self):
        self.service.pool.shutdown(wait=True)
        self.tmp.cleanup()

    def render(self, *, lakes=True, minimum=0., raw=None, bounds=(0., 0., 800., 800.)):
        return self.service.physical(self.entry, self.raw if raw is None else raw, bounds,
                                     render_lakes=lakes, min_lake_area_m2=minimum)

    def test_lake_only_authority_keeps_dem_and_emerged_islands(self):
        height, water, report = self.render()
        np.testing.assert_array_equal(height, self.raw)
        np.testing.assert_array_equal(water[..., 3], 1)
        np.testing.assert_array_equal(water[34:44, 34:44, 1:3], 0)
        self.assertGreater(int(np.count_nonzero(water[..., 2] == 2)), 500)
        self.assertTrue((self.raw[water[..., 2] == 2] < 5).all())
        self.assertEqual(report['hydro_max_cut_m'], 0)
        self.assertTrue(report['hydro_render_lakes'])
        self.assertTrue(np.isfinite(water).all())

    def test_area_filter_reuses_separate_grid_without_changing_legacy(self):
        self.assertFalse(DISPLAY_LAKES)
        old_height, old_water, _ = self.render(lakes=False)
        legacy_grid = self.entry['water_grid']
        self.render()
        grid = self.entry['v2_lake_water_grid']
        self.assertIsNot(grid, legacy_grid)
        _, hidden, _ = self.render(minimum=160001.)
        np.testing.assert_array_equal(hidden[..., 1:3], 0)
        self.assertIs(self.entry['v2_lake_water_grid'], grid)
        new_height, new_water, _ = self.render(lakes=False)
        np.testing.assert_array_equal(new_height, old_height)
        np.testing.assert_array_equal(new_water, old_water)
        self.assertIs(self.entry['water_grid'], legacy_grid)

    def test_cropped_request_keeps_the_same_lake_shore(self):
        height, water, _ = self.render()
        cropped, wet, _ = self.render(raw=self.raw[:, 20:60], bounds=(200., 0., 600., 800.))
        np.testing.assert_array_equal(cropped, height[:, 20:60])
        np.testing.assert_allclose(wet, water[:, 20:60], atol=1e-6)

    def test_invalid_display_area_is_rejected(self):
        for area in (-1., np.nan, np.inf):
            with self.subTest(area=area), self.assertRaises(ValueError):
                self.render(minimum=area)


if __name__ == '__main__':
    unittest.main()

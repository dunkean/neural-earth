"""Coastal glacier provenance is global, deterministic and chart independent."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools._bootstrap import activate
activate()

from types import SimpleNamespace
import unittest
import numpy as np
from terrain_glacier import sample_glacier_proximity


def world(bounds=(0., 0., 400000., 200000.), shape=(32, 64), temperature=-12.):
    field = lambda x: np.full(shape, x, np.float32)
    return SimpleNamespace(bounds=bounds, width=shape[1], height=shape[0], height_m=field(-100),
        layers=dict(temperature_summer=field((temperature+45)/90),
                    temperature_winter=field((temperature-8+45)/90),
                    precip_summer=field(.3), precip_winter=field(.3)))


class GlacierProvenanceTests(unittest.TestCase):
    def test_cold_ocean_and_inland_cold_land_do_not_supply_icebergs(self):
        for height in (-100, 500):
            source = world()
            source.height_m.fill(height)
            actual = sample_glacier_proximity(source, [10000, 20000], [40000], {'world_topology':'plane'})
            np.testing.assert_array_equal(actual, 0.)

    def test_cold_coastal_land_supplies_a_bounded_continuous_proxy(self):
        source = world()
        source.height_m[16, 30] = 500
        settings = {'world_topology':'plane'}
        xs = np.arange(150000., 350001., 5000.)
        values = sample_glacier_proximity(source, xs, [103125.], settings)
        self.assertGreater(float(values.max()), .99)
        self.assertEqual(float(values[0, -1]), 0.)
        cropped = sample_glacier_proximity(source, xs[7:22], [103125.], settings)
        np.testing.assert_array_equal(cropped, values[:, 7:22])
        self.assertLess(float(np.max(np.abs(np.diff(values)))), .11)

    def test_warm_coasts_and_missing_climate_are_not_glacier_sources(self):
        for source in (world(temperature=20), world()):
            source.height_m[16, 30] = 500
            if source.layers['temperature_summer'].max() < .7:
                del source.layers['precip_winter']
            actual = sample_glacier_proximity(source, [190625.], [103125.], {'world_topology':'plane'})
            np.testing.assert_array_equal(actual, 0.)

    def test_antimeridian_and_rotated_polar_chart_agree(self):
        source = world(bounds=(-20e6,-10e6,20e6,10e6),shape=(64,512))
        source.height_m[32,511] = 1000
        settings = {'world_topology':'sphere'}
        x, y = -20e6+1000,156250.
        seam = sample_glacier_proximity(source,[x,x+40e6],[y],settings)
        self.assertGreater(float(seam[0,0]), .3)
        self.assertAlmostEqual(float(seam[0,0]),float(seam[0,1]),places=6)
        from terrain_polar import to_chart
        px, py = to_chart(x,y,settings)
        rotated = sample_glacier_proximity(source,[float(px)],[float(py)],settings,polar=True)
        np.testing.assert_allclose(rotated,seam[:,:1],atol=1e-6,rtol=0)


if __name__ == '__main__':
    unittest.main()

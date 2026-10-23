"""CPU composition order only; not visual proof or a fragment BRDF renderer."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools._bootstrap import activate
activate()

from types import SimpleNamespace
from unittest.mock import patch
import unittest
import numpy as np
from terrain_render_v2 import colorize_surface
from terrain_soil import NAMES


class RenderV2CompositionTests(unittest.TestCase):
    def setUp(self):
        self.n = 32
        self.xs = (np.arange(self.n)+.5)*30
        self.ys = self.xs.copy()
        self.elevation = np.full((self.n, self.n), 100, np.float32)
        constant = lambda v: np.full((16, 32), v, np.float32)
        layers = dict(temperature_summer=constant(75/90), temperature_winter=constant(70/90),
                      precip_summer=constant(.8), precip_winter=constant(.8))
        values = (.48, .36, .23, .55, .40, .29, .72, .20, .08)
        layers.update({key: constant(v) for key, v in zip(NAMES, values)})
        self.world = SimpleNamespace(bounds=(0., 0., 960., 960.), width=32, height=16,
                                     height_m=constant(0), layers=layers, seed=42, periodic_longitude=False)
        self.open = np.array([.4, .3, .2], np.float32)
        self.closed = np.array([.2, .3, .15], np.float32)
        self.sediment = np.array([.8, .7, .5], np.float32)

    def composed(self, *, footprint=30., leaf=1., snow=0., woody=.75, beach=0., flats=0., diagnostic='render-v2', lighting=None):
        shape = self.elevation.shape
        field = lambda v: np.full(shape, v, np.float32)
        environment = {key: field(0) for key in ('temperature', 'potential', 'moisture', 'exposure', 'mineral', 'sand',
                       'clay', 'organic', 'rock', 'grass', 'woody', 'height', 'needle', 'leaf', 'snow', 'ice', 'tone', 'glacier')}
        environment.update(temperature=field(24), potential=field(2), moisture=field(2), mineral=field(.5),
                           grass=field(1), woody=field(woody), height=field(20), leaf=field(leaf), snow=field(snow), tone=field(.5))

        def captured(*args, **kwargs):
            kwargs['capture'].update(open=np.broadcast_to(self.open, (*shape, 3)),
                canopy_leaf_on=np.broadcast_to(self.closed, (*shape, 3)),
                woody=field(woody), grass=field(1), rock=field(0), hot=field(24), perennial=field(0),
                beach=field(beach), flats=field(flats), sediment=np.broadcast_to(self.sediment, (*shape, 3)))

        with patch('terrain_render.surface_material', side_effect=captured), patch('terrain_render_v2.evaluate',
                   side_effect=lambda *a, **k: {key: value.copy() for key, value in environment.items()}):
            return colorize_surface(self.world, self.xs, self.ys, self.elevation, footprint,
                                    {'world_topology': 'plane'}, mode=diagnostic, lighting=lighting)

    def test_retained_material_mean_survives_all_footprints(self):
        expected = self.open*.25+self.closed*.75
        for footprint in (.1, 3., 30., 240., 7680.):
            with self.subTest(footprint=footprint):
                np.testing.assert_allclose(self.composed(footprint=footprint),
                    np.broadcast_to(expected, (*self.elevation.shape, 3)), atol=1e-6)

    def test_beach_excludes_canopy_and_keeps_sediment(self):
        rgb = self.composed(beach=.8)
        expected = self.open*.2+self.sediment*.8
        np.testing.assert_allclose(rgb, np.broadcast_to(expected, rgb.shape), atol=1e-6)
        diagnostic = self.composed(beach=.8, diagnostic='render-v2-vegetation')
        np.testing.assert_array_equal(diagnostic[..., :2], 0)

    def test_low_flats_overpaint_mean_canopy_without_excluding_estuary_trees(self):
        rgb = self.composed(flats=1.)
        expected = (self.open*.25+self.closed*.75)*.2+self.sediment*.9*.8
        np.testing.assert_allclose(rgb, np.broadcast_to(expected, rgb.shape), atol=1e-6)
        diagnostic = self.composed(flats=1., diagnostic='render-v2-vegetation')
        np.testing.assert_array_equal(diagnostic[..., 1], np.float32(.75))

    def test_snow_covers_the_beach_after_sediment(self):
        rgb = self.composed(beach=.8, flats=1., snow=1.)
        np.testing.assert_allclose(rgb, np.broadcast_to([.88, .91, .91], rgb.shape), atol=1e-6)

    def test_cpu_relief_lighting_retains_strength_and_sun_direction(self):
        y, x = np.mgrid[:self.n, :self.n]
        self.elevation = (1000+200*np.sin(x/6)*np.cos(y/8)).astype(np.float32)
        plain = self.composed()
        np.testing.assert_array_equal(plain, self.composed(lighting={'strength': 0}))
        a = self.composed(lighting={'azimuth': 315, 'exaggeration': 6})
        b = self.composed(lighting={'azimuth': 135, 'exaggeration': 6})
        self.assertGreater(float(a[..., 1].max()-a[..., 1].min()), .08)
        self.assertGreater(float(np.abs(a-b).mean()), .02)


if __name__ == '__main__':
    unittest.main()

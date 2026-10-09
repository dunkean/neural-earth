"""Biome appearance on current DEMs, slope priority and shared GPU fixtures."""
import json
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from terrain_biomes import colorize_biomes, parse_rock_slope, sample_biome_fields
from terrain_lighting import relief_intensity


def fields(code=9, settings=None):
    world = SimpleNamespace(bounds=(-1,-1,1,1),width=2,height=2,
                            layers={'koppen':np.full((2,2),code,np.uint8)})
    return sample_biome_fields(world,[-.5,.5],[-.5,.5],settings or {})


def fixture_cases():
    return [dict(name='forest-flat',height=1000,slope=0,resolution=30,threshold=40),
            dict(name='forest-steep',height=1000,slope=45,resolution=30,threshold=40),
            dict(name='snow-flat',height=6000,slope=0,resolution=30,threshold=40),
            dict(name='snow-steep',height=6000,slope=45,resolution=30,threshold=40),
            dict(name='snow-below-threshold',height=6000,slope=39,resolution=30,threshold=40),
            dict(name='snow-coarse',height=6000,slope=45,resolution=60,threshold=40),
            dict(name='snow-fine',height=6000,slope=45,resolution=15,threshold=40),
            dict(name='changed-threshold',height=6000,slope=45,resolution=30,threshold=50),
            dict(name='ocean',height=-1000,slope=0,resolution=30,threshold=40),
            dict(name='snow-flat-lit',height=6000,slope=0,resolution=30,threshold=40,lighting=True),
            dict(name='snow-shaded',height=8000,slope=30,resolution=30,threshold=40,lighting=True),
            dict(name='snow-sunlit',height=8000,slope=-30,resolution=30,threshold=40,lighting=True)]


def heights(case):
    x = (np.arange(304,dtype=np.float32)-152)*case['resolution']
    return np.broadcast_to(case['height']+x*np.float32(np.tan(np.deg2rad(case['slope']))),
                           (304,304)).copy()


class BiomeTests(unittest.TestCase):
    def test_steep_rock_overrides_every_land_class_including_ice(self):
        case=dict(height=6000,slope=45,resolution=30,threshold=40)
        for code in range(1,31):
            with self.subTest(code=code):
                actual=colorize_biomes(heights(case),fields(code),30,40)[152,152]
                np.testing.assert_allclose(actual,[.42,.38,.32],atol=1e-6)

    def test_snow_and_living_biomes_remain_below_threshold(self):
        forest=colorize_biomes(np.full((8,8),1000,np.float32),fields(),30,40)[4,4]
        snow=colorize_biomes(np.full((8,8),6000,np.float32),fields(),30,40)[4,4]
        self.assertGreater(forest[1],forest[0]*2)
        np.testing.assert_allclose(snow,[.92,.93,.96],atol=1e-6)

    def test_slope_rule_starts_at_lod_zero_and_continues_at_finer_lods(self):
        for resolution in (60,30,15,7.5,3.75):
            case=dict(height=6000,slope=45,resolution=resolution)
            actual=colorize_biomes(heights(case),fields(),resolution,40)[152,152]
            np.testing.assert_allclose(actual,[.92,.93,.96] if resolution>30 else [.42,.38,.32],atol=1e-6)

    def test_threshold_is_configurable_and_validated(self):
        case=dict(height=6000,slope=45,resolution=30)
        np.testing.assert_allclose(colorize_biomes(heights(case),fields(),30,50)[152,152],[.92,.93,.96],atol=1e-6)
        for value in (0,90,float('nan'),float('inf')):
            with self.assertRaises(ValueError):parse_rock_slope(value)

    def test_ocean_mask_uses_current_dem_and_new_land_inherits_land_color(self):
        world=SimpleNamespace(bounds=(-4,-2,4,2),width=4,height=2,
                              layers={'koppen':np.array([[0,0,9,0],[0,0,9,0]],np.uint8)})
        f=sample_biome_fields(world,[-3,-1,1,3],[-1,1],{})
        self.assertTrue(np.all(f[1]>f[0]))
        ocean=colorize_biomes(np.full((2,4),-1000,np.float32),f,30,40)
        land=colorize_biomes(np.full((2,4),1000,np.float32),f,30,40)
        self.assertTrue(np.all(ocean[...,2]>ocean[...,1]))
        self.assertTrue(np.all(land[...,1]>land[...,2]))

    def test_custom_palette_and_altitude_settings_are_preserved(self):
        f=fields(settings={'orogen_biome_color_cfb':[.8,.1,.2],
                          'orogen_biome_temperate_alpine':4,'orogen_biome_temperate_snow':6,
                          'orogen_biome_mid_darkening':0})
        actual=colorize_biomes(np.full((8,8),1000,np.float32),f,30,40)[4,4]
        np.testing.assert_allclose(actual,[.8,.1,.2],atol=1e-6)

    def test_adjacent_dem_halos_produce_identical_slope_colors(self):
        resolution=30
        yy,xx=np.mgrid[:304,:560]
        dem=(6000+500*np.sin(xx*.1)+150*np.cos(yy*.17)).astype(np.float32)
        whole=colorize_biomes(dem,fields(),resolution,40)[24:280,24:536]
        a=colorize_biomes(dem[:,:304],fields(),resolution,40)[24:280,24:280]
        b=colorize_biomes(dem[:,256:560],fields(),resolution,40)[24:280,24:280]
        np.testing.assert_array_equal(np.concatenate([a,b],axis=1),whole)

    def test_snow_retains_relief_shadows_and_lighting_controls(self):
        shadow=heights(dict(height=8000,slope=30,resolution=30))
        sun=heights(dict(height=8000,slope=-30,resolution=30))
        snowy=colorize_biomes(shadow,fields(),30,40)
        shadow_rgb=snowy[152,152]*relief_intensity(shadow,30)[152,152]
        sun_rgb=snowy[152,152]*relief_intensity(sun,30)[152,152]
        self.assertGreater(sun_rgb.mean()-shadow_rgb.mean(),.3)
        np.testing.assert_allclose(relief_intensity(shadow,30,{'strength':0}),1)
        np.testing.assert_allclose(relief_intensity(shadow,30,{'ambient':1}),1)


if __name__ == '__main__':
    if '--fixtures' in sys.argv:
        f=fields()
        cases=fixture_cases()
        for case in cases:
            dem=heights(case)
            rgb=colorize_biomes(dem,f,case['resolution'],case['threshold'])[152,152]
            if case.get('lighting'):rgb*=relief_intensity(dem,case['resolution'])[152,152]
            case['expected']=(rgb*255).tolist()
        print(json.dumps(dict(fields=f[:,0,0].tolist(),cases=cases)))
    else:
        unittest.main()

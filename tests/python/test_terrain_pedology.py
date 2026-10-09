"""Regional pedology is coarse, source-based, and controls only Render's earth."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import unittest
import numpy as np
from terrain_pedology import NAMES,TYPES,PALETTE,generate_pedology,sample_pedology,colorize_pedology
from terrain_render import surface_material,colorize_surface
from test_terrain_render import world,SOIL


class PedologyTests(unittest.TestCase):
    def test_normalized_deterministic_compositions_and_palette(self):
        atlas=world()
        a=generate_pedology(atlas.height_m,atlas.layers,42,atlas.bounds)
        b=generate_pedology(atlas.height_m,atlas.layers,42,atlas.bounds)
        for name in NAMES:
            np.testing.assert_array_equal(a[name],b[name])
            self.assertTrue(np.isfinite(a[name]).all())
            self.assertGreaterEqual(a[name].min(),0);self.assertLessEqual(a[name].max(),1)
        fractions=np.stack([a['pedology_'+n] for n in TYPES],axis=-1)
        np.testing.assert_allclose(fractions.sum(-1),1,atol=2e-7)
        np.testing.assert_allclose(np.stack([a[n] for n in NAMES[:3]],axis=-1),fractions@PALETTE,atol=1e-7)

    def test_climate_changes_regional_types_and_old_atlases_derive_once(self):
        atlas=world();a=generate_pedology(atlas.height_m,atlas.layers,42,atlas.bounds)
        hot=dict(atlas.layers,temperature_summer=np.full_like(atlas.height_m,.85),temperature_winter=np.full_like(atlas.height_m,.8),
                 precip_summer=np.full_like(atlas.height_m,1.5),precip_winter=np.full_like(atlas.height_m,1.5))
        b=generate_pedology(atlas.height_m,hot,42,atlas.bounds)
        self.assertGreater(b['pedology_ferrallitic'].mean(),a['pedology_ferrallitic'].mean()+.1)
        sample_pedology(atlas,[0],[0],{})
        cached=next(iter(atlas._terrain_pedology.values()))
        sample_pedology(atlas,[100],[100],{})
        self.assertIs(next(iter(atlas._terrain_pedology.values())),cached)

    def test_pedology_uses_raw_coast_and_ignores_dem_and_material_controls(self):
        atlas=world();atlas.height_m[:,:32]=-200
        xs=np.linspace(-19e6,19e6,64);ys=np.linspace(-9e6,9e6,32)
        a=colorize_surface(atlas,xs,ys,np.full((32,64),9000,np.float32),30,{},mode='pedology')
        b=colorize_surface(atlas,xs,ys,np.full((32,64),-1000,np.float32),7680,{},
                           {'season':0,'snow':2,'moisture':1},mode='pedology')
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a,colorize_pedology(atlas,xs,ys,{}))
        self.assertGreater(a[16,50,0],a[16,10,0])

    def test_render_earth_uses_pedology_and_soil_stays_separate(self):
        h=np.full((4,4),300,np.float32);g=np.zeros((4,4,2),np.float32);point=np.zeros((4,4,3),np.float32)
        soil=np.broadcast_to(SOIL[:,None,None],(9,4,4));seasons=np.broadcast_to(np.array([30,0,0,20,0,0],np.float32)[:,None,None],(6,4,4))
        args=(h,g,np.zeros((4,4),np.float32),point,g,30,soil,seasons,42,{'variation':0,'snow':0})
        red=np.broadcast_to(np.array([.6,.25,.15],np.float32)[:,None,None],(3,4,4))
        grey=np.broadcast_to(np.array([.45,.45,.4],np.float32)[:,None,None],(3,4,4))
        a=surface_material(*args,pedology=red);b=surface_material(*args,pedology=grey)
        self.assertGreater(np.max(np.abs(a-b)),.15)
        np.testing.assert_array_equal(surface_material(*args,bare=True,pedology=red),surface_material(*args,bare=True,pedology=grey))

    def test_regions_filter_high_frequency_source_relief(self):
        atlas=world();atlas.height_m=np.full((256,512),2500,np.float32)
        atlas.layers={name:np.full_like(atlas.height_m,value[0,0]) for name,value in atlas.layers.items()}
        a=generate_pedology(atlas.height_m,atlas.layers,42,atlas.bounds)
        yy,xx=np.indices(atlas.height_m.shape)
        b=generate_pedology(atlas.height_m+500*((xx+yy)%2*2-1),atlas.layers,42,atlas.bounds)
        self.assertLess(np.max(np.abs(a['pedology_red']-b['pedology_red'])),.003)

if __name__=='__main__':unittest.main()

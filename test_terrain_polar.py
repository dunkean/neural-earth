"""Spherical distances, physical units and independent polar NN identity."""
from copy import deepcopy
import unittest
import numpy as np
from terrain_geometry import world_bounds
from terrain_polar import PolarConditioning, chart_manifest, from_chart, sample_raster, to_chart, _directions


class PolarChartTests(unittest.TestCase):
    settings = {'world_topology':'sphere'}

    def test_rotation_roundtrip_at_seams_and_both_poles(self):
        bounds=world_bounds(self.settings)
        x=np.linspace(bounds[0],bounds[2],97)[None,:]
        y=np.linspace(bounds[1],bounds[3],49)[:,None]
        px,py=to_chart(x,y,self.settings)
        gx,gy=from_chart(px,py,self.settings)
        for a,b in zip(_directions(gx,gy,bounds),_directions(x,y,bounds)):
            np.testing.assert_allclose(a,np.broadcast_to(b,a.shape),atol=2e-15)
        np.testing.assert_allclose(to_chart(0,bounds[1],self.settings),(-10e6,0),atol=1e-8)
        np.testing.assert_allclose(to_chart(0,bounds[3],self.settings),(10e6,0),atol=1e-8)

    def test_physical_30m_cells_have_equal_spacing_at_poles(self):
        for diameter in (12732.395447,1000):
            settings=dict(self.settings,world_diameter_km=diameter)
            bounds=world_bounds(settings);radius=(bounds[2]-bounds[0])/(2*np.pi)
            for pole in (-1,1):
                x=pole*(bounds[2]-bounds[0])/4
                gx,gy=from_chart(np.array([x,x+30,x]),np.array([0,0,30]),settings)
                xyz=np.stack(_directions(gx,gy,bounds),axis=-1)
                distances=2*radius*np.arcsin(np.linalg.norm(xyz[1:]-xyz[0],axis=-1)/2)
                np.testing.assert_allclose(distances,[30,30],atol=1e-6)

    def test_sampling_converges_to_one_height_at_each_pole(self):
        fields=np.arange(64,dtype=np.float32).reshape(1,4,16)
        x=np.linspace(-20e6,20e6,51)
        for y,row in ((-10e6,0),(10e6,-1)):
            samples=sample_raster(fields,x,np.full_like(x,y),world_bounds(self.settings))
            np.testing.assert_allclose(samples,fields[0,row].mean(),atol=1e-6)

    def test_conditioning_keeps_physical_height_and_climate_units(self):
        class Source:
            def sample(self,xs,ys):
                out=np.empty((5,len(ys),len(xs)),np.float32)
                for channel,value in enumerate((900,17,700,1200,45)):out[channel]=value
                return out
        factory=PolarConditioning(Source(),self.settings,width=32,height=16)
        physical=factory.sample_raw(-2,-2,2,2)
        for channel,value in enumerate((900,17,700,1200,45)):
            np.testing.assert_allclose(physical[channel],value)
        model=factory(-2,-2,2,2).numpy()
        np.testing.assert_allclose(model[0],30)
        np.testing.assert_array_equal(model[1:],physical[1:])
        self.assertAlmostEqual(factory.latitude(-10e6,0),90)
        self.assertAlmostEqual(factory.latitude(10e6,0),-90)

    def test_chart_identity_is_distinct_and_keeps_parent_immutable(self):
        parent={'geography':{'coordinate_system':'flat-metre'},'world_hash':'original','seed_u64':'42'}
        original=deepcopy(parent);rotated=chart_manifest(parent)
        self.assertEqual(parent,original)
        self.assertNotEqual(rotated['world_hash'],parent['world_hash'])
        self.assertEqual(rotated,chart_manifest(parent))

    def test_custom_source_identity_survives_chart_wrapping(self):
        from types import SimpleNamespace
        source=SimpleNamespace(_terrain_generation_profile='orogen--gfixture',
            heightmap=SimpleNamespace(metadata={'height_sha256':'fixture'}),
            sample=lambda x,y:np.zeros((5,len(y),len(x)),np.float32))
        factory=PolarConditioning(source,self.settings,width=4,height=2)
        self.assertEqual(factory._terrain_generation_profile,source._terrain_generation_profile)
        self.assertIs(factory.heightmap,source.heightmap)
        self.assertEqual(factory.generation_settings,self.settings)


if __name__=='__main__':unittest.main()

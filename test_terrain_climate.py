"""Climate must reproduce model climate at compact grid nodes, without CUDA."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
import torch
from terrain_climate import sample_coarse_climate,colorize

sys.path.insert(0,str(Path(__file__).parent/'terrain-diffusion'))
from terrain_diffusion.inference.world_pipeline import WorldPipeline


class ClimateTests(unittest.TestCase):
    def test_sparse_negative_and_lod7_samples_are_bounded_and_terminate(self):
        queries=[]
        class Field:
            def __getitem__(self,index):
                _,ys,xs=index
                shape=(ys.stop-ys.start,xs.stop-xs.start)
                queries.append(shape)
                data=torch.zeros((7,*shape),dtype=torch.float32)
                data[0]=20;data[2]=18;data[3]=500;data[4]=1000;data[5]=30;data[-1]=1
                return data
        world=SimpleNamespace(coarse=Field())
        xs=np.arange(-638976.,-626856.+1,1212.)
        result=sample_coarse_climate(world,xs,np.array([0.]))
        self.assertEqual(result.shape,(5,1,len(xs)))
        self.assertTrue(np.isfinite(result).all())
        for tx in (-2,1,4):
            coords=tx*256*128+np.linspace(-23.5*128,279.5*128,33)
            result=sample_coarse_climate(world,coords,coords)
            self.assertEqual(result.shape,(5,33,33))
        self.assertLessEqual(max(max(shape) for shape in queries),66)
    def test_grid_nodes_match_model_bio_channels_and_lapse_rate(self):
        class Field:
            def __getitem__(self,index):
                _,ys,xs=index
                yy,xx=torch.meshgrid(torch.arange(ys.start,ys.stop),torch.arange(xs.start,xs.stop),indexing='ij')
                height=(500+xx*15+yy*20).float()
                values=torch.zeros((7,*height.shape))
                values[0]=height.clamp_min(0).sqrt()
                values[1]=values[0]
                values[2]=22-.0065*height+yy*.04
                values[3]=800+xx*2
                values[4]=1200+yy*5
                values[5]=30+xx*.03
                values[6]=1
                return values
        world=SimpleNamespace(coarse=Field())
        nodes=np.arange(33)+.5
        compact=sample_coarse_climate(world,nodes,nodes)
        compact[0]+=compact[4]*200
        exact=WorldPipeline._compute_climate(world,0,0,33,33,torch.full((33,33),200.),scale=8).numpy()
        np.testing.assert_allclose(compact,exact,rtol=2e-6,atol=2e-5)

    def test_temperature_lapse_and_dry_cold_palette(self):
        c=np.zeros((5,33,33),dtype=np.float32)
        c[0]=20;c[2]=1000;c[4]=-.0065
        low=colorize(np.zeros((10,10)),c,'temperature')
        high=colorize(np.full((10,10),4000.),c,'temperature')
        self.assertGreater(low[0,0,0],high[0,0,0])
        self.assertLess(low[0,0,2],high[0,0,2])
        wet=colorize(np.full((10,10),100.),c,'biomes')
        c[2]=0
        dry=colorize(np.full((10,10),100.),c,'biomes')
        self.assertGreater(dry[0,0,0],wet[0,0,0])
        c[0]=-25
        polar=colorize(np.full((10,10),-1000.),c,'biomes')
        np.testing.assert_allclose(polar[0,0],[.82,.93,.98],atol=1e-6)


if __name__=='__main__':unittest.main()

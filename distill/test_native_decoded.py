"""Native decoder alignment, full two-stage coverage and frozen-decoder gradients."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from distill.native_decoded import (DecodedRegion,OriginalDecoder,decoded_region_forward,
                                   physical_terms,relative_tpi_loss)
from distill.native_fusion import blend,refinement_inputs,window_outputs
from terrain_diffusion.inference.world_pipeline import gaussian_noise_patch,linear_weight_window


class SpatialOperator(torch.nn.Module):
    def __init__(self,gain=.07):
        super().__init__()
        self.gain=torch.nn.Parameter(torch.tensor(gain))

    def forward(self,inputs,noise_labels,conditional_inputs):
        return self.gain*(inputs+.1*inputs.mean((-2,-1),keepdim=True))+noise_labels[:,None,None,None]*.13


class DecoderNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=torch.nn.Conv2d(5,1,1,bias=False)
        self.conv.weight.data.fill_(.05)

    def forward(self,inputs,**kwargs):
        return self.conv(inputs)


def scene(path,y=-128,x=64):
    torch.manual_seed(18)
    teacher=SpatialOperator()
    weight=linear_weight_window(64,'cpu',torch.float32)
    initial_origins=[(y+dy,x+dx) for dy in range(-64,129,32) for dx in range(-64,129,32)]
    ref_origins=[(y+dy,x+dx) for dy in range(-32,97,32) for dx in range(-32,97,32)]
    inputs0=torch.randn(49,5,64,64)
    labels0=torch.full((49,),1.5625);labels1=torch.full((25,),.609375)
    condition=torch.zeros(74,58)
    with torch.no_grad():
        target0=teacher(inputs0,labels0,[condition[:49]])
        initial=blend(window_outputs(inputs0,target0,labels0,torch.float32),initial_origins,(y-64,x-64),256,weight)
        crops=torch.stack([initial[:,yy-y+64:yy-y+128,xx-x+64:xx-x+128] for yy,xx in ref_origins])
        noise=torch.from_numpy(np.stack([gaussian_noise_patch(12345+5820,yy,xx,64,64,
            channels=5,tile_h=64,tile_w=64) for yy,xx in ref_origins]))
        inputs1=refinement_inputs(crops,noise,labels1,torch.float32)
        target1=teacher(inputs1,labels1,[condition[49:]])
        final=blend(window_outputs(inputs1,target1,labels1,torch.float32),ref_origins,(y-32,x-32),192,weight)
        final=final[:,32:160,32:160]
        target=final[:-1]/final[-1:]
    np.savez(path,inputs=torch.cat((inputs0,inputs1)).numpy(),labels=torch.cat((labels0,labels1)).numpy(),
        condition=condition.numpy(),target=torch.cat((target0,target1)).numpy(),
        contexts=np.array([(0,yy//32,xx//32) for yy,xx in initial_origins+ref_origins]),
        initial_weighted=initial[:,32:224,32:224].numpy(),initial_origin=np.array([y-32,x-32]),
        final_latents=target.numpy(),final_mask=np.ones((1,128,128),bool),
        metadata=np.array(json.dumps(dict(seed=12345,profile='natural',split='train',y=y,x=x,native_size=128))))
    return teacher,weight


class NativeDecodedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_alignment_covers_all_three_coarse_grid_phases_and_negative_coordinates(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scene.npz'
            for y,x in ((-192,-96),(-160,32),(-128,64)):
                teacher,weight=scene(path,y,x)
                region=DecodedRegion(path)
                self.assertEqual(region.core[0]%96,0)
                self.assertEqual(region.core[1]%96,0)
                self.assertEqual(len(region.init_origins),25)
                self.assertEqual(len(region.ref_origins),9)
                output=decoded_region_forward(teacher,region,weight)
                torch.testing.assert_close(output['core'],region.target,rtol=0,atol=1e-6)

    def test_decoded_height_sends_finite_gradients_to_all34_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scene.npz'
            teacher,weight=scene(path)
            region=DecodedRegion(path)
            decoder=OriginalDecoder(model=DecoderNetwork())
            exact=decoded_region_forward(teacher,region,weight)
            terms=physical_terms(exact,region,decoder,torch.ones(5))
            self.assertEqual(float(terms['height_mae_m'].detach()),0.)
            student=SpatialOperator(.1)
            result=decoded_region_forward(student,region,weight,checkpointed=True)
            result['velocity0'].retain_grad();result['velocity1'].retain_grad()
            terms=physical_terms(result,region,decoder,torch.ones(5))
            loss=terms['height_mae_m']/100.+.5*terms['height_gradient']+.25*terms['height_tpi']
            loss.backward()
            for phase in ('velocity0','velocity1'):
                gradient=result[phase].grad
                self.assertTrue(torch.isfinite(gradient).all())
                self.assertTrue(torch.all(gradient.abs().flatten(1).sum(1)>0))
            self.assertGreater(float(student.gain.grad.abs()),0.)
            self.assertTrue(all(p.grad is None for p in decoder.parameters()))

    def test_tpi_detects_displacement_while_allowing_constant_height_bias(self):
        yy,xx=torch.meshgrid(torch.arange(384.),torch.arange(384.),indexing='ij')
        height=(500.-50.*torch.exp(-((xx-192-10.*torch.sin(yy/50))/8.)**2))[None,None]
        self.assertLess(float(relative_tpi_loss(height+20.,height)),1e-8)
        self.assertGreater(float(relative_tpi_loss(torch.roll(height,20,-1),height)),.1)

    def test_fresh_noise_retargets_both_stages_without_changing_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scene.npz'
            teacher,weight=scene(path)
            region=DecodedRegion(path)
            original=region.target.clone()
            core=region.core
            region.refresh_noise(teacher,weight,87654)
            self.assertEqual(region.core,core)
            self.assertEqual(region.noise_seed,87654)
            self.assertGreater(float((original-region.target).abs().mean()),.01)
            output=decoded_region_forward(teacher,region,weight)
            torch.testing.assert_close(output['core'],region.target,rtol=0,atol=1e-6)
            expected=torch.from_numpy(gaussian_noise_patch(87654+5819,core[0]*8,
                core[1]*8,512,512,channels=1,tile_h=512,tile_w=512))[None]
            torch.testing.assert_close(region.decoder_noise,expected,rtol=0,atol=0)


if __name__=='__main__':
    unittest.main()

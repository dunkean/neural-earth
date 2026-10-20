"""Two-noise locality, transfer coordinates and frozen one-pass equivalence."""
import unittest
from types import SimpleNamespace

import torch

from distill.native_onepass import ARCHITECTURE, OnePassBase, extend_noise_kernel, noise_windows, load_onepass
from distill.native_onepass_runtime import OnePassRuntime, install_onepass


def tiny_saved():
    config=dict(image_size=64,in_channels=10,out_channels=5,model_channels=8,model_channel_mults=[1,2],
        layers_per_block=1,emb_channels=16,noise_emb_dims=8,attn_resolutions=[],midblock_attention=False,
        conditional_inputs=[['tensor',58,1.]],block_kwargs=None)
    model=OnePassBase(config,.5)
    with torch.no_grad():model.net.out_gain.fill_(.5)
    return dict(architecture=ARCHITECTURE,solver_evaluations=1,teacher_solver_evaluations=2,
                config=dict(model.config),gain_anchor=.5,ema=model.state_dict())


class OnePassTests(unittest.TestCase):
    def test_added_noise_does_not_move_constant_channel(self):
        source=torch.arange(4*6*3*3).reshape(4,6,3,3).float()
        expanded=extend_noise_kernel(source)
        self.assertTrue(torch.equal(expanded[:,:5],source[:,:5]))
        self.assertTrue(torch.equal(expanded[:,10:11],source[:,5:6]))
        self.assertEqual(float(expanded[:,5:10].abs().sum()),0.)

    def test_both_noise_streams_are_world_local_on_both_axes(self):
        windows=noise_windows(615123,[(-32,-32),(-32,0),(0,-32)])
        self.assertTrue(torch.equal(windows[0,:,:,32:],windows[1,:,:,:32]))
        self.assertTrue(torch.equal(windows[0,:,32:,:],windows[2,:,:32,:]))
        self.assertFalse(torch.equal(windows[0,:5],windows[0,5:]))
        self.assertTrue(torch.equal(windows,noise_windows(615123,[(-32,-32),(-32,0),(0,-32)])))

    def test_frozen_runtime_matches_all_five_window_fields(self):
        torch.manual_seed(11)
        saved=tiny_saved()
        model=load_onepass(saved).requires_grad_(False)
        runtime=OnePassRuntime(saved)
        noises=noise_windows(615123,[(0,0),(0,32)])
        condition=torch.randn(2,58)
        with torch.no_grad():
            expected=model.windows(noises,condition)
            actual=runtime(noises,condition)
        torch.testing.assert_close(actual,expected,atol=1e-6,rtol=1e-6)

    def test_loader_rejects_teacher_t1_substitution(self):
        saved=tiny_saved();saved['teacher_solver_evaluations']=1
        with self.assertRaisesRegex(ValueError,'two-pass'):
            load_onepass(saved)

    def test_runtime_tile_adapter_matches_explicit_world_noise_and_fusion(self):
        from infinite_tensor import InfiniteTensor, TensorWindow, MemoryTileStore
        from terrain_diffusion.inference.world_pipeline import WorldPipeline, linear_weight_window
        from distill.native_fusion import blend
        torch.manual_seed(12)
        runtime=OnePassRuntime(tiny_saved())
        field=torch.tensor([100.,20.,15.,800.,900.,70.,1.])[:,None,None].expand(7,4,4).clone()
        coarse=InfiniteTensor(shape=(7,None,None),f=lambda ctx:field,
            output_window=TensorWindow(size=(7,4,4),stride=(7,1,1)))
        world=SimpleNamespace(seed=615123,device='cpu',kwargs={'histogram_raw':[0.]*5},coarse=coarse,
                              latents_batch_size=16,tile_store=MemoryTileStore())
        world._process_latent_conditioning=lambda *args,**kwargs:WorldPipeline._process_latent_conditioning(world,*args,**kwargs)
        install_onepass(world,runtime)
        tensor=world._build_latent_stage()
        actual=tensor[:,0:32,0:32]
        origins=[(-32,-32),(-32,0),(0,-32),(0,0)]
        condition=world._process_latent_conditioning(field[None],torch.zeros(1,5),
            torch.tensor([14.99,11.65,15.87,619.26,833.12,69.40,.66]),
            torch.tensor([21.72,21.78,10.40,452.29,738.09,34.59,.47]),torch.tensor(0.))
        with torch.no_grad():windows=runtime(noise_windows(world.seed,origins),condition.expand(4,-1))
        expected=blend(windows,origins,(-32,-32),96,linear_weight_window(64,'cpu',torch.float32))[:,32:64,32:64]
        torch.testing.assert_close(actual,expected,atol=1e-5,rtol=1e-5)
        self.assertEqual(world._distill_onepass_counts['windows'],4)


if __name__=='__main__':unittest.main()

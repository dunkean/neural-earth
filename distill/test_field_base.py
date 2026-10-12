import unittest

import torch

from distill.field_base import FieldBase,condition_at,local_attention
from terrain_diffusion.models.edm_unet import EDMUnet2D
from terrain_diffusion.models.unet_block import UNetBlock


class FieldBaseTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(112)

    def test_condition_centres_and_world_offsets(self):
        grid=torch.arange(58*3*3,dtype=torch.float32).reshape(1,58,3,3)
        values=condition_at(grid,97,97,1,[(16,16)]).reshape(1,97,97,58)
        for y in range(3):
            for x in range(3):
                torch.testing.assert_close(values[0,16+32*y,16+32*x],grid[0,:,y,x])
        shifted=condition_at(grid,65,65,1,[(-16,-16)]).reshape(1,65,65,58)
        torch.testing.assert_close(shifted[0,:65,:65],values[0,32:97,32:97])

    def test_constant64_forward_preserves_full_teacher_computation(self):
        teacher=EDMUnet2D(image_size=64,in_channels=5,out_channels=5,model_channels=8,
            model_channel_mults=[1,2],layers_per_block=1,attn_resolutions=[32],
            conditional_inputs=[('tensor',58,1.)],block_kwargs=dict(channels_per_head=8)).eval()
        with torch.no_grad():
            teacher.out_gain.fill_(1.)
            for module in teacher.modules():
                if hasattr(module,'emb_gain'):module.emb_gain.fill_(.5)
        field=FieldBase(dict(teacher.config)).eval();field.net.load_state_dict(teacher.state_dict())
        x=torch.randn(2,5,64,64);labels=torch.tensor([1.5625,.609375]);condition=torch.randn(2,58)
        a=teacher(x,noise_labels=labels,conditional_inputs=[condition])
        b=field(x,labels,condition[:,:,None,None],[(0,0),(64,-128)],[(32,32),(32,32)])
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
        b.square().mean().backward()
        self.assertGreater(float(field.net.enc['64x64_conv'].weight.grad.abs().sum()),0.)

    def test_local_attention_has_no_global_field_reduction(self):
        block=UNetBlock(16,16,0,attention=True,channels_per_head=8).eval()
        x=torch.randn(1,16,16,24)
        full=local_attention(block,x,[(0,0)],8)
        # One source window8x8 at each physical64 boundary. Nearby field size
        # and a shifted crop must not alter the same globally anchored window.
        crop=local_attention(block,x[:,:,8:16,8:24],[(64,64)],8)
        torch.testing.assert_close(crop,full[:,:,8:16,8:24],atol=1e-6,rtol=1e-5)
        expected=block.attn(x[:,:,:8,:8])
        torch.testing.assert_close(full[:,:,:8,:8],expected,atol=1e-6,rtol=1e-5)


if __name__=='__main__':unittest.main()

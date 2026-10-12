import unittest

import torch

from distill.native_fp8 import NativeFP8Base,fake_e4m3
from distill.native_fp8_training import operator_loss
from terrain_diffusion.models.edm_unet import EDMUnet2D


class NativeFP8Tests(unittest.TestCase):
    def setUp(self):torch.manual_seed(213);torch.set_num_threads(2)

    def test_quantization_clips_finitely_and_keeps_straight_through_gradients(self):
        x=torch.tensor([-1000.,-.13,0.,.13,1000.],requires_grad=True)
        y=fake_e4m3(x,torch.tensor(.01))
        self.assertTrue(y.isfinite().all());self.assertAlmostEqual(float(y[-1].detach()),4.48,places=5)
        y.sum().backward();self.assertTrue(torch.equal(x.grad,torch.ones_like(x)))
        b=x.detach().to(torch.bfloat16).requires_grad_()
        q=fake_e4m3(b,torch.tensor(.01))
        expected=((b.float()/.01).clamp(-448,448).to(torch.float8_e4m3fn).float()*.01).bfloat16()
        self.assertTrue(torch.equal(q,expected))
        q.float().sum().backward();self.assertTrue(torch.equal(b.grad,torch.ones_like(b)))

    def test_parent_computation_exact_when_quantization_is_disabled(self):
        teacher=EDMUnet2D(image_size=64,in_channels=5,out_channels=5,model_channels=8,
            model_channel_mults=[1,2],layers_per_block=1,attn_resolutions=[32],
            conditional_inputs=[('tensor',58,1.)],block_kwargs=dict(channels_per_head=8)).eval()
        with torch.no_grad():teacher.out_gain.fill_(1.)
        model=NativeFP8Base(dict(teacher.config)).eval();model.load_teacher(teacher.state_dict())
        x=torch.randn(2,5,64,64);labels=torch.tensor([1.5625,.609375]);condition=torch.randn(2,58)
        self.assertTrue(torch.equal(teacher(x,noise_labels=labels,conditional_inputs=[condition]),model(x,labels,[condition])))
        self.assertNotIn('out_conv',model.quantized_layers);self.assertNotIn('enc.64x64_conv',model.quantized_layers)
        with self.assertRaises(ValueError):model.set_quantized(True)
        for n in model.quantized_layers:model.net.get_submodule(n)._fp8_scale.fill_(.02)
        model.set_quantized(True);y=model(x,labels,[condition]);y.square().mean().backward()
        self.assertTrue(y.isfinite().all())
        self.assertGreater(float(model.net.out_conv.weight.grad.abs().sum()),0.)

    def test_first_channel0_operator_weight_is_twice_other_first_channels(self):
        inputs=torch.zeros(2,5,64,64);labels=torch.tensor([1.5625,.609375])
        batch=dict(inputs=inputs,labels=labels,mountain=torch.zeros(2,dtype=torch.bool))
        target=torch.zeros_like(inputs);variance=torch.ones(5)
        calibration=dict(first_power=torch.ones(3,5),second_power=torch.ones(3,5),second_variance=torch.ones(5))
        values=[]
        for channel in (0,1):
            predicted=target.clone();predicted[0,channel]=.1
            _,terms=operator_loss(predicted,batch,target,variance,calibration)
            values.append(float(terms['weighted_operator']))
        self.assertAlmostEqual(values[0],2*values[1],places=7)


if __name__=='__main__':unittest.main()

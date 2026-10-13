import tempfile
import unittest
from pathlib import Path
import copy

import torch

from distill.common import REVISION
from distill.native_weight_quant import (NativeWeightQuantBase, encode, fake_weight,
    pack_nibbles, unpack_nibbles, export_packed, load_packed)
from terrain_diffusion.models.edm_unet import EDMUnet2D


class NativeWeightQuantTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(314); torch.set_num_threads(2)

    def model(self):
        net=EDMUnet2D(image_size=64,in_channels=5,out_channels=5,model_channels=8,
            model_channel_mults=[1,2],layers_per_block=1,attn_resolutions=[32],
            conditional_inputs=[('tensor',58,1.)],block_kwargs=dict(channels_per_head=8)).eval()
        with torch.no_grad():net.out_gain.fill_(1.)
        model=NativeWeightQuantBase(dict(net.config),block=512).eval()
        model.net.load_state_dict(net.state_dict())
        return net,model

    def test_all_nibbles_roundtrip_and_validation(self):
        codes=torch.arange(16,dtype=torch.uint8).repeat(4)
        self.assertTrue(torch.equal(unpack_nibbles(pack_nibbles(codes)),codes))
        with self.assertRaises(ValueError):pack_nibbles(torch.tensor([17,0],dtype=torch.uint8))

    def test_quantization_handles_padding_zero_and_has_identity_ste(self):
        for scheme in ('nf4','int8'):
            w=torch.randn(3,5,3,3,requires_grad=True)
            codes,scale,actual=encode(w,scheme)
            self.assertEqual(codes.numel(),512);self.assertTrue(actual.isfinite().all())
            q=fake_weight(w,scheme);self.assertTrue(torch.equal(q,actual))
            q.sum().backward();self.assertTrue(torch.equal(w.grad,torch.ones_like(w)))
            self.assertTrue(torch.equal(encode(torch.zeros(513),scheme)[2],torch.zeros(513)))

    def test_bf16_parent_identity_without_quantization_and_real_gradients(self):
        teacher,model=self.model();teacher.bfloat16()
        x=torch.randn(2,5,64,64);t=torch.tensor([1.5625,.609375]);c=torch.randn(2,58)
        model.quantized=False
        self.assertTrue(torch.equal(teacher(x.bfloat16(),t.bfloat16(),[c.bfloat16()]).float(),model(x,t,[c])))
        model.quantized=True;y=model(x,t,[c]);y.square().mean().backward()
        self.assertGreater(float(model.net.out_conv.weight.grad.abs().sum()),0.)
        self.assertGreater(float(model.net.get_submodule(model.quantized_layers[0]).weight.grad.abs().sum()),0.)

    def test_packed_forward_matches_fake_quantization_and_rejects_corruption(self):
        _,model=self.model();x=torch.randn(2,5,64,64);t=torch.tensor([1.5625,.609375]);c=torch.randn(2,58)
        saved=dict(config=model.config,ema=model.state_dict(),scheme='nf4',block=512,
            quantized_layers=model.quantized_layers,step=1,teacher_revision=REVISION,teacher_weights_sha256='test')
        with torch.no_grad():expected=model(x,t,[c])
        with tempfile.TemporaryDirectory(prefix='native-weight-pack-') as tmp:
            p=Path(tmp);manifest=export_packed(saved,p);packed=load_packed(p)
            with torch.no_grad():actual=packed(x,t,[c])
            self.assertTrue(torch.equal(expected,actual),float((actual-expected).abs().max()))
            self.assertLess(manifest['bytes'],sum(v.numel()*4 for v in saved['ema'].values()))
            with (p/'weights.bin').open('r+b') as f:f.write(b'bad')
            with self.assertRaises(ValueError):load_packed(p)

    def test_no_grad_cache_clears_and_matches_uncached(self):
        _,model=self.model();x=torch.randn(2,5,64,64);t=torch.tensor([1.5625,.609375]);c=torch.randn(2,58)
        with torch.no_grad():
            expected=model(x,t,[c])
            with model.cached_weights():
                self.assertTrue(torch.equal(model(x,t,[c]),expected))
                self.assertTrue(torch.equal(model(x,t,[c]),expected))
        self.assertFalse(model._cache);self.assertFalse(model._cache_enabled)

    def test_ema_copy_and_bf16_snapshot_preserve_the_evaluated_forward(self):
        _,model=self.model();clone=copy.deepcopy(model)
        for name in clone.quantized_layers:
            self.assertIs(clone.net.get_submodule(name)._weight_quant_owner,clone)
        state={n:v.bfloat16() for n,v in model.state_dict().items()}
        clone.load_state_dict(state)
        x=torch.randn(2,5,64,64);t=torch.tensor([1.5625,.609375]);c=torch.randn(2,58)
        with torch.no_grad():self.assertTrue(torch.equal(model(x,t,[c]),clone(x,t,[c])))

    def test_training_loss_targets_all_channels_and_first_channel0_twice(self):
        from distill.native_weight_quant_train import loss_terms
        target=torch.zeros(2,5,64,64);batch=dict(inputs=target.clone(),labels=torch.tensor([1.5625,.609375]))
        loss,terms=loss_terms(target.clone(),target,batch,torch.ones(5))
        self.assertEqual(float(loss),0.)
        values=[]
        for channel in (0,1):
            prediction=target.clone();prediction[0,channel]=.1
            _,terms=loss_terms(prediction,target,batch,torch.ones(5))
            values.append(float(terms['weighted_operator']))
        self.assertAlmostEqual(values[0],2*values[1],places=7)


if __name__=='__main__':unittest.main()

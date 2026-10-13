import unittest

import torch

from distill.field_base import FieldBase
from distill.field_phase import FieldPhaseBase,phase_features
from distill.field_phase_train import resume_optimizer
from terrain_diffusion.models.edm_unet import EDMUnet2D


class FieldPhaseTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19);torch.set_num_threads(2)
        self.config=dict(EDMUnet2D(image_size=64,in_channels=5,out_channels=5,
            model_channels=8,model_channel_mults=[1,2],layers_per_block=1,
            attn_resolutions=[],conditional_inputs=[('tensor',58,1.)],
            block_kwargs=dict(channels_per_head=8)).config)

    def test_world_features_remain_identical_across_overlapping_fields(self):
        a=phase_features([(-64,-96)],96,96,1,'cpu',torch.float32)
        b=phase_features([(-32,-64)],64,64,1,'cpu',torch.float32)
        self.assertTrue(torch.equal(a[:,32:,32:],b))
        self.assertTrue(torch.equal(a,phase_features([(0,-32)],96,96,1,'cpu',torch.float32)))

    def test_zero_adapter_identity_and_nonzero_gradient(self):
        parent=FieldBase(self.config).eval()
        with torch.no_grad():
            parent.net.out_gain.fill_(1.)
            for module in parent.modules():
                if hasattr(module,'emb_gain'):module.emb_gain.fill_(.5)
        model=FieldPhaseBase(self.config).eval();model.load_parent(parent.state_dict())
        inputs=(torch.randn(1,5,64,64),torch.ones(1),torch.randn(1,58,3,3),[(-32,0)],[(0,32)])
        self.assertTrue(torch.equal(parent(*inputs),model(*inputs)))
        model(*inputs).square().mean().backward()
        self.assertGreater(float(model.net.field_phase_stem.grad.abs().sum()),0.)
        self.assertGreater(float(model.net.field_phase_embedding.grad.abs().sum()),0.)

    def test_adam_moments_follow_legacy_names_despite_root_adapter_insertion(self):
        parent=FieldBase(self.config)
        opt=torch.optim.Adam(parent.parameters(),lr=3e-5,betas=(.9,.99),fused=True)
        for index,p in enumerate(parent.parameters()):
            if index%2:p.grad=torch.randn_like(p)
        opt.step()
        model=FieldPhaseBase(self.config);model.load_parent(parent.state_dict())
        resumed=resume_optimizer(model,dict(optimizer=opt.state_dict()))
        for name,p in parent.named_parameters():
            q=dict(model.named_parameters())[name]
            if not opt.state.get(p):
                self.assertFalse(resumed.state.get(q));continue
            for key in ('step','exp_avg','exp_avg_sq'):
                self.assertTrue(torch.equal(opt.state[p][key],resumed.state[q][key]))
        self.assertEqual(len(resumed.param_groups[1]['params']),2)
        self.assertFalse(resumed.state.get(model.net.field_phase_stem))


if __name__=='__main__':unittest.main()

import unittest
import torch

from distill.field_phase import FieldPhaseBase
from distill.field_heads import FieldHeadsBase,combine_heads,head_weights,head_loss
from distill.field_heads_train import resume_optimizer
from distill.field_phase_train import ADAPTERS
from terrain_diffusion.models.edm_unet import EDMUnet2D


class FieldHeadsTests(unittest.TestCase):
    def setUp(self):torch.manual_seed(120);torch.set_num_threads(2)

    def test_blending_matches_native_four_window_parities_and_world_overlap(self):
        from terrain_diffusion.inference.world_pipeline import linear_weight_window
        window=linear_weight_window(64,'cpu',torch.float32)
        origins=[(-32,64)]
        weights=head_weights(origins,96,96,'cpu')
        for y,x in ((0,0),(31,63),(32,64),(95,95)):
            for py in (0,1):
                for px in (0,1):
                    self.assertEqual(weights[0,2*py+px,0,y,x],window[(y-32-32*py)%64,(x+64-32*px)%64])
        moved=head_weights([(0,96)],64,64,'cpu')
        self.assertTrue(torch.equal(moved,weights[:,:,:,32:,32:]))
        values=torch.randn(1,1,5,96,96).expand(-1,4,-1,-1,-1)
        torch.testing.assert_close(combine_heads(values,origins),values[:,0],rtol=1e-6,atol=1e-6)

    def test_uncovered_targets_cannot_affect_head_supervision(self):
        target=torch.zeros(4,5,256,256);mask=torch.zeros(4,1,256,256);mask[:,:,32:224,32:224]=1
        prediction=torch.zeros(1,4,5,256,256,requires_grad=True)
        with torch.no_grad():prediction[:,:,:,:32]=100
        self.assertEqual(float(head_loss(prediction,target,mask,torch.ones(5))[0].detach()),0.)
        loss,_=head_loss(prediction+1,target,mask,torch.ones(5));loss.backward()
        self.assertTrue((prediction.grad[:,:,:,32:224,32:224].abs().sum((0,2,3,4))>0).all())
        self.assertEqual(float(prediction.grad[:,:,:,:32].abs().sum()),0.)

    def test_copied_heads_preserve_parent_velocity_and_existing_state(self):
        config=dict(EDMUnet2D(image_size=64,in_channels=5,out_channels=5,model_channels=8,
            model_channel_mults=[1,2],layers_per_block=1,attn_resolutions=[],
            conditional_inputs=[('tensor',58,1.)],block_kwargs=dict(channels_per_head=8)).config)
        parent=FieldPhaseBase(config).eval()
        with torch.no_grad():parent.net.out_gain.fill_(1.)
        model=FieldHeadsBase(config).eval();model.load_parent(parent.state_dict())
        args=(torch.randn(1,5,64,64),torch.ones(1)*1.5625,torch.randn(1,58,3,3),[(-32,0)],[(32,32)])
        expected=parent(*args)
        original,_,heads=model(*args,return_head_velocities=True)
        self.assertTrue(torch.equal(original,expected))
        for i in range(4):self.assertTrue(torch.equal(heads[:,i],expected))
        for name,value in parent.state_dict().items():self.assertTrue(torch.equal(model.state_dict()[name],value))

    def test_two_inherited_adam_groups_follow_names_and_new_heads_start_fresh(self):
        config=dict(EDMUnet2D(image_size=64,in_channels=5,out_channels=5,model_channels=8,
            model_channel_mults=[1,2],layers_per_block=1,attn_resolutions=[],
            conditional_inputs=[('tensor',58,1.)],block_kwargs=dict(channels_per_head=8)).config)
        parent=FieldPhaseBase(config)
        groups=[[p for n,p in parent.named_parameters() if n not in ADAPTERS],
                [p for n,p in parent.named_parameters() if n in ADAPTERS]]
        optimizer=torch.optim.Adam([dict(params=groups[0],lr=3e-5),dict(params=groups[1],lr=1e-4)],fused=True)
        for i,p in enumerate(parent.parameters()):
            if i%2:p.grad=torch.randn_like(p)
        optimizer.step()
        model=FieldHeadsBase(config);model.load_parent(parent.state_dict())
        resumed=resume_optimizer(model,optimizer.state_dict())
        for name,p in parent.named_parameters():
            q=dict(model.named_parameters())[name]
            if not optimizer.state.get(p):self.assertFalse(resumed.state.get(q));continue
            for key in ('step','exp_avg','exp_avg_sq'):
                self.assertTrue(torch.equal(optimizer.state[p][key],resumed.state[q][key]))
        self.assertEqual(len(resumed.param_groups[2]['params']),5)
        self.assertTrue(all(not resumed.state.get(p) for p in resumed.param_groups[2]['params']))


if __name__=='__main__':unittest.main()

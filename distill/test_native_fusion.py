"""Native signs, weighted dependencies, student-state refinement and both seams."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from distill.native_fusion import (Region, audit_region, blend, overlap_loss,
    refinement_inputs, region_forward, region_loss, window_outputs)
from terrain_diffusion.inference.world_pipeline import gaussian_noise_patch, linear_weight_window


class Operator(torch.nn.Module):
    def __init__(self,gain=.07):
        super().__init__()
        self.gain=torch.nn.Parameter(torch.tensor(gain))

    def forward(self,inputs,noise_labels,conditional_inputs):
        return self.gain*inputs+noise_labels[:,None,None,None]*.13


class NativeFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def make_scene(self,path):
        y,x=-128,64
        init_origins=[(y+dy,x+dx) for dy in (0,32,64,96) for dx in (0,32,64,96)]
        ref_origins=[(y+dy,x+dx) for dy in (32,64) for dx in (32,64)]
        torch.manual_seed(18)
        inputs0=torch.randn(16,5,64,64)
        labels0=torch.full((16,),1.5625)
        labels1=torch.full((4,),.609375)
        condition=torch.zeros(20,58)
        teacher=Operator()
        weight=linear_weight_window(64,'cpu',torch.float32)
        with torch.no_grad():
            target0=teacher(inputs0,labels0,[condition[:16]])
            initial=blend(window_outputs(inputs0,target0,labels0,torch.float32),init_origins,(y,x),160,weight)
            crops=torch.stack([initial[:,yy-y:yy-y+64,xx-x:xx-x+64] for yy,xx in ref_origins])
            noise=torch.from_numpy(np.stack([gaussian_noise_patch(12345+5820,yy,xx,64,64,
                channels=5,tile_h=64,tile_w=64) for yy,xx in ref_origins]))
            inputs1=refinement_inputs(crops,noise,labels1,torch.float32)
            target1=teacher(inputs1,labels1,[condition[16:]])
            final=blend(window_outputs(inputs1,target1,labels1,torch.float32),ref_origins,(y+32,x+32),96,weight)
            core=final[:-1,32:64,32:64]/final[-1:,32:64,32:64]
            field=torch.zeros(5,128,128)
            field[:,64:96,64:96]=core
        arrays=dict(inputs=torch.cat((inputs0,inputs1)).numpy(),labels=torch.cat((labels0,labels1)).numpy(),
            condition=condition.numpy(),target=torch.cat((target0,target1)).numpy(),
            contexts=np.array([(0,yy//32,xx//32) for yy,xx in init_origins+ref_origins]),
            final_latents=field.numpy(),final_mask=np.ones((1,128,128),bool),
            initial_weighted=initial.numpy(),initial_origin=np.array([y,x]),
            metadata=np.array(json.dumps(dict(seed=12345,split='train',y=y,x=x,native_size=128))))
        np.savez(path,**arrays)
        return teacher,weight

    def test_velocity_sign_and_bf16_expression_order(self):
        inputs=torch.tensor([[[[.7]]]])
        velocity=torch.tensor([[[[.4]]]])
        labels=torch.tensor([.609375])
        for dtype in (torch.float32,torch.bfloat16):
            x=(inputs.to(dtype)*.5)
            t=labels.to(dtype)[:,None,None,None]
            expected=(torch.cos(t)*x-torch.sin(t)*.5*(-velocity.to(dtype))).float()/.5
            torch.testing.assert_close(window_outputs(inputs,velocity,labels,dtype),expected,rtol=0,atol=0)
        # Positive network velocity increases the next sample in the native solver.
        self.assertGreater(float(window_outputs(inputs,velocity,labels,torch.float32)),
                           float(window_outputs(inputs,torch.zeros_like(velocity),labels,torch.float32)))

    def test_scene_reconstruction_and_gradients_through_student_blending(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scene.npz'
            teacher,weight=self.make_scene(path)
            region=Region(path)
            audit=audit_region(region,weight)
            self.assertEqual(audit['original_input_max_abs'],0.)
            self.assertEqual(audit['fused_input_max_abs'],0.)
            self.assertEqual(audit['final_core_max_abs'],0.)
            result=region_forward(teacher,region,weight)
            loss,terms=region_loss(result,region,teacher)
            self.assertEqual(float(loss.detach()),0.)
            student=Operator(.1)
            result=region_forward(student,region,weight,checkpointed=True)
            self.assertGreater(float((result['inputs1']-region.stages[1]['inputs']).abs().max().detach()),.001)
            loss,terms=region_loss(result,region,teacher)
            loss.backward()
            self.assertTrue(torch.isfinite(student.gain.grad))
            self.assertGreater(float(student.gain.grad.abs()),.001)

    def test_overlap_checks_horizontal_and_vertical_teacher_differences(self):
        origins=[(0,0),(0,32),(32,0),(32,32)]
        target=torch.stack([torch.full((5,64,64),float(i)) for i in range(4)])
        # The teacher's nonzero disagreement itself must incur no penalty.
        self.assertEqual(float(overlap_loss(target,target,origins)),0.)
        prediction=target.clone()
        prediction[1]+=1.
        self.assertGreater(float(overlap_loss(prediction,target,origins)),0.)
        prediction=target.clone()
        prediction[2]+=1.
        self.assertGreater(float(overlap_loss(prediction,target,origins)),0.)
        with self.assertRaisesRegex(ValueError,'coverage'):
            refinement_inputs(torch.zeros(1,6,64,64),torch.zeros(1,5,64,64),torch.ones(1))

    def test_overlap_target_uses_the_same_student_state(self):
        class ChangedInitialization(Operator):
            def forward(self,inputs,noise_labels,conditional_inputs):
                value=super().forward(inputs,noise_labels,conditional_inputs)
                return value+(noise_labels>1.)[:,None,None,None]*.2
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scene.npz'
            teacher,weight=self.make_scene(path)
            region=Region(path)
            result=region_forward(ChangedInitialization(),region,weight)
            loss,terms=region_loss(result,region,teacher)
            self.assertGreater(float(loss.detach()),0.)
            self.assertEqual(float(terms['refinement_operator'].detach()),0.)
            self.assertEqual(float(terms['overlap'].detach()),0.)

    def test_fresh_noise_retargets_both_teacher_stages_deterministically(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scene.npz'
            teacher,weight=self.make_scene(path)
            a,b=Region(path),Region(path)
            original=a.target.clone()
            a.refresh_noise(teacher,weight,1_000_001)
            b.refresh_noise(teacher,weight,1_000_001)
            torch.testing.assert_close(a.target,b.target,rtol=0,atol=0)
            self.assertGreater(float((a.target-original).abs().mean()),.01)
            prediction=region_forward(teacher,a,weight)
            loss,terms=region_loss(prediction,a,teacher)
            self.assertEqual(float(loss.detach()),0.)


if __name__ == '__main__':
    unittest.main()

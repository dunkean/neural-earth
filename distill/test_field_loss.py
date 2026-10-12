import unittest
from types import SimpleNamespace

import numpy as np
import torch

from distill.field_loss import field_loss,shared_crop_loss
from distill.field_train import select
from distill.native_mixed_bank import KINDS,VARIANTS
from distill.wet_mountains import KIND


class FieldLossTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_targets_identity_and_every_channel_gradient(self):
        torch.manual_seed(125)
        scene=SimpleNamespace(initial=torch.randn(5,192,192),target=torch.randn(5,128,128))
        variance=torch.ones(5);cal=dict(first_power=torch.ones(3,5),second_power=torch.ones(3,5))
        zero,_=field_loss(dict(initial192=scene.initial[None],core=scene.target[None]),scene,variance,cal)
        self.assertEqual(float(zero),0.)
        initial=(scene.initial[None]+.1).requires_grad_();final=(scene.target[None]+.1).requires_grad_()
        loss,_=field_loss(dict(initial192=initial,core=final),scene,variance,cal,mountain=True)
        loss.backward()
        self.assertTrue((initial.grad.abs().sum((0,2,3))>0).all())
        self.assertTrue((final.grad.abs().sum((0,2,3))>0).all())

    def test_shared_world_overlap_checks_both_axes(self):
        field=torch.randn(1,5,160,160)
        a=field[:,:,:128,:128]
        for axis,b in (('x',field[:,:,:128,32:160]),('y',field[:,:,32:160,:128])):
            self.assertEqual(float(shared_crop_loss(a,b,axis,torch.ones(5))),0.)
            self.assertGreater(float(shared_crop_loss(a,b+.1,axis,torch.ones(5))),0.)

    def test_sampling_has_exact_full_cycle_category_and_branch_balance(self):
        records=[dict(case_kind=k,coarse_variant=v,metadata=dict(profile='natural'))
                 for k in (None,KIND,*KINDS) for v in VARIANTS]
        counts={k:0 for k in (None,KIND,*KINDS)};branches={v:0 for v in VARIANTS}
        for i in range(240):
            row=select(records,np.random.default_rng(i),i)
            counts[row['case_kind']]+=1;branches[row['coarse_variant']]+=1
        self.assertEqual(counts[None],144);self.assertEqual(counts[KIND],60)
        self.assertTrue(all(counts[k]==9 for k in KINDS))
        self.assertEqual(branches,{v:80 for v in VARIANTS})


if __name__=='__main__':unittest.main()

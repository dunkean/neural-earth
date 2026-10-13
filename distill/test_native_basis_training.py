import unittest

import numpy as np
import torch

from distill.native_basis_training import choose, objective
from distill.native_lowrank import FactorMPConv
from distill.native_lowrank_inference import FrozenFactor
from distill.native_mixed_bank import KINDS, VARIANTS
from distill.wet_mountains import KIND


class BasisTrainingTests(unittest.TestCase):
    def test_sampling_balances_phases_branches_and_independent_wet_mountains(self):
        records=[]
        for c,kind in enumerate((None,*KINDS,KIND)):
            for seed in range(1000+c*100,1020+c*100):
                for variant in VARIANTS:
                    records.append(dict(metadata=dict(seed=seed,split='train',profile='natural'),
                                        case_kind=kind,coarse_variant=variant))
        counts={None:0,KIND:0,'old':0}
        for step in range(50):
            selected=choose(records,np.random.default_rng(step),step)
            self.assertEqual(len({r['metadata']['seed'] for _,r in selected}),8)
            for phase in (0,1):
                rows=[r for p,r in selected if p==phase]
                self.assertEqual({r['coarse_variant'] for r in rows},set(VARIANTS))
                self.assertEqual(sum(r['case_kind']==KIND for r in rows),1)
            for _,r in selected:
                kind=r['case_kind']; counts[kind if kind in (None,KIND) else 'old']+=1
        self.assertEqual(counts,{None:240,KIND:100,'old':60})

    def test_frozen_factor_preserves_bf16_rounding_exactly(self):
        factor=FactorMPConv(9,12,(3,3),7).eval()
        with torch.no_grad():
            factor.spatial.normal_(); factor.pointwise.normal_()
            x=torch.randn(2,9,32,32).bfloat16()
            torch.testing.assert_close(FrozenFactor(factor)(x),factor(x),rtol=0,atol=0)

    def test_all_first_phase_feature_channels_receive_gradients(self):
        target=torch.zeros(4,5,32,32)
        prediction=target.clone(); prediction[:2,3]=.1; prediction.requires_grad_()
        labels=torch.tensor([1.5625,1.5625,.609375,.609375])
        cal=dict(first_power=torch.ones(3,5),second_power=torch.ones(3,5),second_variance=torch.ones(5))
        loss,terms=objective(prediction,target,labels,target,torch.ones(5),cal,torch.zeros(4,dtype=torch.bool))
        self.assertGreater(float(terms['first_x0'].detach()),0.)
        loss.backward(); self.assertGreater(float(prediction.grad[:2,3].abs().sum()),0.)

    def test_identity_and_mountain_energy_weight(self):
        torch.manual_seed(17)
        target=torch.randn(4,5,32,32)
        labels=torch.tensor([1.5625,1.5625,.609375,.609375])
        cal=dict(first_power=torch.ones(3,5),second_power=torch.ones(3,5),second_variance=torch.ones(5))
        zeros=torch.zeros(4,dtype=torch.bool)
        loss,_=objective(target,target,labels,target,torch.ones(5),cal,zeros)
        self.assertEqual(float(loss),0.)
        _,a=objective(target*1.2,target,labels,target,torch.ones(5),cal,zeros)
        _,b=objective(target*1.2,target,labels,target,torch.ones(5),cal,~zeros)
        for phase in ('first','second'):
            torch.testing.assert_close(b[phase+'_band_energy'],2*a[phase+'_band_energy'])
            torch.testing.assert_close(b[phase+'_band_position'],a[phase+'_band_position'])


if __name__=='__main__': unittest.main()

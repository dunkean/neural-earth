import unittest

import torch

from distill.student import StudentConfig
from distill.tile40_flow_network import TimeFlow,widen_time_flow


class TimeFlowTests(unittest.TestCase):
    def test_widening_preserves_a_nonzero_learned_map_and_breaks_weight_symmetry(self):
        torch.manual_seed(817)
        source=TimeFlow(StudentConfig(stage='base',width=16,depth=4)).eval()
        with torch.no_grad():
            source.head.weight.normal_(std=.03)
            for module in source.modules():
                if hasattr(module,'modulation'):module.modulation.weight.normal_(std=.01)
        target=TimeFlow(StudentConfig(stage='base',width=32,depth=4)).eval()
        widen_time_flow(source,target)
        fields=torch.randn(1,32,32,32);fields[:,5].fill_(.7)
        with torch.no_grad():
            torch.testing.assert_close(target(fields),source(fields),atol=2e-5,rtol=2e-5)
        weight=target.down[0].weight
        self.assertGreater(float((weight[:,0::2]-weight[:,1::2]).detach().abs().sum()),0.)

    def test_zero_head_is_the_exact_gaussian_bridge_velocity(self):
        torch.manual_seed(2901)
        model = TimeFlow(StudentConfig(stage='base', width=16, depth=4)).eval()
        fields = torch.randn(2, 32, 32, 32)
        fields[0, 5].fill_(.2)
        fields[1, 5].fill_(.8)
        actual = model(fields)
        time = fields[:, 5:6]
        expected = (2*time-1)/((1-time).square()+time.square())*fields[:, :5]
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        self.assertTrue(torch.isfinite(model.head.weight.grad).all())
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0.)

    def test_time_modulation_changes_output_for_a_fixed_state(self):
        model = TimeFlow(StudentConfig(stage='base', width=16, depth=4)).eval()
        with torch.no_grad():
            model.head.weight.normal_(std=.02)
            for module in model.modules():
                if hasattr(module, 'modulation'):
                    module.modulation.weight.normal_(std=.03)
        fields = torch.randn(1, 32, 32, 32)
        fields[:, :5].zero_()
        fields[:, 5].fill_(.1)
        a = model(fields)
        fields[:, 5].fill_(.9)
        b = model(fields)
        self.assertGreater(float((a-b).detach().abs().mean()), .001)


if __name__ == '__main__':
    unittest.main()

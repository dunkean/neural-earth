"""CPU/gradient fallbacks preserve training and eager decoder semantics."""
import unittest

import torch

from distill.inference import student_graph_forward


class GraphFallbackTests(unittest.TestCase):
    def test_cpu_decoder_keeps_gradients(self):
        model=torch.nn.Conv2d(2,1,1).eval()
        value=torch.ones(1,2,4,4,requires_grad=True)
        actual=student_graph_forward(model,value,stage='decoder')
        torch.testing.assert_close(actual,model(value))
        actual.sum().backward()
        self.assertIsNotNone(model.weight.grad)
        self.assertIsNotNone(value.grad)
        self.assertNotIn('_distill_decoder_graph',model.__dict__)

    def test_unknown_stage_is_rejected(self):
        with self.assertRaises(ValueError):
            student_graph_forward(torch.nn.Identity(),torch.zeros(1),stage='base')


if __name__=='__main__':unittest.main()

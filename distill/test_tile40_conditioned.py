import unittest
from dataclasses import asdict
from pathlib import Path
import tempfile
import torch
from distill.student import Student,StudentConfig,load_student
from distill.tile40_conditioned import ConditionedStudent,transfer,ARCHITECTURE


class ConditionedTests(unittest.TestCase):
    def test_learned_modulation_keeps_whole_tile_and_window_equivalence(self):
        torch.manual_seed(921)
        model=ConditionedStudent(StudentConfig(stage='base',width=16,depth=2,dilations=(1,2))).eval()
        with torch.no_grad():
            model.head.weight.normal_(std=.03)
            for modules in (model.enc_mod,model.mid_mod,model.dec_mod):
                for mod in modules:mod.affine.weight.normal_(std=.01)
        whole=torch.randn(1,32,384,384)
        window=whole[...,96:256,96:256]
        with torch.no_grad():
            actual=model(window)[...,48:112,48:112]
            expected=model(whole)[...,144:208,144:208]
        torch.testing.assert_close(actual,expected,atol=2e-5,rtol=2e-5)

    def test_zero_modulation_preserves_backbone_and_receives_training_gradients(self):
        torch.manual_seed(345)
        source=Student(StudentConfig(stage='base',width=16,depth=2,dilations=(1,2))).eval()
        with torch.no_grad():source.head.weight.normal_(std=.04)
        target=ConditionedStudent(source.config).eval();transfer(source,target)
        inputs=torch.randn(1,32,64,64)
        torch.testing.assert_close(source(inputs),target(inputs),atol=0.,rtol=0.)
        target(inputs).square().mean().backward()
        self.assertTrue(all(torch.isfinite(mod.affine.weight.grad).all() for mod in target.enc_mod))
        self.assertGreater(float(target.enc_mod[0].affine.weight.grad.abs().sum()),0.)
        with tempfile.TemporaryDirectory() as folder:
            file=Path(folder)/'model.pt'
            torch.save(dict(config=asdict(target.config),model=target.state_dict(),conditioning_architecture=ARCHITECTURE),file)
            loaded,_=load_student(file)
            torch.testing.assert_close(loaded(inputs),target(inputs),atol=0.,rtol=0.)


if __name__=='__main__':unittest.main()

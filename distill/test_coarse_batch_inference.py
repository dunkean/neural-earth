"""Multi-window coarse inference preserves ordering, RNG positions and fusion."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from distill.inference import install
from distill.student import StudentConfig


class TinyCoarse(torch.nn.Module):
    def __init__(self):
        super().__init__();self.config=StudentConfig(stage='coarse')
    def forward(self,features):return features[:,:6]*.5


class CoarseBatchTests(unittest.TestCase):
    def test_batch_matches_scalar_windows_and_uses_one_forward(self):
        model=TinyCoarse().eval()
        def world():
            return SimpleNamespace(device='cpu',seed=617531,kwargs={'coarse_pooling':1,'cond_snr':[3.]*5},
                _build_coarse_stage=lambda:SimpleNamespace())
        def prepare(world,ctxs,*args,**kwargs):
            _,y,x=ctxs[0]
            return SimpleNamespace(conditions=[torch.full((1,5,64,64),float(y+x)),
                                               *[torch.tensor([float(i)]) for i in range(5)]])
        with patch('distill.teacher.physical_coarse_scaling',return_value=(np.zeros(6),np.ones(6))),\
             patch('terrain_inference._coarse_batch',side_effect=prepare):
            scalar=world();batch=world()
            install(scalar,{'coarse':model})
            install(batch,{'coarse':model},batch_coarse=True)
            contexts=[(0,-1,0),(0,0,-1),(0,0,0),(0,1,1)]
            expected=scalar._build_coarse_stage()._f(contexts)
            actual=batch._build_coarse_stage()._f(contexts)
        for a,b in zip(actual,expected):self.assertTrue(torch.equal(a,b))
        self.assertEqual(scalar._distill_counts['coarse']['calls'],4)
        self.assertEqual(batch._distill_counts['coarse']['calls'],1)
        self.assertEqual(batch._distill_counts['coarse']['output_pixels'],4*64**2)


if __name__=='__main__':unittest.main()

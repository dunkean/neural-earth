"""Device-level rounding gate for unit-convolution replacement (no weights)."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import os
from pathlib import Path
import sys
import unittest
import weakref
from unittest.mock import patch
import torch
sys.path.insert(0,str(_REPO_ROOT/'terrain-diffusion'))
import terrain_nn_constants as constants


class ExactResamplingTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('TERRAIN_TEST_CUDA')=='1' and torch.cuda.is_available(),'Dedicated CUDA validation')
    def test_prewarm_retention_and_dynamic_input_replay(self):
        from terrain_cuda_graphs import CudaGraphModel,prewarm_base_forms
        class Base(torch.nn.Module):
            config={'in_channels':5,'conditional_inputs':[['tensor',58,1.]]}
            def __init__(self):
                super().__init__()
                with torch.inference_mode(False):self.weight=torch.nn.Parameter(torch.ones((),device='cuda',dtype=torch.bfloat16))
            def forward(self,x,noise_labels,conditional_inputs,**unused):
                return x*self.weight+conditional_inputs[0].sum(1).view(-1,1,1,1)+noise_labels.view(-1,1,1,1)
        with torch.inference_mode():
            for capacity in (4,16):
                wrapped=CudaGraphModel(Base(),max_buckets=capacity,max_batch=16)
                result=prewarm_base_forms(wrapped,16)
                self.assertEqual(result['retained_batches'],list(range(17-capacity,17)))
                self.assertEqual(result['fully_warmed'],capacity==16)
                self.assertEqual(result['forms'],capacity)
                for n in result['retained_batches']:
                    x=torch.randn((n,5,64,64),device='cuda',dtype=torch.bfloat16)
                    labels=torch.randn(n,device='cuda',dtype=torch.bfloat16)
                    cond=[torch.randn((n,58),device='cuda',dtype=torch.bfloat16)]
                    expected=wrapped.model(x,noise_labels=labels,conditional_inputs=cond)
                    actual=wrapped(x,noise_labels=labels,conditional_inputs=cond)
                    self.assertTrue(torch.equal(expected.view(torch.uint8),actual.view(torch.uint8)))
                before=wrapped.stats()['capture_calls']
                self.assertEqual(before,16)
                wrapped._clear_buckets()
            disabled=CudaGraphModel(Base(),max_bytes=0)
            with patch.object(disabled,'forward',side_effect=AssertionError('Disabled prewarm ran model')):
                self.assertFalse(prewarm_base_forms(disabled,16)['enabled'])
            wrong_signature=CudaGraphModel(Base())
            x=torch.zeros((1,5,64,64),device='cuda',dtype=torch.bfloat16)
            labels=torch.ones(1,device='cuda',dtype=torch.bfloat16)
            wrong_signature(x,labels,[torch.zeros((1,57),device='cuda',dtype=torch.bfloat16)])
            with patch.object(wrong_signature,'forward',return_value=x):
                result=prewarm_base_forms(wrong_signature,1)
            self.assertEqual(result['retained_batches'],[])
            self.assertFalse(result['fully_warmed'])
            wrong_signature._clear_buckets()

    @unittest.skipUnless(os.environ.get('TERRAIN_TEST_CUDA')=='1' and torch.cuda.is_available(),'Dedicated CUDA validation')
    def test_binary_sum_preserves_rounding_and_zero_sign(self):
        with torch.inference_mode():
            for dtype in (torch.bfloat16,torch.float16,torch.float32):
                for shape in ((1,320,64,64),(16,8,8,8)):
                    a=torch.randn(shape,device='cuda',dtype=dtype);b=torch.randn_like(a)
                    a.flatten()[:4]=torch.tensor([0.,-0.,0.,-0.],device='cuda',dtype=dtype)
                    b.flatten()[:4]=torch.tensor([0.,0.,-0.,-0.],device='cuda',dtype=dtype)
                    for weights in (None,.3,.5,[.1,.9],[1.,-1.]):
                        expected=constants._original_sum([a,b],weights)
                        actual=constants._sum([a,b],weights)
                        self.assertTrue(torch.equal(expected.view(torch.uint8),actual.view(torch.uint8)),(dtype,shape,weights))

    @unittest.skipUnless(os.environ.get('TERRAIN_TEST_CUDA')=='1' and torch.cuda.is_available(),'Dedicated CUDA validation')
    def test_small_budget_and_capture_failure_release_buffers_before_eager_fallback(self):
        from terrain_cuda_graphs import CudaGraphModel
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                with torch.inference_mode(False):self.weight=torch.nn.Parameter(torch.ones((),device='cuda'))
                self.inputs=[]
            def forward(self,x,**unused):
                self.inputs.append(weakref.ref(x))
                if len(self.inputs)==4:
                    assert self.inputs[0]() is None,'Capture input survived into eager fallback'
                return x*self.weight
        with torch.inference_mode():
            x=torch.randn((1,1,8,8),device='cuda');labels=torch.zeros(1,device='cuda')
            model=Model();limited=CudaGraphModel(model,max_bytes=0)
            self.assertTrue(torch.equal(limited(x,labels,[]),x))
            self.assertEqual(limited.stats()['memory_fallbacks'],1)
            model=Model();wrapped=CudaGraphModel(model)
            with patch('torch.cuda.graph',side_effect=RuntimeError('injected capture OOM')):
                self.assertTrue(torch.equal(wrapped(x,labels,[]),x))
            self.assertEqual(len(model.inputs),4)
            self.assertEqual(wrapped.stats()['buckets'],0)
            self.assertIn('injected capture OOM',wrapped.stats()['capture_errors'][0])
    @unittest.skipUnless(os.environ.get('TERRAIN_TEST_CUDA')=='1' and torch.cuda.is_available(),
                         'Dedicated CUDA validation; opt in with TERRAIN_TEST_CUDA=1')
    def test_slice_and_repeat_preserve_finite_values_and_signed_zero(self):
        with torch.inference_mode():
            for dtype in (torch.bfloat16,torch.float16,torch.float32):
                value=torch.randn((2,3,17,19),device='cuda',dtype=dtype)
                value[0,0,:2,:2]=torch.tensor([[0.,-0.],[-0.,0.]],device='cuda',dtype=dtype)
                for factor in (1,2,3):
                    for mode in ('down','up','up_bilinear','keep'):
                        expected=constants._original_resample(value,mode,factor)
                        actual=constants._resample(value,mode,factor)
                        self.assertTrue(torch.equal(expected.view(torch.uint8),actual.view(torch.uint8)),(dtype,mode,factor))


if __name__=='__main__':unittest.main()

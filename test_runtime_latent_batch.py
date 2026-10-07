"""CPU numerical gate for the real batched latent preparation helper.

The reference retains the pre-change per-window GPU-path expressions. The
base network is stubbed so input/conditioning precision and weighted outputs
can be checked directly, including non-finite coarse values and zero weights.
No CUDA runtime call or checkpoint is needed.
"""
from pathlib import Path
import ast
import inspect
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).parent/'terrain-diffusion'))
from terrain_diffusion.inference.world_pipeline import WorldPipeline, gaussian_noise_patch, linear_weight_window
from terrain_diffusion.models import mp_layers
from terrain_inference import _latent_gpu_batch
from terrain_nn_constants import _concat, _original_concat
import terrain_inference


def assert_tensor_bytes(test, actual, expected):
    test.assertEqual(actual.shape,expected.shape)
    test.assertEqual(actual.dtype,expected.dtype)
    test.assertEqual(actual.device.type,'cpu')
    test.assertEqual(expected.device.type,'cpu')
    # A length-one expanded label is considered contiguous despite stride 0;
    # explicit copies provide stride 1 before changing the element size.
    actual_bytes=torch.empty(actual.shape,dtype=actual.dtype).copy_(actual).view(torch.uint8)
    expected_bytes=torch.empty(expected.shape,dtype=expected.dtype).copy_(expected).view(torch.uint8)
    test.assertTrue(torch.equal(actual_bytes,expected_bytes))


def prechange_conditioning(value,histogram,means,stds):
    # Frozen cuda-resident-window-v2 path: process each batch-one window before
    # concatenating the actual base batch. Keep this independent of new code.
    cond=value[:-1]/value[-1:]
    cond=torch.cat([cond,torch.ones((1,4,4),device=cond.device)],dim=0)[None]
    cond=(cond-means.to(cond.device).view(1,-1,1,1))/stds.to(cond.device).view(1,-1,1,1)
    cond=cond.nan_to_num(float(means[0]))
    result=_original_concat([
        cond[:,0:1].flatten(1),cond[:,1:2].flatten(1),
        cond[:,2:6,1:3,1:3].mean(dim=(2,3)),cond[:,6:7].flatten(1),
        histogram.to(cond.device),
        torch.tensor([-0.5*np.sqrt(12)],device=cond.device,dtype=torch.float32).view(-1,1),
    ],dim=1).float()
    return cond,result


class CapturingBase:
    def __call__(self,x,*,noise_labels,conditional_inputs):
        self.inputs=x.clone()
        self.labels=noise_labels.clone()
        self.conditions=conditional_inputs[0].clone()
        return deterministic_prediction(x)


def deterministic_prediction(x):
    return (x.float()*.37+.11).to(x.dtype)


class LatentBatchCpuTests(unittest.TestCase):
    def test_terrestrial_noise_switch_preserves_reference_configuration(self):
        from terrain_conditioning import CONDITIONING_SNR
        profile=terrain_inference.InferenceProfile(name='cpu-contract',
            cached_weights=False,cuda_graphs=False,gpu_windows=False)
        baseline=[.5]*5
        world=SimpleNamespace(device='cpu',kwargs={'cond_snr':baseline.copy(),
            'frequency_mult':[1.]*5,'drop_water_pct':.5},
            torch_compile=False,tile_store=None,coarse_model=object(),base_model=object(),
            decoder_model=object())
        terrain_inference.configure_world(world,profile,world_profile='terrestrial-earthlike')
        self.assertEqual(world.kwargs['cond_snr'],list(CONDITIONING_SNR))
        terrain_inference.configure_world(world,profile,world_profile='natural')
        self.assertEqual(world.kwargs['cond_snr'],baseline)
        self.assertEqual(world.kwargs['frequency_mult'],[1.]*5)
        self.assertEqual(world.kwargs['drop_water_pct'],.5)

    @classmethod
    def setUpClass(cls):
        cls.old_threads=torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    @torch.inference_mode()
    def check_batch(self,n,nonfinite,second_pass):
        base=CapturingBase()
        world=SimpleNamespace(_dtype=torch.bfloat16,device=torch.device('cpu'),seed=123456,base_model=base)
        scheduler=SimpleNamespace(config=SimpleNamespace(sigma_data=.5))
        means=torch.tensor([14.99,11.65,15.87,619.26,833.12,69.40,.66])
        stds=torch.tensor([21.72,21.78,10.40,452.29,738.09,34.59,.47])
        histogram=torch.tensor([[.1,.2,.3,.4,.5]])
        t=torch.tensor(.61072594 if second_pass else 1.5645464)
        seed_offset=5820 if second_pass else 5819
        weight=linear_weight_window(64,'cpu',torch.float32)
        ctxs=[(0,-3+i,5-i) for i in range(n)]
        conds=[]
        for i in range(n):
            cond=(torch.arange(7*4*4,dtype=torch.float32).reshape(7,4,4)+i*7.3)*.73
            cond[-1]=torch.linspace(.25,1.25,16).reshape(4,4)
            if nonfinite:
                cond[0,0,0]=float('nan')
                cond[2,1,1]=float('nan')
                cond[3,1,2]=float('inf')
                cond[4,2,1]=-float('inf')
                cond[-1,0,1]=0 # Nonzero numerator/zero weight gives infinities.
                cond[:,3,3]=0 # Zero numerator/zero weight gives NaNs.
            conds.append(cond)
        samples=None
        if second_pass:
            samples=[]
            for i in range(n):
                value=(torch.arange(6*64*64,dtype=torch.float32).reshape(6,64,64)%31-i)*.035
                value[-1]=torch.linspace(.5,1.5,64)[None,:].expand(64,64)
                samples.append(value)
        expected_processed=[]
        expected_inputs=[]
        expected_outputs=[]
        labels=torch.as_tensor(t,dtype=world._dtype,device=world.device)
        view=labels.view(1,1,1,1)
        for index,(ctx,cond) in enumerate(zip(ctxs,conds)):
            normalized,processed=prechange_conditioning(cond,histogram,means,stds)
            upstream_input=cond[:-1]/cond[-1:]
            upstream_input=torch.cat([upstream_input,torch.ones((1,4,4))],dim=0)[None]
            upstream_processed=WorldPipeline._process_latent_conditioning(
                SimpleNamespace(seed=world.seed),upstream_input,histogram,means,stds,torch.tensor(0.),
                seed_offset=ctx[1]*65536+ctx[2])
            assert_tensor_bytes(self,processed,upstream_processed)
            self.assertTrue(torch.isfinite(normalized).all())
            if not nonfinite:
                self.assertTrue(torch.isfinite(processed).all())
            else:
                self.assertEqual(float(normalized[0,0,0,0]),float(means[0]))
                self.assertEqual(float(normalized[0,3,1,2]),torch.finfo(torch.float32).max)
                self.assertEqual(float(normalized[0,4,2,1]),-torch.finfo(torch.float32).max)
            expected_processed.append(processed)
            sample=torch.zeros((1,5,64,64),dtype=world._dtype)
            if samples is not None:
                sample=torch.as_tensor(samples[index],device=world.device,dtype=world._dtype)
                sample=sample[:-1]/sample[-1:]*scheduler.config.sigma_data
            noise=torch.from_numpy(gaussian_noise_patch(world.seed+seed_offset,ctx[1]*32,ctx[2]*32,64,64,channels=5,tile_h=64,tile_w=64))[None].to(dtype=world._dtype)
            z=noise*scheduler.config.sigma_data
            x_t=torch.cos(view)*sample+torch.sin(view)*z
            expected_inputs.append(x_t/scheduler.config.sigma_data)
            pred=-deterministic_prediction(x_t/scheduler.config.sigma_data)
            output=(torch.cos(view)*x_t-torch.sin(view)*scheduler.config.sigma_data*pred).float()/scheduler.config.sigma_data
            expected_outputs.append(torch.cat([output[0]*weight[None],weight[None]],dim=0))
        captured_float=[]
        def capture_concat(values,dim=1,w=None):
            result=_concat(values,dim,w)
            captured_float.append(result.clone())
            return result
        forbidden=AssertionError('CPU test attempted a CUDA call')
        with patch.object(mp_layers,'mp_concat',capture_concat), \
             patch('torch.cuda.current_device',side_effect=forbidden), \
             patch('torch.cuda.synchronize',side_effect=forbidden), \
             patch('torch.cuda.Event',side_effect=forbidden), \
             patch('torch.cuda.Stream',side_effect=forbidden), \
             patch('torch.cuda.is_available',side_effect=forbidden), \
             patch('torch.cuda.current_stream',side_effect=forbidden), \
             patch('torch.cuda.mem_get_info',side_effect=forbidden):
            actual=_latent_gpu_batch(world,ctxs,samples,conds,t,scheduler,weight,histogram,means,stds,seed_offset)
        self.assertEqual(len(captured_float),1)
        assert_tensor_bytes(self,captured_float[0],torch.cat(expected_processed))
        assert_tensor_bytes(self,base.conditions,torch.cat(expected_processed).to(world._dtype))
        assert_tensor_bytes(self,base.inputs,torch.cat(expected_inputs))
        assert_tensor_bytes(self,base.labels,labels.expand(n))
        for result,expected in zip(actual,expected_outputs):
            assert_tensor_bytes(self,result,expected)

    def test_finite_and_nonfinite_conditioning_in_actual_batches(self):
        for n in (1,4,9,16):
            for nonfinite in (False,True):
                for second_pass in (False,True):
                    with self.subTest(n=n,nonfinite=nonfinite,second_pass=second_pass):
                        self.check_batch(n,nonfinite,second_pass)

    def test_gate_rejects_deletion_of_nonzero_prediction_term(self):
        # Mutate an in-memory test copy, never the frozen runtime source.
        tree=ast.parse(inspect.getsource(terrain_inference._latent_gpu_batch))
        changed=False
        for node in ast.walk(tree):
            if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Sub) and isinstance(node.left,ast.BinOp) and isinstance(node.left.left,ast.Name) and node.left.left.id=='cosine':
                node.op=ast.Add()
                node.right=ast.Constant(value=0)
                changed=True
        self.assertTrue(changed)
        namespace=dict(vars(terrain_inference))
        exec(compile(ast.fix_missing_locations(tree),'<prediction-term-mutation>','exec'),namespace)
        with patch(__name__+'._latent_gpu_batch',namespace['_latent_gpu_batch']):
            with self.assertRaises(AssertionError):
                self.check_batch(4,False,True)


if __name__=='__main__':
    unittest.main()

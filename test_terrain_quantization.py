"""CPU tests of offline preparation; no GPU/backend performance claims."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from terrain_quantization import (ROLE, ActivationCollector, FrozenMP, freeze_base,
    export_qdq, quantize_weight, require_scope, tensor_sha, identity, build_tensorrt, sha_file,
    calibration_positions, resolve_exclusions, convolution_coverage, tensorrt_network_flags)
from terrain_diffusion.models.edm_unet import EDMUnet2D
from prepare_terrain_quantization import load_frozen


def small_base(*, levels=1, attention=False, image_size=8):
    torch.manual_seed(42)
    model = EDMUnet2D(image_size=image_size, in_channels=5, model_channels=8,
        model_channel_mults=list(range(1, levels+1)), layers_per_block=1, midblock_attention=attention,
        block_kwargs={'channels_per_head':4},
        conditional_inputs=[('tensor', 58, 1.)], fourier_scale='pos').eval()
    with torch.no_grad():
        model.out_gain.fill_(.7)
        for name, parameter in model.named_parameters():
            if name.endswith('emb_gain'):
                parameter.fill_(.31)
    return model


def inputs(n=2):
    return torch.randn(n,5,8,8), torch.linspace(.2,1.1,n), torch.randn(n,58)


class QuantizationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.cuda_guard = patch.object(torch.cuda, '_lazy_init', side_effect=AssertionError('CPU-only test'))
        self.cuda_guard.start()

    def tearDown(self):
        self.cuda_guard.stop()

    def test_scope_refuses_native_and_other_networks(self):
        require_scope()
        for kind, role in [('coarse',ROLE),('decoder',ROLE),('base','native'),('base','preview')]:
            with self.assertRaises(ValueError):
                require_scope(kind,role)

    @torch.no_grad()
    def test_frozen_gains_embeddings_exact_and_source_unchanged(self):
        model = small_base()
        before = {n:tensor_sha(v) for n,v in model.state_dict().items()}
        frozen, specs = freeze_base(model)
        self.assertTrue(specs)
        self.assertIsInstance(frozen.conditional_layers[0], FrozenMP)
        for n in (1,3):
            x,t,c = inputs(n)
            expected = model(x,t,[c])
            actual = frozen(x,t,[c])
            self.assertTrue(torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)))
            self.assertGreater(float(actual.abs().max()),0)
        self.assertEqual(before,{n:tensor_sha(v) for n,v in model.state_dict().items()})
        self.assertFalse(any(p.requires_grad for p in frozen.parameters()))
        with self.assertRaises(ValueError):
            freeze_base(model,max_weight_bytes=1)

    @torch.no_grad()
    def test_freeze_bf16_and_serialized_roundtrip(self):
        model = small_base().to(torch.bfloat16)
        frozen,specs = freeze_base(model)
        x,t,c = [v.bfloat16() for v in inputs()]
        expected = model(x,t,[c])
        self.assertTrue(torch.equal(expected.view(torch.uint8),frozen(x,t,[c]).view(torch.uint8)))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frozen.pt'
            torch.save(dict(role=ROLE,config=json.loads(json.dumps(dict(model.config))),
                            specs=specs,state=frozen.state_dict()),path)
            loaded=load_frozen(path)
            self.assertTrue(torch.equal(expected.view(torch.uint8),loaded(x,t,[c]).view(torch.uint8)))

    def test_collector_bounded_real_inputs_and_cleanup(self):
        model=small_base()
        collector=ActivationCollector(max_calls=2,max_values_per_layer=7).install(model)
        with torch.no_grad():
            for k in range(4):
                collector.start(dict(seed=k,profile='natural',pass_index=k%2))
                x,t,c=inputs()
                model(x,t,[c])
        collector.close()
        result=collector.result()
        self.assertEqual(len(result['contexts']),2)
        self.assertTrue(all(v['calls']==2 for v in result['layers'].values()))
        self.assertTrue(all(v['diagnostic_sample_count']<=7 for v in result['layers'].values()))
        self.assertTrue(all(not m._forward_pre_hooks for m in model.modules()))
        self.assertEqual(result['calibration_identity'],collector.result()['calibration_identity'])

    def test_nonfinite_and_empty_calibration_rejected(self):
        collector=ActivationCollector()
        with self.assertRaises(ValueError):
            collector.result()
        collector.start(dict(seed=0))
        with self.assertRaises(ValueError):
            collector.observe('bad',torch.tensor([1.,float('nan')]))

    def test_sparse_indices_large_lengths_never_round_out_of_bounds(self):
        # Allocate at most 1024 indices, not the enormous source activations.
        for length in (1, 3, 2**24, 2**24+1, 2**24+4, 2**31+7, 2**40):
            for take in (1, min(3,length), min(1024,length)):
                positions=calibration_positions(length,take)
                expected=[i*(length-1)//max(1,take-1) for i in range(take)]
                self.assertEqual(positions.dtype,torch.int64)
                self.assertEqual(positions.tolist(),expected)
                self.assertGreaterEqual(int(positions.min()),0)
                self.assertLess(int(positions.max()),length)
                if take>1:
                    self.assertEqual(int(positions[-1]),length-1)
                    self.assertTrue(bool(torch.all(positions[1:]>positions[:-1])))
        for length,take in ((0,1),(3,0),(3,4),(2**63,3)):
            with self.assertRaises(ValueError):
                calibration_positions(length,take)

    def test_quantization_axes_rounding_and_error(self):
        weights=np.array([[[[0.,1.,-1.]]],[[[0.,10.,-10.]]]],dtype=np.float32)
        q,s=quantize_weight(weights)
        self.assertEqual(q.dtype,np.int8)
        np.testing.assert_allclose(s,[1/127,10/127])
        reconstructed=q.astype(np.float32)*s[:,None,None,None]
        self.assertTrue(np.all(np.abs(reconstructed-weights)<=s[:,None,None,None]/2+1e-6))
        self.assertEqual(identity(dict(a=1,b=2)),identity(dict(b=2,a=1)))
        self.assertNotEqual(identity(dict(a=1)),identity(dict(a=2)))

    @torch.no_grad()
    def test_qdq_graph_has_calibrated_integer_weights_and_sensitive_exclusion(self):
        import onnx
        from onnx import numpy_helper
        from onnx.reference import ReferenceEvaluator
        model,specs=freeze_base(small_base(levels=2,attention=True))
        sample=inputs()
        collector=ActivationCollector().install(model)
        collector.start(dict(seed=42,profile='natural',pass_index=0))
        model(sample[0],sample[1],[sample[2]])
        collector.close()
        with tempfile.TemporaryDirectory() as tmp:
            report=export_qdq(model,sample,collector.result(),Path(tmp)/'out',exclude=('out_conv',))
            self.assertFalse(report['runtime_enabled'])
            self.assertFalse(report['integer_gpu_execution_proven'])
            self.assertFalse(report['physical_holdout_passed'])
            self.assertTrue(report['converted_layers'])
            self.assertTrue(any(v['layer']=='out_conv' for v in report['skipped_layers']))
            self.assertTrue(any(v['layer']=='enc.8x8_conv' and v['reason']=='sensitive-exclusion'
                                for v in report['skipped_layers']))
            self.assertTrue(any(v['op_type']=='ConvTranspose' and v['reason'].startswith('convtranspose-')
                                for v in report['skipped_layers']))
            self.assertEqual(report['runtime']['onnx'],onnx.__version__)
            graph=onnx.load(Path(tmp)/'out/base-int8-qdq.onnx')
            onnx.checker.check_model(graph)
            self.assertEqual(sum(n.op_type in ('Conv','ConvTranspose') for n in graph.graph.node),
                             len(report['converted_layers'])+len(report['skipped_layers']))
            values={v.name:numpy_helper.to_array(v) for v in graph.graph.initializer}
            self.assertTrue(any(v.dtype==np.int8 and v.ndim==4 for v in values.values()))
            dqs=[n for n in graph.graph.node if n.op_type=='DequantizeLinear' and n.input[0].endswith('_w')]
            self.assertEqual(len(dqs),len(report['converted_layers']))
            for node in dqs:
                self.assertEqual(values[node.input[1]].shape,(values[node.input[0]].shape[0],))
                self.assertEqual(next(a.i for a in node.attribute if a.name=='axis'),0)
            # Execute the exported graph on held-out random inputs and a batch
            # unlike tracing. This is a CPU graph check, not terrain fidelity.
            plain=ReferenceEvaluator(str(Path(tmp)/'out/base-fp32-conversion.onnx'))
            quantized=ReferenceEvaluator(graph)
            for count in (1,3):
                heldout=inputs(count)
                feeds=dict(zip(('sample','noise_labels','conditions'),(x.numpy() for x in heldout)))
                expected=model(heldout[0],heldout[1],[heldout[2]]).numpy()
                actual=plain.run(None,feeds)[0]
                np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-5)
                approximate=quantized.run(None,feeds)[0]
                self.assertEqual(approximate.shape,expected.shape)
                self.assertTrue(np.isfinite(approximate).all())
                self.assertGreater(float(np.max(np.abs(approximate-expected))),0.)

    def test_builder_budget_validation_before_optional_import(self):
        with self.assertRaises(ValueError):
            build_tensorrt('unused.onnx','unused',workspace_mib=2049)
        with self.assertRaises(ValueError):
            build_tensorrt('unused.onnx','unused',role='native')

    def test_builder_refuses_altered_export_receipt_before_tensorrt_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'base-int8-qdq.onnx'
            path.write_bytes(b'identity-test-only')
            receipt=dict(role=ROLE,converted_layers=[dict(layer='a')],runtime_enabled=False,
                         files={path.name:sha_file(path)})
            receipt['artifact_identity']=identity(receipt)
            receipt['converted_layers']=[dict(layer='tampered')]
            (Path(tmp)/'export.json').write_text(json.dumps(receipt),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'manifest identity mismatch'):
                build_tensorrt(path,Path(tmp)/'engine')

    def test_sensitive_input_layer_derived_from_config_and_unknown_exclusions_rejected(self):
        for size in (8,16,512):
            model,_=freeze_base(small_base(image_size=size))
            first=f'enc.{size}x{size}_conv'
            self.assertIn(first,resolve_exclusions(model))
            self.assertIn(first,resolve_exclusions(model,exclude=('out_conv',)))
            with self.assertRaisesRegex(ValueError,'matches no frozen layer'):
                resolve_exclusions(model,exclude=('misspelled_layer',))
        model.register_to_config(image_size=256)
        with self.assertRaisesRegex(ValueError,'input convolution missing'):
            resolve_exclusions(model)

    def test_convolution_inventory_includes_dynamic_weights_and_transpose(self):
        from onnx import helper
        initializers={'model.conv.weight':object()}
        cases=[('Conv','runtime_weight','weight-is-not-an-initializer'),
               ('ConvTranspose','model.conv.weight','convtranspose-not-supported-by-this-quantizer'),
               ('Conv','model.conv.weight','sensitive-exclusion')]
        for op,weight,reason in cases:
            node=helper.make_node(op,['x',weight],['y'],name='test')
            result=convolution_coverage(node,initializers,{'conv':{}},('conv',))
            self.assertEqual(result['reason'],reason)
            self.assertEqual(result['op_type'],op)
        node=helper.make_node('Conv',['x','model.conv.weight'],['y'])
        self.assertNotIn('reason',convolution_coverage(node,initializers,{'conv':{}},()))
        self.assertEqual(convolution_coverage(node,initializers,{},())['reason'],'no-layer-activation-calibration')

    def test_strongly_typed_tensor_rt_api_policy_without_backend_install(self):
        api=SimpleNamespace(__version__='10.13.3',NetworkDefinitionCreationFlag=SimpleNamespace(STRONGLY_TYPED=1))
        self.assertEqual(tensorrt_network_flags(api),2)
        for version in ('8.6.1','11.3.0'):
            api.__version__=version
            with self.assertRaisesRegex(ValueError,'TensorRT10.x'):
                tensorrt_network_flags(api)
        api.__version__='10.13.3'
        api.NetworkDefinitionCreationFlag=SimpleNamespace()
        with self.assertRaisesRegex(ValueError,'STRONGLY_TYPED'):
            tensorrt_network_flags(api)


if __name__=='__main__':
    unittest.main()

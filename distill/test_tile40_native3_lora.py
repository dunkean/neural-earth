from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import torch

from distill.test_tile40_native3 import TinyNet
from distill.tile40_native3 import Native3Base,codes_for,export_packed
from distill.tile40_native3_lora import ConditioningResidual,export_residual,load_residual,read_float_base


class ToyProjectionNet(TinyNet):
    def forward(self,inputs,noise_labels,conditional_inputs):
        x=self.conv(self.enc['first'](inputs))
        embedding=self.projection(conditional_inputs[0])+self.noise_linear(noise_labels[:,None].expand(-1,8))
        return self.out_conv(x+embedding[:,:,None,None])


class ConditioningResidualTests(unittest.TestCase):
    def test_unrounded_raw_weights_are_preserved_for_residual_addition(self):
        torch.manual_seed(17)
        base=Native3Base(ToyProjectionNet(),block=32)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';export_packed(base,source)
            state,_=read_float_base(source)
            spec=base.quant_specs['projection.weight']
            scale=getattr(base,spec['scale_name']).half()
            expected=codes_for(base.net.projection.weight,scale,4,32)[1].detach()
            self.assertTrue(torch.equal(state['projection.weight'],expected))
            self.assertFalse(torch.equal(expected,expected.bfloat16().float()))

    def test_frozen_base_real_update_and_self_contained_roundtrip(self):
        torch.manual_seed(19)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';export_packed(Native3Base(ToyProjectionNet(),32),source)
            state,manifest=read_float_base(source)
            net=ToyProjectionNet();net.load_state_dict(state)
            model=ConditioningResidual(net,manifest,rank=4)
            inputs=torch.randn(2,5,10,10);labels=torch.tensor([1.5625,.609375]);condition=torch.randn(2,8)
            optimizer=torch.optim.SGD(model.adapters.parameters(),lr=.5)
            expected=model(inputs,labels,[condition])
            expected.square().mean().backward()
            self.assertTrue(all(p.grad is None for p in model.net.parameters()))
            self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.adapters.values() if p.grad is not None),0)
            optimizer.step()
            with torch.no_grad():expected=model(inputs,labels,[condition])
            output=Path(directory)/'residual'
            exported=export_residual(model,source,output,step=1)
            self.assertEqual(exported['total_artifact_bytes'],sum(p.stat().st_size for p in output.rglob('*') if p.is_file()))
            with patch('terrain_diffusion.models.edm_unet.EDMUnet2D.from_config',side_effect=lambda cfg:ToyProjectionNet()):
                loaded,_=load_residual(output)
            with torch.no_grad():actual=loaded(inputs.bfloat16(),noise_labels=labels.bfloat16(),conditional_inputs=[condition.bfloat16()]).float()
            self.assertTrue(torch.equal(actual,expected))
            data=bytearray((output/'adapters.bin').read_bytes());data[0]^=1;(output/'adapters.bin').write_bytes(data)
            with self.assertRaises(ValueError):load_residual(output)

    def test_size_gate_counts_source_base(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';export_packed(Native3Base(ToyProjectionNet(),32),source)
            state,manifest=read_float_base(source)
            net=ToyProjectionNet();net.load_state_dict(state)
            model=ConditioningResidual(net,manifest,rank=4)
            output=Path(directory)/'oversize'
            with self.assertRaises(ValueError):export_residual(model,source,output,limit=10)
            self.assertFalse(output.exists())

    def test_residual_layout_metadata_is_authenticated(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source';export_packed(Native3Base(ToyProjectionNet(),32),source)
            state,manifest=read_float_base(source)
            net=ToyProjectionNet();net.load_state_dict(state)
            output=Path(directory)/'residual'
            export_residual(ConditioningResidual(net,manifest,4),source,output)
            path=output/'manifest.json';metadata=json.loads(path.read_text())
            metadata['adapters'][next(iter(metadata['adapters']))]['offset']+=2
            path.write_text(json.dumps(metadata))
            with self.assertRaises(ValueError):load_residual(output)


if __name__=='__main__':unittest.main()

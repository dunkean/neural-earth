import tempfile
import unittest
from pathlib import Path

import torch

from distill.common import REVISION
from distill.native_inference import load_native
from distill.native_weight_runtime import runtime_model
from distill import test_native_weight_quant as fixtures


class NativeWeightRuntimeTests(unittest.TestCase):
    def test_baked_native_forward_matches_qat_and_loader_accepts_ema_only(self):
        torch.set_num_threads(2);torch.manual_seed(31)
        _,model=fixtures.NativeWeightQuantTests().model()
        saved=dict(architecture='native-weight-quant',config=model.config,scheme=model.scheme,
            block=model.block,quantized_layers=model.quantized_layers,ema=model.state_dict(),
            teacher_revision=REVISION,step=20000)
        x=torch.randn(2,5,64,64);labels=torch.tensor([1.5625,.609375]);conds=[torch.randn(2,58)]
        with torch.no_grad():expected=model(x,labels,conds)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'ema.pt';torch.save(saved,p)
            native,metadata=load_native(p)
            self.assertTrue(native._terrain_custom_base)
            self.assertEqual(metadata['step'],20000)
            with torch.no_grad():actual=native(x.bfloat16(),labels.bfloat16(),[conds[0].bfloat16()]).float()
            self.assertTrue(torch.equal(expected,actual),float((actual-expected).abs().max()))
            self.assertTrue(all(not p.requires_grad for p in native.parameters()))


if __name__=='__main__':unittest.main()

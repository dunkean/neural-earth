import copy
import tempfile
import unittest
from pathlib import Path

import torch

from distill.common import REVISION
from distill.native_weight_artifact_audit import audit
from distill.native_weight_quant import export_packed
from distill import test_native_weight_quant as fixtures


class NativeWeightArtifactAuditTests(unittest.TestCase):
    def test_full_inventory_matches_and_wrong_source_weights_are_detected(self):
        torch.set_num_threads(2);torch.manual_seed(44)
        _,model=fixtures.NativeWeightQuantTests().model()
        saved=dict(architecture='native-weight-quant',config=model.config,scheme=model.scheme,
            block=model.block,quantized_layers=model.quantized_layers,ema=model.state_dict(),step=30000,
            teacher_revision=REVISION,teacher_weights_sha256='toy')
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder);export_packed(saved,path)
            report=audit(saved,path,'cpu')
            self.assertEqual(report['status'],'complete')
            self.assertEqual(report['tensor_count'],len(saved['ema']))
            self.assertEqual(report['quantized_kernel_count'],len(model.quantized_layers))
            wrong=copy.deepcopy(saved);wrong['ema']['net.out_gain']+=1.
            report=audit(wrong,path,'cpu')
            self.assertEqual(report['status'],'failed')
            self.assertEqual([r['tensor'] for r in report['rows'] if not r['exact']],['net.out_gain'])
            with self.assertRaisesRegex(ValueError,'step'):audit(saved|dict(step=30001),path,'cpu')


if __name__=='__main__':unittest.main()

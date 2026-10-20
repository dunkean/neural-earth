"""The delivered artifacts contain exactly the selected inference states."""
import hashlib
import json
from pathlib import Path
import sys
import unittest
import subprocess
from unittest.mock import patch

import torch
from distill.common import REPO,bootstrap


class SelectedWeightsTests(unittest.TestCase):
    def test_five_inference_only_models_match_the_manifest(self):
        root=REPO/'distill/weights'
        manifest=json.loads((root/'manifest.json').read_text())
        self.assertEqual(set(manifest['exports']),{'coarse8','coarse2','base_residual8k','decoder64','decoder16'})
        self.assertEqual({p.name for p in root.glob('*.pt')},{v['file'] for v in manifest['exports'].values()})
        for key,item in manifest['exports'].items():
            path=root/item['file'];payload=path.read_bytes()
            self.assertEqual(hashlib.sha256(payload).hexdigest(),item['sha256'])
            self.assertEqual(len(payload),item['bytes']);self.assertLess(len(payload),100000000)
            saved=torch.load(path,map_location='cpu',weights_only=True)
            self.assertTrue(saved['inference_only'])
            self.assertEqual(saved['step'],item['training_step'])
            self.assertFalse({'optimizer','ema','rng_state'} & saved.keys())
            if key=='base_residual8k':
                self.assertEqual(saved['selected_weights'],'raw')
                self.assertTrue(saved['contract']['residual_altitude'])
                self.assertEqual(saved['contract']['coarse_sha256'],manifest['exports']['coarse2']['sha256'])

    def test_application_imports_do_not_change_gpu_or_storage_environment(self):
        import os
        with patch.dict(sys.modules,{'terrain_app':object()}):
            before=dict(os.environ)
            bootstrap()
            self.assertEqual(dict(os.environ),before)

    def test_deployed_inference_does_not_import_research_trainers(self):
        code='''import sys
from distill import tile40_runtime,tile40_flow_runtime,tile40_export_utils
for name in ('distill.teacher','distill.tile40_decoder_pilot','distill.tile40_reference_data','distill.native_weight_quant'):
    assert name not in sys.modules,name
'''
        subprocess.run([sys.executable,'-c',code],cwd=REPO,check=True,timeout=20)


if __name__=='__main__':unittest.main()

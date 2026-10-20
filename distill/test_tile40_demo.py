import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from distill.tile40_demo import routes


class DemoRoutesTests(unittest.TestCase):
    def test_serves_models_and_runtime_without_exposing_teacher_fixtures(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            web=root/'webgpu';web.mkdir();(web/'vendor').mkdir()
            models=root/'models';models.mkdir()
            digest=lambda file:hashlib.sha256(file.read_bytes()).hexdigest()
            for name in ('tile40_runtime.mjs','noise.mjs'):
                (web/name).write_text('// runtime\n')
            (models/'model.onnx').write_bytes(b'model')
            (models/'expected-height.bin').write_bytes(b'hidden fixture')
            manifest=dict(models={'decoder':dict(file='model.onnx',bytes=5,sha256=digest(models/'model.onnx'))},
                provenance=dict(runtime_sha256=digest(web/'tile40_runtime.mjs'),noise_sha256=digest(web/'noise.mjs')))
            (models/'manifest.json').write_text(json.dumps(manifest))
            with patch('distill.tile40_demo.REPO',root):
                files=routes(models)
                self.assertIn('/model/model.onnx',files)
                self.assertNotIn('/model/expected-height.bin',files)
                (web/'noise.mjs').write_text('// changed\n')
                with self.assertRaisesRegex(ValueError,'runtime differs'):
                    routes(models)
                snapshot=models/'runtime-snapshot';snapshot.mkdir()
                (snapshot/'tile40_runtime.mjs').write_text('// runtime\n')
                (snapshot/'noise.mjs').write_text('// runtime\n')
                frozen=routes(models)
                self.assertEqual(frozen['/runtime/noise.mjs'],snapshot/'noise.mjs')


if __name__=='__main__':
    unittest.main()

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from distill.bundle_coarse import prepare
from distill.common import REVISION


class BundleCoarseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = dict(teacher_revision=REVISION, exports={'decoder': {'sha256': 'keep'}})
        for number in (1, 2):
            run = self.root/f'coarse{number}'
            (run/'export').mkdir(parents=True)
            source, payload = b'checkpoint', f'weights{number}'.encode()
            (run/'best.pt').write_bytes(source)
            (run/'export/coarse.pt').write_bytes(payload)
            export = dict(stage='coarse', inference_only=True,
                source=dict(step=195000, sha256=hashlib.sha256(source).hexdigest(),
                    config=dict(stage='coarse', solver_steps=number)),
                export=dict(file='coarse.pt', bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()))
            for name, value in (
                ('status.json', dict(status='complete', step=200000)),
                ('export/manifest.json', export),
                ('delivery.json', dict(status='complete', training_steps=200000, selected_step=195000,
                    validation_count=384, finite=True, export_max_abs=0.,
                    smoke_indices=[0, 192, 383], export=export))):
                (run/name).write_text(json.dumps(value))

    def test_preserves_other_models_and_full_run_best_selection(self):
        weights, result = prepare(self.root, self.manifest)
        self.assertEqual(set(weights), {'coarse1.pt', 'coarse2.pt'})
        self.assertEqual(result['exports']['decoder'], self.manifest['exports']['decoder'])
        self.assertEqual(result['exports']['coarse2']['training_step'], 195000)
        self.assertEqual(result['exports']['coarse2']['last_training_step'], 200000)
        self.assertNotIn('coarse2', self.manifest['exports'])

    def test_rejects_partial_second_run_before_returning_any_replacement(self):
        (self.root/'coarse2/status.json').write_text(json.dumps(dict(status='interrupted', step=199999)))
        with self.assertRaisesRegex(ValueError, 'Coarse2'):
            prepare(self.root, self.manifest)

    def test_rejects_changed_weights_and_changed_selected_source(self):
        for filename in ('export/coarse.pt', 'best.pt'):
            path = self.root/'coarse1'/filename
            original = path.read_bytes()
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError):
                prepare(self.root, self.manifest)
            path.write_bytes(original)

    def test_rejects_missing_delivery_after_training(self):
        (self.root/'coarse2/delivery.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Coarse2'):
            prepare(self.root, self.manifest)


if __name__ == '__main__':
    unittest.main()

"""Review switching validates choices and separates all neural cache identities."""
import json
from pathlib import Path
import tempfile
import unittest

from distill.common import atomic_json
from distill.live_runtime import LiveModels, STAGES, catalog, validate_selection


class LiveModelTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name)/'config.json'
        self.selection = {stage: 'teacher' for stage in STAGES}
        self.options = {stage: {'teacher': {'label': 'Original', 'path': None},
                               'student': {'label': 'Student', 'path': 'fixed.pt', 'sha256': stage}}
                        for stage in STAGES}
        atomic_json(self.path, dict(selection=self.selection, revision='a', reset='initial'))

    def test_paths_and_cross_stage_choices_cannot_be_submitted(self):
        for value in (None, {}, dict(self.selection, extra='teacher'),
                      dict(self.selection, base='/tmp/untrusted.pt')):
            with self.assertRaises(ValueError):
                validate_selection(value, self.options)

    def test_switch_is_immutable_until_restart_and_changes_identity(self):
        old = LiveModels(self.path, self.options)
        baseline = old.identity()
        proposed = dict(self.selection, base='student')
        old.save_request(dict(selection=proposed))
        self.assertEqual(old.identity(), baseline)
        new = LiveModels(self.path, self.options)
        self.assertEqual(new.selection, proposed)
        self.assertEqual(new.identity()['reset'], baseline['reset'])
        self.assertNotEqual(new.identity(), baseline)

    def test_reset_retains_selection_and_invalidates_neural_identity(self):
        old = LiveModels(self.path, self.options)
        old.save_request(dict(selection=self.selection, reset=True))
        new = LiveModels(self.path, self.options)
        self.assertEqual(new.selection, old.selection)
        self.assertNotEqual(new.identity()['reset'], old.identity()['reset'])
        self.assertNotEqual(new.public()['revision'], old.public()['revision'])
        self.assertEqual(json.loads(self.path.read_text())['selection'], self.selection)

    def test_coarse_four_is_optional_and_keeps_other_choices(self):
        root = Path(self.directory.name)
        self.assertNotIn('solver4', catalog(root)['coarse'])
        path = root/'live/weights/coarse4/coarse.pt'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'fixed-four-step-export')
        options = catalog(root)
        selection = dict(self.selection, coarse='solver4')
        self.assertEqual(validate_selection(selection, options), selection)
        self.assertEqual(options['coarse']['solver4']['path'], str(path))
        self.assertNotIn('solver4', options['base'])
        self.assertNotIn('solver4', options['decoder'])


if __name__ == '__main__':
    unittest.main()

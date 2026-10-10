"""Review switching validates choices and separates all neural cache identities."""
import json
from pathlib import Path
import tempfile
import unittest

from distill.common import atomic_json
from distill.live_runtime import (DEFAULT_SELECTION, LiveModels, STAGES, catalog,
                                 prepare_config, validate_selection)


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

    def install_review_exports(self):
        root = Path(self.directory.name)
        for relative in ('live/weights/coarse4/coarse.pt',
                         'final/weights/candidate-1/decoder.pt'):
            path = root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'fixed-export')
        return root

    def test_fresh_config_uses_coarse_four_original_base_and_distilled_decoder(self):
        root = self.install_review_exports()
        path = root/'new-config.json'
        models = LiveModels(path, catalog(root))
        self.assertEqual(models.selection, DEFAULT_SELECTION)
        self.assertEqual(json.loads(path.read_text())['selection'], DEFAULT_SELECTION)
        self.assertEqual(prepare_config(path, catalog(root)), models.config)

    def test_missing_default_export_fails_without_writing_config(self):
        path = Path(self.directory.name)/'new-config.json'
        with self.assertRaisesRegex(ValueError, 'Missing default model exports:.*coarse/solver4'):
            prepare_config(path, catalog(self.directory.name))
        self.assertFalse(path.exists())

    def test_base_exports_are_never_selectable(self):
        root = self.install_review_exports()
        for number in range(1, 5):
            path = root/f'final/weights/candidate-{number}/base.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'retired-base')
        options = catalog(root)
        self.assertEqual(set(options['base']), {'teacher'})
        with self.assertRaises(ValueError):
            validate_selection(dict(DEFAULT_SELECTION, base='initial128'), options)

    def test_legacy_base_is_retired_without_resetting_or_changing_other_choices(self):
        root = self.install_review_exports()
        for choice in ('initial128', 'coupled128', 'refined128', 'refined192'):
            with self.subTest(choice=choice):
                atomic_json(self.path, dict(selection=dict(DEFAULT_SELECTION, base=choice),
                                           revision='old', reset='keep-cache-namespace'))
                models = LiveModels(self.path, catalog(root))
                self.assertEqual(models.selection, DEFAULT_SELECTION)
                self.assertEqual(models.config['reset'], 'keep-cache-namespace')
                self.assertNotEqual(models.config['revision'], 'old')
                self.assertEqual(json.loads(self.path.read_text()), models.config)


if __name__ == '__main__':
    unittest.main()

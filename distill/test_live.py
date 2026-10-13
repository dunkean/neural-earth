"""Review switching validates choices and separates all neural cache identities."""
import json
import hashlib
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from flask import Flask

from distill.common import atomic_json
from distill.live_runtime import (DEFAULT_SELECTION, LiveModels, STAGES, catalog,
                                 prepare_config, validate_selection)
from distill.live_server import register_review_routes


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
        self.assertNotIn('solver4', catalog(root, bundled_root=root/'bundled')['coarse'])
        path = root/'live/weights/coarse4/coarse.pt'
        path.parent.mkdir(parents=True)
        path.write_bytes(b'fixed-four-step-export')
        options = catalog(root, bundled_root=root/'bundled')
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
        models = LiveModels(path, catalog(root, bundled_root=root/'bundled'))
        self.assertEqual(models.selection, DEFAULT_SELECTION)
        self.assertEqual(json.loads(path.read_text())['selection'], DEFAULT_SELECTION)
        self.assertEqual(prepare_config(path, catalog(root, bundled_root=root/'bundled')), models.config)

    def test_missing_default_export_fails_without_writing_config(self):
        path = Path(self.directory.name)/'new-config.json'
        with self.assertRaisesRegex(ValueError, 'Missing default model exports:.*coarse/solver4'):
            prepare_config(path, catalog(self.directory.name, bundled_root=Path(self.directory.name)/'bundled'))
        self.assertFalse(path.exists())

    def test_short_coarse_options_preserve_the_four_step_default(self):
        root = self.install_review_exports()
        for evaluations in (2, 1):
            path = root/f'live/weights/coarse{evaluations}/coarse.pt'
            path.parent.mkdir(parents=True)
            path.write_bytes(f'fixed-{evaluations}-step-export'.encode())
        options = catalog(root, bundled_root=root/'bundled')
        self.assertIn('solver2', options['coarse'])
        self.assertIn('solver1', options['coarse'])
        models = LiveModels(root/'new-config.json', options)
        self.assertEqual(models.selection, DEFAULT_SELECTION)
        for evaluations in (2, 1):
            selection = dict(DEFAULT_SELECTION, coarse=f'solver{evaluations}')
            models.save_request(dict(selection=selection))
            self.assertEqual(LiveModels(models.path, options).selection, selection)

    def test_full_coarse_replaces_pilot_only_after_matching_export_delivery(self):
        root = self.install_review_exports()
        pilot = root/'live/weights/coarse1/coarse.pt'
        pilot.parent.mkdir(parents=True)
        pilot.write_bytes(b'preserved-pilot')
        full = root/'ckpt/coarse-full/coarse1/export/coarse.pt'
        full.parent.mkdir(parents=True)
        full.write_bytes(b'full-export')
        options = lambda: catalog(root, bundled_root=root/'bundled')['coarse']['solver1']
        self.assertEqual(options()['path'], str(pilot))
        delivery = full.parent.parent/'delivery.json'
        proof = dict(status='complete', export_max_abs=0.,
                     export=dict(export=dict(sha256=hashlib.sha256(full.read_bytes()).hexdigest())))
        atomic_json(delivery, proof)
        self.assertEqual(options()['path'], str(full))
        full.write_bytes(b'changed-after-verification')
        self.assertEqual(options()['path'], str(pilot))

    def test_bundled_weights_work_without_the_external_training_directory(self):
        root = Path(self.directory.name)
        bundled = root/'bundled'
        bundled.mkdir()
        for name in ('coarse8', 'coarse4', 'coarse2', 'coarse1', 'decoder'):
            (bundled/(name+'.pt')).write_bytes(name.encode())
        options = catalog(root/'empty-data', bundled_root=bundled)
        models = LiveModels(root/'new-config.json', options)
        self.assertEqual(models.selection, DEFAULT_SELECTION)
        self.assertEqual(options['coarse']['solver4']['path'], str(bundled/'coarse4.pt'))
        self.assertEqual(options['coarse']['solver2']['path'], str(bundled/'coarse2.pt'))
        self.assertEqual(options['coarse']['solver1']['path'], str(bundled/'coarse1.pt'))
        self.assertEqual(options['decoder']['distilled200k']['path'], str(bundled/'decoder.pt'))
        self.assertEqual(set(options['base']), {'teacher'})

    def test_base_exports_are_never_selectable(self):
        root = self.install_review_exports()
        for number in range(1, 5):
            path = root/f'final/weights/candidate-{number}/base.pt'
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b'retired-base')
        options = catalog(root, bundled_root=root/'bundled')
        self.assertEqual(set(options['base']), {'teacher'})
        with self.assertRaises(ValueError):
            validate_selection(dict(DEFAULT_SELECTION, base='initial128'), options)

    def test_native_candidate_requires_matching_full_tensor_audit(self):
        root=self.install_review_exports()
        native=root/'ckpt/native-weight-nf4-100k/step070000.pt'
        native.parent.mkdir(parents=True);native.write_bytes(b'new-native-export')
        proof=root/'eval/native-nf4-artifact70000-5090-audit.json'
        proof.parent.mkdir()
        options=lambda:catalog(root,bundled_root=root/'bundled')
        self.assertNotIn('nf4_70k',options()['base'])
        atomic_json(proof,dict(status='complete',tensor_count=174,
            checkpoint_sha256=hashlib.sha256(native.read_bytes()).hexdigest()))
        self.assertEqual(options()['base']['nf4_70k']['format'],'native')
        self.assertEqual(LiveModels(root/'new-config.json',options()).selection,DEFAULT_SELECTION)
        native.write_bytes(b'changed-native-export')
        self.assertNotIn('nf4_70k',options()['base'])

    def test_native_counter_preserves_forward_and_separates_world_counts(self):
        import torch
        from distill.native_live_runtime import NativeLiveBase
        net=torch.nn.Linear(2,2).eval().requires_grad_(False)
        net._terrain_custom_base=True
        x=torch.ones(1,2)
        first=dict(calls=0,output_pixels=0);second=dict(calls=0,output_pixels=0)
        a,b=NativeLiveBase(net,first),NativeLiveBase(net,second)
        self.assertTrue(torch.equal(a(x),net(x)))
        self.assertEqual(first['calls'],1);self.assertEqual(second['calls'],0)
        self.assertTrue(a._terrain_custom_base)
        b(x);self.assertEqual(first['calls'],1);self.assertEqual(second['calls'],1)

    def test_compact_b75_requires_both_pinned_files_and_persists_selection(self):
        from distill.compact_release import RELEASE
        root = self.install_review_exports()
        compact = root/'releases'/RELEASE
        compact.mkdir(parents=True)
        payloads = {'manifest.json': b'fixed-manifest', 'weights.bin': b'fixed-weights'}
        hashes = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
        with patch('distill.compact_release.HASHES', hashes):
            for name, payload in payloads.items():
                (compact/name).write_bytes(payload)
            options = catalog(root, bundled_root=root/'bundled')
            self.assertEqual(options['base']['nf4_b75']['format'], 'scalar-packed')
            models = LiveModels(root/'new-config.json', options)
            models.save_request(dict(selection=dict(DEFAULT_SELECTION, base='nf4_b75')))
            self.assertEqual(LiveModels(models.path, options).selection['base'], 'nf4_b75')
            (compact/'manifest.json').write_bytes(b'changed')
            self.assertNotIn('nf4_b75', catalog(root, bundled_root=root/'bundled')['base'])

    def test_compact_base_loads_once_and_installs_through_native_interface(self):
        import torch
        root = Path(self.directory.name)
        artifact = root/'weights.bin'
        artifact.write_bytes(b'packed')
        options = {stage: {'teacher': dict(label='Original', path=None)} for stage in STAGES}
        options['base']['nf4_b75'] = dict(label='B75', path=str(artifact), directory=str(root),
            sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(), format='scalar-packed')
        atomic_json(self.path, dict(selection=dict.fromkeys(STAGES, 'teacher') | {'base': 'nf4_b75'},
                                   revision='compact', reset='keep'))
        models = LiveModels(self.path, options)
        net = torch.nn.Conv2d(5, 5, 1).eval()
        world = SimpleNamespace(device='cpu')
        with patch('distill.compact_release.verified', return_value=True), \
             patch('distill.native_scalar_pack.runtime_scalar_packed', return_value=net) as load:
            models.install(world)
            models.install(SimpleNamespace(device='cpu'))
            load.assert_called_once_with(str(root), 'cpu')
        inputs = torch.zeros(1, 5, 8, 8)
        self.assertTrue(torch.equal(world.base_model(inputs), net(inputs)))
        self.assertEqual(world._distill_counts['base']['calls'], 1)

    def test_native_install_preserves_other_student_counters(self):
        import torch
        from types import SimpleNamespace
        from distill.native_live_runtime import install
        counts=dict(coarse=dict(calls=7,output_pixels=16384),decoder=dict(calls=3,output_pixels=256))
        world=SimpleNamespace(_distill_counts=counts)
        model=torch.nn.Conv2d(5,5,1).eval().requires_grad_(False)
        install(world,model)
        inputs=torch.zeros(2,5,8,8)
        self.assertTrue(torch.equal(world.base_model(inputs),model(inputs)))
        self.assertEqual(world._distill_counts['base'],dict(calls=1,output_pixels=128))
        self.assertEqual(counts['coarse']['calls'],7)
        self.assertEqual(counts['decoder']['calls'],3)

    def test_legacy_base_is_retired_without_resetting_or_changing_other_choices(self):
        root = self.install_review_exports()
        for choice in ('initial128', 'coupled128', 'refined128', 'refined192'):
            with self.subTest(choice=choice):
                atomic_json(self.path, dict(selection=dict(DEFAULT_SELECTION, base=choice),
                                           revision='old', reset='keep-cache-namespace'))
                models = LiveModels(self.path, catalog(root, bundled_root=root/'bundled'))
                self.assertEqual(models.selection, DEFAULT_SELECTION)
                self.assertEqual(models.config['reset'], 'keep-cache-namespace')
                self.assertNotEqual(models.config['revision'], 'old')
                self.assertEqual(json.loads(self.path.read_text()), models.config)


class StandardViewerTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name)/'models.json'
        self.selection = dict(DEFAULT_SELECTION)
        self.options = catalog()
        atomic_json(self.path, dict(selection=self.selection, revision='initial', reset='keep-orogen'))
        self.models = LiveModels(self.path, self.options)
        self.server = SimpleNamespace(app=Flask(__name__), PROFILE='test-profile',
            RESTART_COMMAND=['python', 'terrain_server.py'], runtimes={}, runtimes_lock=threading.RLock(),
            generation_coordinator=SimpleNamespace(lock=threading.RLock(), suspend=Mock()))
        self.server.app.add_url_rule('/', view_func=lambda: '<html></html>')
        self.restart = Mock()
        with patch('distill.live_runtime.current', return_value=self.models):
            register_review_routes(self.server, restart_request=self.restart)
        self.client = self.server.app.test_client()

    def test_standard_viewer_exposes_menu_choices_and_controls(self):
        response = self.client.get('/')
        self.assertIn(b'/distill/live-controls.js', response.data)
        self.assertIn(b'url.searchParams.set("nn_engine","exact")', response.data)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        with self.client.get('/distill/live-controls.js') as controls:
            self.assertEqual(controls.status_code, 200)
        response = self.client.get('/api/distill/models')
        self.assertEqual(response.json['selection'], DEFAULT_SELECTION)
        self.assertEqual(response.headers['Cache-Control'], 'no-store')
        self.assertIn('teacher', {option['id'] for option in response.json['options']['coarse']})
        self.assertIn('solver4', {option['id'] for option in response.json['options']['coarse']})

    def test_applying_original_models_uses_standard_restart_and_preserves_cache_reset(self):
        selection = {stage: 'teacher' for stage in STAGES}
        response = self.client.post('/api/distill/models', json=dict(selection=selection))
        self.assertEqual(response.status_code, 202)
        self.restart.assert_called_once_with()
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved['selection'], selection)
        self.assertEqual(saved['reset'], 'keep-orogen')
        self.assertEqual(saved['revision'], response.json['revision'])
        self.assertEqual(self.models.selection, DEFAULT_SELECTION, 'Old process remains immutable until restart')
        self.assertEqual(self.client.post('/api/distill/models', json=dict(selection=selection)).status_code, 409)

    def test_choices_remain_accessible_while_the_gpu_is_busy(self):
        runtime = SimpleNamespace(lock=threading.Lock(), live_worlds=Mock())
        runtime.lock.acquire()
        self.addCleanup(runtime.lock.release)
        self.server.runtimes[0] = runtime
        response = self.client.get('/api/distill/models')
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['selection'], DEFAULT_SELECTION)
        runtime.live_worlds.assert_not_called()

    def test_reset_retains_selected_models_and_changes_only_the_neural_namespace(self):
        response = self.client.post('/api/distill/models', json=dict(selection=self.selection, reset=True))
        self.assertEqual(response.status_code, 202)
        saved = json.loads(self.path.read_text())
        self.assertEqual(saved['selection'], self.selection)
        self.assertNotEqual(saved['reset'], 'keep-orogen')

    def test_invalid_choice_or_unavailable_restart_does_not_change_models(self):
        before = self.path.read_bytes()
        response = self.client.post('/api/distill/models', json=dict(selection=dict(self.selection, base='solver4')))
        self.assertEqual(response.status_code, 400)
        self.server.RESTART_COMMAND = None
        response = self.client.post('/api/distill/models', json=dict(selection=self.selection))
        self.assertEqual(response.status_code, 501)
        self.restart.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_distillation_import_preserves_standard_runtime_environment(self):
        from distill.common import bootstrap
        with patch.dict('sys.modules', terrain_app=Mock()), patch.dict('os.environ', {}, clear=True):
            bootstrap()
            import os
            self.assertEqual(dict(os.environ), {})


if __name__ == '__main__':
    unittest.main()

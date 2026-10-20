"""Only the selected models are offered; migration preserves saved world state."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from distill.common import atomic_json
from distill.live_runtime import DEFAULT_SELECTION,LiveModels,STAGES,catalog,prepare_config,validate_selection


class LiveModelTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory();self.addCleanup(self.directory.cleanup)
        self.root=Path(self.directory.name);self.path=self.root/'config.json'
        self.bundled=self.root/'bundled';self.bundled.mkdir()
        for name in ['coarse8.pt','coarse2.pt','decoder.pt','decoder16-10k.pt']:
            (self.bundled/name).write_bytes(name.encode())
        self.options=catalog(self.root,bundled_root=self.bundled)

    def test_fresh_world_defaults_are_selected_versions(self):
        models=LiveModels(self.path,self.options)
        self.assertEqual(models.selection,dict(coarse='solver8',base='teacher',decoder='distilled200k'))
        self.assertEqual(prepare_config(self.path,self.options),models.config)

    def test_only_allowlisted_versions_are_offered_even_if_research_files_exist(self):
        for name in ['coarse1.pt','coarse4.pt','base.pt','nf4.pt']:
            (self.bundled/name).write_bytes(b'retired')
        options=catalog(self.root,bundled_root=self.bundled)
        self.assertEqual(set(options['coarse']),{'teacher','solver8','solver2'})
        self.assertEqual(set(options['base']),{'teacher'})
        self.assertEqual(set(options['decoder']),{'teacher','distilled200k','compact10k'})
        self.assertEqual(LiveModels(self.path,options).public()['tile40_url'],'/tile40/')

    def test_external_research_checkpoints_cannot_inject_choices(self):
        native=self.root/'ckpt/native-weight-nf4-100k/step070000.pt'
        native.parent.mkdir(parents=True);native.write_bytes(b'research')
        proof=self.root/'eval/native-nf4-artifact70000-5090-audit.json'
        proof.parent.mkdir();atomic_json(proof,dict(status='complete',tensor_count=174,
            checkpoint_sha256=hashlib.sha256(native.read_bytes()).hexdigest()))
        self.assertEqual(set(catalog(self.root,bundled_root=self.bundled)['base']),{'teacher'})

    def test_paths_retired_choices_and_finite_base_cannot_enter_the_world_pipeline(self):
        for value in [None,{},dict(DEFAULT_SELECTION,extra='teacher'),dict(DEFAULT_SELECTION,base='/tmp/a.pt'),
                      dict(DEFAULT_SELECTION,coarse='solver4'),dict(DEFAULT_SELECTION,base='base_residual8k')]:
            with self.subTest(value=value),self.assertRaises(ValueError):validate_selection(value,self.options)

    def test_originals_and_compact_decoder_are_selectable(self):
        for selection in [dict.fromkeys(STAGES,'teacher'),dict(DEFAULT_SELECTION,coarse='solver2',decoder='compact10k')]:
            self.assertEqual(validate_selection(selection,self.options),selection)

    def test_model_change_is_immutable_until_restart(self):
        old=LiveModels(self.path,self.options);before=old.identity()
        selection=dict(DEFAULT_SELECTION,coarse='solver2',decoder='compact10k')
        old.save_request(dict(selection=selection))
        self.assertEqual(old.identity(),before)
        new=LiveModels(self.path,self.options)
        self.assertEqual(new.selection,selection)
        self.assertNotEqual(new.identity(),before)
        self.assertEqual(new.config['reset'],old.config['reset'])

    def test_reset_changes_only_neural_namespace(self):
        old=LiveModels(self.path,self.options)
        old.save_request(dict(selection=old.selection,reset=True))
        new=LiveModels(self.path,self.options)
        self.assertEqual(new.selection,old.selection)
        self.assertNotEqual(new.config['reset'],old.config['reset'])
        self.assertNotEqual(new.identity(),old.identity())

    def test_migrates_retired_coarse_and_base_without_losing_reset(self):
        for coarse,base,target in [('solver4','nf4_70k','solver8'),('solver1','initial128','solver2'),
                                   ('solver2','refined192','solver2')]:
            atomic_json(self.path,dict(selection=dict(coarse=coarse,base=base,decoder='compact10k'),revision='old',reset='keep'))
            model=LiveModels(self.path,self.options)
            self.assertEqual(model.selection,dict(coarse=target,base='teacher',decoder='compact10k'))
            self.assertEqual(model.config['reset'],'keep')
            self.assertNotEqual(model.config['revision'],'old')

    def test_missing_default_fails_before_writing_settings(self):
        (self.bundled/'coarse8.pt').unlink()
        with self.assertRaisesRegex(ValueError,'coarse/solver8'):
            LiveModels(self.path,catalog(self.root,bundled_root=self.bundled))
        self.assertFalse(self.path.exists())

    def test_manifest_tampering_is_rejected(self):
        model=dict(file='coarse8.pt',sha256='wrong',stage='coarse')
        atomic_json(self.bundled/'manifest.json',dict(exports={'coarse8':model}))
        with self.assertRaisesRegex(ValueError,'manifest'):
            catalog(self.root,bundled_root=self.bundled)


class NativeResearchTests(unittest.TestCase):
    setUp = LiveModelTests.setUp

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
        root = self.root
        compact = root/'releases'/RELEASE
        compact.mkdir(parents=True)
        payloads = {'manifest.json': b'fixed-manifest', 'weights.bin': b'fixed-weights'}
        hashes = {name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()}
        with patch('distill.compact_release.HASHES', hashes):
            for name, payload in payloads.items():
                (compact/name).write_bytes(payload)
            options = catalog(root, bundled_root=self.bundled)
            self.assertEqual(options['base']['nf4_b75']['format'], 'scalar-packed')
            models = LiveModels(root/'new-config.json', options)
            models.save_request(dict(selection=dict(DEFAULT_SELECTION, base='nf4_b75')))
            self.assertEqual(LiveModels(models.path, options).selection['base'], 'nf4_b75')
            (compact/'manifest.json').write_bytes(b'changed')
            self.assertNotIn('nf4_b75', catalog(root, bundled_root=self.bundled)['base'])


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



if __name__=='__main__':unittest.main()

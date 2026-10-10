"""CPU integration of real server world construction and coarse admission.

Only server startup is skipped: its exact world-construction functions run
with the real WorldPipeline, inference configuration, manifest builder and
CoarsePreparation. Checkpoint loading and native raster creation are mocked.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path

import ast
from contextlib import nullcontext
from collections import OrderedDict
from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from flask import Flask, Response, has_request_context, jsonify, request

ROOT = _REPO_ROOT
sys.path.insert(0, str(ROOT / 'terrain-diffusion'))
from terrain_diffusion.inference.world_pipeline import WorldPipeline
from terrain_coarse import CoarsePreparation
from terrain_conditioning import CONDITIONING_SNR, WORLD_PROFILES
from terrain_jobs import JobCancelled, QueueFull
from terrain_generation_session import GenerationCancelled
import terrain_inference
import terrain_manifest
from terrain_generation import resolve_generation, register_generation
from terrain_geometry import profile_bounds
from terrain_climate import MODES
from terrain_orogen_layers import MODES as OROGEN_MODES
from terrain_snr_layer import MODES as SNR_MODES


class NoForwardModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, *args, **kwargs):
        raise AssertionError('Pipeline admission must not execute a network')


def native_receipt(seed, style, **options):
    receipt = {name: None for name in ('native_config', 'attempts', 'raster', 'hypsometry')}
    receipt.update(requested_seed_u64=str(seed), selected_seed_u64=str(seed),
                   selected_attempt=0, style=style, raw_height_sha256='fixture-raw',
                   height_sha256='fixture-height', sign_preserved=True,
                   generator_version='orogen-fixture', raster={'width':1024,'height':512})
    return receipt


class ServerWorldConstructionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        profile = terrain_inference.InferenceProfile(
            name='cpu-admission', cached_weights=False, cuda_graphs=False, gpu_windows=False)
        profile_data = dict(asdict(profile), version=terrain_inference.VERSION)
        reference = terrain_manifest.build_manifest(0, file_hashes=False,
                                                     inference_profile=profile_data)
        reference['files'] = {'fixture-checkpoint': {'sha256': 'c' * 64}}
        shared = WorldPipeline(seed=42, cond_snr=[.5] * 5, frequency_mult=[1.] * 5,
                               drop_water_pct=.5, dtype='bf16', log_mode='none')
        for name in ('coarse_model', 'base_model', 'decoder_model'):
            setattr(shared, name, NoForwardModel())
        self.loader = Mock(return_value=shared)
        self.app = Flask(__name__)
        self.app.testing = True
        self.namespace = dict(build_manifest=terrain_manifest.build_manifest,
            resolve_generation=resolve_generation,
            register_generation=register_generation,
            metadata=lambda seed,profile:dict(seed=str(seed),generation_profile=profile),
            generation_profile=lambda:request.args.get("world_profile","natural"),
            profile_data=profile_data, deepcopy=deepcopy, _reference_manifest=reference,
            terrestrial_file_snapshot=lambda generator='native':deepcopy(reference['files']),
            lru_cache=lru_cache, hashlib=hashlib, json=json,
            load_pipeline=self.loader, WorldPipeline=WorldPipeline, COARSE_DEVICE=0,
            runtime_profile=profile, coarse_stream_setting={'coarse':1}, OUTPUT=Path(self.temporary.name),
            WORLD_BOUNDS=(-20e6, -10e6, 20e6, 10e6), profile_bounds=profile_bounds,
            CoarsePreparation=CoarsePreparation, worlds=OrderedDict(), active_seed=None,
            all_gpu_locks=lambda:nullcontext([]), synchronize_gpus=Mock(),
            share_windows=lambda world, kind: world,
            app=self.app, jsonify=jsonify, Response=Response, request=request,
            MODES=MODES+OROGEN_MODES+SNR_MODES, SNR_MODES=SNR_MODES, OROGEN_MODES=OROGEN_MODES,
            has_request_context=has_request_context, subprocess=subprocess,
            GenerationCancelled=GenerationCancelled, jobs=SimpleNamespace(paused=False), finish_generation=lambda token:None,
            pin_cache_io=lambda key:lambda function:function, TILE=256, NATIVE=30, MIN_LOD=-2,
            JobCancelled=JobCancelled, QueueFull=QueueFull)
        # The real per-device runtime holder, with the coarse device as current.
        self.runtime = SimpleNamespace(index=0, shared_pipeline=None, worlds=self.namespace['worlds'])
        self.namespace['current_runtime'] = lambda: self.runtime
        self.namespace['physical_tile'] = lambda seed,lod,tx,ty,**options:self.namespace['get_world'](
            seed, request.args.get('world_profile', 'natural'))
        tree = ast.parse((source_path('terrain_server.py', root=ROOT)).read_text(encoding='utf-8'))
        names = {'generation_failure', 'world_manifest', '_create_world', 'get_world',
                 '_tile_coordinates', 'height_tile', 'close_world', 'display_mode', 'world_info'}
        nodes = [node for node in tree.body if getattr(node, 'name', None) in names]
        self.assertEqual({node.name for node in nodes}, names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source_path('terrain_server.py', root=ROOT)), 'exec'),
             self.namespace)
        for target, kwargs in (
            ('terrain_manifest.bootstrap_metadata', {'side_effect': native_receipt}),
            ('terrain_orogen.bootstrap_metadata', {'side_effect': native_receipt}),
            ('terrain_manifest._runtime_versions', {'return_value': {'cpu-fixture': True}}),
            ('terrain_conditioning.make_conditioning_factory', {'return_value': SimpleNamespace(
                heightmap=SimpleNamespace(metadata={'height_sha256': 'fixture-height'}))}),
            ('terrain_inference.make_synthetic_map_factory', {'return_value': Mock(name='natural-fields')}),
            ('torch.cuda.current_device', {'side_effect': AssertionError('Unexpected CUDA access')}),
            ('torch.cuda.mem_get_info', {'side_effect': AssertionError('Unexpected CUDA access')}),
            ('torch.cuda.synchronize', {'side_effect': AssertionError('Unexpected CUDA access')}),
        ):
            active = patch(target, **kwargs)
            active.start()
            self.addCleanup(active.stop)
        self.addCleanup(self.close_worlds)

    def close_worlds(self):
        for world in self.namespace['worlds'].values():
            world.close()

    def test_full_orogen_configuration_is_accepted_by_world_endpoint(self):
        settings=resolve_generation('orogen').settings
        settings['orogen_temperature_equator']=33.
        settings['orogen_biome_color_af']=[.125,.5,.875]
        encoded=json.dumps(settings)
        self.assertGreater(len(encoded),8192)
        with patch('terrain_generation.REGISTRY_ROOT',Path(self.temporary.name)/'registry'):
            response=self.app.test_client().get('/api/world',query_string=dict(
                seed='42',world_profile='orogen',generation=encoded))
            self.assertEqual(response.status_code,200,response.json)
            descriptor=resolve_generation(response.json['generation_profile'])
            self.assertEqual(descriptor.settings,settings)

    def rehash(self, manifest):
        payload = {key:value for key,value in manifest.items() if key != 'world_hash'}
        manifest['world_hash'] = hashlib.sha256(terrain_manifest._canonical(payload)).hexdigest()
        return manifest

    def test_server_get_world_installs_all_supported_profiles_with_real_pipeline(self):
        get_world = self.namespace['get_world']
        for profile in WORLD_PROFILES:
            with self.subTest(profile=profile):
                world = get_world(42, profile)
                manifest = world._terrain_manifest
                preparation = world._terrain_coarse_preparation
                self.assertIsInstance(world, WorldPipeline)
                self.assertEqual(world.device.type, 'cpu')
                self.assertEqual(world._terrain_world_profile, profile)
                self.assertEqual(manifest['ablation'], 'A0' if profile == 'natural' else profile)
                self.assertEqual(manifest['world_profile'], profile)
                self.assertEqual(manifest['conditioning']['cond_snr'],
                                 [.5] * 5 if profile == 'natural' else list(CONDITIONING_SNR))
                self.assertEqual(terrain_manifest.world_identity(manifest), preparation.world_hash)
                self.assertIs(preparation._tensor, world.coarse)
                self.assertTrue(preparation._manifest_path.is_file())
                self.assertEqual(preparation.network_windows, 0)
                self.assertIs(get_world(42, profile), world)
        self.loader.assert_called_once_with(42, device=0)

    def test_coarse_admission_rejects_other_terrestrial_profile_and_retired_labels(self):
        world = self.namespace['get_world'](42, 'terrestrial-earthlike')
        other = self.namespace['world_manifest'](42, 'terrestrial-continents')
        with self.assertRaisesRegex(ValueError, 'ablation does not match'):
            CoarsePreparation(Path(self.temporary.name) / 'other-world', other).install(world)
        manifest = deepcopy(world._terrain_manifest)
        manifest['world_profile'] = 'terrestrial-continents'
        self.rehash(manifest)
        with self.assertRaisesRegex(ValueError, 'world profile does not match'):
            CoarsePreparation(Path(self.temporary.name) / 'wrong-profile', manifest).install(world)
        for label in ('earth', 'macro-a1', 'macro-a2', 'macro-a3', 'macro-a4'):
            with self.subTest(label=label):
                world._terrain_world_profile = label
                with self.assertRaisesRegex(ValueError, 'world profile is unsupported'):
                    CoarsePreparation(Path(self.temporary.name) / 'retired', world._terrain_manifest).install(world)

    def test_coarse_admission_requires_terrestrial_profile_and_matching_height(self):
        world = self.namespace['get_world'](42, 'terrestrial-earthlike')
        manifest = world._terrain_manifest
        del world._terrain_world_profile
        with self.assertRaisesRegex(ValueError, 'pipeline requires a world profile'):
            CoarsePreparation(Path(self.temporary.name) / 'missing-profile', manifest).install(world)
        world._terrain_world_profile = 'terrestrial-earthlike'
        del world.synthetic_map_factory
        with self.assertRaisesRegex(ValueError, 'factory requires bootstrap height metadata'):
            CoarsePreparation(Path(self.temporary.name) / 'missing-factory', manifest).install(world)
        world.synthetic_map_factory = SimpleNamespace(
            heightmap=SimpleNamespace(metadata={'height_sha256': 'another-height'}))
        with self.assertRaisesRegex(ValueError, 'bootstrap height does not match'):
            CoarsePreparation(Path(self.temporary.name) / 'wrong-height', manifest).install(world)

    def test_precision_mismatch_rejected_and_upstream_none_dtype_admits_fp32(self):
        world = self.namespace['get_world'](42, 'terrestrial-earthlike')
        world._dtype = torch.float16
        with self.assertRaisesRegex(ValueError, 'precision does not match'):
            CoarsePreparation(Path(self.temporary.name) / 'wrong-precision', world._terrain_manifest).install(world)
        fp32 = WorldPipeline(seed=42, cond_snr=[.5] * 5, dtype=None, log_mode='none')
        self.addCleanup(fp32.close)
        for name in ('coarse_model', 'base_model', 'decoder_model'):
            setattr(fp32, name, NoForwardModel())
        terrain_inference.configure_world(fp32, self.namespace['runtime_profile'],
                                          world_profile='terrestrial-earthlike')
        fp32.bind()
        manifest = terrain_manifest.build_manifest(42, 'terrestrial-earthlike',
            precision='fp32', file_hashes=False, inference_profile=self.namespace['profile_data'])
        manifest.update(files=deepcopy(self.namespace['_reference_manifest']['files']), complete=True)
        self.rehash(manifest)
        preparation = CoarsePreparation(Path(self.temporary.name) / 'fp32', manifest).install(fp32)
        self.assertIsNone(fp32._dtype)
        self.assertIs(preparation._tensor, fp32.coarse)
        self.assertEqual(preparation.network_windows, 0)

    def test_actual_height_endpoint_returns_json503_for_internal_admission_errors(self):
        client = self.app.test_client()
        for failure in ('init', 'install'):
            with self.subTest(failure=failure):
                constructor = Mock()
                if failure == 'init':
                    constructor.side_effect = ValueError('fixture invalid manifest')
                else:
                    constructor.return_value.install.side_effect = ValueError('fixture wrong height')
                with patch.dict(self.namespace, CoarsePreparation=constructor):
                    response = client.get('/height/natural-v1/42/4/0/0.bin?world_profile=terrestrial-earthlike')
                    self.assertEqual(response.status_code, 503)
                    self.assertEqual(response.json['stage'], 'generation')
                    self.assertIn('Coarse world admission failed: fixture', response.json['error'])
                self.assertNotIn(('terrestrial-earthlike', 42), self.namespace['worlds'])
        bad_request = client.get('/height/natural-v1/42/12/0/0.bin')
        self.assertEqual(bad_request.status_code, 400)
        self.assertEqual(bad_request.json['error'], 'Invalid coordinates')

    def test_configured_natural_world_admits_exact_factory_and_rejects_other_settings(self):
        token=register_generation('natural', {'cond_snr':[.2,.5,.5,.5,.5],
                                               'frequency_mult':[1.7,1.,1.,1.,1.]})
        descriptor=resolve_generation(token)
        factory=SimpleNamespace(_terrain_generation_profile=token,
                                generation_settings=descriptor.settings)
        with patch('terrain_conditioning.make_conditioning_factory',return_value=factory):
            world=self.namespace['get_world'](42,token)
        self.assertEqual(world.kwargs['cond_snr'],descriptor.settings['cond_snr'])
        self.assertEqual(world.kwargs['frequency_mult'],descriptor.settings['frequency_mult'])
        self.assertEqual(world._terrain_manifest['conditioning']['generation_settings'],descriptor.settings)
        self.assertIsNone(world._terrain_manifest['bootstrap'])
        self.assertNotEqual(world._terrain_manifest['world_hash'],
                            self.namespace['world_manifest'](42,'natural')['world_hash'])
        factory.generation_settings={**descriptor.settings,'cond_snr':[.3]*5}
        with self.assertRaisesRegex(ValueError,'settings do not match conditioning factory'):
            CoarsePreparation(Path(self.temporary.name)/'wrong-settings',world._terrain_manifest).install(world)
        factory.generation_settings=descriptor.settings
        factory._terrain_generation_profile='natural'
        with self.assertRaisesRegex(ValueError,'factory does not match pipeline settings'):
            CoarsePreparation(Path(self.temporary.name)/'wrong-factory',world._terrain_manifest).install(world)

    def test_configuring_back_to_natural_restores_checkpoint_parameters(self):
        token=register_generation('natural', {'cond_snr':[.2]*5,
            'frequency_mult':[2.]*5,'drop_water_pct':.6})
        descriptor=resolve_generation(token)
        with patch('terrain_conditioning.make_conditioning_factory',return_value=SimpleNamespace(
                _terrain_generation_profile=token,generation_settings=descriptor.settings)):
            world=self.namespace['get_world'](42,token)
        terrain_inference.configure_world(world,self.namespace['runtime_profile'],world_profile='natural')
        self.assertEqual(world.kwargs['cond_snr'],[.5]*5)
        self.assertEqual(world.kwargs['frequency_mult'],[1.]*5)
        self.assertEqual(world.kwargs['drop_water_pct'],.5)


if __name__ == '__main__':
    unittest.main()

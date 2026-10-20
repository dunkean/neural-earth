import base64
import json
from pathlib import Path
import tempfile
import unittest
import zlib
from types import SimpleNamespace
from unittest import mock

from flask import Flask
from terrain_share import ShareStore, register_share_routes, MAX_BYTES
from terrain_share_stages import restore_stages


class ShareTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.app = Flask(__name__)
        register_share_routes(self.app, self.root)
        self.client = self.app.test_client()
        self.generation = dict(seed='18446744073709551615', profile='natural',
                               settings={'cond_snr': [0.001, 1, 1, 1, 1]}, worldIdentity='original')
        self.rendering = dict(materials={'snow_color': [.94, .96, .97]}, gpuRender=False)

    def tearDown(self):
        self.directory.cleanup()

    def save(self):
        response = self.client.post('/api/share', json=dict(generation=self.generation, rendering=self.rendering))
        self.assertEqual(response.status_code, 200)
        return response.json

    def test_deterministic_generation_uid_independent_of_rendering_and_restart(self):
        first = self.save()
        self.assertEqual(len(first['generation']), 19)
        self.rendering['gpuRender'] = True
        second = self.save()
        self.assertEqual(first['generation'], second['generation'])
        self.assertNotEqual(first['rendering'], second['rendering'])
        self.assertEqual(ShareStore(self.root).load(first['generation'], 'g')['seed'], self.generation['seed'])
        self.generation['settings']['cond_snr'][0] = .002
        self.assertNotEqual(first['generation'], self.save()['generation'])

    def test_portable_recipe_imports_into_an_empty_installation(self):
        saved = self.save()
        other = Flask('other')
        register_share_routes(other, self.root / 'other')
        client = other.test_client()
        path = '/api/share/' + saved['generation'] + '/' + saved['rendering']
        self.assertEqual(client.get(path).status_code, 404)
        response = client.post('/api/share/import', json=saved)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json['generation']['settings'], self.generation['settings'])
        self.assertEqual(response.json['rendering'], self.rendering)
        self.assertEqual(client.get(path).json, response.json)

    def test_damage_unknown_uid_and_invalid_seed_are_errors(self):
        saved = self.save()
        path = self.root / (saved['generation'] + '.json')
        path.write_text('{}')
        self.assertEqual(self.client.get('/api/share/' + saved['generation'] + '/' + saved['rendering']).status_code, 400)
        self.assertEqual(self.client.get('/api/share/g' + 'a' * 18 + '/r' + 'a' * 18).status_code, 404)
        saved['generation'] = 'g' + 'a' * 18
        self.assertEqual(self.client.post('/api/share/import', json=saved).status_code, 400)
        self.generation['seed'] = str(2**64)
        self.assertEqual(self.client.post('/api/share', json=dict(generation=self.generation, rendering=self.rendering)).status_code, 400)

    def test_decompression_limit(self):
        token = base64.urlsafe_b64encode(zlib.compress(b' ' * (MAX_BYTES + 1))).decode().rstrip('=')
        response = self.client.post('/api/share/import', json=dict(recipe=token, generation='g'+'a'*18, rendering='r'+'a'*18))
        self.assertEqual(response.status_code, 400)

    def test_retained_stages_rebuild_older_dependencies_and_remap_local_ids(self):
        import threading
        from terrain_generation import _defaults
        # A retained climate uses old erosion, while the current height uses new erosion.
        groups = dict(relief={'orogen_detail'}, erosion={'orogen_hydraulic'}, climate={'orogen_temperature_offset'})
        nodes = {
            'old-relief': dict(stage='relief', parent=None, settings={'orogen_detail': 20000}),
            'old-erosion': dict(stage='erosion', parent='old-relief', settings={'orogen_hydraulic': .25}),
            'old-climate': dict(stage='climate', parent='old-erosion', settings={'orogen_temperature_offset': 2}),
            'new-erosion': dict(stage='erosion', parent='old-relief', settings={'orogen_hydraulic': .75}),
        }
        for node in nodes.values():
            node.update(seed=42, width=2048, height=1024, style='continents' if node['stage']=='relief' else None)
        settings = _defaults('orogen')
        settings.update(orogen_detail=90000, orogen_relief_stage='old-relief',
                        orogen_erosion_stage='new-erosion', orogen_climate_stage='old-climate')
        calls = []

        def build(stage, seed, style, config, width, height, parent, relief):
            calls.append((stage, config, parent, relief))
            return dict(id='local-' + str(len(calls))), True

        core = SimpleNamespace(WIDTH=2048, HEIGHT=1024, _LOCK=threading.RLock(), style_config=lambda style, options: options)
        stages = SimpleNamespace(STAGE_KEYS=groups, build_stage=build, load_stage=mock.Mock(side_effect=ValueError('missing')))
        with mock.patch.dict('sys.modules', terrain_orogen=core, terrain_orogen_stages=stages):
            restored = restore_stages(dict(seed='42', profile='orogen', settings=settings, stages=nodes))
        self.assertEqual([c[0] for c in calls], ['relief', 'erosion', 'erosion', 'climate'])
        self.assertEqual(calls[0][1]['detail'], 20000, 'old retained relief settings win over current settings')
        self.assertEqual(calls[1][1]['hydraulic'], .75)
        self.assertEqual(calls[2][1]['hydraulic'], .25)
        self.assertEqual(calls[3][2]['id'], 'local-3', 'climate uses its older erosion parent')
        self.assertEqual(calls[3][3]['id'], 'local-1', 'climate uses its original relief')
        self.assertEqual(restored['settings']['orogen_erosion_stage'], 'local-2')
        self.assertEqual(restored['settings']['orogen_climate_stage'], 'local-4')
        self.assertEqual(settings['orogen_erosion_stage'], 'new-erosion', 'original portable recipe is unchanged')


if __name__ == '__main__':
    unittest.main()

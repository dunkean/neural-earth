"""PNG tiles must cover their world bounds without displaying their halo."""
import ast
from contextlib import nullcontext
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from flask import Flask, Response, has_request_context, jsonify, request, send_file

from terrain_climate import MODES as CLIMATE_MODES, colorize
from terrain_jobs import JobCancelled, QueueFull
from terrain_orogen_layers import EXTRA, MODES, render, sample
from terrain_snr_layer import MODES as SNR_MODES
from terrain_priority import tile_relief_stats

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'terrain-diffusion'))
from terrain_diffusion.inference.relief_map import get_relief_map


class BiomeTileRenderingTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.app = Flask(__name__)
        self.app.testing = True
        self.climate = np.zeros((5, 33, 33), np.float32)
        self.climate[0] = 20
        self.climate[2] = 2000
        self.climate[4] = -.0065
        self.elevations = {lod: np.full((304, 304), 100, np.float32)
                           for lod in (9, 4, 0, -1)}
        # Learned terrain changes a coast and adds a snowy peak in the same tile.
        for lod in (4, 0, -1):
            self.elevations[lod][:, :100] = -1000
            self.elevations[lod][:, 200:] = 5000
        self.physical = Mock(side_effect=self.physical_tile)
        self.diagnostic = Mock(side_effect=AssertionError('Biomes must use current terrain'))
        self.snr_diagnostic = Mock(return_value=Response(b'SNR', mimetype='image/png'))
        self.namespace = dict(app=self.app, request=request, has_request_context=has_request_context,
            jsonify=jsonify, Response=Response, send_file=send_file,
            np=np, io=io, json=json, Image=Image, time=time, tile_relief_stats=tile_relief_stats,
            TILE=256, HALO=24, NATIVE=30, MIN_LOD=-3, CLIMATE_SIZE=33,
            WORLD_BOUNDS=(-20e6, -10e6, 20e6, 10e6), PROFILE='fixture',
            MODES=CLIMATE_MODES+MODES+SNR_MODES, SNR_MODES=SNR_MODES, OROGEN_MODES=MODES, CACHE=Path(temporary.name),
            resolve_generation=lambda wp: SimpleNamespace(bootstrap_generator='orogen',
                settings={'climate_source':'orogen'}),
            generation_profile=lambda: 'orogen',
            pin_cache_io=lambda key: lambda function: function,
            orogen_diagnostic_tile=self.diagnostic, snr_diagnostic_tile=self.snr_diagnostic, physical_tile=self.physical,
            selected_coarse_interpolation=lambda lod: None,
            latent_preview_source=lambda lod: None, _coarse_cache_level=lambda lod, _: lod,
            _atomic_bytes=lambda path, data: path.write_bytes(data),
            _record_tile_disk=lambda *args: None, image_slots=threading.BoundedSemaphore(2),
            span=lambda *args, **kwargs: nullcontext(),
            get_relief_map=get_relief_map, colorize=colorize,
            JobCancelled=JobCancelled, QueueFull=QueueFull)
        tree = ast.parse((ROOT / 'terrain_server.py').read_text(encoding='utf-8'))
        names = {'display_mode', '_tile_coordinates', 'height_tile', 'tile',
                 'render_elevation', '_tile_headers'}
        nodes = [node for node in tree.body if getattr(node, 'name', None) in names]
        self.assertEqual({node.name for node in nodes}, names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]),
                     str(ROOT / 'terrain_server.py'), 'exec'), self.namespace)

    def physical_tile(self, seed, lod, tx, ty, **options):
        stage = {9:'conditioning-preview', 4:'coarse', 0:'decoder', -1:'decoder-refinement'}[lod]
        report = dict(stage=stage, resolution=30*2**lod, seconds=0, width=304, halo=24)
        return Path('unused.npy'), report, False, (self.elevations[lod], self.climate)

    def test_snr_modes_route_to_policy_without_terrain_inference(self):
        for mode in SNR_MODES:
            response = self.app.test_client().get(f'/tiles/natural-v1/42/4/-1/2.png?profile=fixture&mode={mode}')
            self.assertEqual(response.status_code, 200)
            self.snr_diagnostic.assert_called_with(42, 4, -1, 2, 'orogen', mode)
        self.physical.assert_not_called()
        self.diagnostic.assert_not_called()
        response = self.app.test_client().get('/height/natural-v1/42/4/0/0.bin?mode=snr-elevation')
        self.assertEqual(response.status_code, 400)

    def test_orogen_biomes_follow_preview_coarse_decoder_and_refinement_heights(self):
        images = {}
        client = self.app.test_client()
        for lod, stage in ((9, 'conditioning-preview'), (4, 'coarse'),
                           (0, 'decoder'), (-1, 'decoder-refinement')):
            with self.subTest(lod=lod):
                response = client.get(f'/tiles/natural-v1/42/{lod}/0/0.png?profile=fixture&mode=biomes')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers['X-Terrain-Stage'], stage)
                self.assertEqual(float(response.headers['X-Terrain-Elevation-Max']),
                                 100 if lod == 9 else 5000)
                with Image.open(io.BytesIO(response.data)) as image:
                    self.assertEqual(image.size, (256, 256))
                    images[lod] = np.array(image)
                response.close()
        lowland = images[9][128, 128]
        self.assertGreater(lowland[1], lowland[2])
        for lod in (4, 0, -1):
            self.assertGreater(images[lod][128, 20, 2], images[lod][128, 20, 1], 'New coast is ocean')
            self.assertGreater(images[lod][128, 230].min(), 200, 'New high peak is snowy')
        self.assertEqual([call.args[1] for call in self.physical.call_args_list], [9, 4, 0, -1])
        self.diagnostic.assert_not_called()

    def test_gpu_biomes_receive_the_current_height_and_climate(self):
        client = self.app.test_client()
        for lod in (9, 4, 0, -1):
            with self.subTest(lod=lod):
                response = client.get(f'/height/natural-v1/42/{lod}/0/0.bin?profile=fixture&mode=biomes&climate=1')
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers['X-Terrain-Climate-Width'], '33')
                values = np.frombuffer(response.data, dtype='<f4')
                np.testing.assert_array_equal(values[:304*304], self.elevations[lod].ravel())
                np.testing.assert_array_equal(values[304*304:], self.climate.ravel())
        self.diagnostic.assert_not_called()


class DiagnosticTileRenderingTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.app.testing = True
        yy, xx = np.mgrid[:32, :64]
        ramp = (xx + yy) / 96.
        self.atlas = SimpleNamespace(bounds=(-20e6, -10e6, 20e6, 10e6),
                                     width=64, height=32)
        layers = dict(plates=(xx // 8 + yy // 8 * 8).astype(np.uint16),
                      crust=(xx % 9 < 4).astype(np.uint8),
                      boundaries=(xx // 4 % 4).astype(np.uint8),
                      convergence=ramp * 4 - 2, uplift=ramp / 2)
        for field, _ in EXTRA.values():
            layers.setdefault(field, ramp)
        layers['superPlates'] = xx // 16
        for channel in range(3):
            layers[f'biome_{channel}'] = np.roll(ramp, channel * 5, axis=1)
            layers[f'koppen_color_{channel}'] = (xx // (channel + 2) % 3) / 2.
        for prefix in ('wind', 'ocean_current'):
            for season in ('summer', 'winter'):
                layers[f'{prefix}_east_{season}'] = ramp - .5
                layers[f'{prefix}_north_{season}'] = np.flip(ramp, axis=1) - .5
        layers['height'] = ramp * 5000 - 500
        self.atlas.layers = layers
        self.atlas.sample_height_m = lambda xs, ys: sample(
            self.atlas, layers['height'], xs, ys)
        self.namespace = dict(request=request, PROFILE='fixture', TILE=256, HALO=24,
            NATIVE=30, WORLD_BOUNDS=self.atlas.bounds, Response=Response,
            np=np, io=io, Image=Image, time=time, tile_relief_stats=tile_relief_stats,
            world_manifest=lambda seed, wp: {}, world_identity=lambda manifest: 'fixture',
            resolve_generation=lambda wp: SimpleNamespace(
                bootstrap_style='earthlike', bootstrap_options={}),
            render_orogen_layer=render)
        tree = ast.parse((ROOT / 'terrain_server.py').read_text(encoding='utf-8'))
        names = {'orogen_diagnostic_tile', '_tile_headers'}
        nodes = [node for node in tree.body if getattr(node, 'name', None) in names]
        self.assertEqual({node.name for node in nodes}, names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]),
                     str(ROOT / 'terrain_server.py'), 'exec'), self.namespace)
        provider = SimpleNamespace(get_heightmap=lambda *args, **kwargs: self.atlas)
        active = patch.dict(sys.modules, terrain_orogen=provider)
        active.start()
        self.addCleanup(active.stop)

    def tile(self, mode, lod=9, tx=-1, ty=-1, binary=False):
        with self.app.test_request_context('/?profile=fixture&world_identity=fixture'):
            response = self.namespace['orogen_diagnostic_tile'](
                42, lod, tx, ty, 'orogen', mode, binary=binary)
        self.assertEqual(response.status_code, 200)
        if binary:
            return response, np.frombuffer(response.data, dtype='<f4').reshape(304, 304)
        self.assertEqual(response.mimetype, 'image/png')
        with Image.open(io.BytesIO(response.data)) as image:
            self.assertEqual(image.size, (256, 256))
            return response, np.array(image)

    def test_all_diagnostic_pngs_exclude_halo_at_positive_and_negative_coordinates(self):
        for mode in MODES:
            for lod, tx, ty in ((9, -1, -1), (4, 2, 1)):
                with self.subTest(mode=mode, lod=lod, tx=tx, ty=ty):
                    _, actual = self.tile(mode, lod, tx, ty)
                    resolution = 30 * 2**lod
                    xs = (tx * 256 + np.arange(-24, 280) + .5) * resolution
                    ys = (ty * 256 + np.arange(-24, 280) + .5) * resolution
                    expected = render(self.atlas, mode, xs, ys)[24:280, 24:280]
                    expected = (np.clip(expected, 0, 1) * 255).astype(np.uint8)
                    np.testing.assert_array_equal(actual, expected)

    def test_adjacent_tiles_match_one_continuous_render_in_both_axes(self):
        for mode in MODES:
            if mode == 'orogen-height':
                continue  # Relief shading uses a halo; covered by the test above.
            for axis in (0, 1):
                with self.subTest(mode=mode, axis=axis):
                    _, first = self.tile(mode)
                    _, second = self.tile(mode, tx=0 if axis == 1 else -1,
                                          ty=0 if axis == 0 else -1)
                    actual = np.concatenate((first, second), axis=axis)
                    xs = (-256 + np.arange(512 if axis == 1 else 256) + .5) * 15360
                    ys = (-256 + np.arange(512 if axis == 0 else 256) + .5) * 15360
                    expected = (np.clip(render(self.atlas, mode, xs, ys), 0, 1) * 255).astype(np.uint8)
                    np.testing.assert_array_equal(actual, expected)

    def test_binary_height_keeps_physical_halo(self):
        response, actual = self.tile('orogen-height', binary=True)
        self.assertEqual(response.mimetype, 'application/octet-stream')
        self.assertEqual(response.headers['X-Terrain-Width'], '304')
        self.assertEqual(response.headers['X-Terrain-Halo'], '24')
        coords = (-256 + np.arange(-24, 280) + .5) * 15360
        np.testing.assert_array_equal(actual, self.atlas.sample_height_m(coords, coords))


if __name__ == '__main__':
    unittest.main()

"""CPU contracts for real InfiniteTensor fusion, scheduling and coarse replay."""
from pathlib import Path
import hashlib
import json
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent / 'terrain-diffusion'))
from infinite_tensor import InfiniteTensor, MemoryTileStore, TensorWindow
from terrain_diffusion.inference.world_pipeline import linear_weight_window

from terrain_coarse import CoarsePreparation
from terrain_window_scheduler import WorldWindowScheduler, read_rect, scheduler_status
from terrain_inference import choose_profile


def _manifest(seed):
    payload = {'seed_u64': str(seed), 'ablation': 'A0',
               'conditioning': {'cond_snr': [.5]*5},
               'generation': {'precision': 'fp32'},
               'complete': True, 'files': {'mock-checkpoint': {'sha256': 'c'*64}}}
    payload['world_hash'] = hashlib.sha256(json.dumps(
        payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return payload


def _preparation(*args, **kwargs):
    # These tests inspect committed files immediately; async contracts have
    # dedicated concurrency tests in test_terrain_coarse_async.py.
    return CoarsePreparation(*args, async_persistence=False, **kwargs)


def _world(cache_bytes=2000, *, base_batch=2, batch_sensitive=False):
    store = MemoryTileStore(cache_size_bytes=cache_bytes)
    calls = {'coarse': [], 'base0': [], 'base1': [], 'decoder': []}
    coarse_window = TensorWindow(size=(7, 4, 4), stride=(7, 2, 2))
    latent_window = TensorWindow(size=(6, 4, 4), stride=(6, 2, 2))
    decoder_window = TensorWindow(size=(2, 4, 4), stride=(2, 2, 2))

    def coarse_f(indices):
        calls['coarse'].extend(indices)
        result = []
        weight = linear_weight_window(4, 'cpu', torch.float32)
        for _, i, j in indices:
            value = torch.ones((7, 4, 4), dtype=torch.float32)
            value[:6] *= (2 * i + 3 * j + 4) * weight
            value[-1] = weight
            result.append(value)
        return result

    coarse = InfiniteTensor((7, None, None), coarse_f, coarse_window,
                            batch_size=1, tile_store=store, tensor_id='coarse')

    def base_f(name):
        def f(indices, inputs):
            calls[name].extend(indices)
            result = []
            for index, value in zip(indices, inputs):
                average = (value[0] / value[-1]).mean().item()
                output = torch.ones((6, 4, 4), dtype=torch.float32)
                output[:5] *= average + index[1] / 10 + index[2] / 20 + (
                    len(indices) * .001 if batch_sensitive else 0)
                result.append(output)
            return result
        return f

    base0 = InfiniteTensor((6, None, None), base_f('base0'), latent_window,
                           args=(coarse,), args_windows=(TensorWindow(
                               size=(7, 4, 4), stride=(7, 2, 2)),),
                           batch_size=base_batch, tile_store=store, tensor_id='base0')
    base1 = InfiniteTensor((6, None, None), base_f('base1'), latent_window,
                           args=(base0,), args_windows=(latent_window,),
                           batch_size=base_batch, tile_store=store, tensor_id='base1')

    def decoder_f(indices, inputs):
        calls['decoder'].extend(indices)
        result = []
        for value in inputs:
            output = torch.ones((2, 4, 4), dtype=torch.float32)
            output[0] = (value[0] / value[-1]).mean()
            result.append(output)
        return result

    decoder = InfiniteTensor((2, None, None), decoder_f, decoder_window,
                             args=(base1,), args_windows=(latent_window,),
                             batch_size=1, tile_store=store, tensor_id='decoder')
    world = SimpleNamespace(coarse=coarse, latents=base1, residual=decoder, calls=calls)
    world._terrain_window_scheduler = WorldWindowScheduler(world)
    return world


class WindowSchedulerTests(unittest.TestCase):
    def test_profile_batch_does_not_follow_free_vram(self):
        with patch.dict(os.environ, {'TERRAIN_CANONICAL_LATENTS': '0'}, clear=True), \
             patch('torch.cuda.get_device_capability', return_value=(8, 6)), \
             patch('torch.cuda.mem_get_info', side_effect=[(4*1024**3, 24*1024**3),
                                                        (20*1024**3, 24*1024**3)]):
            self.assertEqual(choose_profile(0).latent_batch, 16)
            self.assertEqual(choose_profile(0).latent_batch, 16)
        with patch.dict(os.environ, {'TERRAIN_CANONICAL_LATENTS': '1'}, clear=True), \
             patch('torch.cuda.get_device_capability', return_value=(8, 6)), \
             patch('torch.cuda.mem_get_info', return_value=(4*1024**3, 24*1024**3)):
            profile = choose_profile(0)
            self.assertEqual(profile.latent_batch, 1)
            self.assertTrue(profile.canonical_latents)

    def test_scalar_base_batch_is_order_independent_with_shape_sensitive_model(self):
        whole = _world(base_batch=1, batch_sensitive=True)
        expected = read_rect(whole, 'decoder', -2, -2, 8, 8)
        split = _world(base_batch=1, batch_sensitive=True)
        output = torch.empty_like(expected)
        pieces = [(-2, -2, 3, 2), (-2, 2, 3, 8), (3, -2, 8, 2), (3, 2, 8, 8)]
        for k in (3, 1, 0, 2):
            i1, j1, i2, j2 = pieces[k]
            output[:, i1+2:i2+2, j1+2:j2+2] = read_rect(split, 'decoder', i1, j1, i2, j2)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        torch.testing.assert_close(read_rect(split, 'decoder', -2, -2, 8, 8), expected,
                                   rtol=0, atol=0)

    def test_full_crop_subcrops_permutations_and_eviction(self):
        whole = _world()
        expected = read_rect(whole, 'decoder', -2, -2, 8, 8)
        sliced = _world()
        pieces = [(slice(-2, 3), slice(-2, 2)), (slice(-2, 3), slice(2, 8)),
                  (slice(3, 8), slice(-2, 2)), (slice(3, 8), slice(2, 8))]
        result = torch.empty_like(expected)
        for k in (3, 0, 2, 1):
            row, col = pieces[k]
            patch = read_rect(sliced, 'decoder', row.start, col.start, row.stop, col.stop)
            result[:, row.start+2:row.stop+2, col.start+2:col.stop+2] = patch
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        # A tiny LRU evicts outputs between requests. Regeneration remains the
        # same canonical function of global window coordinates.
        again = read_rect(sliced, 'decoder', -2, -2, 8, 8)
        torch.testing.assert_close(again, expected, rtol=0, atol=0)
        metrics = scheduler_status(sliced)
        self.assertGreater(metrics['decoder']['generated'], 0)
        self.assertGreater(metrics['decoder']['regenerated_estimate'], 0)
        self.assertGreater(metrics['coarse']['unique_windows_estimate'], 0)
        self.assertEqual(metrics['coarse']['metadata_history_bytes'], 1 << 20)

    def test_cancel_between_quanta_and_resume(self):
        world = _world(cache_bytes=10_000)
        checks = 0
        def cancel():
            nonlocal checks
            checks += 1
            if checks == 3:
                raise RuntimeError('obsolete request')
        with self.assertRaisesRegex(RuntimeError, 'obsolete'):
            read_rect(world, 'coarse', 0, 0, 8, 8, check=cancel)
        self.assertGreater(len(world.calls['coarse']), 0)
        done_before = len(world.calls['coarse'])
        resumed = read_rect(world, 'coarse', 0, 0, 8, 8)
        reference = read_rect(_world(), 'coarse', 0, 0, 8, 8)
        torch.testing.assert_close(resumed, reference, rtol=0, atol=0)
        self.assertGreater(len(world.calls['coarse']), done_before)

    def test_complete_fusion_has_positive_weights(self):
        world = _world()
        fused = read_rect(world, 'coarse', -3, -3, 7, 7)
        self.assertTrue(torch.all(fused[-1] > 0))
        # Each pixel includes all its contributors, including negative-index
        # windows; a split along an overlap must keep the exact same sum.
        left = read_rect(world, 'coarse', -3, -3, 7, 2)
        right = read_rect(world, 'coarse', -3, 2, 7, 7)
        torch.testing.assert_close(torch.cat([left, right], dim=2), fused, rtol=0, atol=0)

    def test_public_stage_read_has_output_admission_limit(self):
        with self.assertRaisesRegex(ValueError, 'output budget'):
            read_rect(_world(), 'coarse', 0, 0, 10_000, 10_000)


class CoarsePreparationTests(unittest.TestCase):
    def test_live_preparations_share_coverage_and_corruption(self):
        bounds=(0,0,7680,7680)
        manifest=_manifest(31)
        with tempfile.TemporaryDirectory() as temporary:
            foreground,background=_world(),_world()
            a=_preparation(temporary,manifest,bounds=bounds).install(foreground)
            b=_preparation(temporary,manifest,bounds=bounds).install(background)
            self.assertIs(a._persisted_indices,b._persisted_indices)
            self.assertEqual(a.status()['complete_windows'],0)
            a.step(foreground,budget_windows=a.status()['total_windows'])
            self.assertEqual(b.status()['complete_windows'],b.status()['total_windows'])
            b.step(background,budget_windows=1)
            self.assertEqual(b.network_windows,0)
            self.assertEqual(background.calls['coarse'],[])
            index=next(iter(a._required_indices(0,0,1,1)))
            self.assertTrue(b.complete(0,0,1,1))
            a._window_path(index).write_bytes(b'')
            self.assertFalse(b.complete(0,0,1,1))
            self.assertEqual(a.status()['complete_windows'],a.status()['total_windows']-1)
            self.assertEqual(a.status()['disk_bytes'],
                             sum(p.stat().st_size for p in a.windows_dir.glob('*.npy')))

    def test_live_preparations_share_namespace_quota(self):
        bounds=(0,0,7680,7680)
        manifest=_manifest(32)
        with tempfile.TemporaryDirectory() as temporary:
            foreground,background=_world(),_world()
            budget=2*576  # Two 7x4x4 float32 .npy payloads.
            a=_preparation(temporary,manifest,bounds=bounds,
                                budget_bytes=budget).install(foreground)
            b=_preparation(temporary,manifest,bounds=bounds,
                                budget_bytes=budget).install(background)
            a.step(foreground,budget_windows=2)
            self.assertEqual(a.status()['disk_bytes'],budget)
            self.assertEqual(b.status()['complete_windows'],2)
            b.step(background,budget_windows=1)
            self.assertTrue(a.status()['disk_budget_exhausted'])
            self.assertTrue(b.status()['disk_budget_exhausted'])
            self.assertEqual(sum(p.stat().st_size for p in a.windows_dir.glob('*.npy')),budget)
            self.assertEqual(b.status()['disk_bytes'],budget)

    def test_checkpoint_window_geometry_uses_exact_linear_weight_taper(self):
        store = MemoryTileStore(cache_size_bytes=2_000_000)
        weight = linear_weight_window(64, 'cpu', torch.float32)
        def model(indices):
            return [torch.cat([weight[None].expand(6,-1,-1), weight[None]])
                    for _ in indices]
        coarse = InfiniteTensor((7,None,None), model,
                                TensorWindow(size=(7,64,64), stride=(7,48,48)),
                                batch_size=1, tile_store=store, tensor_id='coarse')
        world = SimpleNamespace(coarse=coarse, latents=coarse, residual=coarse)
        world._terrain_window_scheduler = WorldWindowScheduler(world)
        with tempfile.TemporaryDirectory() as temporary:
            prep = _preparation(temporary, _manifest(12),
                                     bounds=(0,0,7680,7680)).install(world)
            prep.step(world, budget_windows=1)
            index = next(iter(prep._persisted_indices))
            saved = np.load(prep._window_path(index), allow_pickle=False)
            np.testing.assert_array_equal(saved[-1], weight.numpy())
            self.assertEqual(prep._window_path(index).stat().st_size, 114816)

    def test_budget_resume_disk_reuse_and_mip_provenance(self):
        manifest = _manifest(11)
        bounds = (0, 0, 6 * 7680, 6 * 7680)
        with tempfile.TemporaryDirectory() as temporary:
            world = _world()
            prep = _preparation(temporary, manifest, bounds=bounds).install(world)
            first = prep.step(world, budget_windows=1)
            self.assertEqual(first['network_windows'], 1)
            self.assertEqual(first['complete_windows'], 1)
            self.assertIsNone(prep.read_mip(world, 0, 0, 0, 4, 4))
            self.assertFalse(prep.ready_for_samples(np.array([128., 640.]), np.array([128., 640.])))
            while prep.status()['complete_windows'] < prep.status()['total_windows']:
                prep.step(world, budget_windows=3)
            self.assertEqual(prep.status()['coverage'], 1)
            self.assertTrue(prep.ready_for_samples(np.array([128., 640.]), np.array([128., 640.])))
            # The server clips LOD 7–12 tile samples to finite world bounds
            # before readiness and field sampling. The prepared eight-cell
            # apron must then cover even tiles straddling a world edge.
            for lod in (7, 12):
                step = 1 << lod
                tile = (np.arange(-24, 280) + .5) * step
                clipped = np.clip(tile, 0, 6*256)
                self.assertTrue(prep.ready_for_samples(clipped, clipped), f'LOD {lod}')
            mip = prep.read_mip(world, 1, 0, 0, 3, 3)
            self.assertEqual(mip['data'].shape, (3, 3))
            self.assertEqual(mip['source'], 'learned-coarse')
            self.assertFalse(mip['final_dem_mip'])
            self.assertTrue(np.isfinite(mip['data']).all())
            # A new pipeline and RAM store rehydrate the same weighted windows
            # without calling the network.
            reopened = _world()
            again = _preparation(temporary, manifest, bounds=bounds).install(reopened)
            replay = again.read_mip(reopened, 1, 0, 0, 3, 3)
            np.testing.assert_array_equal(replay['data'], mip['data'])
            self.assertEqual(again.network_windows, 0)
            self.assertGreater(again.disk_hits, 0)
            required = next(iter(again._required_indices(0, 0, 6, 6)))
            for damaged in (b'', b'\x93NUMPY\x01'):
                again._window_path(required).write_bytes(damaged)
                reopened.coarse.clear_cache()
                self.assertFalse(again.complete(0, 0, 6, 6))
                self.assertIsNone(again.read_mip(reopened, 1, 0, 0, 3, 3))
                replay = again.read_mip(reopened, 1, 0, 0, 3, 3, generate=True)
                self.assertTrue(np.isfinite(replay['data']).all())
            self.assertGreaterEqual(again.network_windows, 2)

    def test_shape_correct_weight_corruption_rejected_before_ready(self):
        manifest = _manifest(4)
        bounds = (0, 0, 6 * 7680, 6 * 7680)
        with tempfile.TemporaryDirectory() as temporary:
            world = _world()
            prep = _preparation(temporary, manifest, bounds=bounds).install(world)
            prep.read_mip(world, 0, 0, 0, 2, 2, generate=True)
            index = next(iter(prep._required_indices(0, 0, 2, 2)))
            path = prep._window_path(index)
            sidecar = prep._sidecar_path(index)
            original = np.load(path, allow_pickle=False)
            for weight in (np.zeros_like(original[-1]), original[-1] + np.float32(.1)):
                corrupted = original.copy()
                corrupted[-1] = weight
                with path.open('wb') as stream:
                    np.save(stream, corrupted, allow_pickle=False)
                # Even if the ordinary payload metadata is updated, the
                # canonical weight channel must fail the cheap readiness gate.
                metadata = json.loads(sidecar.read_text())
                payload = path.read_bytes()
                metadata['bytes'] = len(payload)
                metadata['mtime_ns'] = path.stat().st_mtime_ns
                metadata['sha256'] = hashlib.sha256(payload).hexdigest()
                sidecar.write_text(json.dumps(metadata))
                world.coarse.clear_cache()
                self.assertFalse(prep.complete(0, 0, 2, 2))
                self.assertIsNone(prep.read_mip(world, 0, 0, 0, 2, 2))
                prep.read_mip(world, 0, 0, 0, 2, 2, generate=True)
                np.testing.assert_array_equal(np.load(path)[-1], original[-1])

    def test_identity_collision_rejected(self):
        bounds = (0, 0, 2 * 7680, 2 * 7680)
        with tempfile.TemporaryDirectory() as temporary:
            world = _world()
            manifest = _manifest(1)
            _preparation(temporary, manifest, bounds=bounds).install(world)
            with self.assertRaisesRegex(ValueError, 'geometry'):
                _preparation(temporary, manifest,
                                  bounds=(0, 0, 3 * 7680, 3 * 7680)).install(_world())
            wrong_seed = _world()
            wrong_seed.seed = 2
            with self.assertRaisesRegex(ValueError, 'seed'):
                _preparation(temporary, manifest, bounds=bounds).install(wrong_seed)
            wrong_snr = _world()
            wrong_snr.seed = 1
            wrong_snr.kwargs = {'cond_snr': [.1]*5}
            with self.assertRaisesRegex(ValueError, 'SNR'):
                _preparation(temporary, manifest, bounds=bounds).install(wrong_snr)

    def test_disk_budget_stops_background_without_breaking_demand(self):
        with tempfile.TemporaryDirectory() as temporary:
            world = _world()
            prep = _preparation(temporary, _manifest(8),
                                     bounds=(0, 0, 2*7680, 2*7680),
                                     budget_bytes=1).install(world)
            output = read_rect(world, 'coarse', 0, 0, 2, 2)
            self.assertTrue(torch.isfinite(output).all())
            self.assertTrue(prep.status()['disk_budget_exhausted'])
            self.assertEqual(prep.status()['complete_windows'], 0)
            before = len(world.calls['coarse'])
            prep.step(world, budget_windows=1)
            self.assertEqual(len(world.calls['coarse']), before)

    def test_background_repairs_window_already_in_ram(self):
        with tempfile.TemporaryDirectory() as temporary:
            world = _world(cache_bytes=100_000)
            # Simulate a read before persistence was installed.
            window_index = (0, -4, -4)
            world._terrain_window_scheduler.ensure_window(world.coarse, window_index)
            prep = _preparation(temporary, _manifest(9),
                                     bounds=(0, 0, 2*7680, 2*7680)).install(world)
            self.assertIn(window_index, prep._planned_indices)
            prep._cursor = prep._indices.index(window_index)
            prep.step(world, budget_windows=1)
            self.assertTrue(prep._valid_metadata(window_index))


if __name__ == '__main__':
    unittest.main()

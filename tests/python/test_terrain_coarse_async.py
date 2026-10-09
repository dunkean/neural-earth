"""CPU-only persistence races, immutable snapshots and failure contracts."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np
import torch

from test_terrain_windows import _manifest, _world
from terrain_coarse import CoarsePreparation, CoarsePersistenceError
from terrain_window_scheduler import read_rect


class AsyncCoarseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.preparations = []
        self.releases = []
        self.cuda_guard = patch('torch.cuda.synchronize', side_effect=AssertionError('CPU test called CUDA'))
        self.cuda_guard.start()

    def tearDown(self):
        for event in self.releases:
            event.set()
        for preparation in self.preparations:
            try:
                preparation.close(timeout=5)
            except CoarsePersistenceError:
                pass
        self.cuda_guard.stop()
        self.directory.cleanup()

    def preparation(self, *, max_pending=8, budget_bytes=2*1024**3):
        world = _world(cache_bytes=100_000)
        preparation = CoarsePreparation(self.directory.name, _manifest(701),
            bounds=(0, 0, 7680, 7680), max_pending=max_pending,
            budget_bytes=budget_bytes).install(world)
        self.preparations.append(preparation)
        return preparation, world

    def block_writer(self, preparation):
        entered, release = threading.Event(), threading.Event()
        self.releases.append(release)
        original = preparation._commit_window

        def commit(index, array):
            entered.set()
            if not release.wait(5):
                raise TimeoutError('Test writer was not released')
            return original(index, array)

        preparation._commit_window = commit
        return entered, release

    def value(self, preparation, index):
        return preparation._model_f([index])[0]

    def test_pending_snapshot_reused_by_peer_before_durable_commit(self):
        first, world = self.preparation()
        second, peer = self.preparation()
        entered, release = self.block_writer(first)
        index = first._indices[0]
        expected = self.value(first, index)
        original = expected.clone()
        self.assertTrue(first._save_window(index, expected))
        self.assertTrue(entered.wait(2))
        expected.zero_()  # Caller/storage reuse must not corrupt accepted CPU data.
        self.assertFalse(second._valid_metadata(index))
        self.assertEqual(second.status()['complete_windows'], 0)
        restored = peer.coarse._f([index])[0]
        self.assertTrue(torch.equal(restored, original))
        restored.zero_()  # Reader cannot mutate the shared pending snapshot.
        self.assertTrue(torch.equal(first._load_window(index), original))
        self.assertEqual(peer.calls['coarse'], [])
        self.assertEqual(second.status()['persistence']['pending_hits'], 2)
        with self.assertRaises(TimeoutError):
            first.flush(timeout=.01)
        release.set()
        first.flush(timeout=5)
        self.assertTrue(second._valid_metadata(index))
        np.testing.assert_array_equal(np.load(first._window_path(index)), original.numpy())
        self.assertEqual(second.status()['persistence']['committed'], 1)

    def test_bounded_queue_applies_observable_backpressure(self):
        preparation, _ = self.preparation(max_pending=1)
        entered, release = self.block_writer(preparation)
        a, b = preparation._indices[:2]
        preparation._save_window(a, self.value(preparation, a))
        self.assertTrue(entered.wait(2))
        with ThreadPoolExecutor(max_workers=1) as caller:
            submitted = threading.Event()

            def enqueue():
                submitted.set()
                return preparation._save_window(b, self.value(preparation, b))

            future = caller.submit(enqueue)
            self.assertTrue(submitted.wait(2))
            with preparation._namespace.condition:
                # Wait on the Condition, rather than assuming thread timing.
                self.assertTrue(preparation._namespace.condition.wait_for(
                    lambda: preparation._namespace.backpressure_count == 1, timeout=2))
            self.assertFalse(future.done())
            self.assertEqual(preparation.status()['persistence']['pending_windows'], 1)
            release.set()
            self.assertTrue(future.result(timeout=5))
        preparation.flush(timeout=5)
        status = preparation.status()['persistence']
        self.assertEqual(status['high_water'], 1)
        self.assertEqual(status['committed'], 2)
        self.assertGreater(status['backpressure_seconds'], 0)

    def test_pending_quota_shared_and_rejected_before_copy(self):
        first, _ = self.preparation(budget_bytes=576)
        second, _ = self.preparation(budget_bytes=576)
        entered, release = self.block_writer(first)
        a, b = first._indices[:2]
        first._save_window(a, self.value(first, a))
        self.assertTrue(entered.wait(2))
        self.assertFalse(second._save_window(b, self.value(second, b)))
        status = first.status()
        self.assertTrue(status['disk_budget_exhausted'])
        self.assertEqual(status['persistence']['reserved_disk_bytes'], 576)
        self.assertEqual(status['disk_bytes'], 0)
        release.set()
        first.flush(timeout=5)
        self.assertEqual(first.status()['disk_bytes'], 576)
        self.assertEqual(first.status()['persistence']['reserved_disk_bytes'], 0)

    def test_partial_commit_failure_reported_without_false_readiness(self):
        import terrain_coarse
        preparation, world = self.preparation()
        index = preparation._indices[0]
        original = terrain_coarse._atomic

        def fail_sidecar(path, payload):
            if path == preparation._sidecar_path(index):
                raise OSError('simulated full disk')
            return original(path, payload)

        with patch.object(terrain_coarse, '_atomic', side_effect=fail_sidecar):
            preparation._save_window(index, self.value(preparation, index))
            with self.assertRaisesRegex(CoarsePersistenceError, 'simulated full disk'):
                preparation.flush(timeout=5)
        self.assertFalse(preparation._valid_metadata(index))
        status = preparation.status()
        self.assertEqual(status['complete_windows'], 0)
        self.assertEqual(status['persistence']['errors'], 1)
        self.assertEqual(status['persistence']['pending_windows'], 0)
        self.assertEqual(status['disk_bytes'], preparation._window_path(index).stat().st_size)
        with self.assertRaises(CoarsePersistenceError):
            preparation.step(world)

    def test_close_drains_but_does_not_close_peer_and_receipt_rejects_tamper(self):
        first, _ = self.preparation()
        peer, _ = self.preparation()
        index = first._indices[0]
        value = self.value(first, index)
        first._save_window(index, value)
        first.close(timeout=5)
        first.close(timeout=5)
        self.assertTrue(torch.equal(peer._load_window(index), value))
        with self.assertRaisesRegex(CoarsePersistenceError, 'closed'):
            first._save_window(first._indices[1], value)
        path = first._window_path(index)
        array = np.load(path)
        array[0, 0, 0] += 1
        with path.open('wb') as stream:
            np.save(stream, array)
        sidecar_path = first._sidecar_path(index)
        metadata = json.loads(sidecar_path.read_text())
        metadata['mtime_ns'] = path.stat().st_mtime_ns
        sidecar_path.write_text(json.dumps(metadata))  # Deliberately retain old SHA.
        self.assertIsNone(peer._load_window(index))
        self.assertTrue(peer._save_window(index, value))
        peer.flush(timeout=5)
        self.assertTrue(torch.equal(peer._load_window(index), value))

    def test_real_fusion_replay_and_final_step_durable(self):
        preparation, world = self.preparation()
        expected = read_rect(world, 'coarse', 0, 0, 2, 2).clone()
        preparation.flush(timeout=5)
        peer, reopened = self.preparation()
        actual = read_rect(reopened, 'coarse', 0, 0, 2, 2)
        self.assertTrue(torch.equal(expected, actual))
        self.assertEqual(reopened.calls['coarse'], [])
        result = preparation.step(world, budget_windows=preparation.status()['total_windows'])
        self.assertEqual(result['complete_windows'], result['total_windows'])
        self.assertEqual(result['persistence']['pending_windows'], 0)


if __name__ == '__main__':
    unittest.main()

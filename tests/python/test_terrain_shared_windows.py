"""Latent/decoder windows shared across GPUs: computed once, never twice."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import threading
from types import SimpleNamespace
import unittest

import torch
from infinite_tensor import InfiniteTensor, MemoryTileStore, TensorWindow

from terrain_shared_windows import SharedWindows


class Stage:
    """A fake stage tensor with the InfiniteTensor attributes the wrapper uses."""
    def __init__(self, uuid='init_residual_map', offset=0.):
        self.uuid, self.args, self.device = uuid, (), torch.device('cpu')
        self.offset, self.calls = offset, []
        self.started, self.release = threading.Event(), threading.Event()
        self.release.set()
        self.fail = False
        self._f = self.compute

    def compute(self, ctxs, *args):
        self.calls.append(list(ctxs))
        self.started.set()
        self.release.wait(5)
        if self.fail:
            raise RuntimeError('cancelled producer')
        # Two "GPUs" give slightly different numbers for the same window.
        return [torch.full((2, 2), float(sum(ctx))+self.offset) for ctx in ctxs]


def world_with(stage):
    return SimpleNamespace(residual=stage, latents=None)


class SharedWindowTests(unittest.TestCase):
    def test_second_gpu_reuses_published_windows(self):
        registry = SharedWindows()
        first, second = Stage(offset=0.), Stage(offset=.25)
        registry.install(world_with(first), ('final', 'world'))
        registry.install(world_with(second), ('final', 'world'))
        a = first._f([(0, 0, 1), (0, 1, 1)])
        b = second._f([(0, 1, 1), (0, 2, 1)])
        self.assertTrue(torch.equal(a[1], b[0]))
        self.assertEqual(second.calls, [[(0, 2, 1)]])
        self.assertEqual(registry.status()['own_hits']+registry.status()['shared_hits'], 1)

    def test_window_claimed_by_another_gpu_is_awaited_not_recomputed(self):
        registry = SharedWindows()
        first, second = Stage(offset=0.), Stage(offset=.25)
        registry.install(world_with(first), ('final', 'world'))
        registry.install(world_with(second), ('final', 'world'))
        first.release.clear()
        results = {}
        producer = threading.Thread(target=lambda: results.setdefault('a', first._f([(0, 5, 5)])))
        producer.start()
        self.assertTrue(first.started.wait(2))
        consumer = threading.Thread(target=lambda: results.setdefault('b', second._f([(0, 5, 5), (0, 6, 6)])))
        consumer.start()
        # The consumer computes its own claim first, then waits for the producer.
        for _ in range(100):
            if second.calls:
                break
            threading.Event().wait(.01)
        self.assertEqual(second.calls, [[(0, 6, 6)]])
        first.release.set()
        producer.join(5); consumer.join(5)
        self.assertTrue(torch.equal(results['a'][0], results['b'][0]))
        self.assertEqual(registry.status()['waits'], 1)

    def test_failed_producer_releases_its_claim(self):
        registry = SharedWindows()
        first, second = Stage(offset=0.), Stage(offset=.25)
        registry.install(world_with(first), ('final', 'world'))
        registry.install(world_with(second), ('final', 'world'))
        first.release.clear()
        first.fail = True
        errors = []
        def produce():
            try:
                first._f([(0, 7, 7)])
            except RuntimeError as exc:
                errors.append(exc)
        producer = threading.Thread(target=produce)
        producer.start()
        self.assertTrue(first.started.wait(2))
        results = []
        consumer = threading.Thread(target=lambda: results.append(second._f([(0, 7, 7)])))
        consumer.start()
        threading.Event().wait(.1)
        first.release.set()
        producer.join(5); consumer.join(5)
        self.assertEqual(len(errors), 1)
        self.assertEqual(results[0][0][0, 0].item(), 14.25)
        self.assertEqual(registry.status()['claims'], 0)

    def test_namespaces_and_inactive_mode_do_not_share(self):
        registry = SharedWindows()
        final, preview = Stage(), Stage(offset=.5)
        registry.install(world_with(final), ('final', 'world'))
        registry.install(world_with(preview), ('preview', 'world'))
        final._f([(0, 1, 1)])
        preview._f([(0, 1, 1)])
        self.assertEqual(preview.calls, [[(0, 1, 1)]])
        single = SharedWindows(active=lambda: False)
        a, b = Stage(), Stage()
        single.install(world_with(a), ('final', 'world'))
        single.install(world_with(b), ('final', 'world'))
        a._f([(0, 1, 1)]); b._f([(0, 1, 1)])
        self.assertEqual(b.calls, [[(0, 1, 1)]])
        self.assertEqual(single.status()['windows'], 0)

    def test_budget_evicts_oldest_windows(self):
        registry = SharedWindows(budget_bytes=3*16)
        stage = Stage()
        registry.install(world_with(stage), ('final', 'world'))
        stage._f([(0, i, i) for i in range(5)])
        self.assertEqual(registry.status()['windows'], 3)
        self.assertEqual(registry.status()['evicted'], 2)

    def test_real_infinite_tensors_in_separate_stores_share_every_window(self):
        registry = SharedWindows()
        calls = []
        def make(offset):
            def f(ctxs):
                calls.append((offset, list(ctxs)))
                return [torch.full((1, 4, 4), offset+ctx[1]*10+ctx[2]) for ctx in ctxs]
            return InfiniteTensor(shape=(1, None, None), f=f, output_window=TensorWindow(size=(1, 4, 4), stride=(1, 2, 2)),
                                  batch_size=4, tile_store=MemoryTileStore(), tensor_id='init_residual_map')
        first, second = make(0.), make(1000.)
        registry.install(SimpleNamespace(residual=first), ('final', 'w'))
        registry.install(SimpleNamespace(residual=second), ('final', 'w'))
        a = first[:, 0:8, 0:8]
        b = second[:, 0:8, 0:8]
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(all(offset == 0. for offset, _ in calls))


if __name__ == '__main__':
    unittest.main()

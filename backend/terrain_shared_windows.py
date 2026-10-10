"""Latent/decoder windows computed once and shared by every GPU of the plan.

Different GPU models produce slightly different BF16 windows. When two
neighbouring tiles are computed on two GPUs, each GPU would otherwise compute
its own copy of the windows they share, and the tiles would meet with a small
step. Here the first GPU that needs a window claims it, computes it and
publishes it; any other GPU waits for and copies that same window. Every
window then has exactly one value, whichever GPU produced it.

A wrapped stage function partitions each batch into published windows (copied
to the caller's device), windows claimed by another GPU (awaited after the
caller's own work) and new claims (computed now, as one smaller batch). A
caller computes its claims before waiting, and claims are only made inside the
stage function once its inputs exist, so two GPUs cannot wait on each other.
Failed or cancelled claims are released; a waiter then computes the window.

Keys include the world identity, the store kind (final, preview, polar) and the
stage tensor id, so windows never cross worlds or isolated preview stores.
With a single detail GPU the wrapper is a pass-through.
"""
from collections import OrderedDict
import threading

import torch

STAGES = ('init_latent_map', 'step_latent_map_0', 'latent_map_T1', 'init_residual_map')


def _bytes(tensor):
    return tensor.numel() * tensor.element_size()


class SharedWindows:
    def __init__(self, budget_bytes=1024**3, *, active=lambda: True, check=lambda: None):
        self.budget_bytes = int(budget_bytes)
        self.active = active
        self.check = check
        self.condition = threading.Condition()
        self.values = OrderedDict()
        self.claims = {}
        self.bytes = 0
        self.metrics = dict(published=0, shared_hits=0, own_hits=0, waits=0,
                            released_claims=0, evicted=0, wait_seconds=0.0)

    def install(self, world, namespace):
        """Wrap every latent/decoder stage tensor of ``world``."""
        seen = set()
        def visit(tensor):
            if tensor is None or id(tensor) in seen:
                return
            seen.add(id(tensor))
            for dependency in getattr(tensor, 'args', ()):
                visit(dependency)
            if tensor.uuid in STAGES:
                self._wrap(tensor, (namespace, tensor.uuid), world)
        visit(getattr(world, 'residual', None))
        visit(getattr(world, 'latents', None))
        return self

    def _wrap(self, tensor, prefix, world):
        if getattr(tensor, '_terrain_shared_windows', None) is self:
            return
        original = tensor._f
        registry = self
        device_of = lambda: torch.device(tensor.device) if hasattr(tensor, 'device') else None

        def shared(ctxs, *args):
            if not registry.active():
                return original(ctxs, *args)
            return registry._call(prefix, original, list(ctxs), args, device_of(), world)

        tensor._terrain_shared_original_f = original
        tensor._f = shared
        tensor._terrain_shared_windows = self

    def _call(self, prefix, original, ctxs, args, device, world):
        owner = threading.get_ident()
        outputs = [None] * len(ctxs)
        mine, foreign = [], []
        with self.condition:
            for position, ctx in enumerate(ctxs):
                key = (prefix, tuple(ctx))
                value = self.values.get(key)
                if value is not None:
                    self.values.move_to_end(key)
                    outputs[position] = value
                    self.metrics['own_hits' if value.device == device else 'shared_hits'] += 1
                elif key in self.claims and self.claims[key] != owner:
                    foreign.append(position)
                else:
                    self.claims[key] = owner
                    mine.append(position)
        try:
            if mine:
                computed = original([ctxs[p] for p in mine], *[[arg[p] for p in mine] for arg in args])
                if len(computed) != len(mine):
                    raise RuntimeError('Stage returned an incomplete batch')
                with self.condition:
                    for position, value in zip(mine, computed):
                        outputs[position] = value
                        self._publish((prefix, tuple(ctxs[position])), value)
                    mine = []
                    self.condition.notify_all()
            for position in foreign:
                outputs[position] = self._await(prefix, ctxs[position], original, args, position, device)
        finally:
            if mine:
                with self.condition:
                    for position in mine:
                        key = (prefix, tuple(ctxs[position]))
                        if self.claims.get(key) == owner:
                            del self.claims[key]
                            self.metrics['released_claims'] += 1
                    self.condition.notify_all()
        return [value if value.device == device else value.to(device) for value in outputs]

    def _await(self, prefix, ctx, original, args, position, device):
        import time
        key = (prefix, tuple(ctx))
        started = time.perf_counter()
        with self.condition:
            self.metrics['waits'] += 1
            while True:
                value = self.values.get(key)
                if value is not None:
                    self.metrics['wait_seconds'] += time.perf_counter() - started
                    self.metrics['shared_hits'] += 1
                    return value
                if key not in self.claims:
                    # The producer failed or was cancelled: compute it here.
                    self.claims[key] = threading.get_ident()
                    break
                self.condition.wait(.5)
                self.condition.release()
                try:
                    self.check()
                finally:
                    self.condition.acquire()
        try:
            value = original([ctx], *[[arg[position]] for arg in args])[0]
        except BaseException:
            with self.condition:
                if self.claims.get(key) == threading.get_ident():
                    del self.claims[key]
                    self.metrics['released_claims'] += 1
                self.condition.notify_all()
            raise
        with self.condition:
            self._publish(key, value)
            self.condition.notify_all()
        return value

    def _publish(self, key, value):
        """Caller holds the condition. Windows are immutable after production."""
        self.claims.pop(key, None)
        previous = self.values.pop(key, None)
        if previous is not None:
            self.bytes -= _bytes(previous)
        self.values[key] = value
        self.bytes += _bytes(value)
        self.metrics['published'] += 1
        while self.bytes > self.budget_bytes and len(self.values) > 1:
            _, evicted = self.values.popitem(last=False)
            self.bytes -= _bytes(evicted)
            self.metrics['evicted'] += 1

    def clear(self):
        with self.condition:
            self.values.clear()
            self.bytes = 0
            self.condition.notify_all()

    def status(self):
        with self.condition:
            return dict(self.metrics, windows=len(self.values), bytes=self.bytes,
                        budget_bytes=self.budget_bytes, claims=len(self.claims),
                        wait_seconds=round(self.metrics['wait_seconds'], 4))

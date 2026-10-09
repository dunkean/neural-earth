"""Window observability and safe cancellation for the real InfiniteTensor graph.

InfiniteTensor owns overlap fusion and the weighted numerator/denominator.  This
module deliberately leaves that arithmetic in its store: a read is published
only after every intersecting window, including all upstream passes, exists.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from hashlib import blake2b
import math
from threading import local
from time import perf_counter

MAX_READ_ELEMENTS = 4 * 1024 * 1024


@dataclass
class StageCounters:
    requests: int = 0
    cache_hits: int = 0
    generated: int = 0
    regenerated: int = 0
    seconds: float = 0.0
    # One MiB per stage, fixed for arbitrary exploration duration. Bloom
    # membership and cardinality are estimates, never exact cache accounting.
    seen_bits: bytearray = field(default_factory=lambda: bytearray(1 << 20), repr=False)
    set_bits: int = 0

    def seen_before(self, index: tuple[int, ...]) -> bool:
        digest = blake2b(repr(index).encode('ascii'), digest_size=16).digest()
        bit_count = len(self.seen_bits) * 8
        positions = [int.from_bytes(digest[offset:offset+4], 'little') % bit_count
                     for offset in (0, 4, 8)]
        present = True
        for position in positions:
            byte, bit = divmod(position, 8)
            mask = 1 << bit
            if not self.seen_bits[byte] & mask:
                present = False
                self.seen_bits[byte] |= mask
                self.set_bits += 1
        return present

    def snapshot(self) -> dict:
        bit_count = len(self.seen_bits) * 8
        estimated_unique = (None if self.set_bits >= bit_count else
                            round(-bit_count / 3 * math.log1p(-self.set_bits / bit_count)))
        return dict(requests=self.requests, cache_hits=self.cache_hits,
                    generated=self.generated, regenerated_estimate=self.regenerated,
                    unique_windows_estimate=estimated_unique,
                    metadata_history_bytes=len(self.seen_bits),
                    request_scope='internal_dependencies_and_external_reads',
                    generated_scope='store_materializations_including_disk_replay',
                    seconds_inclusive=round(self.seconds, 6))


class WorldWindowScheduler:
    """Observe and cancel actual coarse, base-pass and decoder model batches.

    Upstream dependency enumeration and batch partitioning remain untouched.
    Decomposing a request before its dependencies are materialized can change
    BF16 base batch shapes and, in turn, terrain metres. A caller may provide
    a thread-local interest check before each real ``f`` batch; completed
    windows remain reusable after cancellation.
    """

    def __init__(self, world):
        self.world = world
        self._thread = local()
        self._counters = defaultdict(StageCounters)
        for tensor in (world.coarse, world.latents, world.residual):
            self._install_recursive(tensor)

    def _install_recursive(self, tensor):
        for dependency in tensor.args:
            self._install_recursive(dependency)
        if getattr(tensor, '_terrain_scheduled', False):
            return
        name = tensor.uuid
        original = tensor._ensure_processed
        original_f = tensor._f
        scheduler = self

        def observed(window_indices):
            # InfiniteTensor still performs its original complete dependency
            # traversal and decides every batch boundary. Only count before
            # and after; there is no request decomposition here.
            indices = list(window_indices)
            counters = scheduler._counters[name]
            unique = list(dict.fromkeys(indices))
            pending = []
            for index in unique:
                counters.requests += 1
                if tensor._store.is_window_processed(tensor.uuid, index):
                    counters.cache_hits += 1
                else:
                    pending.append(index)
            start = perf_counter()
            try:
                original(indices)
            finally:
                counters.seconds += perf_counter() - start
                for index in pending:
                    if tensor._store.is_window_processed(tensor.uuid, index):
                        counters.generated += 1
                        if counters.seen_before(index):
                            counters.regenerated += 1

        def checked_f(*args):
            check = getattr(scheduler._thread, 'check', None)
            if check is not None:
                check()
            return original_f(*args)

        tensor._terrain_original_ensure = original
        tensor._terrain_original_f = original_f
        tensor._ensure_processed = observed
        tensor._f = checked_f
        tensor._terrain_scheduled = True

    @contextmanager
    def scope(self, check=None):
        previous = getattr(self._thread, 'check', None)
        self._thread.check = check
        try:
            yield self
        finally:
            self._thread.check = previous

    def status(self) -> dict:
        return {name: value.snapshot() for name, value in sorted(self._counters.items())}

    def ensure_window(self, tensor, index: tuple[int, ...], *, check=None) -> bool:
        """Materialize exactly one output window (plus its real dependencies).

        Returns True if it was already in the active tensor store. This is the
        background worker's scheduling quantum; asking for its pixel rectangle
        would accidentally request the neighbouring overlap windows as well.
        """
        already = tensor._store.is_window_processed(tensor.uuid, index)
        effective_check = check if check is not None else getattr(self._thread, 'check', None)
        if effective_check is not None:
            effective_check()
        tensor._store.begin_access(tensor.uuid)
        try:
            tensor._ensure_processed([index])
        finally:
            tensor._store.end_access(tensor.uuid)
        return already

    def ensure_windows(self, tensor, indices, *, check=None):
        """Submit a coarse group without decomposing any upstream stages."""
        effective_check = check if check is not None else getattr(self._thread, 'check', None)
        if effective_check is not None:
            effective_check()
        tensor._store.begin_access(tensor.uuid)
        try:
            with self.scope(effective_check):
                tensor._ensure_processed(indices)
        finally:
            tensor._store.end_access(tensor.uuid)


def stage_tensor(world, stage: str):
    if stage == 'coarse':
        return world.coarse
    if stage in ('latent', 'base'):
        return world.latents
    if stage in ('decoder', 'residual'):
        return world.residual
    raise ValueError(f'Unknown neural stage: {stage}')


def read_rect(world, stage: str, i1: int, j1: int, i2: int, j2: int, *, check=None):
    """Return the fully fused stage rectangle with a cancellation boundary.

    Coordinates use the selected stage's output cells, row first.  This calls
    the real InfiniteTensor read, so its store performs the same ordered sum
    as an ordinary pipeline read and no partial weighted average can escape.
    """
    if i2 <= i1 or j2 <= j1:
        raise ValueError('Rectangle must have positive width and height')
    tensor = stage_tensor(world, stage)
    channels = getattr(tensor, 'shape', (1,))[0]
    if (i2-i1) * (j2-j1) * channels > MAX_READ_ELEMENTS:
        raise ValueError('Neural stage read exceeds bounded output budget')
    scheduler = getattr(world, '_terrain_window_scheduler', None)
    if scheduler is None:
        return tensor[:, i1:i2, j1:j2]
    with scheduler.scope(check):
        return tensor[:, i1:i2, j1:j2]


def ensure_rect(world, stage: str, i1: int, j1: int, i2: int, j2: int, *, check=None):
    """Prefetch contributors; call ``read_rect`` when fused pixels are needed."""
    if i2 <= i1 or j2 <= j1:
        raise ValueError('Rectangle must have positive width and height')
    tensor = stage_tensor(world, stage)
    channels = getattr(tensor, 'shape', (1,))[0]
    if (i2-i1) * (j2-j1) * channels > MAX_READ_ELEMENTS:
        raise ValueError('Neural stage prefetch exceeds bounded output budget')
    region = (slice(0, tensor.shape[0]), slice(i1, i2), slice(j1, j2))
    scheduler = getattr(world, '_terrain_window_scheduler', None)
    if scheduler is None:
        return tensor[:, i1:i2, j1:j2]
    with scheduler.scope(check):
        tensor._store.begin_access(tensor.uuid)
        try:
            tensor._ensure_processed_range([region])
        finally:
            tensor._store.end_access(tensor.uuid)


def scheduler_status(world) -> dict:
    scheduler = getattr(world, '_terrain_window_scheduler', None)
    return scheduler.status() if scheduler is not None else {}

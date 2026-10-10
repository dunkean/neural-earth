"""Persistent, resumable preparation of learned 7.68 km coarse windows.

The persisted object is each model-produced *weighted* seven-channel window.
The live InfiniteTensor store still sums every overlapping contributor in its
native order. A saved window can therefore be reused after RAM eviction or a
process restart without accepting an incomplete weighted average.
"""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path
import re
import threading
from time import perf_counter
from uuid import uuid4
from weakref import WeakValueDictionary

import numpy as np
import torch
from terrain_profiling import span

from terrain_window_scheduler import read_rect
from terrain_manifest import world_identity
from terrain_conditioning import WORLD_PROFILES
from terrain_generation import resolve_generation
from terrain_diffusion.inference.world_pipeline import linear_weight_window

COARSE_METRES = 7680
# The climate sampler requests ci-8:ceil(max/256)+1+8, so the far edge
# needs nine cells beyond the finite grid's exclusive stop.
CONTEXT_CELLS = 9
DEFAULT_BOUNDS = (-20_000_000, -10_000_000, 20_000_000, 10_000_000)
SCHEMA = 'learned-coarse-windows-v3'


@dataclass
class _NamespaceState:
    """Accounting shared by live preparations of one on-disk world."""
    budget_bytes: int
    persisted_indices: set[tuple[int, ...]]
    verified_stamps: OrderedDict
    disk_bytes: int = 0
    disk_budget_exhausted: bool = False
    max_pending: int = 8
    condition: threading.Condition = field(default_factory=threading.Condition)
    io_lock: threading.RLock = field(default_factory=threading.RLock)
    pending: OrderedDict = field(default_factory=OrderedDict)
    reserved_bytes: int = 0
    future: object = None
    error: str | None = None
    submitted: int = 0
    committed: int = 0
    errors: int = 0
    pending_hits: int = 0
    high_water: int = 0
    backpressure_count: int = 0
    backpressure_seconds: float = 0.0
    transfer_wait_seconds: float = 0.0
    write_seconds: float = 0.0


@dataclass
class _PendingWrite:
    owner: object
    index: tuple[int, ...]
    cpu: torch.Tensor
    source: torch.Tensor | None
    event: object
    reserved_bytes: int

    def wait(self):
        # Wait for this D2H only, never synchronize all work on the device.
        if self.event is not None:
            self.event.synchronize()


class CoarsePersistenceError(RuntimeError):
    pass


_NAMESPACE_STATES: WeakValueDictionary[Path, _NamespaceState] = WeakValueDictionary()
_NAMESPACE_LOCK = threading.Lock()
# Executor shutdown drains accepted writes before Python/CUDA teardown. A
# namespace has only one active drain; two worlds may persist independently.
_PERSIST_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix='terrain-coarse-save')


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid4().hex + '.tmp')
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_bytes(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False).encode('utf-8')


@dataclass(frozen=True)
class CoarseGrid:
    i1: int
    j1: int
    i2: int
    j2: int

    @classmethod
    def from_bounds(cls, bounds):
        x0, y0, x1, y1 = bounds
        if not x1 > x0 or not y1 > y0:
            raise ValueError('World bounds must have positive area')
        return cls(math.floor(y0 / COARSE_METRES), math.floor(x0 / COARSE_METRES),
                   math.ceil(y1 / COARSE_METRES), math.ceil(x1 / COARSE_METRES))


class CoarsePreparation:
    """Install a per-world disk cache and prepare the finite world by budgets.

    ``manifest`` must be the canonical ``terrain_manifest.build_manifest``
    mapping. Its ``world_hash`` namespaces all windows and mip products.
    Install after world.bind(), before the first coarse access. The normal
    viewport path and background preparation then share one persisted cache.
    """

    def __init__(self, root: str | Path, manifest: dict, *, bounds=DEFAULT_BOUNDS,
                 budget_bytes: int = 2 * 1024**3, max_pending: int = 8,
                 async_persistence: bool = True):
        if (not isinstance(manifest, dict) or
                not re.fullmatch(r'[0-9a-f]{16,64}', str(manifest.get('world_hash', '')))):
            raise ValueError('Coarse cache requires a world manifest with world_hash')
        world_identity(manifest)
        if not all(key in manifest for key in ('seed_u64', 'ablation', 'conditioning', 'generation')):
            raise ValueError('Persistent coarse requires seed, ablation, conditioning and generation identity')
        if budget_bytes <= 0:
            raise ValueError('budget_bytes must be positive')
        if not isinstance(max_pending, int) or max_pending < 1:
            raise ValueError('max_pending must be a positive integer')
        self.max_pending = max_pending
        self.async_persistence = async_persistence
        self._closed = False
        self.manifest = dict(manifest)
        self.world_hash = manifest['world_hash']
        self.bounds = tuple(bounds)
        self.grid = CoarseGrid.from_bounds(bounds)
        self.root = Path(root) / SCHEMA / self.world_hash
        self.budget_bytes = int(budget_bytes)
        self.windows_dir = self.root / 'windows'
        self.mips_dir = self.root / 'mips'
        self._cursor_path = self.root / 'progress.json'
        self._manifest_path = self.root / 'manifest.json'
        self._contract_path = self.root / 'contract.json'
        self._tensor = None
        self._namespace = None
        self._indices: tuple[tuple[int, ...], ...] = ()
        self._planned_indices: set[tuple[int, ...]] = set()
        self._persisted_indices: set[tuple[int, ...]] = set()
        self._disk_bytes = 0
        self._verified_stamps: OrderedDict[tuple[int, ...], tuple] = OrderedDict()
        self._expected_weight = None
        self._weight_digest = None
        self._cursor = 0
        self._priority_order = None
        self._priority_cursor = 0
        self._priority_signature = None
        self._priority_geometry = None
        self.disk_hits = 0
        self.network_windows = 0
        self.model_synchronized_seconds = 0.0
        self.model_submission_seconds = 0.0
        self.persistence_seconds = 0.0
        self.disk_budget_exhausted = False
        self._model_f = None
        # Multi-GPU: ``remote(indices)`` computes and persists missing windows on
        # the plan's coarse device. This world then never runs the coarse model,
        # so one world never mixes coarse windows from different GPU models.
        self.remote = None
        self.remote_windows = 0

    @property
    def _disk_bytes(self):
        return (self._namespace.disk_bytes if self._namespace is not None
                else self._unshared_disk_bytes)

    @_disk_bytes.setter
    def _disk_bytes(self, value):
        if self._namespace is None:
            self._unshared_disk_bytes = value
        else:
            self._namespace.disk_bytes = value

    @property
    def disk_budget_exhausted(self):
        return (self._namespace.disk_budget_exhausted if self._namespace is not None
                else self._unshared_disk_budget_exhausted)

    @disk_budget_exhausted.setter
    def disk_budget_exhausted(self, value):
        if self._namespace is None:
            self._unshared_disk_budget_exhausted = value
        else:
            self._namespace.disk_budget_exhausted = value

    def install(self, world):
        tensor = world.coarse
        if hasattr(world, 'seed') and str(world.seed) != self.manifest['seed_u64']:
            raise ValueError('Coarse manifest seed does not match pipeline')
        if hasattr(world, 'kwargs') and list(world.kwargs['cond_snr']) != self.manifest['conditioning']['cond_snr']:
            raise ValueError('Coarse manifest conditioning SNR does not match pipeline')
        if hasattr(world, '_dtype'):
            actual_precision = {'torch.bfloat16': 'bf16', 'torch.float32': 'fp32',
                                'torch.float16': 'fp16'}.get(str(world._dtype or torch.float32))
            if actual_precision != self.manifest['generation']['precision']:
                raise ValueError('Coarse manifest precision does not match pipeline')
        declared_profile = self.manifest.get('world_profile', 'natural')
        descriptor = resolve_generation(declared_profile)
        if declared_profile != 'natural' and not hasattr(world, '_terrain_world_profile'):
            raise ValueError('Coarse terrestrial pipeline requires a world profile')
        if hasattr(world, '_terrain_world_profile'):
            profile_name = world._terrain_world_profile
            try:
                resolve_generation(profile_name)
            except ValueError:
                raise ValueError('Coarse pipeline world profile is unsupported')
            actual_ablation = 'A0' if profile_name == 'natural' else profile_name
            if actual_ablation != self.manifest['ablation']:
                raise ValueError('Coarse manifest ablation does not match pipeline')
            if declared_profile != profile_name:
                raise ValueError('Coarse manifest world profile does not match pipeline')
        if not descriptor.is_default:
            factory = getattr(world, 'synthetic_map_factory', None)
            if getattr(factory, '_terrain_generation_profile', None) != declared_profile:
                raise ValueError('Coarse generation factory does not match pipeline settings')
            if getattr(factory, 'generation_settings', None) != descriptor.settings:
                raise ValueError('Coarse generation settings do not match conditioning factory')
        if descriptor.needs_bootstrap:
            factory = getattr(world, 'synthetic_map_factory', None)
            heightmap = getattr(factory, 'heightmap', None)
            metadata = getattr(heightmap, 'metadata', None)
            if not isinstance(metadata, dict):
                raise ValueError('Coarse terrestrial factory requires bootstrap height metadata')
            bootstrap = self.manifest.get('bootstrap')
            expected_height = bootstrap.get('height_sha256') if isinstance(bootstrap, dict) else None
            if (not isinstance(expected_height, str) or not expected_height
                    or metadata.get('height_sha256') != expected_height):
                raise ValueError('Coarse manifest bootstrap height does not match conditioning factory')
        if self.manifest['conditioning'].get('generation_settings', descriptor.settings) != descriptor.settings:
            raise ValueError('Coarse manifest generation settings do not match pipeline')
        if hasattr(world, '_terrain_profile'):
            declared = self.manifest['generation'].get('inference_profile', {})
            for key in ('coarse_batch', 'latent_batch', 'decoder_batch', 'canonical_latents'):
                if key in declared and declared[key] != getattr(world._terrain_profile, key):
                    raise ValueError(f'Coarse manifest {key} does not match pipeline')
        from terrain_inference import coarse_stream_count
        if tensor.batch_size != 1 and not (
                getattr(getattr(world, '_terrain_profile', None), 'coarse_batch', None) == 1
                and tensor.batch_size == coarse_stream_count(world)):
            raise ValueError('Persistent coarse requires the validated batch-one profile')
        if getattr(tensor, '_terrain_coarse_preparation', None) is self:
            return self
        if getattr(tensor, '_terrain_coarse_preparation', None) is not None:
            raise ValueError('Coarse tensor already has a different persistent identity')
        if tensor.output_window.size[1] != tensor.output_window.size[2]:
            raise ValueError('Expected square coarse windows')
        self._expected_weight = linear_weight_window(
            tensor.output_window.size[1], 'cpu', torch.float32).numpy()
        self._weight_digest = sha256(self._expected_weight.tobytes()).hexdigest()
        header = BytesIO()
        np.lib.format.write_array_header_1_0(header, dict(
            descr='<f4', fortran_order=False, shape=tuple(tensor.output_window.size)))
        self._payload_bytes = header.tell() + int(np.prod(tensor.output_window.size)) * 4
        manifest_path = self._manifest_path
        if manifest_path.exists():
            stored = json.loads(manifest_path.read_text(encoding='utf-8'))
            if stored != self.manifest:
                raise ValueError('Coarse manifest collision; refusing mixed worlds')
        else:
            _atomic(manifest_path, _json_bytes(self.manifest))
        contract = dict(schema=SCHEMA, world_hash=self.world_hash,
                        bounds=self.bounds, grid=vars(self.grid), context_cells=CONTEXT_CELLS,
                        tensor_shape=tensor.shape,
                        output_window=tensor.output_window.to_dict(),
                        dtype=str(tensor.dtype), batch_size=1,
                        weight_sha256=self._weight_digest)
        if self._contract_path.exists():
            if json.loads(self._contract_path.read_text(encoding='utf-8')) != json.loads(_json_bytes(contract)):
                raise ValueError('Coarse stage geometry or precision changed for persisted world')
        else:
            _atomic(self._contract_path, _json_bytes(contract))
        self._tensor = tensor
        # Climate's 15-cell local baseline requires a nine-cell far apron.
        # Include it in the resumable finite plan so edge tiles can become
        # learned-ready without an unplanned synchronous neural burst.
        region = (slice(0, tensor.shape[0]),
                  slice(self.grid.i1 - CONTEXT_CELLS, self.grid.i2 + CONTEXT_CELLS),
                  slice(self.grid.j1 - CONTEXT_CELLS, self.grid.j2 + CONTEXT_CELLS))
        self._indices = tuple(tensor.output_window.intersecting_windows(
            region, tensor_shape=tensor.shape))
        self._planned_indices = set(self._indices)
        # Server foreground and background worlds can be installed before
        # either computes a window. Adopt one live namespace ledger rather
        # than freezing each installation's initial directory listing.
        with _NAMESPACE_LOCK:
            key = self.root.resolve()
            state = _NAMESPACE_STATES.get(key)
            if state is None:
                state = _NamespaceState(self.budget_bytes, set(), OrderedDict(), max_pending=self.max_pending)
                self._namespace = state
                self._persisted_indices = state.persisted_indices
                self._verified_stamps = state.verified_stamps
                self._disk_bytes = sum(path.stat().st_size
                                       for path in self.windows_dir.glob('*.npy'))
                for index in self._indices:
                    self._valid_metadata(index)
                _NAMESPACE_STATES[key] = state
            else:
                if state.budget_bytes != self.budget_bytes:
                    raise ValueError('Coarse namespace already uses a different disk budget')
                if state.max_pending != self.max_pending:
                    raise ValueError('Coarse namespace already uses a different pending budget')
                self._namespace = state
                self._persisted_indices = state.persisted_indices
                self._verified_stamps = state.verified_stamps
        if self._cursor_path.exists():
            try:
                progress = json.loads(self._cursor_path.read_text(encoding='utf-8'))
                if progress.get('world_hash') == self.world_hash:
                    self._cursor = int(progress.get('cursor', 0)) % max(len(self._indices), 1)
            except (OSError, ValueError, TypeError):
                pass
        original = tensor._f
        self._model_f = original
        preparation = self

        def persisted(ctxs, *args):
            preparation._check_error()
            outputs = [None] * len(ctxs)
            missing = []
            missing_positions = []
            for position, index in enumerate(ctxs):
                cached = preparation._load_window(index)
                if cached is None:
                    missing.append(index)
                    missing_positions.append(position)
                else:
                    outputs[position] = cached.to(tensor.device)
            if missing:
                remote = preparation.remote
                start = perf_counter()
                computed = remote(missing, *args) if remote is not None else original(missing, *args)
                preparation.model_submission_seconds += perf_counter() - start
                if len(computed) != len(missing):
                    raise RuntimeError('Coarse model returned an incomplete batch')
                if remote is not None:
                    # The coarse device already queued these windows for disk.
                    for position, output in zip(missing_positions, computed):
                        outputs[position] = output.to(tensor.device)
                        preparation.remote_windows += 1
                    return outputs
                persistence_start = perf_counter()
                for position, index, output in zip(missing_positions, missing, computed):
                    preparation._save_window(index, output)
                    outputs[position] = output
                    preparation.network_windows += 1
                preparation.persistence_seconds += perf_counter() - persistence_start
            return outputs

        tensor._f = persisted
        tensor._terrain_coarse_preparation = self
        world._terrain_coarse_preparation = self
        return self

    def _window_path(self, index) -> Path:
        _, i, j = index
        return self.windows_dir / f'{i}_{j}.npy'

    def _sidecar_path(self, index) -> Path:
        return self._window_path(index).with_suffix('.json')

    def _invalidate(self, index):
        was_known = index in self._persisted_indices or index in self._verified_stamps
        self._persisted_indices.discard(index)
        self._verified_stamps.pop(index, None)
        if was_known:
            # A peer may discover an external truncation. Reconcile bytes on
            # that rare path; normal per-window preparation stays O(1).
            self._disk_bytes = sum(path.stat().st_size
                                   for path in self.windows_dir.glob('*.npy'))
            if self._disk_bytes < self.budget_bytes:
                self.disk_budget_exhausted = False

    def _valid_metadata(self, index) -> bool:
        state = self._namespace
        with state.condition:
            if index in state.pending:
                return False  # Readable in RAM, but not yet durable.
        with state.io_lock:
            with state.condition:
                if index in state.pending:
                    return False
            return self._valid_metadata_locked(index)

    def _valid_metadata_locked(self, index) -> bool:
        """Cheap readiness: identity, header, size, and canonical weight bytes.

        The full file digest is checked on load. Stable file/sidecar stamps let
        repeated tile readiness avoid reopening unchanged window payloads.
        """
        path, sidecar_path = self._window_path(index), self._sidecar_path(index)
        try:
            file_stat, meta_stat = path.stat(), sidecar_path.stat()
            stamp = (file_stat.st_size, file_stat.st_mtime_ns,
                     meta_stat.st_size, meta_stat.st_mtime_ns)
            if self._verified_stamps.get(index) == stamp:
                self._verified_stamps.move_to_end(index)
                if index in self._planned_indices:
                    self._persisted_indices.add(index)
                return True
            metadata = json.loads(sidecar_path.read_text(encoding='utf-8'))
            expected = dict(schema=SCHEMA, world_hash=self.world_hash,
                            index=list(index), weight_sha256=self._weight_digest,
                            shape=list(self._tensor.output_window.size),
                            dtype='float32')
            if any(metadata.get(key) != value for key, value in expected.items()):
                raise ValueError('Window metadata identity mismatch')
            if (metadata.get('bytes') != file_stat.st_size or
                    metadata.get('mtime_ns') != file_stat.st_mtime_ns):
                raise ValueError('Window metadata size/time mismatch')
            with path.open('rb') as stream:
                version = np.lib.format.read_magic(stream)
                if version == (1, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
                elif version == (2, 0):
                    shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
                else:
                    raise ValueError('Unsupported NumPy window format')
                offset = stream.tell()
                if (shape != tuple(expected['shape']) or fortran or
                        dtype != np.dtype('<f4') or
                        offset + int(np.prod(shape)) * 4 != file_stat.st_size):
                    raise ValueError('Window header does not match expected array')
                stream.seek(offset + (shape[0] - 1) * shape[1] * shape[2] * 4)
                weight_bytes = stream.read(shape[1] * shape[2] * 4)
                if sha256(weight_bytes).hexdigest() != self._weight_digest:
                    raise ValueError('Window fusion weights differ from canonical taper')
            self._verified_stamps[index] = stamp
            if len(self._verified_stamps) > 8192:
                self._verified_stamps.popitem(last=False)
            if index in self._planned_indices:
                self._persisted_indices.add(index)
            return True
        except (OSError, ValueError, EOFError, KeyError, TypeError):
            self._invalidate(index)
            return False

    def _load_window(self, index):
        self._check_error()
        state = self._namespace
        with state.condition:
            pending = state.pending.get(index)
        if pending is not None:
            return self._read_pending(pending)
        with state.io_lock:
            with state.condition:
                pending = state.pending.get(index)
            if pending is None:
                return self._load_disk_window(index)
        return self._read_pending(pending)

    def _read_pending(self, pending):
        with span('coarse.persistence.pending_read', world=self.world_hash):
            pending.wait()
            result = pending.cpu.clone()
            self._validate_array(result.numpy())
        with self._namespace.condition:
            self._namespace.pending_hits += 1
        return result

    def _load_disk_window(self, index):
        path = self._window_path(index)
        if not self._valid_metadata_locked(index):
            return None
        try:
            payload = path.read_bytes()
            metadata = json.loads(self._sidecar_path(index).read_text(encoding='utf-8'))
            if sha256(payload).hexdigest() != metadata['sha256']:
                raise ValueError('Window digest mismatch')
            value = np.load(BytesIO(payload), allow_pickle=False)
            if (value.dtype != np.float32 or
                    tuple(value.shape) != tuple(self._tensor.output_window.size) or
                    not np.isfinite(value).all() or
                    not np.array_equal(value[-1], self._expected_weight)):
                raise ValueError('Invalid learned coarse output')
            self.disk_hits += 1
            return torch.from_numpy(np.array(value, copy=True))
        except (OSError, ValueError, EOFError, KeyError, TypeError):
            self._invalidate(index)
            return None

    def _save_window(self, index, value) -> bool:
        """Accept a bounded immutable snapshot; True means accepted, not durable.

        Pending snapshots are shared across live worlds and included in quota
        reservations. Full queues block admission, never discard a learned
        window or accumulate unbounded detached GPU outputs.
        """
        state = self._namespace
        index = tuple(index)
        with state.condition:
            if self._closed:
                raise CoarsePersistenceError('Coarse preparation is closed')
            self._check_error()
            if index in state.pending:
                return True
            if len(state.pending) >= state.max_pending:
                start = perf_counter()
                state.backpressure_count += 1
                state.condition.notify_all()
                with span('coarse.persistence.backpressure', world=self.world_hash):
                    while len(state.pending) >= state.max_pending and not state.error and not self._closed:
                        state.condition.wait()
                state.backpressure_seconds += perf_counter() - start
                self._check_error()
                if self._closed:
                    raise CoarsePersistenceError('Coarse preparation is closed')
                if index in state.pending:
                    return True
            path = self._window_path(index)
            previous = path.stat().st_size if path.exists() else 0
            reservation = max(0, self._payload_bytes - previous)
            if state.disk_bytes + state.reserved_bytes + reservation > state.budget_bytes:
                state.disk_budget_exhausted = True
                return False
            with span('coarse.persistence.snapshot', world=self.world_hash, index=list(index)):
                if tuple(value.shape) != tuple(self._tensor.output_window.size):
                    raise RuntimeError('Invalid learned coarse window shape')
                if value.is_cuda:
                    # copy_ is submitted on the producer stream after all model
                    # work. Hold source storage until the transfer event fires.
                    cpu = torch.empty(tuple(value.shape), dtype=torch.float32,
                                      device='cpu', pin_memory=True)
                    source = value.detach()
                    cpu.copy_(source, non_blocking=True)
                    event = torch.cuda.Event()
                    event.record(torch.cuda.current_stream(value.device))
                else:
                    cpu = value.detach().to('cpu', dtype=torch.float32).clone()
                    source = event = None
                    self._validate_array(cpu.numpy())
            state.pending[index] = _PendingWrite(self, index, cpu, source, event, reservation)
            state.reserved_bytes += reservation
            state.submitted += 1
            state.high_water = max(state.high_water, len(state.pending))
            if state.future is None:
                try:
                    state.future = _PERSIST_EXECUTOR.submit(self._drain, state)
                except BaseException:
                    pending = state.pending.pop(index)
                    state.reserved_bytes -= reservation
                    pending.wait()
                    state.condition.notify_all()
                    raise
        if not self.async_persistence:
            self.flush()
        return True

    def _validate_array(self, array):
        if (tuple(array.shape) != tuple(self._tensor.output_window.size) or
                array.dtype != np.float32 or
                not np.isfinite(array).all() or
                not np.array_equal(array[-1], self._expected_weight)):
            raise RuntimeError('Invalid learned coarse window')

    @staticmethod
    def _drain(state):
        while True:
            with state.condition:
                if not state.pending:
                    state.future = None
                    state.condition.notify_all()
                    return
                index, pending = next(iter(state.pending.items()))
            started = perf_counter()
            succeeded = False
            disk_delta = 0
            try:
                with span('coarse.persistence.transfer_wait', world=pending.owner.world_hash):
                    pending.wait()
                with state.condition:
                    state.transfer_wait_seconds += perf_counter() - started
                pending.source = None
                started = perf_counter()
                with span('coarse.persistence.commit', world=pending.owner.world_hash, index=list(index)):
                    with state.io_lock:
                        disk_delta = pending.owner._commit_window(index, pending.cpu.numpy())
                succeeded = True
            except BaseException as exc:
                with state.condition:
                    state.errors += 1
                    state.error = f'{type(exc).__name__}: {exc}'[:1000]
                # Accounting failures must not kill the drain with unrelated
                # accepted writes still waiting forever in the queue.
                try:
                    with state.io_lock:
                        pending.owner._invalidate(index)
                        actual_bytes = sum(path.stat().st_size for path in pending.owner.windows_dir.glob('*.npy'))
                        with state.condition:
                            state.disk_bytes = actual_bytes
                except OSError:
                    pass
            finally:
                with state.condition:
                    state.write_seconds += perf_counter() - started
                    state.committed += int(succeeded)
                    if succeeded:
                        state.disk_bytes += disk_delta
                    state.reserved_bytes -= pending.reserved_bytes
                    state.pending.pop(index, None)
                    state.condition.notify_all()

    def _commit_window(self, index, array):
        self._validate_array(array)
        buffer = BytesIO()
        np.save(buffer, array, allow_pickle=False)
        path = self._window_path(index)
        previous = path.stat().st_size if path.exists() else 0
        payload = buffer.getvalue()
        if len(payload) != self._payload_bytes:
            raise RuntimeError('Coarse payload size differs from reserved bytes')
        _atomic(path, payload)
        file_stat = path.stat()
        sidecar = dict(schema=SCHEMA, world_hash=self.world_hash,
                       index=list(index), shape=list(self._tensor.output_window.size),
                       dtype='float32', weight_sha256=self._weight_digest,
                       bytes=len(payload), mtime_ns=file_stat.st_mtime_ns,
                       sha256=sha256(payload).hexdigest())
        _atomic(self._sidecar_path(index), _json_bytes(sidecar))
        self._verified_stamps.pop(index, None)
        if not self._valid_metadata_locked(index):
            raise RuntimeError('Newly persisted coarse window failed verification')
        if index in self._planned_indices:
            self._persisted_indices.add(index)
        return len(payload) - previous

    def _check_error(self):
        state = self._namespace
        if state is not None and state.error:
            raise CoarsePersistenceError(f'Coarse persistence failed: {state.error}')

    def flush(self, timeout=None):
        """Wait for shared namespace commits; propagate asynchronous failures."""
        state = self._namespace
        if state is None:
            return
        deadline = None if timeout is None else perf_counter() + timeout
        with state.condition:
            while state.pending:
                remaining = None if deadline is None else deadline - perf_counter()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError('Timed out flushing coarse persistence')
                state.condition.wait(remaining)
        self._check_error()

    def close(self, timeout=None):
        """Reject this producer's new writes, drain shared writes, report errors.

        Other preparations sharing this namespace remain usable. Repeated
        close/flush calls are safe; close must precede world/CUDA teardown.
        """
        if self._namespace is not None:
            with self._namespace.condition:
                self._closed = True
                self._namespace.condition.notify_all()
        self.flush(timeout)

    def _save_cursor(self):
        _atomic(self._cursor_path, _json_bytes(dict(
            schema=SCHEMA, world_hash=self.world_hash, cursor=self._cursor,
            total_windows=len(self._indices))))

    def prioritize(self, mask, focus=None):
        """Reorder missing windows around the camera; keep all sea windows.

        Classification uses the entire overlapping model footprint and one
        conservative land-mask cell around it. Already committed/pending
        windows still get skipped by step, with unchanged fusion and storage.
        The caller owns the GPU lock; view updates only publish a new focus.
        """
        signature = (id(mask), tuple(focus) if focus is not None else None)
        if signature == self._priority_signature:
            return
        if self._priority_geometry is None or self._priority_signature[0] != id(mask):
            x0, y0, x1, y1 = mask['bounds']
            dx, dy = (x1-x0)/mask['width'], (y1-y0)/mask['height']
            window = self._tensor.output_window
            geometry = []
            for index in self._indices:
                left = index[-1]*window.stride[-1]*COARSE_METRES
                top = index[-2]*window.stride[-2]*COARSE_METRES
                right = left+window.size[-1]*COARSE_METRES
                bottom = top+window.size[-2]*COARSE_METRES
                ix0 = max(0, min(mask['width']-1, math.floor((left-x0)/dx)-1))
                ix1 = max(ix0+1, min(mask['width'], math.ceil((right-x0)/dx)+1))
                iy0 = max(0, min(mask['height']-1, math.floor((top-y0)/dy)-1))
                iy1 = max(iy0+1, min(mask['height'], math.ceil((bottom-y0)/dy)+1))
                sea = not any('1' in mask['rows'][y][ix0:ix1] for y in range(iy0,iy1))
                geometry.append((index, sea, left, top, right, bottom))
            self._priority_geometry = geometry
        bounds = focus if focus is not None else self.bounds
        cx, cy = (bounds[0]+bounds[2])/2, (bounds[1]+bounds[3])/2
        def rank(item):
            index, sea, left, top, right, bottom = item
            visible = left < bounds[2] and right > bounds[0] and top < bounds[3] and bottom > bounds[1]
            distance = ((left+right)/2-cx)**2+((top+bottom)/2-cy)**2
            return sea, not visible, distance, index
        self._priority_order = tuple(item[0] for item in sorted(self._priority_geometry,key=rank))
        self._priority_cursor = 0
        self._priority_signature = signature

    def step(self, world, budget_windows: int = 1, *, check=None) -> dict:
        """Run at most ``budget_windows`` missing coarse model windows.

        The caller owns its GPU lock. Interest/cancellation is checked before
        each group. A cancelled step leaves atomic completed files reusable.
        """
        if world.coarse is not self._tensor:
            raise ValueError('Preparation is not installed on this world')
        if budget_windows < 1:
            raise ValueError('budget_windows must be positive')
        self._check_error()
        if self._closed:
            raise CoarsePersistenceError('Coarse preparation is closed')
        total = len(self._indices)
        if len(self._persisted_indices) >= total or self.disk_budget_exhausted:
            return self.status()
        generated_before = self.network_windows + self.remote_windows
        examined = 0
        while examined < total and self.network_windows + self.remote_windows - generated_before < budget_windows:
            group = []
            remaining = budget_windows - (self.network_windows + self.remote_windows - generated_before)
            group_size = min(remaining, self._tensor.batch_size or 1)
            while examined < total and len(group) < group_size:
                if self._priority_order is None:
                    index = self._indices[self._cursor]
                else:
                    index = self._priority_order[self._priority_cursor]
                    self._priority_cursor = (self._priority_cursor+1) % total
                self._cursor = (self._cursor + 1) % total
                examined += 1
                with self._namespace.condition:
                    pending = index in self._namespace.pending
                if index not in self._persisted_indices and not pending:
                    group.append(index)
            if not group:
                break
            if check is not None:
                check()
            # Use the installed real InfiniteTensor scheduler and cache. The
            # weighted window is persisted by the wrapped coarse f.
            scheduler = getattr(world, '_terrain_window_scheduler', None)
            if scheduler is not None:
                if len(group) == 1:
                    scheduler.ensure_window(self._tensor, group[0], check=check)
                else:
                    scheduler.ensure_windows(self._tensor, group, check=check)
            else:
                self._tensor._store.begin_access(self._tensor.uuid)
                try:
                    self._tensor._ensure_processed(group)
                finally:
                    self._tensor._store.end_access(self._tensor.uuid)
            for index in group:
                with self._namespace.condition:
                    pending = index in self._namespace.pending
                if not pending and self._load_window(index) is None:
                    if self.disk_budget_exhausted:
                        break
                    # Repair RAM-only materializations or damaged disk windows.
                    if check is not None:
                        check()
                    start = perf_counter()
                    if self.remote is not None:
                        self.remote([index])
                        self.model_submission_seconds += perf_counter() - start
                        self.remote_windows += 1
                        continue
                    output = self._model_f([index])[0]
                    self.model_submission_seconds += perf_counter() - start
                    persist_start = perf_counter()
                    self._save_window(index, output)
                    self.persistence_seconds += perf_counter() - persist_start
                    self.network_windows += 1
            if self.disk_budget_exhausted:
                break
        with self._namespace.condition:
            pending_plan = set(self._namespace.pending).intersection(self._planned_indices)
            all_submitted = len(self._persisted_indices | pending_plan) >= total
        if all_submitted:
            # One final drain gives background completion its durable meaning;
            # intermediate quanta leave disk work off the compute lane.
            self.flush()
        self._save_cursor()
        return dict(self.status(), quantum_windows=self.network_windows+self.remote_windows-generated_before)

    def _required_indices(self, i1, j1, i2, j2):
        region = (slice(0, self._tensor.shape[0]), slice(i1, i2), slice(j1, j2))
        return self._tensor.output_window.intersecting_windows(
            region, tensor_shape=self._tensor.shape)

    def complete(self, i1, j1, i2, j2) -> bool:
        """True only when every contributor for this coarse rectangle is saved."""
        return all(self._valid_metadata(index)
                   for index in self._required_indices(i1, j1, i2, j2))

    def ready_for_samples(self, xs_native, ys_native, climate_halo: int = 8) -> bool:
        """Check sparse height and bounded climate sample groups without NN work.

        ``xs_native``/``ys_native`` are 1-D native 30 m pixel coordinates, as
        passed to the server's coarse sampler. Every bilinear neighbour and
        the 15-cell climate baseline halo must have all weighted contributors.
        Groups follow the server's 48-cell sparse block partition, avoiding a
        false requirement to prepare the entire space between distant samples.
        """
        xs = np.asarray(xs_native, dtype=np.float64)
        ys = np.asarray(ys_native, dtype=np.float64)
        if xs.ndim != 1 or ys.ndim != 1 or not len(xs) or not len(ys):
            raise ValueError('Sample coordinates must be nonempty 1-D arrays')
        if not np.isfinite(xs).all() or not np.isfinite(ys).all():
            raise ValueError('Nonfinite sample coordinates')
        if climate_halo < 0:
            raise ValueError('climate_halo must be nonnegative')
        checked = set()
        def complete_once(i1,j1,i2,j2):
            for index in self._required_indices(i1,j1,i2,j2):
                if index not in checked:
                    if not self._valid_metadata(index):
                        return False
                    checked.add(index)
            return True
        xc, yc = xs / 256 - .5, ys / 256 - .5
        # The server reads one dense rectangle below this area threshold.
        # Group-only readiness would otherwise miss contributors in its middle.
        dense_i1 = math.floor(float(yc.min())) - 1
        dense_i2 = math.ceil(float(yc.max())) + 2
        dense_j1 = math.floor(float(xc.min())) - 1
        dense_j2 = math.ceil(float(xc.max())) + 2
        if (dense_i2-dense_i1) * (dense_j2-dense_j1) <= 512 * 512:
            if not complete_once(dense_i1, dense_j1, dense_i2, dense_j2):
                return False
        # A caller with wider than 16-cell sample gaps may have climate points
        # between them; require the entire climate rectangle conservatively.
        if ((len(xs) > 1 and np.max(np.diff(np.sort(xs))) > 4096) or
                (len(ys) > 1 and np.max(np.diff(np.sort(ys))) > 4096)):
            return complete_once(
                math.floor(float(ys.min() / 256)) - climate_halo,
                math.floor(float(xs.min() / 256)) - climate_halo,
                math.ceil(float(ys.max() / 256)) + 1 + climate_halo,
                math.ceil(float(xs.max() / 256)) + 1 + climate_halo)
        x_groups, y_groups = np.floor(xc / 48).astype(np.int64), np.floor(yc / 48).astype(np.int64)
        for yg in np.unique(y_groups):
            selected_y = ys[y_groups == yg]
            centres_y = yc[y_groups == yg]
            elev_i1 = math.floor(float(centres_y.min())) - 1
            elev_i2 = math.ceil(float(centres_y.max())) + 2
            clim_i1 = math.floor(float(selected_y.min() / 256)) - climate_halo
            clim_i2 = math.ceil(float(selected_y.max() / 256)) + 1 + climate_halo
            for xg in np.unique(x_groups):
                selected_x = xs[x_groups == xg]
                centres_x = xc[x_groups == xg]
                elev_j1 = math.floor(float(centres_x.min())) - 1
                elev_j2 = math.ceil(float(centres_x.max())) + 2
                clim_j1 = math.floor(float(selected_x.min() / 256)) - climate_halo
                clim_j2 = math.ceil(float(selected_x.max() / 256)) + 1 + climate_halo
                if not complete_once(min(elev_i1, clim_i1), min(elev_j1, clim_j1),
                                     max(elev_i2, clim_i2), max(elev_j2, clim_j2)):
                    return False
        return True

    def read_mip(self, world, level: int, i1: int, j1: int, i2: int, j2: int,
                 *, generate: bool = False, check=None) -> dict | None:
        """Average decoded learned coarse metres in aligned 2**level blocks.

        The result is a coarse approximation, never labelled as a final DEM
        mip. Without ``generate``, a partial source returns None rather than a
        biased average of available windows.
        """
        if not 0 <= level <= 12 or i2 <= i1 or j2 <= j1:
            raise ValueError('Invalid mip level or rectangle')
        scale = 1 << level
        si1 = self.grid.i1 + i1 * scale
        sj1 = self.grid.j1 + j1 * scale
        si2 = self.grid.i1 + i2 * scale
        sj2 = self.grid.j1 + j2 * scale
        if (si1 < self.grid.i1 or sj1 < self.grid.j1 or
                si2 > self.grid.i2 or sj2 > self.grid.j2):
            raise ValueError('Mip rectangle outside world grid')
        if (si2 - si1) * (sj2 - sj1) > 1024 * 1024:
            raise ValueError('Mip read exceeds host buffer budget')
        if not generate and not self.complete(si1, sj1, si2, sj2):
            return None
        fused = read_rect(world, 'coarse', si1, sj1, si2, sj2, check=check)
        values = fused.to('cpu', dtype=torch.float32).numpy()
        denominator = values[-1]
        if not np.isfinite(values).all() or not np.all(denominator > 0):
            raise RuntimeError('Incomplete or invalid coarse fusion')
        sqrt_height = values[0] / denominator
        metres = np.sign(sqrt_height) * np.square(sqrt_height)
        mip = metres.reshape(i2-i1, scale, j2-j1, scale).mean(axis=(1, 3))
        return dict(data=np.ascontiguousarray(mip, dtype=np.float32),
                    source='learned-coarse', source_resolution=COARSE_METRES,
                    resolution=COARSE_METRES * scale, level=level,
                    complete=True, final_dem_mip=False, world_hash=self.world_hash)

    def status(self) -> dict:
        complete = len(self._persisted_indices)
        total = len(self._indices)
        persistence = {}
        if self._namespace is not None:
            state = self._namespace
            with state.condition:
                persistence = dict(
                    async_enabled=self.async_persistence,
                    pending_windows=len(state.pending), max_pending=state.max_pending,
                    pending_bytes=sum(p.cpu.numel() * p.cpu.element_size() for p in state.pending.values()),
                    reserved_disk_bytes=state.reserved_bytes,
                    submitted=state.submitted, committed=state.committed,
                    pending_hits=state.pending_hits, high_water=state.high_water,
                    backpressure_count=state.backpressure_count,
                    backpressure_seconds=round(state.backpressure_seconds, 6),
                    transfer_wait_seconds=round(state.transfer_wait_seconds, 6),
                    write_seconds=round(state.write_seconds, 6),
                    errors=state.errors, last_error=state.error)
        return dict(world_hash=self.world_hash, source='learned-coarse',
                    source_resolution=COARSE_METRES, complete_windows=complete,
                    total_windows=total, coverage=complete/total if total else 1.0,
                    cursor=self._cursor, disk_hits=self.disk_hits,
                    network_windows=self.network_windows,
                    remote_windows=self.remote_windows, remote_coarse=self.remote is not None,
                    model_synchronized_seconds=round(self.model_synchronized_seconds, 6),
                    model_submission_seconds=round(self.model_submission_seconds, 6),
                    model_timing='host submission only; no device synchronization',
                    persistence_seconds=round(self.persistence_seconds, 6),
                    persistence_timing='host enqueue/backpressure; worker transfer/write timings in persistence',
                    persistence=persistence,
                    disk_bytes=self._disk_bytes, budget_bytes=self.budget_bytes,
                    disk_budget_exhausted=self.disk_budget_exhausted,
                    final_dem_mip=False)

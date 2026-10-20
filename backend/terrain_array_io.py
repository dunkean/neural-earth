"""Parallel hashing and direct, optionally deferred, npz persistence.

Generation artifacts are immutable. Hash identities and file contents match
the sequential ``sha256(array.tobytes())`` / ``np.savez`` forms exactly; only
the scheduling changes. Deferred writes keep large HDD writes off the request
path; readers consult the in-memory artifact first, then the completed file.
"""
from concurrent.futures import ThreadPoolExecutor
import atexit
import hashlib
import os
from pathlib import Path
import tempfile
import threading
import weakref

import numpy as np

# hashlib, file writes and numpy elementwise kernels release the GIL.
POOL = ThreadPoolExecutor(max_workers=max(1, min(16, os.cpu_count() or 1)), thread_name_prefix='terrain-array-io')
_WRITER = ThreadPoolExecutor(max_workers=1, thread_name_prefix='terrain-persist')
_PENDING = {}
_PENDING_LOCK = threading.Lock()
# Enabled by the long-running server, which flushes before exiting. Library
# callers and tests keep synchronous, immediately visible persistence.
deferred = False


# Digests of read-only arrays, keyed by identity: stage layers composed into
# an atlas are hashed once. A weak reference guards against reused ids.
_DIGESTS = {}
_DIGESTS_LOCK = threading.Lock()


def _frozen(array):
    # Neither this array nor any array it views can change.
    while isinstance(array, np.ndarray):
        if array.flags.writeable:
            return False
        array = array.base
    return True


def sha256_array(array):
    array = np.asarray(array)
    frozen = _frozen(array)
    with _DIGESTS_LOCK:
        known = _DIGESTS.get(id(array))
    if frozen and known is not None and known[0]() is array:
        return known[1]
    digest = hashlib.sha256(array.data if array.flags.c_contiguous else array.tobytes()).hexdigest()
    if frozen:
        key = id(array)
        def forget(_, key=key):
            with _DIGESTS_LOCK:
                if key in _DIGESTS and _DIGESTS[key][0]() is None:
                    del _DIGESTS[key]
        with _DIGESTS_LOCK:
            _DIGESTS[key] = (weakref.ref(array, forget), digest)
    return digest


def sha256_arrays(arrays):
    names = list(arrays)
    return dict(zip(names, POOL.map(sha256_array, [arrays[k] for k in names])))


def parallel_rows(function, rows, *, chunks=None):
    """Evaluate ``function(start, stop)`` over row blocks and concatenate.

    Small blocks keep numpy temporaries cache-resident, which matters more
    than the thread count for long elementwise chains.

    Only for row-local computations: every output row depends on its input row
    alone, so block boundaries cannot change any value.
    """
    chunks = chunks or POOL._max_workers
    bounds = np.linspace(0, rows, min(rows, chunks)+1).astype(int)
    parts = list(POOL.map(lambda i: function(bounds[i], bounds[i+1]), range(len(bounds)-1)))
    if isinstance(parts[0], dict):
        return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    return np.concatenate(parts)


def _write_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.tmp', delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns


def save_npz(path, arrays, *, on_complete=None):
    """Atomically write an uncompressed npz; deferred when enabled.

    Returns the file's (size, mtime_ns) signature, or None while deferred;
    ``on_complete`` then receives it once the file is in place.
    """
    path = Path(path)
    if not deferred:
        signature = _write_npz(path, arrays)
        if on_complete is not None:
            on_complete(signature)
        return signature
    holder = []
    def task():
        try:
            signature = _write_npz(path, arrays)
            if on_complete is not None:
                on_complete(signature)
        except Exception as error:
            print(f'Deferred write of {path} failed: {error}', flush=True)
            raise
        finally:
            with _PENDING_LOCK:
                # Registration completes under this lock before we can get here.
                if _PENDING.get(path) is holder[0]:
                    del _PENDING[path]
    with _PENDING_LOCK:
        holder.append(_WRITER.submit(task))
        _PENDING[path] = holder[0]
    return None


def pending(path):
    with _PENDING_LOCK:
        return Path(path) in _PENDING


def wait(path, timeout=None):
    """Complete a deferred write of this path, if one is queued."""
    with _PENDING_LOCK:
        future = _PENDING.get(Path(path))
    if future is not None:
        future.result(timeout=timeout)


def flush(timeout=None):
    """Complete every deferred write; called before process exit."""
    with _PENDING_LOCK:
        futures = list(_PENDING.values())
    for future in futures:
        future.result(timeout=timeout)


atexit.register(flush)

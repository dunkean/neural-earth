"""Bounded RAM delivery and ordered persistence of regenerable physical tiles.

A full disk queue skips persistence; it never blocks generation. One writer
preserves replacement order. The supplied writer commits the receipt last.
"""
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import threading
from terrain_profiling import span, trace


class PhysicalDelivery:
    def __init__(self, max_bytes=48*1024**2, max_pending=8):
        self.max_bytes = max_bytes
        self.entries = OrderedDict()
        self.bytes = 0
        self.lock = threading.Lock()
        self.slots = threading.BoundedSemaphore(max_pending)
        self.writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix='terrain-persist')
        self.pending = 0
        self.written = self.skipped = self.failed = 0
        self.memory_only = 0
        self.errors = deque(maxlen=8)

    def get(self, key):
        with self.lock:
            entry = self.entries.get(key)
            if entry:
                self.entries.move_to_end(key)
                return entry[0], dict(entry[1]), entry[2]

    def publish(self, key, path, report, arrays, write, identifier=None):
        size = sum(a.nbytes for a in arrays)
        with self.lock:
            previous = self.entries.pop(key, None)
            if previous:
                self.bytes -= previous[3]
            if size <= self.max_bytes:
                self.entries[key] = (path, dict(report), arrays, size)
                self.bytes += size
            while self.bytes > self.max_bytes:
                self.bytes -= self.entries.popitem(last=False)[1][3]
        if write is None:
            with self.lock:
                self.memory_only += 1
            return
        if not self.slots.acquire(blocking=False):
            with self.lock:
                self.skipped += 1
            return
        with self.lock:
            self.pending += 1
        def persist():
            try:
                with trace(identifier), span('tile.persist', key=key):
                    write()
                with self.lock:
                    self.written += 1
            except Exception as exc:
                with self.lock:
                    self.failed += 1
                    self.errors.append(str(exc)[:400])
            finally:
                with self.lock:
                    self.pending -= 1
                self.slots.release()
        try:
            self.writer.submit(persist)
        except RuntimeError:
            with self.lock:
                self.pending -= 1
            self.slots.release()
            raise

    def status(self):
        with self.lock:
            return dict(bytes=self.bytes, max_bytes=self.max_bytes, entries=len(self.entries),
                        pending=self.pending, written=self.written, skipped=self.skipped,
                        memory_only=self.memory_only,
                        failed=self.failed, errors=list(self.errors))

    def flush(self):
        self.writer.submit(lambda: None).result(timeout=300)

    def close(self):
        self.writer.shutdown(wait=True)

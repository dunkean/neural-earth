"""Quota for regenerable tile groups in one explicitly selected cache root.

Initial exports, model files and other numerical profiles are never scanned or
removed. File IO is pinned until the response closes, including Windows streams.
"""
from dataclasses import dataclass, field
from pathlib import Path
import re
import threading
import time

WORLD_PROFILES = frozenset(('natural', 'terrestrial-gondwana', 'terrestrial-continents',
                            'terrestrial-earthlike', 'terrestrial-archipelago'))


@dataclass
class CacheGroup:
    files: dict = field(default_factory=dict)
    accessed: float = 0


class CacheLease:
    def __init__(self, owner, key):
        self.owner, self.key, self.released = owner, key, False
        with owner.condition:
            owner.pins[key] = owner.pins.get(key, 0)+1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()

    def release(self):
        with self.owner.condition:
            if self.released:
                return
            self.released = True
            count = self.owner.pins.get(self.key, 1)-1
            if count:
                self.owner.pins[self.key] = count
            else:
                self.owner.pins.pop(self.key, None)
            self.owner.condition.notify_all()


class TerrainDiskCache:
    def __init__(self, root, version, budget_bytes=16*1024**3,
                 protected_keys=lambda: set(), grace_seconds=3):
        self.root = Path(root).resolve()
        self.version, self.budget_bytes = version, int(budget_bytes)
        if self.budget_bytes <= 0:
            raise ValueError('Disk quota must be positive')
        self.protected_keys, self.grace_seconds = protected_keys, grace_seconds
        self.condition = threading.Condition()
        self.groups, self.pins = {}, {}
        self.bytes = 0
        self.evictions = self.evicted_bytes = self.failed_evictions = 0
        self.indexed = False
        self.closed = False
        self.thread = threading.Thread(target=self._run, name='terrain-disk-cache', daemon=True)
        self.thread.start()

    def acquire(self, key):
        return CacheLease(self, key)

    def _safe(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError('Un fichier de cache sort du profil actif')
        return path

    def _read_files(self, paths):
        files = {}
        for path in paths:
            path = self._safe(path)
            try:
                stat = path.stat()
                if path.is_file():
                    files[path] = stat.st_size
            except FileNotFoundError:
                pass
        return files

    def record(self, key, paths):
        """Constant-size bookkeeping; a group contains at most four tile files."""
        files = self._read_files(paths)
        with self.condition:
            group = self.groups.setdefault(key, CacheGroup())
            self.bytes += sum(files.values())-sum(group.files.values())
            group.files = files
            group.accessed = time.time()
            self.condition.notify_all()

    def _key(self, path):
        parts = path.relative_to(self.root).parts
        world_profile=''
        if parts and (parts[0] in WORLD_PROFILES or any(
                re.fullmatch(re.escape(profile)+r'--g[0-9a-f]{24}', parts[0])
                for profile in WORLD_PROFILES)):
            world_profile=parts[0]+'/'
            parts=parts[1:]
        if parts and parts[0] == 'physical-v1':
            parts = parts[1:]
        if len(parts) == 3 and parts[0].isdigit() and re.fullmatch(r'-?\d+(?:-source3)?', parts[1]):
            match = re.fullmatch(r'(-?\d+)_(-?\d+)\.(?:(?:climate|relief|biomes|temperature|precipitation)\.)?(npy|json|png)', parts[2])
            if match:
                level=parts[1].split('-source3')[0]
                suffix='/source3' if parts[1].endswith('-source3') else ''
                return f'{self.version}/{world_profile}{int(parts[0])}/{int(level)}/{int(match[1])}/{int(match[2])}{suffix}'
        if len(parts)==2 and parts[0].isdigit() and (parts[1]=='world.json' or re.fullmatch(r'overview(?:\.(relief|biomes|temperature|precipitation))?\.(png|json)',parts[1])):
            return f'{self.version}/{world_profile}{int(parts[0])}/overview'
        return None

    def _index(self):
        # Only this numerical profile is walked, once, on a background thread.
        for candidate in self.root.rglob('*'):
            if self.closed:
                return
            try:
                path = self._safe(candidate)
                if not path.is_file():
                    continue
                key = self._key(path)
                if key is None:
                    continue
                stat = path.stat()
                with self.condition:
                    group = self.groups.setdefault(key, CacheGroup())
                    old = group.files.get(path, 0)
                    group.files[path] = stat.st_size
                    self.bytes += stat.st_size-old
                    group.accessed = max(group.accessed, stat.st_mtime)
            except (OSError, ValueError):
                # External symlinks and transient writes are ignored, never deleted.
                continue
        with self.condition:
            self.indexed = True
            self.condition.notify_all()

    def trim(self):
        protected = self.protected_keys()
        now = time.time()
        with self.condition:
            # Never race startup stat snapshots with unlink/reinsertion of a group.
            if not self.indexed or self.bytes <= self.budget_bytes:
                return
            candidates = sorted(((key, group) for key, group in self.groups.items()
                                 if not self.pins.get(key) and key not in protected
                                 and now-group.accessed >= self.grace_seconds),
                                key=lambda item: item[1].accessed)
            for key, group in candidates:
                if self.bytes <= self.budget_bytes:
                    break
                removed = 0
                # The lock prevents a new lease between eligibility and unlink.
                for path, size in list(group.files.items()):
                    try:
                        self._safe(path).unlink(missing_ok=True)
                        group.files.pop(path)
                        removed += size
                    except (OSError, ValueError):
                        self.failed_evictions += 1
                self.bytes -= removed
                self.evicted_bytes += removed
                if not group.files:
                    self.groups.pop(key, None)
                    self.evictions += 1
                else:
                    group.accessed = now  # Back off after Windows sharing errors.

    def _run(self):
        self._index()
        while not self.closed:
            self.trim()
            with self.condition:
                self.condition.wait(1)

    def status(self):
        with self.condition:
            return dict(bytes=self.bytes, budget_bytes=self.budget_bytes, groups=len(self.groups),
                        pinned_groups=len(self.pins), evictions=self.evictions,
                        evicted_bytes=self.evicted_bytes, failed_evictions=self.failed_evictions,
                        indexed=self.indexed, over_budget=self.bytes>self.budget_bytes)

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        self.thread.join(timeout=5)

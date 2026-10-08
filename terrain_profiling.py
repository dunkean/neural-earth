"""Opt-in, bounded request timeline. CUDA timings never synchronize serving."""
from collections import deque
from contextlib import contextmanager, nullcontext
import os
import threading
import time

ENABLED = os.environ.get('TERRAIN_PROFILE', '0') == '1'
CUDA = os.environ.get('TERRAIN_PROFILE_CUDA', '0') == '1'
_events = deque(maxlen=8192)
_pending = deque(maxlen=256)
_lock = threading.Lock()
_local = threading.local()
_noop = nullcontext()
_dropped_events = 0
_dropped_cuda = 0


def set_enabled(enabled, cuda=False):
    global ENABLED, CUDA
    ENABLED, CUDA = bool(enabled), bool(cuda)


def _append(event):
    global _dropped_events
    with _lock:
        _dropped_events += len(_events) == _events.maxlen
        _events.append(event)


def instant(name, **fields):
    if ENABLED:
        _append(dict(name=name, ph='i', ts=time.time_ns()/1000,
                     trace=getattr(_local, 'trace', None), args=fields))


@contextmanager
def _span(name, fields, gpu):
    global _dropped_cuda
    started = time.perf_counter_ns()
    epoch = time.time_ns()/1000
    trace = getattr(_local, 'trace', None)
    begin = end = None
    if gpu and CUDA:
        import torch
        if torch.cuda.is_available():
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        if end is not None:
            end.record()
        event = dict(name=name, ph='X', ts=epoch,
                     dur=(time.perf_counter_ns()-started)/1000,
                     trace=trace, args=dict(fields, failed=failed))
        _append(event)
        if end is not None:
            with _lock:
                _dropped_cuda += len(_pending) == _pending.maxlen
                _pending.append((begin, end, name, trace, epoch))


def span(name, *, gpu=False, **fields):
    return _span(name, fields, gpu) if ENABLED else _noop


@contextmanager
def measured_lock(lock, name='lock.wait'):
    with span(name):
        lock.__enter__()
    try:
        yield
    finally:
        lock.__exit__(None, None, None)


@contextmanager
def trace(identifier):
    previous = getattr(_local, 'trace', None)
    _local.trace = identifier
    try:
        yield
    finally:
        _local.trace = previous


def snapshot():
    """Read ready events; intervals include GPU submission gaps, not pure kernels."""
    with _lock:
        ready, waiting = [], []
        global _dropped_events
        for begin, end, name, identifier, epoch in _pending:
            if end.query():
                ready.append(dict(name=name+'.cuda_interval', trace=identifier,
                                  ts=epoch,
                                  milliseconds=begin.elapsed_time(end)))
            else:
                waiting.append((begin, end, name, identifier, epoch))
        _pending.clear()
        _pending.extend(waiting)
        for e in ready:
            _dropped_events += len(_events) == _events.maxlen
            _events.append(dict(name=e.pop('name'), ph='i', ts=e.pop('ts'),
                                trace=e.pop('trace'), args=e))
        return dict(enabled=ENABLED, cuda_intervals=CUDA, capacity=_events.maxlen,
                    dropped_events=_dropped_events, dropped_cuda=_dropped_cuda,
                    pending_cuda=len(_pending), clock='epoch microseconds; monotonic durations',
                    notes='CUDA intervals may include submission gaps; draw submission is not presentation.',
                    traceEvents=[dict(e, pid=1, tid=str(e.get('trace') or 'server')) for e in _events])

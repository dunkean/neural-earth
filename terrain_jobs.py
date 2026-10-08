"""Small, backend-independent queue for shared terrain requests.

Only the compute lane is serialized. CPU encoders run independently afterwards.
Camera subscriptions cancel *queued* work; an already launched compute finishes
and may populate the cache. This is intentionally not a neural-stage DAG yet.
"""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import threading
import time
from terrain_profiling import instant, span, trace


class JobCancelled(Exception):
    pass


class QueueFull(Exception):
    pass


@dataclass
class Job:
    key: str
    compute: object
    finalize: object
    priority: float
    sequence: int
    created: float = field(default_factory=time.monotonic)
    subscribers: set = field(default_factory=set)
    legacy: bool = False
    state: str = 'queued'
    event: object = field(default_factory=threading.Event)
    result: object = None
    error: object = None
    started: float = 0
    compute_seconds: float = 0
    lane: str = 'gpu'
    admitted: float = 0
    slot_wait_seconds: float = 0
    queue_seconds: float = 0

    def wait(self, timeout=300):
        if not self.event.wait(timeout):
            raise TimeoutError('Le calcul de terrain a dépassé le délai de réponse')
        if self.error:
            raise self.error
        return self.result


class TerrainJobs:
    def __init__(self, max_pending=128, session_ttl=35, cpu_workers=2):
        self.condition = threading.Condition()
        self.jobs = OrderedDict()
        self.views = {}
        self.max_pending = max_pending
        self.session_ttl = session_ttl
        self.sequence = 0
        self.current = threading.local()
        self.closed = False
        self.worker = None
        self.preview_workers = []
        self.cpu = ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix='terrain-encode')
        self.preview_cpu = ThreadPoolExecutor(max_workers=cpu_workers, thread_name_prefix='terrain-preview-encode')
        # Avoid unbounded detached elevation arrays if rendering is slower than CUDA.
        self.cpu_slots = threading.Semaphore(cpu_workers + 1)
        self.preview_slots = threading.Semaphore(cpu_workers + 1)
        self.metrics = dict(submitted=0, deduplicated=0, cancelled_queued=0,
                            cancelled_computing=0,
                            completed=0, failed=0, obsolete_completed=0,
                            compute_seconds=0.0, encode_seconds=0.0,
                            last_queue_seconds=0.0)

    def update_view(self, session, epoch, wants):
        """wants is {job_key: priority}; old epochs cannot resurrect old work."""
        with self.condition:
            old = self.views.get(session)
            if old and epoch < old['epoch']:
                return False
            if old and epoch == old['epoch']:
                old['touched'] = time.monotonic()
                return True
            if not old and len(self.views) >= 32:
                self._expire_locked()
                if len(self.views) >= 32:
                    raise QueueFull('Trop de sessions actives')
            self.views[session] = dict(epoch=epoch, wants=wants, touched=time.monotonic())
            for job in list(self.jobs.values()):
                if session in job.subscribers and job.key not in wants:
                    job.subscribers.discard(session)
                if session in job.subscribers:
                    self._priority_locked(job)
                self._cancel_if_unused_locked(job)
            self.condition.notify_all()
            return True

    def release_view(self, session):
        with self.condition:
            self._release_locked(session)
            self.condition.notify_all()

    def _release_locked(self, session):
        self.views.pop(session, None)
        for job in list(self.jobs.values()):
            job.subscribers.discard(session)
            self._priority_locked(job)
            self._cancel_if_unused_locked(job)

    def _expire_locked(self):
        now = time.monotonic()
        for session, view in list(self.views.items()):
            if now - view['touched'] > self.session_ttl:
                self._release_locked(session)

    def _priority_locked(self, job):
        priorities = [self.views[s]['wants'][job.key] for s in job.subscribers
                      if s in self.views and job.key in self.views[s]['wants']]
        if priorities:
            job.priority = min(priorities)

    def _cancel_if_unused_locked(self, job):
        if job.state == 'queued' and not job.subscribers and not job.legacy:
            job.state = 'cancelled'
            job.error = JobCancelled('La caméra ne demande plus cette tuile')
            job.event.set()
            self.jobs.pop(job.key, None)
            self.metrics['cancelled_queued'] += 1

    def submit(self, key, compute, finalize=lambda value: value,
               session=None, epoch=0, priority=2000, lane='gpu'):
        if lane not in ('gpu', 'cpu'):
            raise ValueError('Unknown terrain compute lane')
        with self.condition:
            self._expire_locked()
            if self.closed:
                raise JobCancelled('Le moteur est arrêté')
            if session:
                view = self.views.get(session)
                if not view or epoch > view['epoch'] or key not in view['wants']:
                    raise JobCancelled('Cette demande appartient à une ancienne caméra')
                view['touched'] = time.monotonic()
                priority = view['wants'][key]
            job = self.jobs.get(key)
            deduplicated = job is not None
            if job:
                self.metrics['deduplicated'] += 1
            else:
                if len(self.jobs) >= self.max_pending:
                    raise QueueFull('File de terrain pleine')
                self.sequence += 1
                job = Job(key, compute, finalize, priority, self.sequence)
                job.lane = lane
                self.jobs[key] = job
                self.metrics['submitted'] += 1
            if session:
                job.subscribers.add(session)
            else:
                job.legacy = True
            self._priority_locked(job)
            instant('job.deduplicated' if deduplicated else 'job.admitted', key=key, sequence=job.sequence, lane=job.lane)
            if job.lane == 'gpu' and self.worker is None:
                self.worker = threading.Thread(target=self._run, name='terrain-compute', daemon=True)
                self.worker.start()
            if job.lane == 'cpu' and not self.preview_workers:
                for index in range(2):
                    worker = threading.Thread(target=self._run, args=('cpu',), name=f'terrain-preview-{index}', daemon=True)
                    self.preview_workers.append(worker)
                    worker.start()
            self.condition.notify_all()
            return job

    def _run(self, lane='gpu'):
        slots = self.cpu_slots if lane == 'gpu' else self.preview_slots
        encoder = self.cpu if lane == 'gpu' else self.preview_cpu
        while True:
            with self.condition:
                self._expire_locked()
                ready = [j for j in self.jobs.values() if j.state == 'queued' and j.lane == lane]
                if not ready:
                    if self.closed:
                        return
                    self.condition.wait(1)
                    continue
                # Priority classes occupy bands of 1000. Aging only breaks ties
                # within a class and cannot promote background over coverage.
                now = time.monotonic()
                job = min(ready, key=lambda j: (int(j.priority // 1000),
                          j.priority % 1000 - min(50, now-j.created), j.sequence))
                job.state = 'computing'
                job.admitted = now
                job.queue_seconds = now-job.created
                self.metrics['last_queue_seconds'] = job.queue_seconds
            slot_started = time.monotonic()
            with trace(job.sequence), span('job.finalizer_backpressure', lane=lane):
                slots.acquire()
            try:
                self.current.job = job
                job.slot_wait_seconds = time.monotonic()-slot_started
                self.check_current_interest()
                job.started = time.monotonic()
                with trace(job.sequence), span('job.compute', key=job.key, lane=lane):
                    value = job.compute()
                job.compute_seconds = time.monotonic()-job.started
                with self.condition:
                    job.state = 'encoding'
                    self.metrics['compute_seconds'] += job.compute_seconds
                encoder.submit(self._finish, job, value)
            except BaseException as error:
                slots.release()
                self._complete(job, error=error)
            finally:
                self.current.job = None

    def check_current_interest(self):
        """Safe quantum boundary: never interrupt a CUDA kernel or half window."""
        job = getattr(self.current, 'job', None)
        if job is None or job.legacy:
            return
        with self.condition:
            self._expire_locked()
            if not job.subscribers:
                raise JobCancelled('La caméra a changé entre deux blocs de calcul')

    def _finish(self, job, value):
        started = time.monotonic()
        try:
            self.current.job = job
            if job.lane == 'cpu':
                self.check_current_interest()
            with trace(job.sequence), span('job.finalize', key=job.key, lane=job.lane):
                result = job.finalize(value)
            self._complete(job, result=result, encode_seconds=time.monotonic()-started)
        except BaseException as error:
            self._complete(job, error=error, encode_seconds=time.monotonic()-started)
        finally:
            self.current.job = None
            (self.cpu_slots if job.lane == 'gpu' else self.preview_slots).release()

    def _complete(self, job, result=None, error=None, encode_seconds=0):
        with self.condition:
            job.result, job.error = result, error
            cancelled = isinstance(error, JobCancelled)
            job.state = 'cancelled' if cancelled else ('failed' if error else 'done')
            self.metrics['cancelled_computing' if cancelled else ('failed' if error else 'completed')] += 1
            self.metrics['encode_seconds'] += encode_seconds
            if not job.subscribers and not job.legacy:
                self.metrics['obsolete_completed'] += 1
            self.jobs.pop(job.key, None)
            job.event.set()
            self.condition.notify_all()

    def status(self):
        with self.condition:
            self._expire_locked()
            return dict(self.metrics, sessions=len(self.views),
                        queued=sum(j.state == 'queued' for j in self.jobs.values()),
                        computing=sum(j.state == 'computing' for j in self.jobs.values()),
                        encoding=sum(j.state == 'encoding' for j in self.jobs.values()),
                        jobs=[dict(key=j.key, state=j.state, priority=j.priority, lane=j.lane,
                                   queue_seconds=j.queue_seconds, slot_wait_seconds=j.slot_wait_seconds,
                                   compute_seconds=j.compute_seconds)
                              for j in self.jobs.values()])

    def protected_keys(self):
        with self.condition:
            self._expire_locked()
            return set(self.jobs).union(*(set(v['wants']) for v in self.views.values()))

    def close(self):
        with self.condition:
            self.closed = True
            for job in list(self.jobs.values()):
                if job.state == 'queued':
                    job.legacy = False
                    job.subscribers.clear()
                    self._cancel_if_unused_locked(job)
            self.condition.notify_all()
        if self.worker:
            self.worker.join(timeout=10)
        for worker in self.preview_workers:
            worker.join(timeout=10)
        self.cpu.shutdown(wait=True)
        self.preview_cpu.shutdown(wait=True)

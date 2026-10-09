"""Latest-request generation, exclusive GPU admission and safe cancellation."""
from collections import OrderedDict
from contextlib import contextmanager
import subprocess
import threading


class GenerationCancelled(Exception):
    pass


class GenerationToken:
    def __init__(self):
        self.cancelled = threading.Event()
        self.lock = threading.Lock()
        self.processes = set()

    def check(self):
        if self.cancelled.is_set():
            raise GenerationCancelled('Generation superseded by a new request')

    def cancel(self):
        self.cancelled.set()
        with self.lock:
            for process in self.processes:
                if process.poll() is None:
                    try:
                        process.terminate()
                    except OSError:
                        pass  # The process can exit between poll and terminate.


class GenerationCoordinator:
    def __init__(self, jobs, background):
        self.jobs, self.background = jobs, background
        self.lock = threading.RLock()
        self.current = None
        self.externally_paused = False
        self.epochs = OrderedDict()

    def begin(self, session=None, epoch=0):
        with self.lock:
            if self.externally_paused:
                raise GenerationCancelled('GPU reserved for a new generation')
            if session:
                if epoch <= self.epochs.get(session, -1):
                    raise GenerationCancelled('Outdated generation request')
                self.epochs[session] = epoch
                self.epochs.move_to_end(session)
                while len(self.epochs) > 64:
                    self.epochs.popitem(last=False)
            if self.current is not None:
                self.current.cancel()
            self.current = token = GenerationToken()
            self.background.stop()
            self.jobs.pause()
            return token

    def finish(self, token):
        with self.lock:
            if self.current is token:
                self.current = None
                if not self.externally_paused:
                    self.jobs.resume()

    def suspend(self, paused):
        with self.lock:
            self.externally_paused = paused
            if paused:
                if self.current is not None:
                    self.current.cancel()
                self.background.stop()
                self.jobs.pause()
            elif self.current is None:
                self.jobs.resume()


_current = threading.local()


@contextmanager
def generation_scope(token):
    previous = getattr(_current, 'token', None)
    _current.token = token
    try:
        token.check()
        yield
        token.check()
    finally:
        _current.token = previous


def check_generation():
    token = getattr(_current, 'token', None)
    if token is not None:
        token.check()


@contextmanager
def cancellable_process(*args, **kwargs):
    """Cancel only child processes belonging to this generation request."""
    token = getattr(_current, 'token', None)
    check_generation()
    process = subprocess.Popen(*args, **kwargs)
    if token is not None:
        with token.lock:
            token.processes.add(process)
            if token.cancelled.is_set() and process.poll() is None:
                process.terminate()
    try:
        with process:
            try:
                check_generation()
                yield process
            except BaseException:
                if process.poll() is None:
                    process.kill()
                raise
    except Exception:
        check_generation()  # Report cancellation rather than a broken pipe.
        raise
    finally:
        if token is not None:
            with token.lock:
                token.processes.discard(process)
    check_generation()


def run_process(args, *, capture_output=False, timeout=None, check=False, **kwargs):
    if capture_output:
        kwargs.update(stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with cancellable_process(args, **kwargs) as process:
        stdout, stderr = process.communicate(timeout=timeout)
    if check and process.returncode:
        raise subprocess.CalledProcessError(process.returncode, args, stdout, stderr)
    return subprocess.CompletedProcess(args, process.returncode, stdout, stderr)

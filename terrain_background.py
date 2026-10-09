"""One-window background quanta through the shared, priority-aware compute lane."""
import threading
import time
from collections import OrderedDict


class CoarseBackground:
    def __init__(self, jobs, compute, status, *, window_quantum=None):
        self.jobs, self.compute, self.status_function = jobs, compute, status
        self.window_quantum = window_quantum
        self.lock = threading.Lock()
        self.generation = 0
        self.task = None
        self.closed = False
        self.focuses = OrderedDict()

    def start(self, seed, profile, max_windows=None):
        if max_windows is not None and not 1 <= max_windows <= 8192:
            raise ValueError('Budget coarse must be between 1 and 8192 windows')
        with self.lock:
            if (self.task and self.task['state'] == 'running' and
                    self.task['seed'] == str(seed) and self.task['world_profile'] == profile and
                    self.task['budget'] == max_windows):
                return dict(self.task)
            self.generation += 1
            token = self.generation
            self.task = dict(seed=str(seed), world_profile=profile, budget=max_windows,
                             quanta=0, state='running', error=None, started=time.time())
            task = self.task
        threading.Thread(target=self._run, args=(token,task),
                         name='coarse-background', daemon=True).start()
        return self.status()

    def set_focus(self, seed, profile, bounds):
        with self.lock:
            key = (int(seed), profile)
            self.focuses[key] = tuple(bounds)
            self.focuses.move_to_end(key)
            while len(self.focuses) > 32:
                self.focuses.popitem(last=False)

    def focus(self, seed, profile):
        with self.lock:
            return self.focuses.get((int(seed), profile))

    def stop(self):
        with self.lock:
            self.generation += 1
            if self.task and self.task['state']=='running':
                self.task['state']='stopped'
        return self.status()

    def _active(self, token):
        with self.lock:
            return not self.closed and self.generation==token

    def _run(self, token, task):
        try:
            quantum = 0
            windows = 0
            while task['budget'] is None or windows < task['budget']:
                if not self._active(token):
                    return
                # Stopping a task cancels the next safe quantum. The currently
                # launched NN window always finishes and persists its result.
                count = self.window_quantum() if self.window_quantum else 1
                if task['budget'] is not None:
                    count = min(count, task['budget']-windows)
                def compute():
                    if not self._active(token):
                        return None
                    if self.window_quantum:
                        return self.compute(int(task['seed']),task['world_profile'],budget_windows=count)
                    return self.compute(int(task['seed']),task['world_profile'])
                key=f"coarse-preparation/{token}/{quantum}"
                # Every visible priority is <5000, including explicitly
                # requested sea. Background only consumes the idle GPU lane.
                result=self.jobs.submit(key,compute,priority=5000).wait()
                if not self._active(token):
                    return
                with self.lock:
                    task['quanta']+=1
                    task['progress']=result
                    windows += result.get('quantum_windows', count) if result else 0
                    task['computed_windows'] = windows
                quantum += 1
                if result and result.get('disk_budget_exhausted'):
                    with self.lock:
                        if self.generation==token:
                            task['state']='disk-budget-exhausted'
                    return
                if result and result['complete_windows']>=result['total_windows']:
                    break
            with self.lock:
                if self.generation==token:
                    task['state']='complete' if task['budget'] is None else 'budget-complete'
        except Exception as exc:
            with self.lock:
                if self.generation==token:
                    task.update(state='failed',error=str(exc))

    def status(self):
        with self.lock:
            if not self.task:
                return {'state':'idle'}
            return dict(self.task, focus_bounds=self.focuses.get(
                (int(self.task['seed']),self.task['world_profile'])))

    def close(self):
        self.stop()
        with self.lock:
            self.closed=True

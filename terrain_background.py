"""One-window background quanta through the shared, priority-aware compute lane."""
import threading
import time


class CoarseBackground:
    def __init__(self, jobs, compute, status):
        self.jobs, self.compute, self.status_function = jobs, compute, status
        self.lock = threading.Lock()
        self.generation = 0
        self.task = None
        self.closed = False

    def start(self, seed, profile, max_windows=64):
        if not 1 <= max_windows <= 8192:
            raise ValueError('Budget coarse must be between 1 and 8192 windows')
        with self.lock:
            self.generation += 1
            token = self.generation
            self.task = dict(seed=str(seed), world_profile=profile, budget=max_windows,
                             quanta=0, state='running', error=None, started=time.time())
            task = self.task
        threading.Thread(target=self._run, args=(token,task),
                         name='coarse-background', daemon=True).start()
        return self.status()

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
            for quantum in range(task['budget']):
                if not self._active(token):
                    return
                # Stopping a task cancels the next safe quantum. The currently
                # launched NN window always finishes and persists its result.
                def compute():
                    if not self._active(token):
                        return None
                    return self.compute(int(task['seed']),task['world_profile'])
                key=f"coarse-preparation/{token}/{quantum}"
                result=self.jobs.submit(key,compute,priority=4000).wait()
                if not self._active(token):
                    return
                with self.lock:
                    task['quanta']+=1
                    task['progress']=result
                if result and result.get('disk_budget_exhausted'):
                    with self.lock:
                        if self.generation==token:
                            task['state']='disk-budget-exhausted'
                    return
                if result and result['complete_windows']>=result['total_windows']:
                    break
            with self.lock:
                if self.generation==token:
                    task['state']='budget-complete'
        except Exception as exc:
            with self.lock:
                if self.generation==token:
                    task.update(state='failed',error=str(exc))

    def status(self):
        with self.lock:
            return dict(self.task) if self.task else {'state':'idle'}

    def close(self):
        self.stop()
        with self.lock:
            self.closed=True

"""Add5k NF4 training blocks while matched terrain quality keeps improving."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from distill.common import DATA, REPO, atomic_json, external_path
from distill.jobs import state
from distill.native_onepass_progress_policy import POLICY, evidence, select
from distill.native_weight_quant import file_sha


def reports(step, selection):
    paths = [DATA/'_base'/f'base-onepass-residual-nf4-{selection}-{step:06d}-{coarse}.json'
             for coarse in ('coarse2', 'coarse4')]
    return evidence([json.loads(path.read_text()) for path in paths])


class Controller:
    def __init__(self, args):
        self.args = args
        self.root = external_path(args.root)
        self.output = self.root/'adaptive'
        self.output.mkdir(exist_ok=True)
        self.stopped = False
        self.child = None
        self.history = []
        self.best = None
        self.plateau = 0
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self.stop)

    def stop(self, signum, frame):
        self.stopped = True
        if self.child is not None and self.child.poll() is None:
            self.child.terminate()

    def status(self, status, **fields):
        atomic_json(self.output/'run-status.json', dict(status=status, **fields))

    def execute(self, command, phase, target):
        with self.args.gpu_lock.open('a') as lock:
            while not self.stopped:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    self.status('waiting_gpu', phase=phase, target=target)
                    time.sleep(2)
            if self.stopped:
                raise InterruptedError('Adaptive continuation interrupted.')
            self.child = subprocess.Popen([sys.executable, '-u', *command], cwd=REPO)
            self.status('running', phase=phase, target=target, child_pid=self.child.pid)
            code = self.child.wait()
            self.child = None
            if code or self.stopped:
                raise RuntimeError(f'{phase} exited{code};inspect before retry.')

    def observe(self, step):
        candidates = [reports(step, selection) for selection in ('ema', 'raw')]
        reference = self.best if self.best is not None else next(value for value in candidates if value['selection'] == 'raw')
        winner, comparison = select(candidates, reference)
        baseline = self.best is None
        if baseline or comparison['improved']:
            self.best = winner
            self.plateau = 0
        else:
            self.plateau += 1
        record = dict(step=step, selection=winner['selection'], comparison=comparison,
                      baseline=baseline, plateau_blocks=self.plateau, best_step=self.best['step'],
                      best_selection=self.best['selection'])
        self.history.append(record)
        atomic_json(self.output/'history.json', self.history)
        atomic_json(self.output/'best.json', dict(step=self.best['step'], selection=self.best['selection'],
            accepted=False, policy=POLICY,
            reports=[str(DATA/'_base'/f'base-onepass-residual-nf4-{self.best["selection"]}-{self.best["step"]:06d}-{coarse}.json')
                     for coarse in ('coarse2', 'coarse4')]))
        print(json.dumps(record), flush=True)
        return comparison['eligible'] and self.plateau < POLICY['plateau_patience']

    def plates(self, root, step):
        report_paths = []
        for selection in ('ema', 'raw'):
            checkpoint = root/'train'/('raw' if selection == 'raw' else '')/f'step{step:06d}.pt'
            pack = root/'packed'/f'{selection}-{step:06d}'
            audit_path = pack/'audit.json'
            if not audit_path.exists():
                if pack.exists():
                    raise ValueError('Incomplete pack;preserve and inspect it.')
                self.execute(['-m', 'distill.native_onepass_residual_export', '--checkpoint', str(checkpoint),
                              '--output', str(pack)], f'export:{step}:{selection}', step)
            audit = json.loads(audit_path.read_text())
            if audit['status'] != 'complete' or audit['source_sha256'] != file_sha(checkpoint):
                raise ValueError('Packed checkpoint audit mismatch.')
            for coarse in ('coarse4', 'coarse2'):
                report = DATA/'_base'/f'base-onepass-residual-nf4-{selection}-{step:06d}-{coarse}.json'
                if not report.exists():
                    self.execute(['-m', 'distill.native_progressive_evaluate', '--checkpoint', str(checkpoint),
                                  '--packed', str(pack), '--coarse', coarse,
                                  '--label', 'onepass-residual-nf4-'+selection], f'quality:{step}:{selection}:{coarse}', step)
                saved = json.loads(report.read_text())
                if (not report.with_suffix('.png').exists() or saved['representation'] != 'packed-nf4' or
                        saved['checkpoint_sha256'] != audit['source_sha256']):
                    raise ValueError('Plate does not match the audited packed checkpoint.')
                from distill.native_onepass_gallery import publish
                publish(report, report.with_suffix('')/'gallery')
                report_paths.append(str(report))
        atomic_json(root/f'milestone{step:06d}.json', dict(status='complete', step=step,
                    reports=report_paths, accepted=False))

    def stage(self, previous, start, target):
        root = self.output/f'stage-{target:06d}'
        finished = root/'run-status.json'
        if finished.exists():
            status = json.loads(finished.read_text())
            if status['status'] == 'complete' and status['step'] == target:
                return root
            if status['status'] in ('failed', 'interrupted'):
                raise ValueError('Previous extension failed or was interrupted;inspect before explicit resumption.')
        if shutil.disk_usage(self.root).free < 12*2**30:
            raise OSError('Insufficient checkpoint space for the next5k block;best weights retained.')
        root.mkdir(exist_ok=True)
        source, parent = previous/'train/latest.pt', root/'parent.pt'
        if not parent.exists():
            os.link(source, parent)
        # Hardlink an immutable completed state;atomic trainer saves cannot alter it.
        if file_sha(parent) != file_sha(source):
            raise ValueError('Frozen continuation parent changed.')
        request = dict(parent_sha256=file_sha(parent), parent_step=start, target_steps=target,
                       policy=POLICY, continuation='Matched terrain progress;retain Adam/EMA/FP32 masters/RNG.')
        if (root/'request.json').exists() and json.loads((root/'request.json').read_text()) != request:
            raise ValueError('Extension request changed.')
        atomic_json(root/'request.json', request)
        common = ['-m', 'distill.native_onepass_residual_train', '--preparation', str(self.args.preparation),
                  '--resume-source', str(parent), '--steps', str(target), '--nf4', '--snapshot-every', '1000']
        try:
            smoke = root/'smoke'
            if not (smoke/'gpu-smoke.json').exists():
                self.execute(common+['--output', str(smoke), '--smoke'], f'smoke:{start}:{target}', target)
            if json.loads((smoke/'gpu-smoke.json').read_text())['status'] != 'complete':
                raise ValueError('Extension smoke failed.')
            smoke_pack = root/'smoke-packed'
            if not (smoke_pack/'audit.json').exists():
                if smoke_pack.exists():
                    raise ValueError('Incomplete extension smoke pack;inspect before retry.')
                self.execute(['-m', 'distill.native_onepass_residual_export', '--checkpoint',
                              str(smoke/f'step{start+4:06d}.pt'), '--output', str(smoke_pack)],
                              f'packed-smoke:{target}', target)
            progress = root/'train/status.json'
            current = json.loads(progress.read_text()) if progress.exists() else dict(step=start)
            if current.get('status') == 'failed':
                raise ValueError('Failed trainer;no automatic retry.')
            step = current['step']
            while step < target and not self.stopped:
                endpoint = min(target, (step//1000+1)*1000)
                self.execute(common+['--output', str(root/'train'), '--smoke-root', str(smoke),
                                     '--until', str(endpoint)], f'train:{step}:{endpoint}', target)
                saved = json.loads(progress.read_text())
                if saved['step'] != endpoint or saved['status'] not in ('segment_complete', 'complete'):
                    raise ValueError('Extension segment incomplete.')
                step = endpoint
            if self.stopped:
                raise InterruptedError('Extension interrupted.')
            self.plates(root, target)
            atomic_json(finished, dict(status='complete', step=target, accepted=False))
            return root
        except Exception as exc:
            atomic_json(finished, dict(status='interrupted' if self.stopped else 'failed',
                        target=target, error=str(exc)))
            raise

    def run(self):
        while not self.stopped:
            initial = state(DATA/'jobs'/self.args.initial_job)
            if not initial['live']:
                if initial.get('status') != 'complete' or initial.get('returncode') != 0:
                    raise ValueError('Initial20k run failed;inspect before continuing.')
                break
            self.status('waiting_initial_run', initial_job=self.args.initial_job, initial_target=20000,
                        extension_steps=5000, hard_step_limit=None)
            time.sleep(15)
        if self.stopped:
            raise InterruptedError('Adaptive continuation interrupted.')
        should_continue = True
        for step in (10000, 15000, 20000):
            should_continue = self.observe(step)
        previous, step = self.root, 20000
        while should_continue and not self.stopped:
            target = step+POLICY['extension_steps']
            previous = self.stage(previous, step, target)
            step = target
            should_continue = self.observe(step)
        self.status('interrupted' if self.stopped else 'stopped', step=step,
                    reason='terrain_regression' if not self.history[-1]['comparison']['eligible'] else 'plateau',
                    best_step=self.best['step'], best_selection=self.best['selection'], accepted=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--gpu-lock', type=Path, required=True)
    parser.add_argument('--initial-job', default='base-onepass-residual-nf4-20k-4090')
    args = parser.parse_args()
    controller = Controller(args)
    policy = controller.output/'policy.json'
    if not policy.exists():
        raise ValueError('Explicit adaptive training authorization required.')
    if json.loads(policy.read_text())['policy'] != POLICY:
        raise ValueError('Adaptive policy changed.')
    try:
        controller.run()
    except Exception as exc:
        controller.status('interrupted' if controller.stopped else 'storage_limited' if isinstance(exc, OSError) else 'failed',
                          error=str(exc))
        raise


if __name__ == '__main__':
    main()

"""Durable 20k NF4 continuation with audited packs and six-scene milestones."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

from distill.common import DATA, REPO, atomic_json, external_path
from distill.native_weight_quant import file_sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--gpu-lock', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=20000)
    args = parser.parse_args()
    root, parent = external_path(args.root), external_path(args.parent)
    root.mkdir(parents=True, exist_ok=True)
    if args.steps != 20000:
        raise ValueError('This run implements the authorized20k total endpoint.')
    request = root/'request.json'
    if not request.exists() or json.loads(request.read_text())['parent_sha256'] != file_sha(parent):
        raise ValueError('The frozen parent must match the explicit training request.')
    stopped, child = False, None

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.terminate()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)

    def execute(command, phase):
        nonlocal child
        with args.gpu_lock.open('a') as lock:
            while not stopped:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    atomic_json(root/'run-status.json', dict(status='waiting_gpu', phase=phase, target=20000))
                    time.sleep(2)
            if stopped:
                raise InterruptedError('20k continuation interrupted.')
            child = subprocess.Popen([sys.executable, '-u', *command], cwd=REPO)
            atomic_json(root/'run-status.json', dict(status='running', phase=phase, target=20000, child_pid=child.pid))
            code = child.wait()
            child = None
            if code or stopped:
                raise RuntimeError(f'{phase} exited{code};inspect before retry.')

    common = ['-m', 'distill.native_onepass_residual_train', '--preparation', str(args.preparation),
              '--resume-source', str(parent), '--steps', '20000', '--nf4', '--snapshot-every', '1000']

    def plates(step):
        reports = []
        for selection in ('ema', 'raw'):
            checkpoint = root/'train'/('raw' if selection == 'raw' else '')/f'step{step:06d}.pt'
            pack = root/'packed'/f'{selection}-{step:06d}'
            audit_path = pack/'audit.json'
            if not audit_path.exists():
                if pack.exists():
                    raise ValueError('Incomplete pack;preserve it and inspect the failed audit.')
                execute(['-m', 'distill.native_onepass_residual_export', '--checkpoint', str(checkpoint),
                         '--output', str(pack)], f'export:{step}:{selection}')
            audit = json.loads(audit_path.read_text())
            if audit['status'] != 'complete' or audit['source_sha256'] != file_sha(checkpoint):
                raise ValueError('Packed checkpoint audit mismatch.')
            for coarse in ('coarse4', 'coarse2'):
                label = 'onepass-residual-nf4-'+selection
                report = DATA/'_base'/f'base-{label}-{step:06d}-{coarse}.json'
                if not report.exists():
                    execute(['-m', 'distill.native_progressive_evaluate', '--checkpoint', str(checkpoint),
                             '--packed', str(pack), '--coarse', coarse, '--label', label],
                            f'quality:{step}:{selection}:{coarse}')
                saved = json.loads(report.read_text())
                if (not report.with_suffix('.png').exists() or saved['representation'] != 'packed-nf4' or
                        saved['checkpoint_sha256'] != audit['source_sha256']):
                    raise ValueError('Milestone plate does not match the audited packed checkpoint.')
                from distill.native_onepass_gallery import publish
                publish(report, report.with_suffix('')/'gallery')
                reports.append(str(report))
        atomic_json(root/f'milestone{step:06d}.json', dict(status='complete', step=step,
                    reports=reports, accepted=False))

    try:
        smoke = root/'smoke'
        proof = smoke/'gpu-smoke.json'
        if not proof.exists():
            execute(common+['--output', str(smoke), '--smoke'], 'resume-nf4-smoke')
        if json.loads(proof.read_text())['status'] != 'complete':
            raise ValueError('NF4 continuation smoke failed.')
        # Prove the actual packed path before committing to the long run.
        smoke_step = json.loads((smoke/'status.json').read_text())['step']
        smoke_pack = root/'smoke-packed'
        if not (smoke_pack/'audit.json').exists():
            if smoke_pack.exists():
                raise ValueError('Incomplete smoke pack;inspect before retry.')
            execute(['-m', 'distill.native_onepass_residual_export', '--checkpoint',
                     str(smoke/f'step{smoke_step:06d}.pt'), '--output', str(smoke_pack)], 'nf4-packed-smoke')
        progress = root/'train/status.json'
        current = json.loads(progress.read_text()) if progress.exists() else dict(step=smoke_step-4)
        if current.get('status') == 'failed':
            raise ValueError('Previous trainer failed;no automatic retry.')
        step = current['step']
        milestones = (2000, 5000, 10000, 15000, 20000)
        for mark in milestones:
            if mark <= step:
                plates(mark)
        while step < 20000 and not stopped:
            target = min(20000, (step//1000+1)*1000)
            execute(common+['--output', str(root/'train'), '--smoke-root', str(smoke),
                            '--until', str(target)], f'train:{step}:{target}')
            saved = json.loads(progress.read_text())
            if saved['step'] != target or saved['status'] not in ('segment_complete', 'complete'):
                raise ValueError('Trainer did not finish its requested segment.')
            step = target
            if step in milestones:
                plates(step)
        atomic_json(root/'run-status.json', dict(status='interrupted' if stopped else 'complete',
                    step=step, target=20000, accepted=False,
                    note='20k training and real compact-weight plates;quality acceptance remains visual.'))
    except Exception as exc:
        atomic_json(root/'run-status.json', dict(status='interrupted' if stopped else 'failed',
                    target=20000, error=str(exc)))
        raise


if __name__ == '__main__':
    main()

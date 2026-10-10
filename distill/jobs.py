"""Durable tmux jobs with explicit process handles and terminal exit status.

python -m distill.jobs start teacher-smoke --gpu 0 -- python -m distill.teacher ...
python -m distill.jobs status teacher-smoke
python -m distill.jobs stop teacher-smoke

No automatic restart: inspect a live handle or terminal status first, then resume
the existing dataset/checkpoint explicitly. Polling does not trigger GPU work.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

from distill.common import DATA, REPO, atomic_json


def proc_identity(pid):
    try:
        # The comm field may contain spaces or ')'; start time is field 22.
        data = Path(f'/proc/{int(pid)}/stat').read_text().rsplit(')', 1)[1].split()
        if data[0] == 'Z':
            return None
        return data[19]
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def state(directory):
    path = directory/'status.json'
    result = json.loads(path.read_text()) if path.exists() else dict(status='missing')
    session = f"distill-{directory.name}"
    tmux = subprocess.run(['tmux', 'has-session', '-t', session], capture_output=True).returncode == 0
    worker = (bool(result.get('worker_identity')) and result.get('worker_pid') is not None and
              proc_identity(result['worker_pid']) == result.get('worker_identity'))
    child = (bool(result.get('child_identity')) and result.get('child_pid') is not None and
             proc_identity(result['child_pid']) == result.get('child_identity'))
    return result | dict(tmux_live=tmux, worker_live=worker, child_live=child,
                         live=tmux or worker or child)


def worker(job_path):
    job = json.loads(job_path.read_text())
    directory = job_path.parent
    environment = dict(os.environ) | job['environment']
    status = dict(status='running', command=job['command'], gpu=job['gpu'],
                  started_at=time.time(), worker_pid=os.getpid(), worker_identity=proc_identity(os.getpid()))
    with (directory/'output.log').open('a', buffering=1) as output:
        output.write(f"\nSTART {time.strftime('%Y-%m-%d %H:%M:%S')} {shlex.join(job['command'])}\n")
        process = subprocess.Popen(job['command'], cwd=REPO, env=environment, stdout=output,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        status.update(child_pid=process.pid, child_identity=proc_identity(process.pid))
        atomic_json(directory/'status.json', status)
        def forward(signum, frame):
            if process.poll() is None:
                # Signal only the trainer, not its DataLoader workers. The
                # trainer finishes its step and shuts its workers down itself.
                process.send_signal(signal.SIGTERM)
        for signum in (signal.SIGTERM, signal.SIGINT):
            signal.signal(signum, forward)
        code = process.wait()
        status.update(status='complete' if code == 0 else 'failed', returncode=code, finished_at=time.time())
        atomic_json(directory/'status.json', status)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['start', 'status', 'stop', 'worker'])
    parser.add_argument('name')
    parser.add_argument('--gpu', choices=['0', '1', 'cpu'], default='0')
    parser.add_argument('--compact', action='store_true')
    args, command = parser.parse_known_args()
    if args.action == 'worker':
        raise SystemExit(worker(Path(args.name)))
    if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyz0123456789-_' for c in args.name):
        parser.error('Job name must contain only lowercase letters, digits, - and _.')
    directory = DATA/'jobs'/args.name
    current = state(directory)
    if args.action == 'status':
        if args.compact:
            current = {key: current.get(key) for key in ('status', 'live', 'child_pid', 'returncode', 'started_at', 'finished_at')}
        print(json.dumps(current, indent=2 if not args.compact else None))
    elif args.action == 'stop':
        if current['worker_live']:
            os.kill(current['worker_pid'], signal.SIGTERM)
            print('Requested graceful stop; wait for the saved checkpoint and terminal status.')
        elif current['child_live']:
            os.kill(current['child_pid'], signal.SIGTERM)
            print('Requested graceful stop of the surviving child.')
        else:
            print('No verified live worker/child to stop.')
    else:
        if current['live']:
            raise SystemExit('Existing job is live. Do not start a duplicate.')
        if command and command[0] == '--':
            command = command[1:]
        if not command:
            parser.error('Pass a command after --.')
        directory.mkdir(parents=True, exist_ok=True)
        environment = {key: os.environ[key] for key in ('PATH', 'HF_HOME', 'HF_HUB_CACHE',
                       'TERRAIN_RUNTIME_ROOT', 'TERRAIN_OUTPUT_ROOT', 'DISTILL_ROOT') if key in os.environ}
        environment.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='' if args.gpu == 'cpu' else args.gpu,
                           TERRAIN_CUDA_DEVICE='0', TERRAIN_GPU_MODE='single', TERRAIN_CUDA_DEVICES='0')
        job = dict(command=command, environment=environment, gpu=args.gpu, created_at=time.time())
        if (directory/'job.json').exists():
            atomic_json(directory/f'job-{time.time_ns()}.json', json.loads((directory/'job.json').read_text()))
        atomic_json(directory/'job.json', job)
        atomic_json(directory/'status.json', dict(status='starting', command=command, gpu=args.gpu))
        launcher = shlex.join([sys.executable, '-m', 'distill.jobs', 'worker', str(directory/'job.json')])
        subprocess.run(['tmux', 'new-session', '-d', '-s', f'distill-{args.name}', '-c', str(REPO), launcher], check=True)
        print(json.dumps(dict(job=args.name, session=f'distill-{args.name}', directory=str(directory))))


if __name__ == '__main__':
    main()

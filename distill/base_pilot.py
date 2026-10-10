"""Durable native base pilot, queued behind the short coarse trials on GPU0."""
from __future__ import annotations

import json
import signal
import subprocess
import sys
import time

from distill.common import DATA, REPO, atomic_json
from distill.jobs import state


def main():
    process = None
    stopped = False
    output = DATA/'ckpt/base-native-pilot'

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    # Keep the viewer on GPU1 and avoid concurrent training on GPU0.
    while not stopped:
        coarse = state(DATA/'jobs/coarse-short-trials')
        if not coarse['live']:
            if coarse.get('status') != 'complete' or coarse.get('returncode') != 0:
                raise RuntimeError('Finish the short coarse trials before the native base pilot.')
            break
        time.sleep(10)
    if stopped:
        return
    for action in ('prepare', 'train'):
        path = (DATA/'crops/base-native-pilot' if action == 'prepare' else output)/'status.json'
        current = json.loads(path.read_text()) if path.exists() else {}
        if current.get('status') == 'complete' and (action == 'prepare' or current.get('step') == 1000):
            continue
        print(json.dumps(dict(native_base_phase=action)), flush=True)
        process = subprocess.Popen([sys.executable, '-u', '-m', 'distill.native_base', action], cwd=REPO)
        code = process.wait()
        if stopped:
            return
        if code:
            raise RuntimeError(f'Native base {action} failed with exit {code}.')
        current = json.loads(path.read_text())
        if current.get('status') != 'complete':
            raise RuntimeError(f'Native base {action} did not complete.')
    atomic_json(output/'pilot.json', dict(status='complete', training_steps=1000,
                teacher_derived=True, production_stages=2, accepted=False,
                physical_quality_evaluation_pending=True))


if __name__ == '__main__':
    main()

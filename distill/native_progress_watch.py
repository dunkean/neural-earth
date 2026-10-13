"""Publish evaluated base checkpoints on the second GPU without racing a trainer.

The job polls at five-minute intervals. Atomic hardlinks pin the exact evaluated
EMA while the producer replaces best.pt. Only two intermediate raw checkpoints
are retained; all rendered stages retain their exact checksum and provenance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import uuid

import torch

from distill.common import DATA, REPO, atomic_json, external_path
from distill.jobs import state


def checksum(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--checkpoint-root', type=Path, required=True)
    parser.add_argument('--wait-job')
    parser.add_argument('--target-step', type=int, required=True)
    parser.add_argument('--interval', type=int, default=300)
    parser.add_argument('--output', type=Path, default=DATA/'_base')
    parser.add_argument('--matched-eval', action='store_true', help='Evaluate each published EMA on the independent matched bank.')
    args = parser.parse_args()
    stop = threading.Event()
    def request_stop(signum, frame):
        stop.set()  # Finish any running GPU child before releasing its snapshot.
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, request_stop)
    root, output = external_path(args.checkpoint_root), external_path(args.output)
    if args.interval < 60 or args.target_step <= 0:
        raise ValueError('Positive target and sparse polling required.')
    if args.wait_job:
        while True:
            if stop.is_set():
                return
            upstream = state(DATA/'jobs'/args.wait_job)
            if not upstream['live']:
                if upstream['status'] != 'complete' or upstream.get('returncode') != 0:
                    raise RuntimeError('Inspect failed preceding GPU job before continuing.')
                break
            stop.wait(args.interval)
    snapshots = root/'progress-snapshots'
    snapshots.mkdir(parents=True, exist_ok=True)
    proof = root/'progress-publication.json'
    publications = json.loads(proof.read_text()).get('publications', []) if proof.exists() else []
    observed = None
    while not stop.is_set():
        trainer = state(DATA/'jobs'/args.job)
        terminal = not trainer['live']
        if terminal and (trainer['status'] != 'complete' or trainer.get('returncode') != 0):
            raise RuntimeError('Producer stopped unsuccessfully; inspect before resuming.')
        source = root/f'step{args.target_step}.pt' if terminal else root/'best.pt'
        if source.exists() and (terminal or source.stat().st_mtime_ns != observed):
            snapshot = snapshots/f'.publication-{uuid.uuid4().hex}.pt'
            os.link(source, snapshot)
            observed = snapshot.stat().st_mtime_ns
            try:
                saved = torch.load(snapshot, map_location='cpu', weights_only=True)
                step = saved['step']
                del saved
                sha = checksum(snapshot)
                image_proof = output/f'{step:06d}.json'
                previous = json.loads(image_proof.read_text()) if image_proof.exists() else None
                if previous and previous['checkpoint_sha256'] != sha:
                    raise ValueError('Existing stage belongs to another checkpoint.')
                if previous is None:
                    if step in (20000, 30000):
                        retained = snapshots/f'step{step}.pt'
                        if retained.exists():
                            if checksum(retained) != sha:
                                raise ValueError('Retained stage checkpoint differs.')
                        else:
                            os.link(snapshot, retained)
                        render_source = retained
                    else:
                        render_source = snapshot
                    subprocess.run([sys.executable, '-u', '-m', 'distill.native_progress_images',
                        '--checkpoints', str(render_source), '--output', str(output)], cwd=REPO, check=True)
                    published = json.loads(image_proof.read_text())
                    published.update(snapshot_retained=step in (20000, 30000),
                        producer_checkpoint_root=str(root), source_checkpoint=str(source))
                    if terminal:
                        published.update(checkpoint=str(source), snapshot_retained=True)
                    atomic_json(image_proof, published)
                    publications.append(dict(step=step, sha256=sha, image=str(output/f'{step:06d}.png')))
                    atomic_json(proof, dict(status='running', publications=publications))
                    print(json.dumps(publications[-1]), flush=True)
                if args.matched_eval:
                    evaluation = DATA/f'eval/native160x3-step{step}-matched-validation.json'
                    if evaluation.exists():
                        prior = json.loads(evaluation.read_text())
                        if prior['status'] != 'complete' or prior['results'][0]['sha256'] != sha:
                            raise ValueError('Existing matched evaluation belongs to another checkpoint.')
                    else:
                        subprocess.run([sys.executable, '-u', '-m', 'distill.native_multilod_eval',
                            '--checkpoints', str(snapshot), '--output', str(evaluation), '--rare-shape'],
                            cwd=REPO, check=True)
                if terminal:
                    if step != args.target_step:
                        raise ValueError('Producer did not complete its requested training palier.')
                    atomic_json(proof, dict(status='complete', publications=publications, accepted=False))
                    return
            finally:
                snapshot.unlink(missing_ok=True)
        elif terminal:
            raise FileNotFoundError('Completed producer is missing its final immutable checkpoint.')
        stop.wait(args.interval)
    atomic_json(proof, dict(status='interrupted', publications=publications, accepted=False))


if __name__ == '__main__':
    main()

"""Two finite matched-budget shoreline trials, resumable through distill.jobs.

Run on GPU 0, leaving the live viewer on GPU 1. Each trial starts from the
same immutable parent EMA with reset optimizer. Only shore_weight differs.
Signals reach the active trainer; interrupted jobs require explicit resume.
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time

import torch

from distill.common import DATA, REPO, atomic_json
from distill.decoded_loss import initialize
from distill.shore_probe import fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=2000)
    args = parser.parse_args()
    if args.steps != 2000:
        parser.error('This fixed comparison uses exactly 2000 steps; use a separate trial for another budget.')
    root = DATA/'shore-trial'
    root.mkdir(parents=True, exist_ok=True)
    source = DATA/'ckpt/candidates/base-step100000-f25dbf12474a.pt'
    decoder = DATA/'ckpt/candidates/decoder-step200000-3a6303b22b45.pt'
    audit = DATA/'ckpt/base-decoded/paired-audit.json'
    contract = dict(steps=args.steps, source_sha256=fingerprint(source), decoder_sha256=fingerprint(decoder),
                    audit_sha256=fingerprint(audit),
                    code_sha256={name: fingerprint(REPO/'distill'/name) for name in
                                 ('shore_trial.py', 'shore_probe.py', 'train.py', 'decoded_loss.py',
                                  'student.py', 'features.py')})
    if (root/'contract.json').exists() and json.loads((root/'contract.json').read_text()) != contract:
        raise ValueError('Trial inputs/code changed; do not silently resume this comparison.')
    atomic_json(root/'contract.json', contract)
    child = None
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if child is not None and child.poll() is None:
            child.send_signal(signal.SIGTERM)
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, stop)
    def run(command):
        nonlocal child
        if stopped:
            raise RuntimeError('Interrupted; resume this same trial explicitly.')
        child = subprocess.Popen(command, cwd=REPO)
        code = child.wait()
        child = None
        if code or stopped:
            raise RuntimeError(f'Child stopped ({code}); inspect state before resuming.')
    def probe(name, checkpoint):
        output = root/f'{name}-validation.json'
        if output.exists():
            report = json.loads(output.read_text())
            if report['checkpoint']['sha256'] != fingerprint(checkpoint):
                raise ValueError('Existing validation refers to different weights.')
            return
        run([sys.executable, '-m', 'distill.shore_probe', '--checkpoint', str(checkpoint),
             '--decoder', str(decoder), '--audit', str(audit), '--output', str(output)])
    try:
        probe('parent', source)
        for name, weight in (('control', 0.), ('shore', .1)):
            output = root/name
            latest = output/'latest.pt'
            if not latest.exists():
                initialize(source, decoder, audit, output, DATA/'eval/rare-training-coverage.json')
            saved = torch.load(latest, map_location='cpu', weights_only=True)
            if saved['step'] > args.steps:
                raise ValueError('Checkpoint exceeds the matched budget.')
            if saved['step'] < args.steps:
                atomic_json(root/'status.json', dict(status='running', phase=name, updated_at=time.time()))
                run([sys.executable, '-m', 'distill.train', '--stage', 'base', '--dataset', str(DATA/'crops/main'),
                     '--output', str(output), '--resume', str(latest), '--width', '128', '--steps', str(args.steps),
                     '--batch', '1', '--lr', '.0001', '--workers', '2', '--train-size', '64',
                     '--eval-every', '2000', '--val-count', '41', '--save-seconds', '120', '--seed', '8675309',
                     '--height-weight', '8', '--height-mae-weight', '.02', '--spectral-band-weight', '.05',
                     '--shore-weight', str(weight), '--allow-data-growth'])
            terminal = json.loads((output/'status.json').read_text())
            if terminal.get('status') != 'complete' or terminal.get('step') != args.steps:
                raise RuntimeError('A trial did not complete its matched budget.')
            probe(name, latest)
        atomic_json(root/'status.json', dict(status='complete', accepted=False, updated_at=time.time(),
            note='Matched trials and common proxy validation complete; physical inspection still required.'))
    except Exception as exc:
        atomic_json(root/'status.json', dict(status='interrupted' if stopped else 'failed',
                                           error=str(exc), updated_at=time.time()))
        raise


if __name__ == '__main__':
    main()

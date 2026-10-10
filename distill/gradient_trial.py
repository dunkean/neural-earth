"""Finite relative-height-gradient trial against the completed matched control.

GPU0 only; GPU1's live viewer and all selected runtime weights stay unchanged.
SIGTERM saves the active trainer, then stops the workflow. Resume explicitly
with the same command after verifying the old job handle is terminal.
"""
from __future__ import annotations

import json
import signal
import subprocess
import sys
import time

import torch

from distill.common import DATA, REPO, atomic_json
from distill.shore_probe import fingerprint


def main():
    root = DATA/'gradient-trial'
    output = root/'student'
    source = DATA/'ckpt/candidates/base-step100000-f25dbf12474a.pt'
    decoder = DATA/'ckpt/candidates/decoder-step200000-3a6303b22b45.pt'
    audit = DATA/'ckpt/base-decoded/paired-audit.json'
    control = DATA/'shore-trial/control/latest.pt'
    saved_control = torch.load(control, map_location='cpu', weights_only=True)
    saved_student = torch.load(output/'latest.pt', map_location='cpu', weights_only=True)
    expected = dict(steps=2000, batch=1, lr=.0001, train_size=64, seed=8675309,
                    spectral_weight=.02, spectral_band_weight=.05, gradient_weight=.05,
                    height_weight=8., height_mae_weight=.02, decoder_loss_weight=1.)
    if any(saved_control['arguments'][k] != v for k, v in expected.items()):
        raise ValueError('The completed control does not match this trial.')
    if (saved_control['step'] != 2000 or saved_control['arguments'].get('shore_weight', 0.) or
            saved_control['arguments'].get('decoded_gradient_weight', 0.) or
            saved_control['warm_start']['source_sha256'] != fingerprint(source) or
            saved_student['warm_start']['source_sha256'] != fingerprint(source) or
            saved_control['config'] != saved_student['config'] or
            saved_control['sampling_policy'] != saved_student['sampling_policy']):
        raise ValueError('Trial/control initial source, architecture, sampling or budget differs.')
    contract = dict(source_sha256=fingerprint(source), control_sha256=fingerprint(control),
                    decoder_sha256=fingerprint(decoder), audit_sha256=fingerprint(audit),
                    common_arguments=expected, decoded_gradient_weight=.01,
                    code_sha256={name: fingerprint(REPO/'distill'/name) for name in
                                 ('gradient_trial.py', 'shore_probe.py', 'train.py', 'decoded_loss.py',
                                  'student.py', 'features.py')},
                    original_control_code=json.loads((DATA/'shore-trial/contract.json').read_text())['code_sha256'])
    if (root/'contract.json').exists() and json.loads((root/'contract.json').read_text()) != contract:
        raise ValueError('Inputs or code changed; do not silently resume this trial.')
    atomic_json(root/'contract.json', contract)
    child, stopped = None, False
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
            raise RuntimeError('Workflow interrupted; explicit resume required.')
        child = subprocess.Popen(command, cwd=REPO)
        code = child.wait()
        child = None
        if code or stopped:
            raise RuntimeError(f'Child stopped ({code}); inspect state before resuming.')
    def probe(name, checkpoint):
        report = root/f'{name}-validation.json'
        if report.exists():
            if json.loads(report.read_text())['checkpoint']['sha256'] != fingerprint(checkpoint):
                raise ValueError('Existing validation has different weights.')
            return
        run([sys.executable, '-m', 'distill.shore_probe', '--checkpoint', str(checkpoint),
             '--decoder', str(decoder), '--audit', str(audit), '--output', str(report)])
    try:
        probe('parent', source)
        probe('control', control)
        if saved_student['step'] < 2000:
            command = [sys.executable, '-m', 'distill.train', '--stage', 'base',
                       '--dataset', str(DATA/'crops/main'), '--output', str(output),
                       '--resume', str(output/'latest.pt'), '--width', '128', '--workers', '2',
                       '--eval-every', '2000', '--val-count', '41', '--save-seconds', '120',
                       '--decoded-gradient-weight', '.01', '--allow-data-growth']
            for key, value in expected.items():
                command.extend(['--'+key.replace('_', '-'), str(value)])
            atomic_json(root/'status.json', dict(status='running', phase='training', updated_at=time.time()))
            run(command)
        terminal = json.loads((output/'status.json').read_text())
        if terminal.get('status') != 'complete' or terminal.get('step') != 2000:
            raise RuntimeError('Trial did not complete exactly 2000 steps.')
        probe('gradient', output/'latest.pt')
        atomic_json(root/'status.json', dict(status='complete', accepted=False, updated_at=time.time(),
            note='Common validation completed; physical rare cases and cold tiles required before promotion.'))
    except Exception as exc:
        atomic_json(root/'status.json', dict(status='interrupted' if stopped else 'failed', error=str(exc), updated_at=time.time()))
        raise


if __name__ == '__main__':
    main()

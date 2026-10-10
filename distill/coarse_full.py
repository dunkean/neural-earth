"""Full 1/2-pass coarse runs, resuming pilots without replacing their artifacts.

Run one process per GPU through distill.jobs. The budget uses the complete
training bank and validation split. Export checks prove execution, not artistic
quality; the user reviews these coarse variants in the app.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys

from distill.common import DATA, REPO, atomic_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluations', type=int, choices=(1, 2), required=True)
    parser.add_argument('--steps', type=int, default=200000)
    args = parser.parse_args()
    root = DATA/'ckpt/coarse-full'/f'coarse{args.evaluations}'
    source = DATA/'coarse-short-trials'/f'coarse{args.evaluations}'/'latest.pt'
    root.mkdir(parents=True, exist_ok=True)
    contract = dict(evaluations=args.evaluations, target_steps=args.steps,
                    initial_checkpoint=str(source), initial_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    dataset=str(DATA/'crops/main'), batch=4, lr=2e-5, validation_count=384,
                    keep_steps=[25000, 50000, 100000, 150000, 200000],
                    detailed_physical_audit=False)
    if (root/'contract.json').exists() and json.loads((root/'contract.json').read_text()) != contract:
        raise ValueError('Full coarse run contract changed; inspect the existing run.')
    atomic_json(root/'contract.json', contract)
    process = None
    stopped = False

    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        if process is not None and process.poll() is None:
            process.send_signal(signal.SIGTERM)

    for signum in (signal.SIGINT, signal.SIGTERM):
        signal.signal(signum, stop)
    status_path = root/'status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    if status.get('status') != 'complete' or status.get('step') != args.steps:
        latest = root/'latest.pt'
        command = [sys.executable, '-u', '-m', 'distill.train', '--stage', 'coarse',
            '--coarse-solver-steps', str(args.evaluations), '--width', '128', '--output', str(root),
            '--resume', str(latest if latest.exists() else source), '--steps', str(args.steps),
            '--batch', '4', '--lr', '2e-5', '--workers', '2', '--eval-every', '5000',
            '--val-count', '384', '--spectral-band-weight', '.05', '--height-mae-weight', '.05',
            '--save-seconds', '120', '--keep-steps', *[str(s) for s in contract['keep_steps'] if s <= args.steps]]
        process = subprocess.Popen(command, cwd=REPO)
        code = process.wait()
        if stopped:
            return
        if code:
            raise RuntimeError(f'Full coarse training failed with exit{code}.')
        status = json.loads(status_path.read_text())
        if status.get('status') != 'complete' or status.get('step') != args.steps:
            raise RuntimeError('Full coarse training did not finish its budget.')
    import torch
    from distill.dataset import Crops
    from distill.export import export_stage
    from distill.inference import coarse_graph_forward
    from distill.student import load_student
    torch.set_num_threads(4)
    selected = root/'best.pt'
    model, saved = load_student(selected, 'cuda')
    if saved['step'] <= 5000:
        raise ValueError('A two-crop pilot checkpoint cannot select the full-validation model.')
    manifest = export_stage(selected, root/'export')
    exported, _ = load_student(root/'export/coarse.pt', 'cuda')
    exported.solver.prepare_capture()
    validation = Crops(DATA/'crops/main', 'coarse', 'val')
    with torch.inference_mode():
        for index in (0, len(validation)//2, len(validation)-1):
            sample = validation[index][0][None].cuda()
            expected = model(sample)
            actual = coarse_graph_forward(exported, sample)
            if not torch.isfinite(actual).all():
                raise FloatingPointError('Non-finite exported coarse.')
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    atomic_json(root/'delivery.json', dict(status='complete', training_steps=args.steps,
        selected_step=saved['step'], validation_count=384, finite=True, export_max_abs=0.,
        smoke_indices=[0, len(validation)//2, len(validation)-1], export=manifest,
        preserved_pilot=str(source.parent), artistic_quality='user review', physical_acceptance=False))
    print(json.dumps(dict(full_coarse=args.evaluations, status='complete', selected_step=saved['step'])), flush=True)


if __name__ == '__main__':
    main()

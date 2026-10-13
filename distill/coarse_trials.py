"""Short 2/1-evaluation coarse trials for user review, without a physical audit.

Run through distill.jobs on GPU0. Each trial starts from the preserved coarse4
EMA, keeps its full production windows and is exported independently. An
interrupted trial resumes its optimizer; completed trials are not retrained.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import signal
import subprocess
import sys

import torch

from distill.common import DATA, REPO, atomic_json
from distill.export import export_stage
from distill.student import load_student
from distill.dataset import Crops
from distill.inference import coarse_graph_forward


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=str, default=str(DATA/'live/weights/coarse4/coarse.pt'))
    parser.add_argument('--steps', type=int, default=5000)
    args = parser.parse_args()
    from pathlib import Path
    source = Path(args.source).resolve()
    root = DATA/'coarse-short-trials'
    contract = dict(source=str(source), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    training_steps=args.steps, evaluations=[2, 1], seed=8675309,
                    physical_evaluation=False)
    if (root/'contract.json').exists() and json.loads((root/'contract.json').read_text()) != contract:
        raise ValueError('The coarse trial contract changed.')
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
    torch.set_num_threads(4)
    for evaluations in (2, 1):
        output = root/f'coarse{evaluations}'
        status_path = output/'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if status.get('status') != 'complete' or status.get('step') != args.steps:
            command = [sys.executable, '-u', '-m', 'distill.train', '--stage', 'coarse',
                '--width', '128', '--coarse-solver-steps', str(evaluations), '--output', str(output),
                '--steps', str(args.steps), '--batch', '4', '--lr', '2e-5', '--workers', '2',
                '--eval-every', str(args.steps), '--val-count', '2', '--spectral-band-weight', '.05',
                '--height-mae-weight', '.05', '--save-seconds', '120']
            latest = output/'latest.pt'
            command += ['--resume', str(latest)] if latest.exists() else ['--initialize-from', str(source)]
            process = subprocess.Popen(command, cwd=REPO)
            code = process.wait()
            if stopped:
                atomic_json(root/'status.json', dict(status='interrupted', evaluations=evaluations))
                return
            if code:
                raise RuntimeError(f'Coarse{evaluations} training failed with exit {code}.')
            status = json.loads(status_path.read_text())
            if status.get('status') != 'complete' or status.get('step') != args.steps:
                raise RuntimeError('Coarse trial stopped before completing its short budget.')
        export_dir = DATA/f'live/weights/coarse{evaluations}'
        manifest = export_stage(output/'best.pt', export_dir)
        # One complete production window: finite output, EMA/export agreement and
        # the same whole-solver CUDA Graph path the app uses. No site/LOD audit.
        sample = Crops(DATA/'crops/main', 'coarse', 'val')[0][0][None].cuda()
        model, _ = load_student(export_dir/'coarse.pt', 'cuda')
        parent, _ = load_student(output/'best.pt', 'cuda')
        model.solver.prepare_capture()
        with torch.inference_mode():
            expected = parent(sample)
            actual = coarse_graph_forward(model, sample)
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        atomic_json(export_dir/'smoke.json', dict(evaluations=evaluations, finite=True,
                    export_max_abs=0., shape=list(actual.shape), export=manifest['export']))
        del parent, model, expected, actual, sample
        torch.cuda.empty_cache()
        atomic_json(root/'status.json', dict(status='running', exported_evaluations=evaluations))
        print(json.dumps(dict(exported_coarse=evaluations, training_steps=args.steps)), flush=True)
    atomic_json(root/'status.json', dict(status='complete', exported_evaluations=[2, 1],
                default_unchanged=4, physical_evaluation=False))


if __name__ == '__main__':
    main()

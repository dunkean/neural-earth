"""Swap only the FP32 output gain between B75 and B80 for a causal level check."""
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch

from distill.common import DATA, REPO, atomic_json, external_path
from distill.native_weight_quant import file_sha
from distill.native_weight_quant_evaluate import evaluate


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve the original cross-gain probe.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = DATA/'ckpt/native-nf4-level-70k-80k/b'
    paths = {step: root/f'step{step:06d}.pt' for step in (75000, 80000)}
    source = {step: torch.load(path, map_location='cpu', weights_only=True, mmap=True)
              for step, path in paths.items()}
    gain = {step: source[step]['ema']['net.out_gain'].detach().clone()
            for step in source}
    reports = {}
    with tempfile.TemporaryDirectory(prefix='nf4-gain-cross-', dir=DATA/'eval') as folder:
        for step, other in ((75000, 80000), (80000, 75000)):
            saved = source[step].copy()
            saved['ema'] = source[step]['ema'].copy()
            saved['ema']['net.out_gain'] = gain[other]
            saved['gain_cross_probe'] = dict(source_step=step, output_gain_step=other,
                                             no_fitting=True)
            path = Path(folder)/f'step{step:06d}-gain{other:06d}.pt'
            torch.save(saved, path)
            label = f'nf4levelb{step//1000}gain{other//1000}'
            report = evaluate(path, DATA/'_base', label, 'coarse4')
            reports[label] = dict(checkpoint_sha256=file_sha(path), rows=report['rows'])
    summaries = {name: {str(lod): dict(
        mean_mae_m=float(np.mean([r['mae_m'] for r in report['rows'] if r['lod'] == lod])),
        mean_abs_bias_m=float(np.mean([abs(r['height_bias_m']) for r in report['rows'] if r['lod'] == lod])))
        for lod in (3, 0)} for name, report in reports.items()}
    atomic_json(output, dict(status='complete', reports=reports, summaries=summaries,
        gains={str(step): float(value) for step, value in gain.items()},
        sources={str(step): file_sha(path) for step, path in paths.items()},
        source_code_sha256=file_sha(REPO/'distill/native_level_gain_cross_probe.py'),
        gpu=torch.cuda.get_device_name(), updates=0,
        scope='Only one FP32 output scalar exchanged. Temporary derived checkpoints removed; original snapshots untouched.'))
    print(json.dumps(dict(status='complete', summaries=summaries)), flush=True)


if __name__ == '__main__':
    main()

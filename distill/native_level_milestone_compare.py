"""Deterministic comparison of saved NF4 level checkpoints at 75k and 80k."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import DATA, REPO, atomic_json, external_path
from distill.native_weight_quant import file_sha
from distill.native_weight_quant_evaluate import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Comparison already exists; preserve the original result.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = DATA/'ckpt/native-nf4-level-70k-80k'
    cases = [
        ('start70', DATA/'ckpt/native-nf4-level-init/step070000-identity.pt', 'nf4levelstartmilestone'),
        ('a75', root/'a/step075000.pt', 'nf4levela75milestone'),
        ('b75', root/'b/step075000.pt', 'nf4levelb75milestone'),
        ('b75_repeat', root/'b/step075000.pt', 'nf4levelb75repeatmilestone'),
        ('a80', root/'a/step080000.pt', 'nf4levela80milestone'),
        ('b80', root/'b/step080000.pt', 'nf4levelb80milestone'),
    ]
    reports = {}
    for name, checkpoint, label in cases:
        step = 70000 if name == 'start70' else int(name[1:3])*1000
        stem = DATA/'_base'/f'native-weight-{label}-coarse4-{step:06d}'
        if stem.with_suffix('.json').exists() or stem.with_suffix('.png').exists():
            raise ValueError(f'Comparison output already exists: {stem}')
        reports[name] = evaluate(checkpoint, DATA/'_base', label, 'coarse4')
        rows = reports[name]['rows']
        if len(rows) != 14 or any(not np.isfinite(row['mae_m']) for row in rows):
            raise ValueError(f'Incomplete metrics: {name}')
    if reports['b75']['rows'] != reports['b75_repeat']['rows']:
        raise ValueError('Repeated B75 metrics differ.')
    summaries = {}
    for name, report in reports.items():
        summaries[name] = {str(lod): dict(
            mean_mae_m=float(np.mean([r['mae_m'] for r in report['rows'] if r['lod'] == lod])),
            mean_abs_bias_m=float(np.mean([abs(r['height_bias_m']) for r in report['rows'] if r['lod'] == lod])),
            mean_high_frequency_rms_ratio=float(np.mean([r['high_frequency_rms_ratio'] for r in report['rows'] if r['lod'] == lod])),
        ) for lod in (3, 0)}
    result = dict(status='complete', reports=reports, summaries=summaries,
                  b75_repeat_metrics_exact=True, gpu=torch.cuda.get_device_name(), updates=0,
                  evaluation_settings=dict(cudnn_benchmark=False, cudnn_deterministic=True, tf32=False),
                  source_sha256=file_sha(REPO/'distill/native_level_milestone_compare.py'))
    atomic_json(output, result)
    print(json.dumps(dict(status='complete', summaries=summaries,
                          b75_repeat_metrics_exact=True)), flush=True)


if __name__ == '__main__':
    main()

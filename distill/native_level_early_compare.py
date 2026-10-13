"""Deterministic matched72k A/B progress, with an independent B repeat."""
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
        raise ValueError('Preserve prior comparisons; no implicit rerun.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root = DATA/'ckpt/native-nf4-level-70k-80k'
    cases = [('start', DATA/'ckpt/native-nf4-level-init/step070000-identity.pt', 'nf4levelstartdet'),
             ('a', root/'a/early/step072000.pt', 'nf4levelaearlydet'),
             ('b', root/'b/early/step072000.pt', 'nf4levelbearlydet'),
             ('b_repeat', root/'b/early/step072000.pt', 'nf4levelbearlyrepeatdet')]
    reports = {}
    for name, checkpoint, label in cases:
        stem = DATA/'_base'/f'native-weight-{label}-coarse4-{70000 if name == "start" else 72000:06d}'
        if stem.with_suffix('.json').exists() or stem.with_suffix('.png').exists():
            raise ValueError('Use fresh labels after investigating a previous terminal attempt.')
        reports[name] = evaluate(checkpoint, DATA/'_base', label, 'coarse4')
        reports[name].update(evaluation_settings=dict(cudnn_benchmark=False,
            cudnn_deterministic=True, tf32=False), wrapper_sha256=file_sha(REPO/'distill/native_level_early_compare.py'))
        atomic_json(stem.with_suffix('.json'), reports[name])
    if reports['b']['rows'] != reports['b_repeat']['rows']:
        raise ValueError('Repeated deterministic B metrics differ; inspect before interpreting A/B.')
    for name in ('start', 'a', 'b'):
        rows = reports[name]['rows']
        if len(rows) != 14 or any(not np.isfinite(r['mae_m']) for r in rows):
            raise ValueError('Complete finite14-row matched reports required.')
    summaries = {}
    for name in ('start', 'a', 'b'):
        summaries[name] = {str(lod): dict(
            mean_mae_m=float(np.mean([r['mae_m'] for r in reports[name]['rows'] if r['lod'] == lod])),
            mean_abs_bias_m=float(np.mean([abs(r['height_bias_m']) for r in reports[name]['rows'] if r['lod'] == lod])))
            for lod in (3, 0)}
    atomic_json(output, dict(status='complete', reports=reports, summaries=summaries,
        b_repeat_metrics_exact=True, gpu=torch.cuda.get_device_name(), updates=0,
        scope='Descriptive early2000-update pair, same GPU, geometry and deterministic flags; not final selection.',
        code_sha256={name:file_sha(REPO/'distill'/name) for name in
                     ('native_level_early_compare.py', 'native_weight_quant_evaluate.py')}))
    print(json.dumps(dict(status='complete', summaries=summaries, b_repeat_metrics_exact=True)), flush=True)


if __name__ == '__main__':
    main()

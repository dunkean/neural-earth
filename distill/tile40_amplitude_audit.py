"""Read-only decoded amplitude baseline for the40km tile research segment."""
import argparse
import json
import math
from pathlib import Path
import statistics

from distill.common import atomic_json, external_path
from distill.native_weight_quant import file_sha


def audit(report_path):
    report = json.loads(report_path.read_text())
    if report['status'] != 'complete' or report['representation'] != 'packed-nf4':
        raise ValueError('A completed actual packed NF4 evaluation is required.')
    selected = [row for row in report['rows'] if row['kind'] == 'wet-high-mountain' and row['lod'] == 0]
    if len(selected) < 3:
        raise ValueError('At least three independent mountain scenes required.')
    ratios, correlations, scenes = [], [], []
    for row in selected:
        shapes = [value for value in row['shape']['topographic_position'] if value['available']]
        values = [value['tpi_rms_ratio'] for value in shapes]
        if not values or any(value is None or not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError('Finite positive decoded relief amplitudes required.')
        ratios.extend(values)
        correlations.extend(value['tpi_correlation'] for value in shapes
                            if value['tpi_correlation'] is not None)
        scenes.append(dict(seed=row['seed'], bias_m=row['height_bias_m'], mae_m=row['mae_m'],
                           rms_ratios=values, hf_rms_ratio=row['high_frequency_rms_ratio']))
    gap = statistics.mean(abs(math.log(value)) for value in ratios)
    return dict(status='complete', report=str(report_path), report_sha256=file_sha(report_path),
        checkpoint_sha256=report['checkpoint_sha256'], packed_weights_sha256=report['packed_weights_sha256'],
        step=report['step'], gpu=report['gpu'], weight_bytes=report['weight_bytes'], scenes=scenes,
        metrics=dict(amplitude_gap=gap, mean_amplitude_ratio=statistics.mean(ratios),
                     minimum_amplitude_ratio=min(ratios), maximum_amplitude_ratio=max(ratios),
                     morphology_correlation=statistics.mean(correlations),
                     mean_height_mae_m=statistics.mean(row['mae_m'] for row in selected)),
        scope='Diagnostic on three wet mountains;not a40km tile,all-climate,or browser acceptance benchmark.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.report)
    output = external_path(args.output)
    atomic_json(output, result)
    for name, value in result['metrics'].items():
        print(f'METRIC {name}={value:.9g}')
    print(f'ARTIFACT decoded_amplitude={output}')


if __name__ == '__main__':
    main()

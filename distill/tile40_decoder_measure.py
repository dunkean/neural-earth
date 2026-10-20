"""Read audited decoder-component validation;never claim full-tile acceptance."""
import json
import math

from distill.common import DATA
from distill.native_weight_quant import file_sha


def main():
    root = DATA/'tile40-20261014'
    run = root/'decoder16-pilot-v1'
    contract = json.loads((run/'contract.json').read_text())
    references = root/'reference-guided-v2'
    if file_sha(references/'manifest.json') != contract['dataset_manifest_sha256']:
        raise ValueError('Validation reference contract changed.')
    manifest = json.loads((references/'manifest.json').read_text())
    expected_cases = {case['file'] for case in manifest['cases'] if case['split']=='val'}
    report = json.loads((run/'validation-020000.json').read_text())
    if report['step'] != 20000:
        raise ValueError('Wrong measured checkpoint step.')
    selected = report['ema']
    if {row['case']['file'] for row in selected['rows']} != expected_cases or len(selected['rows']) != 30:
        raise ValueError('Validation omitted a seed,climate or relief group.')
    export = root/'decoder16-step20k-browser-v1'
    artifact = json.loads((export/'manifest.json').read_text())
    if artifact['step'] != 20000 or file_sha(run/'step-020000.pt') != artifact['checkpoint_sha256']:
        raise ValueError('Browser export used different checkpoint weights.')
    if file_sha(export/artifact['model']) != artifact['model_sha256'] or artifact['model_bytes'] >= 100000000:
        raise ValueError('Artifact identity or size changed.')
    score = selected['score']
    if not math.isfinite(score) or score < 0:
        raise ValueError('Nonfinite component measurement.')
    print('SCOPE Oracle-latent decoder component;not a sketch-to-tile product or full40km benchmark.')
    print(f'METRIC decoder_component_score={score}')
    for group, values in selected['groups'].items():
        print(f'METRIC {group}_height_mae_m={values["height_mae_m"]}')
        print(f'METRIC {group}_amplitude_gap={values["amplitude_gap"]}')
    print(f'METRIC decoder_model_bytes={artifact["model_bytes"]}')
    print(f'ARTIFACT component_validation={run/"validation-020000.json"}')
    print(f'ARTIFACT decoder_export={export/"manifest.json"}')


if __name__ == '__main__':
    main()

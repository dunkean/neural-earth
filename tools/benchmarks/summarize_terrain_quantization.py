"""CPU-only synthesis of measured quantization experiments and displayed climate."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path

import argparse
import json
from pathlib import Path
import statistics

import numpy as np

from experiment_terrain_quantization import error, trusted_manifest
from terrain_climate import expand_climate
from terrain_quantization import ROLE, require_scope, sha_file, write_json


def read(path):
    result = json.loads(path.read_text(encoding='utf-8'))
    require_scope(role=result['role'])
    if result.get('runtime_enabled') is not False or result.get('state') != 'complete':
        raise ValueError(f'Expected complete offline result: {path}')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    physical = read(root/'physical-validation/physical.json')
    matched = read(root/'matched-bf16-trt-admitted/matched.json')
    fp32 = read(root/'matched-bf16-trt-fp32-admitted/matched.json')
    for receipt in (matched,fp32):
        profile=receipt['reference_profile']
        if not (profile['cached_weights'] and profile['cuda_graphs'] and profile['exact_kernels']
                and profile['attention_backend']=='reference'):
            raise ValueError('Matched reference profile is not admitted')
        device=str(receipt['selected_cuda_device'])
        for field in ('reference_kernel_status','reference_kernel_status_after'):
            status=receipt[field]
            if not status['requested'] or not status['effective_by_device'].get(device,False):
                raise ValueError('Matched custom kernels not effective')
        graph=receipt['reference_graph_stats']
        if graph['fallback_calls'] or graph['memory_fallbacks'] or graph['capture_errors']:
            raise ValueError('Matched graphs fell back')
    for row in matched['physical_pairs']:
        for pair in row['pairs']:
            if not all(value['byte_exact'] for backend in pair['admission'].values()
                       for value in backend['revisit'].values()):
                raise ValueError('Matched physical revisit changed arrays')
    proof = read(root/'holdout-packets-trt/packets.json')
    engine = root/'trt-qdq-attempt2'
    build = json.loads((engine/'build.json').read_text(encoding='utf-8'))
    float_engine = root/'trt-fp32'
    float_build = json.loads((float_engine/'build.json').read_text(encoding='utf-8'))
    source_export=trusted_manifest(root.parent/'quant-export-debug-2/export.json','prepared-not-validated')
    for receipt in (physical,matched,fp32,proof):
        if receipt['export_identity'] != source_export['artifact_identity']:
            raise ValueError('Mixed export identities')
        # The original packet receipt omits capture_identity. Its verified
        # source export supplies that relation; do not retrofit old receipts.
        if receipt.get('capture_identity',source_export['capture_identity']) != source_export['capture_identity']:
            raise ValueError('Mixed capture identities')
    for receipt in (fp32,proof):
        if receipt['holdout_identity'] != matched['holdout_identity']:
            raise ValueError('Mixed holdout identities')
    for directory,expected,consumers,variant in (
        (engine,build,(proof,matched),'qdq'),
        (float_engine,float_build,(fp32,),'fp32')):
        require_scope(role=expected['role'])
        if expected['state'] != 'built-not-executed' or expected['variant'] != variant:
            raise ValueError('Unexpected engine build state/variant')
        if expected['plan_sha'] != sha_file(directory/'base-preview.plan'):
            raise ValueError('Engine plan SHA mismatch')
        if expected['export_identity'] != matched['export_identity']:
            raise ValueError('Engine/source export identity mismatch')
        filename='base-int8-qdq.onnx' if variant == 'qdq' else 'base-fp32-conversion.onnx'
        if expected['onnx_sha'] != source_export['files'][filename]:
            raise ValueError('Engine/source ONNX SHA mismatch')
        for receipt in consumers:
            actual=receipt['engine_build']
            if actual['plan_sha'] != expected['plan_sha'] or actual['export_identity'] != expected['export_identity']:
                raise ValueError('Engine/measurement identity mismatch')
    matched_physical={tuple(x['site'][key] for key in ('seed','profile','tx','ty')):x
                      for x in matched['physical_pairs']}
    for row in physical['variants']['trt']:
        key=tuple(row['site'][field] for field in ('seed','profile','tx','ty'))
        corroboration=matched_physical[key]
        if not all(pair['elevation_error_m']==row['elevation_error_m'] and
                   pair['climate_errors']==row['climate_errors'] for pair in corroboration['pairs']):
            raise ValueError('Historical physical TRT errors disagree with plan-bound matched errors')
    layers = json.loads((engine/'layers.json').read_text(encoding='utf-8'))['Layers']
    convs = [x for x in layers if x.get('ParameterType') == 'Convolution']
    int8_convs = [x for x in convs if x['Weights']['Type'] == 'Int8']
    kernels = json.loads((root/'holdout-packets-trt/trt-kernels.json').read_text(encoding='utf-8'))['kernels']
    integer_kernels = [x for x in kernels if 'i8' in x.lower() and 'tensor16x8x32' in x.lower()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    variants = {}
    with np.load(root/'physical-validation/physical.npz', allow_pickle=False) as arrays:
        for variant, items in physical['variants'].items():
            if variant == 'reference':
                continue
            values = []
            for index, item in enumerate(items):
                elev, climate = arrays[f'{variant}_{index}_elev'], arrays[f'{variant}_{index}_climate']
                ref, rc = arrays[f'reference_{index}_elev'], arrays[f'reference_{index}_climate']
                c, bc = expand_climate(climate,elev.shape), expand_climate(rc,ref.shape)
                temperature = c[0]+c[4]*np.maximum(elev,0)
                baseline_temperature = bc[0]+bc[4]*np.maximum(ref,0)
                values.append(dict(site=item['site'], elevation_error_m=item['elevation_error_m'],
                    physical_gate_passed=item['physical_gate_passed'], climate_errors=item['climate_errors'],
                    displayed_temperature_error_c=error(temperature,baseline_temperature),
                    coastline_sign_disagreement=item['coastline_sign_disagreement']))
            variants[variant] = values
    if physical['physical_holdout_passed'] or any(
        item['physical_gate_passed'] for rows in variants.values() for item in rows):
        raise ValueError('This rejection synthesis requires every tested physical gate to fail')
    receipts = ['physical-validation/physical.json','physical-validation/physical.npz',
        'matched-bf16-trt-admitted/matched.json','matched-bf16-trt-fp32-admitted/matched.json',
        'holdout-packets-trt/packets.json','holdout-packets-trt/trt-kernels.json',
        'trt-qdq-attempt2/build.json','trt-qdq-attempt2/layers.json','trt-fp32/build.json',
        'trt-qdq-attempt2.log','trt-fp32.log']
    report = dict(role=ROLE, runtime_enabled=False, state='complete', verdict='rejected-keep-quantization-off',
        producer_sha=sha_file(__file__), evidence={name:sha_file(source_path(name, root=root)) for name in receipts},
        integer_gpu_execution_proven=bool(int8_convs and integer_kernels),
        integer_coverage=dict(int8_convolutions=len(int8_convs),
            convolution_family_layers=len(convs), layers=len(layers), actual_unique_cuda_kernels=len(kernels),
            actual_int8_tensorcore_kernel_names=integer_kernels,
            warning='Layer counts include three float deconvolutions and do not imply a MAC-weighted coverage percentage.'),
        network_packet_median_speedups_vs_current_bf16=[x['median_paired_speedup'] for x in matched['packets']],
        fp32_network_packet_median_speedups_vs_current_bf16=[x['median_paired_speedup'] for x in fp32['packets']],
        physical_paired_speedups=[dict(site=x['site'],median=x['median_paired_speedup'],
                                      values=[p['speedup'] for p in x['pairs']]) for x in matched['physical_pairs']],
        physical_median_speedup=float(statistics.median(x['median_paired_speedup'] for x in matched['physical_pairs'])),
        reference_cuda_graphs=matched['reference_graph_stats'], variants=variants,
        reference_profile=matched['reference_profile'],
        reference_kernel_status=matched['reference_kernel_status_after'],
        stream_evidence='Both admitted producer revisions explicitly construct torch.cuda.Stream(device=selected). Handles are recorded; no portable default-stream classification is inferred from their numeric values.',
        historical_physical_trt_attribution='Original physical receipt lacks plan_sha; identical complete elevation/climate error dictionaries on all four sites are corroborated by the plan-bound matched physical pairs. Original receipt unchanged.',
        gate=physical['gate'], physical_holdout_passed=physical['physical_holdout_passed'],
        split='Historical calibration seeds42/0/43 immutable. Seeds101/202/303/404 began as holdout; after observing diagnostics they are validation/tuning only. No candidate passed, so no final acceptance or untouched-final-holdout certification is claimed.',
        limitations=['Only base model preview investigated; coarse and decoder unchanged.',
            'Every world/store was private and fresh. No approximate latent stored or activated in production.',
            'No full native decoder corpus/browser performance claim. LOD3 gate already failed.',
            'The first Torch packet run used cuDNN default TF32 policy and is diagnostic only; physical/matched runs explicitly disabled TF32.',
            'CUDA trace proof was recorded on the default stream; reported speedups use nondefault stream matched AB/BA runs.',
            'Desktop GPU activity is uncontrolled. Timing conclusions are specific to this RTX3090/runtime.',
            'Display temperature uses the exact existing CPU palette reconstruction formula; no independent biome classifier ground truth.'])
    write_json(args.output,report)
    print(json.dumps(dict(verdict=report['verdict'],integer_proven=report['integer_gpu_execution_proven'],
                          physical_speedup=report['physical_median_speedup'])),flush=True)


if __name__ == '__main__':
    main()

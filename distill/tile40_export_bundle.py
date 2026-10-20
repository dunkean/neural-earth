"""Export all three models and a complete browser numerical-reference fixture."""
import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from distill.common import DATA, REPO, atomic_json, atomic_write, external_path
from distill.features import COND_MEANS, COND_STDS, base_features
from distill.native_weight_quant import file_sha
from distill.student import load_student
from distill.tile40_conditioning import CLIMATES
from distill.tile40_export_coarse import ExportCoarse
from distill.tile40_runtime import Tile40
from distill.tile40_export_utils import EncoderRegion,compact_decoder_input







def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder', type=Path, required=True)
    parser.add_argument('--decoder', type=Path, default=DATA/'tile40-20261014/decoder16-pilot-v1/step-020000.pt')
    parser.add_argument('--dataset', type=Path, default=DATA/'tile40-20261014/reference-coarse2-guided-v4')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--encoder-weights', choices=('raw', 'ema'), default='ema')
    parser.add_argument('--decoder-weights', choices=('raw', 'ema'), default='ema')
    parser.add_argument('--compact-decoder-input', action='store_true')
    parser.add_argument('--whole-decoder', action='store_true',help='Export one1856px finite-root decoder call.')
    parser.add_argument('--flow',action='store_true',help='Export the complete latent flow,including its spatial prior and integration.')
    parser.add_argument('--finite-flow',action='store_true',help='A single finite tile;no infinite-world dependency halo.')
    parser.add_argument('--flow-steps',type=int,default=2,help='Midpoint intervals;two model calls per interval.')
    args = parser.parse_args()
    if args.flow_steps < 1 or (args.finite_flow and not args.flow):
        parser.error('Positive flow intervals and --flow with --finite-flow are required.')
    root = DATA/'tile40-20261014'
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve prior bundles;use a fresh output.')
    output.mkdir(parents=True)
    torch.set_num_threads(8)
    dataset = args.dataset
    plan = json.loads((dataset/'manifest.json').read_text())
    train = next(row for row in plan['cases'] if row['split']=='train')
    case = next(row for row in plan['cases'] if row['split']=='val' and row['relief']=='mountains' and row['climate']=='arid')
    with np.load(dataset/train['file'], allow_pickle=False) as item:
        parameters = json.loads(str(item['metadata']))['world_config']['parameters']
    with np.load(dataset/case['file'], allow_pickle=False) as item:
        elevation = item['sketch_m'].copy()
    encoder, saved = load_student(args.encoder, ema=args.encoder_weights=='ema')
    if args.flow:
        from distill.tile40_flow_pilot import FlowEncoder,FiniteFlowEncoder
        encoder_class=FiniteFlowEncoder if args.finite_flow else FlowEncoder
        encoder=encoder_class(encoder,saved['contract'],steps=args.flow_steps)
    decoder, decoder_saved = load_student(args.decoder, ema=args.decoder_weights=='ema')
    coarse, _ = load_student(REPO/'distill/weights/coarse2.pt')
    generator = Tile40(ExportCoarse(coarse.solver), encoder, decoder, parameters,
                       whole_decoder=args.whole_decoder)
    actual = generator.generate(elevation, case['climate'], case['seed'], trace=True)
    halo = encoder.halo
    inputs = base_features(actual['coarse'], generator.coarse_origin, generator.coarse_origin, case['seed'], -64-halo, -64-halo,
                           320+2*halo, parameters['histogram_raw'])[None]
    cropped = EncoderRegion(encoder).eval().requires_grad_(False)
    with torch.no_grad():
        expected = cropped(inputs).numpy()
    artifact = output/'encoder-f32.onnx'
    torch.onnx.export(cropped, inputs, str(artifact), input_names=['inputs'], output_names=['latents'],
                      opset_version=17, dynamo=False, do_constant_folding=True, external_data=False)
    onnx.checker.check_model(str(artifact))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(str(artifact), sess_options=options, providers=['CPUExecutionProvider'])
    exported = session.run(None, {'inputs':inputs.numpy()})[0]
    max_abs = float(np.max(np.abs(exported-expected)))
    if not np.isfinite(exported).all() or max_abs > .0005:
        raise ValueError(f'Encoder export changed the numerical reference:{max_abs=}')
    models = dict(encoder=dict(file=artifact.name, output_name='latents', sha256=file_sha(artifact),
        bytes=artifact.stat().st_size, checkpoint_sha256=file_sha(args.encoder), step=saved['step'],
        selected_weights=args.encoder_weights, export_max_abs=max_abs, input_shape=list(inputs.shape)))
    for stage, directory in [('coarse', root/'coarse2-browser-v3')]:
        previous = json.loads((directory/'manifest.json').read_text())
        source = directory/previous['model']
        if file_sha(source) != previous['model_sha256']:
            raise ValueError('A frozen component export changed.')
        destination = output/f'{stage}-f32.onnx'
        # Real files are portable through the Windows UNC bridge and packaging.
        # These frozen ONNX exports share storage through regular hard links.
        os.link(source, destination)
        models[stage] = dict(file=destination.name, output_name=previous.get('output_name', 'residual'),
            sha256=previous['model_sha256'], bytes=source.stat().st_size,
            checkpoint_sha256=previous['checkpoint_sha256'], step=previous['step'])
    from distill.features import decoder_features
    if args.whole_decoder:
        fixture = decoder_features(torch.from_numpy(actual['latents'][:4, 32:264, 32:264]),
                                   case['seed'], -256, -256,1856,noise_tile=512)[None]
    else:
        fixture = decoder_features(torch.from_numpy(actual['latents'][:4, 64:128, 64:128]), case['seed'], 0, 0)[None]
    with torch.no_grad():
        expected_residual = decoder(fixture).numpy()
    selected_decoder = decoder
    selected_fixture = fixture
    destination = output/'decoder-f32.onnx'
    torch_max_abs = 0.
    if args.compact_decoder_input:
        destination = output/'decoder-five-inputs-f32.onnx'
        selected_decoder = compact_decoder_input(decoder)
        selected_fixture = fixture[:, :5]
        with torch.no_grad():
            folded_residual = selected_decoder(selected_fixture).numpy()
        torch_max_abs = float(np.max(np.abs(folded_residual-expected_residual)))
        if torch_max_abs > .0001:
            raise ValueError(f'Dropping zero channels changed the decoder:{torch_max_abs=}')
    torch.onnx.export(selected_decoder, selected_fixture, str(destination), input_names=['inputs'], output_names=['residual'],
        opset_version=17, dynamo=False, do_constant_folding=True, external_data=False)
    onnx.checker.check_model(str(destination))
    decoder_session = ort.InferenceSession(str(destination), sess_options=options, providers=['CPUExecutionProvider'])
    decoder_actual = decoder_session.run(None, {'inputs':selected_fixture.numpy()})[0]
    decoder_max_abs = float(np.max(np.abs(decoder_actual-expected_residual)))
    if not np.isfinite(decoder_actual).all() or decoder_max_abs > .0001:
        raise ValueError(f'Decoder export changed the reference:{decoder_max_abs=}')
    models['decoder'] = dict(file=destination.name, output_name='residual',
        input_channels=int(selected_fixture.shape[1]), input_shape=list(selected_fixture.shape),
        layout='finite-root' if args.whole_decoder else 'overlap512',
        sha256=file_sha(destination), bytes=destination.stat().st_size,
        checkpoint_sha256=file_sha(args.decoder), step=decoder_saved['step'], selected_weights=args.decoder_weights,
        zero_channel_fold_torch_max_abs=torch_max_abs, export_max_abs=decoder_max_abs)
    if any(model['bytes'] >= 100000000 for model in models.values()):
        raise ValueError('An exported model exceeds100MB.')
    theta = torch.atan(torch.tensor(parameters['cond_snr'])).bfloat16()
    parameters.update(cos_theta=torch.cos(theta).float().tolist(), sin_theta=torch.sin(theta).float().tolist(),
        labels=torch.log(torch.tan(theta)/8).float().tolist())
    means, stds = np.asarray(parameters['coarse_means'], np.float32), np.asarray(parameters['coarse_stds'], np.float32)
    means[1], stds[1] = means[0]-means[1], stds[0]
    parameters.update(physical_means=means.tolist(), physical_stds=stds.tolist())
    axis = torch.arange(-5, 6, dtype=torch.float32)
    laplacian = torch.exp(-.5*(axis/5.).square())
    laplacian /= laplacian.sum()
    gaussian = np.exp(-.5*(np.arange(-32, 33, dtype=np.float64)/8.)**2)
    gaussian /= gaussian.sum()
    atomic_json(output/'input.json', dict(sketch_m=elevation.tolist(), climate=case['climate'], seed=case['seed']))
    atomic_write(output/'expected-height.bin', lambda handle:handle.write(actual['height_m'].astype('<f4').tobytes()))
    manifest = dict(schema=1, models=models, total_model_bytes=sum(model['bytes'] for model in models.values()),
        encoder_halo=halo, coarse_origin=generator.coarse_origin, coarse_size=generator.coarse_size,
        parameters=parameters, condition_means=COND_MEANS.tolist(), condition_stds=COND_STDS.tolist(),
        climates=CLIMATES, kernels=dict(laplacian=laplacian.tolist(), guidance=gaussian.tolist()),
        side_m=40000., pixels=1334, spacing_m=40000./1334, fixture_case=case,
        input_file='input.json', input_sha256=file_sha(output/'input.json'),
        expected_file='expected-height.bin', expected_sha256=file_sha(output/'expected-height.bin'),
        native_reference_timing=actual['timing'], dataset_manifest_sha256=file_sha(dataset/'manifest.json'),
        encoder_training_manifest_sha256=saved.get('contract',{}).get('dataset_manifest_sha256'),
        sampling=dict(kind='conditional-flow',evaluations=2*args.flow_steps,intervals=args.flow_steps,
            finite_tile=args.finite_flow,colored_prior=saved['contract'].get('colored_prior',False)) if args.flow else dict(kind='one-pass'),
        scope='Complete input-to-height browser candidate;not yet quality-accepted or shipped.',
        provenance=dict(runtime_sha256=file_sha(REPO/'webgpu/tile40_runtime.mjs'),
                        noise_sha256=file_sha(REPO/'webgpu/noise.mjs')))
    atomic_json(output/'manifest.json', manifest)
    snapshot=output/'runtime-snapshot';snapshot.mkdir()
    import shutil
    for name in ('tile40_runtime.mjs','noise.mjs'):
        shutil.copy2(REPO/'webgpu'/name,snapshot/name)
    print(json.dumps(dict(models=models, total_model_bytes=manifest['total_model_bytes'], native_reference_timing=actual['timing'])), flush=True)


if __name__ == '__main__':
    main()

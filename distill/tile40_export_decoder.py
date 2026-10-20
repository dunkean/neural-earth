"""Export and verify the compact decoder on a real TRAIN fixture for browser timing."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from distill.common import DATA, atomic_json, atomic_write, external_path
from distill.native_basis_training import inventory
from distill.native_decoded import DecodedRegion
from distill.native_weight_quant import file_sha
from distill.student import load_student
from distill.tile40_decoder_projection import project_decoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DATA/'final/weights/candidate-1/decoder.pt')
    parser.add_argument('--output', type=Path, default=DATA/'tile40-20261014/decoder-browser-v1')
    parser.add_argument('--width', type=int, help='Benchmark a projected initialization;not a trained smaller model.')
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve previous exports;use a fresh output directory.')
    output.mkdir(parents=True)
    torch.set_num_threads(8)
    model, saved = load_student(args.checkpoint, 'cpu')
    model.requires_grad_(False)
    if model.config.stage != 'decoder':
        raise ValueError('A decoder checkpoint is required.')
    source_width = model.config.width
    if args.width is not None and args.width != source_width:
        model = project_decoder(model, args.width)
        model.requires_grad_(False)
    records, _, _ = inventory()
    record = next(row for row in records if row['case_kind'] == 'wet-high-mountain' and row['coarse_variant'] == 'coarse4')
    region = DecodedRegion(Path(record['root'])/record['file'], 'cpu')
    fields = torch.nn.functional.interpolate(region.target[None, :4], size=(512, 512), mode='nearest')
    inputs = torch.cat([region.decoder_noise, fields, torch.zeros(1, 11, 512, 512)], dim=1).float()
    with torch.no_grad():
        expected = model(inputs).numpy()
    artifact = output/'decoder-f32-512.onnx'
    torch.onnx.export(model, inputs, str(artifact), input_names=['inputs'], output_names=['residual'],
                      opset_version=17, dynamo=False, do_constant_folding=True, external_data=False)
    onnx.checker.check_model(str(artifact))
    if artifact.stat().st_size >= 100000000:
        raise ValueError('Browser artifact exceeds the100MB model budget.')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(str(artifact), sess_options=options, providers=['CPUExecutionProvider'])
    actual = session.run(None, {'inputs':inputs.numpy()})[0]
    delta = actual.astype(np.float64)-expected.astype(np.float64)
    max_abs, rmse = float(np.max(np.abs(delta))), float(np.sqrt(np.mean(delta*delta)))
    if actual.shape != expected.shape or not np.isfinite(actual).all() or max_abs > .0005:
        raise ValueError(f'CPU export changed the decoder:{max_abs=} {rmse=}')
    timings = []
    for _ in range(3):
        before = time.perf_counter()
        session.run(None, {'inputs':inputs.numpy()})
        timings.append(time.perf_counter()-before)
    atomic_write(output/'inputs.bin', lambda handle:handle.write(inputs.numpy().astype('<f4').tobytes()))
    atomic_write(output/'expected.bin', lambda handle:handle.write(expected.astype('<f4').tobytes()))
    report = dict(status='complete', model=artifact.name, model_bytes=artifact.stat().st_size,
        model_sha256=file_sha(artifact), checkpoint_sha256=file_sha(args.checkpoint), step=saved['step'],
        input_shape=list(inputs.shape), output_shape=list(expected.shape), input_dtype='float32',
        input_file='inputs.bin', input_sha256=file_sha(output/'inputs.bin'),
        expected_file='expected.bin', expected_sha256=file_sha(output/'expected.bin'),
        cpu_forward_comparison=dict(max_abs=max_abs, rmse=rmse), native_ort_seconds=timings,
        fixture=dict(root=record['root'], file=record['file'], split='train', seed=record['metadata']['seed']),
        versions=dict(torch=torch.__version__, onnx=onnx.__version__, ort=ort.__version__),
        model_width=model.config.width, source_width=source_width,
        projected_initialization=model.config.width != source_width,
        scope='One512px decoder window only;native timing is not browser or full40km timing.')
    atomic_json(output/'manifest.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

"""Export the existing two-step hidden coarse context for browser verification."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnx
import onnxruntime as ort
import torch

from distill.common import DATA, REPO, atomic_json, atomic_write, external_path
from distill.features import coarse_features
from distill.native_weight_quant import file_sha
from distill.student import load_student
from distill.tile40_export_utils import ExportCoarse





def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=REPO/'distill/weights/coarse2.pt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve existing exports;use a fresh output directory.')
    output.mkdir(parents=True)
    torch.set_num_threads(8)
    model, saved = load_student(args.checkpoint, 'cpu')
    if model.config.stage != 'coarse' or model.config.solver_steps != 2:
        raise ValueError('The frozen two-step coarse context is required.')
    model.requires_grad_(False)
    fixture = sorted((DATA/'crops/main/coarse').glob('train*.npz'))[0]
    with np.load(fixture, allow_pickle=False) as sample:
        meta = json.loads(str(sample['metadata']))
        inputs = coarse_features(sample['condition'], sample['labels'], meta['seed'], meta['y'], meta['x'])[None]
    with torch.no_grad():
        expected = model(inputs).numpy()
    exported = ExportCoarse(model.solver).eval()
    with torch.no_grad():
        torch.testing.assert_close(exported(inputs), torch.from_numpy(expected), atol=0., rtol=0.)
    artifact = output/'coarse2-f32-64.onnx'
    torch.onnx.export(exported, inputs, str(artifact), input_names=['inputs'], output_names=['coarse'],
        opset_version=17, dynamo=False, do_constant_folding=True, external_data=False)
    graph = onnx.load(str(artifact))
    repaired = []
    for node in graph.graph.node:
        attributes = {attribute.name:attribute for attribute in node.attribute}
        # Upstream MPConv uses a one-element PyTorch padding tuple. The legacy
        # exporter emits two pads for a2D Conv;ONNX needs begin/end per axis.
        if node.op_type == 'Conv' and len(attributes['kernel_shape'].ints) == 2 and len(attributes['pads'].ints) == 2:
            pads = list(attributes['pads'].ints)
            attributes['pads'].ints[:] = pads+pads
            repaired.append(node.name)
    onnx.checker.check_model(graph)
    onnx.save(graph, str(artifact))
    if artifact.stat().st_size >= 100000000:
        raise ValueError('The complete artifact exceeds100MB.')
    options = ort.SessionOptions()
    options.intra_op_num_threads = 8
    session = ort.InferenceSession(str(artifact), sess_options=options, providers=['CPUExecutionProvider'])
    actual = session.run(None, {'inputs':inputs.numpy()})[0]
    delta = actual.astype(np.float64)-expected.astype(np.float64)
    max_abs = float(np.abs(delta).max())
    if actual.shape != expected.shape or not np.isfinite(actual).all() or max_abs > .001:
        raise ValueError(f'Export changed the context:{max_abs=}')
    # Different conditioning and labels catch accidental trace-time input constants.
    varied = inputs.clone()
    varied[:, 6:11] += .17
    varied[:, 11:] -= .09
    with torch.no_grad():
        varied_expected = model(varied).numpy()
    varied_actual = session.run(None, {'inputs':varied.numpy()})[0]
    varied_max_abs = float(np.abs(varied_actual-varied_expected).max())
    if not np.isfinite(varied_actual).all() or varied_max_abs > .001:
        raise ValueError(f'Export froze conditioning or labels:{varied_max_abs=}')
    timings = []
    for _ in range(3):
        before = time.perf_counter()
        session.run(None, {'inputs':inputs.numpy()})
        timings.append(time.perf_counter()-before)
    atomic_write(output/'inputs.bin', lambda handle:handle.write(inputs.numpy().astype('<f4').tobytes()))
    atomic_write(output/'expected.bin', lambda handle:handle.write(expected.astype('<f4').tobytes()))
    scope = 'One64px hidden coarse-context window;not a full40km terrain generation.'
    report = dict(status='complete', stage='coarse', model=artifact.name,
        model_bytes=artifact.stat().st_size, model_sha256=file_sha(artifact),
        checkpoint_sha256=file_sha(args.checkpoint), step=saved['step'], output_name='coarse',
        input_shape=list(inputs.shape), output_shape=list(expected.shape), input_dtype='float32',
        input_file='inputs.bin', input_sha256=file_sha(output/'inputs.bin'),
        expected_file='expected.bin', expected_sha256=file_sha(output/'expected.bin'),
        cpu_forward_comparison=dict(max_abs=max_abs, varied_max_abs=varied_max_abs,
                                    rmse=float(np.sqrt(np.mean(delta*delta)))),
        native_ort_seconds=timings, fixture=dict(file=str(fixture), metadata=meta),
        scope=scope, browser_scope=scope, broadcast_padding_repairs=repaired,
        versions=dict(torch=torch.__version__, onnx=onnx.__version__, ort=ort.__version__))
    atomic_json(output/'manifest.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()

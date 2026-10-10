"""Check optimized coarse inference against the preserved previous implementation.

This is a numerical check, not a performance measurement. Both eager and graph
replay must match the prior solver exactly on complete held-out feature windows.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from distill.common import REPO, atomic_json, external_path
from distill.dataset import Crops
from distill.features import base_features, decoder_features
from distill.inference import base_inputs, decoder_inputs, coarse_graph_forward
from distill.student import load_student


def legacy_forward(path):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'CoarseSolver')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_forward')
    fn.name = 'legacy_forward'
    module = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    namespace = {'torch': torch}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace['legacy_forward']


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--legacy-source', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, saved = load_student(args.checkpoint, 'cuda:0')
    if not model.config.solver_steps:
        parser.error('A trained coarse solver checkpoint is required.')
    old = legacy_forward(args.legacy_source)
    dataset = Crops(args.dataset, 'coarse', 'val', halo=0)
    indices = sorted(set(int(i*(len(dataset)-1)/5) for i in range(6)))
    # Preserve outputs before exact runtime constant aliases are installed.
    references = []
    for index in indices:
        inputs = dataset[index][0][None].to('cuda:0')
        with torch.autocast('cuda', enabled=False):
            references.append(old(model.solver, inputs).clone())
    model.solver.prepare_capture()
    report = dict(checkpoint=str(args.checkpoint), checkpoint_digest=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                  legacy_source_digest=hashlib.sha256(args.legacy_source.read_bytes()).hexdigest(),
                  gpu=torch.cuda.get_device_name(), stage='coarse', step=saved['step'], rows=[],
                  note='Exact numerical comparison; GPU may share training work. No speed claim.')
    report['student_source_digests'] = {'code:'+name: hashlib.sha256((REPO/name).read_bytes()).hexdigest()
        for name in ('distill/student.py', 'distill/features.py', 'distill/inference.py', 'distill/coarse_solver.py')}
    static = dataset[indices[0]][0][None].to('cuda:0')
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            model(static)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
        output = model(static)
    for index, reference in zip(indices, references):
        inputs = dataset[index][0][None].to('cuda:0')
        eager = model(inputs)
        static.copy_(inputs)
        graph.replay()
        captured = output.clone()
        runtime = coarse_graph_forward(model, inputs)
        row = dict(file=dataset.paths[index].name,
                   eager_exact=torch.equal(eager, reference), graph_exact=torch.equal(captured, reference),
                   runtime_graph_exact=torch.equal(runtime, reference),
                   eager_max_abs=float((eager-reference).abs().max()),
                   graph_max_abs=float((captured-reference).abs().max()))
        if not torch.isfinite(reference).all() or not torch.isfinite(captured).all():
            raise FloatingPointError('Non-finite coarse comparison.')
        report['rows'].append(row)
        print(json.dumps(row), flush=True)
    report['feature_rows'] = []
    for index in indices:
        path = dataset.paths[index]
        with np.load(Path(args.dataset)/'base'/path.name, allow_pickle=False) as sample:
            m = json.loads(str(sample['metadata']))
            size = 256+2*192
            parameters = (sample['coarse'], m['coarse_y'], m['coarse_x'], m['seed'],
                          m['y']-192, m['x']-192, size, sample['histogram'])
            reference_features = base_features(*parameters)
            features = base_inputs(*parameters, 'cuda:0')
            base_exact = torch.equal(features.cpu(), reference_features)
        with np.load(Path(args.dataset)/'decoder'/path.name, allow_pickle=False) as sample:
            m = json.loads(str(sample['metadata']))
            latents = torch.from_numpy(sample['latents'].astype('float32'))
            reference_features = decoder_features(latents, m['seed'], m['y'], m['x'])
            features = decoder_inputs(latents.to('cuda:0'), m['seed'], m['y'], m['x'], 512, 'cuda:0')
            decoder_exact = torch.equal(features.cpu(), reference_features)
        report['feature_rows'].append(dict(file=path.name, base_exact=base_exact, decoder_exact=decoder_exact))
    report['exact_passed'] = all(r['eager_exact'] and r['graph_exact'] and r['runtime_graph_exact'] for r in report['rows'])
    report['exact_passed'] &= all(r['base_exact'] and r['decoder_exact'] for r in report['feature_rows'])
    atomic_json(external_path(args.output), report)
    if not report['exact_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

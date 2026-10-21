"""Build native160x2 one-pass weights, packed size evidence and CUDA controls."""
import argparse
import json
from pathlib import Path

import torch

from distill.common import DATA, REVISION, REPO, atomic_json, atomic_write, external_path
from distill.native_basis_training import inventory
from distill.native_decoded import DecodedRegion
from distill.native_onepass import initialize, ARCHITECTURE, noise_windows
from distill.native_onepass_runtime import OnePassRuntime
from distill.native_progressive_prepare import configure
from distill.native_weight_quant import file_sha
from distill.native_scalar_precision import ScalarPrecisionBase
from distill.native_scalar_pack import export_scalar_packed


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve previous preparation.')
    configure()
    source = torch.load(args.source, map_location='cpu', weights_only=True, mmap=True)
    model, mapping = initialize(source)
    saved = dict(architecture=ARCHITECTURE, config=model.config, gain_anchor=model.gain_anchor,
        solver_evaluations=1, teacher_solver_evaluations=2, window_size=64, stride=32, input_noise_offsets=[5819,5820],
        teacher_revision=REVISION, teacher_weights_sha256=source['teacher_weights_sha256'],
        parent_sha256=file_sha(args.source), transfer_mapping=mapping, step=0, accepted=False,
        model=model.state_dict(), ema={name:value.bfloat16() if name != 'net.out_gain' else value.float()
                                     for name,value in model.state_dict().items()})
    output.mkdir(parents=True)
    atomic_write(output/'initial.pt', lambda handle:torch.save(saved, handle))
    # Actual nibble artifact; initial PTQ quality is not promoted or accepted.
    quant = ScalarPrecisionBase(model.config)
    quant.set_precision('anchored', model.gain_anchor)
    quant.load_state_dict(saved['ema'])
    packed_source = dict(architecture='native-weight-quant-scalar', scalar_precision='anchored-fp32-v1',
        gain_anchor=model.gain_anchor, config=model.config, scheme='nf4', block=512,
        quantized_layers=quant.quantized_layers, ema=saved['ema'], teacher_revision=REVISION,
        teacher_weights_sha256=source['teacher_weights_sha256'], step=0)
    packed = export_scalar_packed(packed_source, output/'initial-ptq-size')
    if packed['bytes'] >= 100000000:
        raise ValueError('Architecture misses the actual weights.bin100MB budget.')
    del quant, model
    model, _ = initialize(source)
    model = model.cuda().eval()
    runtime = OnePassRuntime(saved, 'cuda')
    records, _, _ = inventory()
    checks = []
    for kind in ('desert-plain', 'wet-complex-coast', 'wet-high-mountain'):
        row = next(r for r in records if r['case_kind'] == kind and r['coarse_variant'] == 'coarse4')
        region = DecodedRegion(Path(row['root'])/row['file'], 'cuda')
        noises = noise_windows(row['metadata']['seed'], region.ref_origins, 'cuda')
        condition = region.stages[1]['condition']
        prediction = model.windows(noises, condition)
        inference = runtime(noises, condition)
        if not torch.equal(prediction, inference):
            raise ValueError(f'Frozen one-pass runtime changed windows:{kind}:{float((prediction-inference).abs().max())}')
        if not torch.isfinite(inference).all():
            raise FloatingPointError('Nonfinite one-pass output.')
        checks.append(dict(kind=kind, seed=row['metadata']['seed'], windows=len(noises), exact=True))
    # A real TRAIN update proves that the initially unused second noise can learn.
    with torch.enable_grad():
        model.requires_grad_(True)
        noises = noises[:1]; condition = condition[:1]
        prediction = model.windows(noises, condition)
        loss = prediction.square().mean()
        loss.backward()
    first = next(iter(model.net.enc.values())).weight
    gradient = first.grad[:,5:10].abs().sum()
    if not torch.isfinite(gradient) or float(gradient) <= 0:
        raise ValueError('Second noise has no learnable input gradient.')
    atomic_json(output/'proof.json', dict(status='complete', initial_sha256=file_sha(output/'initial.pt'),
        source_sha256=file_sha(args.source), parameters=sum(p.numel() for p in model.parameters()),
        initial_nf4_weights_bytes=packed['bytes'], initial_nf4_sha256=packed['sha256'],
        download_budget_satisfied=True, ptq_quality_accepted=False, forward_checks=checks,
        second_noise_gradient=float(gradient), gpu=torch.cuda.get_device_name(),
        code_sha256={name:file_sha(REPO/'distill'/name) for name in ('native_onepass.py','native_onepass_runtime.py')},
        note='Size/forward/backward controls; one-pass fidelity and final3x speed remain unproven.'))
    print(json.dumps(dict(status='complete', weights_bytes=packed['bytes'], runtime_exact=len(checks))), flush=True)


if __name__ == '__main__':
    main()

"""Real TRAIN forward audit of compact scalar storage, including a sub-BF16 gain."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from distill.common import REPO, atomic_json, external_path
from distill.native_basis_training import inventory, WindowBank
from distill.native_decoded import DecodedRegion, OriginalDecoder, decoded_region_forward
from distill.decoded_loss import reconstructed_height
from distill.native_scalar_precision import load_scalar_precision
from distill.native_scalar_pack import export_scalar_packed, load_scalar_packed, runtime_scalar_packed
from distill.native_weight_quant import encode, file_sha
from terrain_diffusion.models.mp_layers import normalize


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reuse-existing-packed', action='store_true',
                        help='Explicitly audit the retained export from a previous terminal failed audit.')
    args = parser.parse_args()
    output = external_path(args.output)
    if (output/'audit.json').exists():
        raise ValueError('Preserve existing audits.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = torch.load(args.source, map_location='cpu', weights_only=True, mmap=True)
    if source['architecture'] != 'native-weight-quant':
        raise ValueError('Prototype audit starts from the immutable legacy70k source.')
    anchor = float(source['ema']['net.out_gain'].bfloat16().float())
    # Deliberately exercise a value which would be lost by BF16 serialization.
    gain = source['ema']['net.out_gain'].float() + .00006103515625
    if torch.equal(gain.bfloat16().float(), gain):
        raise ValueError('The test gain must exercise sub-BF16 scalar precision.')
    saved = source | dict(architecture='native-weight-quant-scalar', scalar_precision='anchored-fp32-v1',
        gain_anchor=anchor, ema=source['ema'] | {'net.out_gain': gain}, evaluation_only=True, accepted=False)
    if args.reuse_existing_packed:
        manifest = json.loads((output/'packed/manifest.json').read_text())
    else:
        manifest = export_scalar_packed(saved, output/'packed', 'cuda')
    original = load_scalar_precision(saved, 'cuda')
    packed = load_scalar_packed(output/'packed', 'cuda')
    runtime = runtime_scalar_packed(output/'packed', 'cuda')
    eligible = {'net.'+name+'.weight' for name in saved['quantized_layers']}
    checked = []
    for name, value in packed.state_dict().items():
        expected = saved['ema'][name].to(value.device).to(torch.float32 if name == 'net.out_gain' else torch.bfloat16)
        if name in eligible:
            effective = normalize(expected.float())/math.sqrt(expected[0].numel())
            expected = encode(effective, saved['scheme'], saved['block'])[2].bfloat16()
        if not torch.equal(value, expected.to(value.dtype)):
            raise ValueError('Packed effective tensor differs: '+name)
        checked.append(name)
    records, _, _ = inventory()
    bank = WindowBank(records)
    checks = []
    with original.cached_weights(), packed.cached_weights():
        for index in range(3):
            batch, _ = bank.batch(records, np.random.default_rng(123+index), 70000+index)
            actual = original(batch['inputs'], batch['labels'], [batch['condition']])
            candidate = packed(batch['inputs'], batch['labels'], [batch['condition']])
            baked = runtime(batch['inputs'].bfloat16(), noise_labels=batch['labels'].bfloat16(),
                            conditional_inputs=[batch['condition'].bfloat16()]).float()
            if not torch.equal(actual, candidate):
                raise ValueError('Packed scalar forward differs.')
            if not torch.equal(actual, baked):
                raise ValueError('Compact inference runtime differs.')
            checks.append(dict(index=index, windows=len(actual), max_difference=0.))
        record = next(r for r in records if r['case_kind'] == 'wet-high-mountain' and r['coarse_variant'] == 'coarse4')
        region = DecodedRegion(Path(record['root'])/record['file'], 'cuda')
        from terrain_diffusion.inference.world_pipeline import linear_weight_window
        weight = linear_weight_window(64, 'cuda', torch.float32)
        # Adapt only the call convention for the frozen runtime net.
        def frozen_forward(inputs, labels, conditions):
            return runtime(inputs.bfloat16(), noise_labels=labels.bfloat16(),
                           conditional_inputs=[v.bfloat16() for v in conditions]).float()
        fields = [decoded_region_forward(model, region, weight) for model in (original, packed, frozen_forward)]
        if any(not torch.equal(fields[0][k], r[k]) for r in fields[1:]
               for k in ('velocity0', 'inputs1', 'velocity1', 'core')):
            raise ValueError('Packed two-stage scalar field differs.')
        decoder = OriginalDecoder('cuda')
        heights = [reconstructed_height(decoder(r['core'][None], region.decoder_noise), r['core'][None]) for r in fields]
        if any(not torch.equal(heights[0], h) for h in heights[1:]):
            raise ValueError('Packed decoded physical field differs.')
    atomic_json(output/'audit.json', dict(status='complete', source_sha256=file_sha(args.source),
        scope='TRAIN-only inference/storage probe with gain displaced1/16BF16step; not a fitted model.',
        tensor_checks=checked, forward_checks=checks, two_stage_field_exact=True, decoded_field_exact=True,
        compact_runtime_exact=True, runtime_body_dtype=str(runtime.out_conv.weight.dtype),
        runtime_gain_dtype=str(runtime.out_gain.dtype),
        gain_fp32=float(gain), gain_anchor=anchor, gain_bf16=float(gain.bfloat16().float()),
        stored_bytes=manifest['bytes'], manifest_bytes=(output/'packed/manifest.json').stat().st_size,
        gpu=torch.cuda.get_device_name(), weight_updates=0, served_weights_changed=False,
        code_sha256={name: file_sha(REPO/'distill'/name) for name in
                     ('native_scalar_pack.py', 'native_scalar_pack_audit.py')}))
    print(json.dumps(dict(status='complete', tensors=len(checked), packed_bytes=manifest['bytes'],
                         gpu=torch.cuda.get_device_name())), flush=True)


if __name__ == '__main__':
    main()

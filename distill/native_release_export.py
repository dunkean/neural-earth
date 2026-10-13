"""Export and audit a trained scalar NF4 base for a versioned GitHub release."""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import zipfile

import numpy as np
import torch

from distill.common import REPO, atomic_json, external_path
from distill.native_basis_training import inventory, WindowBank
from distill.native_decoded import DecodedRegion, decoded_region_forward
from distill.native_scalar_precision import load_scalar_precision
from distill.native_scalar_pack import export_scalar_packed, load_scalar_packed, runtime_scalar_packed
from distill.native_weight_quant import encode, file_sha
from terrain_diffusion.models.mp_layers import normalize


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--release', required=True)
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists():
        raise ValueError('Preserve the existing release; use a new version directory.')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    if saved.get('architecture') != 'native-weight-quant-scalar' or saved.get('selected_weights') != 'ema':
        raise ValueError('A trained scalar EMA snapshot is required.')
    manifest = export_scalar_packed(saved, output, 'cuda')
    original = load_scalar_precision(saved, 'cuda')
    manifest.update(release=args.release, source_checkpoint_sha256=file_sha(args.checkpoint),
                    selected_weights='ema', source_arm=saved.get('arm'),
                    license='MIT', parameter_count=sum(value.numel() for value in original.parameters()),
                    upstream_model='xandergos/terrain-diffusion-30m')
    atomic_json(output/'manifest.json', manifest)
    packed = load_scalar_packed(output, 'cuda')
    runtime = runtime_scalar_packed(output, 'cuda')
    eligible = {'net.'+name+'.weight' for name in saved['quantized_layers']}
    checked = []
    for name, value in packed.state_dict().items():
        expected = saved['ema'][name].to(value.device).to(
            torch.float32 if name == 'net.out_gain' else torch.bfloat16)
        if name in eligible:
            effective = normalize(expected.float())/math.sqrt(expected[0].numel())
            expected = encode(effective, saved['scheme'], saved['block'])[2].bfloat16()
        if not torch.equal(value, expected.to(value.dtype)):
            raise ValueError('Packed tensor changed: '+name)
        checked.append(name)
    def frozen_forward(inputs, labels, conditions):
        return runtime(inputs.bfloat16(), noise_labels=labels.bfloat16(),
                       conditional_inputs=[value.bfloat16() for value in conditions]).float()
    records, _, _ = inventory()
    bank = WindowBank(records)
    forward_checks, region_checks = [], []
    with original.cached_weights(), packed.cached_weights():
        for index in range(3):
            batch, _ = bank.batch(records, np.random.default_rng(2026101300+index), saved['step']+index)
            outputs = [model(batch['inputs'], batch['labels'], [batch['condition']])
                       for model in (original, packed, frozen_forward)]
            if any(not torch.equal(outputs[0], value) for value in outputs[1:]):
                raise ValueError('Packed window inference changed.')
            forward_checks.append(dict(index=index, windows=len(outputs[0]), max_difference=0.))
        from terrain_diffusion.inference.world_pipeline import linear_weight_window
        weight = linear_weight_window(64, 'cuda', torch.float32)
        for kind in ('desert-plain', 'temperate-plain', 'wet-complex-coast',
                     'low-plain-sea', 'wet-high-mountain'):
            record = next(row for row in records if row['case_kind'] == kind and row['coarse_variant'] == 'coarse4')
            if record['metadata']['split'] != 'train':
                raise ValueError('Export audit must use TRAIN captures.')
            region = DecodedRegion(Path(record['root'])/record['file'], 'cuda')
            fields = [decoded_region_forward(model, region, weight)
                      for model in (original, packed, frozen_forward)]
            if any(not torch.equal(fields[0][key], value[key]) for value in fields[1:]
                   for key in ('velocity0', 'inputs1', 'velocity1', 'core')):
                raise ValueError('Packed native two-stage field changed: '+kind)
            region_checks.append(dict(kind=kind, seed=record['metadata']['seed'], max_difference=0.))
    audit = dict(status='complete', release=args.release, source_checkpoint_sha256=file_sha(args.checkpoint),
                 weights_sha256=manifest['sha256'], gpu=torch.cuda.get_device_name(),
                 tensor_checks=checked, window_checks=forward_checks, region_checks=region_checks,
                 compact_runtime_exact=True, updates=0,
                 code_sha256={name:file_sha(REPO/'distill'/name) for name in
                              ('native_release_export.py', 'native_scalar_pack.py', 'native_scalar_precision.py')})
    atomic_json(output/'audit.json', audit)
    shutil.copyfile(REPO/'terrain-diffusion/LICENSE', output/'LICENSE-TEACHER.txt')
    (output/'NOTICE.txt').write_text(
        'Derived from xandergos/terrain-diffusion-30m (Alexander Goslin), MIT license.\n'
        f'Teacher revision: {manifest["teacher_revision"]}\n'
        'NF4 codebook follows QLoRA / bitsandbytes (MIT); codec implemented independently.\n'
        'https://github.com/bitsandbytes-foundation/bitsandbytes\n'
        'Convolutions and activations remain BF16 in the reference runtime.\n'
        'NF4 reduces download size; native four-bit arithmetic is not supplied.\n')
    pending = output/(args.release+'.zip.pending')
    archive = output/(args.release+'.zip')
    members = sorted(path for path in output.iterdir() if path.is_file() and path.name != pending.name)
    with zipfile.ZipFile(pending, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as handle:
        for path in members:
            handle.write(path, path.name)
    with zipfile.ZipFile(pending) as handle:
        if handle.testzip() is not None:
            raise ValueError('Release archive is corrupt.')
    os.replace(pending, archive)
    (output/'SHA256SUMS.txt').write_text(''.join(f'{file_sha(path)}  {path.name}\n'
        for path in members+[archive]))
    print(json.dumps(dict(status='complete', release=args.release, weights_bytes=manifest['bytes'],
                          zip_bytes=archive.stat().st_size, weights_sha256=manifest['sha256'],
                          tensors=len(checked), fields=len(region_checks), gpu=torch.cuda.get_device_name())), flush=True)


if __name__ == '__main__':
    main()

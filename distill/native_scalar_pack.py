"""Compact NF4 format preserving the anchored FP32 output gain.

Existing BF16-gain artifacts and quantizer code remain unchanged. Internal
kernels retain their original packed codes/scales; only out_gain adds two bytes.
"""
import copy
import json
import math
from pathlib import Path
import shutil
import tempfile

import numpy as np
import torch

from distill.common import atomic_json, atomic_write, external_path
from distill.native_scalar_precision import ScalarPrecisionBase
from distill.native_weight_quant import export_packed, file_sha, unpack_nibbles


def replace_gain(legacy_file, legacy_manifest, output_file, gain):
    """Preserve every unrelated payload byte and move its offsets by two bytes."""
    manifest = copy.deepcopy(legacy_manifest)
    row = manifest['tensors']['net.out_gain']
    if row['dtype'] != 'bfloat16' or row['shape'] != [] or row['bytes'] != 2:
        raise ValueError('Exactly one BF16 scalar must be replaced.')
    start = row['offset']
    value = gain.detach().cpu().float()
    if value.shape != torch.Size([]) or not torch.isfinite(value):
        raise ValueError('Finite scalar gain required.')
    payload = value.numpy().astype('<f4', copy=False).tobytes()
    def write(handle):
        with Path(legacy_file).open('rb') as old:
            remaining = start
            while remaining:
                chunk = old.read(min(remaining, 1024*1024))
                if not chunk:
                    raise ValueError('Truncated prefix.')
                handle.write(chunk)
                remaining -= len(chunk)
            if len(old.read(2)) != 2:
                raise ValueError('Truncated gain.')
            handle.write(payload)
            shutil.copyfileobj(old, handle)
    atomic_write(output_file, write)
    for name, entry in manifest['tensors'].items():
        if name == 'net.out_gain':
            entry.update(dtype='float32', bytes=4)
        elif entry['offset'] > start:
            entry['offset'] += 2
        if entry.get('scale_offset', -1) > start:
            entry['scale_offset'] += 2
    manifest['bytes'] += 2
    manifest['sha256'] = file_sha(output_file)
    return manifest


@torch.no_grad()
def export_scalar_packed(saved, output, device='cpu'):
    if saved.get('architecture') != 'native-weight-quant-scalar' or saved.get('scalar_precision') != 'anchored-fp32-v1':
        raise ValueError('Explicit anchored scalar checkpoint required.')
    output = external_path(output)
    if (output/'manifest.json').exists() or (output/'weights.bin').exists():
        raise ValueError('Preserve existing artifacts; use a fresh output.')
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.legacy-staging-', dir=output) as staging:
        legacy = export_packed(saved | dict(architecture='native-weight-quant'), Path(staging), device)
        manifest = replace_gain(Path(staging)/'weights.bin', legacy, output/'weights.bin',
                                saved['ema']['net.out_gain'])
    manifest.update(architecture='native-weight-quant-scalar-packed',
        scalar_precision='anchored-fp32-v1', gain_anchor=float(saved['gain_anchor']),
        accepted=False, accelerated_arithmetic=False,
        note='BF16 activations and convolutions; FP32 anchored out_gain; baked NF4 kernels bypass normalization.')
    atomic_json(output/'manifest.json', manifest)
    return manifest


def read_tensor(handle, name, row, manifest):
    count = math.prod(row['shape'])
    handle.seek(row['offset'])
    payload = handle.read(row['bytes'])
    if len(payload) != row['bytes']:
        raise ValueError('Truncated tensor payload.')
    if row['dtype'] == 'float32':
        if name != 'net.out_gain' or row['shape'] != [] or row['bytes'] != 4:
            raise ValueError('Only the scalar output gain may be FP32.')
        value = torch.from_numpy(np.frombuffer(payload, dtype='<f4').copy())
    elif row['dtype'] == 'bfloat16':
        if row['bytes'] != count*2:
            raise ValueError('Invalid BF16 tensor length.')
        value = torch.from_numpy(np.frombuffer(payload, dtype='<u2').copy()).view(torch.bfloat16)
    elif row['dtype'] in ('nf4', 'int8'):
        codes = torch.from_numpy(np.frombuffer(payload, dtype=np.uint8).copy())
        if row['dtype'] == 'nf4':
            codes = unpack_nibbles(codes)
        handle.seek(row['scale_offset'])
        scale_payload = handle.read(row['scale_count']*4)
        if len(scale_payload) != row['scale_count']*4:
            raise ValueError('Truncated quantization scales.')
        scales = torch.from_numpy(np.frombuffer(scale_payload, dtype='<f4').copy())
        if codes.numel() != scales.numel()*manifest['block'] or count > codes.numel():
            raise ValueError('Invalid quantized block lengths.')
        values = (torch.tensor(manifest['codebook'])[codes.long()] if row['dtype'] == 'nf4'
                  else (codes.float()-128)/127)
        value = (values.reshape(-1, manifest['block'])*scales[:, None]).flatten()[:count].bfloat16()
    else:
        raise ValueError('Unsupported scalar artifact tensor dtype.')
    if not torch.isfinite(value).all():
        raise ValueError('Nonfinite reconstructed tensor.')
    return value.reshape(row['shape'])


def load_scalar_packed(output, device='cpu'):
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text())
    if (manifest['architecture'] != 'native-weight-quant-scalar-packed' or
            manifest['scalar_precision'] != 'anchored-fp32-v1' or manifest['endian'] != 'little' or
            manifest['file'] != 'weights.bin'):
        raise ValueError('Unsupported scalar packed contract.')
    artifact = output/'weights.bin'
    if artifact.stat().st_size != manifest['bytes'] or file_sha(artifact) != manifest['sha256']:
        raise ValueError('Scalar packed weights changed.')
    model = ScalarPrecisionBase(manifest['config'], manifest['scheme'], manifest['block'],
                                manifest['quantized_layers']).eval()
    if set(model.state_dict()) != set(manifest['tensors']):
        raise ValueError('Packed scalar tensor inventory mismatch.')
    state = {}
    with artifact.open('rb') as handle:
        for name, row in manifest['tensors'].items():
            state[name] = read_tensor(handle, name, row, manifest)
    model.load_state_dict(state)
    model._packed = True
    model.set_precision('anchored', manifest['gain_anchor'])
    return model.to(device).requires_grad_(False)


@torch.no_grad()
def runtime_scalar_packed(output, device='cpu'):
    owner = load_scalar_packed(output, device)
    gain = owner.net.out_gain.detach().float().clone()
    owner.net.bfloat16()
    owner.net.out_gain = torch.nn.Parameter(gain, requires_grad=False)
    net = owner.net.eval().requires_grad_(False)
    net._terrain_custom_base = True
    net._terrain_weight_quant_scheme = owner.scheme
    return net

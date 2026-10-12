"""Weight-only native QAT and a documented, nibble-packed inference artifact.

NF4 lookup values follow QLoRA / bitsandbytes (MIT), functional.get_4bit_type:
https://github.com/bitsandbytes-foundation/bitsandbytes/blob/main/bitsandbytes/functional.py
The implementation below is independent. No bitsandbytes dependency or kernel.
Activations retain native BF16; this is storage compression, not faster arithmetic.
"""
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
from types import MethodType

import numpy as np
import torch
from torch.nn import functional as F

from distill.common import atomic_json, atomic_write, external_path
from distill.native_base import NativeBase
from terrain_diffusion.models.mp_layers import MPConv, normalize

NF4 = (-1., -.6961928009986877, -.5250730514526367, -.39491748809814453,
       -.28444138169288635, -.18477343022823334, -.09105003625154495, 0.,
       .07958029955625534, .16093020141124725, .24611230194568634,
       .33791524171829224, .44070982933044434, .5626170039176941,
       .7229568362236023, 1.)


def encode(weight, scheme='nf4', block=512):
    if scheme not in ('nf4', 'int8') or block < 16:
        raise ValueError('NF4 or int8 and a positive block size >=16 required.')
    flat = weight.detach().float().reshape(-1)
    padded = F.pad(flat, (0, (-flat.numel()) % block)).reshape(-1, block)
    scale = padded.abs().amax(dim=1).clamp_min(1e-30)
    unit = padded / scale[:, None]
    if scheme == 'nf4':
        table = weight.new_tensor(NF4, dtype=torch.float32)
        midpoints = (table[1:] + table[:-1]) * .5
        codes = torch.bucketize(unit.contiguous(), midpoints).to(torch.uint8)
        actual = table[codes.long()] * scale[:, None]
    else:
        codes = ((unit * 127).round().clamp(-127, 127) + 128).to(torch.uint8)
        actual = (codes.float() - 128) * (scale[:, None] / 127)
    return codes.reshape(-1), scale, actual.reshape(-1)[:flat.numel()].reshape(weight.shape)


def fake_weight(weight, scheme='nf4', block=512):
    rounded = encode(weight, scheme, block)[2]
    return rounded.detach() + (weight - weight.detach())


def pack_nibbles(codes):
    flat = codes.reshape(-1).to(torch.uint8)
    if flat.numel() % 2 or torch.any(flat > 15):
        raise ValueError('Even number of four-bit codes required.')
    return flat[0::2] | (flat[1::2] << 4)


def unpack_nibbles(packed):
    result = torch.empty(packed.numel() * 2, dtype=torch.uint8)
    result[0::2], result[1::2] = packed & 15, packed >> 4
    return result


def weight_conv(layer, inputs, gain=1):
    owner = layer._weight_quant_owner
    if owner._packed:
        weight = layer.weight.to(inputs.dtype)
        if not isinstance(gain, (int, float)) or gain != 1:
            weight = (layer.weight.float() * gain).to(inputs.dtype)
    elif not owner.quantized:
        return MPConv.forward(layer, inputs, gain)
    else:
        weight = owner._cache.get(layer._weight_quant_name)
        if weight is None:
            effective = normalize(layer.weight.float()) / math.sqrt(layer.weight[0].numel())
            weight = fake_weight(effective, owner.scheme, owner.block)
            if owner._cache_enabled:
                if torch.is_grad_enabled():
                    raise ValueError('Weight cache is only valid for frozen/no-grad forwards.')
                owner._cache[layer._weight_quant_name] = weight.to(inputs.dtype)
        weight = (weight * gain).to(inputs.dtype)
    return F.conv2d(inputs, weight, padding=0 if layer.no_padding else weight.shape[-1] // 2,
                    groups=layer.groups)


class NativeWeightQuantBase(NativeBase):
    def __init__(self, config, scheme='nf4', block=512, layers=None):
        super().__init__(config)
        if scheme not in ('nf4', 'int8') or block < 16 or block % 2:
            raise ValueError('Valid quantization scheme and even block size required.')
        self.scheme, self.block = scheme, block
        first = 'enc.' + next(iter(self.net.enc))
        eligible = [n for n, m in self.net.named_modules() if isinstance(m, MPConv)
                    and m.weight.ndim == 4 and n not in (first, 'out_conv')]
        self.quantized_layers = list(eligible if layers is None else layers)
        if not self.quantized_layers or any(n not in eligible for n in self.quantized_layers):
            raise ValueError('Only native internal MP convolutions may be quantized.')
        self.quantized, self._packed = True, False
        self._cache, self._cache_enabled = {}, False
        for name in self.quantized_layers:
            layer = self.net.get_submodule(name)
            # An ordinary reference avoids registering the owner as a submodule.
            object.__setattr__(layer, '_weight_quant_owner', self)
            layer._weight_quant_name = name
            layer.forward = MethodType(weight_conv, layer)

    @contextmanager
    def cached_weights(self):
        if self._cache_enabled:
            raise ValueError('Nested caches are not supported.')
        self._cache_enabled = True
        try:
            yield
        finally:
            self._cache.clear()
            self._cache_enabled = False

    def forward(self, inputs, noise_labels, conditional_inputs):
        self.net.eval()
        state = {n: v.to(torch.bfloat16) for n, v in
                 list(self.net.named_parameters()) + list(self.net.named_buffers())}
        with torch.autocast(device_type=inputs.device.type, enabled=False):
            return torch.func.functional_call(self.net, state, (inputs.to(torch.bfloat16),),
                dict(noise_labels=noise_labels.to(torch.bfloat16),
                     conditional_inputs=[v.to(torch.bfloat16) for v in conditional_inputs])).float()


def load_weight_quant(saved, device='cpu'):
    if saved['architecture'] != 'native-weight-quant':
        raise ValueError('Weight-only native checkpoint required.')
    model = NativeWeightQuantBase(saved['config'], saved['scheme'], saved['block'],
                                 saved['quantized_layers']).to(device).eval()
    model.load_state_dict(saved['ema'])
    return model.requires_grad_(False)


def file_sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


@torch.no_grad()
def export_packed(saved, output, device='cpu'):
    """Store effective quantized convolutions, BF16 sensitive tensors and FP32 scales.

    C order; little endian; NF4 even code in low nibble, odd code in high nibble.
    Effective kernels already include global MP normalization and fan-in at gain1.
    Reconstructed quantized convolutions MUST NOT normalize those kernels again.
    """
    output = external_path(output)
    header_path = output / 'manifest.json'
    if header_path.exists():
        raise ValueError('Preserve existing packed artifacts; use a fresh directory.')
    tensors, offset = {}, 0
    eligible = {'net.' + n + '.weight' for n in saved['quantized_layers']}
    def write(f):
        nonlocal offset
        for name, original in saved['ema'].items():
            value = original.detach().cpu().bfloat16()
            row = dict(shape=list(value.shape), offset=offset)
            if name in eligible:
                # Match the device's reduction/rounding used during QAT. A CPU
                # global norm can move a code boundary versus the CUDA norm.
                effective = normalize(value.to(device).float()) / math.sqrt(value[0].numel())
                codes, scale, _ = encode(effective, saved['scheme'], saved['block'])
                codes = pack_nibbles(codes) if saved['scheme'] == 'nf4' else codes
                payload = codes.cpu().numpy().tobytes()
                scale_bytes = scale.cpu().numpy().astype('<f4', copy=False).tobytes()
                row.update(dtype=saved['scheme'], bytes=len(payload), scale_offset=offset+len(payload),
                           scale_count=scale.numel(), scale_dtype='float32', effective_gain=1)
                f.write(payload); f.write(scale_bytes)
                offset += len(payload) + len(scale_bytes)
            else:
                payload = value.contiguous().view(torch.uint16).numpy().astype('<u2', copy=False).tobytes()
                row.update(dtype='bfloat16', bytes=len(payload))
                f.write(payload); offset += len(payload)
            tensors[name] = row
    artifact = output / 'weights.bin'
    atomic_write(artifact, write)
    manifest = dict(schema=1, architecture='native-weight-quant-packed', scheme=saved['scheme'],
        block=saved['block'], codebook=list(NF4) if saved['scheme']=='nf4' else None,
        config=saved['config'], quantized_layers=saved['quantized_layers'], step=saved['step'],
        teacher_revision=saved['teacher_revision'], teacher_weights_sha256=saved['teacher_weights_sha256'],
        source_contract=saved.get('contract'), tensors=tensors, file='weights.bin', bytes=artifact.stat().st_size,
        sha256=file_sha(artifact), endian='little', quantization_device=str(device), accepted=False, accelerated_arithmetic=False,
        note='BF16 activations; weight-only packed storage. Effective quantized kernels bypass MP re-normalization.')
    atomic_json(header_path, manifest)
    return manifest


def load_packed(output, device='cpu'):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    artifact = output / manifest['file']
    if artifact.stat().st_size != manifest['bytes'] or file_sha(artifact) != manifest['sha256']:
        raise ValueError('Packed weights changed.')
    if manifest['architecture']!='native-weight-quant-packed' or manifest['endian']!='little':
        raise ValueError('Unsupported packed format.')
    model = NativeWeightQuantBase(manifest['config'], manifest['scheme'], manifest['block'],
                                 manifest['quantized_layers']).eval()
    if set(model.state_dict()) != set(manifest['tensors']):
        raise ValueError('Packed tensor inventory mismatch.')
    state = {}
    with artifact.open('rb') as f:
        for name, row in manifest['tensors'].items():
            f.seek(row['offset']); payload = f.read(row['bytes'])
            count = math.prod(row['shape'])
            if row['dtype']=='bfloat16':
                if row['bytes']!=count*2: raise ValueError('Invalid BF16 tensor length.')
                value = torch.from_numpy(np.frombuffer(payload,dtype='<u2').copy()).view(torch.bfloat16)
            else:
                codes = torch.from_numpy(np.frombuffer(payload,dtype=np.uint8).copy())
                if row['dtype']=='nf4': codes = unpack_nibbles(codes)
                elif row['dtype']!='int8': raise ValueError('Invalid packed tensor dtype.')
                f.seek(row['scale_offset'])
                scale = torch.from_numpy(np.frombuffer(f.read(row['scale_count']*4),dtype='<f4').copy())
                if codes.numel()!=scale.numel()*manifest['block']:raise ValueError('Incomplete quantized blocks.')
                values = torch.tensor(manifest['codebook'])[codes.long()] if row['dtype']=='nf4' else (codes.float()-128)/127
                value = (values.reshape(-1,manifest['block'])*scale[:,None]).reshape(-1)[:count].bfloat16()
            state[name] = value.reshape(row['shape'])
    model.load_state_dict(state)
    model._packed = True
    return model.to(device).requires_grad_(False)

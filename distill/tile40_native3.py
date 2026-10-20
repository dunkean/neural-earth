"""Original native MP U-Net:3-bit convolutions,4-bit projections,BF16 inputs.

Only stored weights are compressed. Native global MP normalization,attention,
two-stage interface and BF16 arithmetic remain in the original architecture.
QAT learns FP16 block scales and uses straight-through weight gradients. The
climate/noise input projections retain BF16 precision. No speed claim is made.
"""
from contextlib import contextmanager
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from distill.common import REVISION, atomic_json, atomic_write, external_path
from distill.native_weight_quant import file_sha
from distill.tile40_teacher_quant import pack3, unpack3

ARCHITECTURE = 'native-mp-gaussian3-4-v1'
TABLES = {
    3: (-2.1519457,-1.3439093,-.7560053,-.2450942,.2450942,.7560053,1.3439093,2.1519457),
    4: (-2.7325896,-2.0690172,-1.6180464,-1.2562312,-.9423403,-.6567591,-.3880494,-.1283950,
         .1283950,.3880494,.6567591,.9423403,1.2562312,1.6180464,2.0690172,2.7325896),
}


def codes_for(weight, scale, bits, block):
    flat = weight.detach().float().reshape(-1)
    padded = F.pad(flat,(0,(-flat.numel())%block)).reshape(-1,block)
    if bits not in TABLES or scale.numel()!=len(padded):
        raise ValueError('Supported codebook and positive complete block scales required.')
    centers = padded.new_tensor(TABLES[bits])
    boundaries = (centers[:-1]+centers[1:])*.5
    codes = torch.bucketize((padded/scale.float()[:,None]).contiguous(),boundaries)
    actual = (centers[codes]*scale.float()[:,None]).reshape(-1)[:flat.numel()].reshape(weight.shape)
    return codes.to(torch.uint8).reshape(-1),actual


@torch.no_grad()
def initial_scale(weight,bits,block):
    flat = weight.detach().float().reshape(-1)
    values = F.pad(flat,(0,(-flat.numel())%block)).reshape(-1,block)
    scale = values.square().mean(1).sqrt().clamp_min(1e-12)
    table = values.new_tensor(TABLES[bits])
    boundaries = (table[:-1]+table[1:])*.5
    for _ in range(4):
        code = torch.bucketize((values/scale[:,None]).contiguous(),boundaries)
        selected = table[code]
        scale = (values*selected).sum(1)/selected.square().sum(1).clamp_min(1e-12)
    # Both the forward and artifact use the same representable scales.
    return scale.clamp_min(1e-7).half()


def fake_gaussian(weight,scale,bits,block):
    quantized = codes_for(weight,scale,bits,block)[1]
    return quantized+(weight-weight.detach())


def represented_scale(value):
    value = value.clamp_min(1e-7)
    return value+(value.half().float()-value).detach()


class Native3Base(nn.Module):
    def __init__(self,net,block=256):
        super().__init__()
        if block<16 or block%8:
            raise ValueError('A multiple-of8 block size >=16 is required.')
        self.net,self.block = net,block
        self.config = {k:v for k,v in dict(net.config).items() if not k.startswith('_')}
        first = 'enc.'+next(iter(net.enc))+'.weight'
        sensitive = {first,'out_conv.weight','noise_linear.weight','conditional_layers.0.weight'}
        self.quant_specs = {}
        for name,value in net.named_parameters():
            if value.ndim not in (2,4) or name in sensitive:
                continue
            bits = 4 if value.ndim==2 else 3
            key = 'scale_'+str(len(self.quant_specs))
            self.register_parameter(key,nn.Parameter(initial_scale(value,bits,block).float()))
            self.quant_specs[name] = dict(bits=bits,scale_name=key)
        self._cache = None
        self.quantized = True

    @contextmanager
    def cached_weights(self):
        if self._cache is not None or torch.is_grad_enabled():
            raise ValueError('Quantized weight cache requires a single no-grad scope.')
        self._cache = {}
        try:
            yield
        finally:
            self._cache = None

    def forward(self,inputs,noise_labels,conditional_inputs,**kwargs):
        self.net.eval()
        state = {}
        for name,value in list(self.net.named_parameters())+list(self.net.named_buffers()):
            if name in self.quant_specs and self.quantized:
                spec = self.quant_specs[name]
                if self._cache is not None and name in self._cache:
                    state[name] = self._cache[name]
                    continue
                scale = represented_scale(getattr(self,spec['scale_name']))
                value = fake_gaussian(value,scale,spec['bits'],self.block)
            value = value.bfloat16()
            state[name] = value
            if self._cache is not None:
                self._cache[name] = value
        with torch.autocast(device_type=inputs.device.type,enabled=False):
            return torch.func.functional_call(self.net,state,(inputs.bfloat16(),),
                dict(noise_labels=noise_labels.bfloat16(),
                     conditional_inputs=[v.bfloat16() for v in conditional_inputs],**kwargs)).float()


def _pack_codes(codes,bits):
    codes = codes.cpu().numpy().astype(np.uint8)
    if bits==3:
        return pack3(codes)
    return (codes[::2]|(codes[1::2]<<4)).tobytes()


@torch.no_grad()
def export_packed(model,output,*,step=0,selection='raw',state=None,contract=None,limit=100000000,header_reserve=131072):
    output = external_path(output)
    if output.exists():
        raise ValueError('Preserve packed artifacts;use a fresh directory.')
    state = model.net.state_dict() if state is None else state
    inventory = {}
    estimate = 0
    for name,value in state.items():
        if name in model.quant_specs:
            spec = model.quant_specs[name]
            blocks = math.ceil(value.numel()/model.block)
            estimate += blocks*(model.block*spec['bits']//8+2)
        else:
            estimate += value.numel()*2
    if header_reserve<0:
        raise ValueError('Header reserve cannot be negative.')
    if estimate+header_reserve>=limit:
        raise ValueError(f'Packed base plus header reserve exceeds budget:{estimate}')
    output.mkdir(parents=True)
    offset = 0
    def write(handle):
        nonlocal offset
        for name,value in state.items():
            row = dict(shape=list(value.shape),offset=offset)
            if name in model.quant_specs:
                spec = model.quant_specs[name]
                scale = getattr(model,spec['scale_name']).clamp_min(1e-7).half()
                # EMA may be on another device;perform codes on the scale device.
                codes,_ = codes_for(value.to(scale.device),scale,spec['bits'],model.block)
                payload = _pack_codes(codes,spec['bits'])
                scales = scale.cpu().numpy().astype('<f2').tobytes()
                row.update(dtype='gaussian'+str(spec['bits']),bytes=len(payload),
                    scale_offset=offset+len(payload),scale_count=scale.numel(),scale_bytes=len(scales))
                handle.write(payload);handle.write(scales)
                offset += len(payload)+len(scales)
            else:
                payload = value.detach().bfloat16().contiguous().cpu().view(torch.uint16).numpy().astype('<u2').tobytes()
                row.update(dtype='bfloat16',bytes=len(payload))
                handle.write(payload);offset += len(payload)
            inventory[name] = row
    atomic_write(output/'weights.bin',write)
    manifest = dict(schema=1,architecture=ARCHITECTURE,config=model.config,block=model.block,
        codebooks={str(k):list(v) for k,v in TABLES.items()},tensors=inventory,
        step=step,selection=selection,teacher_revision=REVISION,contract=contract,
        file='weights.bin',bytes=(output/'weights.bin').stat().st_size,sha256=file_sha(output/'weights.bin'),
        storage='Raw MP weights;retain original MP normalization;BF16 activations;FP16 learned block scales.',
        quality_accepted=False,size_scope='Base only;original coarse/decoder are outside this budget.')
    atomic_json(output/'manifest.json',manifest)
    actual = sum(p.stat().st_size for p in output.iterdir() if p.is_file())
    if actual>=limit:
        raise ValueError(f'Packed base exceeds strict limit:{actual}')
    return manifest|dict(total_artifact_bytes=actual)


def read_packed(output):
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text())
    artifact = output/manifest['file']
    if manifest['architecture']!=ARCHITECTURE or file_sha(artifact)!=manifest['sha256'] or artifact.stat().st_size!=manifest['bytes']:
        raise ValueError('Packed architecture or bytes changed.')
    state = {}
    with artifact.open('rb') as handle:
        for name,row in manifest['tensors'].items():
            handle.seek(row['offset']);payload=handle.read(row['bytes'])
            count = math.prod(row['shape'])
            if row['dtype']=='bfloat16':
                if len(payload)!=count*2:
                    raise ValueError('Incomplete BF16 tensor.')
                value = torch.from_numpy(np.frombuffer(payload,dtype='<u2').copy()).view(torch.bfloat16)
            else:
                bits = int(row['dtype'].removeprefix('gaussian'))
                packed = np.frombuffer(payload,dtype=np.uint8)
                codes = unpack3(payload) if bits==3 else np.stack((packed&15,packed>>4),axis=1).reshape(-1)
                handle.seek(row['scale_offset'])
                scales = torch.from_numpy(np.frombuffer(handle.read(row['scale_bytes']),dtype='<f2').copy()).float()
                if len(codes)!=len(scales)*manifest['block']:
                    raise ValueError('Incomplete quantized blocks.')
                values = torch.tensor(manifest['codebooks'][str(bits)])[torch.from_numpy(codes.astype(np.int64))]
                value = (values.reshape(-1,manifest['block'])*scales[:,None]).reshape(-1)[:count].bfloat16()
            state[name] = value.reshape(row['shape'])
    return state,manifest


def load_packed(output,device='cpu'):
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    state,manifest = read_packed(output)
    model = EDMUnet2D.from_config(manifest['config']).eval()
    model.load_state_dict(state,strict=True)
    return model.to(device=device,dtype=torch.bfloat16).requires_grad_(False),manifest

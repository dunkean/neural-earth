"""Small continuous corrections of native conditioning projections under100MB.

The source3/4bit base stays immutable. FP16 low-rank residuals are added to raw
MP projection weights before original normalization and BF16 arithmetic. The
runtime loader merges them into the unchanged original U-Net architecture.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
import torch
from torch import nn

from distill.common import atomic_json,atomic_write,external_path
from distill.native_weight_quant import file_sha
from distill.tile40_native3 import ARCHITECTURE as BASE_ARCHITECTURE
from distill.tile40_teacher_quant import unpack3

ARCHITECTURE='native-mp-gaussian34-lora-v1'


def read_float_base(source):
    source=Path(source)
    manifest=json.loads((source/'manifest.json').read_text())
    artifact=source/manifest['file']
    if manifest['architecture']!=BASE_ARCHITECTURE or file_sha(artifact)!=manifest['sha256']:
        raise ValueError('Immutable source base changed.')
    state={}
    with artifact.open('rb') as handle:
        for name,row in manifest['tensors'].items():
            handle.seek(row['offset']);payload=handle.read(row['bytes'])
            count=math.prod(row['shape'])
            if row['dtype']=='bfloat16':
                value=torch.from_numpy(np.frombuffer(payload,dtype='<u2').copy()).view(torch.bfloat16).float()
            else:
                bits=int(row['dtype'].removeprefix('gaussian'))
                packed=np.frombuffer(payload,dtype=np.uint8)
                codes=unpack3(payload) if bits==3 else np.stack((packed&15,packed>>4),axis=1).reshape(-1)
                handle.seek(row['scale_offset'])
                scale=torch.from_numpy(np.frombuffer(handle.read(row['scale_bytes']),dtype='<f2').copy()).float()
                values=torch.tensor(manifest['codebooks'][str(bits)])[torch.from_numpy(codes.astype(np.int64))]
                value=(values.reshape(-1,manifest['block'])*scale[:,None]).reshape(-1)[:count]
            state[name]=value.reshape(row['shape'])
    return state,manifest


def fp16_value(value):
    return value+(value.half().float()-value).detach()


class ConditioningResidual(nn.Module):
    def __init__(self,net,source_manifest,rank=8):
        super().__init__()
        if rank<1:raise ValueError('Positive low-rank capacity required.')
        self.net=net.eval().requires_grad_(False)
        self.rank=rank
        self.adapters=nn.ParameterDict()
        self.layers={}
        state=dict(net.named_parameters())
        for name,row in source_manifest['tensors'].items():
            if row['dtype']!='gaussian4' or name=='logvar_linear.weight':continue
            value=state[name];out_channels,in_channels=value.shape
            r=min(rank,out_channels,in_channels)
            key='layer'+str(len(self.layers))
            self.adapters[key+'_a']=nn.Parameter(torch.randn(r,in_channels,device=value.device)*.02)
            self.adapters[key+'_b']=nn.Parameter(torch.zeros(out_channels,r,device=value.device))
            self.layers[name]=key
        self._static={name:value.detach().bfloat16() for name,value in
            list(net.named_parameters())+list(net.named_buffers())}

    def forward(self,inputs,noise_labels,conditional_inputs):
        state=dict(self._static)
        for name,key in self.layers.items():
            a,b=(fp16_value(self.adapters[key+suffix]) for suffix in ('_a','_b'))
            raw=self.net.get_parameter(name)
            state[name]=(raw+b@a).bfloat16()
        with torch.autocast(device_type=inputs.device.type,enabled=False):
            return torch.func.functional_call(self.net,state,(inputs.bfloat16(),),
                dict(noise_labels=noise_labels.bfloat16(),conditional_inputs=[x.bfloat16() for x in conditional_inputs])).float()


def from_base(source,device='cpu',rank=8):
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    state,manifest=read_float_base(source)
    net=EDMUnet2D.from_config(manifest['config'])
    net.load_state_dict(state,strict=True)
    return ConditioningResidual(net.to(device),manifest,rank),manifest


@torch.no_grad()
def export_residual(model,source,output,*,step=0,selection='raw',adapters=None,contract=None,limit=100000000):
    source=Path(source);output=external_path(output)
    if output.exists():raise ValueError('Preserve low-rank artifacts;use a fresh directory.')
    source_manifest=json.loads((source/'manifest.json').read_text())
    adapters=model.adapters.state_dict() if adapters is None else adapters
    estimate=sum(p.stat().st_size for p in source.iterdir() if p.is_file())+sum(v.numel()*2 for v in adapters.values())+32768
    if estimate>=limit:raise ValueError(f'Base plus residuals exceed budget:{estimate}')
    (output/'base').mkdir(parents=True)
    os.link(source/source_manifest['file'],output/'base'/source_manifest['file'])
    shutil.copyfile(source/'manifest.json',output/'base/manifest.json')
    rows={};offset=0
    def write(handle):
        nonlocal offset
        for name,value in adapters.items():
            payload=value.detach().cpu().numpy().astype('<f2').tobytes()
            rows[name]=dict(shape=list(value.shape),offset=offset,bytes=len(payload))
            handle.write(payload);offset+=len(payload)
    atomic_write(output/'adapters.bin',write)
    adapter_sha=file_sha(output/'adapters.bin')
    base_manifest_sha=file_sha(output/'base/manifest.json')
    layout=json.dumps(dict(layers=model.layers,adapters=rows),sort_keys=True)
    fingerprint=hashlib.sha256((source_manifest['sha256']+base_manifest_sha+adapter_sha+layout).encode()).hexdigest()
    manifest=dict(schema=1,architecture=ARCHITECTURE,base_directory='base',
        base_sha256=source_manifest['sha256'],base_manifest_sha256=base_manifest_sha,source_base=str(source.resolve()),
        adapter_file='adapters.bin',adapter_sha256=adapter_sha,adapters=rows,layers=model.layers,
        rank=model.rank,sha256=fingerprint,step=step,selection=selection,contract=contract,
        source_step=source_manifest['step'],teacher_revision=source_manifest['teacher_revision'],
        storage='Add FP16 residual B@A to raw dequantized FP32 projection;then cast BF16 and apply original MP normalization.',
        size_scope='Entire base including source weights,adapters and both manifests;decoder excluded.',quality_accepted=False)
    atomic_json(output/'manifest.json',manifest)
    actual=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    if actual>=limit:raise ValueError(f'Complete residual base exceeds strict cap:{actual}')
    return manifest|dict(total_artifact_bytes=actual)


def load_residual(output,device='cpu'):
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    output=Path(output);manifest=json.loads((output/'manifest.json').read_text())
    if manifest['architecture']!=ARCHITECTURE:raise ValueError('Unexpected residual architecture.')
    state,base=read_float_base(output/manifest['base_directory'])
    artifact=output/manifest['adapter_file']
    adapter_sha=file_sha(artifact)
    base_manifest_sha=file_sha(output/manifest['base_directory']/'manifest.json')
    layout=json.dumps(dict(layers=manifest['layers'],adapters=manifest['adapters']),sort_keys=True)
    fingerprint=hashlib.sha256((base['sha256']+base_manifest_sha+adapter_sha+layout).encode()).hexdigest()
    if adapter_sha!=manifest['adapter_sha256'] or base_manifest_sha!=manifest['base_manifest_sha256'] or fingerprint!=manifest['sha256']:
        raise ValueError('Residual bytes or metadata changed.')
    values={}
    with artifact.open('rb') as handle:
        for name,row in manifest['adapters'].items():
            handle.seek(row['offset'])
            array=np.frombuffer(handle.read(row['bytes']),dtype='<f2').copy().reshape(row['shape'])
            values[name]=torch.from_numpy(array).to(device=device,dtype=torch.float32)
    for name,key in manifest['layers'].items():
        state[name]=state[name].to(device)+values[key+'_b']@values[key+'_a']
    net=EDMUnet2D.from_config(base['config'])
    net.load_state_dict({name:value.bfloat16() for name,value in state.items()},strict=True)
    return net.to(device=device,dtype=torch.bfloat16).eval().requires_grad_(False),manifest

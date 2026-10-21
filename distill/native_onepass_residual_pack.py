"""NF4 one-pass body plus its200-byte FP32 learned noise head.

All model payload bytes count toward the100MB target. Quantized internal kernels
are baked; nonquantized MP and anchored output kernels are cached for inference.
"""
import json
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from distill.common import atomic_json, atomic_write, external_path
from distill.native_scalar_precision import ScalarPrecisionBase
from distill.native_scalar_pack import export_scalar_packed, runtime_scalar_packed
from distill.native_onepass_residual import ARCHITECTURE as FLOAT_ARCHITECTURE
from distill.native_fusion import window_outputs
from distill.native_phase_loss import PHASE_LABELS
from distill.native_progressive_width import anchored_scale
from distill.native_progressive_runtime import frozen_output
from distill.native_weight_quant import file_sha
from terrain_diffusion.models.mp_layers import normalize

ARCHITECTURE='native-two-noise-onepass-residual-nf4'


class ResidualQuantOnePass(ScalarPrecisionBase):
    def __init__(self, config, gain_anchor, layers=None):
        if config['in_channels']!=10 or config['out_channels']!=5:
            raise ValueError('Native two-noise interface required.')
        super().__init__(config,'nf4',512,layers)
        self.set_precision('anchored',gain_anchor)
        self.residual=nn.Conv2d(10,5,1,bias=False,dtype=torch.float32)
        nn.init.zeros_(self.residual.weight)
        for name,parameter in self.net.named_parameters():
            if parameter.ndim==0 and name!='out_gain':parameter.requires_grad_(False)

    def windows(self,noises,condition):
        label=torch.full((len(noises),),PHASE_LABELS[0],device=noises.device,dtype=torch.bfloat16)
        first=(torch.sin(label[:,None,None,None])*noises[:,:5].bfloat16()).float()
        velocity=self(torch.cat((first,noises[:,5:].float()),dim=1),label,[condition])
        with torch.autocast(device_type=noises.device.type,enabled=False):
            return window_outputs(first,velocity,label)+F.conv2d(noises.float(),self.residual.weight.float())


def quantized_source(saved):
    if saved.get('architecture') not in (FLOAT_ARCHITECTURE,ARCHITECTURE):
        raise ValueError('Explicit residual one-pass source required.')
    model=ResidualQuantOnePass(saved['config'],saved['gain_anchor'],saved.get('quantized_layers'))
    model.load_state_dict(saved['ema'])
    return model


@torch.no_grad()
def export_residual_packed(saved,output,device='cpu'):
    output=external_path(output)
    if output.exists():raise ValueError('Preserve prior artifacts.')
    model=quantized_source(saved)
    body_source=dict(saved,architecture='native-weight-quant-scalar',scalar_precision='anchored-fp32-v1',
        scheme='nf4',block=512,quantized_layers=model.quantized_layers,
        ema={name:value for name,value in saved['ema'].items() if name.startswith('net.')})
    manifest=export_scalar_packed(body_source,output,device=device)
    values=saved['ema']['residual.weight'].detach().cpu().float()
    if values.shape!=(5,10,1,1) or not torch.isfinite(values).all():
        raise ValueError('Finite5x10 FP32 head required.')
    atomic_write(output/'residual.bin',lambda handle:handle.write(values.numpy().astype('<f4').tobytes()))
    total=manifest['bytes']+(output/'residual.bin').stat().st_size
    if total>=100000000:raise ValueError('Composite model misses the100MB payload budget.')
    manifest['onepass_wrapper']=dict(architecture=ARCHITECTURE,solver_evaluations=1,teacher_solver_evaluations=2,
        noise_offsets=[5819,5820],window_size=64,stride=32,download_weight_bytes=total,
        residual=dict(file='residual.bin',dtype='float32',shape=[5,10,1,1],bytes=200,sha256=file_sha(output/'residual.bin')),
        accepted=False,quality_scope='PTQ unless source checkpoint explicitly completed QAT.')
    atomic_json(output/'manifest.json',manifest)
    return manifest


class ResidualQuantRuntime(nn.Module):
    def __init__(self, saved, device='cpu'):
        super().__init__()
        self.model = quantized_source(saved).to(device).eval().requires_grad_(False)

    def cached_weights(self):
        return self.model.cached_weights()

    def forward(self, noises, condition):
        return self.model.windows(noises, condition)


class ResidualPackedRuntime(nn.Module):
    def __init__(self,output,device='cpu'):
        super().__init__()
        output=Path(output);manifest=json.loads((output/'manifest.json').read_text())
        wrapper=manifest.get('onepass_wrapper',{})
        if (wrapper.get('architecture')!=ARCHITECTURE or wrapper.get('solver_evaluations')!=1 or
                wrapper.get('teacher_solver_evaluations')!=2):
            raise ValueError('Unsupported composite one-pass manifest.')
        head=wrapper['residual']
        artifact=output/'residual.bin'
        if (head['file']!='residual.bin' or head['dtype']!='float32' or head['shape']!=[5,10,1,1] or
                head['bytes']!=200 or artifact.stat().st_size!=200 or file_sha(artifact)!=head['sha256'] or
                wrapper['download_weight_bytes']!=manifest['bytes']+200):
            raise ValueError('Residual payload or composite byte count changed.')
        self.net=runtime_scalar_packed(output,device)
        self.config=dict(manifest['config'])
        from terrain_inference import prepare_models
        forwards={name:self.net.get_submodule(name).forward for name in manifest['quantized_layers']}
        out=self.net.out_conv;gain=self.net.out_gain.detach().float()
        kernel=normalize(out.weight.float())*anchored_scale(gain,manifest['gain_anchor'],out.weight[0].numel())
        prepare_models(SimpleNamespace(coarse_model=self.net,base_model=self.net,decoder_model=self.net))
        for name,forward in forwards.items():self.net.get_submodule(name).forward=forward
        out._runtime_out_weight=kernel.bfloat16();out.forward=MethodType(frozen_output,out)
        self.register_buffer('label',torch.tensor(PHASE_LABELS[0],device=device,dtype=torch.bfloat16))
        self.residual=nn.Conv2d(10,5,1,bias=False,dtype=torch.float32,device=device)
        with torch.no_grad():
            self.residual.weight.copy_(torch.from_numpy(np.frombuffer(artifact.read_bytes(),dtype='<f4').copy()).reshape(5,10,1,1))
        self.eval().requires_grad_(False)

    def forward(self,noises,condition):
        first=torch.sin(self.label)*noises[:,:5].bfloat16()
        inputs=torch.cat((first,noises[:,5:].bfloat16()),dim=1)
        velocity=self.net(inputs,noise_labels=self.label.expand(len(inputs)),conditional_inputs=[condition.bfloat16()])
        with torch.autocast(device_type=noises.device.type,enabled=False):
            return window_outputs(first,velocity,self.label.expand(len(inputs)))+F.conv2d(noises.float(),self.residual.weight.float())

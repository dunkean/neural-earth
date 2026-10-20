"""Original decoder control for generated latents;not a browser-sized product."""
from types import SimpleNamespace
import torch
from torch import nn
from distill.native_decoded import OriginalDecoder


class OriginalDecoderControl(nn.Module):
    def __init__(self,device='cuda',teacher=None):
        super().__init__()
        self.teacher=teacher if teacher is not None else OriginalDecoder(device)
        self.config=SimpleNamespace(stage='decoder')

    def forward(self,inputs):
        if inputs.shape[1:]!=(16,512,512):
            raise ValueError('Canonical16-channel decoder feature window required.')
        fields=inputs[:,1:5,::8,::8]
        latents=torch.cat((fields,fields.new_zeros(len(fields),1,64,64)),dim=1)
        return self.teacher(latents,inputs[:,:1])

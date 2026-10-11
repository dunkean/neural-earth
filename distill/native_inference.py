"""Load native base EMA weights into the unchanged production solver interface."""
from __future__ import annotations

import torch

from distill.common import REVISION


def load_native(source,device='cpu'):
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    saved=torch.load(source,map_location='cpu',weights_only=True)
    config=saved['config']
    if (config.get('in_channels') != 5 or config.get('out_channels') != 5 or
            config.get('conditional_inputs') != [['tensor',58,1.0]]):
        raise ValueError('Native base requires the teacher five-channel /58-value interface.')
    revision=saved.get('teacher_revision') or saved.get('dataset_manifest',{}).get('revision')
    if revision != REVISION:
        raise ValueError('Native base teacher revision is missing or mismatched.')
    state=saved.get('ema',saved['model'])
    if not state or any(not name.startswith('net.') for name in state):
        raise ValueError('Native checkpoint weights must belong to the native net.')
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Non-finite native base weights.')
    model=EDMUnet2D.from_config(config)
    model.load_state_dict({name[4:]:value for name,value in state.items()},strict=True)
    dtype=torch.bfloat16 if torch.device(device).type=='cuda' else torch.float32
    return model.to(device=device,dtype=dtype).eval().requires_grad_(False),saved

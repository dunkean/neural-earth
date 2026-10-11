"""Native64-latent regions aligned to both base and original decoder grids.

The source scene contains all25 initialization and9 refinement dependencies.
Height/shape supervision uses the original frozen512/384 decoder and its
unchanged one-step noise/velocity expression. Full pipeline checks still follow.
"""
from __future__ import annotations

import json
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from distill.native_fusion import (Region,blend,forward_batches,overlap_loss,
    refinement_inputs,window_outputs)
from distill.native_base import teacher_source
from distill.decoded_loss import reconstructed_height,relative_height_gradient_loss


class DecodedRegion(Region):
    def __init__(self,path,device='cpu'):
        from terrain_diffusion.inference.world_pipeline import gaussian_noise_patch
        with np.load(path,allow_pickle=False) as sample:
            arrays={key:sample[key].copy() for key in sample.files if key!='metadata'}
            self.metadata=json.loads(str(sample['metadata']))
        y,x=self.metadata['y'],self.metadata['x']
        if y%32 or x%32 or self.metadata['native_size']!=128:
            raise ValueError('Decoded native regions require a128 scene on the base32 grid.')
        self.core=tuple(((value+95)//96)*96 for value in (y,x))
        oy,ox=self.core[0]-y,self.core[1]-x
        if oy not in (0,32,64) or ox not in (0,32,64):
            raise ValueError('No complete64-latent decoder window in this scene.')
        self.dtype=torch.bfloat16 if torch.device(device).type=='cuda' else torch.float32
        self.init_origin=(self.core[0]-64,self.core[1]-64)
        self.init_origins=[(self.core[0]+dy,self.core[1]+dx)
                           for dy in (-64,-32,0,32,64) for dx in (-64,-32,0,32,64)]
        self.ref_origins=[(self.core[0]+dy,self.core[1]+dx)
                          for dy in (-32,0,32) for dx in (-32,0,32)]
        labels=np.unique(arrays['labels'])
        if len(labels)!=2:
            raise ValueError('Complete native two-stage calls required.')
        self.stages=[]
        for label,origins in ((labels.max(),self.init_origins),(labels.min(),self.ref_origins)):
            lookup={}
            for index in np.flatnonzero(arrays['labels']==label):
                context=arrays['contexts'][index]
                key=(int(context[1])*32,int(context[2])*32)
                if key in lookup:
                    raise ValueError('Duplicate native phase/window.')
                lookup[key]=index
            try:
                selected=[lookup[origin] for origin in origins]
            except KeyError as exc:
                raise ValueError(f'Incomplete decoded native dependencies: {exc}') from exc
            self.stages.append({key:torch.from_numpy(arrays[key][selected]).to(device)
                                for key in ('inputs','labels','condition','target')})
        self.noise=torch.from_numpy(np.stack([gaussian_noise_patch(self.metadata['seed']+5820,
            yy,xx,64,64,channels=5,tile_h=64,tile_w=64) for yy,xx in self.ref_origins])).to(device)
        initial=torch.from_numpy(arrays['initial_weighted']).to(device)
        origin=arrays['initial_origin']
        self.teacher_initial=torch.stack([initial[:,yy-int(origin[0]):yy-int(origin[0])+64,
            xx-int(origin[1]):xx-int(origin[1])+64] for yy,xx in self.ref_origins])
        self.target=torch.from_numpy(arrays['final_latents'][:,oy:oy+64,ox:ox+64]).to(device)
        self.mask=torch.from_numpy(arrays['final_mask'][:,oy:oy+64,ox:ox+64]).to(device)
        if self.target.shape!=(5,64,64) or not self.mask.all():
            raise ValueError('Complete64-square teacher target and mask required.')
        self.decoder_noise=torch.from_numpy(gaussian_noise_patch(self.metadata['seed']+5819,
            self.core[0]*8,self.core[1]*8,512,512,channels=1,tile_h=512,tile_w=512))[None].to(device)

    def core_crop(self,refined,weight):
        origin=(self.core[0]-32,self.core[1]-32)
        canvas=blend(refined,self.ref_origins,origin,128,weight)
        weighted=canvas[:,32:96,32:96]
        if torch.any(weighted[-1:]<=0):
            raise ValueError('Decoded native core has incomplete coverage.')
        return weighted[:-1]/weighted[-1:]

    @torch.no_grad()
    def refresh_noise(self,teacher,weight,seed):
        # Larger canvas than Region32. Geometry remains fixed for a given scene.
        from terrain_diffusion.inference.world_pipeline import gaussian_noise_patch
        first,second=self.stages
        device=first['inputs'].device
        noise0=torch.from_numpy(np.stack([gaussian_noise_patch(seed+5819,y,x,64,64,
            channels=5,tile_h=64,tile_w=64) for y,x in self.init_origins])).to(device)
        zeros=noise0.new_zeros((len(noise0),6,64,64));zeros[:,-1:]=1.
        first['inputs']=refinement_inputs(zeros,noise0,first['labels'],self.dtype)
        first['target']=forward_batches(teacher,first['inputs'].to(self.dtype),first['labels'].to(self.dtype),first['condition'].to(self.dtype)).float()
        initial=blend(window_outputs(first['inputs'],first['target'],first['labels'],self.dtype),
                      self.init_origins,self.init_origin,192,weight)
        self.teacher_initial=self.ref_crops(initial)
        self.noise=torch.from_numpy(np.stack([gaussian_noise_patch(seed+5820,y,x,64,64,
            channels=5,tile_h=64,tile_w=64) for y,x in self.ref_origins])).to(device)
        second['inputs']=refinement_inputs(self.teacher_initial,self.noise,second['labels'],self.dtype)
        second['target']=forward_batches(teacher,second['inputs'].to(self.dtype),second['labels'].to(self.dtype),second['condition'].to(self.dtype)).float()
        self.target=self.core_crop(window_outputs(second['inputs'],second['target'],second['labels'],self.dtype),weight)
        self.decoder_noise=torch.from_numpy(gaussian_noise_patch(seed+5819,self.core[0]*8,self.core[1]*8,
            512,512,channels=1,tile_h=512,tile_w=512))[None].to(device)
        self.noise_seed=int(seed)


def decoded_region_forward(model,region,weight,checkpointed=False):
    first,second=region.stages
    velocity0=forward_batches(model,first['inputs'],first['labels'],first['condition'],checkpointed=checkpointed)
    initial=blend(window_outputs(first['inputs'],velocity0,first['labels'],region.dtype),
                  region.init_origins,region.init_origin,192,weight)
    inputs1=refinement_inputs(region.ref_crops(initial),region.noise,second['labels'],region.dtype)
    velocity1=forward_batches(model,inputs1,second['labels'],second['condition'],checkpointed=checkpointed)
    output1=window_outputs(inputs1,velocity1,second['labels'],region.dtype)
    return dict(core=region.core_crop(output1,weight),inputs1=inputs1,velocity0=velocity0,velocity1=velocity1,output1=output1)


class OriginalDecoder(nn.Module):
    def __init__(self,device='cpu',model=None):
        super().__init__()
        dtype=torch.bfloat16 if torch.device(device).type=='cuda' else torch.float32
        if model is None:
            from terrain_diffusion.models.edm_unet import EDMUnet2D
            model=EDMUnet2D.from_pretrained(str(teacher_source().parent/'decoder_model'))
            if model.config.in_channels!=5 or model.config.out_channels!=1 or model.config.conditional_inputs:
                raise ValueError('Pinned original decoder interface changed.')
            from types import SimpleNamespace
            from terrain_inference import prepare_models
            model=model.to(device=device,dtype=dtype).eval().requires_grad_(False)
            prepare_models(SimpleNamespace(coarse_model=model,base_model=model,decoder_model=model))
        self.model=model.to(device=device,dtype=dtype).eval().requires_grad_(False)
        self.register_buffer('label',torch.atan(torch.tensor(80./.5)).to(device=device,dtype=dtype))

    def forward(self,latents,gaussian):
        if latents.shape[1:]!=(5,64,64) or gaussian.shape[1:]!=(1,512,512):
            raise ValueError('Original decoder requires complete64/512 aligned windows.')
        with torch.autocast(device_type=latents.device.type,enabled=False):
            dtype=self.label.dtype
            fields=F.interpolate(latents[:,:4].to(dtype),size=(512,512),mode='nearest')
            t=self.label.view(1,1,1,1)
            x_t=torch.cos(t)*torch.zeros_like(gaussian,dtype=dtype)+torch.sin(t)*(gaussian.to(dtype)*.5)
            inputs=torch.cat((x_t/.5,fields),dim=1)
            velocity=self.model(inputs,noise_labels=self.label.expand(len(inputs)),conditional_inputs=[])
            return (torch.cos(t)*x_t-torch.sin(t)*.5*(-velocity)).float()/.5


def blur(values,sigma):
    radius=int(math.ceil(3*sigma))
    axis=torch.arange(-radius,radius+1,device=values.device,dtype=torch.float32)
    kernel=torch.exp(-.5*(axis/sigma).square());kernel=kernel/kernel.sum()
    values=F.conv2d(F.pad(values,(radius,radius,0,0),mode='reflect'),kernel.view(1,1,1,-1))
    return F.conv2d(F.pad(values,(0,0,radius,radius),mode='reflect'),kernel.view(1,1,-1,1))


def relative_tpi_loss(prediction,target):
    losses=[]
    for sigma in (2.,8.,32.):
        margin=int(math.ceil(3*sigma))
        reference=(target-blur(target,sigma))[...,margin:-margin,margin:-margin]
        actual=(prediction-blur(prediction,sigma))[...,margin:-margin,margin:-margin]
        losses.append((actual-reference).square().mean()/reference.detach().square().mean().clamp_min(1.))
    return torch.stack(losses).mean()


def physical_terms(result,region,decoder,channel_variance):
    with torch.no_grad():
        reference_residual=decoder(region.target[None],region.decoder_noise)
        reference_height=reconstructed_height(reference_residual,region.target[None])
    residual=decoder(result['core'][None],region.decoder_noise)
    height=reconstructed_height(residual,result['core'][None])
    mask=torch.ones_like(height)
    low=result['core'][4:5]*38.6-31.4
    low_target=region.target[4:5]*38.6-31.4
    low=low.sign()*low.square();low_target=low_target.sign()*low_target.square()
    variance=channel_variance.to(height.device).clamp_min(1e-3)[:,None,None]
    return dict(height_mae_m=(height-reference_height).abs().mean(),
        low_height_mae_m=(low-low_target).abs().mean(),
        height_gradient=relative_height_gradient_loss(height,reference_height,mask,floor_m=.02),
        height_tpi=relative_tpi_loss(height,reference_height),
        normalized_latent=((result['core']-region.target).square()/variance).mean())


def decoded_loss(result,region,decoder,teacher,channel_variance):
    terms=physical_terms(result,region,decoder,channel_variance)
    first,second=region.stages
    with torch.no_grad():
        target1=forward_batches(teacher,result['inputs1'].detach().to(region.dtype),
            second['labels'].to(region.dtype),second['condition'].to(region.dtype)).float()
    terms['operator']=(result['velocity0']-first['target']).square().mean()+\
                       (result['velocity1']-target1).square().mean()
    terms['overlap']=overlap_loss(result['output1'],window_outputs(result['inputs1'].detach(),target1,
        second['labels'],region.dtype),region.ref_origins)
    loss=terms['height_mae_m']/100.+.25*terms['low_height_mae_m']/100.+.5*terms['height_gradient']+\
         .25*terms['height_tpi']+.1*terms['normalized_latent']+.02*terms['operator']+.01*terms['overlap']
    return loss,terms

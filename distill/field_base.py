"""Full teacher-width field UNet with spatial FiLM and world-aligned attention.

All teacher parameters keep their original names/shapes. Larger fields change
the function and need training. A64 square with constant conditions and aligned
attention remains a numerical control. Halo64 is a training/evaluation choice,
not a claimed bound on this network's convolutional receptive field.
"""
import math
from types import MethodType

import torch
from torch import nn
from torch.nn import functional as F

from distill.native_base import NativeBase
from terrain_diffusion.models import mp_layers


def condition_at(conditions,height,width,spacing,centres):
    if conditions.ndim!=4 or conditions.shape[1]!=58:
        raise ValueError('Dense conditioning requires Bx58xgridYxgridX.')
    if conditions.shape[-2:]==(1,1):
        return conditions[:,:,0,0]
    grids=[]
    for cy,cx in centres:
        ys=(torch.arange(height,device=conditions.device,dtype=torch.float32)*spacing-cy)/32
        xs=(torch.arange(width,device=conditions.device,dtype=torch.float32)*spacing-cx)/32
        gy,gx=torch.meshgrid(ys,xs,indexing='ij')
        grids.append(torch.stack((2*gx/(conditions.shape[-1]-1)-1,
                                 2*gy/(conditions.shape[-2]-1)-1),dim=-1))
    # Interpolate physical window positions in FP32, then use the teacher dtype.
    values=F.grid_sample(conditions.float(),torch.stack(grids),mode='bilinear',
                        padding_mode='border',align_corners=True).to(conditions.dtype)
    return values.permute(0,2,3,1).reshape(-1,58)


def embedding_at(net,conditions,labels,height,width,spacing,centres):
    values=condition_at(conditions,height,width,spacing,centres)
    if conditions.shape[-2:]==(1,1):
        return net.compute_embeddings(labels,[values])
    repeated=labels[:,None].expand(-1,height*width).reshape(-1)
    return net.compute_embeddings(repeated,[values]).reshape(len(labels),height,width,-1)


def local_attention(block,inputs,origins,spacing):
    window=max(1,64//spacing)
    if spacing>64 or 64%spacing:
        raise ValueError('Attention must use a fixed64-latent world extent.')
    result=[]
    for i,(oy,ox) in enumerate(origins):
        if oy%spacing or ox%spacing:
            raise ValueError('World origin must align with the model downsampling grid.')
        top,left=(oy//spacing)%window,(ox//spacing)%window
        h,w=inputs.shape[-2:]
        bottom,right=(-h-top)%window,(-w-left)%window
        padded=F.pad(inputs[i:i+1],(left,right,top,bottom))
        hp,wp=padded.shape[-2:];ny,nx=hp//window,wp//window
        tiles=padded.reshape(1,padded.shape[1],ny,window,nx,window).permute(0,2,4,1,3,5)
        tiles=tiles.reshape(ny*nx,padded.shape[1],window,window)
        attended=block.attn(tiles)
        merged=attended.reshape(1,ny,nx,padded.shape[1],window,window)
        merged=merged.permute(0,3,1,4,2,5).reshape(1,padded.shape[1],hp,wp)
        result.append(merged[:,:,top:top+h,left:left+w])
    return torch.cat(result)


def block_forward(block,inputs,embedding,origins,spacing):
    x=block.resample(inputs)
    if block.mode=='enc':
        if block.conv_skip is not None:x=block.conv_skip(x)
        x=mp_layers.normalize(x,dim=1)
    if block.conv_type=='mobile':
        y=block.conv_res0[0](block.activation(x));y=block.conv_res0[1](block.activation(y))
    else:y=block.conv_res0(block.activation(x))
    if block.emb_linear is not None:
        c=block.emb_linear(embedding,gain=block.emb_gain)+1
        if embedding.ndim==2:
            c=c/torch.sqrt(torch.mean(c**2,dim=1,keepdim=True)+1e-8)
            c=c[:,:,None,None]
        else:
            c=c/torch.sqrt(torch.mean(c**2,dim=-1,keepdim=True)+1e-8)
            c=c.permute(0,3,1,2)
        y=block.activation(y*c.to(y.dtype))
    else:y=block.activation(y)
    y=block.conv_res1(y)
    if block.mode=='dec' and block.conv_skip is not None:x=block.conv_skip(x)
    if x.shape[-2:]!=y.shape[-2:]:
        raise ValueError('Field pupil uses padded same-size residual convolutions.')
    x=mp_layers.mp_sum([x,y],w=block.res_balance)
    if block.num_heads:
        x=mp_layers.mp_sum([x,local_attention(block,x,origins,spacing)],w=block.attn_balance)
    return torch.clip(x,-block.clip_act,block.clip_act) if block.clip_act is not None else x


def field_forward(net,x,noise_labels,conditions,origins,centres,return_features=False):
    initial_size=x.shape[-2:];embeddings={};features={}
    def emb_for(height,width,spacing):
        key=(height,width,spacing)
        if key not in embeddings:
            embeddings[key]=embedding_at(net,conditions,noise_labels,height,width,spacing,centres)
            if hasattr(net,'field_phase_embedding'):
                from distill.field_phase import add_phase_embedding
                embeddings[key]=add_phase_embedding(net,embeddings[key],origins,height,width,spacing)
        return embeddings[key]
    x=torch.cat((x,torch.ones_like(x[:,:1])),dim=1);skips=[]
    for name,block in net.enc.items():
        if 'conv' in name:
            x=block(x)
            if hasattr(net,'field_phase_stem'):
                from distill.field_phase import phase_features
                phase=phase_features(origins,*initial_size,1,x.device,x.dtype).permute(0,3,1,2)
                x=x+F.conv2d(phase,net.field_phase_stem.to(x.dtype))
        else:
            height=x.shape[-2]//2 if block.resample_mode=='down' else x.shape[-2]
            width=x.shape[-1]//2 if block.resample_mode=='down' else x.shape[-1]
            spacing=initial_size[0]//height
            x=block_forward(block,x,emb_for(height,width,spacing),origins,spacing)
        skips.append(x)
        if return_features:features['enc.'+name]=x
    for name,block in net.dec.items():
        if 'block' in name:x=mp_layers.mp_concat([x,skips.pop()],w=net.concat_balance)
        height=x.shape[-2]*2 if block.resample_mode=='up' else x.shape[-2]
        width=x.shape[-1]*2 if block.resample_mode=='up' else x.shape[-1]
        spacing=initial_size[0]//height
        x=block_forward(block,x,emb_for(height,width,spacing),origins,spacing)
        if return_features:features['dec.'+name]=x
    result=net.out_conv(x,gain=net.out_gain)
    return (result,features) if return_features else result


class FieldBase(NativeBase):
    def __init__(self,config):
        super().__init__(config)
        if config['in_channels']!=5 or config['out_channels']!=5:
            raise ValueError('Keep the original five-channel interface in both phases.')
        self.net.forward=MethodType(field_forward,self.net)

    def forward(self,inputs,noise_labels,conditions,origins,centres,return_features=False):
        if inputs.ndim!=4 or inputs.shape[-2:]!=inputs.shape[-2:][::-1]:
            raise ValueError('Square field inputs are required.')
        if len(origins)!=len(inputs) or len(centres)!=len(inputs):
            raise ValueError('Every field requires explicit world and condition-grid origins.')
        self.net.eval();dtype=torch.bfloat16 if inputs.is_cuda else torch.float32
        state={name:value.to(dtype) for name,value in
               list(self.net.named_parameters())+list(self.net.named_buffers())}
        with torch.autocast(device_type=inputs.device.type,enabled=False):
            result=torch.func.functional_call(self.net,state,(inputs.to(dtype),),dict(
                noise_labels=noise_labels.to(dtype),conditions=conditions.to(dtype),
                origins=origins,centres=centres,return_features=return_features))
            return (result[0].float(),result[1]) if return_features else result.float()

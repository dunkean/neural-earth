"""Compact physical climate, using the model's actual coarse conditioning.

Transport stores sea-level BIO1, raw BIO4, BIO12, BIO15 and lapse rate.
Temperature at the displayed elevation is reconstructed in the shader.
"""
import numpy as np
import torch
from scipy.ndimage import map_coordinates

CLIMATE_SIZE = 33
MODES = ('relief', 'biomes', 'temperature', 'precipitation')


@torch.inference_mode()
def sample_coarse_climate(world, xs, ys, check=None, _local=False):
    from terrain_diffusion.inference.postprocessing import local_baseline_temperature_torch
    stride = 256
    xs,ys=np.asarray(xs),np.asarray(ys)
    width=int(np.ceil(xs.max()/stride)-np.floor(xs.min()/stride))+18
    height=int(np.ceil(ys.max()/stride)-np.floor(ys.min()/stride))+18
    if not _local and (width>64 or height>64):
        # A sparse worldwide output must not materialize the rectangle between
        # its points. Every group retains the original 8-cell climate context.
        output=np.empty((5,len(ys),len(xs)),np.float32)
        xgroups=np.floor(xs/stride/48).astype(np.int64)
        ygroups=np.floor(ys/stride/48).astype(np.int64)
        for gy in np.unique(ygroups):
            yi=np.flatnonzero(ygroups==gy)
            for gx in np.unique(xgroups):
                xi=np.flatnonzero(xgroups==gx)
                if check:
                    check()
                output[:,yi[:,None],xi]=sample_coarse_climate(world,xs[xi],ys[yi],check=check,_local=True)
        return output
    ci, cj = int(np.floor(np.min(ys)/stride)), int(np.floor(np.min(xs)/stride))
    ei, ej = int(np.ceil(np.max(ys)/stride))+1, int(np.ceil(np.max(xs)/stride))+1
    from terrain_window_scheduler import read_rect
    data = read_rect(world,'coarse',ci-8,cj-8,ei+8,ej+8,check=check).float()
    coarse = data[:-1]/data[-1:].clamp_min(1e-8)
    elevation = torch.sign(coarse[0])*coarse[0].clamp_min(0).square()
    baseline, beta = local_baseline_temperature_torch(coarse[2], elevation, win=15, fallback_threshold=.02)
    features = torch.cat([baseline, coarse[3:6,7:-7,7:-7], beta], dim=0).unsqueeze(0)
    # Match WorldPipeline._compute_climate's half-cell and align_corners=False.
    y = torch.as_tensor(ys, device=data.device, dtype=torch.float32)/stride-ci+1
    x = torch.as_tensor(xs, device=data.device, dtype=torch.float32)/stride-cj+1
    yy, xx = torch.meshgrid(y*2/features.shape[-2]-1,x*2/features.shape[-1]-1,indexing='ij')
    grid = torch.stack([xx,yy],dim=-1).unsqueeze(0)
    return torch.nn.functional.grid_sample(features,grid,mode='bilinear',padding_mode='border',align_corners=False)[0].cpu().numpy()


def expand_climate(climate, shape):
    h,w = shape
    yy,xx = np.meshgrid(np.linspace(0,climate.shape[1]-1,h),np.linspace(0,climate.shape[2]-1,w),indexing='ij')
    return np.stack([map_coordinates(plane,[yy,xx],order=1,mode='nearest') for plane in climate])


def colorize(elevation, climate, mode):
    """CPU fallback palette mirrored by terrain_renderer.js (heuristic biomes)."""
    c = expand_climate(climate,elevation.shape)
    temp = c[0]+c[4]*np.maximum(elevation,0)
    rain = np.maximum(c[2],0)
    if mode=='temperature':
        t=np.clip((temp+35)/70,0,1)[...,None]
        rgb=(1-t)*np.array([.18,.38,.88])+t*np.array([.95,.22,.08])
    elif mode=='precipitation':
        t=np.clip(np.log1p(rain)/np.log(4001),0,1)[...,None]
        rgb=(1-t)*np.array([.78,.61,.32])+t*np.array([.08,.36,.72])
    else:
        season=np.maximum(c[1],0)/100  # WorldClim BIO4 is temperature std x100.
        effective_rain=rain/(1+np.maximum(c[3],0)/200)
        dry=np.clip((450+np.maximum(temp,0)*25+season*8-effective_rain)/650,0,1)[...,None]
        wet=np.clip(effective_rain/2500,0,1)[...,None]
        rgb=(1-wet)*np.array([.42,.57,.28])+wet*np.array([.08,.35,.20])
        rgb=(1-dry)*rgb+dry*np.array([.82,.69,.42])
        rock=np.clip((elevation-2300)/2200,0,1)[...,None]
        rgb=(1-rock)*rgb+rock*np.array([.52,.48,.43])
        snow=np.clip((1.5-temp)/7,0,1)[...,None]
        rgb=(1-snow)*rgb+snow*np.array([.93,.97,.99])
    if mode=='biomes':
        ocean=np.array([.10,.30,.50])+np.clip(1+elevation/6000,0,1)[...,None]*np.array([.18,.24,.20])
        ice=np.clip((-2-temp)/10,0,1)[...,None]
        ocean=(1-ice)*ocean+ice*np.array([.82,.93,.98])
        rgb=np.where((elevation<0)[...,None],ocean,rgb)
    return np.clip(rgb,0,1)

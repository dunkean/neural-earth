"""Rasterize the original spherical Orogen triangulation with CUDA tensors."""
import numpy as np
import torch

def rasterize(points, triangles, nearest, xyz, fields, device):
    triangles=np.asarray(triangles,np.int64).reshape(-1,3)
    # Incident triangle table: a nearest vertex's fan contains the query.
    order=np.argsort(triangles.ravel(),kind='stable'); vertices=triangles.ravel()[order]
    counts=np.bincount(vertices,minlength=len(points)); offsets=np.cumsum(counts)-counts
    fan=np.full((len(points),int(counts.max())), -1,np.int64)
    fan[vertices,np.arange(len(vertices))-np.repeat(offsets,counts)]=order//3
    tri=torch.as_tensor(triangles,device=device)
    p=torch.as_tensor(points,dtype=torch.float32,device=device)
    matrices=p[tri].transpose(1,2)
    inverse=torch.linalg.inv(matrices)
    fan=torch.as_tensor(fan,device=device)
    names=list(fields); data=torch.as_tensor(np.stack([fields[k] for k in names],axis=1),device=device)
    categorical={'plates','crust','boundaries','koppen','superPlates'}
    nearest_only=torch.as_tensor([k in categorical or k.startswith('koppen_color_') for k in names],device=device)
    # Fields stay resident and transfer once; per-chunk, per-field copies
    # each synchronized the device.
    projected=torch.empty((len(names),len(xyz)),dtype=data.dtype,device=device)
    fallback=torch.zeros((),dtype=torch.int64,device=device)
    with torch.inference_mode():
        for start in range(0,len(xyz),32768):
            end=min(start+32768,len(xyz));q=torch.as_tensor(xyz[start:end],device=device)
            neighbors=torch.as_tensor(nearest[start:end],device=device); n=neighbors[:,0]; candidates=fan[neighbors].reshape(end-start,-1)
            weights=(inverse[candidates.clamp_min(0)]*q[:,None,None,:]).sum(dim=-1)
            score=weights.min(dim=-1).values.masked_fill_(candidates<0,-torch.inf)
            chosen=score.argmax(dim=1);row=torch.arange(end-start,device=device)
            valid=score[row,chosen]>=-1e-4;fallback+=(~valid).sum()
            w=weights[row,chosen];w=w.clamp_min(0);w=w/w.sum(dim=1,keepdim=True)
            t=tri[candidates[row,chosen].clamp_min(0)]
            result=(data[t]*w[:,:,None]).sum(dim=1)
            nearest_value=data[n]
            result=torch.where(valid[:,None],result,nearest_value)
            projected[:,start:end]=torch.where(nearest_only,nearest_value,result).T
        projected=projected.cpu().numpy()
    torch.cuda.synchronize(device)
    projected.flags.writeable=False
    return dict(zip(names,projected)),int(fallback)

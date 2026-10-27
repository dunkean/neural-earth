"""Sample immutable climate/soil inputs once, before authority compilation/IPC."""
import numpy as np


CLIMATE=('temperature_summer','temperature_winter','precip_summer','precip_winter')


def prepare(layers,shape,grid):
    if '__resource_ts' in layers:
        return {k:v for k,v in layers.items() if k.startswith('__resource_')}
    import terrain_resources as R
    h,w=shape;x0,y0,x1,y1=grid.bounds
    xx=((grid.x-x0)%(x1-x0))/(x1-x0)*w-.5
    yy=np.clip((grid.y-y0)/(y1-y0)*h-.5,0,h-1)
    ix,iy=np.floor(xx).astype(np.int64),np.floor(yy).astype(np.int64)
    cols=np.unique(np.r_[ix%w,(ix+1)%w]);rows=np.unique(np.r_[iy,np.minimum(iy+1,h-1)])
    c0,c1=np.searchsorted(cols,ix%w),np.searchsorted(cols,(ix+1)%w)
    r0,r1=np.searchsorted(rows,iy),np.searchsorted(rows,np.minimum(iy+1,h-1))
    tx,ty=(xx-ix)[None,:],(yy-iy)[:,None]
    index=np.ix_(rows,cols)
    nodes=[];out={}
    for name,short in zip(CLIMATE,('ts','tw','ps','pw')):
        a=np.asarray(layers[name])[index].astype(np.float64)
        a=-45+90*a if short in ('ts','tw') else 1000*np.maximum(a,0)
        nodes.append(a)
        top=a[r0[:,None],c0]*(1-tx)+a[r0[:,None],c1]*tx
        bottom=a[r1[:,None],c0]*(1-tx)+a[r1[:,None],c1]*tx
        out['__resource_'+short]=(top*(1-ty)+bottom*ty).astype(np.float32)
    ts,tw,ps,pw=nodes
    pet=150+55*np.maximum((ts+tw)/2+5,0)
    out['__resource_aridity']=1-R.smoothstep(.12,.55,(ps+pw)/pet)
    for name in ('sandy','calcareous','clayey','ferrallitic','organic','podzolic','mineral'):
        key='pedology_'+name
        out['__resource_'+key]=(R.sample_grid(layers[key],grid.x,grid.y,bounds=grid.bounds)
                               if key in layers else np.full(grid.shape,1/7))
    return out

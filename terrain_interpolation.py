"""Bounded, monotone coarse preview interpolation on the existing read halo."""
import numpy as np


def _hermite(a,b,c,d,t):
    a,b,c,d=(value.astype(np.float64,copy=False) for value in (a,b,c,d))
    left=b-a
    middle=c-b
    right=d-c
    def slope(p,q):
        same=(p>0)&(q>0)|(p<0)&(q<0)
        # Harmonic mean, computed without reciprocal overflow near zero.
        denominator=p+q
        result=np.zeros_like(denominator)
        np.divide(2*p*q,denominator,out=result,where=same)
        return result
    m0=slope(left,middle)
    m1=slope(middle,right)
    t2=t*t
    t3=t2*t
    value=(2*t3-3*t2+1)*b+(t3-2*t2+t)*m0+(-2*t3+3*t2)*c+(t3-t2)*m1
    # Guard rounding drift at flat extrema and reproduce source knots exactly.
    return np.clip(value,np.minimum(b,c),np.maximum(b,c))


def monotone_grid(field,x,y):
    """Separable C1 Hermite samples; every value stays inside its central cell.

    x/y are local source coordinates. The caller already reads one neighbour
    either side for bilinear sampling. At an exact terminal knot, the fourth
    neighbour is unnecessary: clamping gathers reproduces that knot exactly.
    No additional source reads or neural dependencies are introduced.
    """
    field=np.asarray(field,dtype=np.float32)
    x=np.asarray(x,dtype=np.float64)
    y=np.asarray(y,dtype=np.float64)
    if field.ndim!=2 or min(field.shape)<2 or x.ndim!=1 or y.ndim!=1:
        raise ValueError('Expected a rectangular field and coordinate vectors')
    if not np.isfinite(field).all() or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('Interpolation inputs must be finite')
    if (x<0).any() or (x>field.shape[1]-1).any() or (y<0).any() or (y>field.shape[0]-1).any():
        raise ValueError('Interpolation coordinates outside source field')
    ix=np.floor(x).astype(np.int64)
    iy=np.floor(y).astype(np.int64)
    fx=(x-ix).astype(np.float32)
    fy=(y-iy).astype(np.float32)
    cols=np.clip(ix[:,None]+np.arange(-1,3),0,field.shape[1]-1)
    rows=np.clip(iy[:,None]+np.arange(-1,3),0,field.shape[0]-1)
    values=field[:,cols]
    across=_hermite(values[:,:,0],values[:,:,1],values[:,:,2],values[:,:,3],fx[None,:])
    values=across[rows,:]
    return _hermite(values[:,0,:],values[:,1,:],values[:,2,:],values[:,3,:],fy[:,None]).astype(np.float32)

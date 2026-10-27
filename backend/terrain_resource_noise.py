"""Fused separable province interpolation, warmed before the first request."""
import numpy as np
try:
    from numba import njit
except ImportError:
    njit=None


def _interpolate(table,ii,jj,fx,fy):
    out=np.empty((len(jj),len(ii)),np.float64)
    period=table.shape[1]
    for y in range(len(jj)):
        for x in range(len(ii)):
            i,j=ii[x],jj[y]
            # Preserve the reference's four additions and multiplication order.
            value=(1-fx[x])*(1-fy[y])*table[j,i]
            value=value+(1-fx[x])*fy[y]*table[j+1,i]
            value=value+fx[x]*(1-fy[y])*table[j,(i+1)%period]
            out[y,x]=value+fx[x]*fy[y]*table[j+1,(i+1)%period]
    return out


kernel=njit(cache=True,nogil=True)(_interpolate) if njit is not None else None
ready=False


def prepare():
    global ready
    if kernel is not None:
        kernel(np.zeros((2,2)),np.zeros(1,np.int64),np.zeros(1,np.int64),np.zeros(1),np.zeros(1))
        ready=True

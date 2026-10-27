"""Exact separable squared EDT, evaluating only requested atlas nodes.

Compiled once at startup, never on a world request. Horizontal nearest sites
are found in two scans; vertical lower envelopes are evaluated only in the
columns/rows needed by authority interpolation. No approximate coast or tract.
"""
import math
import numpy as np
try:
    from numba import njit
except ImportError:
    njit=None


def _distance(mask,rows,cols,spacing):
    h,w=mask.shape
    squared=np.empty((h,w),np.int64)
    inf=1 << 50
    for y in range(h):
        last=-w*2
        for x in range(w):
            if mask[y,x]: last=x
            squared[y,x]=(x-last)**2 if last>=0 else inf
        last=w*3
        for x in range(w-1,-1,-1):
            if mask[y,x]: last=x
            if last<w and (last-x)**2<squared[y,x]:
                squared[y,x]=(last-x)**2
    result=np.empty((len(rows),len(cols)),np.float32)
    v=np.empty(h,np.int32);z=np.empty(h+1,np.float64)
    for col in range(len(cols)):
        x=cols[col];k=-1
        for q in range(h):
            if squared[q,x] == inf: continue
            if k<0:
                k=0;v[0]=q;z[0]=-math.inf;z[1]=math.inf
                continue
            s=((squared[q,x]+q*q)-(squared[v[k],x]+v[k]*v[k]))/(2.*q-2.*v[k])
            while s<=z[k]:
                k-=1
                s=((squared[q,x]+q*q)-(squared[v[k],x]+v[k]*v[k]))/(2.*q-2.*v[k])
            k+=1;v[k]=q;z[k]=s;z[k+1]=math.inf
        k=0
        for row in range(len(rows)):
            y=rows[row]
            while z[k+1]<y: k+=1
            dy=(v[k]-y)*spacing;dx=math.sqrt(squared[v[k],x])*spacing
            result[row,col]=math.sqrt(dy*dy+dx*dx)
    return result


kernel=njit(cache=True,nogil=True)(_distance) if njit is not None else None
ready=False


def prepare():
    global ready
    if kernel is not None:
        mask=np.zeros((4,8),bool);mask[1,2]=True
        kernel(mask,np.arange(4,dtype=np.int64),np.arange(8,dtype=np.int64),1.)
        ready=True

"""Optional exact CUDA spatial search for the Orogen v3 atlas projection."""
from __future__ import annotations

from terrain_paths import REPO_ROOT

from functools import lru_cache
from pathlib import Path
import threading

import numpy as np

ROOT = REPO_ROOT
GPU = ROOT / 'native' / 'orogen' / 'gpu'
VERSION = 'orogen-v3-cuda-projection-v1'
_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def implementation_identity():
    try:
        import cupy as cp
        from terrain_device import select_cuda_device
        selected = select_cuda_device()['selected']
        cp.cuda.Device(selected['index']).use()
        # Compilation is probed too: CUDA driver alone does not guarantee NVRTC.
        a = cp.zeros(1, dtype=cp.float32)
        cp.RawKernel('extern "C" __global__ void probe(float *a){a[0]=1;}', 'probe')((1,), (1,), (a,))
        if a.get()[0] != 1:
            raise RuntimeError('CUDA kernel probe failed')
        return dict(version=VERSION, available=True, cupy_version=cp.__version__,
                    runtime_version=cp.cuda.runtime.runtimeGetVersion(), device=selected)
    except Exception as error:
        return dict(version=VERSION, available=False, fallback='CPU cKDTree', reason=str(error))


@lru_cache(maxsize=128)
def _kernel(source, name='run'):
    import cupy as cp
    # Match JS double intermediates; fused multiply-add would alter thresholds.
    kernel = cp.RawKernel(source, name, options=('--std=c++17', '--fmad=false'))
    kernel.compile()
    return kernel


def nearest_regions(points, query):
    """GPU grid with a conservative stopping bound; exact Euclidean k=4."""
    import cupy as cp
    cp.cuda.Device(implementation_identity()['device']['index']).use()
    if len(points)<4 or points.shape[1:]!=(3,) or query.shape[1:]!=(3,):
        raise ValueError('GPU nearest lookup expects at least four XYZ points')
    with _LOCK:
        bins=64
        p=cp.asarray(points,dtype=cp.float32)
        q=cp.asarray(query,dtype=cp.float32)
        # The input is already on the CPU. Build this small index with NumPy;
        # CuPy's CUB reduction otherwise requires optional CUDA header packages.
        coords=np.clip(np.floor((points+1)*(.5*bins)),0,bins-1).astype(np.int32)
        keys=coords[:,0]+bins*(coords[:,1]+bins*coords[:,2])
        order=cp.asarray(np.argsort(keys,kind='stable').astype(np.int32))
        count=np.bincount(keys,minlength=bins**3)
        offsets=cp.asarray(np.concatenate((np.zeros(1,np.int32),np.cumsum(count,dtype=np.int32))))
        out=cp.empty((len(query),4),cp.int32)
        _kernel((GPU/'nearest.cu').read_text(),'nearest4')(((len(query)+127)//128,),(128,),
            (p,order,offsets,q,out,np.int32(len(query)),np.int32(bins)))
        result=out.get()
        if np.any(result<0):raise RuntimeError('GPU nearest lookup failed')
        return result.astype(np.int64)

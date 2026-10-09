"""Opt-in Orogen CUDA kernels, owned by the existing Python process.

The original JavaScript remains the CPU reference and orchestration layer.
Independent region loops use NVRTC kernels; experimental graph propagation
uses deterministic gather passes. No runtime import occurs for CPU defaults.
"""
from __future__ import annotations

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path

from functools import lru_cache
import json
from pathlib import Path
import struct
import subprocess
from terrain_generation_session import check_generation, cancellable_process, run_process
import threading
import time

import numpy as np

ROOT = REPO_ROOT
GPU = ROOT / 'native' / 'orogen' / 'gpu'
VERSION = 'orogen-cuda-graph-v1'
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
        return dict(version=VERSION, available=False, fallback='original CPU', reason=str(error))


def prepare(directory, flags):
    target = directory / 'gpu-vendor'
    from shutil import which, copyfile
    copyfile(ROOT / 'native/orogen/climate-parameters.json', directory / 'climate-parameters.json')
    result = run_process([which('node'), str(GPU / 'build.mjs'), str(target), json.dumps(flags)],
                            capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError('Orogen GPU adapter failed: ' + result.stderr[-4000:])
    return target


_DISTANCE = r'''
extern "C" __global__ void distance(const int* off,const int* adj,
 const unsigned char* stop,const float* src,float* dst,int n,int* changed) {
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;
 float best=src[r];
 if(!stop[r])for(int i=off[r];i<off[r+1];i++)best=fminf(best,src[adj[i]]+1.f);
 dst[r]=best;if(best<src[r])atomicExch(changed,1);
}
'''
_STRESS = r'''
extern "C" __global__ void stress(const int* off,const int* adj,
 const int* plate,const unsigned char* ocean,const float* src,const float* sf,
 float* dst,float* outsf,int n,float decay,float subDecay) {
 int r=blockIdx.x*blockDim.x+threadIdx.x;if(r>=n)return;
 float best=src[r],factor=sf[r];
 if(!ocean[plate[r]])for(int i=off[r];i<off[r+1];i++){
  int nb=adj[i];if(plate[nb]!=plate[r])continue;
  float value=src[nb]*(sf[nb]>.5f?subDecay:decay);
  if(value>=.005f&&value>best){best=value;factor=sf[nb];}
 }
 dst[r]=best;outsf[r]=factor;
}
'''


@lru_cache(maxsize=128)
def _kernel(source, name='run'):
    import cupy as cp
    # Match JS double intermediates; fused multiply-add would alter thresholds.
    kernel = cp.RawKernel(source, name, options=('--std=c++17', '--fmad=false'))
    kernel.compile()
    return kernel


def _read(stream, count):
    parts = bytearray()
    while len(parts) < count:
        chunk = stream.read(count-len(parts))
        if not chunk:
            raise RuntimeError('Orogen CUDA bridge closed before completion')
        parts.extend(chunk)
    return parts


def _reply(stream, header, data=()):
    encoded = json.dumps(header, separators=(',', ':')).encode()
    stream.write(struct.pack('<I', len(encoded)))
    stream.write(encoded)
    for block in data:
        stream.write(block)
    stream.flush()


def run(node, directory, request, flags, external_erosion=None):
    """Run one Node orchestration with a framed, synchronous CUDA pipe."""
    import cupy as cp
    # CUDA's current device is thread-local, while the identity probe is cached.
    cp.cuda.Device(implementation_identity()['device']['index']).use()
    from terrain_bootstrap import _json_bytes
    target = prepare(directory, flags)
    request.update(gpuVendorDirectory=str(target), gpuRuntime=str(GPU / 'runtime.mjs'))
    if external_erosion is not None:
        request['externalErosion'] = True
    (directory / 'request.json').write_bytes(_json_bytes(request))
    buffers = {}
    timings = dict(kernel_seconds=0., bridge_seconds=0., upload_bytes=0, readback_bytes=0,
                   compile_seconds=0., graph_seconds=0., kernels=0, propagation_steps=0,
                   graph_steps={})
    begun = time.perf_counter()
    external_result = None
    stats = None
    with _LOCK, (directory / 'node.log').open('w+b') as log:
        with cancellable_process([node, str(ROOT / 'native/orogen/pipeline.mjs'), str(directory / 'request.json'), str(directory)],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log) as process:
            try:
                while True:
                    check_generation()
                    size = struct.unpack('<I', _read(process.stdout, 4))[0]
                    if size > 32*1024*1024:
                        raise RuntimeError('Invalid CUDA bridge frame')
                    header = json.loads(_read(process.stdout, size))
                    if header['cmd'] == 'finish':
                        stats = header['stats']
                        _reply(process.stdin, {})
                        break
                    if header['cmd'] == 'externalErosion':
                        if external_erosion is None:
                            raise RuntimeError('Unexpected external erosion request')
                        response, external_result = external_erosion()
                        _reply(process.stdin, response)
                        continue
                    at = time.perf_counter()
                    for array in header['arrays']:
                        identity = array['id']
                        if array['bytes']:
                            host = np.frombuffer(_read(process.stdout, array['bytes']), dtype=array['dtype'])
                            if len(host) != array['length']:
                                raise RuntimeError('CUDA bridge shape mismatch')
                            if identity in buffers:
                                buffers[identity].set(host)
                            else:
                                buffers[identity] = cp.asarray(host)
                            timings['upload_bytes'] += array['bytes']
                        elif identity not in buffers:
                            if array['length'] == 0:
                                buffers[identity] = cp.empty(0, dtype=array['dtype'])
                            else:
                                raise RuntimeError(f'Missing CUDA input buffer: {header.get("label", header.get("kind"))}, {array}')
                    if header['cmd'] == 'kernel':
                        kt = time.perf_counter()
                        kernel = _kernel(header['source'])
                        timings['compile_seconds'] += time.perf_counter()-kt
                        kt = time.perf_counter()
                        n = header['bound']
                        batch=header.get('batch')
                        count=batch['iterations'] if batch else 1
                        current=dict(buffers)
                        for iteration in range(count):
                            check_generation()
                            for dest,src in batch['before'] if batch else ():
                                cp.copyto(current[dest],current[src])
                            args=tuple(current[arg['array']] if 'array' in arg else
                                       np.int32(arg['scalar']) if arg.get('integer') else np.float64(arg['scalar'])
                                       for arg in header['args'])
                            kernel(((n+127)//128,), (128,), args)
                            for dest,src in batch['after'] if batch else ():
                                cp.copyto(current[dest],current[src])
                            for a,b in batch['swaps'] if batch else ():
                                current[a],current[b]=current[b],current[a]
                        cp.cuda.get_current_stream().synchronize()
                        timings['kernel_seconds'] += time.perf_counter()-kt
                        timings['kernels'] += count
                    elif header['cmd'] == 'special':
                        fields = {key: buffers[val['array']] if 'array' in val else val['scalar']
                                  for key, val in header['fields'].items()}
                        graph_at=time.perf_counter()
                        steps = _special(header['kind'], fields)
                        cp.cuda.get_current_stream().synchronize()
                        timings['graph_seconds'] += time.perf_counter()-graph_at
                        timings['propagation_steps'] += steps
                        timings['graph_steps'][header['kind']]=timings['graph_steps'].get(header['kind'],0)+steps
                    else:
                        raise RuntimeError('Unknown CUDA command')
                    output = [buffers[i].get().tobytes() for i in header['outputs']]
                    _reply(process.stdin, dict(outputs=[dict(array=i, bytes=len(b)) for i, b in zip(header['outputs'], output)]), output)
                    timings['readback_bytes'] += sum(map(len, output))
                    timings['bridge_seconds'] += time.perf_counter()-at
                process.stdin.close()
                if process.wait(timeout=600):
                    raise RuntimeError('Orogen CUDA orchestration failed')
            except Exception as error:
                if process.poll() is None:
                    try:
                        _reply(process.stdin, {'error': str(error)[-4000:]})
                    except (OSError, ValueError):
                        pass
                    process.kill()
                process.wait()
                for attribute in ('stdin','stdout'):
                    stream=getattr(process,attribute)
                    try:
                        if stream:stream.close()
                    except OSError:
                        pass
                    setattr(process,attribute,None)
                log.seek(0)
                raise RuntimeError(str(error) + '\n' + log.read().decode(errors='replace')[-6000:]) from error
    timings['seconds'] = time.perf_counter()-begun
    timings['peak_buffer_bytes'] = sum(a.nbytes for a in buffers.values())
    return dict(identity=implementation_identity(), flags=flags, timings=timings, coverage=stats), external_result


def _special(kind, f):
    import cupy as cp
    n = len(f['adjOffset'])-1
    grid, block = ((n+127)//128,), (128,)
    if kind == 'erode':
        return _erode_graph(f)
    if kind == 'distance':
        src = cp.where(f['seeds'] != 0, cp.float32(0), cp.float32(cp.inf))
        dst = cp.empty_like(src)
        changed = cp.zeros(1, cp.int32)
        kernel = _kernel(_DISTANCE, 'distance')
        # A connected graph can have diameter N-1; never silently truncate.
        for step in range(n):
            check_generation()
            changed.fill(0)
            kernel(grid, block, (f['adjOffset'], f['adjList'], f['stops'], src, dst, np.int32(n), changed))
            src, dst = dst, src
            if not changed.get()[0]:
                f['result'][:] = src
                return step+1
        raise RuntimeError('GPU distance field did not converge')
    if kind == 'stress':
        src, sf = f['r_stress'].copy(), f['r_subductFactor'].copy()
        dst, outsf = cp.empty_like(src), cp.empty_like(sf)
        kernel = _kernel(_STRESS, 'stress')
        for step in range(int(f['numPasses'])):
            check_generation()
            kernel(grid, block, (f['adjOffset'], f['adjList'], f['r_plate'], f['plateIsOcean'],
                                src, sf, dst, outsf, np.int32(n), np.float32(f['decayFactor']), np.float32(f['subductDecayFactor'])))
            src, dst = dst, src
            sf, outsf = outsf, sf
        f['r_stress'][:] = src
        f['r_subductFactor'][:] = sf
        return int(f['numPasses'])
    raise RuntimeError('Unknown GPU graph operation')


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
        coords=cp.clip(cp.floor((p+1)*(.5*bins)),0,bins-1).astype(cp.int32)
        keys=coords[:,0]+bins*(coords[:,1]+bins*coords[:,2])
        order=cp.argsort(keys).astype(cp.int32)
        count=cp.bincount(keys,minlength=bins**3)
        offsets=cp.concatenate((cp.zeros(1,cp.int32),cp.cumsum(count,dtype=cp.int32)))
        out=cp.empty((len(query),4),cp.int32)
        _kernel((GPU/'nearest.cu').read_text(),'nearest4')(((len(query)+127)//128,),(128,),
            (p,order,offsets,q,out,np.int32(len(query)),np.int32(bins)))
        result=out.get()
        if np.any(result<0):raise RuntimeError('GPU nearest lookup failed')
        return result.astype(np.int64)


def _erode_graph(f):
    import cupy as cp
    source=(GPU/'erosion.cu').read_text()
    n=len(f['r_elevation']);grid,block=((n+127)//128,),(128,)
    args=(f['adjOffset'],f['adjList'])
    sea=f['r_isOcean'];dist=f['neighborDist']
    height=f['r_elevation'].copy();dst=cp.empty_like(height)
    changed=cp.zeros(1,cp.int32);steps=0
    def solve(name,initial,build,limit=n):
        nonlocal steps
        src=initial.copy();out=cp.empty_like(src);kernel=_kernel(source,name)
        for i in range(limit):
            check_generation()
            changed.fill(0)
            kernel(grid,block,build(src,out)+(changed,))
            src,out=out,src;steps+=1
            if not changed.get()[0]:return src
        raise RuntimeError('GPU '+name+' did not converge')
    if f['hIters']>0 and bool(cp.any(sea).get()):
        initial=cp.where(sea!=0,height,cp.float32(1.e30))
        # Minimax fill from all water bodies, not the reference canyon-carving heap.
        height=solve('flood',initial,lambda a,b:args+(sea,height,a,b,np.int32(n)))
    target=cp.empty(n,cp.int32);length=cp.empty(n,cp.float32)
    total=max(int(f['hIters']),int(f['tIters']),int(f['gIters']))
    for iteration in range(total):
        check_generation()
        if iteration<f['gIters'] and f['glacialStrength']>0:
            _kernel(source,'glaciers')(grid,block,args+(dist,sea,f['r_xyz'],height,dst,np.int32(n),
                np.float32(f['glacialStrength']),np.float32(1/max(1,f['gIters']))))
            height,dst=dst,height
        if iteration<f['hIters']:
            _kernel(source,'receivers')(grid,block,args+(dist,sea,height,target,length,np.int32(n)))
            area=cp.where(sea!=0,cp.float32(0),cp.float32(1))
            area=solve('accumulate',area,lambda a,b:args+(target,sea,a,b,np.int32(n)))
            old=height.copy()
            height=solve('incision',height,lambda a,b:(target,length,sea,old,area,a,b,np.int32(n),
                np.float32(f['K']),np.float32(f['m']),np.float32(f['dt'])))
            _kernel(source,'deposition')(grid,block,args+(target,length,sea,old,height,dst,np.int32(n)))
            height,dst=dst,height
        if iteration<f['tIters']:
            _kernel(source,'thermal')(grid,block,args+(dist,sea,height,dst,np.int32(n),
                np.float32(f['talusSlope']),np.float32(f['kThermal'])))
            height,dst=dst,height
    if not bool(cp.isfinite(height).all().get()):raise RuntimeError('GPU graph erosion produced non-finite heights')
    f['r_elevation'][:]=height
    return steps

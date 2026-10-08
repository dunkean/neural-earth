"""City Generator surface erosion on an optional native WebGPU device.

The bundled WGSL retains City Generator's tiled convergence solver and
D-infinity flow / implicit incision. Heights are metres, longitude wraps,
latitude changes physical neighbour distances, and ocean samples stay exact.
There is no terrain generation or percentile normalization in this adapter.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import math
import threading
from terrain_generation_session import check_generation
import time

import numpy as np

VERSION = 'city-surface-periodic-v1'
SHADER = Path(__file__).parent / 'native' / 'city_erosion' / 'surface.wgsl'
_LOCK = threading.RLock()


@lru_cache(maxsize=1)
def _runtime():
    try:
        import wgpu
    except ImportError as exc:
        raise RuntimeError('City GPU erosion requires the optional dependency: '
                           'python -m pip install -r requirements-city-gpu.txt') from exc
    try:
        adapter = wgpu.gpu.request_adapter_sync(power_preference='high-performance')
        if adapter is None or adapter.info['adapter_type'] == 'CPU':
            raise RuntimeError('No hardware WebGPU adapter available')
        device = adapter.request_device_sync(required_limits={
            'max-storage-buffer-binding-size': adapter.limits['max-storage-buffer-binding-size'],
            'max-buffer-size': adapter.limits['max-buffer-size'],
        })
        layout = device.create_bind_group_layout(entries=[
            dict(binding=0, visibility=wgpu.ShaderStage.COMPUTE,
                 buffer=dict(type='uniform', has_dynamic_offset=True, min_binding_size=192)),
            dict(binding=1, visibility=wgpu.ShaderStage.COMPUTE, buffer=dict(type='read-only-storage')),
            *[dict(binding=i, visibility=wgpu.ShaderStage.COMPUTE, buffer=dict(type='storage'))
              for i in (3, 5, 6, 7)],
        ])
        module = device.create_shader_module(code=SHADER.read_text(encoding='utf-8'))
        pipeline_layout = device.create_pipeline_layout(bind_group_layouts=[layout])
        names = ('loadHeight', 'surfaceFlowInit', 'receivers', 'flow', 'graphInit',
                 'start', 'clearNext', 'solveStats', 'graphFinish', 'thermalFlux',
                 'thermalApply', 'copyHeight', 'diffuse', 'packHeight')
        pipelines = {name:device.create_compute_pipeline(layout=pipeline_layout,
                     compute=dict(module=module, entry_point=name)) for name in names}
        for mode in (1, 3):
            pipelines[f'solveTile:{mode}'] = device.create_compute_pipeline(layout=pipeline_layout,
                compute=dict(module=module, entry_point='solveTile', constants={'SOLVE_MODE': mode}))
        return wgpu, adapter, device, layout, pipelines
    except Exception as exc:
        raise RuntimeError(f'City GPU erosion could not initialize: {exc}. '
                           'Select Orogen for CPU erosion.') from exc


def implementation_identity():
    with _LOCK:
        wgpu, adapter, _, _, _ = _runtime()
        return dict(version=VERSION, wgpu_version=wgpu.__version__, adapter=dict(adapter.info),
                    shader_sha256=hashlib.sha256(SHADER.read_bytes()).hexdigest())


def erode(height_m, *, strength=1., iterations=12, talus=.6, motif_km=300., world_height_m=20_000_000.):
    source = np.ascontiguousarray(height_m, dtype='<f4')
    if (source.ndim != 2 or source.shape[1] != 2*source.shape[0]
            or source.shape[0] < 16 or source.shape[1] > 4096 or not np.isfinite(source).all()):
        raise ValueError('City erosion requires a finite 2:1 float32 world atlas (width 32..4096)')
    if (isinstance(iterations, bool) or int(iterations) != iterations or not 1 <= iterations <= 64
            or not math.isfinite(strength) or not 0 <= strength <= 2
            or not math.isfinite(talus) or not .01 <= talus <= 4
            or not math.isfinite(motif_km) or not 10 <= motif_km <= 5000):
        raise ValueError('Invalid City erosion strength, iterations, talus or motif scale')
    if not math.isfinite(world_height_m) or world_height_m <= 0:
        raise ValueError('Invalid world height')
    if strength == 0:
        return source.copy(), dict(engine=VERSION, seconds=0., iterations=0, identity=None)
    with _LOCK:
        started = time.perf_counter()
        wgpu, adapter, device, layout, pipelines = _runtime()
        setup_seconds = time.perf_counter()-started
        h, w = source.shape
        count = source.size
        if count*112 > device.limits['max-storage-buffer-binding-size']:
            raise RuntimeError('City erosion atlas exceeds this GPU storage buffer limit; select Orogen.')
        buffers = []
        control = None
        def create(size, usage):
            buffer = device.create_buffer(size=size, usage=usage)
            buffers.append(buffer)
            return buffer
        try:
            uniform = create(65536, wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST)
            input_buffer = create(source.nbytes, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_DST)
            device.queue.write_buffer(input_buffer, 0, source)
            nodes = create(count*112, wgpu.BufferUsage.STORAGE)
            control = create(1088, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC | wgpu.BufferUsage.COPY_DST)
            output = create(source.nbytes, wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
            tiles = create(math.ceil(w/16)*math.ceil(h/16)*8, wgpu.BufferUsage.STORAGE)
            group = device.create_bind_group(layout=layout, entries=[
                dict(binding=0, resource=dict(buffer=uniform, size=192)),
                *[dict(binding=i, resource=dict(buffer=b)) for i,b in
                  ((1,input_buffer),(3,nodes),(5,control),(6,output),(7,tiles))],
            ])
            cell = world_height_m/h
            motif = motif_km*1000.
            parameters = np.zeros(48, np.float32)
            parameters[2:4] = [w, motif/2400.]
            parameters[4] = cell
            parameters[15] = talus
            parameters[18] = strength*.1125
            parameters[22] = strength*.40*cell/(math.sqrt(.004)*motif)
            parameters[26] = .95
            parameters[44:46] = [w, h]
            variants = {}
            def offset(mode=0, phase=0, diffusion=0.):
                key = (mode, phase, diffusion)
                if key not in variants:
                    config = parameters.copy()
                    config[24:26] = [phase, mode]
                    config[27] = diffusion
                    variants[key] = len(variants)*256
                    device.queue.write_buffer(uniform, variants[key], config)
                return variants[key]
            offset()
            for mode in (1,3):
                for phase in (0,1): offset(mode,phase)
            offset(diffusion=min(.1,strength*.025))
            # The upstream solver validates completion and retries from the
            # original input, never accepts a truncated drainage/height solve.
            attempts = []
            budgets = (128,256,512) if count>=512*1024 else (64,128,256,512)
            for budget in budgets:
                check_generation()
                encoder = device.create_command_encoder()
                encoder.clear_buffer(control)
                compute = encoder.begin_compute_pass()
                def dispatch(name, mode=0, phase=0, diffusion=0., groups=None):
                    compute.set_pipeline(pipelines[name])
                    compute.set_bind_group(0, group, [offset(mode,phase,diffusion)])
                    compute.dispatch_workgroups(groups if groups is not None else math.ceil(count/256))
                def solve(mode):
                    dispatch('graphInit',mode)
                    dispatch('start',mode,groups=1)
                    for step in range(budget):
                        check_generation()
                        dispatch('clearNext',mode,step&1,groups=1)
                        compute.set_pipeline(pipelines[f'solveTile:{mode}'])
                        compute.dispatch_workgroups(math.ceil(w/16),math.ceil(h/16))
                    dispatch('solveStats',mode,groups=1)
                    dispatch('graphFinish',mode)
                dispatch('loadHeight')
                for _ in range(int(iterations)):
                    check_generation()
                    for name in ('surfaceFlowInit','receivers','flow'): dispatch(name)
                    solve(1)
                    solve(3)
                    for name in ('thermalFlux','thermalApply','copyHeight'): dispatch(name)
                    dispatch('diffuse',diffusion=min(.1,strength*.025))
                    dispatch('copyHeight')
                dispatch('packHeight')
                compute.end()
                submitted = time.perf_counter()
                device.queue.submit([encoder.finish()])
                status = np.frombuffer(device.queue.read_buffer(control), dtype='<u4')
                attempts.append(dict(budget=budget, errors=int(status[2]),
                    max_solve_steps=status[267:271].tolist(),
                    submit_seconds=time.perf_counter()-submitted))
                if status[2] == 0:
                    result = np.frombuffer(device.queue.read_buffer(output),dtype='<f4').copy().reshape(h,w)
                    if not np.isfinite(result).all() or not np.array_equal(result[source<=0],source[source<=0]):
                        raise RuntimeError('City erosion produced invalid heights or changed ocean samples')
                    return result, dict(engine=VERSION, seconds=time.perf_counter()-started,
                        setup_seconds=setup_seconds, iterations=int(iterations), attempts=attempts,
                        workspace_bytes=sum(b.size for b in buffers), identity=implementation_identity())
            raise RuntimeError('City GPU erosion did not converge after 512 tiled passes; select Orogen.')
        finally:
            # Cancellation still drains submitted WebGPU work before releasing the GPU.
            if control is not None:
                device.queue.read_buffer(control,0,4)
            for buffer in buffers: buffer.destroy()

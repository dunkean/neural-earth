"""Small BF16 CUDA kernels, compiled by PyTorch's NVRTC and captured by its graphs.

No tensor storage is owned by the driver; PyTorch allocates outputs and owns all
inputs. Modules remain resident while captured graphs may reference functions.
Only inference, contiguous BF16 CUDA tensors are admitted. No CPU copy occurs.
"""
from __future__ import annotations

import ctypes as C
from ctypes.util import find_library
import hashlib
import os
import sys
from pathlib import Path
import threading

import torch

SOURCE = r'''
typedef unsigned short bf;
__device__ __forceinline__ float unpack(bf x) { return __uint_as_float((unsigned)x << 16); }
__device__ __forceinline__ bf pack(float x) {
    bf result; asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(result) : "f"(x)); return result;
}
extern "C" __global__ void binary_sum(const bf* a, const bf* b,
        const bf* weights, const bf* norm, bf* out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // Match each materialized BF16 product, sum, +0 and division separately.
    float p = unpack(pack(unpack(a[i]) * unpack(weights[0])));
    float q = unpack(pack(unpack(b[i]) * unpack(weights[1])));
    float sum = unpack(pack(p + q));
    sum = unpack(pack(sum + 0.0f));
    out[i] = pack(sum / unpack(norm[0]));
}
extern "C" __global__ void binary_concat(const bf* a, const bf* b,
        const bf* factor_a, const bf* factor_b, bf* out,
        long long n, long long plane_a, long long plane_b) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    long long plane = plane_a + plane_b, batch = i / plane, j = i % plane;
    out[i] = j < plane_a ? pack(unpack(a[batch * plane_a + j]) * unpack(factor_a[0]))
                         : pack(unpack(b[batch * plane_b + j - plane_a]) * unpack(factor_b[0]));
}
extern "C" __global__ void silu_scaled(const bf* x, bf* out, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float value = unpack(x[i]);
    float activated = unpack(pack(value / (1.0f + expf(-value))));
    out[i] = pack(activated / 0.596f);
}
'''
SOURCE_SHA256 = hashlib.sha256(SOURCE.encode()).hexdigest()
WRAPPER_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
_packs = {}
_lock = threading.RLock()


class KernelUnavailable(RuntimeError):
    pass


def _signature(dll, name, restype, argtypes):
    function = getattr(dll, name)
    function.restype, function.argtypes = restype, argtypes
    return function


def _load_library(loader, candidates, label):
    errors = []
    for candidate in dict.fromkeys(str(p) for p in candidates if p):
        try:
            return loader(candidate)
        except OSError as error:
            errors.append(f'{candidate}: {error}')
    raise KernelUnavailable(f'{label} unavailable: ' + '; '.join(errors))


def _cuda_libraries():
    root = Path(torch.__file__).resolve().parent
    if os.name == 'nt':
        return (_load_library(C.WinDLL, [root / 'lib' / 'nvrtc64_120_0.dll',
                                        *sorted((root / 'lib').glob('nvrtc64_*.dll'))], 'NVRTC'),
                _load_library(C.WinDLL, ['nvcuda.dll'], 'CUDA driver'))
    if not sys.platform.startswith('linux'):
        raise KernelUnavailable('Native CUDA kernels support Windows and Linux')
    # Linux Torch wheels install NVRTC in the sibling nvidia namespace package.
    # Prefer the compiler bundled with Torch before trying a system toolkit.
    libraries = [*sorted((root.parent / 'nvidia' / 'cuda_nvrtc' / 'lib').glob('libnvrtc.so*')),
                 *sorted((root / 'lib').glob('libnvrtc.so*'))]
    for variable in ('CUDA_HOME', 'CUDA_PATH'):
        if os.environ.get(variable):
            libraries.extend(sorted((Path(os.environ[variable]) / 'lib64').glob('libnvrtc.so*')))
    libraries.extend([find_library('nvrtc'), 'libnvrtc.so.12', 'libnvrtc.so'])
    return (_load_library(C.CDLL, libraries, 'NVRTC'),
            _load_library(C.CDLL, [find_library('cuda'), 'libcuda.so.1',
                                  '/usr/lib/wsl/lib/libcuda.so.1'], 'CUDA driver'))


class KernelPack:
    def __init__(self, device):
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            # Ask Torch to initialize CUDA; verify the current driver context below.
            torch.empty(0, device=self.device)
            capability = torch.cuda.get_device_capability(self.device)
            if capability[0] < 8:
                raise KernelUnavailable('BF16 kernels require compute capability >= 8')
            self.nvrtc, self.driver = _cuda_libraries()
            self._bind()
            context = C.c_void_p()
            self.check(self.ctx_current(C.byref(context)))
            self.context = context.value
            if not self.context:
                raise KernelUnavailable('No active CUDA context')
            program = C.c_void_p()
            self.check_nvrtc(self.create(C.byref(program), SOURCE.encode(), b'terrain_kernels.cu', 0, None, None))
            options = [f'--gpu-architecture=compute_{capability[0]}{capability[1]}'.encode(),
                       b'--std=c++14', b'--fmad=false', b'--prec-div=true', b'--prec-sqrt=true']
            try:
                result = self.compile(program, len(options), (C.c_char_p * len(options))(*options))
                if result:
                    size = C.c_size_t()
                    self.log_size(program, C.byref(size))
                    log = C.create_string_buffer(size.value)
                    self.log(program, log)
                    raise KernelUnavailable(log.value.decode(errors='replace')[:4000])
                size = C.c_size_t()
                self.check_nvrtc(self.ptx_size(program, C.byref(size)))
                ptx = C.create_string_buffer(size.value)
                self.check_nvrtc(self.ptx(program, ptx))
                self.module = C.c_void_p()
                self.check(self.load(C.byref(self.module), C.cast(ptx, C.c_void_p), 0, None, None))
                self.functions = {}
                for name in ('binary_sum', 'binary_concat', 'silu_scaled'):
                    function = C.c_void_p()
                    self.check(self.get_function(C.byref(function), self.module, name.encode()))
                    self.functions[name] = function
            finally:
                self.destroy(C.byref(program))

    def _bind(self):
        pointer = C.c_void_p
        pp = C.POINTER(pointer)
        self.create = _signature(self.nvrtc, 'nvrtcCreateProgram', C.c_int,
                                 [pp, C.c_char_p, C.c_char_p, C.c_int, C.POINTER(C.c_char_p), C.POINTER(C.c_char_p)])
        self.compile = _signature(self.nvrtc, 'nvrtcCompileProgram', C.c_int,
                                  [pointer, C.c_int, C.POINTER(C.c_char_p)])
        self.destroy = _signature(self.nvrtc, 'nvrtcDestroyProgram', C.c_int, [pp])
        for field, name, tail in (('log_size', 'nvrtcGetProgramLogSize', C.POINTER(C.c_size_t)),
                                  ('ptx_size', 'nvrtcGetPTXSize', C.POINTER(C.c_size_t)),
                                  ('log', 'nvrtcGetProgramLog', pointer), ('ptx', 'nvrtcGetPTX', pointer)):
            setattr(self, field, _signature(self.nvrtc, name, C.c_int, [pointer, tail]))
        self.ctx_current = _signature(self.driver, 'cuCtxGetCurrent', C.c_int, [pp])
        self.load = _signature(self.driver, 'cuModuleLoadDataEx', C.c_int,
                               [pp, pointer, C.c_uint, pointer, pointer])
        self.get_function = _signature(self.driver, 'cuModuleGetFunction', C.c_int,
                                       [pp, pointer, C.c_char_p])
        self.launch = _signature(self.driver, 'cuLaunchKernel', C.c_int,
                                 [pointer, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint, C.c_uint,
                                  C.c_uint, pointer, pp, pp])

    @staticmethod
    def check(result):
        if result:
            raise RuntimeError(f'CUDA driver error {result}')

    @staticmethod
    def check_nvrtc(result):
        if result:
            raise KernelUnavailable(f'NVRTC error {result}')

    def run(self, name, tensors, integers, elements):
        if not elements:
            return
        # Never compile/load or synchronize here, including during graph capture.
        with torch.cuda.device(self.device):
            stream = torch.cuda.current_stream(self.device)
            context = C.c_void_p()
            self.check(self.ctx_current(C.byref(context)))
            if context.value != self.context:
                raise RuntimeError('CUDA context changed after kernel preparation')
            arguments = [C.c_void_p(t.data_ptr()) for t in tensors]
            arguments += [C.c_longlong(value) for value in integers]
            pointers = (C.c_void_p * len(arguments))(*(C.cast(C.byref(a), C.c_void_p) for a in arguments))
            self.check(self.launch(self.functions[name], (elements + 255) // 256, 1, 1, 256, 1, 1,
                                   0, C.c_void_p(stream.cuda_stream), pointers, None))
            # Inputs may originate on another allocator stream. Protect them
            # until this launch completes; no device synchronization is needed.
            for tensor in tensors:
                tensor.record_stream(stream)


def prepare(device):
    device = torch.device(device)
    if device.type != 'cuda':
        raise KernelUnavailable('CUDA tensor required')
    index = device.index if device.index is not None else torch.cuda.current_device()
    with _lock:
        if index not in _packs:
            with torch.cuda.device(index):
                if torch.cuda.is_current_stream_capturing():
                    raise KernelUnavailable('Prepare kernels before CUDA graph capture')
            pack = KernelPack(torch.device('cuda', index))
            # expf/compiler implementations can vary. Admit SiLU only after an
            # exhaustive finite BF16 gate on this device, before any capture.
            with torch.cuda.device(index), torch.inference_mode():
                values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).to(pack.device)
                values = values[torch.isfinite(values)]
                actual = torch.empty_like(values)
                pack.run('silu_scaled', [values, actual], [values.numel()], values.numel())
                expected = torch.nn.functional.silu(values) / .596
                if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
                    raise KernelUnavailable('Finite BF16 SiLU numerical admission failed on this device')
                a = (torch.arange(64, dtype=torch.float32).reshape(1,2,4,8) / 4 - 8).to(pack.device, torch.bfloat16)
                b = -a.flip(-1)
                a.flatten()[:4] = torch.tensor([0.,-0.,0.,-0.],device=pack.device,dtype=a.dtype)
                b.flatten()[:4] = torch.tensor([0.,0.,-0.,-0.],device=pack.device,dtype=a.dtype)
                for pair in ((.3,.7),(.5,.5),(1.,-1.)):
                    weights = torch.tensor(pair,device=pack.device,dtype=a.dtype)
                    norm = torch.linalg.vector_norm(weights)
                    actual = torch.empty_like(a)
                    pack.run('binary_sum',[a,b,weights,norm,actual],[a.numel()],a.numel())
                    expected = torch.stack([a*weights[0],b*weights[1]]).sum(0) / norm
                    if not torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)):
                        raise KernelUnavailable('BF16 sum numerical admission failed on this device')
                b = torch.cat([b,b[:,:1]],dim=1)
                weights = torch.tensor([.3,.7],device=pack.device,dtype=a.dtype)
                c = torch.sqrt(torch.tensor(5,device=pack.device,dtype=a.dtype) / torch.square(weights).sum())
                factors = [c / float(size)**.5 * weights[i] for i,size in enumerate((2,3))]
                actual = torch.empty((1,5,4,8),device=pack.device,dtype=a.dtype)
                pack.run('binary_concat',[a,b,*factors,actual],[actual.numel(),a.numel(),b.numel()],actual.numel())
                expected = torch.cat([a*factors[0],b*factors[1]],dim=1)
                if not torch.equal(actual.view(torch.uint8),expected.view(torch.uint8)):
                    raise KernelUnavailable('BF16 concat numerical admission failed on this device')
            _packs[index] = pack
        return _packs[index]


def _admitted(tensors):
    return (not torch.is_grad_enabled() and tensors and tensors[0].is_cuda and
            all(t.device == tensors[0].device and t.dtype == torch.bfloat16 and t.is_contiguous() for t in tensors))


def binary_sum(a, b, weights, norm):
    if not _admitted([a, b, weights, norm]) or a.shape != b.shape or weights.numel() != 2 or norm.numel() != 1:
        return None
    output = torch.empty_like(a)
    prepare(a.device).run('binary_sum', [a, b, weights, norm, output], [a.numel()], a.numel())
    return output


def binary_concat(a, b, factors, dim):
    if (dim != 1 or a.ndim < 2 or b.ndim != a.ndim or len(factors) != 2 or
            not _admitted([a, b, *factors]) or a.shape[0] != b.shape[0] or a.shape[2:] != b.shape[2:] or
            any(f.numel() != 1 for f in factors)):
        return None
    output = torch.empty((a.shape[0], a.shape[1] + b.shape[1], *a.shape[2:]), device=a.device, dtype=a.dtype)
    planes = (a.shape[1:].numel(), b.shape[1:].numel())
    prepare(a.device).run('binary_concat', [a, b, *factors, output], [output.numel(), *planes], output.numel())
    return output


def silu_scaled(x):
    if not _admitted([x]):
        return None
    output = torch.empty_like(x)
    prepare(x.device).run('silu_scaled', [x, output], [x.numel()], x.numel())
    return output


def status():
    with _lock:
        return dict(source_sha256=SOURCE_SHA256, wrapper_sha256=WRAPPER_SHA256, devices=sorted(_packs),
                    compiler='nvrtc', intermediate_precision='explicit-bf16', fma=False,
                    finite_bf16_silu_admitted_devices=sorted(_packs),
                    sum_concat_probe_admitted_devices=sorted(_packs))

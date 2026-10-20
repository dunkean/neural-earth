"""Bounded CUDA graphs for true one-pass native windows, including embeddings."""
import threading

import torch
from torch import nn

from distill.native_onepass_runtime import install_onepass


class OnePassGraph(nn.Module):
    def __init__(self, runtime, exact_kernels=False, enabled=True):
        super().__init__()
        self.runtime=runtime
        self.config=runtime.config
        self._lock=threading.Lock()
        self._cache=None
        self._prepared=False
        self.exact_kernels=bool(exact_kernels)
        self.enabled=bool(enabled)
        self.eval().requires_grad_(False)

    def forward(self,noises,condition):
        if not self.enabled or not noises.is_cuda or torch.is_grad_enabled() or self.training:
            return self.runtime(noises,condition)
        if noises.shape[1:]!=(10,64,64) or condition.shape!=(len(noises),58):
            raise ValueError('Complete64px two-noise windows and58 conditioning values required.')
        with self._lock:
            if not self._prepared:
                from types import SimpleNamespace
                from terrain_nn_constants import prepare_constants
                prepare_constants(SimpleNamespace(device=noises.device,coarse_model=self.runtime.net,
                    base_model=self.runtime.net,decoder_model=self.runtime.net),exact_kernels=self.exact_kernels)
                self._prepared=True
            key=(tuple(noises.shape),tuple(noises.stride()),noises.dtype,noises.device,
                 tuple(condition.shape),condition.dtype,tuple(condition.stride()),
                 tuple((id(p),p._version) for p in self.runtime.parameters()))
            if self._cache is None or self._cache[0]!=key:
                self._cache=None
                static_noise=noises.clone();static_condition=condition.clone()
                stream=torch.cuda.Stream(device=noises.device)
                stream.wait_stream(torch.cuda.current_stream(noises.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):reference=self.runtime(static_noise,static_condition)
                torch.cuda.current_stream(noises.device).wait_stream(stream)
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream,capture_error_mode='thread_local'):
                    output=self.runtime(static_noise,static_condition)
                graph.replay()
                if not torch.equal(reference,output):
                    raise RuntimeError('One-pass graph differs from eager windows.')
                self._cache=(key,graph,static_noise,static_condition,output)
            _,graph,static_noise,static_condition,output=self._cache
            static_noise.copy_(noises);static_condition.copy_(condition)
            graph.replay()
            return output.clone()


def install_onepass_graphs(world,runtime,exact_kernels=False,enabled=True):
    from types import MethodType
    original=world._process_latent_conditioning
    def process(self,image,histogram,means,stds,noise_level,seed_offset=0):
        # The dataset's58-value conditioning uses CPU float32 arithmetic.
        # GPU-resident coarse windows must not mix with CPU histogram tensors.
        return original(image.cpu(),histogram.cpu(),means.cpu(),stds.cpu(),noise_level.cpu(),
                        seed_offset=seed_offset)
    world._process_latent_conditioning=MethodType(process,world)
    captured=OnePassGraph(runtime,exact_kernels,enabled)
    install_onepass(world,captured)
    return captured

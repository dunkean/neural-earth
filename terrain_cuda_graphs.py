"""Bounded CUDA graph buckets for eager Terrain Diffusion eval networks.

Opt-in only after per-GPU numerical/latency validation. No input padding, model
approximation or solver changes. The graph owns static buffers and callers get
an independent output clone; replaying another request cannot overwrite it.
"""
from collections import OrderedDict
import threading

import torch


class CudaGraphModel(torch.nn.Module):
    def __init__(self, model, *, max_buckets=4, max_batch=None):
        super().__init__()
        self.model = model
        self.max_buckets = max_buckets
        self.max_batch = max_batch
        self._buckets = OrderedDict()
        self._lock = threading.Lock()
        self.replay_calls = 0
        self.capture_calls = 0
        self.fallback_calls = 0
        self.capture_errors = []
        self._failed_keys = set()
        self._parameter_versions = None
        self.eval()

    @property
    def config(self):
        return self.model.config

    def compute_embeddings(self, *args, **kwargs):
        return self.model.compute_embeddings(*args, **kwargs)

    def _clear_buckets(self):
        for bucket in self._buckets.values():
            bucket[-1].synchronize()
        self._buckets.clear()

    def _key(self, x, labels, conditions, embeds):
        return (tuple(x.shape), x.dtype, x.device,
                tuple(labels.shape), labels.dtype,
                tuple((tuple(t.shape), t.dtype, t.device) for t in conditions),
                (tuple(embeds.shape), embeds.dtype, embeds.device) if embeds is not None else None)

    def forward(self, x, noise_labels, conditional_inputs, return_logvar=False, precomputed_embeds=None):
        conditions = conditional_inputs or []
        if self.training or torch.is_grad_enabled() or not x.is_cuda or return_logvar or (self.max_batch is not None and x.shape[0] > self.max_batch):
            self.fallback_calls += 1
            return self.model(x, noise_labels=noise_labels, conditional_inputs=conditions,
                              return_logvar=return_logvar, precomputed_embeds=precomputed_embeds)
        key = self._key(x, noise_labels, conditions, precomputed_embeds)
        with self._lock:
            versions = tuple((id(p), p._version) for p in self.model.parameters())
            if versions != self._parameter_versions:
                self._clear_buckets()
                self._failed_keys.clear()
                self._parameter_versions = versions
            if key in self._failed_keys:
                self.fallback_calls += 1
                return self.model(x, noise_labels=noise_labels, conditional_inputs=conditions, precomputed_embeds=precomputed_embeds)
            bucket = self._buckets.get(key)
            if bucket is None:
                try:
                    static_x = x.clone()
                    static_labels = noise_labels.clone()
                    static_conditions = [value.clone() for value in conditions]
                    static_embeds = precomputed_embeds.clone() if precomputed_embeds is not None else None
                    # Warmup side stream initialises cuDNN/cuBLAS and any lazy
                    # MP eval weights before capture, as required by PyTorch.
                    warm = torch.cuda.Stream(device=x.device)
                    warm.wait_stream(torch.cuda.current_stream(x.device))
                    with torch.cuda.stream(warm):
                        for _ in range(3):
                            warm_reference = self.model(static_x, noise_labels=static_labels, conditional_inputs=static_conditions, precomputed_embeds=static_embeds)
                    torch.cuda.current_stream(x.device).wait_stream(warm)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=warm):
                        static_output = self.model(static_x, noise_labels=static_labels, conditional_inputs=static_conditions, precomputed_embeds=static_embeds)
                    # Every device/bucket gets its own exact first-use gate.
                    # This one scalar check is confined to capture setup.
                    graph.replay()
                    if not torch.equal(warm_reference, static_output):
                        raise RuntimeError('CUDA graph failed exact eager-output validation for this device/bucket')
                    last_use = torch.cuda.Event()
                    bucket = (graph, static_x, static_labels, static_conditions, static_embeds, static_output, last_use)
                    self._buckets[key] = bucket
                    self.capture_calls += 1
                    while len(self._buckets) > self.max_buckets:
                        _, evicted = self._buckets.popitem(last=False)
                        # Return private capture allocations only after their
                        # last replay/output clone has finished.
                        evicted[-1].synchronize()
                except RuntimeError as exc:
                    # Failure is observable; this bucket is not retried on every
                    # camera request. The validated eager path stays available.
                    self._failed_keys.add(key)
                    self.capture_errors.append(str(exc)[:400])
                    self.fallback_calls += 1
                    return self.model(x, noise_labels=noise_labels, conditional_inputs=conditions, precomputed_embeds=precomputed_embeds)
            self._buckets.move_to_end(key)
            graph, static_x, static_labels, static_conditions, static_embeds, static_output, last_use = bucket
            stream = torch.cuda.current_stream(x.device)
            # The Python lock serialises submission. The event also protects
            # static buffers when different callers use different CUDA streams.
            stream.wait_event(last_use)
            static_x.copy_(x)
            static_labels.copy_(noise_labels)
            for target, value in zip(static_conditions, conditions):
                target.copy_(value)
            if static_embeds is not None:
                static_embeds.copy_(precomputed_embeds)
            graph.replay()
            self.replay_calls += 1
            output = static_output.clone()
            last_use.record(stream)
            return output

    def stats(self):
        return {'buckets': len(self._buckets), 'replay_calls': self.replay_calls,
                'capture_calls': self.capture_calls, 'fallback_calls': self.fallback_calls,
                'capture_errors': list(self.capture_errors)}

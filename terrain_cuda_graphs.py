"""Bounded CUDA graph buckets for eager Terrain Diffusion eval networks.

Opt-in only after per-GPU numerical/latency validation. No input padding, model
approximation or solver changes. The graph owns static buffers and callers get
an independent output clone; replaying another request cannot overwrite it.
"""
from collections import OrderedDict
import os
import threading

import torch
from terrain_profiling import span


class CudaGraphModel(torch.nn.Module):
    def __init__(self, model, *, max_buckets=4, max_batch=None, max_bytes=None):
        super().__init__()
        self.model = model
        self.max_buckets = max_buckets
        self.max_batch = max_batch
        self.max_bytes = max_bytes if max_bytes is not None else int(os.environ.get('TERRAIN_GRAPH_CACHE_MIB', '2048')) * 1024**2
        if self.max_bytes < 0 or max_buckets < 1:
            raise ValueError('Invalid CUDA graph cache budget')
        self._bucket_bytes = {}
        self.memory_fallbacks = 0
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
        self._bucket_bytes.clear()

    def _evict(self):
        key, bucket = self._buckets.popitem(last=False)
        bucket[-1].synchronize()
        self._bucket_bytes.pop(key, None)

    def _key(self, x, labels, conditions, embeds):
        return (tuple(x.shape), x.dtype, x.device,
                tuple(labels.shape), labels.dtype,
                tuple((tuple(t.shape), t.dtype, t.device) for t in conditions),
                (tuple(embeds.shape), embeds.dtype, embeds.device) if embeds is not None else None)

    def forward(self, x, noise_labels, conditional_inputs, return_logvar=False, precomputed_embeds=None):
        with span('nn.forward', gpu=True, shape=list(x.shape), model=type(self.model).__name__):
            return self._forward(x, noise_labels, conditional_inputs, return_logvar, precomputed_embeds)

    def _forward(self, x, noise_labels, conditional_inputs, return_logvar=False, precomputed_embeds=None):
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
                # Bound both shapes and measured private-pool allocation. Evict
                # before capture; a conservative first-shape estimate preserves
                # 1 GiB for the actual inference path. Runtime OOM still falls
                # back to eager, since external GPU consumers can race admission.
                estimate = max(64 * 1024**2, x.numel() * x.element_size() * 256)
                if self._bucket_bytes:
                    estimate = max(estimate, max(self._bucket_bytes.values()))
                while self._buckets and (len(self._buckets) >= self.max_buckets or
                        sum(self._bucket_bytes.values()) + estimate > self.max_bytes):
                    self._evict()
                free, _ = torch.cuda.mem_get_info(x.device)
                reclaimable = torch.cuda.memory_reserved(x.device)-torch.cuda.memory_allocated(x.device)
                if estimate > self.max_bytes or free+reclaimable < estimate+1024**3:
                    self.memory_fallbacks += 1
                    self.fallback_calls += 1
                    return self.model(x, noise_labels=noise_labels, conditional_inputs=conditions, precomputed_embeds=precomputed_embeds)
                static_x = static_labels = static_conditions = static_embeds = None
                graph = static_output = warm_reference = bucket = warm = last_use = None
                try:
                    allocated_before = torch.cuda.memory_allocated(x.device)
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
                    captured_bytes = max(0, torch.cuda.memory_allocated(x.device)-allocated_before)
                    if captured_bytes > self.max_bytes:
                        # Release the just-captured pool before running eager.
                        last_use.record()
                        last_use.synchronize()
                        del bucket, graph, static_x, static_labels, static_conditions, static_embeds, static_output, warm_reference
                        self.memory_fallbacks += 1
                        self._failed_keys.add(key)
                        self.fallback_calls += 1
                        return self.model(x, noise_labels=noise_labels, conditional_inputs=conditions, precomputed_embeds=precomputed_embeds)
                    while self._buckets and sum(self._bucket_bytes.values())+captured_bytes > self.max_bytes:
                        self._evict()
                    self._buckets[key] = bucket
                    self._bucket_bytes[key] = captured_bytes
                    self.capture_calls += 1
                except RuntimeError as exc:
                    # Failure is observable; this bucket is not retried on every
                    # camera request. The validated eager path stays available.
                    self._failed_keys.add(key)
                    self.capture_errors.append(str(exc)[:400])
                    # A partial graph/private pool must not survive into the
                    # eager OOM fallback through locals or exception frames.
                    if warm is not None:
                        warm.synchronize()
                    torch.cuda.current_stream(x.device).synchronize()
                    bucket = graph = static_x = static_labels = static_conditions = None
                    static_embeds = static_output = warm_reference = last_use = warm = None
                    exc.__traceback__ = None
                    torch.cuda.empty_cache()
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
        with self._lock:
            return {'buckets': len(self._buckets), 'replay_calls': self.replay_calls,
                    'pool_bytes': sum(self._bucket_bytes.values()), 'max_bytes': self.max_bytes,
                    'memory_fallbacks': self.memory_fallbacks,
                    'capture_calls': self.capture_calls, 'fallback_calls': self.fallback_calls,
                    'capture_errors': list(self.capture_errors)}

@torch.inference_mode()
def prewarm_base_forms(model, max_batch):
    """Capture real base input shapes once at startup; no terrain is generated.

    All tail batch sizes occur during navigation. Warming only batch16 leaves
    first-use captures (~300 ms each on the measured GPU) in foreground jobs.
    Capacity/free-memory guards remain those of CudaGraphModel.
    """
    import time
    if not isinstance(model,CudaGraphModel) or model.max_bytes==0:
        return dict(enabled=False,reason='graphs disabled or zero budget')
    specs=model.config.get('conditional_inputs',[])
    if len(specs)!=1 or specs[0][0]!='tensor':
        return dict(enabled=False,reason='unsupported base conditioning signature')
    parameter=next(model.parameters());device,dtype=parameter.device,parameter.dtype
    if device.type!='cuda':return dict(enabled=False,reason='not CUDA')
    limit=min(max_batch,model.max_batch or max_batch)
    started=time.perf_counter()
    expected_keys={}
    for batch in range(1,limit+1):
        x=torch.zeros((batch,model.config['in_channels'],64,64),device=device,dtype=dtype)
        labels=torch.ones(batch,device=device,dtype=dtype)
        conditions=[torch.zeros((batch,specs[0][1]),device=device,dtype=dtype)]
        expected_keys[batch]=model._key(x,labels,conditions,None)
        output=model(x,noise_labels=labels,conditional_inputs=conditions)
        del output,x,labels,conditions
    torch.cuda.synchronize(device)
    with model._lock:
        retained=[batch for batch,key in expected_keys.items() if key in model._buckets]
    return dict(enabled=True,requested_forms=limit,forms=len(retained),retained_batches=retained,
                fully_warmed=retained==list(range(1,limit+1)),
                seconds=round(time.perf_counter()-started,4),**model.stats())

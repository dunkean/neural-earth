"""Batch-one coarse stream pool, shared by serving and offline experiments.

Only the twenty-step solver is dispatched here. The caller prepares inputs and
owns the store: results are returned in input order, with a CUDA event join on
the caller's stream. Each slot has its own graph exec, buffers and scheduler.
"""
from dataclasses import dataclass
import threading

import torch

from terrain_coarse_graph import CoarseSolverAdapter
from terrain_cuda_graphs import CudaGraphModel


@dataclass(frozen=True)
class CoarseSolverInputs:
    sample: torch.Tensor
    labels: torch.Tensor
    conditions: tuple[torch.Tensor, ...]  # image, then scalar conditions
    embeds: torch.Tensor | None

    def tensors(self):
        return (self.sample, self.labels, *self.conditions,
                *((self.embeds,) if self.embeds is not None else ()))

    def validate(self, device):
        if self.sample.shape[0] != 1:
            raise ValueError('Each coarse stream must keep batch size 1')
        if self.labels.shape[0] != 20:
            raise ValueError('Coarse solver requires twenty noise labels')
        if not self.conditions or self.conditions[0].shape[0] != 1:
            raise ValueError('Missing batch-one coarse conditioning image')
        if any(t.device != device for t in self.tensors()):
            raise ValueError('All inputs must be on the pool device')


class CoarseStreamPool:
    """Explicitly prewarmed, bounded pool for one host submitter.

    Warmup/capture never runs while another slot is in flight. No silent eager
    fallback is accepted: that would invalidate the concurrency experiment.
    Input producers must use the caller's current stream (or join it first).
    """
    def __init__(self, model, scheduler, device, *, streams=1,
                 max_bytes=2 * 1024**3):
        if streams not in (1, 2, 4, 8, 16):
            raise ValueError('Use 1, 2, 4, 8 or 16 coarse streams')
        if max_bytes < 1:
            raise ValueError('Graph memory budget must be positive')
        self.device = torch.device(device)
        if self.device.type != 'cuda':
            raise ValueError('CoarseStreamPool requires CUDA')
        if self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.model = model
        self._lock = threading.Lock()
        self._closed = False
        self._ready = False
        self.slots = []
        for _ in range(streams):
            graph = CudaGraphModel(CoarseSolverAdapter(model, scheduler, self.device),
                                   max_buckets=1, max_batch=1,
                                   max_bytes=max_bytes // streams)
            self.slots.append((torch.cuda.Stream(device=self.device), graph))

    def _versions(self):
        return tuple((id(p), p._version) for p in self.model.parameters())

    @staticmethod
    def _call(graph, inputs):
        return graph(inputs.sample, noise_labels=inputs.labels,
                     conditional_inputs=list(inputs.conditions),
                     precomputed_embeds=inputs.embeds)

    @torch.inference_mode()
    def warm(self, inputs, expected):
        """Capture serially and require byte equality to the current solver."""
        with self._lock:
            if self._closed or self._ready:
                raise RuntimeError('Pool must be new before warmup')
            inputs.validate(self.device)
            caller = torch.cuda.current_stream(self.device)
            for stream, graph in self.slots:
                stream.wait_stream(caller)
                with torch.cuda.stream(stream):
                    actual = self._call(graph, inputs)
                stream.synchronize()
                if not graph._buckets or graph.fallback_calls:
                    raise RuntimeError(f'Coarse graph capture refused: {graph.stats()}')
                if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
                    raise RuntimeError('Stream graph differs from the reference solver')
            self._parameter_versions = self._versions()
            self._ready = True

    @torch.inference_mode()
    def run(self, inputs):
        """Submit at most one input per slot; join without blocking the host.

        Event durations measure whole solver calls (including copies/clone),
        not pure kernel time. All returned outputs are ordered and usable on
        the caller stream. The pool must outlive their GPU completion.
        """
        with self._lock:
            if self._closed or not self._ready:
                raise RuntimeError('Warm the pool before submission')
            if not 1 <= len(inputs) <= len(self.slots):
                raise ValueError('Submit between one and the slot count inputs')
            if self._versions() != self._parameter_versions:
                raise RuntimeError('Model weights changed; build a new pool')
            # Validate the whole group before enqueuing any work; recapture on
            # a new shape would synchronize other streams during submission.
            for item, (_, graph) in zip(inputs, self.slots):
                item.validate(self.device)
                key = graph._key(item.sample, item.labels, item.conditions, item.embeds)
                if key not in graph._buckets:
                    raise ValueError('Input shape/dtype was not prewarmed')
            caller = torch.cuda.current_stream(self.device)
            outputs, intervals = [], []
            for item, (stream, graph) in zip(inputs, self.slots):
                stream.wait_stream(caller)
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                with torch.cuda.stream(stream):
                    begin.record(stream)
                    output = self._call(graph, item)
                    end.record(stream)
                    for tensor in item.tensors():
                        tensor.record_stream(stream)
                outputs.append(output)
                intervals.append((begin, end))
            # Join only after all sibling submissions: joining inside the loop
            # would make the next slot depend on the previous solver.
            for output, (_, end) in zip(outputs, intervals):
                caller.wait_event(end)
                output.record_stream(caller)
            return outputs, intervals

    def stats(self):
        return dict(streams=len(self.slots), ready=self._ready, closed=self._closed,
                    slots=[graph.stats() for _, graph in self.slots])

    def close(self):
        with self._lock:
            if not self._closed:
                for stream, graph in self.slots:
                    stream.synchronize()
                    graph._clear_buckets()
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()

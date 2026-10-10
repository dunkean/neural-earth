"""Offline stage hooks for accepted-candidate evaluation, with unchanged upstream."""
from __future__ import annotations

from collections import deque
from types import MethodType
import threading

import torch
import torch.nn.functional as F
from infinite_tensor import InfiniteTensor, TensorWindow

from distill.features import COND_MEANS, COND_STDS, phase
from distill.student import load_student
from terrain_noise import noise_patch


def normalized(value):
    return (value[:-1]/value[-1:].clamp_min(1e-12)).float()


def noise(seed, y, x, h, w, channels, tile=64):
    """Same bytes as ``features.noise``, from the shared tile cache."""
    return torch.from_numpy(noise_patch(int(seed), int(y), int(x), h, w,
                                        channels=channels, tile_h=tile, tile_w=tile))


def upload(value, device):
    """Host-to-device copy that does not wait for queued GPU work.

    A pageable copy synchronises the stream, so the CPU could not prepare the
    next window while the GPU computes this one. Pinned staging is recycled by
    PyTorch's host allocator only after the copy has completed.
    """
    if torch.device(device).type != 'cuda':
        return value.to(device)
    return value.pin_memory().to(device, non_blocking=True)


def base_inputs(coarse, coarse_y, coarse_x, seed, y, x, size, histogram, device):
    """Assemble the CPU feature schema directly on the inference device."""
    if y % 32 or x % 32 or size % 32:
        raise ValueError('Base inputs must align to the 32-latent coarse grid.')
    cy, cx = y//32-int(coarse_y), x//32-int(coarse_x)
    field = torch.as_tensor(coarse, dtype=torch.float32)[:, cy:cy+size//32, cx:cx+size//32]
    if field.shape != (6, size//32, size//32):
        raise ValueError('Stored conditioning does not cover the requested halo.')
    # Retain the original CPU normalization and trigonometric arithmetic.
    field = (field-torch.from_numpy(COND_MEANS)[:, None, None])/torch.from_numpy(COND_STDS)[:, None, None]
    inputs = torch.zeros(32, size, size, device=device, dtype=torch.float32)
    inputs[:5].copy_(upload(noise(seed+5819, y, x, size, size, 5), device))
    inputs[5:10].copy_(upload(noise(seed+5820, y, x, size, size, 5), device))
    inputs[10:16].copy_(F.interpolate(upload(field[None], device), size=(size, size), mode='nearest')[0])
    inputs[16].fill_((1-.66)/.47)
    inputs[17:21].copy_(upload(phase(y, x, size, size), device))
    inputs[21:26].copy_(upload(torch.as_tensor(histogram, dtype=torch.float32), device)[:, None, None])
    return inputs


def decoder_inputs(latents, seed, y, x, size, device):
    inputs = torch.zeros(16, size, size, device=device, dtype=torch.float32)
    inputs[:1].copy_(upload(noise(seed+5819, y, x, size, size, 1, tile=size), device))
    fields = torch.as_tensor(latents, dtype=torch.float32, device=device)
    inputs[1:5].copy_(F.interpolate(fields[None], size=(size, size), mode='nearest')[0])
    return inputs


def coarse_inputs(condition, labels, seed, y, x, device):
    condition = torch.as_tensor(condition, dtype=torch.float32, device=device)
    h, w = condition.shape[-2:]
    inputs = torch.zeros(16, h, w, device=device, dtype=torch.float32)
    inputs[:6].copy_(upload(noise(seed+1, y, x, h, w, 6), device))
    inputs[6:11].copy_(condition)
    inputs[11:].copy_(torch.as_tensor(labels, dtype=torch.float32, device=device)[:, None, None])
    return inputs


def autocast(stage):
    # The coarse solver disables autocast itself. No cast cache: a graph must
    # not capture cast weights owned by an autocast region that later exits.
    return torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False)


class StudentGraphs:
    """Batch-one CUDA graphs of one student, one slot per CUDA stream.

    Every slot is captured separately and must equal its eager output byte for
    byte, so a window gives the same result in any slot. Slots of a group run
    concurrently: small coarse windows cannot fill the GPU one at a time.
    """
    def __init__(self, model, stage, example, slots):
        self.key = self.key_for(model, example, slots)
        self.slots = []
        device = example.device
        caller = torch.cuda.current_stream(device)
        for _ in range(slots):
            stream = torch.cuda.Stream(device=device)
            static = example.clone()
            stream.wait_stream(caller)
            with autocast(stage), torch.cuda.stream(stream):
                for _ in range(3):
                    reference = model(static)
            caller.wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with autocast(stage), torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
                output = model(static)
            graph.replay()
            if not torch.equal(reference, output):
                raise RuntimeError(f'{stage} student graph differs from its eager output.')
            self.slots.append((stream, graph, static, output))

    @staticmethod
    def key_for(model, example, slots):
        return (tuple(example.shape), tuple(example.stride()), example.dtype, example.device, slots,
                tuple((id(p), p._version) for p in model.parameters()))

    def run(self, inputs):
        """At most one input per slot; outputs are ready on the caller stream."""
        caller = torch.cuda.current_stream(inputs[0].device)
        outputs = []
        for value, (stream, graph, static, output) in zip(inputs, self.slots):
            stream.wait_stream(caller)
            with torch.cuda.stream(stream):
                static.copy_(value)
                graph.replay()
                result = output.clone()
            value.record_stream(stream)
            outputs.append(result)
        for result, (stream, *_) in zip(outputs, self.slots):
            caller.wait_stream(stream)
            result.record_stream(caller)
        return outputs

    def close(self):
        for stream, *_ in self.slots:
            stream.synchronize()
        self.slots.clear()


def graph_lock(model):
    return model.__dict__.setdefault('_distill_graph_lock', threading.RLock())


def student_graphs(model, stage, example, slots):
    """Share one bounded capture per student across worlds borrowing it."""
    with graph_lock(model):
        key = StudentGraphs.key_for(model, example, slots)
        cached = model.__dict__.get('_distill_graphs')
        if cached is None or cached.key != key:
            if cached is not None:
                cached.close()
            model.__dict__.pop('_distill_graphs', None)
            model.__dict__['_distill_graphs'] = StudentGraphs(model, stage, example, slots)
        return model.__dict__['_distill_graphs']


def run_graphs(model, stage, inputs, slots):
    outputs = []
    with graph_lock(model):
        for offset in range(0, len(inputs), slots):
            group = inputs[offset:offset+slots]
            outputs += student_graphs(model, stage, group[0], slots).run(group)
    return outputs


def coarse_graph_forward(model, inputs):
    """Single-slot whole-solver replay, as checked by check_coarse_capture."""
    if not inputs.is_cuda or torch.is_grad_enabled():
        return model(inputs)
    return run_graphs(model, 'coarse', [inputs], 1)[0]


class FiniteGuard:
    """Non-finite student outputs are refused without a per-window host sync.

    Each flag is copied to pinned memory and read once its event completed;
    a failure surfaces on a later window of the same world, never silently.
    """
    def __init__(self, limit=16):
        self.pending = deque()
        self.limit = limit

    def add(self, stage, outputs):
        flag = torch.stack([torch.isfinite(value).all() for value in outputs]).all()
        if not flag.is_cuda:
            if not flag:
                raise FloatingPointError(f'Non-finite {stage} student output.')
            return
        host = flag.to('cpu', non_blocking=True)
        event = torch.cuda.Event()
        event.record()
        self.pending.append((stage, host, event))
        while self.pending and (len(self.pending) > self.limit or self.pending[0][2].query()):
            stage, host, event = self.pending.popleft()
            event.synchronize()
            if not host.item():
                self.pending.clear()
                raise FloatingPointError(f'Non-finite {stage} student output.')


def prepare_students(models, device, profile=None):
    """One-time placement of shared students: solver constants, conv layout."""
    for stage, model in models.items():
        if model.__dict__.get('_distill_prepared') == str(device):
            continue
        if stage == 'coarse' and model.config.solver_steps and torch.device(device).type == 'cuda':
            model.solver.prepare_capture(exact_kernels=getattr(profile, 'exact_kernels', False),
                                         attention_backend=getattr(profile, 'attention_backend', 'reference'))
        if not model.config.solver_steps:
            # Convolutional students run 1.2-1.3x faster in channels_last; BF16
            # error to the FP32 network is unchanged (decoder RMS 8.9e-4).
            model.to(memory_format=torch.channels_last)
        model.__dict__['_distill_prepared'] = str(device)
    return models


def graphs_enabled(stage, device, profile):
    if torch.device(device).type != 'cuda' or not getattr(profile, 'cuda_graphs', False):
        return False
    return getattr(profile, 'coarse_solver_graphs' if stage == 'coarse' else 'decoder_graphs', False)


def example_input(stage, device, size):
    channels = dict(base=32, coarse=16, decoder=16)[stage]
    value = torch.zeros(1, channels, size, size, device=device)
    return value if stage == 'coarse' else value.contiguous(memory_format=torch.channels_last)


@torch.inference_mode()
def prewarm(models, device, profile, *, decoder_size=512, coarse_streams=1):
    """Capture the graphs the selected students will replay, before the first view."""
    import time
    prepare_students(models, device, profile)
    results = {}
    for stage, model in models.items():
        if stage == 'base' or not graphs_enabled(stage, device, profile):
            continue
        started = time.perf_counter()
        slots = coarse_streams if stage == 'coarse' else 1
        student_graphs(model, stage, example_input(stage, device, 64 if stage == 'coarse' else decoder_size), slots)
        torch.cuda.synchronize(device)
        results[stage] = dict(enabled=True, fully_warmed=True, student=True, streams=slots,
                              seconds=round(time.perf_counter()-started, 4))
    return results


def install(world, checkpoints, tile_size=512):
    """Install before bind. Mapping is stage -> checkpoint; combinations supported."""
    if tile_size < 32 or tile_size % 32:
        raise ValueError('Base tile size must align to 32.')
    students = {}
    for stage, path in checkpoints.items():
        if isinstance(path, torch.nn.Module):
            model = path
        else:
            model, checkpoint = load_student(path, world.device)
        if model.config.stage != stage:
            raise ValueError(f'{path}: expected {stage}, found {model.config.stage}.')
        students[stage] = model
    profile = getattr(world, '_terrain_profile', None)
    prepare_students(students, world.device, profile)
    world._distill_students = students
    world._distill_counts = {stage: dict(calls=0, output_pixels=0) for stage in students}
    guard = FiniteGuard()

    def forward(stage, batch, slots=1):
        """Batch-one student calls; graph slots run up to ``slots`` concurrently."""
        model = students[stage]
        inputs = [value[None].to(world.device) for value in batch]
        if stage != 'coarse':
            inputs = [value.contiguous(memory_format=torch.channels_last) for value in inputs]
        if graphs_enabled(stage, world.device, profile) and not torch.is_grad_enabled():
            outputs = run_graphs(model, stage, inputs, slots)
        else:
            with autocast(stage):
                outputs = [model(value) for value in inputs]
        outputs = [value[0].float() for value in outputs]
        guard.add(stage, outputs)
        world._distill_counts[stage]['calls'] += len(outputs)
        return outputs

    if 'base' in students:
        halo = students['base'].halo
        total = tile_size+2*halo

        def build_base(self):
            def predict(ctxs, conditions):
                outputs = []
                for (_, iy, ix), condition in zip(ctxs, conditions):
                    y, x = iy*tile_size-halo, ix*tile_size-halo
                    field = normalized(condition).cpu().numpy()
                    features = base_inputs(field, y//32, x//32, self.seed, y, x,
                                           total, self.kwargs['histogram_raw'], self.device)
                    pred = forward('base', [features])[0][:, halo:halo+tile_size, halo:halo+tile_size]
                    self._distill_counts['base']['output_pixels'] += tile_size**2
                    outputs.append(torch.cat([pred, pred.new_ones(1, tile_size, tile_size)]))
                return outputs
            return InfiniteTensor(shape=(6, None, None), f=predict,
                output_window=TensorWindow(size=(6, tile_size, tile_size)), args=(self.coarse,),
                args_windows=(TensorWindow(size=(7, total//32, total//32),
                    stride=(7, tile_size//32, tile_size//32), offset=(0, -halo//32, -halo//32)),),
                device=self.device, tile_store=self.tile_store, batch_size=1,
                tensor_id='distill_final_latent')
        world._build_latent_stage = MethodType(build_base, world)

    if 'coarse' in students:
        original = world._build_coarse_stage

        def build_coarse(self):
            from distill.teacher import physical_coarse_scaling
            from terrain_inference import _coarse_batch, coarse_stream_count
            from terrain_diffusion.inference.world_pipeline import linear_weight_window
            from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
            tensor = original()
            if self.kwargs['coarse_pooling'] != 1:
                raise ValueError('Coarse student currently requires pool=1.')
            scheduler = EDMDPMSolverMultistepScheduler(sigma_min=.002, sigma_max=80, sigma_data=.5)
            weight = linear_weight_window(64, 'cpu', torch.float32).to(self.device)
            t_cond = torch.atan(torch.tensor(self.kwargs['cond_snr'])).to(self.device, dtype=torch.bfloat16)
            labels = [v.detach().view(-1) for v in torch.log(torch.tan(t_cond)/8)]
            means, stds = physical_coarse_scaling(self)
            means = torch.as_tensor(means, device=self.device)[:, None, None]
            stds = torch.as_tensor(stds, device=self.device)[:, None, None]

            def predict(ctxs):
                features = []
                for ctx in ctxs:
                    prepared = _coarse_batch(self, [ctx], scheduler, weight, t_cond, labels, 1, prepare_only=True)
                    condition = prepared.conditions[0][0]
                    scalars = torch.stack([v[0] for v in prepared.conditions[1:]])
                    features.append(coarse_inputs(condition, scalars, self.seed, ctx[1]*48, ctx[2]*48, self.device))
                # The world's stream count groups windows; each stays batch-one.
                preds = forward('coarse', features, slots=max(1, coarse_stream_count(self)))
                self._distill_counts['coarse']['output_pixels'] += 64**2*len(preds)
                return [torch.cat([(pred*stds+means)*weight[None], weight[None]]) for pred in preds]
            tensor._f = predict
            return tensor
        world._build_coarse_stage = MethodType(build_coarse, world)

    if 'decoder' in students:
        def decoder_windows(ctxs, latents, weight, size, stride, noise_seed=None):
            """Every decoder window of this world, refinement octaves included."""
            seed = world.seed if noise_seed is None else noise_seed
            weight = weight.to(world.device)
            features = [decoder_inputs(normalized(latent)[:4], seed, ctx[1]*stride, ctx[2]*stride,
                                       size, world.device) for ctx, latent in zip(ctxs, latents)]
            preds = forward('decoder', features)
            world._distill_counts['decoder']['output_pixels'] += size**2*len(preds)
            return [torch.cat([pred*weight[None], weight[None]]) for pred in preds]
        # terrain_inference._decoder_batch dispatches here (refinement too);
        # the stage override also covers worlds built by the upstream pipeline.
        world._terrain_decoder_windows = decoder_windows
        original_decoder = world._build_decoder_stage

        def build_decoder(self):
            from terrain_diffusion.inference.world_pipeline import linear_weight_window
            tensor = original_decoder()
            size, stride = self.decoder_tile_size, self.decoder_tile_stride
            weight = linear_weight_window(size, 'cpu', torch.float32).to(self.device)
            tensor._f = lambda ctxs, latents: decoder_windows(ctxs, latents, weight, size, stride)
            return tensor
        world._build_decoder_stage = MethodType(build_decoder, world)
    return world

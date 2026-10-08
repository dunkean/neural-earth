"""Local, reversible optimisations of the unchanged Terrain Diffusion networks.

No checkpoint or upstream source is modified. Natural conditioning, solver,
seeds, window geometry and BF16 precision remain those of WorldPipeline. The
optional Earth profile supplies continental and latitude-aware climate inputs;
its physically nonnegative climate projection is explicit and versioned. FP32
CUDA window caches retain upstream accumulation order. Configure before bind.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import os
from types import MethodType
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from infinite_tensor import InfiniteTensor, TensorWindow
from terrain_diffusion.inference.world_pipeline import (
    WorldPipeline, gaussian_noise_patch, linear_weight_window,
)
from terrain_diffusion.models.mp_layers import MPConv, MPConvResample, normalize
from terrain_diffusion.models import mp_layers
from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from terrain_diffusion.inference.synthetic_map import make_synthetic_map_factory

VERSION = 'cuda-resident-window-v2'


@dataclass(frozen=True)
class InferenceProfile:
    name: str
    coarse_batch: int = 1
    latent_batch: int = 16
    decoder_batch: int = 1
    cached_weights: bool = True
    cached_coarse_embeddings: bool = True
    scalar_sync_removed: bool = True
    batched_latent_transfer: bool = True
    gpu_windows: bool = True
    cuda_graphs: bool = True
    graph_max_base_batch: int = 16
    decoder_graphs: bool = True
    coarse_solver_graphs: bool = True
    canonical_latents: bool = False
    exact_kernels: bool = True
    attention_backend: str = 'reference'


def choose_profile(device=None) -> InferenceProfile:
    """Conservative defaults; explicit overrides allow measured GPU tuning.

    The BF16 latent batch is numerical state, so free VRAM cannot select it.
    Defaults stay at the validated scalar coarse/decoder and latent-16 path.
    Actual CUDA allocation decides admission when memory is scarce.
    """
    device = device or torch.cuda.current_device()
    free, _ = torch.cuda.mem_get_info(device)
    gib = free / 1024**3
    def batch(name, default, maximum):
        value = int(os.environ.get(name, default))
        if not 1 <= value <= maximum:
            raise ValueError(f'{name} must be between 1 and {maximum}')
        return value
    def flag(name, default):
        value = os.environ.get(name, '1' if default else '0')
        if value not in ('0', '1'):
            raise ValueError(f'{name} must be 0 or 1')
        return value == '1'
    graph_flag = os.environ.get('TERRAIN_CUDA_GRAPHS', '1')
    if graph_flag not in ('0', '1'):
        raise ValueError('TERRAIN_CUDA_GRAPHS must be 0 or 1')
    canonical_flag = os.environ.get('TERRAIN_CANONICAL_LATENTS', '0')
    if canonical_flag not in ('0', '1'):
        raise ValueError('TERRAIN_CANONICAL_LATENTS must be 0 or 1')
    canonical_latents = canonical_flag == '1'
    latent_batch = batch('TERRAIN_LATENT_BATCH', 1 if canonical_latents else 16, 32)
    if canonical_latents and latent_batch != 1:
        raise ValueError('Canonical latents require TERRAIN_LATENT_BATCH=1')
    # A transient VRAM load must not silently select a different world.
    return InferenceProfile(
        name=f'cuda-{torch.cuda.get_device_capability(device)[0]}-{int(gib)}g',
        coarse_batch=batch('TERRAIN_COARSE_BATCH', 1, 1),
        latent_batch=latent_batch,
        decoder_batch=batch('TERRAIN_DECODER_BATCH', 1, 1),
        cuda_graphs=graph_flag == '1',
        graph_max_base_batch=batch('TERRAIN_GRAPH_MAX_BASE_BATCH', 16, 32),
        decoder_graphs=flag('TERRAIN_DECODER_GRAPHS', True),
        coarse_solver_graphs=flag('TERRAIN_COARSE_SOLVER_GRAPHS', True),
        canonical_latents=canonical_latents,
        exact_kernels=flag('TERRAIN_EXACT_KERNELS', True),
        attention_backend=os.environ.get('TERRAIN_ATTENTION_BACKEND', 'reference'),
    )


def _gain_key(gain):
    if isinstance(gain, torch.Tensor):
        return ('tensor', id(gain), gain._version, gain.device, gain.dtype)
    return ('scalar', gain)


def _cached_weight(module, x, gain):
    key = (module.weight._version, module.weight.device, module.weight.dtype,
           x.dtype, _gain_key(gain))
    cached = module.__dict__.get('_terrain_weight_cache')
    if cached is None or cached[0] != key:
        # Preserve *the complete original expression and rounding order*, even
        # when gain is a learned scalar. Scaling a rounded BF16 weight afterwards
        # would not be equivalent.
        w = normalize(module.weight.to(torch.float32))
        w = (w * (gain / np.sqrt(w[0].numel()))).to(x.dtype)
        module.__dict__['_terrain_weight_cache'] = (key, w)
    return module.__dict__['_terrain_weight_cache'][1]


def _mp_forward(self, x, gain=1):
    if self.training or torch.is_grad_enabled():
        return self.__dict__['_terrain_original_forward'](x, gain)
    w = _cached_weight(self, x, gain)
    if isinstance(self, MPConvResample):
        upsampled = mp_layers.resample(x, mode=self.resample_mode, factor=self.stride)
        if self.resample_mode == 'down':
            y = F.conv2d(x, w, stride=self.stride, padding=0)
        else:
            y = F.conv_transpose2d(x, w, stride=self.stride, padding=0)
        return mp_layers.mp_sum([y, upsampled], w=self.skip_weight)
    if w.ndim == 2:
        return F.linear(x, w)
    return F.conv2d(x, w, padding=(0 if self.no_padding else w.shape[-1] // 2,), groups=self.groups)


def prepare_models(world):
    """Reuse eval weights; invalidate automatically on parameter version changes."""
    count = 0
    for model in (world.coarse_model, world.base_model, world.decoder_model):
        model.eval()
        for module in model.modules():
            if isinstance(module, (MPConv, MPConvResample)):
                if '_terrain_original_forward' not in module.__dict__:
                    module.__dict__['_terrain_original_forward'] = module.forward
                    module.forward = MethodType(_mp_forward, module)
                count += 1
    return count


def restore_models(world):
    """Reference harness can restore the unchanged upstream forwards."""
    for model in (world.coarse_model, world.base_model, world.decoder_model):
        for module in model.modules():
            original = module.__dict__.pop('_terrain_original_forward', None)
            if original is not None:
                module.forward = original
            module.__dict__.pop('_terrain_weight_cache', None)


def _coarse_batch(world, ctxs, scheduler, weight_window, t_cond, cond_inputs, pool_size):
    dtype = world._dtype or torch.float32
    means = torch.tensor(world.kwargs['coarse_means'])
    stds = torch.tensor(world.kwargs['coarse_stds'])
    synthetic, cond_noises, sample_noises = [], [], []
    regional = None
    embed_key = tuple((id(p), p._version) for p in world.coarse_model.parameters())
    if world.__dict__.get('_terrain_coarse_embed_key') != embed_key:
        world.__dict__.pop('_terrain_coarse_embeds', None)
        world.__dict__.pop('_terrain_snr_cache', None)
        world.__dict__['_terrain_coarse_embed_key'] = embed_key
    for _, i, j in ctxs:
        i1, j1 = i * (48 // pool_size) * pool_size, j * (48 // pool_size) * pool_size
        value = world._conditioning_model_input(i1, i1 + 64, j1, j1 + 64)
        if world._terrain_snr_active:
            from terrain_snr import window_snr
            from terrain_geometry import world_latitude
            chart_latitude = getattr(world.synthetic_map_factory, 'latitude', None)
            lat = (chart_latitude((j1 + 32) * 7680, (i1 + 32) * 7680) if chart_latitude else
                   float(world_latitude((i1 + 32) * 7680, world._terrain_generation_settings)))
            regional = window_snr(world._terrain_generation_settings, world.kwargs['cond_snr'], value,
                                  latitude=lat)
        synthetic.append((value - means[[0, 2, 3, 4, 5], None, None]) / stds[[0, 2, 3, 4, 5], None, None])
        cond_noises.append(gaussian_noise_patch(world.seed, i1, j1, 64, 64, channels=5, tile_h=64, tile_w=64))
        sample_noises.append(gaussian_noise_patch(world.seed + 1, i1, j1, 64, 64, channels=6, tile_h=64, tile_w=64))
    synthetic = torch.stack(synthetic).to(world.device, dtype=dtype)
    cond_noise = torch.from_numpy(np.stack(cond_noises)).to(world.device, dtype=dtype)
    if regional is not None:
        # Production coarse batches remain scalar. Cache all conditioning and
        # timestep embeddings per quantized regional policy, bounded per world.
        if len(ctxs) != 1:
            raise ValueError('Regional SNR currently requires scalar coarse batches')
        cache = world.__dict__.setdefault('_terrain_snr_cache', OrderedDict())
        policy = regional[0]
        cached = cache.get(policy)
        if cached is None:
            regional_t = torch.atan(torch.tensor(policy)).to(world.device, dtype=dtype)
            regional_inputs = [v.detach().view(-1) for v in torch.log(torch.tan(regional_t)/8)]
            cached = [regional_t, regional_inputs, None]
            cache[policy] = cached
            while len(cache) > 64:
                cache.popitem(last=False)
        cache.move_to_end(policy)
        t_cond, cond_inputs = cached[:2]
    t_view = t_cond.view(1, -1, 1, 1)
    cond_img = torch.cos(t_view) * synthetic + torch.sin(t_view) * cond_noise
    scheduler.set_timesteps(20)
    sample = torch.from_numpy(np.stack(sample_noises)).to(world.device, dtype=dtype) * scheduler.sigmas[0]

    # The original cnoise.item() waits for CUDA on *every* solver step. Compute
    # labels with the same CUDA operation and pass tensors without a host read.
    scalar_labels = world.__dict__.get('_terrain_coarse_labels')
    if scalar_labels is None:
        scalar_labels = [scheduler.trigflow_precondition_noise(sigma.to(world.device).view(-1)).to(dtype)
                         for sigma in scheduler.sigmas[:-1]]
        world.__dict__['_terrain_coarse_labels'] = scalar_labels
    embeds = cached[2] if regional is not None else world.__dict__.get('_terrain_coarse_embeds')
    if embeds is None and world._terrain_profile.cached_coarse_embeddings:
        embeds = [world.coarse_model.compute_embeddings(label, cond_inputs) for label in scalar_labels]
        if regional is not None:
            cached[2] = embeds
        else:
            world.__dict__['_terrain_coarse_embeds'] = embeds
    n = len(ctxs)
    conditions = [value.expand(n) for value in cond_inputs]
    if world._terrain_profile.cuda_graphs and world._terrain_profile.coarse_solver_graphs:
        from terrain_coarse_graph import run_coarse_solver
        sample = run_coarse_solver(world, scheduler, sample, cond_img, scalar_labels, conditions, embeds)
    else:
        for idx, (t, sigma) in enumerate(zip(scheduler.timesteps, scheduler.sigmas)):
            sigma = sigma.to(world.device)
            scaled = scheduler.precondition_inputs(sample, sigma)
            model_in = torch.cat([scaled, cond_img], dim=1).to(dtype)
            model_out = world.coarse_model(
                model_in, noise_labels=scalar_labels[idx].expand(n), conditional_inputs=conditions,
                precomputed_embeds=embeds[idx].expand(n, -1) if embeds is not None else None,
            )
            # CPU t is sufficient for the scheduler's step index and avoids its
            # initial CUDA->CPU scalar transfer. All sample arithmetic remains CUDA.
            sample = scheduler.step(model_out, t, sample).prev_sample
    sample = (sample if world._terrain_profile.gpu_windows else sample.cpu()).float() / scheduler.config.sigma_data
    stds, means = stds.to(sample.device), means.to(sample.device)
    sample = sample * stds.view(1, -1, 1, 1) + means.view(1, -1, 1, 1)
    sample[:, 1] = sample[:, 0] - sample[:, 1]
    outputs = []
    for value in sample:
        if pool_size > 1:
            value = world._pool_coarse_conditioning(value, pool_size)
        outputs.append(torch.cat([value * weight_window[None], weight_window[None]], dim=0))
    return outputs


def _build_coarse(self):
    from terrain_coarse_graph import clear_coarse_solver
    clear_coarse_solver(self)
    pool = self.kwargs['coarse_pooling']
    scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, sigma_max=80, sigma_data=0.5)
    cache_device = self.device if self._terrain_profile.gpu_windows else 'cpu'
    weight = linear_weight_window(64 // pool, 'cpu', torch.float32).to(cache_device)
    t_cond = torch.atan(torch.tensor(self.kwargs['cond_snr'])).to(self.device, dtype=self._dtype or torch.float32)
    cond_inputs = [v.detach().view(-1) for v in torch.log(torch.tan(t_cond) / 8)]
    # Rebuild invalidates embeddings when generation conditioning changes.
    self.__dict__.pop('_terrain_coarse_embeds', None)
    self.__dict__.pop('_terrain_coarse_labels', None)
    self.__dict__.pop('_terrain_snr_cache', None)
    return InfiniteTensor(
        shape=(7, None, None),
        f=lambda ctxs: _coarse_batch(self, ctxs, scheduler, weight, t_cond, cond_inputs, pool),
        output_window=TensorWindow(size=(7, 64 // pool, 64 // pool), stride=(7, 48 // pool, 48 // pool)),
        batch_size=self._terrain_profile.coarse_batch, device=cache_device,
        tile_store=self.tile_store, tensor_id='base_coarse_map',
    )


def _latent_inference(self, ctxs, samples, cond_imgs, t, scheduler, weight_window,
                      histogram_raw, cond_means, cond_stds, seed_offset=0):
    if not ctxs:
        return []
    dtype = self._dtype or torch.float32
    if self._terrain_profile.gpu_windows:
        return _latent_gpu_batch(self, ctxs, samples, cond_imgs, t, scheduler,
                                 weight_window, histogram_raw, cond_means, cond_stds, seed_offset)
    samples = samples if samples is not None else [None] * len(ctxs)
    t_tensor = torch.as_tensor(t, dtype=dtype, device=self.device)
    t_view = t_tensor.view(1, 1, 1, 1)
    model_inputs, conditions, noisy_samples = [], [], []
    for ctx, sample, cond in zip(ctxs, samples, cond_imgs):
        if sample is None:
            sample = torch.zeros((1, 5, 64, 64), device=self.device, dtype=dtype)
        else:
            sample = torch.as_tensor(sample, device=self.device, dtype=dtype)
            sample = sample[:-1] / sample[-1:] * scheduler.config.sigma_data
        cond = cond[:-1] / cond[-1:]
        cond = torch.cat([cond, torch.ones((1, 4, 4), device=cond.device)], dim=0)[None]
        processed = self._process_latent_conditioning(
            cond, histogram_raw, cond_means, cond_stds, torch.tensor(0.0),
            seed_offset=ctx[1] * 65536 + ctx[2])
        conditions.append(processed.to(self.device, dtype=dtype))
        noise = torch.from_numpy(gaussian_noise_patch(
            self.seed + seed_offset, ctx[1] * 32, ctx[2] * 32, 64, 64,
            channels=5, tile_h=64, tile_w=64,
        ))[None].to(self.device, dtype=dtype)
        z = noise * scheduler.config.sigma_data
        x_t = torch.cos(t_view) * sample + torch.sin(t_view) * z
        model_inputs.append(x_t / scheduler.config.sigma_data)
        noisy_samples.append(x_t)
    n = len(ctxs)
    pred = -self.base_model(torch.cat(model_inputs), noise_labels=t_tensor.expand(n),
                            conditional_inputs=[torch.cat(conditions)])
    sample = torch.cos(t_view) * torch.cat(noisy_samples) - torch.sin(t_view) * scheduler.config.sigma_data * pred
    # One D2H transfer per batch instead of one synchronising transfer per window.
    sample = sample.cpu().float() / scheduler.config.sigma_data
    return [torch.cat([value * weight_window[None], weight_window[None]], dim=0) for value in sample]


def _latent_gpu_batch(world, ctxs, samples, cond_imgs, t, scheduler, weight,
                      histogram, means, stds, seed_offset):
    """Prepare the unchanged actual base batch with one compact H2D copy.

    InfiniteTensor still selects every context and every batch boundary. All
    pointwise operations retain their original precision and expression order;
    the climate reduction still covers each individual 2x2 crop. In particular
    the upstream batch-one NaN replacement is applied to every window, rather
    than interpreting its old batch-axis indexing on the combined batch.
    """
    dtype = world._dtype or torch.float32
    n = len(ctxs)
    cache = world.__dict__.setdefault('_terrain_latent_constants', {})
    key = id(t)
    constants = cache.get(key)
    if constants is None:
        t_tensor = torch.as_tensor(t, dtype=dtype, device=world.device)
        view = t_tensor.view(1, 1, 1, 1)
        constants = (t_tensor, torch.cos(view), torch.sin(view),
                     means.to(world.device).view(1, -1, 1, 1),
                     stds.to(world.device).view(1, -1, 1, 1),
                     histogram.to(world.device),
                     torch.tensor([-0.5*np.sqrt(12)], device=world.device, dtype=torch.float32).view(1, 1),
                     weight.to(world.device), float(means[0]))
        cache[key] = constants
    labels, cosine, sine, means_gpu, stds_gpu, histogram_gpu, noise_level, weight_gpu, nan_fill = constants
    if samples is None:
        sample = torch.zeros((n, 5, 64, 64), device=world.device, dtype=dtype)
    else:
        sample = torch.stack([value.to(device=world.device, dtype=dtype) for value in samples])
        sample = sample[:, :-1] / sample[:, -1:] * scheduler.config.sigma_data
    cond = torch.stack(cond_imgs)
    cond = cond[:, :-1] / cond[:, -1:]
    cond = torch.cat([cond, torch.ones((n, 1, 4, 4), device=cond.device)], dim=1)
    cond = ((cond-means_gpu)/stds_gpu).nan_to_num(nan_fill)
    processed = mp_layers.mp_concat([
        cond[:, 0:1].flatten(1), cond[:, 1:2].flatten(1),
        cond[:, 2:6, 1:3, 1:3].mean(dim=(2, 3)), cond[:, 6:7].flatten(1),
        histogram_gpu.expand(n, -1), noise_level.expand(n, -1),
    ], dim=1).float().to(dtype)
    # The same portable noise patches in the same window order are assembled
    # on the host first. This replaces sixteen synchronising pageable copies.
    noise = torch.from_numpy(np.stack([gaussian_noise_patch(
        world.seed+seed_offset, ctx[1]*32, ctx[2]*32, 64, 64,
        channels=5, tile_h=64, tile_w=64) for ctx in ctxs])).to(world.device, dtype=dtype)
    z = noise*scheduler.config.sigma_data
    x_t = cosine*sample+sine*z
    pred = -world.base_model(x_t/scheduler.config.sigma_data, noise_labels=labels.expand(n),
                             conditional_inputs=[processed])
    sample = (cosine*x_t-sine*scheduler.config.sigma_data*pred).float()/scheduler.config.sigma_data
    return [torch.cat([value*weight_gpu[None], weight_gpu[None]], dim=0) for value in sample]


def _build_latent(self):
    # Keep the upstream T=2 dependency graph and fusion barriers unchanged.
    self.__dict__.pop('_terrain_latent_constants', None)
    tensor = WorldPipeline._build_latent_stage(self)
    if self._terrain_profile.gpu_windows:
        def move(node):
            for arg in node.args:
                move(arg)
            node.to(self.device)
        move(tensor)
        # Upstream closures capture a CPU weight window. _latent_inference
        # explicitly moves this tiny fixed window, not any neural output.
    return tensor


def _decoder_batch(world, ctxs, latents, scheduler, weight, t_list, size, stride, *, noise_seed=None):
    dtype = world._dtype or torch.float32
    lc = world.latent_compression
    normalised = [(value[:-1] / value[-1:])[:4].view(1, 4, size // lc, size // lc) for value in latents]
    # Transfer compact 64² latents then expand; the original transfers 512².
    upsampled = F.interpolate(torch.cat(normalised).to(world.device, dtype=dtype),
                              size=(size, size), mode='nearest')
    n = len(ctxs)
    sample = torch.zeros((n, 1, size, size), device=world.device, dtype=dtype)
    for idx, t in enumerate(t_list):
        noises = [gaussian_noise_patch((world.seed if noise_seed is None else noise_seed) + 5819 + idx, ctx[1] * stride, ctx[2] * stride,
                                     size, size, channels=1, tile_h=size, tile_w=size) for ctx in ctxs]
        noise = torch.from_numpy(np.stack(noises)).to(world.device, dtype=dtype)
        t = t.view(1, 1, 1, 1).to(world.device, dtype=dtype)
        z = noise * scheduler.config.sigma_data
        x_t = torch.cos(t) * sample + torch.sin(t) * z
        model_in = torch.cat([x_t / scheduler.config.sigma_data, upsampled], dim=1)
        pred = -world.decoder_model(model_in, noise_labels=t.reshape(1).expand(n), conditional_inputs=[])
        sample = torch.cos(t) * x_t - torch.sin(t) * scheduler.config.sigma_data * pred
    sample = (sample if world._terrain_profile.gpu_windows else sample.cpu()).float() / scheduler.config.sigma_data
    return [torch.cat([value * weight[None], weight[None]], dim=0) for value in sample]


def _build_decoder(self):
    size, stride, lc = self.decoder_tile_size, self.decoder_tile_stride, self.latent_compression
    scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, sigma_max=80, sigma_data=0.5)
    cache_device = self.device if self._terrain_profile.gpu_windows else 'cpu'
    weight = linear_weight_window(size, 'cpu', torch.float32).to(cache_device)
    times = [torch.atan(scheduler.sigmas[0] / scheduler.config.sigma_data)]
    return InfiniteTensor(
        shape=(2, None, None),
        f=lambda ctxs, latents: _decoder_batch(self, ctxs, latents, scheduler, weight, times, size, stride),
        output_window=TensorWindow(size=(2, size, size), stride=(2, stride, stride)),
        args=(self.latents,),
        args_windows=(TensorWindow(size=(6, size // lc, size // lc), stride=(6, stride // lc, stride // lc)),),
        batch_size=self._terrain_profile.decoder_batch, device=cache_device,
        tile_store=self.tile_store, tensor_id='init_residual_map',
    )


def _init_conditioning(self):
    if getattr(self, '_terrain_polar_conditioning', None) is not None:
        self.synthetic_map_factory = self._terrain_polar_conditioning
        return
    world_profile = getattr(self, '_terrain_world_profile', 'natural')
    if world_profile != 'natural':
        from terrain_conditioning import make_conditioning_factory
        self.synthetic_map_factory = make_conditioning_factory(self.seed, world_profile)
        return
    # Upstream uses `seed or random.randint`, so zero was not a reproducible
    # world. Its factory masks seeds to 31 bits and uses no other base seed:
    # 2**31 therefore yields exactly the intended seeds [1,2,3,4,5] for seed 0.
    # All non-zero worlds keep their original factory and RNG streams.
    factory_seed = self.seed if self.seed != 0 else (1 << 31)
    self.synthetic_map_factory = make_synthetic_map_factory(
        seed=factory_seed, frequency_mult=self.kwargs['frequency_mult'],
        drop_water_pct=self.kwargs['drop_water_pct'])


def _build_hierarchy(self):
    # Install on every bind/rebuild. All direct reads then use the bounded real
    # stage graph, including complete upstream dependencies and fusion.
    WorldPipeline._build_hierarchy(self)
    # A previous persistence wrapper belongs to the discarded coarse tensor.
    # Reinstallation with the current manifest is explicit after a rebuild.
    self.__dict__.pop('_terrain_coarse_preparation', None)
    self.__dict__.pop('_terrain_refinement_fields', None)
    from terrain_window_scheduler import WorldWindowScheduler
    self._terrain_window_scheduler = WorldWindowScheduler(self)


def configure_world(world, profile=None, *, allow_experimental=False, world_profile=None):
    """Apply optimisations before bind; existing windows are never mixed.

    A world with an existing cache is rebuilt explicitly. Models may be shared
    between seed-specific worlds; their immutable eval-weight cache is shared.
    """
    profile = profile or choose_profile(world.device)
    world_profile = world_profile or getattr(world, '_terrain_world_profile', 'natural')
    from terrain_generation import resolve_generation
    descriptor = resolve_generation(world_profile)
    from terrain_snr import active as snr_active
    if profile.coarse_batch != 1 and not allow_experimental:
        raise ValueError('Coarse batches >1 rejected: BF16 solver drift changes terrain. Offline experiments only.')
    if profile.decoder_batch != 1 and not allow_experimental:
        raise ValueError('Decoder batches >1 rejected by the 1m numerical gate. Offline experiments only.')
    if profile.canonical_latents and profile.latent_batch != 1:
        raise ValueError('Canonical latent mode requires scalar base batches')
    if profile.attention_backend != 'reference' and not allow_experimental:
        if os.environ.get('TERRAIN_EXPERIMENTAL_INFERENCE') != '1':
            raise ValueError('Approximate attention requires explicit offline experimental admission')
    if world.torch_compile:
        raise ValueError('Realtime profile currently requires the validated eager CUDA path')
    from terrain_nn_constants import validate_engine_config
    # Even graph-disabled worlds share patched model aliases in this process.
    validate_engine_config(exact_kernels=profile.exact_kernels, attention_backend=profile.attention_backend)
    if (getattr(world, '_terrain_profile', None) == profile and
            getattr(world, '_terrain_world_profile', 'natural') == world_profile):
        return world
    world._terrain_generation_settings = descriptor.settings
    world._terrain_snr_active = snr_active(descriptor.settings)
    if profile.cached_weights:
        prepare_models(world)
    if profile.cuda_graphs or profile.exact_kernels or profile.attention_backend != 'reference':
        from terrain_nn_constants import prepare_constants
        prepare_constants(world, exact_kernels=profile.exact_kernels, attention_backend=profile.attention_backend)
    if profile.cuda_graphs:
        from terrain_cuda_graphs import CudaGraphModel
        if not isinstance(world.coarse_model, CudaGraphModel):
            world.coarse_model = CudaGraphModel(world.coarse_model, max_buckets=1)
        if (isinstance(world.base_model, CudaGraphModel) and
                world.base_model.max_batch != profile.graph_max_base_batch):
            world.base_model = world.base_model.model
        if not isinstance(world.base_model, CudaGraphModel):
            world.base_model = CudaGraphModel(world.base_model, max_buckets=max(8,profile.graph_max_base_batch),
                                              max_batch=profile.graph_max_base_batch)
        if profile.decoder_graphs and not isinstance(world.decoder_model, CudaGraphModel):
            world.decoder_model = CudaGraphModel(world.decoder_model, max_buckets=1, max_batch=1)
        elif not profile.decoder_graphs and isinstance(world.decoder_model, CudaGraphModel):
            world.decoder_model = world.decoder_model.model
    else:
        from terrain_cuda_graphs import CudaGraphModel
        # A seed-specific world may borrow models from a graph-enabled loader.
        # Disabling graphs must actually unwrap them for this world; other worlds
        # can retain their shared wrappers and resident capture buffers.
        for name in ('coarse_model', 'base_model', 'decoder_model'):
            model = getattr(world, name)
            if isinstance(model, CudaGraphModel):
                setattr(world, name, model.model)
    world._terrain_profile = profile
    if '_terrain_reference_cond_snr' not in world.__dict__:
        world._terrain_reference_cond_snr = list(world.kwargs['cond_snr'])
        world._terrain_reference_frequency_mult = list(world.kwargs['frequency_mult'])
        world._terrain_reference_drop_water_pct = world.kwargs['drop_water_pct']
    world.kwargs['cond_snr'] = (list(world._terrain_reference_cond_snr) if world_profile=='natural'
                               else list(descriptor.settings['cond_snr']))
    world.kwargs['frequency_mult'] = (list(world._terrain_reference_frequency_mult) if world_profile=='natural'
                                     else list(descriptor.settings['frequency_mult']))
    world.kwargs['drop_water_pct'] = (world._terrain_reference_drop_water_pct if world_profile=='natural'
                                     else descriptor.settings['drop_water_pct'])
    world._terrain_world_profile = world_profile
    world._init_conditioning = MethodType(_init_conditioning, world)
    world._batch_sizes = [b for b in (1, 2, 4, 8, 16, 32) if b <= profile.latent_batch]
    if world._batch_sizes[-1] != profile.latent_batch:
        world._batch_sizes.append(profile.latent_batch)
    world.latents_batch_size = profile.latent_batch
    world._build_coarse_stage = MethodType(_build_coarse, world)
    world._latent_inference = MethodType(_latent_inference, world)
    world._build_latent_stage = MethodType(_build_latent, world)
    world._build_decoder_stage = MethodType(_build_decoder, world)
    world._build_hierarchy = MethodType(_build_hierarchy, world)
    if world.tile_store is not None:
        world.rebuild()
    return world


def inference_status(world):
    if world is None:
        return None
    profile = getattr(world, '_terrain_profile', None)
    weight_bytes = 0
    cached_layers = 0
    for model in (world.coarse_model, world.base_model, world.decoder_model):
        for module in model.modules():
            cached = module.__dict__.get('_terrain_weight_cache')
            if cached is not None:
                cached_layers += 1
                weight_bytes += cached[1].numel() * cached[1].element_size()
    graph_stats = {name: model.stats() for name, model in (
        ('coarse', world.coarse_model), ('base', world.base_model), ('decoder', world.decoder_model)) if hasattr(model, 'stats')}
    if '_terrain_coarse_solver_graph' in world.__dict__:
        graph_stats['coarse_solver'] = world._terrain_coarse_solver_graph.stats()
    from terrain_device import select_cuda_device
    return {'version': VERSION, 'device': str(world.device), 'device_selection': select_cuda_device(),
            'world_profile': getattr(world, '_terrain_world_profile', 'natural'),
            'conditioning_noise': list(world.kwargs['cond_snr']),
            'regional_snr': {'enabled': world._terrain_snr_active, 'cached_policies': len(world.__dict__.get('_terrain_snr_cache', {}))},
            'climate_projection': None,
            'profile': asdict(profile) if profile else None, 'cached_weight_layers': cached_layers,
            'cached_weight_bytes': weight_bytes, 'solver_steps': 20, 'latent_fusion_steps': world.T,
            'precision': str(world._dtype), 'compile': world.torch_compile,
            'pointwise_kernels': __import__('terrain_nn_constants').kernel_status(),
            'window_cache_bytes': getattr(world.tile_store, '_bytes', 0), 'cuda_graphs': graph_stats,
            'coarse_persistence': getattr(world,'_terrain_coarse_preparation').status() if hasattr(world,'_terrain_coarse_preparation') else None,
            'windows': __import__('terrain_window_scheduler').scheduler_status(world)}

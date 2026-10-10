"""Experimental 15 m to 3.75 m cascade using the existing 30 m decoder.

The checkpoint is not a super-resolution model. Each level reuses its four
appearance latents over a smaller physical footprint and a new noise stream.
Only its high-frequency residual is added to the interpolated parent DEM.
Globally aligned 2x2 blocks retain their parent's mean away from the coast;
the coastal band follows the continuous native DEM instead of block offsets.
"""
from __future__ import annotations

from contextlib import nullcontext
import math

import torch
import torch.nn.functional as F
from infinite_tensor import InfiniteTensor, TensorWindow
from terrain_diffusion.inference.world_pipeline import linear_weight_window
from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler

VERSION = 'decoder-cascade-v5-native-coast-sign'
MIN_LOD = -3


def _repeat(value):
    return value.repeat_interleave(2, -2).repeat_interleave(2, -1)


def _native_coast(world, level, i1, j1, i2, j2, *, check=None):
    """Bilinear native heights and a one-native-cell coastal band.

    Use integer origins and local fractions, including at negative/far world
    coordinates. Each fine LOD uses the same 30 m surface: recursive detail
    and mean correction cannot introduce a new land/sea classification here.
    This footprint is contained in the recursive parent's native read halo.
    """
    scale = 2**level
    ni1, nj1 = i1//scale-1, j1//scale-1
    ni2, nj2 = (i2-1)//scale+2, (j2-1)//scale+2
    native = sample_refined(world, 0, ni1, nj1, ni2, nj2, check=check)
    native = native.to(world.device, dtype=torch.float32)[None, None]
    lower = -F.max_pool2d(-native, 3, stride=1, padding=1)
    upper = F.max_pool2d(native, 3, stride=1, padding=1)
    mixed = (lower < 0) & (upper >= 0)

    def coordinates(start, stop, origin):
        cells = torch.arange(start, stop, device=world.device, dtype=torch.int64)
        centres = cells.div(scale, rounding_mode='floor') - origin
        fraction = (cells.remainder(scale).float()+.5)/scale-.5
        left = centres + (fraction < 0).long()*-1
        return centres, left, fraction.remainder(1)

    cy, iy, fy = coordinates(i1, i2, ni1)
    cx, ix, fx = coordinates(j1, j2, nj1)
    z = native[0, 0]
    # Convex interpolation avoids overshoot and preserves the zero surface.
    # Separate products also keep CPU SIMD/tail rounding independent of the
    # requested tile shape (torch.lerp may fuse only its vectorized path).
    top = z[iy[:, None], ix]*(1-fx) + z[iy[:, None], ix+1]*fx
    bottom = z[iy[:, None]+1, ix]*(1-fx) + z[iy[:, None]+1, ix+1]*fx
    surface = top*(1-fy[:, None]) + bottom*fy[:, None]
    return surface[None, None], mixed[:, :, cy[:, None], cx]


def _fractional_latents(world, row, col, size, stride, scale):
    """Read a sub-cell footprint without rounding its global origin to FP32.

    Positions refer to the decoder's 64x64 conditioning sample centres. Only
    local offsets become FP32, retaining centimetre spacing far from the origin.
    Normalize the fused source before interpolating its physical latent values.
    """
    lc = world.latent_compression
    count = size // lc
    row0 = (row * stride + lc / 2) / (lc * scale) - .5
    col0 = (col * stride + lc / 2) / (lc * scale) - .5
    i1, j1 = math.floor(row0), math.floor(col0)
    i2 = math.floor(row0 + (count - 1) / scale) + 2
    j2 = math.floor(col0 + (count - 1) / scale) + 2
    fused = world.latents[:, i1:i2, j1:j2].to(world.device, dtype=torch.float32)
    values = (fused[:-1] / fused[-1:])[None]
    offsets = torch.arange(count, device=values.device, dtype=torch.float32) / scale
    ys = (offsets + (row0 - i1) + .5) * (2 / (i2 - i1)) - 1
    xs = (offsets + (col0 - j1) + .5) * (2 / (j2 - j1)) - 1
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')
    grid = torch.stack((xx, yy), dim=-1)[None]
    value = F.grid_sample(values, grid, mode='bilinear', padding_mode='border',
                          align_corners=False)[0]
    return torch.cat((value, torch.ones_like(value[:1])), dim=0)


def _detail_field(world, level):
    fields = world.__dict__.setdefault('_terrain_refinement_fields', {})
    if level in fields:
        return fields[level]
    from terrain_inference import _decoder_batch

    size, stride = world.decoder_tile_size, world.decoder_tile_stride
    scale, lc = 2**level, world.latent_compression
    if size % lc:
        raise ValueError('Refinement requires aligned decoder conditioning')
    aligned = size % (lc * scale) == 0 and stride % (lc * scale) == 0
    latent_size, latent_stride = size // (lc * scale), stride // (lc * scale)
    scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, sigma_max=80, sigma_data=0.5)
    cache_device = world.device if world._terrain_profile.gpu_windows else 'cpu'
    weight = linear_weight_window(size, 'cpu', torch.float32).to(cache_device)
    times = [torch.atan(scheduler.sigmas[0] / scheduler.config.sigma_data)]

    def generate(ctxs):
        latents = []
        for ctx in ctxs:
            if aligned:
                row, col = ctx[1] * latent_stride, ctx[2] * latent_stride
                value = world.latents[:, row:row + latent_size, col:col + latent_size]
                # Keep the original four refinement levels byte-for-byte.
                latents.append(value.repeat_interleave(scale, -2).repeat_interleave(scale, -1))
            else:
                latents.append(_fractional_latents(world, ctx[1], ctx[2], size, stride, scale))
        return _decoder_batch(world, ctxs, latents, scheduler, weight, times, size, stride,
                              noise_seed=world.seed + level * 104729)

    tensor = InfiniteTensor(
        shape=(2, None, None), f=generate,
        output_window=TensorWindow(size=(2, size, size), stride=(2, stride, stride)),
        batch_size=1, device=cache_device, tile_store=world.tile_store,
        tensor_id=f'refinement_residual_{level}',
    )
    observer = getattr(world, '_terrain_window_scheduler', None)
    if observer is not None:
        observer._install_recursive(tensor)
    fields[level] = tensor
    return tensor


@torch.inference_mode()
def sample_refined(world, level, i1, j1, i2, j2, *, check=None):
    """Read fine-grid cells; level 0 is the native 30 m DEM (coherent coast signs).

    Integer coordinates refer to cell edges at 30 / 2**level metres. Parent
    interpolation and detail projection include a halo, so overlapping tile
    requests agree, including across negative world coordinates.
    """
    if not 0 <= level <= -MIN_LOD:
        raise ValueError('Unsupported refinement level')
    if any(not isinstance(v, int) for v in (i1, j1, i2, j2)):
        raise ValueError('Refinement coordinates must be integers')
    if i2 <= i1 or j2 <= j1 or (i2 - i1) * (j2 - j1) > 1024**2:
        raise ValueError('Refinement read exceeds bounded output budget')
    if check is not None:
        check()
    observer = getattr(world, '_terrain_window_scheduler', None)
    with observer.scope(check) if observer is not None else nullcontext():
        if level == 0:
            # The same coherent-sign native DEM as LOD 0-2 decoder tiles.
            from terrain_coastline import read_native
            return read_native(world, i1, j1, i2, j2).float()

        # One parent cell beyond each aligned child block is enough for both
        # bilinear filters. Cropping removes their outer edge clamping.
        pi1, pj1 = i1 // 2 - 1, j1 // 2 - 1
        pi2, pj2 = (i2 + 1) // 2 + 1, (j2 + 1) // 2 + 1
        parent = sample_refined(world, level - 1, pi1, pj1, pi2, pj2, check=check)
        parent = parent.to(world.device, dtype=torch.float32)[None, None]
        base = F.interpolate(parent, scale_factor=2, mode='bilinear', align_corners=False)
        base = base + _repeat(parent - F.avg_pool2d(base, 2))
        # Mean correction can overshoot even when every neighbour is land.
        # Limit its zero-mean variation to the local parent range; mixed
        # land/sea neighbourhoods still allow interpolated coast crossings.
        tiny = torch.finfo(parent.dtype).tiny
        variation = base - _repeat(parent)
        lower = -F.max_pool2d(-parent, 3, stride=1, padding=1)
        upper = F.max_pool2d(parent, 3, stride=1, padding=1)
        down = F.max_pool2d((-variation).clamp_min(0), 2)
        up = F.max_pool2d(variation.clamp_min(0), 2)
        interpolation_gain = torch.minimum(
            (parent-lower) / down.clamp_min(tiny),
            (upper-parent) / up.clamp_min(tiny)).clamp(0, 1)
        base = _repeat(parent) + variation * _repeat(interpolation_gain)

        # Block mean offsets create detached pixels in mixed-sign terrain,
        # even with bounded interpolation and sign-safe decoder residuals.
        # Reconstruct the coast directly from the native DEM at every level.
        coast, coastal = _native_coast(world, level, 2*pi1, 2*pj1, 2*pi2, 2*pj2,
                                      check=check)
        base = torch.where(coastal, coast, base)

        field = _detail_field(world, level)
        fused = field[:, 2*pi1:2*pi2, 2*pj1:2*pj2].to(world.device)
        detail = (fused[0:1] / fused[1:2]).clamp(-3, 3)[None]
        low = F.interpolate(F.avg_pool2d(detail, 2), scale_factor=2,
                            mode='bilinear', align_corners=False)
        detail = detail - low
        detail = detail - _repeat(F.avg_pool2d(detail, 2))
        # Checkpoint residuals live in signed-sqrt height space. Convert their
        # small perturbation to metres with the parent's local derivative;
        # decrease amplitude with resolution and retain zero block means.
        amplitude = 2 * parent.abs().sqrt()
        amplitude = amplitude * float(world.kwargs['residual_std']) * (0.5**level)
        residual = detail * _repeat(amplitude)
        # Linearizing signed-sqrt heights alone lets noise dominate very low
        # terrain. Bound it smoothly by half the lowest absolute child height:
        # zero at sea level, tending to the original amplitude on high ground.
        # A shared gain per 2x2 block keeps the residual's mean exactly zero;
        # the interpolated base can still reconstruct crossings at the coast.
        peak = F.max_pool2d(residual.abs(), 2)
        height = -F.max_pool2d(-base.abs(), 2)
        gain = height / torch.hypot(height, 2*peak).clamp_min(tiny)
        result = base + residual * _repeat(gain)
        if check is not None:
            check()
        row, col = i1 - 2*pi1, j1 - 2*pj1
        return result[0, 0, row:row + i2-i1, col:col + j2-j1].contiguous()

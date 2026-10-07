"""Generate tiny deterministic cross-language fixtures from upstream code."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "terrain-diffusion"))
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import torch

from terrain_diffusion.scheduler.dpmsolver import EDMDPMSolverMultistepScheduler
from terrain_diffusion.inference.world_pipeline import gaussian_noise_patch, linear_weight_window
from terrain_diffusion.data.laplacian_encoder import laplacian_denoise, laplacian_decode
from terrain_diffusion.inference.postprocessing import local_baseline_temperature_torch
import torch.nn.functional as F


def main() -> None:
    scheduler = EDMDPMSolverMultistepScheduler(sigma_min=0.002, sigma_max=80, sigma_data=0.5)
    scheduler.set_timesteps(20)
    initial = torch.tensor([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7, -0.8], dtype=torch.float32)
    sample = initial.clone()
    snapshots = {}
    for i, t in enumerate(scheduler.timesteps):
        output = torch.tensor([np.sin((i + 1) * 0.19 + j * 0.37) * 0.25 for j in range(8)], dtype=torch.float32)
        sample = scheduler.step(output, t, sample).prev_sample
        if i in (0, 1, 9, 19):
            snapshots[str(i + 1)] = sample.tolist()
    fixture = {
        "source": "terrain_diffusion.scheduler.dpmsolver.EDMDPMSolverMultistepScheduler and world_pipeline",
        "initial": initial.tolist(),
        "sigmas": scheduler.sigmas.tolist(),
        "solver_snapshots": snapshots,
        "weight_8": linear_weight_window(8, "cpu", torch.float32).numpy().tolist(),
        "noise_patch": gaussian_noise_patch(42, -3, 5, 3, 4, 2, tile_h=8, tile_w=8).tolist(),
    }
    weight = linear_weight_window(8, "cpu", torch.float32)
    fused_sum = torch.zeros((2, 8, 14), dtype=torch.float32)
    fused_weight = torch.zeros((8, 14), dtype=torch.float32)
    tiles = []
    for x, values in ((0, (1.0, -2.0)), (6, (3.0, 4.0))):
        output = torch.cat([torch.tensor(values)[:, None, None] * weight[None], weight[None]])
        fused_sum[:, :, x:x+8] += output[:2]
        fused_weight[:, x:x+8] += output[2]
        tiles.append({"y": 0, "x": x, "data": output.numpy().reshape(-1).tolist()})
    fixture["fusion"] = {
        "channels": 2, "tile_size": 8, "y0": 0, "x0": 0, "height": 8, "width": 14,
        "tiles": tiles,
        "expected": (fused_sum / fused_weight[None]).numpy().reshape(-1).tolist(),
    }
    y = torch.arange(64, dtype=torch.float32)[:, None]
    x = torch.arange(64, dtype=torch.float32)[None, :]
    residual = 0.6 * torch.sin(y * 0.21) * torch.cos(x * 0.13)
    low_y = torch.arange(8, dtype=torch.float32)[:, None]
    low_x = torch.arange(8, dtype=torch.float32)[None, :]
    lowres = 2.0 * low_y - 1.5 * low_x + torch.sin(low_x + low_y)
    residual_d, lowres_d = laplacian_denoise(residual, lowres, sigma=5)
    decoded = laplacian_decode(residual_d, lowres_d)
    height = torch.sign(decoded) * torch.square(decoded)
    fixture["laplacian"] = {
        "high_h": 64, "high_w": 64, "low_h": 8, "low_w": 8,
        "residual": residual.numpy().reshape(-1).tolist(),
        "lowres": lowres.numpy().reshape(-1).tolist(),
        "expected_height": height.numpy().reshape(-1).tolist(),
    }
    gy, gx = torch.meshgrid(torch.arange(-8, 9, dtype=torch.float32), torch.arange(-8, 9, dtype=torch.float32), indexing="ij")
    sqrt_e = 12 * torch.sin(gy * 0.31) + 9 * torch.cos(gx * 0.27)
    coarse = torch.stack([
        sqrt_e, sqrt_e - 3,
        18 - 0.002 * torch.clamp(sqrt_e, min=0).square() + gy * 0.15,
        200 + gx * 4, 1100 + gy * 8 + gx * 5, 40 + gx * 0.7,
    ]).float()
    coarse_elev = torch.clamp(coarse[0], min=0).square()
    baseline, beta = local_baseline_temperature_torch(coarse[2], coarse_elev, win=15, fallback_threshold=0.02)
    central = coarse[:, 7:-7, 7:-7]
    features = torch.cat([baseline, beta, central], dim=0).unsqueeze(0)
    hi, hj = torch.meshgrid(torch.arange(64), torch.arange(64), indexing="ij")
    elev = (50 + hi.float() * 1.7 - hj.float() * 0.8).clamp_min(-30)
    u = (hi + 0.5) / 256 + 0.5
    v = (hj + 0.5) / 256 + 0.5
    grid = torch.stack([(v + 0.5) * 2 / 3 - 1, (u + 0.5) * 2 / 3 - 1], dim=-1).unsqueeze(0)
    up = F.grid_sample(features, grid, mode="bilinear", padding_mode="border", align_corners=False).squeeze(0)
    clim = torch.stack([up[0] + up[1] * elev.clamp_min(0), up[5], up[6], up[7], up[1]])
    fixture["climate"] = {
        "coarse_h": 17, "coarse_w": 17, "i1": 0, "j1": 0, "height": 64, "width": 64, "scale": 8,
        "coarse": coarse.numpy().reshape(-1).tolist(),
        "elevation": elev.numpy().reshape(-1).tolist(),
        "expected": clim.numpy().reshape(-1).tolist(),
    }
    destination = ROOT / "webgpu" / "math-fixture.json"
    destination.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()

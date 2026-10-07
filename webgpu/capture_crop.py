"""Capture an upstream natural-world 64px coastal crop and first real feeds.

This records a reference; it does not supply intermediate values to the browser
as a substitute for completing the crop dependency graph.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "terrain-diffusion"))
os.environ.setdefault("HF_HOME", "E:/TerrainDiffusionRuntime/huggingface")
os.environ.setdefault("MPLBACKEND", "Agg")
os.chdir(ROOT / "terrain-diffusion")

import numpy as np
import torch

from terrain_diffusion.inference.world_pipeline import WorldPipeline

REVISION = "9ef8030cb805b433b98ec25c5dddefbac07a9e26"
SNAPSHOT = Path(os.environ["HF_HOME"]) / "hub" / "models--xandergos--terrain-diffusion-30m" / "snapshots" / REVISION
DEST = Path("E:/TerrainDiffusionRuntime/webgpu-models")
CROP = (2560, -3584, 2624, -3520)
SEED = 42
PREFIX = "crop64-coast"


def save_array(name: str, value: torch.Tensor | np.ndarray) -> dict:
    data = value.detach().float().cpu().contiguous().numpy() if isinstance(value, torch.Tensor) else np.asarray(value, dtype=np.float32)
    data = np.ascontiguousarray(data, dtype=np.float32)
    path = DEST / f"{PREFIX}-{name}.bin"
    path.write_bytes(data.tobytes())
    return {"file": path.name, "shape": list(data.shape), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "min": float(data.min()), "max": float(data.max()), "mean": float(data.mean())}


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    calls = {name: 0 for name in ("coarse_model", "base_model", "decoder_model")}
    first = {}
    windows = {"coarse": [], "base_pass1": [], "base_pass2": [], "decoder": []}
    started = time.perf_counter()
    print("Loading natural FP32 checkpoint", flush=True)
    world = WorldPipeline.from_pretrained(str(SNAPSHOT), seed=SEED, dtype=None, torch_compile=False,
                                          latents_batch_size=1, log_mode="none", cache_limit=512 * 1024 * 1024).to("cuda")

    def hook(name):
        def capture(_module, args, kwargs, output):
            calls[name] += 1
            if name in first:
                return
            record = {"x": save_array(f"{name}-x", args[0]),
                      "noise_labels": save_array(f"{name}-noise_labels", kwargs["noise_labels"]),
                      "output": save_array(f"{name}-output", output)}
            for index, condition in enumerate(kwargs.get("conditional_inputs", [])):
                record[f"cond_{index}"] = save_array(f"{name}-cond_{index}", condition)
            first[name] = record
            print(f"Captured {name} first forward, call={calls[name]}", flush=True)
        return capture

    handles = [getattr(world, name).register_forward_hook(hook(name), with_kwargs=True) for name in calls]
    original_coarse = world._coarse_inference
    original_base = world._latent_inference
    original_decoder = world._decoder_inference

    def capture_coarse(_self, ctx, *args, **kwargs):
        result = original_coarse(ctx, *args, **kwargs)
        _, ty, tx = ctx
        i1, j1 = ty * 48, tx * 48
        input_map = world._conditioning_model_input(i1, i1 + 64, j1, j1 + 64)
        index = len(windows["coarse"])
        windows["coarse"].append({"ctx": [int(v) for v in ctx],
                                  "input_map": save_array(f"coarse-{index}-input_map", input_map),
                                  "tile": save_array(f"coarse-{index}-tile", result)})
        return result

    def capture_base(_self, ctxs, samples, *args, **kwargs):
        result = original_base(ctxs, samples, *args, **kwargs)
        stage = "base_pass1" if samples is None or all(item is None for item in samples) else "base_pass2"
        for ctx, tile in zip(ctxs, result, strict=True):
            index = len(windows[stage])
            windows[stage].append({"ctx": [int(v) for v in ctx],
                                   "tile": save_array(f"{stage}-{index}-tile", tile)})
        return result

    def capture_decoder(_self, ctx, *args, **kwargs):
        result = original_decoder(ctx, *args, **kwargs)
        index = len(windows["decoder"])
        windows["decoder"].append({"ctx": [int(v) for v in ctx],
                                   "tile": save_array(f"decoder-{index}-tile", result)})
        return result

    world._coarse_inference = types.MethodType(capture_coarse, world)
    world._latent_inference = types.MethodType(capture_base, world)
    world._decoder_inference = types.MethodType(capture_decoder, world)
    try:
        world.bind()
        with torch.no_grad():
            output = world.get(*CROP, with_climate=True)
        elevation = save_array("elevation", output["elev"])
        climate = save_array("climate", output["climate"])
        data = {
            "checkpoint_revision": REVISION,
            "profile": "natural",
            "precision": "fp32",
            "seed": str(SEED),
            "bbox_row_col": CROP,
            "native_m_per_pixel": 30,
            "source": "unmodified WorldPipeline.from_pretrained(...).get with batch 1",
            "seconds": time.perf_counter() - started,
            "forward_calls": calls,
            "windows": windows,
            "first_real_forward": first,
            "elevation": elevation,
            "climate": climate,
        }
        (DEST / f"{PREFIX}-manifest.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(json.dumps({"seconds": data["seconds"], "forward_calls": calls, "elevation": elevation, "climate": climate}, indent=2), flush=True)
    finally:
        for handle in handles:
            handle.remove()
        world.close()


if __name__ == "__main__":
    main()

"""Exclusive-GPU experiment: raw learned coast context versus altitude noise.

No product defaults or weights are changed. The four climate noise entries
stay .5. Each 64x64 coarse patch is 491.52km across and has real NN contributors.
"""
from __future__ import annotations

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()


import argparse
import json
from pathlib import Path
import time

import numpy as np

from verify_terrestrial_bootstrap import STYLES
from validate_terrestrial_nn import local_comparison, sha, save_board


def tune(cpu_report, output, seeds=(0,), styles=STYLES, noises=(.5, .2, .1, .05)):
    cpu_report, output = Path(cpu_report).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cpu = json.loads(cpu_report.read_text(encoding="utf-8"))
    if cpu["contract_failures"]:
        raise ValueError("CPU contract must pass before tuning")
    cpu_worlds = {(int(row["seed"]), row["style"]): row for row in cpu["worlds"]}
    candidates = json.loads((cpu_report.parent / "local-candidates.json").read_text(encoding="utf-8"))
    sites = {(int(site["seed"]), site["world_profile"].removeprefix("terrestrial-")): site
             for site in candidates if site["kind"] == "coast-median"}
    report = {"schema": "terrestrial-coast-noise-study-v1", "gpu_used": True, "complete": False,
              "cpu_report_sha256": sha(cpu_report), "validator_sha256": sha(__file__),
              "seeds": [str(s) for s in seeds], "styles": list(styles), "elevation_noise": list(noises),
              "climate_noise": [.5] * 4, "cell_m": 7680, "patch_cells": 64,
              "product_defaults_changed": False, "height_mask_or_clipping_used": False, "worlds": {}}
    target = output / "coast-noise-report.json"

    def persist():
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    persist()
    from terrain_app import load_pipeline, resolve_model_source, gpu_calls
    from terrain_inference import configure_world
    from terrain_conditioning import make_conditioning_factory
    from terrain_reference import neural_manifest
    import torch
    checkpoint = Path(resolve_model_source())
    try:
        with torch.inference_mode():
            for seed in seeds:
                for style in styles:
                    site = sites[(seed, style)]
                    profile, key = f"terrestrial-{style}", f"{style}-s{seed}"
                    factory = make_conditioning_factory(seed, profile)
                    expected = cpu_worlds[(seed, style)]["metadata"]
                    if any(factory.heightmap.metadata[name] != expected[name]
                           for name in ("cache_key", "raw_height_sha256", "height_sha256")):
                        raise ValueError("CPU parent source identity changed")
                    ci, cj = site["center_ci"], site["center_cj"]
                    parent = factory.sample_raw(cj-32, ci-32, cj+32, ci+32)
                    context = {"site": site, "source": "actual physical 7680m lattice",
                               "bounds_coarse_x0_y0_x1_y1": [cj-32, ci-32, cj+32, ci+32], "variants": {}}
                    fields, labels = [parent[0]], [f"{key}: physical input; 491.52km patch"]
                    report["worlds"][key] = context
                    for noise in noises:
                        started = time.perf_counter()
                        world = load_pipeline(seed)
                        configure_world(world, world_profile=profile)
                        world.kwargs["cond_snr"] = [float(noise), .5, .5, .5, .5]
                        world.rebuild()
                        stage = world.coarse[:, ci-32:ci+32, cj-32:cj+32].float().cpu().numpy()
                        if not np.isfinite(stage).all() or np.any(stage[-1] <= 0):
                            raise ValueError("Invalid NN contributors")
                        signed = stage[0] / stage[-1]
                        learned = np.sign(signed) * signed**2
                        manifest = neural_manifest(world, profile, checkpoint_source=checkpoint)
                        prefix = f"{key}-noise{noise:g}"
                        npz = output / f"{prefix}.npz"
                        np.savez_compressed(npz, coarse_weighted=stage, raw_learned_coarse_m=learned,
                                            physical_input=parent, manifest_hash=np.array(manifest["world_hash"]))
                        (output / f"{prefix}-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                        comparison = local_comparison(parent[0], learned, 7680)
                        context["variants"][str(noise)] = {"npz": npz.name, "npz_sha256": sha(npz),
                            "manifest_hash": manifest["world_hash"], "actual_cond_snr": list(world.kwargs["cond_snr"]),
                            "seconds": time.perf_counter() - started, "comparison": comparison}
                        fields.append(learned)
                        labels.append(f"Raw NN coarse | altitude noise {noise:g}")
                        print(f"{prefix}: sign changes {comparison['raw_land_sign_disagreement_fraction']:.1%}; zero distances {comparison['zero_contour_distances']['distance_min_median_p95_max_m']}", flush=True)
                        persist()
                    save_board(output / f"{key}-noise-board.png", fields, labels, width=512, height=512)
        report["complete"] = True
        report["cuda_forward_calls"] = dict(gpu_calls)
        persist()
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        persist()
        raise
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--style", choices=STYLES, action="append")
    parser.add_argument("--noise", type=float, action="append")
    args = parser.parse_args()
    tune(args.cpu_report, args.output, tuple(args.seed or (0,)), tuple(args.style or STYLES),
         tuple(args.noise or (.5, .2, .1, .05)))

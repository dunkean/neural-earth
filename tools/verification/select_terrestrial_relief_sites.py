"""CPU-only stress-site selection by measured local parent relief and p99 height."""
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

import numpy as np

from verify_terrestrial_bootstrap import STYLES, sampling_coordinates, area_weights, weighted_quantiles


def select(cpu_report, output, high_relief_seed=0, p99_seed=42, pixels=512):
    from terrain_bootstrap import get_heightmap
    cpu_report, output = Path(cpu_report).resolve(), Path(output).resolve()
    cpu = json.loads(cpu_report.read_text(encoding="utf-8"))
    audited = {(int(row["seed"]), row["style"]): row for row in cpu["worlds"]}
    sites = []
    half_span = pixels * 30 / 2
    for seed, kind in ((high_relief_seed, "high-local-relief"), (p99_seed, "p99-height")):
        for style in STYLES:
            provider = get_heightmap(seed, style)
            if provider.metadata["height_sha256"] != audited[(seed, style)]["metadata"]["height_sha256"]:
                raise ValueError("CPU source differs from stress selection source")
            height = provider.height_m
            xs, ys = sampling_coordinates(height.shape)
            low, high = np.full(height.shape, np.inf), np.full(height.shape, -np.inf)
            # Actual immutable sampler at a 3x3 physical grid across the native
            # footprint. This measures the source, not an invented ridge score.
            for dy in (-half_span, 0., half_span):
                for dx in (-half_span, 0., half_span):
                    sampled = provider.sample_height_m(xs + dx, ys + dy)
                    low = np.minimum(low, sampled)
                    high = np.maximum(high, sampled)
            allowed = (height > 0) & (low > 0) & (np.abs(ys)[:, None] < 8_000_000)
            if kind == "high-local-relief":
                allowed &= height > 500
            if not allowed.any():
                raise ValueError(f"No positive-land stress sites in {style} seed {seed}")
            if kind == "high-local-relief":
                score = np.where(allowed, high - low, -np.inf)
                row, col = np.unravel_index(np.argmax(score), score.shape)
                rule = "maximum actual 3x3 parent height range across15.36km among nonpolar footprints with all9samples land and centre>500m; no NN beauty selection"
                target = None
            else:
                target = weighted_quantiles(height[allowed], area_weights(height.shape)[allowed], (.99,))[0]
                score = np.where(allowed, np.abs(height - target), np.inf)
                row, col = np.unravel_index(np.argmin(score), score.shape)
                rule = "closest centre to spherical-area weighted99th percentile of admissible nonpolar inland height; high altitude does not certify mountain shape"
            x, y = float(xs[col]), float(ys[row])
            site = {"id": f"{style}-s{seed}-{kind}", "seed": str(seed), "world_profile": f"terrestrial-{style}",
                "kind": kind, "center_x_m": x, "center_y_m": y,
                "center_cj": int(np.floor(x / 7680)), "center_ci": int(np.floor(y / 7680)),
                "focus_native_j": int(np.floor(x / 30)), "focus_native_i": int(np.floor(y / 30)),
                "parent_center_height_m": float(height[row, col]),
                "parent_footprint_sample_min_m": float(low[row, col]),
                "parent_footprint_sample_max_m": float(high[row, col]),
                "parent_footprint_3x3_range_m": float(high[row, col] - low[row, col]),
                "target_p99_height_m": target, "native_pixels": pixels, "selection_rule": rule,
                "selection_scope": "source bilinear height only; high local range can be a steep hillside, not a certified mountain massif"}
            sites.append(site)
            print(json.dumps(site), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(sites, indent=2) + "\n", encoding="utf-8")
    return sites


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    select(args.cpu_report, args.output)

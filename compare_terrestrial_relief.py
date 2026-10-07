"""Matched-footprint CUDA relief comparison, raw stages and explicit land masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_erosion, gaussian_filter

from validate_terrestrial_nn import sha, climate_diagnostics


def relief_metrics(height, mask, cell_m=30):
    height, mask = np.asarray(height, np.float64), np.asarray(mask, bool)
    values = height[mask]
    interior = binary_erosion(mask, structure=np.ones((3, 3)), border_value=0)
    slope = np.hypot(*np.gradient(height, cell_m))[interior]
    highpass = (height - gaussian_filter(height, sigma=1000 / cell_m, mode="reflect"))[interior]
    return {"selected_fraction": float(mask.mean()), "selected_pixels": int(mask.sum()),
            "height_quantiles_m": np.quantile(values, [0, .05, .5, .95, 1]).tolist() if values.size else None,
            "relief_p95_minus_p05_m": float(np.quantile(values, .95) - np.quantile(values, .05)) if values.size else None,
            "relief_max_minus_min_m": float(np.ptp(values)) if values.size else None,
            "height_std_m": float(values.std()) if values.size else None,
            "height_highpass_1000m_sigma_std_m": float(highpass.std()) if highpass.size else None,
            "interior_slope_quantiles": np.quantile(slope, [.5, .9, .95, .99]).tolist() if slope.size else None,
            "interior_slope_pixels": int(slope.size)}


def power_spectrum(values, cell_m, units):
    """Plane-detrended, Hann-windowed radial Fourier power; identical footprints."""
    values = np.asarray(values, np.float64)
    y, x = np.indices(values.shape)
    design = np.stack((np.ones(values.size), x.ravel(), y.ravel()), axis=1)
    trend = (design @ np.linalg.lstsq(design, values.ravel(), rcond=None)[0]).reshape(values.shape)
    residual = values - trend
    window = np.hanning(values.shape[0])[:, None] * np.hanning(values.shape[1])[None, :]
    power = np.abs(np.fft.fft2(residual * window))**2 / (values.size * np.square(window).sum())
    fy, fx = np.fft.fftfreq(values.shape[0], cell_m), np.fft.fftfreq(values.shape[1], cell_m)
    radius = np.hypot(fy[:, None], fx[None, :])
    positive = radius > 0
    edges = np.geomspace(radius[positive].min(), radius.max() * (1 + 1e-12), 9)
    bands = np.histogram(radius[positive], bins=edges, weights=power[positive])[0]
    return {"units": units, "cell_m": cell_m, "shape": list(values.shape),
            "method": "least-squares plane removed; 2D Hann; radial summed FFT power normalized by N*window_energy",
            "frequency_edges_cycles_per_km": (edges * 1000).tolist(),
            "power_units_squared_by_band": bands.tolist(), "total_non_dc_power_units_squared": float(bands.sum()),
            "detrended_unwindowed_std": float(residual.std()),
            "area_mask": "whole co-registered context; spectra are not called land-only"}


def enrich_comparison(report_path):
    """CPU-only derived metrics from saved matched raw arrays; no re-inference."""
    report_path = Path(report_path).resolve()
    output = report_path.parent
    report = json.loads(report_path.read_text(encoding="utf-8"))
    from terrain_conditioning import make_conditioning_factory
    for row in report["sites"].values():
        bounds, site = row["native_bounds"], row["site"]
        xs, ys = (np.arange(bounds[0], bounds[2]) + .5) * 30, (np.arange(bounds[1], bounds[3]) + .5) * 30
        parent = make_conditioning_factory(int(site["seed"]), site["world_profile"]).sample(xs, ys)[0]
        parent_land = parent > 0
        arrays = {}
        for name, variant in row["variants"].items():
            with np.load(output / variant["artifact"]["npz"], allow_pickle=False) as data:
                arrays[name] = data["dem_m"]
            variant["native_height_spectrum"] = power_spectrum(arrays[name], 30, "metres")
        common = parent_land & (arrays["noise0.05"] > 0) & (arrays["noise0.1"] > 0) & (arrays["noise0.5"] > 0)
        row["parent_metrics"] = relief_metrics(parent, parent_land)
        for name, variant in row["variants"].items():
            for label, mask in (("own_land", arrays[name] > 0), ("fixed_parent_land", parent_land), ("matched_terrestrial_common_land", common)):
                variant[label] = relief_metrics(arrays[name], mask)
        for label in ("own_land", "fixed_parent_land", "matched_terrestrial_common_land"):
            a, b = row["variants"]["noise0.05"][label], row["variants"]["noise0.5"][label]
            row["ratios_noise005_to_05"][label]["highpass_1000m_sigma_std"] = a["height_highpass_1000m_sigma_std_m"] / b["height_highpass_1000m_sigma_std_m"] if b["height_highpass_1000m_sigma_std_m"] else None
    report["cpu_enrichment_sha256"] = sha(__file__)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def compare(cpu_report, output, pixels=512, sites_json=None, skip_natural=False):
    cpu_report, output = Path(cpu_report).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cpu = json.loads(cpu_report.read_text(encoding="utf-8"))
    if cpu["contract_failures"]:
        raise ValueError("CPU contract failures")
    candidates = json.loads((cpu_report.parent / "local-candidates.json").read_text(encoding="utf-8"))
    wanted = ("earthlike-s0-mountain-median", "earthlike-s0-plain-median",
              "archipelago-s0-mountain-median", "continents-s0-mountain-median")
    sites = json.loads(Path(sites_json).resolve().read_text(encoding="utf-8")) if sites_json else [next(s for s in candidates if s["id"] == name) for name in wanted]
    report = {"schema": "matched-terrestrial-relief-v1", "complete": False, "gpu_used": True,
              "cpu_report_sha256": sha(cpu_report), "comparer_sha256": sha(__file__),
              "conditioning_version": cpu["conditioning_version"], "pixels": pixels, "sites": {},
              "selection_source": str(Path(sites_json).resolve()) if sites_json else "legacy high-elevation-median IDs and one plain; altitude ranking is not morphology certification",
              "quality_thresholds_registered": False,
              "comparison_scope": "same seed and native bounds; own-land slopes and fixed/intersection masks distinguish changed land support"}
    target = output / "relief-comparison.json"

    def persist():
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    persist()
    from terrain_app import load_pipeline, resolve_model_source, gpu_calls
    from terrain_inference import configure_world
    from terrain_reference import export_neural_intermediates
    from terrain_conditioning import make_conditioning_factory, BOOTSTRAP_CONDITIONING_VERSION
    import torch
    if BOOTSTRAP_CONDITIONING_VERSION != report["conditioning_version"]:
        raise ValueError("Conditioning version changed")
    checkpoint = Path(resolve_model_source())
    cpu_worlds = {(int(row["seed"]), row["style"]): row for row in cpu["worlds"]}
    try:
        with torch.inference_mode():
            for site in sites:
                factory = make_conditioning_factory(int(site["seed"]), site["world_profile"])
                expected_parent = cpu_worlds[(int(site["seed"]), site["world_profile"].removeprefix("terrestrial-"))]["metadata"]
                if any(factory.heightmap.metadata[name] != expected_parent[name]
                       for name in ("cache_key", "raw_height_sha256", "height_sha256")):
                    raise ValueError("CPU parent identity differs from matched comparison")
                row, native_arrays = {"site": site, "variants": {}}, {}
                report["sites"][site["id"]] = row
                variants = [("noise0.05", .05), ("noise0.1", .1), ("noise0.5", .5)]
                if not skip_natural:
                    variants.append(("natural", None))
                board = Image.new("RGB", (pixels * len(variants), pixels + 36), "#17212a")
                draw = ImageDraw.Draw(board)
                for col, (name, noise) in enumerate(variants):
                    world = load_pipeline(int(site["seed"]))
                    profile = "natural" if noise is None else site["world_profile"]
                    configure_world(world, world_profile=profile)
                    if noise is not None:
                        world.kwargs["cond_snr"] = [noise, .5, .5, .5, .5]
                    world.rebuild()
                    named_site = {**site, "id": f"{site['id']}-{name}"}
                    artifact = export_neural_intermediates(world, named_site, output,
                                                           native_pixels=pixels, checkpoint_source=checkpoint)
                    with np.load(output / artifact["npz"], allow_pickle=False) as data:
                        dem, bounds = data["dem_m"], data["native_bounds"]
                        coarse = data["coarse_unweighted"][0]
                        coarse_m = np.sign(coarse) * coarse**2
                        lowfreq_sqrt = data["latent_lowfreq"] * 38.6 - 31.4
                        lowfreq_m = np.sign(lowfreq_sqrt) * lowfreq_sqrt**2
                        features = data["latent_normalized"][:4]
                        climate = data["climate"]
                    if row.get("native_bounds") is not None and row["native_bounds"] != bounds.tolist():
                        raise ValueError("Native comparison bounds changed")
                    row["native_bounds"] = bounds.tolist()
                    if "parent_metrics" not in row:
                        xs = (np.arange(bounds[0], bounds[2]) + .5) * 30
                        ys = (np.arange(bounds[1], bounds[3]) + .5) * 30
                        parent = make_conditioning_factory(int(site["seed"]), site["world_profile"]).sample(xs, ys)[0]
                        parent_land = parent > 0
                        row["parent_metrics"] = relief_metrics(parent, parent_land)
                    native_arrays[name] = dem
                    row["variants"][name] = {"artifact": artifact, "actual_cond_snr": list(world.kwargs["cond_snr"]),
                        "own_land": relief_metrics(dem, dem > 0),
                        "fixed_parent_land": relief_metrics(dem, parent_land),
                        "source_land_sign_changed_fraction": float(np.mean((dem > 0) != parent_land)),
                        "raw_climate": climate_diagnostics(climate),
                        "native_height_spectrum": power_spectrum(dem, 30, "metres"),
                        "coarse_height_spectrum": power_spectrum(coarse_m, 7680, "metres"),
                        "latent_lowfreq_height_spectrum": power_spectrum(lowfreq_m, 240, "metres"),
                        "latent_feature_spectra": [power_spectrum(feature, 240, "normalized learned latent units") for feature in features]}
                    board.paste(Image.open(output / artifact["shaded_png"]), (col * pixels, 36))
                    draw.text((col * pixels + 4, 8), f"{site['id']} | {name}", fill="white")
                    print(f"Matched {site['id']} {name}: ownland={row['variants'][name]['own_land']}", flush=True)
                    persist()
                common_land = parent_land & (native_arrays["noise0.05"] > 0) & (native_arrays["noise0.1"] > 0) & (native_arrays["noise0.5"] > 0)
                row["terrestrial_common_land_fraction"] = float(common_land.mean())
                for name, dem in native_arrays.items():
                    row["variants"][name]["matched_terrestrial_common_land"] = relief_metrics(dem, common_land)
                row["ratios_noise005_to_05"] = {}
                for mask_name in ("own_land", "fixed_parent_land", "matched_terrestrial_common_land"):
                    a, b = row["variants"]["noise0.05"][mask_name], row["variants"]["noise0.5"][mask_name]
                    ratio = {"relief_p95_minus_p05": a["relief_p95_minus_p05_m"] / b["relief_p95_minus_p05_m"] if b["relief_p95_minus_p05_m"] else None}
                    ratio["slope_p95"] = a["interior_slope_quantiles"][2] / b["interior_slope_quantiles"][2] if b["interior_slope_quantiles"] and b["interior_slope_quantiles"][2] else None
                    row["ratios_noise005_to_05"][mask_name] = ratio
                board.save(output / f"{site['id']}-relief-board.png")
                persist()
        report["complete"] = True
        report["cuda_forward_calls"] = dict(gpu_calls)
        persist()
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        persist()
        raise
    return enrich_comparison(target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pixels", type=int, default=512)
    parser.add_argument("--sites-json", type=Path, help="CPU-selected explicit sites; default preserves legacy high-elevation median IDs")
    parser.add_argument("--skip-natural", action="store_true", help="three terrestrial noise variants only")
    args = parser.parse_args()
    compare(args.cpu_report, args.output, args.pixels, args.sites_json, args.skip_natural)

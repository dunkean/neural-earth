"""Explicit CUDA quality audit of signed terrestrial parents and raw NN stages.

Run only in the coordinator's exclusive GPU slot. --plan-only is CPU-only.
--quick selects one median coast per style and 8x4=32 learned point probes;
the default selects twelve local sites and 32x16=512 probes per world.
These probes are not a complete learned planetary raster or native coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree

from verify_terrestrial_bootstrap import STYLES, area_weights, render_height, weighted_quantiles


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summary(height, weights=None):
    height = np.asarray(height)
    if weights is None:
        weights = np.full(height.shape, 1 / height.size)
    return {"shape": list(height.shape), "finite": bool(np.isfinite(height).all()),
            "land_fraction": float(weights[height > 0].sum() / weights.sum()),
            "height_quantiles_m": weighted_quantiles(height, weights, (0, .01, .05, .5, .95, .99, 1)),
            "fraction_abs_height_over_20000m": float(weights[np.abs(height) > 20000].sum() / weights.sum())}


def coast_mask(height):
    land = np.asarray(height) > 0
    coast = np.zeros(land.shape, bool)
    vertical, horizontal = land[1:] != land[:-1], land[:, 1:] != land[:, :-1]
    coast[1:] |= vertical
    coast[:-1] |= vertical
    coast[:, 1:] |= horizontal
    coast[:, :-1] |= horizontal
    return coast


def zero_contour_points(height, cell_m):
    """Subcell linear zero crossings on grid edges, in common planar metres."""
    height = np.asarray(height, np.float64)
    points = []
    for axis in (0, 1):
        a, b = (height[:-1], height[1:]) if axis == 0 else (height[:, :-1], height[:, 1:])
        rows, cols = np.where((a > 0) != (b > 0))
        if len(rows):
            fraction = a[rows, cols] / (a[rows, cols] - b[rows, cols])
            y, x = rows.astype(float), cols.astype(float)
            if axis == 0:
                y += fraction
            else:
                x += fraction
            points.append(np.stack((x, y), axis=1) * cell_m)
    return np.concatenate(points) if points else np.empty((0, 2))


def contour_comparison(parent, learned, cell_m):
    source, nn = zero_contour_points(parent, cell_m), zero_contour_points(learned, cell_m)
    distances = {}
    for name, origin, destination in (("source_to_nn", source, nn), ("nn_to_source", nn, source)):
        distances[name] = np.quantile(cKDTree(destination).query(origin)[0], [0, .5, .95, 1]).tolist() if len(origin) and len(destination) else None
    return {"source_crossings": len(source), "nn_crossings": len(nn),
            "distance_min_median_p95_max_m": distances,
            "method": "nearest edge-linear zero-contour crossing in shared planar metres; crop-limited"}


def local_comparison(parent, learned, cell_m=30):
    """Measure raw changed signs and visible local coast offsets, without masks."""
    parent, learned = np.asarray(parent), np.asarray(learned)
    if parent.shape != learned.shape:
        raise ValueError("Local comparison footprints differ")
    source_coast, nn_coast = coast_mask(parent), coast_mask(learned)
    distances = {}
    for name, origin, destination in (("source_to_nn", source_coast, nn_coast),
                                      ("nn_to_source", nn_coast, source_coast)):
        if not origin.any() or not destination.any():
            distances[name] = None
        else:
            values = distance_transform_edt(~destination, sampling=cell_m)[origin]
            distances[name] = np.quantile(values, [0, .5, .95, 1]).tolist()
    result = {"source": summary(parent), "raw_nn": summary(learned),
              "raw_land_sign_disagreement_fraction": float(np.mean((parent > 0) != (learned > 0))),
              "source_land_changed_to_sea_fraction": float(np.mean((parent > 0) & (learned <= 0))),
              "source_sea_changed_to_land_fraction": float(np.mean((parent <= 0) & (learned > 0))),
              "height_difference_quantiles_m": np.quantile(learned - parent, [0, .05, .5, .95, 1]).tolist(),
              "source_coast_present": bool(source_coast.any()), "nn_coast_present": bool(nn_coast.any()),
              "visible_crop_coast_distance_min_median_p95_max_m": distances,
              "coast_distance_limit": "Visible crop only; null when either coastline is absent; no correspondence outside crop"}
    result["zero_contour_distances"] = contour_comparison(parent, learned, cell_m)
    for name, height in (("source", parent), ("raw_nn", learned)):
        slope = np.hypot(*np.gradient(height.astype(np.float64), cell_m))
        result[name]["slope_quantiles"] = np.quantile(slope, [.5, .95, .99, 1]).tolist()
    return result


def coarse_conditioning_at_native(factory, xs, ys):
    """Co-register the actual 7.68km physical input lattice and a native crop."""
    cj0, cj1 = int(np.floor(xs[0] / 7680 - .5)), int(np.ceil(xs[-1] / 7680 - .5)) + 1
    ci0, ci1 = int(np.floor(ys[0] / 7680 - .5)), int(np.ceil(ys[-1] / 7680 - .5)) + 1
    cx, cy = (np.arange(cj0, cj1) + .5) * 7680, (np.arange(ci0, ci1) + .5) * 7680
    lattice = factory.sample(cx, cy)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    coordinates = np.stack((yy, xx), axis=-1)
    fields = np.stack([RegularGridInterpolator((cy, cx), channel)(coordinates)
                       for channel in lattice]).astype(np.float32)
    return fields, {"physical_lattice_shape": list(lattice.shape),
                    "bounds_indices_x0_y0_x1_y1": [cj0, ci0, cj1, ci1], "cell_m": 7680,
                    "interpolation": "bilinear physical fields at native centres"}


def climate_diagnostics(climate):
    """Inspect raw NN physical climate, preserving nonphysical predictions."""
    climate = np.asarray(climate)
    names = ("temperature_c", "temperature_std_c_x100", "precipitation_mm_year", "precipitation_cv_percent", "lapse_c_per_m")
    return {name: {"min": float(channel.min()), "max": float(channel.max()),
                   "negative_fraction": float(np.mean(channel < 0)), "finite": bool(np.isfinite(channel).all()),
                   "negative_is_nonphysical": index in (1, 2, 3)}
            for index, (name, channel) in enumerate(zip(names, climate))}


def enrich_report(report_path):
    """CPU-only diagnostic enrichment from the exact saved raw NN arrays."""
    report_path = Path(report_path).resolve()
    output = report_path.parent
    report = json.loads(report_path.read_text(encoding="utf-8"))
    for record in report["global_probes"].values():
        with np.load(output / record["npz"], allow_pickle=False) as data:
            parent, learned = data["physical_conditioning"][0], data["learned_coarse_m"]
            record["source_flat"] = summary(parent)
            record["raw_nn_flat"] = summary(learned)
            record["flat_land_sign_disagreement_fraction"] = float(np.mean((parent > 0) != (learned > 0)))
            record["spherical_area_scope"] = "Original stratified spherical strip weights; coarse centres snapped by at most3840m; sparse point evidence only"
            if "learned_coarse_channels" in data:
                record["raw_nn_coarse_climate_diagnostics"] = climate_diagnostics(data["learned_coarse_channels"][2:6])
    for record in report["sites"].values():
        with np.load(output / record["artifact"]["npz"], allow_pickle=False) as data:
            record["raw_nn_native_climate_diagnostics"] = climate_diagnostics(data["climate"])
            record["raw_nn_coarse_climate_diagnostics"] = climate_diagnostics(data["coarse_unweighted"][2:6])
    report["cpu_enrichment_sha256"] = sha(__file__)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def select_sites(candidates, styles, seeds, quick=False, site_ids=None, expanded=False):
    lookup = {(c["world_profile"].removeprefix("terrestrial-"), int(c["seed"]), c["kind"]): c for c in candidates}
    if site_ids:
        sites = [c for c in candidates if c["id"] in site_ids]
        if len(sites) != len(set(site_ids)):
            raise ValueError("Requested site IDs missing from CPU candidates")
        return sites
    selected = []
    for index, style in enumerate(styles):
        choices = [(seeds[0], "coast-median")] if quick else [
            (seeds[0], "coast-low-relief"), (seeds[-1], "coast-median"),
            (seeds[index % len(seeds)], "plain-median" if index % 2 else "mountain-median")]
        if expanded and not quick:
            polar_kind = "polar-land-median" if (style, seeds[0], "polar-land-median") in lookup else "polar-sea-median"
            choices = [(seeds[0], "coast-low-relief"), (seeds[-1], "coast-median"),
                       (seeds[0], "plain-median"), (seeds[-1], "mountain-median"), (seeds[0], polar_kind)]
            if style == "earthlike" and len(seeds) > 1:
                second_polar = "polar-coast-median" if (style, seeds[-1], "polar-coast-median") in lookup else "polar-land-median"
                choices.append((seeds[-1], second_polar))
        for seed, kind in choices:
            if (style, seed, kind) not in lookup:
                raise ValueError(f"CPU candidates lack {style} seed {seed} {kind}")
            selected.append(lookup[(style, seed, kind)])
    return selected


def sparse_global(world, factory, width, progress=None):
    """Real complete-contributor NN point reads on a stratified planetary grid."""
    xs = -20e6 + (np.arange(width) + .5) * 40e6 / width
    ys = -10e6 + (np.arange(width // 2) + .5) * 20e6 / (width // 2)
    cj, ci = np.floor(xs / 7680).astype(np.int64), np.floor(ys / 7680).astype(np.int64)
    physical_x, physical_y = (cj + .5) * 7680, (ci + .5) * 7680
    learned = np.empty((len(ci), len(cj)), np.float32)
    channels = np.empty((6, len(ci), len(cj)), np.float32)
    for row, i in enumerate(ci):
        for col, j in enumerate(cj):
            value = world.coarse[:, int(i):int(i)+1, int(j):int(j)+1].float().cpu().numpy()
            if not np.isfinite(value).all() or float(value[-1, 0, 0]) <= 0:
                raise ValueError("NN coarse contributors invalid")
            signed = value[0, 0, 0] / value[-1, 0, 0]
            channels[:, row, col] = value[:-1, 0, 0] / value[-1, 0, 0]
            learned[row, col] = np.sign(signed) * signed**2
        if progress:
            progress(row + 1, len(ci))
    return {"learned_coarse_m": learned, "learned_coarse_channels": channels,
            "physical_conditioning": factory.sample(physical_x, physical_y),
            "sample_x_m": physical_x, "sample_y_m": physical_y, "area_weights": area_weights(learned.shape)}


def save_board(path, fields, labels, width=512, height=256):
    board = Image.new("RGB", (width * len(fields), height + 36), "#17212a")
    draw = ImageDraw.Draw(board)
    for col, (field, label) in enumerate(zip(fields, labels)):
        board.paste(Image.fromarray(render_height(field)).resize((width, height), Image.Resampling.NEAREST), (col * width, 36))
        draw.text((col * width + 6, 6), label, fill="white")
    board.save(path)


def validate(cpu_report, output, seeds=(0, 42), styles=STYLES, *, pixels=512,
             global_width=32, quick=False, site_ids=None, skip_local=False, plan_only=False,
             elevation_cond_noise=None, expanded_sites=False):
    cpu_report, output = Path(cpu_report).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cpu = json.loads(cpu_report.read_text(encoding="utf-8"))
    if cpu.get("contract_failures"):
        raise ValueError("CPU contract failures must be resolved before NN QA")
    candidates = json.loads((cpu_report.parent / "local-candidates.json").read_text(encoding="utf-8"))
    cpu_worlds = {(int(row["seed"]), row["style"]): row for row in cpu["worlds"]}
    for seed in seeds:
        for style in styles:
            if (seed, style) not in cpu_worlds:
                raise ValueError(f"CPU audit lacks {style} seed {seed}")
    sites = [] if skip_local else select_sites(candidates, styles, seeds, quick, site_ids, expanded_sites)
    report = {"schema": "terrestrial-raw-nn-quality-v1", "cpu_report": str(cpu_report),
              "cpu_report_sha256": sha(cpu_report), "validator_sha256": sha(__file__),
              "quick": quick, "gpu_used": False, "complete": False,
              "global_probe_kind": "stratified actual NN coarse point samples; no raster topology or LOD area means",
              "complete_global_nn_generated": False, "complete_native_world_generated": False,
              "global_probe_width": global_width, "pixels": pixels, "seeds": [str(s) for s in seeds],
              "elevation_cond_noise_override": elevation_cond_noise,
              "styles": list(styles), "planned_sites": sites, "global_probes": {}, "sites": {},
              "numerical_failures": [], "shoreline_findings": [],
              "quality_thresholds_registered": False,
              "quality_acceptance": "Measured shoreline/relief deltas require review; numerical success is not aesthetic acceptance",
              "conditioning_version": cpu.get("conditioning_version", "unknown"),
              "quality_claim": "No learned evidence until GPU records are completed"}
    target = output / "neural-report.json"

    def persist():
        target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    persist()
    if plan_only:
        return report
    from terrain_app import load_pipeline, resolve_model_source, gpu_calls
    from terrain_inference import configure_world
    from terrain_reference import export_neural_intermediates, neural_manifest
    from terrain_conditioning import make_conditioning_factory, BOOTSTRAP_CONDITIONING_VERSION
    if report["conditioning_version"] != BOOTSTRAP_CONDITIONING_VERSION:
        raise ValueError("CPU conditioning version differs from current NN conditioning")
    import torch
    checkpoint = Path(resolve_model_source())
    report["checkpoint_source"] = str(checkpoint)
    report["gpu_used"] = True
    try:
        with torch.inference_mode():
            for seed in seeds:
                for style in styles:
                    selected = [s for s in sites if int(s["seed"]) == seed and s["world_profile"] == f"terrestrial-{style}"]
                    if not global_width and not selected:
                        continue
                    profile = f"terrestrial-{style}"
                    factory = make_conditioning_factory(seed, profile)
                    expected_parent = cpu_worlds[(seed, style)]["metadata"]
                    current_parent = factory.heightmap.metadata
                    if any(current_parent[name] != expected_parent[name]
                           for name in ("cache_key", "raw_height_sha256", "height_sha256")):
                        raise ValueError(f"CPU parent identity changed before NN audit: {style} seed {seed}")
                    world = load_pipeline(seed)
                    configure_world(world, world_profile=profile)
                    if elevation_cond_noise is not None:
                        world.kwargs["cond_snr"] = [float(elevation_cond_noise), .5, .5, .5, .5]
                    world.rebuild()
                    manifest = neural_manifest(world, profile, checkpoint_source=checkpoint)
                    key = f"{style}-s{seed}"
                    if global_width:
                        started = time.perf_counter()
                        probe = sparse_global(world, factory, global_width,
                            lambda row, total: print(f"NN {key} global row {row}/{total}", flush=True))
                        path = output / f"{key}-global-probe.npz"
                        np.savez_compressed(path, **probe, manifest_hash=np.array(manifest["world_hash"]))
                        (output / f"{key}-global-manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
                        parent, learned, weights = probe["physical_conditioning"][0], probe["learned_coarse_m"], probe["area_weights"]
                        record = {"seconds": time.perf_counter() - started, "manifest_hash": manifest["world_hash"],
                                  "npz": path.name, "npz_sha256": sha(path), "source": summary(parent, weights),
                                  "raw_nn": summary(learned, weights),
                                  "weighted_land_sign_disagreement_fraction": float(weights[(parent > 0) != (learned > 0)].sum()),
                                  "point_count": int(learned.size), "quality_scope": "Sparse signed point heights; no inferred NN components or coastline length"}
                        report["global_probes"][key] = record
                        save_board(output / f"{key}-global-board.png", [parent, learned],
                                   [f"{key}: physical parent {global_width}x{global_width//2} POINTS", "Raw actual NN coarse POINTS; nearest display"])
                        if not record["raw_nn"]["finite"] or record["raw_nn"]["fraction_abs_height_over_20000m"]:
                            report["numerical_failures"].append({"world": key, "reason": "nonfinite or NN height above 20km"})
                        persist()
                    for site in selected:
                        started = time.perf_counter()
                        artifact = export_neural_intermediates(world, site, output, native_pixels=pixels, checkpoint_source=checkpoint)
                        with np.load(output / artifact["npz"], allow_pickle=False) as data:
                            dem, bounds = data["dem_m"], data["native_bounds"]
                        xs, ys = (np.arange(bounds[0], bounds[2]) + .5) * 30, (np.arange(bounds[1], bounds[3]) + .5) * 30
                        direct = factory.sample(xs, ys)
                        lattice, footprint = coarse_conditioning_at_native(factory, xs, ys)
                        comparison = local_comparison(lattice[0], dem)
                        path = output / f"{site['id']}-source-comparison.npz"
                        np.savez_compressed(path, direct_physical=direct, coarse_lattice_bilinear_physical=lattice,
                                            raw_nn_dem_m=dem, native_bounds=bounds, manifest_hash=np.array(manifest["world_hash"]))
                        save_board(output / f"{site['id']}-native-board.png", [direct[0], lattice[0], dem],
                                   ["Physical parent sampled at 30m", "Actual 7680m input lattice, bilinear view", "Raw NN DEM 30m; no parent land mask"], width=pixels, height=pixels)
                        report["sites"][site["id"]] = {"site": site, "artifact": artifact,
                            "seconds": time.perf_counter() - started, "comparison_npz": path.name,
                            "comparison_npz_sha256": sha(path), "coarse_conditioning_footprint": footprint,
                            "comparison_to_actual_lattice": comparison, "comparison_to_direct_parent": local_comparison(direct[0], dem)}
                        if not comparison["raw_nn"]["finite"] or comparison["raw_nn"]["fraction_abs_height_over_20000m"]:
                            report["numerical_failures"].append({"site": site["id"], "reason": "nonfinite or NN height above 20km"})
                        if comparison["source_coast_present"] and not comparison["nn_coast_present"]:
                            report["shoreline_findings"].append({"site": site["id"],
                                "finding": "source coastline present but raw NN coastline absent from this native crop",
                                "changed_sign_fraction": comparison["raw_land_sign_disagreement_fraction"]})
                        print(f"NN {site['id']} native done: changed signs={comparison['raw_land_sign_disagreement_fraction']:.1%}", flush=True)
                        persist()
        report["complete"] = True
        report["quality_claim"] = "Actual checkpoint coarse/latent/native stages for listed sparse probes and local crops; source coastline changes are measured without masking"
        report["cuda_forward_calls"] = dict(gpu_calls)
        persist()
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        persist()
        raise
    return enrich_report(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--style", choices=STYLES, action="append")
    parser.add_argument("--pixels", type=int)
    parser.add_argument("--global-width", type=int, help="even stratified width; points=width*width/2; zero skips")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument("--site-id", action="append")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--elevation-cond-noise", type=float, help="study-only altitude cond_snr override; four climate channels remain .5")
    parser.add_argument("--expanded-sites", action="store_true", help="all styles: two coasts, plain, mountain, polar; Earthlike adds second-seed polar")
    args = parser.parse_args()
    width = args.global_width if args.global_width is not None else (8 if args.quick else 32)
    if width < 0 or width % 2 or width == 2:
        parser.error("--global-width must be 0 or an even integer at least 4")
    result = validate(args.cpu_report, args.output, tuple(args.seed or ((0,) if args.quick else (0, 42))),
                      tuple(args.style or STYLES), pixels=args.pixels or (256 if args.quick else 512),
                      global_width=width, quick=args.quick, site_ids=args.site_id,
                      skip_local=args.skip_local, plan_only=args.plan_only,
                      elevation_cond_noise=args.elevation_cond_noise, expanded_sites=args.expanded_sites)
    raise SystemExit(1 if result["numerical_failures"] else 0)


if __name__ == "__main__":
    main()

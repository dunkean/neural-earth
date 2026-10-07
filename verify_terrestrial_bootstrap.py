"""Independent CPU audits of native terrestrial parents; no NN/GPU execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import label

STYLES = ("gondwana", "continents", "earthlike", "archipelago")
DEFAULT_SEEDS = (0, 1, 13, 42, 97, 2026, 1234567, 4294967297)
BOUNDS_M = (-20_000_000., -10_000_000., 20_000_000., 10_000_000.)
HEIGHT_KNOTS = np.array([-11000, -6000, -2000, -1, 0, 200, 1000, 2500, 4500, 6500, 9000])
HEIGHT_COLORS = np.array([[5, 18, 40], [14, 47, 82], [32, 91, 125], [98, 161, 175],
                          [217, 211, 159], [99, 142, 83], [139, 143, 94], [144, 119, 96],
                          [162, 149, 137], [222, 220, 213], [247, 246, 241]])


def area_weights(shape):
    """Exact spherical strip areas for an equirectangular full-world raster."""
    h, w = shape
    edges = np.linspace(np.pi / 2, -np.pi / 2, h + 1)
    rows = np.sin(edges[:-1]) - np.sin(edges[1:])
    return np.broadcast_to(rows[:, None] / (2 * w), (h, w))


def weighted_quantiles(values, weights, probabilities=(.01, .05, .25, .5, .75, .95, .99)):
    values, weights = np.asarray(values).ravel(), np.asarray(weights).ravel()
    if not len(values) or float(weights.sum()) <= 0:
        return [None] * len(probabilities)
    order = np.argsort(values, kind="stable")
    v, w = values[order], weights[order]
    cdf = (np.cumsum(w) - .5 * w) / w.sum()
    return [float(x) for x in np.interp(probabilities, cdf, v)]


def periodic_components(mask, weights=None):
    """Four-neighbor labels with explicit longitude seam unions."""
    mask = np.asarray(mask, bool)
    labels, n = label(mask)
    if not n:
        return []
    parent = np.arange(n + 1)

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(labels[:, 0], labels[:, -1]):
        if a and b:
            ra, rb = root(a), root(b)
            parent[rb] = ra
    roots = np.array([root(i) for i in range(n + 1)])
    cells = np.ones(mask.shape, np.float64) if weights is None else np.asarray(weights)
    mass = np.bincount(roots[labels].ravel(), weights=(cells * mask).ravel(), minlength=n + 1)
    return sorted((float(x) for x in mass[1:] if x > 0), reverse=True)


def terrain_metrics(height, area_mode="spherical"):
    h = np.asarray(height, np.float64)
    if h.ndim != 2 or min(h.shape) < 2 or not np.isfinite(h).all():
        raise ValueError("Expected a finite full-world height raster")
    if area_mode not in ("spherical", "flat"):
        raise ValueError("Unknown area convention")
    weights = area_weights(h.shape) if area_mode == "spherical" else np.full(h.shape, 1 / h.size)
    land, sea = h > 0, h <= 0
    land_area = float(weights[land].sum())
    comps = periodic_components(land, weights)
    horizontal = land != np.roll(land, -1, axis=1)
    vertical = land[:-1] != land[1:]
    coast_cells = horizontal | np.roll(horizontal, 1, axis=1)
    coast_cells[:-1] |= vertical
    coast_cells[1:] |= vertical
    # Longitude neighbors share a north/south edge; latitude neighbors an east/west edge.
    radius = 40_000_000 / (2 * np.pi)
    edge_ns = radius * np.pi / h.shape[0]
    edge_ew = radius * 2 * np.pi / h.shape[1] * np.cos(np.linspace(np.pi / 2, -np.pi / 2, h.shape[0] + 1)[1:-1])
    coastline_m = float(horizontal.sum() * edge_ns + (vertical.sum(axis=1) * edge_ew).sum()) if area_mode == "spherical" else float(horizontal.sum() * 20_000_000 / h.shape[0] + vertical.sum() * 40_000_000 / h.shape[1])
    bins = [-12000, -8000, -5000, -2000, 0, 200, 500, 1000, 2000, 3500, 5000, 7000, 10000, 30000]
    hypsometry = np.histogram(h, bins=bins, weights=weights)[0]
    lat = np.pi / 2 - (np.arange(h.shape[0]) + .5) * np.pi / h.shape[0]
    dx = np.maximum(40_000_000 / h.shape[1] * np.cos(lat), 1) if area_mode == "spherical" else np.full(h.shape[0], 40_000_000 / h.shape[1])
    dy = 20_000_000 / h.shape[0]
    slope_x = (np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)) / (2 * dx[:, None])
    slope_y = np.gradient(h, dy, axis=0)
    slope = np.hypot(slope_x, slope_y)
    seam_step = np.abs(h[:, 0] - h[:, -1])
    neighbor_steps = np.abs(h - np.roll(h, 1, axis=1))
    return {
        "shape": list(h.shape), "finite": True, "min_m": float(h.min()), "max_m": float(h.max()),
        "area_weighted_mean_m": float((weights * h).sum()), "land_fraction": land_area,
        "land_components": len(comps), "land_components_at_least_0_1pct_planet": sum(a >= .001 for a in comps),
        "land_components_at_least_5pct_land": sum(a >= .05 * land_area for a in comps),
        "largest_land_fraction_of_planet": comps[0] if comps else 0.,
        "largest_land_fraction_of_land": comps[0] / land_area if comps else 0.,
        "largest_components_planet_fraction": comps[:12], "coast_cell_fraction": float(weights[coast_cells].sum()),
        "raster_coastline_km": coastline_m / 1000,
        "land_elevation_weighted_quantiles_m": weighted_quantiles(h[land], weights[land]),
        "sea_elevation_weighted_quantiles_m": weighted_quantiles(h[sea], weights[sea]),
        "highland_fraction_of_land_2000m": float(weights[h > 2000].sum() / land_area) if land_area else 0.,
        "plain_fraction_of_land_0_300m": float(weights[land & (h < 300)].sum() / land_area) if land_area else 0.,
        "slope_weighted_quantiles": weighted_quantiles(slope[land], weights[land]),
        "longitude_seam_step_weighted_quantiles_m": weighted_quantiles(seam_step, weights[:, 0]),
        "ordinary_longitude_neighbor_step_weighted_quantiles_m": weighted_quantiles(neighbor_steps, weights),
        "hypsometry_bins_m": bins, "hypsometry_planet_fraction": [float(v) for v in hypsometry],
        "metrics_topology": f"four-neighbor with longitude seam; {area_mode} area; source signed heights",
        "area_convention": area_mode,
    }


def raster_style_gate(metrics, style):
    """Independently apply the native broad shape gate to the exported raster.

    Native atlas acceptance uses eight-neighbor cube cells; this audit uses
    four-neighbor equirectangular raster cells with longitude seam unions.
    Neither gate promises a named geographic arrangement or fine relief.
    """
    largest = metrics["largest_land_fraction_of_land"]
    significant = metrics["land_components_at_least_5pct_land"]
    return {"gondwana": largest >= .85,
            "continents": largest <= .65 and significant >= 3,
            "earthlike": largest <= .75 and significant >= 2,
            "archipelago": largest <= .15 and metrics["land_components"] >= 100}[style]


def render_height(height):
    h = np.asarray(height)
    return np.stack([np.interp(h, HEIGHT_KNOTS, HEIGHT_COLORS[:, i]) for i in range(3)], axis=-1).astype(np.uint8)


def sampling_coordinates(shape):
    h, w = shape
    x0, y0, x1, y1 = BOUNDS_M
    return (x0 + (np.arange(w) + .5) * (x1 - x0) / w,
            y0 + (np.arange(h) + .5) * (y1 - y0) / h)


def select_local_candidates(height, seed, style, provider=None):
    """Select ordinary/unfavorable geography deterministically, without a beauty score."""
    h = np.asarray(height)
    xs, ys = sampling_coordinates(h.shape)
    land = h > 0
    coast = (land != np.roll(land, 1, axis=1)) | (land != np.roll(land, -1, axis=1))
    coast[1:] |= land[1:] != land[:-1]
    coast[:-1] |= land[:-1] != land[1:]
    # Stay away from poles; use the whole longitude domain including its seam.
    allowed = np.broadcast_to((np.abs(ys) < 8_000_000)[:, None], h.shape)
    neighborhood = sum(np.roll(np.roll(h, dy, axis=0), dx, axis=1) for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1))) / 4
    roughness = np.abs(h - neighborhood)
    candidates = []

    def pick(name, mask, field, fraction, selection_region=None):
        ij = np.argwhere(mask & (allowed if selection_region is None else selection_region))
        if not len(ij):
            return
        scores = field[ij[:, 0], ij[:, 1]]
        order = np.lexsort((ij[:, 1], ij[:, 0], scores))
        row, col = ij[order[min(len(order) - 1, int(fraction * len(order)))]]
        x, y = float(xs[col]), float(ys[row])
        selection_height = float(h[row, col])
        # A coast cell centre can lie tens of kilometres from its shoreline.
        # Refine a selected sign-changing edge against the immutable physical
        # sampler so a native local crop actually contains a parent shoreline.
        if "coast" in name and provider is not None:
            for di, dj in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                ri, cj = row + di, (col + dj) % h.shape[1]
                if 0 <= ri < h.shape[0] and land[row, col] != land[ri, cj]:
                    dx, dy = dj * (40_000_000 / h.shape[1]), di * (20_000_000 / h.shape[0])
                    lo, hi, start_land = 0., 1., bool(land[row, col])
                    for _ in range(32):
                        mid = (lo + hi) / 2
                        if bool(provider.sample_height_m([x + mid * dx], [y + mid * dy])[0, 0] > 0) == start_land:
                            lo = mid
                        else:
                            hi = mid
                    x, y = x + (lo + hi) / 2 * dx, y + (lo + hi) / 2 * dy
                    break
        candidates.append({"id": f"{style}-s{seed}-{name}", "seed": str(seed), "world_profile": f"terrestrial-{style}",
                           "kind": name, "row": int(row), "col": int(col), "center_x_m": x, "center_y_m": y,
                           "center_cj": int(np.floor(x / 7680)), "center_ci": int(np.floor(y / 7680)),
                           "focus_native_j": int(np.floor(x / 30)), "focus_native_i": int(np.floor(y / 30)),
                           "parent_selection_cell_height_m": selection_height,
                           "parent_center_height_m": float(provider.sample_height_m([x], [y])[0, 0]) if provider else selection_height,
                           "selection_score": float(field[row, col]), "selection_percentile": fraction,
                           "selection_rule": "ranked source geography; coast focus refined to bilinear zero crossing"})

    pick("coast-low-relief", coast, roughness, .1)
    pick("coast-median", coast, roughness, .5)
    positive = h[land & allowed]
    mountain_threshold = float(np.quantile(positive, .75)) if len(positive) else 2000.
    pick("mountain-median", land & (h >= max(1000., mountain_threshold)), h, .5)
    pick("plain-median", land & (h < 500), roughness, .5)
    polar = np.broadcast_to((np.abs(ys) > 8_800_000)[:, None], h.shape)
    pick("polar-coast-median", coast, roughness, .5, polar)
    pick("polar-land-median", land, h, .5, polar)
    pick("polar-sea-median", ~land, h, .5, polar)
    return candidates


def checkpoint_field_diagnostics(fields, coarse_means, coarse_stds):
    """Encode height exactly once, then measure the checkpoint's true z scores."""
    fields = np.asarray(fields, np.float64)
    encoded = fields.copy()
    encoded[0] = np.sign(encoded[0]) * np.sqrt(np.abs(encoded[0]))
    z = (encoded - np.asarray(coarse_means)[:, None, None]) / np.asarray(coarse_stds)[:, None, None]
    weights = area_weights(fields.shape[1:])
    names = ("elevation_signed_sqrt_m", "temperature_c", "temperature_std_c_x100", "precipitation_mm_year", "precipitation_cv_percent")
    result = {}
    for name, raw, normalized in zip(names, fields, z):
        result[name] = {"physical_min": float(raw.min()), "physical_max": float(raw.max()),
            "z_quantiles_spherical": weighted_quantiles(normalized, weights, (.01, .05, .5, .95, .99)),
            "z_mean_spherical": float((weights * normalized).sum()),
            "abs_z_over4_spherical_fraction": float(weights[np.abs(normalized) > 4].sum()),
            "abs_z_over4_flat_fraction": float(np.mean(np.abs(normalized) > 4)),
            "abs_z_max": float(np.max(np.abs(normalized))), "finite": bool(np.isfinite(normalized).all())}
    return result


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def export_batch(output, seeds=DEFAULT_SEEDS, styles=STYLES):
    from terrain_bootstrap import get_heightmap
    from terrain_conditioning import make_conditioning_factory, BOOTSTRAP_CONDITIONING_VERSION
    from terrain_manifest import MODEL_ROOT
    checkpoint = json.loads((MODEL_ROOT / "config.json").read_text(encoding="utf-8"))
    indices = [0, 2, 3, 4, 5]
    means = [checkpoint["coarse_means"][i] for i in indices]
    stds = [checkpoint["coarse_stds"][i] for i in indices]
    diagnostic_x, diagnostic_y = sampling_coordinates((128, 256))

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    reports, candidates, failures = [], [], []
    for seed in seeds:
        panels = []
        natural_fields = make_conditioning_factory(seed, "natural").sample(diagnostic_x, diagnostic_y)
        natural_diagnostics = checkpoint_field_diagnostics(natural_fields, means, stds)
        np.savez_compressed(output / f"natural-s{seed}-conditioning-diagnostic.npz", physical=natural_fields,
                            xs_m=diagnostic_x, ys_m=diagnostic_y, means=np.array(means), stds=np.array(stds))
        for style in styles:
            started = time.perf_counter()
            provider = get_heightmap(seed, style)
            final = np.asarray(provider.height_m, np.float32)
            source = np.asarray(provider.raw_height_m, np.float32)
            if source.shape != final.shape:
                raise ValueError("Source and calibrated rasters must share their grid for morphology comparison")
            metrics, raw_metrics = terrain_metrics(final), terrain_metrics(source)
            flat_metrics = terrain_metrics(final, "flat")
            xs, ys = sampling_coordinates(final.shape)
            probe_x, probe_y = xs[::max(1, len(xs) // 23)], ys[::max(1, len(ys) // 11)]
            sampled = provider.sample_height_m(probe_x, probe_y)
            split = len(probe_x) // 2
            partitioned = np.concatenate((provider.sample_height_m(probe_x[:split], probe_y),
                                          provider.sample_height_m(probe_x[split:], probe_y)), axis=1)
            wrapped = provider.sample_height_m(probe_x + 40_000_000, probe_y)
            reverse = provider.sample_height_m(probe_x[::-1], probe_y[::-1])[::-1, ::-1]
            seam_left = provider.sample_height_m([-20_000_000 - .01], probe_y)
            seam_right = provider.sample_height_m([-20_000_000 + .01], probe_y)
            conditioning = make_conditioning_factory(seed, f"terrestrial-{style}")
            fields = conditioning.sample(probe_x, probe_y)
            checks = {"source_shape_matches": source.shape == final.shape,
                      "exported_raster_passes_broad_style_gate": bool(raster_style_gate(metrics, style)),
                      "source_land_sign_preserved": bool(np.array_equal(source > 0, final > 0)),
                      "sample_partition_exact": bool(np.array_equal(sampled, partitioned)),
                      "sample_reverse_order_exact": bool(np.array_equal(sampled, reverse)),
                      "source_arrays_immutable": not provider.height_m.flags.writeable and not provider.raw_height_m.flags.writeable,
                      "longitude_wrap_max_error_m": float(np.max(np.abs(sampled - wrapped))),
                      "longitude_seam_continuity_error_m": float(np.max(np.abs(seam_left - seam_right))),
                      "conditioning_finite": bool(np.isfinite(fields).all()),
                      "conditioning_shape_matches": fields.shape == (5,) + sampled.shape,
                      "conditioning_height_max_error_m": float(np.max(np.abs(fields[0] - sampled))),
                      "physical_input_altitude_not_clipped": bool(np.array_equal(fields[0], sampled))}
            for name, value in checks.items():
                if (isinstance(value, bool) and not value) or (name.endswith("error_m") and value > .01):
                    failures.append({"seed": str(seed), "style": style, "check": name, "actual": value})
            diagnostic_fields = conditioning.sample(diagnostic_x, diagnostic_y)
            record = {"seed": str(seed), "style": style, "elapsed_s": time.perf_counter() - started,
                      "metadata": provider.metadata, "physical": metrics, "physical_flat": flat_metrics,
                      "native_source": raw_metrics, "checks": checks,
                      "conditioning_version": BOOTSTRAP_CONDITIONING_VERSION,
                      "checkpoint_diagnostics": checkpoint_field_diagnostics(diagnostic_fields, means, stds),
                      "natural_checkpoint_diagnostics_same_seed": natural_diagnostics,
                      "below_datum_non_ocean_spherical_fraction": float(area_weights(final.shape)[(final < 0) & ~provider.physical_ocean].sum()),
                      "below_datum_non_ocean_flat_fraction": float(np.mean((final < 0) & ~provider.physical_ocean)),
                      "physical_ocean_mask_scope": "nearest native diagnostic classification; below-datum non-ocean includes basins and coast reconstruction cells",
                      "calibration_land_fraction_delta": metrics["land_fraction"] - raw_metrics["land_fraction"],
                      "height_float32_sha256": hashlib.sha256(final.tobytes()).hexdigest()}
            reports.append(record)
            directory = output / f"{style}-s{seed}"
            directory.mkdir(exist_ok=True)
            np.savez_compressed(directory / "height.npz", physical_height_m=final, native_source_height_m=source,
                                xs_m=xs, ys_m=ys, sea_level_m=np.array(0.))
            np.savez_compressed(directory / "conditioning-diagnostic.npz", physical=diagnostic_fields,
                                xs_m=diagnostic_x, ys_m=diagnostic_y, means=np.array(means), stds=np.array(stds),
                                conditioning_version=np.array(BOOTSTRAP_CONDITIONING_VERSION))
            (directory / "metrics.json").write_text(json.dumps(record, indent=2, default=_json_default) + "\n", encoding="utf-8")
            candidates.extend(select_local_candidates(final, seed, style, provider))
            panel = Image.new("RGB", (1024, 288), "#17212a")
            ImageDraw.Draw(panel).text((8, 6), f"{style}  seed={seed} | native-derived signed raster (left) / mapped physical height (right) | land={metrics['land_fraction']:.1%}", fill="white")
            panel.paste(Image.fromarray(render_height(source)).resize((512, 256), Image.Resampling.BILINEAR), (0, 32))
            panel.paste(Image.fromarray(render_height(final)).resize((512, 256), Image.Resampling.BILINEAR), (512, 32))
            panels.append(panel)
            print(json.dumps({"seed": str(seed), "style": style, "elapsed_s": record["elapsed_s"],
                              "land": metrics["land_fraction"], "large_masses": metrics["land_components_at_least_0_1pct_planet"],
                              "largest_land": metrics["largest_land_fraction_of_land"], "checks": checks}), flush=True)
        board = Image.new("RGB", (1024, 288 * len(panels)), "#17212a")
        for i, panel in enumerate(panels):
            board.paste(panel, (0, 288 * i))
        board.save(output / f"comparison-s{seed}.png")
    result = {"schema": "terrestrial-cpu-independent-qa-v1", "gpu_used": False,
              "bounds_m": BOUNDS_M, "seeds": [str(s) for s in seeds], "styles": list(styles),
              "worlds": reports, "contract_failures": failures,
              "source_scope": "native_source metrics describe exported bilinear native-derived raster, not original cube atlas cells",
              "conditioning_version": BOOTSTRAP_CONDITIONING_VERSION,
              "checkpoint_normalization": {"coarse_indices": indices, "means": means, "stds": stds,
                  "diagnostic_sampling_shape": [128, 256], "altitude_encoding": "sign(h)*sqrt(abs(h)) exactly once"},
              "quality_claim": "Measured CPU parent topology and physical inputs; learned terrain quality untested"}
    (output / "report.json").write_text(json.dumps(result, indent=2, default=_json_default) + "\n", encoding="utf-8")
    (output / "local-candidates.json").write_text(json.dumps(candidates, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("generated/terrestrial-bootstrap-cpu"))
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--styles", default=",".join(STYLES))
    args = parser.parse_args()
    result = export_batch(args.output, [int(s) for s in args.seeds.split(",")], args.styles.split(","))
    raise SystemExit(1 if result["contract_failures"] else 0)


if __name__ == "__main__":
    main()

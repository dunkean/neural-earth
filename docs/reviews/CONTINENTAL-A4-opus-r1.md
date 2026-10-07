# CONTINENTAL-A4 review (read-only)

**Verdict: ACCEPT WITH LIMITATIONS.** A4 is acceptable as an experimental, non-default profile. It is **not promotable** in its current state. The NN report is now complete: 4 sites plus the 32×16 global probe. It confirms the large-scale land/ocean layout, but both A4 coast sites show a defect that comes from the coast formula itself.

## Checks that passed

- **Same coordinates everywhere.** Each sample depends only on its absolute position. Capsule maths uses float64 and the crop-invariance test is exact.
- **Elevation transform and SNR.** The signed square root is applied exactly once (`terrain_macro.py:290-293`). The pipeline then normalises with the checkpoint's own coarse means/stds (`world_pipeline.py:1022-1023`). No custom imports, no NN output mask. Conditioning noise is 0.5.
- **Profile routing.** `macro-a4` routes to `ContinentalBootstrap` (`terrain_inference.py:400-403`). The server's `generation_profile`, `world_manifest` and `conditioning_preview` all accept and route `macro-a4`.
- **Manifests and cache identity.** A4 has its own `macro-continental-bootstrap-v1-A4` identity, and the complete manifests hash `terrain_macro.py`, `terrain_world.py`, the stats file and the climate sources. `neural_manifest` rejects a world configured with the wrong profile. A0–A3 code paths are unchanged.
- **Rust reuse claims.** I checked against `coast.rs`:
  - The capsule distance matches `coast.rs:341-344` exactly.
  - The land ramp `height*smooth(distance/ramp)` matches `coast.rs:428`.
  - The sea side does **not** match Rust (×2 shelf plus an extra deep term, where Rust uses `ramp*4`). The doc only claims the land side, so this is accurate.
- **Probe decoding.** The probe's `ch0/weight` then signed square works, because coarse channel 0 is the denormalised signed-sqrt elevation (`world_pipeline.py:1049-1056`).
- **Site coordinates.** The validator's coarse cell, native pixel and conditioning resampling all use the same cell-centre convention.
- **Global probe result.** A4 NN land fraction is 0.334, exactly equal to the conditioning, with sign agreement 1.0 and 3 masses kept. This shows the NN keeps the large-scale layout at those points; it is not an LOD11 area result.

## Findings

### P1 — The coastline sits on a flat, near-sea-level band; the NN turns it into flat land and ponds
`terrain_macro.py:390-394`. Both the land ramp and the shelf use smoothstep, which has zero slope at the shoreline on both sides. With ramps of 70–145 km:
- One coarse cell (7.68 km) offshore is about −2 m; two cells is about −7 m.
- Inland is about +33 m per 1,000 m of inland height.
- So the first 1–2 coarse cells on each side of every coast are within a few metres of sea level. This was harmless in Rust at metre scale, but not at 10–145 km.

What the NN produced:
- **coast_seed42 A4:** conditioning land is 50.7%; the NN DEM is 97.5% land with **269** water components. At least ~4% of the crop lies between −0.09 and 0.82 m. The board (`.../d6095794.../bootstrap_coast_seed42/dem-shaded.png`) shows the shelf replaced by a featureless lowland with polygonal patterns. A0 at the same coordinates has 18 water components and its sea floor goes down to −3,662 m.
- **coast_seed0 A4:** conditioning land is 54%; the DEM is 72.6% land with 35 water components. The deepest point across about 15 km of "sea" is −11.8 m.

The flat band is proven from the formula. That it causes the NN result is a strong but unproven hypothesis; only 2 sites were measured.

**Fix:** give the shoreline a non-zero slope. For example, use an ease-out on land (`1-(1-t)^2`) and a linear or ease-out start for the shelf, then add a test for minimum |gradient| across the shoreline at coarse cells. Rerun both coast sites.

### P2 — The ridge sites sit on hard-clipped plateaus
`terrain_macro.py:383,395-396`; `terrain_bootstrap_preview.py:73`.
- Before clipping, inland height can reach about 7.6 km. It is hard-clipped to 5,814.07 m.
- The ridge site is chosen at the global maximum, so by construction it lands on the clip. In the seed0 30.72 km crop, the **median** is exactly 5,814.07, so at least half the crop is flat. For seed42, at least 5% is.
- The NN still produced relief (4.7–7.6 km, above the input support), so this is not a proven output defect. But the "ridge" gate is testing a clipped mesa, not a crest.

**Fix:** use a soft saturation toward the clip value, report the clipped land fraction, and pick the ridge site from the unclipped crest maximum.

### P2 — Global-probe topology numbers are misleading
`terrain_bootstrap_validate.py:84-85` reuse `topology` on a grid of points 1,250 km apart.
- Each point stands for 1.5625M km², so `land_components_at_least_1M_km2` counts every single isolated point as a continent.
- Points 1,250 km apart that touch diagonally count as connected. A0 then appears as one landmass holding 93% of land (343,750,000 km²), while the 1024-wide CPU board gives A0 only 10% of land in its top five masses.

**Fix:** keep only land fraction and sign agreement for the probe, or rename these as point-lattice metrics and drop the km² fields.

### P2 — The "matched native conditioning" is not what the NN saw
`terrain_bootstrap_validate.py:111-119` evaluates the analytic A4 field at 30 m. The NN only sees 7.68 km cells, which is 4×4 values over a 30.72 km crop. So the land-fraction comparison between "conditioning" and DEM is partly sub-cell interpolation. Doc line 75 ("coast input spans −9 to 1,031 m") has the same problem.

**Fix:** also report the conditioning bilinearly upsampled from the coarse cells, as `terrain_reference.py:425-431` already does, and label which is which.

### P2 — The land-fraction test cannot fail
The test's 512×256 lattice in `test_terrain_bootstrap.py:19-20` is the same lattice used to calibrate `sea_offset_m` (`terrain_macro.py:345-348`). That makes the 0.28–0.36 bound guaranteed by construction.

**Fix:** test on an offset or different-resolution lattice.

### P2 — The CPU-evidence guard checks too few files
`terrain_bootstrap_validate.py:45-48` compares only `terrain_macro.py`. The A4 fields also depend on `terrain_world.py` (latitude, lapse rate, source distributions) and on the stats JSON. The stats JSON is hashed in the CPU report but never compared. `topology` is imported from the hashed preview script, which is also not compared.

**Fix:** compare every `source_sha256` entry and add `terrain_world.py`.

### P2 — The climate bounds come from the natural factory, and the 20.6% is not only polar
- The BIO1 floor of −17.5 °C is the natural factory's own clip (`clip(-10)` then ×1.25), not the checkpoint's training support (z = 3.99 there).
- The bounds are recomputed per seed from one natural sample grid (`terrain_macro.py:152-153`), so for example BIO15's upper bound is 194.97 for seed 0 and 198.63 for seed 42.
- Because the lapse rate is applied before clipping, some of the 20.6% is likely high mountains at mid latitudes, not just the poles. That is plausible from the formula but not measured.

**Fix:** split the pre-clip fraction by latitude band and by elevation band, and use seed-independent bounds.

### P2 — Doc and reporting precision
- The ramp formula gives 10.5–145 km, not the "12–150 km" in the doc.
- "Five continental supports" merge into **2–3** masses of at least 1M km² (seed42: 150M and 106M km²). "Top-5 = 99.9%" should be stated together with that count.
- Ocean depth is narrow: median −3,518 m, minimum −4,437 m (natural: −2,274 m and −8,714 m), with no trenches or ridges. Acceptable for this block's land priority, but it should be stated.

## Outstanding gates before promotion
1. Fix the coast datum (P1), then rerun the coast boards for both seeds and compare them visually against A0.
2. Soft elevation saturation, and a ridge site on a real crest.
3. Visual review of the native and LOD11 boards. I looked at one board, but I am not certifying visual quality.
4. A real LOD11 area-mean check (the 32×16 probe is not one), plus more seeds and sites.
5. Climate clipping split by latitude and elevation.

The scope is appropriate: no work from the procedural, erosion or Rust-linkage phases was pulled in or needed. I made no edits and ran no commands.
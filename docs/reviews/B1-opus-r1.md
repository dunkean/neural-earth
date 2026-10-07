I recommend **CHANGES REQUIRED** for B1. Nothing here touches the protected `natural` default, and the manifest/reference part could be accepted on its own. The A1–A3 experiment and its E1 evidence can't support any conclusion yet: one defect undermines the A1 ablation, and the boards can't show the forms the visual gate judges.

This was a read-only review: I ran no code. Reproductions are commands to run, and values marked "expected" are inferred from the code and from `e1_report.json`, not measured.

## Proven defects

**P1-1 — A1 keeps the climate computed for A0's elevation, so it doesn't isolate geography.**
- **Where:** `terrain_macro.py:248-250`. `_natural_at()` (`:256-263`) runs `finalize(raw)` using A0's elevation, which applies the lapse rate, the −10 °C clip and the BIO4 baseline. Only afterwards is the elevation swapped for the macro one. `test_terrain_macro.py:23-24` asserts this behaviour as correct.
- **Effect:** a cell that was a mountain in A0 keeps a cold climate in A1 even when A1 makes it a 200 m plain. Ocean that becomes land keeps its warm sea-level climate.
- **Evidence in `e1_report.json`, site `mountain_seed0`:**
  - A1 and A3 have identical conditioning elevation (94–304 m).
  - A1's DEM comes out at 2,425–3,298 m; A3's at 271–302 m.
  - The only input difference is climate. This matches A0's mountain (DEM 4,967–6,230 m) reappearing through temperature.
  - The same inflation shows on A1 plains: coarse-stage maxima reach 2.4–3.6 km from conditioning that never exceeds 343 m.
- **To reproduce:**
  - `f0=make_macro_factory(0,'A0').sample_raw(699,-529,763,-465)` and `f1=…('A1')…`
  - Then `raw=NaturalConditioning(0).sample_raw(...)`, set `raw[0]=f1[0]`, and run `NaturalConditioning(0).finalize(raw)[1]`.
  - Expected: this BIO1 differs from `f1[1]` by up to tens of °C at the cell centre.
- **Fix:**
  - In `sample()`, substitute the macro elevation into the raw stack before `finalize` for A1/A3.
  - Replace the test with "climate equals `finalize(raw with A1 elevation)`".
  - Rerun E1 for A1. Until then, no A1 conclusion is valid.

**P1-2 — A2/A3 climate has no bound on the range the checkpoint has seen.**
- **Where:** `terrain_macro.py:205-232`.
- **What's missing:**
  - Natural clips BIO1 to [−10, 40] before its stretch, so its floor is −27.5 °C (`:87-88`).
  - A2/A3 draw quantiles from latitude bands all the way to 90°, including Antarctica (`terrain_world.py:79-88`). They then add the lapse rate for up to about 6.9 km of elevation, with no clip.
  - The upstream statistics exclude latitudes beyond ±60°, which is a third of this world's latitude range.
  - Elevation *was* clipped for exactly this out-of-range reason (`:198-201`); climate wasn't.
  - The negative-climate projection exists only for the `earth` profile (`terrain_inference.py:200`).
- **Status:** the missing bound is proven from the code; how far values actually go out of range is not measured.
- **Fix:**
  - Measure, per channel on the global 256×128 lattice, the fraction of cells outside A0's support and the z-score against `coarse_means/stds`.
  - Then pick a policy: clip/compress to A0's support, or limit latitude to ±60° plus a documented out-of-range band.
  - Add a test.

**P1-3 — The E1 boards and report can't support the coast/mountain/plain visual gate.**
- **Crop too small:** the native crop is 128 px, i.e. 3.84 km, half of one 7.68 km coarse cell (`terrain_reference.py:241`). The corpus field `native_dem_window_pixels: 256` is never read.
- **Palette saturates:** `PALETTE_STOPS` tops out at 6,000 m (`:21-25`). The A3 mountain DEM is 6,115–6,878 m, so `macro_mountain_seed0/neural-comparison.png` is a uniform white square. A0's mountain is partly saturated too.
- **No relief shading:** there's no hillshade or profile on the neural boards (`:142`).
- **Missing levels:** the learned coarse and latent stages are saved as arrays only, so the continental and regional levels the audit requires (§13.1) have no board.
- **Footprints don't match:** `write_e1_report` (`:164-201`) summarises conditioning and coarse over 491 km, latent over 15.4 km and the DEM over 3.84 km. That can't show at which stage a coast or mountain disappears, and the docstring's "identical native pixels" is true only for the DEM.
- **Focus point:** the default focus `ci*256` (`:121-122`) is a cell corner, not the centre.
- **Fix:**
  - DEM crops of at least 1,024–2,048 px (30–60 km) with hillshade.
  - A learned-coarse board over about 491 km and a latent low-frequency board over about 250 km.
  - Auto-range legends alongside the fixed palette.
  - Per-stage statistics resampled onto the same footprint.
  - Coast-crossing profiles.

**P1-4 — The E1 evidence isn't tied to a world identity.**
- **Wrong numerical profile recorded:** manifests are built with `inference_profile={}` and default strings `bf16` / `cuda-resident-v1` (`terrain_reference.py:204-219`). The neural run uses `choose_profile(world.device)` (`:257-258`), which is never recorded.
- **Evidence came from a different script version:** `index.json` contains `neural_sites`, which the current `terrain_reference.py` never writes (`:266-268`). That file isn't in the hashed implementation list, so the run can't be verified.
- **Stale output:** there's an extra `A0-neural/` directory from an earlier layout.
- **Unverifiable claim:** whether E1 was rerun after the documented ridge-uplift reduction can't be confirmed read-only. The manifest hash of `terrain_macro.py` would need comparing against the current file.
- **Fix:**
  - Write `world_identity(build_manifest(..., inference_profile=asdict(world._terrain_profile)))` into each neural `metrics.json` and `e1_report.json`.
  - Hash `terrain_reference.py`.
  - Refuse to write the report if the manifest doesn't match the files currently on disk.
  - Purge the stale output.

## P2 — cache identity, contracts, hardening

- **Manifest describes intent, not the runtime.** `terrain_manifest.py:115-128` hard-codes `cond_snr`, `drop_water_pct`, windows, strides and steps instead of reading the world's config. `checkpoint.revision` (`:110`) is claimed even for an arbitrary `checkpoint_source`. The server already has to patch the `earth` manifest by hand (`terrain_server.py:132-137`). **Fix:** derive these from the config/kwargs and assert they match the checkpoint.
- **Identity is too broad.** `:59-76` hashes `terrain_server.py`, `terrain_app.py`, `terrain_jobs.py`, `terrain_disk_cache.py` and `terrain_background.py`. Any server edit therefore invalidates the global coarse cache keyed by `world_hash` (`terrain_coarse.py:88`), which is about 120k coarse forwards to rebuild. **Fix:** split into a generation hash and separate build provenance, and test that a server-only edit keeps the hash.
- **Version name never bumps.** `MACRO_VERSION` (`terrain_macro.py:20`) stayed the same after the ridge change. Tile and physical cache keys use `macro-a3` plus `VERSION`, not the hash (`terrain_server.py:495-496, 549-550`). Only native-mip children check identity (`:344`). This is a B3 gate. **Fix:** golden-value tests pinning a few samples per ablation, which force a version bump, plus hash-keyed caches.
- **A3 basins create uncalibrated inland seas.** `terrain_macro.py:197` subtracts up to 500 m on land after the sea threshold is calibrated, so A3's land fraction is no longer the calibrated one. The 0.25–0.55 test is loose. **Fix:** floor basins above 0 unless they're explicit lakes, and report A3's global land fraction against the target.
- **A3 coasts are flat by construction.** Land rank rises steadily with the continent score, and ridges are multiplied by `shore` (`:183-195`). Coasts therefore always start as low plains; E1 shows −0.9 to 6.9 m. This is a design limitation, not a bug. **Fix:** relief that doesn't depend on distance to the coast, and ridges allowed to reach the shoreline.
- **Possible repeated values in `sea_knots`** (`:156`). `np.interp` requires strictly increasing breakpoints; whether ties actually occur is unproven. **Fix:** apply the same strict-increase guard as `build_quantiles`.
- **Digest cache headroom.** `_cached_digest` has `maxsize=64` and 47 files are hashed today. Past 64, the least-recently-used cache would rehash the 1.1 GB weights on every call.
- **Small items:**
  - Dead `ci0`/`cj0` at `:245-246`.
  - `earth` would produce an "TH" ablation directory (`terrain_reference.py:131`).
  - The constants 38.6/−31.4 are duplicated from `world_pipeline.py:1377`.
  - `write_e1_report` has no unit test.
  - The doc says "model units" where the comparison is in physical units after the square root.

## Measured observation (not a code defect)

A2 changes only climate, yet at `coast_seed0` it removes the protected A0 coast: land goes from 50% to 0%, with the DEM at −2.06 to −0.70 km. The other A0-coordinate sites also shift by hundreds of metres. The coarse model's elevation is highly sensitive to climate.

Any climate scheme must therefore be judged on whether protected sites survive. A useful extra variant: latitude organisation that keeps natural's clip/stretch and BIO4 regression, to separate "organisation" from "distribution shift".

## Confirmed sound

- A0 matches upstream on square windows, and seed 0 is fixed to seeds 1–5.
- The natural path is unchanged for non-zero seeds.
- Cell-centre and row/column conventions are consistent with `world_pipeline.py:886-998`.
- Macro objects and calibration are independent of the requested crop.
- Switching profiles rebuilds the tile store, so A0 and A1 windows don't mix.
- Seeds are stored as decimal strings.
- Physical units are correct.
- The square root is applied once.
- Memory per chunk is bounded.

## Verdict: CHANGES REQUIRED

**Outstanding gates:**
1. Fix P1-1 and P1-2, then rerun E1.
2. Rebuild the boards and co-registered report (P1-3).
3. Bind the evidence to `world_hash` plus the actual numerical profile (P1-4).
4. A natural regression crop compared against `audit-implementation/baseline`; there is currently no evidence that natural is bit-identical.
5. NN-level coverage of the seed-42, negative-coordinate and window-border sites, plus observed good/bad locations.
6. Human visual approval.

I make no claim here about visual quality or performance.

Separately, the Google Calendar, Google Drive and Sentry connectors need authorising in your claude.ai connector settings before they can be used.
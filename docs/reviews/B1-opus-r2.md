# B1 review, corrections r1

**Verdict: CHANGES REQUIRED, but only one narrow item.** One P1 cache-identity defect has to be fixed before the core freeze and the attested rerun. Once that's fixed, the B1 code can be accepted with limitations. `natural` stays the default, and A1–A3 remain experimental.

This was a read-only review: I ran no code or tests. Line numbers refer to the files as read today. Where I say "expected", the value is inferred from code, not measured.

## Previous P1s: status

| Previous finding | Status | Evidence |
|---|---|---|
| Opus P1-1: A1 climate tied to A0's elevation | **Fixed** | `terrain_macro.py:259-263` puts the macro elevation into the raw stack before natural `finalize`. The test at `test_terrain_macro.py:26-29` now checks the coupled version. |
| Opus P1-2: A2/A3 climate unbounded | **Fixed as a contract; quality limitation remains** | Each channel is clipped to the per-seed A0 lattice range (`:240-242`), with diagnostics. The stored seed-0 A3 diagnostic shows 20.6% of BIO1 pinned at −17.5 °C (z p99 = max = 3.99). |
| Opus P1-3: boards and footprints | **Mostly fixed** | Three sites now use 1024² crops, the palette reaches 9,000 m, and hillshade, profiles and coarse/latent boards exist. Footprints are labelled and a native-pixel resampling metric was added. Gaps remain (P2-4, P2-9). |
| Opus P1-4 / Astra R1-01: evidence not bound to the actual run | **Fixed, except the P1 below** | `neural_manifest()` records the real execution profile, `VERSION`, kwargs and capability. NPZ, metrics and manifest are bound together, and stale or mixed artifacts are refused. |
| Astra R1-02: summaries over different footprints | **Fixed** | Summaries are relabelled as context and get explicit footprints. Cross-ablation DEM comparisons require identical native bounds. |

**Contracts I verified:**
- **Conditioning coordinates:** the factory argument order (`cj0, ci0, cj1, ci1`) matches `world_pipeline.py:998`.
- **Coarse grid:** coarse windows start at `i*48`, so the coarse index matches the conditioning cell (`terrain_inference.py:147`).
- **Pixel-centre resampling:** the convention in `_sample_to_native` matches `_compute_climate` (`world_pipeline.py:1439`).
- **Coarse channel 0:** it is signed-sqrt elevation, already denormalised (`terrain_inference.py:189`).
- **Low-frequency constants:** 38.6 / −31.4 match `world_pipeline.py:1377`.
- **Climate channel order:** correct.
- **Profile and seed switches:** both rebuild the conditioning (`world_pipeline.py:749, 770`).
- **Seed 0:** the fix is in the production A0 path (`terrain_inference.py:367`).

**The refusal on the historical run was correct.** `reference-b1r1/manifests/neural/` holds **11** hashes for 8 seed/ablation combinations. A0, A2 and A3 at seed 0 each have two hashes, so the sources changed mid-run. All of these manifests record `latent_batch=16`, `canonical_latents=false`, bf16 and `cuda-resident-window-v2`. Those 32 arrays should stay historical.

## P1

**P1-1: the world hash leaves out code the NN run executes.**
- **Where:** `terrain_manifest.py:59-76`.
- **What's missing:**
  - `edm_unet.py:12` imports `unet_block.py`, which isn't hashed. `UNetBlock.forward` (`unet_block.py:116`) runs on every forward pass. `terrain_nn_constants.py:80` replaces only the attention part.
  - `terrain_reference.py` (the exporter itself) and `terrain_diffusion/common/model_utils.py` aren't hashed either.
- **Impact:** editing the U-Net residual block changes the terrain without changing `world_hash`. The B2/B3 persistent coarse cache keyed on that hash would then serve stale data, and `verify_manifest_files` would still pass the run.
- **Reproduce:** call `build_manifest(0, file_hashes=True)`, add a line to `unet_block.py`, rebuild. The `world_hash` is identical.
- **Fix:**
  - Add `models/unet_block.py`, `common/model_utils.py` and the package `__init__.py` files to the list.
  - Add a test that, after `configure_world` + `bind` on a fake or CPU world, every loaded module from `sys.modules` under `ROOT` or `terrain-diffusion/` is in `manifest["files"]` (or explicitly allow-listed).
  - Verify the exporter separately (see P2-1) rather than adding it to the world identity.

## P2

**P2-1: attestation is too broad and also blind to the exporter.**
- **Too broad:** the reference run doesn't import `terrain_server`, `terrain_background`, `terrain_jobs`, `terrain_disk_cache`, `terrain_coarse`, `terrain_final_mips` or `terrain_climate`, yet edits to them refuse attestation. That is exactly what blocked this run.
- **Exporter unchecked:** `exporter_sha256` is only compared between an NPZ and its own `metrics.json` (`terrain_reference.py:396`). It is never compared between the CPU and NN artifacts, or against `reporter_sha256`.
- **Fix:**
  - Split a generation identity (the execution path only) from build provenance (server files), as Opus asked in r1.
  - In `write_e1_report`, require every artifact's `exporter_sha256` to equal `_sha256(__file__)`.
- **Practical point:** until the split is done, the final run needs the whole B3 lane frozen for its full duration, not just the GPU lane.

**P2-2: the reference DEM depends on run order and crop size.**
- **Where:**
  - One global `pipeline` and its tile store are shared across sites (`terrain_app.py:59-88`; loop at `terrain_reference.py:585-597`).
  - The latent profile is batch-16, non-canonical (`terrain_inference.py:67`).
- **Measured elsewhere:** B2 found a 0.21 m difference between a whole crop and its sub-crops for this profile (`AUDIT_IMPLEMENTATION.md:36`).
- **Impact:** that's negligible for A0-vs-A3 differences of hundreds of metres. It is not negligible when this run later serves as the regression baseline for flat coasts.
- **Fix:**
  - Call `world.rebuild()` before each site/ablation export, and record the requested crop size in the report.
  - Compare future runs only at the same crop size.

**P2-3: the report's "conditioning" stage isn't what the network actually consumed.**
- **Where:** `export_conditioning_site` re-derives the conditioning on the CPU (`terrain_reference.py:157-160`). For A0 that uses the reimplementation, not the upstream factory the NN actually used.
- **Fix:**
  - Save `world._conditioning_model_input(ci-32, ci+32, cj-32, cj+32)` in `stages.npz`.
  - Assert it equals the CPU `model_input`: exactly for macro, within 1e-3 for A0.
  - This also guards against any future factory mis-wiring.

**P2-4: the matched-footprint metric is too small for its purpose.**
- **Problem:** with 256² crops, `matched_native_footprint` covers 7.68 km, which is one coarse cell (`:479-489`). Its land/sea disagreement for "conditioning" or "coarse" just says whether that one cell is land. Yet conditioning and coarse are already on the same 64² grid (`cb == nb`), and no per-cell comparison exists. That comparison is exactly the "where does the coast or mountain disappear" measure the audit asks for (§6.3).
- **Also:** `map_coordinates(mode="nearest")` (`:429`) silently clamps if a footprint ever falls outside its source.
- **Fix:**
  - Add a `matched_coarse_grid` block: MAE in metres and in sqrt space, signed bias, land/sea disagreement and correlation over the 491 km footprint.
  - Do the same on the latent grid.
  - Assert that the native footprint lies inside each source footprint.

**P2-5: one A3-selected NN site sits in the clipped-climate regime.**
- **Where:** `macro_plain_seed0` has `center_ci=1202`, which is about **83°S** (`terrain_world.py:34-36`). The 80–90° WorldClim band is far below the −17.5 °C floor.
- **Consequence:** A2/A3 BIO1 is expected to be pinned across that window. There the site measures the clip, not the A3 design, and the elevation–temperature coupling is lost.
- **Reproduce:** run `make_macro_factory(0,'A3')._climate(xx, yy, elev, clip=False)` over that site's bounds and compare with `climate_support[0]`.
- **Fix:**
  - Report, per site and channel, the fraction of cells at the support bounds.
  - Either re-select an A3 plain within ±60° or label this site as clip-regime evidence.

**P2-6: the `coarse_window_border` crop doesn't contain a window border.**
- **Where:** the default focus is (−12160, 12416), so the crop spans 12288–12544.
- **Problem:** 12288 = 48·256 = 32·384, so the crop's edges sit exactly on the origins of the coarse, latent and decoder window grids. No latent window edge lies inside the crop.
- **Fix:** set `focus_native_i/j` to −12288 / 12288 so the border runs through the middle of the crop.

**P2-7: A3 basins can still create uncalibrated inland seas** (Opus r1 P2, still open).
- **Where:** `terrain_macro.py:204` subtracts up to 500 m after the sea threshold has been calibrated. The 0.25–0.55 land-fraction test is loose, and A3's actual land fraction isn't reported against `water_fraction`.
- **Fix:** keep basins above 0 m unless they are deliberate lakes, and report A3's global land fraction against the target.

**P2-8: misleading manifest labels.**
- **Hard-coded values:** `conditioning.cond_snr`, `drop_water_pct` and windows/strides are hard-coded (`terrain_manifest.py:115-128`) and can contradict `generation.checkpoint_kwargs`. The server already patches `cond_snr` to 0.1 for `earth` (`terrain_server.py:146`).
- **Revision:** `checkpoint.revision` is claimed even for an arbitrary checkpoint source (`:110`).
- **Fix:** derive these values from the world or config, or remove them.
- **Related, still open:** the `sea_knots`/`land_knots` arrays lack the `maximum.accumulate` guard used in `terrain_world.py:132`. This is unproven hardening, not a demonstrated bug.

**P2-9: the diagnostic boards are missing elements.**
- **No footprint markers:** the coarse and latent boards don't show where the DEM and latent footprints sit.
- **No legend or scale:** there is no palette legend, no scale bar, and no auto-range view next to the fixed palette.
- **Downscaling:** 1024² crops are shrunk to 512 with `NEAREST` (`terrain_reference.py:312`), which risks aliasing. The `coast_seed0` board I looked at shows no obvious artifacts, but use `BOX`/`LANCZOS` for downscaling.
- **No A0 baseline:** the climate diagnostics give z-scores without an A0 row for comparison.

**P2-10 (low): 31-bit noise seeds.**
- **Where:** `terrain_macro.py:27, 51, 142`.
- **Effect:** seeds that differ by 2³¹ share the same conditioning noise. For A0 this is inherited from upstream. For the macro variants it can be fixed by deriving noise seeds with `SeedSequence`, with a version bump.
- **Fix:** at minimum, document it in the manifest.

## Not certified
- **Visual quality:** I looked at two historical boards. The `coast_seed0` hillshade board is readable. The flatness of the A3 coast matches the documentation. I'm not certifying quality.
- **Performance:** none was measured.
- **The 20.6% BIO1 pinning:** this is a physical-coherence limitation at high latitude, not a code defect. Use soft compression or limit the latitude range before judging A2/A3 climate.

## Outstanding gates
1. Fix P1-1. Before the freeze, ideally also P2-1, P2-2 and P2-3.
2. Freeze the core, including the upstream U-Net files and the B3 lane. Rerun into a new directory. `e1_report.json` must be written.
3. Compare a `natural` regression crop against `audit-implementation/baseline`; there is still no evidence `natural` is unchanged.
4. Decide the A2/A3 high-latitude climate policy, and fix P2-5 and P2-6 in the corpus.
5. Run more sites through the NN: seed 42, stratified samples, and the user's known good and bad locations.
6. Human visual approval.

The Google Calendar, Google Drive and Sentry connectors need authorising in your claude.ai connector settings; they can't be used until then.
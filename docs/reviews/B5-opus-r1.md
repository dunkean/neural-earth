# B5 review: final integration, natural reference, navigation, coarse world and WebGPU evidence

**Verdict: ACCEPT WITH LIMITATIONS.** I found no P0 or P1 code regression in the CUDA path. Two P2 items are measurement and documentation problems in B5's own deliverable: the frame-time measurement and how the browser evidence is cited. Fix both before B5 is called "final measurements". The other P2s are robustness, performance, or gaps in evidence. All product gates that are still open are listed separately at the end.

This was a read-only review: I edited nothing and ran no commands. `index.html` changed while I was reading it: the `pump()` change that serializes validations appeared between two reads. My findings use the later version (`index.html:80-81`). Without running commands I can't hash the current files against the JSON reports.

## What I checked and found correct

- **Climate.** `terrain_climate.py:35-48` uses the same coarse context, 15-cell regression and centre alignment as upstream `_compute_climate` (`world_pipeline.py:1418-1454`). It stores sea-level baseline plus lapse rate, and BIO channels 3:6.
- **Coarse units.** The persisted weighted windows hold denormalized signed-sqrt height (`world_pipeline.py:1049-1056`). So `read_mip` and `sample_coarse_area` average physical metres correctly (`terrain_server.py:357-362`).
- **Readiness.** `ready_for_samples` uses the same dense and sparse partitions as `sample_field`, and the climate rectangles join up across groups. The climate apron (9 cells) is exactly in the plan, and LOD ≥9 area chunks are pre-checked.
- **Exact mips.** `terrain_final_mips.py` takes only each child's own interior and publishes nothing if a child is missing. Child identities are checked (`terrain_server.py:453-472`).
- **A0 preview.** Seeds, octave counts and seed-0 handling match the upstream factory (`terrain_macro.py:51`, `synthetic_map.py:183`).
- **Identity.** The manifest hashes 107 upstream files plus `infinite_tensor`, model files, data sources and the statistics file. The physical cache path includes the generator identity (`terrain_server.py:72-75`).
- **Evidence.** `natural-reference-fidelity.json` is a real reproduction under the final code: the cached packet was generated in the same session (Seconds header 5.7064). `coarse-world.json` and the E1 report agree with VALIDATION_FINAL.

## Findings

**P2-1: the frame-time gate is measured with an instrument that cannot pass, and the "pan" is not a pan.** (`benchmark_browser.cjs:15-19, 29`; `index.html:94`; `VALIDATION_FINAL.md:29-30`)
- **Reproduction.** The harness sends `ArrowRight` every 500 ms. Each keypress jumps the camera by 100 px × 30 m = 3 km. "6 km/s" is therefore four discrete jumps in 2 s, not a continuous pan.
- The metric is the interval between animation frames. On a 60 Hz display that interval is at least 16.67 ms by construction, so a p95 of 16.8–16.9 ms is display-timing jitter. It is not evidence that frames are too slow.
- The 0.2–0.3 ms "camera handler" figure measures only the synchronous key handler. The actual drawing (`drawFrame` and the WebGPU submit) happens later and is never timed.
- **Fix:**
  - Drive the camera every frame during the test.
  - Record per-frame work time from the start of the frame callback to the end of the draw.
  - Record the refresh rate and the share of frames longer than 1.5× the refresh period.
  - Keep coverage lag as a separate measure.
  - Restate the <16.7 ms gate as **unmeasured**, not "fails".

**P2-2: VALIDATION_FINAL cites browser evidence that doesn't match the final client.** (`VALIDATION_FINAL.md:20-36`; `benchmark_browser.cjs:25`)
- `browser-navigation.json` and `browser-prepared-world.json` were run with `index.html` hash `04ddf…`, before the fix. `browser-promotions.json` and `browser-final-client.json` used `e0631…`. The document still calls them all "sources figées" (frozen sources).
- In the prepared-world run, the "initial-world-settled" view was **100% `conditioning-preview`** even though 6160/6160 coarse windows were persisted. That is the defect Astra found and fixed.
- The document doesn't mention:
  - the 180.9 s timeout before the fix;
  - the 25.4 s run after it, which started with a partly warm cache;
  - the newer `browser-final-client.json`.
- The scenario named "cold-native-reference-region" was served entirely from cache (42/42 hits) in both prepared-world and final-client runs.
- **Fix:** add a client-hash column, mark the pre-fix runs as historical, cite the promotion and final-client evidence, and rename the scenario `native-revisit`.

**P2-3: promotion from preview to learned terrain depends on a world being in memory and on a non-blocking grab of the global GPU lock.** (`terrain_server.py:475-503`)
- After a server restart, the readiness check (`learned_tile_ready`) finds no world in memory and returns False. Cached previews are then served indefinitely, until something else loads a world: a view below LOD 7, a cache miss, or automatic preparation. With `prepare=0` none of these may happen.
- `index.html:80` stops validations competing with each other. But any GPU job in progress still makes the check fail, and the client then doubles its retry interval up to 30 s (`index.html:85`).
- **Fix:**
  - Use a dedicated lock for the readiness ledger instead of the global GPU lock.
  - Build a metadata-only readiness check from the coarse cache's on-disk `contract.json` and window metadata, without loading any models.
  - Return an `X-Terrain-Validation: deferred` header so the client doesn't back off.

**P2-4 (missing evidence): numerical settings are neither pinned nor part of the identity.** (`terrain_manifest.py:76-81`; `terrain_app.py:59-75`)
- Nothing anywhere sets or records cuDNN `benchmark`/`deterministic`, TF32 or matmul-precision flags.
- The persistent caches assume bit-identical replay. cuDNN's heuristic can fall back to a different algorithm when workspace memory is short, which would change results without any visible signal.
- **Fix:** set these flags explicitly when the pipeline loads, add them to `inference_profile`, and run one replay under VRAM pressure.

**P2-5: persistence takes 25% of world preparation (112 s of 439.6 s).** (`terrain_coarse.py:395-409, 450`)
- Each new window is checked twice: `_save_window` re-reads its header and weight plane, then `step` runs `_load_window` (full read, SHA-256, `np.load`, finiteness check).
- The first `install` also validates all 6160 windows while holding the GPU lock.
- **Fix:** have `_save_window` report success and skip the second check for windows just written. Re-measure. This is performance only.

**P2-6: tiles at the world edge are inconsistent across LOD levels.** (`terrain_server.py:373-387`, `625-633`)
- LOD 4–6 sample positions are not clipped to the world bounds; LOD ≥7 are.
- A LOD 6 tile that overlaps the edge reaches about 70 coarse cells beyond the planned 9-cell apron. That triggers unplanned foreground coarse generation and disk writes, and the edge shading differs from LOD 7.
- **Fix:** clip the coarse-stage samples at LOD 4–6 the same way as LOD ≥7. Leave the native LODs (0–3) unclipped.

**P2-7: `CoarsePreparation.read_mip` is unused and its bounds are wrong.** (`terrain_coarse.py:547-566`)
- It claims to accept `level` up to 12. But the grid is 2606 rows and a level-12 block is 4096 rows, so level 12 always raises. Its origin is relative to the grid, not the global tile lattice.
- **Fix:** delete it, or align it to the global origin and use it to build an area-mean pyramid once after preparation. That would directly attack the 8.3 s reconstruction of a LOD 12 tile.

**P2-8 (not reproduced, Windows only): file-handle race on PNG tiles.** (`terrain_server.py:815-835`)
- `tile()` deletes or atomically replaces a PNG that another request may still be sending via `send_file`. On Windows that raises `PermissionError`, which is not caught, so the request returns a 500. The client's retry hides it.
- **Fix:** catch `PermissionError`, or write promoted tiles under new file names.

## Open product gates (not regressions)

- **E1:** no human visual approval yet. The A3 coast is still flat. `natural` stays the default.
- **E2:** NO-GO. The browser crop's maximum height error is 4.79 m against a 1 m limit (the JS CPU diagnostic reaches 0.51 m). The base model under NCHW layout still diverges. The evidence is bound to the source hashes that produced it.
- **E3:**
  - The first native view takes 5.2–5.7 s. The <250 ms target is met only from cache.
  - Sustained cold-pan throughput is unmeasured.
  - The 0.21 m dependence on batch/partition order is baked into persisted tiles, so tile bytes depend on navigation history.
- **E4:**
  - Each seed and each identity costs 440 s and 707 MB.
  - Reconstructing the learned world view costs 8.3 s per LOD 12 tile and about 25 s for one promotion pass.
  - Any server edit invalidates the coarse and physical caches, and nothing removes old identities.
- **LOD coverage:** exact DEM mips exist only at LOD 1–2. LOD 3 is still the latent low-frequency approximation, without the decoder residual.
- **Pending receipt:** `coarse-persistence-fidelity.json` (world hash `311ab…`) and `window-scheduler-fidelity.json` predate the final 107-file identity. Re-run both under it, then update VALIDATION_FINAL with P2-1 and P2-2.

The claude.ai Google Calendar, Google Drive and Sentry connectors need to be authorized in your claude.ai connector settings before they can be used. None of them was needed for this review.
# B2 review: neural window scheduling and learned-coarse persistence

**Verdict: CHANGES REQUIRED.**

The persistence core is sound: weighted per-window saves, atomic writes, the geometry/dtype contract, replay from disk with zero model calls, and fused reads that never return a partial average. But this block's own acceptance gate is "entier/sous-crops, permutations". It is not met, and the docs admit it. On top of that, the learned coarse can never appear at world scale. I only read code; I didn't run anything. Measured numbers below come from the project's own reports. I can't certify visual quality or performance; no measurements of either exist for this block.

## P1: must fix before accepting

### P1-1 — The terrain depends on navigation history (proven, unresolved)
- **Where:** `terrain_inference.py:280-282`, and the design choice at `terrain_window_scheduler.py:34-38`. Batches are 16 latent windows (`latent_batch`) of whatever happens to be missing, unpadded. That gives a variable batch shape, and in BF16 a different numerical path (CUDA graphs for batches ≤4, eager above that).
- **Evidence:** `docs/WINDOW_IMPLEMENTATION.md:61-63` and the `off_full_vs_off_subcrops` field in `verify_terrain_windows.py:81` show 0.210693 m elevation difference between one whole read and four subcrop reads.
- **Consequences:**
  - A tile revisited after the 512 MB LRU evicts its latents is recomputed with a different batch, so it can come back with different heights. This variant hasn't been measured.
  - Neighbouring physical tiles saved to disk at different times can disagree on the latent field they share, which can show up as seams. On a flat coast, 0.2 m can move the shoreline a long way.
- **Reproduce:** the existing report. On CPU: make the mock `base_f` in `test_terrain_windows.py:47-57` add `len(indices)*1e-3`, and `test_full_crop_subcrops_permutations_and_eviction` fails. The current mocks give the same result whatever the batch contains, so they can't catch this.
- **Fix:** make latent batches canonical.
  - When any window in a fixed group is needed (for example `(i//4, j//4)`, 16 windows), compute the whole group in one batch of constant shape. Each window's value then depends only on its coordinates.
  - Alternatively, measure latent batch 1.
  - Then rerun on GPU: whole vs subcrops vs ≥3 permutations vs eviction-then-regeneration, expecting 0 m, on several seeds and on coastal sites.

### P1-2 — Tiles on the world edge never become "learned-ready" (proven by arithmetic)
- **Where:** `terrain_coarse.py:127-135` (the plan is the world plus 8 cells), `:214-215` and `:269` (windows outside the plan never count as saved), and `terrain_server.py:283`.
- **What happens:** `terrain_lod.js:11-16` still requests whole tiles that straddle the world edge. Their samples, plus the 24-sample halo and the climate halo, reach far past the plan.
  - At LOD 12 a tile spans about 31,457 km; at LOD 11, the ±15,729 km rows exceed the world's ±10,000 km height. So every LOD 11–12 tile, and the outer ring of tiles at LOD 7–10, stays on the conditioning preview even at `coverage == 1`.
  - The result is no learned world view, and a visible seam between learned and preview tiles.
- **Reproduce:** in `test_budget_resume_…`, after full coverage, call `prep.ready_for_samples(np.array([128., 4096.]), np.array([128., 128.]))`. It returns `False`: the request needs cell 25, the plan ends at cell 14.
- **Fix:**
  - Clamp the sample coordinates to `WORLD_BOUNDS` in both `sample_physical` and `ready_for_samples`, and return an explicit outside-the-world mask or no-data for those pixels.
  - Add a test that runs a straddling tile at LOD 7 and LOD 12 after full preparation.

### P1-3 — Free VRAM at startup changes the world's identity and its numbers (proven by reading the code)
- **Where:** `terrain_inference.py:54,66`. `latent_batch = min(16, ceiling)`, where `ceiling` depends on free memory: 8 below 10 GiB free. This flows through `terrain_server.py:51` into `inference_profile` and then into `world_hash`.
- **What happens:** the same seed on the same machine gets a different `world_hash` and different heights depending on what else holds VRAM at boot. The learned coarse cache is also thrown away, even though the coarse output doesn't depend on the latent batch.
- **This contradicts** the docstring at `terrain_inference.py:47-49`, which says memory "never [controls] generation quality".
- **Fix:** fix the numerical profile. Memory may only refuse admission, or select an explicitly distinct and declared profile. Once P1-1 is fixed, batch size becomes a pure performance knob.

## P2: should fix

- **Coarse cache identity is wrong in both directions** (`terrain_manifest.py:51-58`, `terrain_server.py:53-57`).
  - Too many inputs: any edit to `terrain_server.py`/`terrain_app.py`, a driver update, the CUDA ordinal or `latent_batch` invalidates about 707 MB (6160 windows × 114,816 B).
  - Too few: no GPU SKU or uuid and no cuDNN version. Two cards with the same compute capability share one cache.
  - Fix: a coarse-stage identity covering coarse weights, conditioning, solver, coarse batch, dtype, device SKU and library versions.
- **Nothing ties the manifest to the actual world** (`terrain_coarse.py:72-77`).
  - It never checks that `manifest.seed_u64 == world.seed`, nor cond_snr, profile or dtype.
  - The `earth` manifest declares `cond_snr [0.5]*5` and `world_profile: natural` (`terrain_manifest.py:82,93`), while the run uses 0.1. The hash still differs, but the stated provenance is false.
  - Fix: check all of these in `install()`.
- **No world bounds, and a full quota breaks requests** (`terrain_server.py:440` accepts |tx| ≤ 1e7; `terrain_coarse.py:210-211`).
  - Tiles below LOD 7 outside the world still run the networks, and their coarse windows are saved and count against the 2 GiB.
  - Once the quota is full, every new coarse window raises inside `f`, so normal requests fail permanently.
  - Fix: reject or clamp out-of-world requests. On quota, stop persisting and flag it in `status()` instead of failing the read.
- **Each background step reads whole files from disk** (`terrain_coarse.py:239,254`).
  - Every examined window gets a full `np.load` plus validation. On an already-complete world, one step reads about 707 MB while holding `gpu_lock`.
  - A damaged file whose window is still in RAM raises "exists only in RAM" every time and puts the background task in the failed state.
  - Fix: test membership with `_persisted_indices` and `exists()`, short-circuit when complete, and in the RAM case call the raw coarse `_f([index])`. The coarse stage has no inputs, so this is safe.
- **The instrumentation misleads** (`terrain_window_scheduler.py:61-85`, `terrain_coarse.py:162-164`).
  - Stage seconds include upstream time, and asynchronous CUDA means they mostly measure kernel launches. `network_seconds` stops before the synchronising `.cpu()` call.
  - `requests`/`cache_hits` also count internal dependency reads.
  - The `unique` sets grow without bound.
  - `quantum_batches` is never used.
  - Fix: CUDA events, exclusive per-stage timings, a separate counter for real external requests, and bounded counters.
- **`read_mip` isn't usable for tiles yet** (`terrain_coarse.py:339-345`).
  - "Aligned" blocks are aligned to the grid origin (−1303, −2605), not to the global tile lattice, and partial blocks at the edges are refused.
  - Neither the server nor the overview uses it. The overview (`terrain_server.py:700`) never gets replaced by the learned coarse.
  - LOD ≥ 9 point-samples the coarse without filtering, so it aliases. This is a B3 gate.
- **Integration problems.**
  - Background preparation calls `get_world`, which sets `active_seed` and evicts the user's world from the 2-world LRU (`terrain_server.py:183-186`).
  - The physical climate cache doesn't hash `terrain_climate.py`.
  - `terrain_climate.py` itself matches upstream: clamp to 0 m, half-cell alignment, the same 5 channels.
- **Validation doesn't prove what the docs say.**
  - The GPU scripts have no pass/fail thresholds: one seed, one crop, one permutation.
  - Cancellation, resume and eviction-then-regeneration are untested on GPU.
  - "manifest/geometry separation" is only tested for geometry (`test_terrain_windows.py:172-180`).

## Outstanding gates
1. P1-1 fixed, then the GPU gate: whole, subcrops, ≥3 permutations and post-eviction regeneration all at 0 m, on several seeds including coastal sites, with assertions in the scripts.
2. A straddling LOD 7–12 tile after full preparation is `learned-ready`, and the edge-seam test passes.
3. Two startups with different free VRAM give the same `world_hash`.
4. A GPU test of cancellation and resume, and of a corrupted window that is still in RAM.
5. E4 measured with synchronised timings: full preparation's wall time, model time, bytes, and lock-hold time.
6. Visual approval of the learned coarse at continental and world scale: not established.
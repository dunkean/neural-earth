# Review r2: BOOTSTRAP-RESTART

I only read files: the code, the tests, and the stored reports under `E:/TerrainDiffusionRuntime`. I ran nothing and edited nothing. The final NN QA run (`terrestrial-bootstrap-v2-nn/final`) still shows `"complete": false`, so I haven't certified it.

## Fixes from r1 that I confirmed
- **Natural no longer touches Rust or World Builder.** `terrain_manifest.py:65-66,78` leaves out `terrain_bootstrap.py` and the native identity for natural. `terrain_server.py:132,144-150` loads the native identity only for terrestrial profiles, and `test_terrain_identity.py:61-73` guards this. `terrain_server.py:29` imports constants only, with no git call or build.
- **Bootstrap errors become JSON.** `/api/world` returns a JSON 503 (`terrain_server.py:620-621`).
- **The heightmap is generated before the GPU lock.** It happens at `:210` and `:730`. Inside the lock only an LRU or NPZ load remains.
- **Rayon 1 vs 4 threads is tested at production size.** `test_terrain_bootstrap_native.py:126-148` uses 256 / 1024×512 with seed 2⁶⁴−1.
- **Climate seeds use the full u64 world seed.** `terrain_conditioning.py:117-128` hashes the whole seed with SplitMix64 and bumps the version.
- **Unused parameters are now null for terrestrial.** `terrain_manifest.py:135-137`.
- **The recorded noise level matches what runs.** The manifest's `cond_snr` equals what `configure_world` actually sets (`terrain_inference.py:440-442`), and `neural_manifest` records the real `world.kwargs`.
- **0.05 is inside the training range.** Training draws the conditioning noise level as `exp(U(−5,5))`, roughly 0.0067 to 148 (`coarse_dataset.py:420`). The noise-level input itself is not out of distribution.

## P1

**P1-1 (proven by reading; a regression introduced by the r1 P1-1 fix): terrestrial tiles survive a change of world identity, so a later bootstrap fix would serve old terrain.**
- **Where it happens:**
  - The cache root comes from `PROFILE`, which hashes only natural's identity (`terrain_server.py:68-71`).
  - Terrestrial tiles live at `CACHE/<profile>/physical-v1/<seed>/...` (`:699-703`), PNGs at `CACHE/<profile>/<seed>/...` (`:824-826`), and the overview at `:889-897`.
  - A cache hit (`:723-728`, `:828-833`, `:895-897`) never compares `report['world_identity']` with the current identity. `valid_physical_report` (`:498-502`) doesn't either; only the LOD 1–2 mip children check it (`:463`).
- **Why it now matters:** since natural no longer hashes `terrain_bootstrap.py` or the native identity, a change to `main.rs`, `terrain_bootstrap.py` (for example a hypsometry v3), ETOPO or the World Builder pin no longer moves `CACHE`.
- **What the user would see:** the client sends the new `world_identity` (`index.html:95`), which passes `_tile_coordinates` (`:634-637`). The server then returns the old height and climate tiles at LOD 0 and 3–11, plus the old overview, with an `X-Terrain-World-Identity` header naming the old world. Coarse windows are not affected because they are keyed by `world_hash` (`terrain_coarse.py:107`).
- **Reproduction:**
  1. Open `terrestrial-earthlike` seed 0 and load a few tiles at LOD 5.
  2. Change one byte of a comment in `terrain_bootstrap.py` and restart.
  3. `/api/world` reports a new identity, but `/height/.../5/x/y.bin` answers `X-Terrain-Cache: hit` with the old report.
- **Fix:**
  - For profiles other than natural, put `world_identity(...)[:16]` into the tile and overview paths. Alternatively, require `report.get('world_identity') == world_identity(world_manifest(seed, profile))` in `valid_physical_report`, and store and compare the identity in the overview's `world.json`.
  - Add a test that changes the identity and expects a cache miss.

**P1-2 (missing evidence, with a measured adverse signal): the 0.05 default was chosen on how closely the NN follows its input, and the stored data show relief loss.**
- **What was compared:** `tune_terrestrial_coasts.py:80` compares the NN coarse output only with its own input (`parent[0]`). Training feeds the true coarse field as the conditioning image (`coarse_dataset.py:421-422`), so lowering the noise makes the model copy the input more closely. Smaller coast displacement and fewer sign changes are therefore expected by construction. That metric cannot detect the audit §4 failure mode (smooth input → "potatoes" and lost mountains), and the input here is a bilinear ~39 km raster.
- **What the stored report already shows** (`coast-noise-report.json`, seed 0, one coast-median window per style, 7.68 km): going from noise 0.5 to 0.05, the NN's p99 slope drops 38% (gondwana), 28% (continents), 46% (earthlike) and 38% (archipelago). Maximum slope drops by 33–55%. Each style moves toward its parent's smooth p99.
- **At 30 m** (`native-noise-0.5` vs `native-noise-0.05`, 512² crops):
  - earthlike-s0 coast: slope p95 falls from 0.103 to 0.049.
  - earthlike-s42 coast: slope p95 falls from 0.074 to 0.022, and the median from 0.0122 to 0.0024.
  - gondwana-s0 coast: slope rises, but the land/sea mix of that crop changed a lot (43% of pixels changed sign vs 13%).
  - So two of three sites lose 55–80% of their slope. That is only three coastal sites with confounded crops, and no mountain or plain site was ever compared across noise levels.
- **Fix (a gate, not a code change, since the user authorized the default):**
  - Run the final NN sites (which already include mountain, plain and polar sites) at 0.05, at 0.5, and with natural, on the same footprints.
  - Report land-only slope quantiles, relief range and the coarse/latent power spectrum per stage.
  - Don't call 0.05 "validated" until that comparison and the visual boards are done.
  - Worth testing: an intermediate value such as 0.1–0.2 for height, or adding back structure in the 15–80 km band of the parent, rather than tightening the lock further.

## P2
- **a. Natural's identity still includes the terrestrial adapter.** `terrain_manifest.py:60` hashes `terrain_conditioning.py` for natural, and that file holds `CONDITIONING_SNR`, the v2 climate and `BootstrapConditioning`. The 0.05 change alone therefore invalidated natural's `PROFILE`/`CACHE` and its 6,160 prepared coarse windows (439.6 s). r1 gate 5 ("natural's cache still reused") can't hold under this layout. Fix: move `NaturalConditioning` into its own module, hash only that for natural, and check the 391,444-byte reference tile again.
- **b. Some bootstrap errors still reach the client as an HTML 500.**
  - A missing or broken `../world-builder-rs` raises `subprocess.CalledProcessError` (`terrain_bootstrap.py:69,72`). That is neither a `RuntimeError` nor an `OSError`, so `/api/world` (`terrain_server.py:618-621`) returns HTML, and `index.html:102` fails inside `r.json()`.
  - `/api/coarse/prepare` (`:260`, read at `index.html:111`), the overview's `metadata()` call (`:892`, outside its `try`) and the tile endpoints don't map `RuntimeError` at all.
  - Fix: wrap the git and cargo failures in `RuntimeError`, and add a 503 branch to those endpoints.
- **c. The snapshot can record a file that isn't the one running.** `terrain_server.py:148` hashes `terrain_bootstrap.py` on the first terrestrial request, while the code in use was imported at startup (`terrain_bootstrap.py:52`). An edit in between gives a manifest naming bytes that never ran, and it disagrees with `python_source_sha256`. Fix: use `terrain_bootstrap._IMPORTED_SOURCE_SHA256`.
- **d. The docs are stale.**
  - `TERRAIN_BOOTSTRAP_CONDITIONING.md:15` still says `CONDITIONING_SNR=(0.5,)*5` and that the adapter doesn't alter the pipeline.
  - `TERRESTRIAL_BOOTSTRAP_QA.md:89-91` says no new default was chosen.
  - `QA.md:45,103-105` points at `terrestrial-bootstrap-final-cpu`, which has v1 climate, no flat metrics and no polar sites. Its documented NN command now raises at `validate_terrestrial_nn.py:217-218`. The current CPU run is `terrestrial-bootstrap-v2-cpu`.
  - The QA table is sphere-only, yet the flat map differs. For earthlike s0 the largest landmass is 58.5% of land on the sphere vs 43.1% on the flat map.
  - The measured temperature extrapolation isn't stated: temperature inputs sit beyond 4σ over 19.2% of the flat map (max 8.4σ), against 0% for natural.
- **e. Flat-map metrics are computed but nothing gates on them.**
  - `verify_terrestrial_bootstrap.py:276` still gates on sphere metrics.
  - The NN global probes are weighted by sphere area only (`validate_terrestrial_nn.py:165,255`).
  - Fix: report the flat values alongside, and gate on whichever convention the product decision names.
- **f. The SNR sweep can't be tied to the current climate.** `tune_terrestrial_coasts.py:22-33` neither checks nor records `conditioning_version`. Fix: add the same check the validator has.
- **g. The heightmap cache grows without limit.** Each new seed writes a permanent NPZ under `CACHE_ROOT` (`terrain_bootstrap.py:33,249`). Since the client starts with a random seed, every fresh visit adds one, with no quota. Fix: measure the NPZ size and add an LRU limit or bring it under the disk budget.

## Verdict: CHANGES REQUIRED
P1-1 is a small, proven cache-identity defect that the identity decoupling introduced; fix it before any further bootstrap change lands. P1-2 doesn't block the default the user chose, but nothing should describe 0.05 as giving credible relief until the matched comparison exists.

**Gates still open:**
1. Fix P1-1 and add the identity-change test.
2. Finish the running final NN QA, then run matched comparisons at 0.05, at 0.5 and with natural on the same sites, with relief and spectrum metrics at coarse and 30 m.
3. Visual approval at continental, regional and 30 m scales (audit §13.1).
4. Split natural's identity (P2-a), then confirm the natural reference tile is byte-identical and its cache is reused.
5. Measure first-open latency (including candidate attempts and `generation_seconds`) and run a browser smoke test.
6. Correct the docs (P2-d).

Separately, the claude.ai Google Calendar, Google Drive and Sentry connectors need authorizing in your claude.ai connector settings before they can be used. None of them was needed for this review.
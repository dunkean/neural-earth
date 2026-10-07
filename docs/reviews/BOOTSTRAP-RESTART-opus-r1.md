# Review: BOOTSTRAP-RESTART (terrestrial World Builder bootstrap)

I only read files. I ran nothing and edited nothing. Two files changed while I was reading them: `terrain_bootstrap.py` gained `_generation_binary` and `terrain_server.py` gained `bootstrap_raster_spacing_m`. I re-read the current `terrain_bootstrap.py` and spot-checked `main.rs`, which was unchanged. Since the height agent is still fixing continent sampling, `main.rs` and `terrain_bootstrap.py` need another look once that lands. Nothing below certifies visual quality, NN behaviour or performance.

## What checks out
- **Old initializers and city-generator code are gone.** Profiles are limited to natural plus the four terrestrial styles (`terrain_conditioning.py:19`), `ABLATIONS = ("A0",)` (`terrain_reference.py:22`), and none of the reviewed files mention `city_generator`, A4, macro or earth.
- **Coordinates are consistent end to end.**
  - Cell centres sit at half-cell offsets.
  - Upstream calls the factory as `(cj0, ci0, cj1, ci1)` (`world_pipeline.py:998`), and the adapter maps that correctly to x/y.
  - The raster's north row matches the climate latitude convention.
  - The server removes the lapse adjustment exactly once (`terrain_server.py:146`).
- **Height units are handled correctly.** Interpolation happens in metres, the signed square root is applied exactly once (`terrain_conditioning.py:221`), and the adapter does not clip.
- **The climate rank mapping matches upstream.** Upstream `build_quantiles` uses `linspace(1e-4, 1-1e-4, 64)`, which matches `terrain_conditioning.py:187`, and the octave counts line up per channel.
- **The heightmap cache is sound.** It checks the namespace and each array's hash, writes atomically, and the new `_generation_binary` refuses to generate if the sources or the binary changed mid-process (`terrain_bootstrap.py:130-143`).
- **The cube-face seam reconstruction is tested.** See `main.rs:175-213`.

## P1 problems

**P1-1 (proven): the natural reference now depends on World Builder and on this experimental code.**
- `terrain_manifest.py:74` puts `bootstrap_native: implementation_identity()` into the manifest for every profile, including natural.
- `:57` hashes `terrain_bootstrap.py` and `terrain_conditioning.py` for natural too.
- At server startup, `terrain_server.py:67-70` builds a natural manifest from those hashes and uses it to name the cache folder for every profile.
- `implementation_identity()` calls git, refuses to continue if the World Builder checkout isn't at the pinned commit or is modified (`terrain_bootstrap.py:68-73`), and may run `cargo build` (`:103-104`).

How to reproduce:
- Move `../world-builder-rs` off the pinned commit, or edit a world-core file, and start the server: it fails on import even if you only use natural.
- Or let the height agent rebuild `main.rs`, or upgrade Rust: the cache folder name changes, so natural's tile cache and its prepared coarse windows (6,160 windows, 439.6 s, 707 MB) start cold again.

Fix:
- Compute the bootstrap identity only for terrestrial profiles and keep it inside the manifest's `bootstrap` block, which already records `height_sha256`.
- Leave the bootstrap files out of natural's identity.
- Key terrestrial NN caches on the output content (`height_sha256` + climate version) rather than the binary hash.
- Then confirm the fixed 391,444-byte natural reference tile is still reproduced exactly.

**P1-2 (proven geometry; misleading validation): the world is a projected sphere, but every check measures the sphere, not the flat map.**
- `main.rs:150-158` samples the planet in latitude/longitude, and `terrain_bootstrap.py:302-303` maps that linearly onto the 40,000 × 20,000 km plane.
- Shapes are therefore stretched east–west by 1/cos(latitude): at least 2× over 33% of the map, at least 4× over 16%, at least 10× over 6.4%, and about 326× in the top and bottom raster rows.
- Yet every check weights by sphere area or uses sphere distances:
  - style acceptance (`main.rs:37-62`)
  - the height-distribution remap (`terrain_bootstrap.py:188`)
  - CPU QA: land fraction, coastline length and slopes (`verify_terrestrial_bootstrap.py:23-28, 79-89`)
  - NN QA (`validate_terrestrial_nn.py:128, 208`)
- Test sites also skip |y| > 8,000 km (`verify:150`).
- Result: land fraction, largest landmass, coastline, the height distribution and NN land/sea disagreement don't describe what the user navigates. Disagreement near the poles is down-weighted by cos(latitude).

How to reproduce: for any exported world, compare land fraction and largest-landmass share with equal weights per cell against the current sphere weights, and count land above 60° latitude.

Fix:
- Record the "unrolled planet" decision explicitly.
- Gate on flat-map (equal-area) metrics, keeping sphere metrics as secondary.
- Add polar sites to the NN QA.
- Then decide how to handle the poles, for example keeping very high latitudes oceanic or treating them explicitly. Any such change gets a new version.

**P1-3 (missing evidence, strong prior): climate inputs at high latitudes may be far outside what the network saw in training.**
- Climate bands run all the way to 90° (`terrain_world.py:185-194`), and BIO1 is not clamped (`terrain_conditioning.py:190`). That is deliberate per the conditioning doc.
- Natural clamps and stretches temperature to a floor of −17.5 °C. Using the means/stds cited in `TERRAIN_BOOTSTRAP_LOCAL_STUDY.md:117-118` (still to be checked against the installed config), Antarctic-band temperatures would sit around −5σ or lower.
- The earlier A4 polar case already showed the network erasing land there: 7.04% NN land against 45.84% in the input.
- Because of P1-2, these latitudes cover about a third of the flat map.

Fix: in the CPU QA, report how far each input channel sits from the checkpoint's normalization, on an equal-area grid, plus the fraction beyond 4σ. Compare against natural. Decide the polar policy before this profile becomes the default.

## P2 problems
- **a. One height distribution for every style.** The target is all of ETOPO, including Antarctic and Greenland ice surfaces (`terrain_bootstrap.py:161-168`), so archipelago islands get the same high-altitude tail as continents. Measure land above 2,000 m and the highest point per island. Consider restricting the target to |lat| ≤ 60° or making it style-aware, as hypsometry v3.
- **b. Inland basins become sea.** Land basins below zero go through the ocean depth distribution (`terrain_bootstrap.py:191`). This is documented in `main.rs:164`, but the area affected is never measured, and `physical_ocean` is diagnostic only.
- **c. The input may be too smooth.** The raster is about 39 km per pixel and bilinear, so the inputs carry no structure between roughly 15 and 78 km. That fits the smooth "potato" coastlines described in the audit (§4). Compare coastline spectra of terrestrial vs natural inputs and of the network's coarse output. The bridge already allows resolution 512 and a 2048×1024 raster.
- **d. Thread-count determinism isn't tested at production size.** The test uses resolution 128 at 256×128 (`test_terrain_bootstrap_native.py:127`); production is 256 at 1024×512. Extend the test.
- **e. The style test is circular.** `test_styles_have_measured_topology` re-checks the same criteria `main.rs` used to accept the world. Only `verify_terrestrial_bootstrap.py`'s raster check is independent.
- **f. The NN validator's exit code says nothing about quality.** It fails only on non-finite values or heights above 20 km (`validate_terrestrial_nn.py:213, 234, 269`). Rename those failures to contract failures, or pre-register real quality thresholds.
- **g. Bootstrap failures surface as an opaque error.** A `RuntimeError` (all 12 candidate worlds rejected, cargo missing, pin mismatch) isn't caught by `/api/world` (`terrain_server.py:592-600`), so the client tries to parse an HTML 500 as JSON. Return a JSON 422/503 with the reason.
- **h. Opening a new terrestrial seed blocks the GPU.** Up to 12 world generations plus the ETOPO remap run inside the GPU lock (`_create_world` → `rebuild` → factory, `terrain_server.py:197-207, 712-720`) and in the metadata request. This is unmeasured. Generate the heightmap before taking the GPU lock, and measure it.
- **i. An unvalidated profile is the default.** The client opens `terrestrial-earthlike` by default and automatically starts an 8,192-window preparation (`index.html:29, 104, 122`). The NN QA isn't finished, the validator needs the GPU to itself, and the audit makes visual approval a blocking gate for changing the default (§13.1). Default to natural, or confirm the user authorized this, and pause automatic preparation during QA.
- **j. Climate noise seeds collide.** Seeds are masked to 31 bits and offset by `104729·k` (`terrain_conditioning.py:31, 158`). u64 seeds that agree modulo 2³¹ share all climate noise, and channels shift across nearby seeds. Derive per-channel seeds with a hash and bump the conditioning version.
- **k. The manifest records parameters terrestrial doesn't use.** `drop_water_pct` and `frequency_mult` appear for terrestrial profiles (`terrain_manifest.py:127-129`). Set them to null there.

## Verdict: CHANGES REQUIRED
There is no P0. P1-1 is a proven coupling of the protected reference to the experimental lane. P1-2 makes the QA report sphere numbers for a flat product.

**Gates still open:**
1. The height agent's continent-sampling fix, followed by another review of `main.rs` and `terrain_bootstrap.py`.
2. CPU QA over 8 seeds × 4 styles, with flat-map metrics and input-normalization statistics.
3. Completed records from `validate_terrestrial_nn`, including polar sites.
4. Visual approval of coast, mountain and plain boards against natural, at continental, regional and 30 m scales.
5. After the P1-1 fix, a check that natural is byte-identical and its cache is still reused.
6. Measured first-open and generation latency, plus a browser LOD smoke test.

Separately, the claude.ai Google Calendar, Google Drive and Sentry connectors need authorizing in your claude.ai connector settings before they can be used. None of them is needed for this review.
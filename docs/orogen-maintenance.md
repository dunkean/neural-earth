# Orogen modifications and upstream tracking

This is the maintenance record for Neural Earth's Orogen integration. It records local changes, the last reviewed upstream revision, and how to assess future updates. Technical source provenance and licenses remain in [PROVENANCE.md](../native/orogen/PROVENANCE.md) and [third-party notices](../THIRD_PARTY_NOTICES.md).

## Integration decision

**2026-10-10: keep the bundled integration and review upstream changes selectively.** A direct replacement with an upstream submodule is deferred: it would require migrating the generator, retained stages, climate settings and optional CUDA adapter together. Merely adding a submodule would track upstream history but would not make the running integration compatible with its updates.

The upstream worker imports Delaunator from an HTTPS CDN. Importing it unchanged with the local Node.js 24.10.0 runtime fails with `ERR_UNSUPPORTED_ESM_URL_SCHEME`. Replacing that import is straightforward; the substantial work lies in preserving our retained-state exports, physical units, settings and GPU transformations while adopting upstream's changed algorithms.

This decision leaves the application generator unchanged. The tracking process below uses an independent checkout and has no runtime dependency on it.

## Source and review baseline

| Item | Recorded value |
| --- | --- |
| Upstream | [raguilar011095/planet_heightmap_generation](https://github.com/raguilar011095/planet_heightmap_generation) |
| Tracked branch | `main` |
| Last review | 2026-10-10 |
| Last reviewed upstream commit | [`cc2662b4edd52231c4f65d8765f3ef12cd82d9b7`](https://github.com/raguilar011095/planet_heightmap_generation/commit/cc2662b4edd52231c4f65d8765f3ef12cd82d9b7) |
| Upstream commit date and subject | 2026-07-03; `Merge pull request #54 from raguilar011095/ra_climate_tuning` |
| Bundled source | `native/orogen/vendor/`, copied on 2026-10-08 from local `../world_builder/orogen/js` |
| Original source revision | Unknown: the copied local checkout had no HEAD commit |
| Integrated upstream revision | Not established; the reviewed commit is a tracking baseline, not the origin of our bundled code |
| License | Bundled GPL-3.0; Delaunator ISC and Acorn MIT licenses retained separately |

Comparisons normalize line endings. Against the original local snapshot, nine JavaScript modules are unchanged, nine are modified, and `climate-config.js` is a local addition. Against the reviewed upstream commit, **five of the 19 bundled JavaScript modules match and 14 differ**. These differences combine upstream evolution and our local adaptations; they are not all Neural Earth changes.

The five matching files are `climate-util.js`, `ocean-land.js`, `rng.js`, `simplex-noise.js` and `sphere-mesh.js`. File equality does not establish compatibility of the entire pipeline or identify the original snapshot's revision.

## Local modifications to preserve

Paths in this table are relative to `native/orogen/` unless another directory is named.

| Area | Files | Local behavior and maintenance constraint |
| --- | --- | --- |
| Server execution | `pipeline.mjs`, `vendor/planet-worker.js`, `vendor/delaunator.cjs` | Run the browser worker through Node, use bundled Delaunator, and export graph fields and timing metadata to Python. Preserve field names and array formats. |
| Generator controls | `vendor/planet-worker.js`, `vendor/elevation.js` | Configurable collision threshold, stress propagation and angular-speed multiplier; optional postprocessing bypass and boundary diagnostics. Preserve the settings contract. |
| Physical height imports | `vendor/planet-worker.js` | Import float32 metres using the inverse elevation curve, pixel-centre sampling and the application's `atan2(z,x)` longitude convention. Preserve sea level, units and wrapping. |
| Coastal height calibration | `vendor/color-map.js`, `vendor/planet-worker.js`, `gpu/runtime.mjs`, `backend/terrain_orogen.py` | `coastal-slope-v1` adds a bounded coastal term to the original land-altitude curve. Node, Python, the CUDA climate helper and the import inverse must agree; this also affects climate. |
| Climate and biome controls | `climate-parameters.json`, `vendor/climate-config.js`, `vendor/wind.js`, `vendor/ocean.js`, `vendor/precipitation.js`, `vendor/heuristic-precip.js`, `vendor/temperature.js`, `vendor/koppen.js`, `vendor/color-map.js` | Shared configurable parameters for the backend and UI, including climate thresholds and biome appearance. Temperature moisture is clamped to its expected range. Defaults and parameter meanings need explicit comparison before accepting upstream tuning. |
| Independent stages | `vendor/planet-worker.js`, `pipeline.mjs`, `backend/terrain_orogen_stages.py` | Export/restore retained state, apply erosion without rebuilding relief, clear climate caches, and rebase historical height conventions. Snapshot compatibility needs a migration decision if upstream state changes. |
| External City erosion | `vendor/planet-worker.js`, `pipeline.mjs`, `backend/terrain_city_erosion.py` | Replace graph elevation and resume climate on the retained mesh. Preserve consistency between the eroded atlas and climate input. |
| Optional Orogen CUDA | `gpu/build.mjs`, `gpu/runtime.mjs`, `gpu/erosion.cu`, `gpu/nearest.cu`, `backend/terrain_orogen_gpu.py` | Transform temporary module copies using AST analysis and specific source patterns. Upstream function signatures and loop changes require checking both transformed loops and CPU fallback coverage. Experimental propagation and erosion can change terrain. |
| Neural conditioning and persistence | `backend/terrain_orogen.py`, `backend/terrain_conditioning.py`, `backend/terrain_orogen_layers.py` | Rasterize graph triangles into the atlas, convert seasonal climate to BIO1/BIO4/BIO12/BIO15, and include implementation hashes in world identity. New source behavior must produce a distinct cache identity. |

The historical six-array bit-equality comparison in the provenance applies to one seed, mesh size and historical height convention. It is not a guarantee of equivalence with today's upstream or optional GPU engines.

## Upstream differences requiring review

The reviewed upstream has features beyond the copied local snapshot:

- **Plate physics and mantle convection:** `plate-physics.js` and related elevation inputs introduce additional retained fields and alter terrain formation.
- **Superplates:** `buildSuperPlates` now takes a coarse mesh and its plate map, with optional high-resolution projection. Our worker uses the older mesh interface.
- **Terrain shaping:** `terrain-config.js`, detail noise, dampening/orogenic fields and `terrain-metrics.js` add controls, dependencies and retained data.
- **Climate:** upstream exports `CLIMATE`, `CLIMATE_DEFAULTS` and tuning helpers, while our modules read the local `climate` catalog. The calibrated defaults and temperature implementation differ. The upstream `tuning/climate/` suite provides useful evaluation infrastructure, but its parameter names are not interchangeable with ours.
- **GPU compatibility:** our adapter recognizes existing functions and source patterns. A successful CPU import alone cannot validate an upstream update for GPU execution.

A future direct integration is worth reconsidering when the headless execution, retained-state contract and configuration can be maintained through a small, explicit adapter. Until then, port useful fixes in separate changes and record their upstream commits.

## Check for updates

Run these commands from the Neural Earth repository root. Clone once into the ignored `tmp/` directory; this checkout is for review only.

```shell
git clone --branch main --single-branch https://github.com/raguilar011095/planet_heightmap_generation.git tmp/orogen-upstream-watch
```

For subsequent reviews:

```shell
git -C tmp/orogen-upstream-watch fetch --prune origin
git -C tmp/orogen-upstream-watch rev-parse origin/main
git -C tmp/orogen-upstream-watch log --oneline cc2662b4edd52231c4f65d8765f3ef12cd82d9b7..origin/main -- js tuning/climate
git -C tmp/orogen-upstream-watch diff --stat cc2662b4edd52231c4f65d8765f3ef12cd82d9b7 origin/main -- js tuning/climate
```

Inspect changes in each affected area, for example:

```shell
git -C tmp/orogen-upstream-watch diff cc2662b4edd52231c4f65d8765f3ef12cd82d9b7 origin/main -- js/planet-worker.js js/elevation.js js/super-plates.js js/climate-config.js js/temperature.js
```

Use the full commit from the last completed review as the starting revision when this record advances. `fetch` updates remote references; the commands compare `origin/main`, so no checkout or pull is needed. Check upstream manually before planned Orogen work or when a relevant fix is announced. No scheduled automation is installed.

For each relevant change, record its commit, affected files, expected benefit and decision: **port**, **defer**, or **irrelevant to the integration**. UI-only changes often need no port; simulation changes require evaluating output and performance rather than copying files wholesale. Reading the latest upstream tree uses `git show origin/main:js/<file>`; diffing it against our vendor tree alone cannot distinguish our adaptations from upstream history.

## Validate an accepted port

Use a separate development change. Document the imported commit and preserve the applicable licenses. Check the affected settings, physical units, graph fields and retained-state compatibility. Changes to height or climate must be recorded as behavioral changes, including their effect on existing snapshots and new cache identities.

With the normal Python environment activated, the existing checks are:

```shell
python -m unittest discover -s tests/python -p "test_terrain_orogen*.py"
python -m unittest discover -s tests/python -p "test_terrain_koppen.py"
```

Run biome checks when climate or palette behavior changes, and the relevant generation-settings/UI checks when their contract changes. Review skipped tests: atlas/stage tests need CUDA, and optional GPU tests need their additional runtime. CPU results do not validate optional GPU compatibility. Compare full generation with the retained-stage chain, verify imports and external erosion when touched, and check representative seeds and mesh sizes for physical fields and timing.

## Review and port log

Add a row after each completed review; record partial ports individually. Update the review baseline and command examples only after all relevant changes through that commit have an explicit decision. A reviewed commit does not become an integrated revision merely because this table advances.

| Review date | Upstream commit | Outcome | Integrated changes / follow-up |
| --- | --- | --- | --- |
| 2026-10-10 | `cc2662b4edd52231c4f65d8765f3ef12cd82d9b7` | Direct integration deferred; selective review retained | No code imported. Worker loading, module differences, climate configuration, retained stages and GPU coupling reviewed. Plate physics, superplates, terrain shaping and climate calibration require separate migration work. |

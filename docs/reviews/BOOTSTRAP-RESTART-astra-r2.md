# Bootstrap restart review — Astra r2

Date: 2026-10-07. Second source review after the Opus r1 correction batch. No GPU execution and no production source edits; only this review document was written. Source continued changing during review. The completed v2 CPU receipt and the earlier complete paired NN studies were read; the final v2 NN run was still in progress.

## Result

Two concrete defects were found and reported during this pass: **P1 stale server-side terrain caches after a native-only revision**, and **P2 unhandled subprocess initialization errors**. Both now have source fixes that were inspected. No additional confirmed open P1/P2 implementation defect was found in the inspected final snapshot.

This result does **not** certify final terrestrial NN quality or navigation performance. Completed final v2 polar/coast/plain/mountain results, visual inspection and runtime verification remain necessary evidence. Defaulting to the requested terrestrial world is authorized by the user's task; this review does not introduce a new approval requirement.

## Findings and inspected corrections

### P1: browser identity changes did not invalidate server disk content

After Natural was decoupled from native source/binary hashes, a native-only revision changed a terrestrial `world_identity` but left the global Natural-derived `PROFILE` unchanged. Physical tiles, rendered PNGs and overview PNGs still occupied `CACHE/profile/seed/...`. Adding `world_identity` to the browser URL prevented browser reuse, but `valid_physical_report()` accepted an old decoder report and the overview route returned an existing PNG unconditionally. A fresh URL could therefore return the old world's terrain.

The inspected correction compares every physical/render report's `world_identity` with the current manifest before treating it as a hit. Overview caching now uses a **per-display-mode** JSON identity receipt, checks the requested identity, and regenerates on mismatch. This also avoids making one newly regenerated mode validate another mode's old PNG. Existing native-mip reads already checked the world identity. Source tests `test_native_only_identity_change_invalidates_height_payload` and `test_overview_identity_receipts_are_per_display_mode` exercise these cases. They were inspected, not independently rerun here.

### P2: Git failures bypassed the JSON bootstrap error response

The `/api/world` correction initially caught `RuntimeError` and `OSError`, but missing/inaccessible World Builder source makes `subprocess.run(check=True)` raise `subprocess.CalledProcessError`, which is neither. Such a failure still produced an HTML 500 response and obscured the cause in the client's JSON parsing. Other generation routes also lacked the runtime handler.

The inspected server now catches `subprocess.SubprocessError` in `/api/world` and registers JSON 503 handlers for `RuntimeError`, `OSError` and `subprocess.SubprocessError` across routes. This addresses the concrete uncaught path. The message may remain terse for a Git command failure, but it is now structured and no longer an HTML parse error.

## Other correction checks

- **Natural/native prerequisite separation:** `_files(..., 'natural')` omits `terrain_bootstrap.py` and does not call native `implementation_identity()`. Natural manifests have `bootstrap_native=None`. The terrestrial source/binary snapshot is lazy. Missing Cargo or an unpinned World Builder checkout no longer blocks Natural through manifest construction. Shared Python runtime/conditioning files still participate in Natural identity, appropriately reflecting that they contain code used by Natural; this does not promise old cache folders survive arbitrary shared-source changes.
- **Conditioning policy:** terrestrial `cond_snr` is now `(0.05, 0.5, 0.5, 0.5, 0.5)`. `configure_world` saves the original per-world Natural values and restores them when selecting Natural; the Natural factory remains the upstream factory with the established seed-zero repair. Neural manifests take the executed SNR from checkpoint kwargs. Terrestrial unused frequency/drop-water controls are now null in the manifest.
- **u64 climate seeds:** v2 mixes the full u64 seed and channel domain before FastNoiseLite's 31-bit reduction, removing systematic equality for seeds separated by `2**31`. The documentation accurately states that finite 31-bit channel seeds can still collide. Natural seed derivation is unchanged.
- **CPU initialization and GPU locks:** normal metadata, background preparation and physical-tile admission resolve the parent manifest before entering the GPU critical section. This removes the ordinary first-generation path identified by Opus from that lock. No latency claim follows from source inspection.
- **QA:** CPU exports now include both flat and spherical metrics, real checkpoint z scores, Natural comparisons and below-datum/non-ocean diagnostics. Polar sites are included in selection. The thread-count test now uses production native/raster dimensions. NN exit status is explicitly a numerical contract result rather than visual acceptance.
- The first review's provenance revalidation and 39,062.5 m source-resolution corrections remain present. No change to validated runtime LOD geometry or decoder arithmetic was identified in this correction batch.

## Measured evidence actually inspected

### v2 CPU batch

Read `E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-cpu/report.json`: **32 worlds**, climate version `native-bootstrap-worldclim-v2`, **zero recorded contract failures**. The independent raster gate currently uses spherical area; flat metrics are additionally reported.

| Style | Flat-map land fraction | Largest component / flat-map land |
| --- | --- | --- |
| Gondwana | 26.29–41.15% | 91.12–99.25% |
| Continents | 26.75–44.06% | 34.74–55.15% |
| Earthlike | 21.58–37.64% | 27.53–68.64% |
| Archipelago | 15.26–19.98% | 6.28–16.46% |

Applying the same broad thresholds to the flat metrics yields **one exception**: archipelago seed `4294967297` has its largest component at **16.4648%** of flat land, above the spherical gate's 15% cap. It passes the recorded spherical gate. This is a measured projection/style distinction, not proof that the world ceases to be an archipelago. Do not claim all 32 satisfy identical flat and spherical thresholds; document the distinction or define a separate explicit flat tolerance.

The normalization concern is now quantified. Across this 256 × 128 diagnostic sampling, BIO1 reaches **12.55 standard deviations** from the checkpoint mean, and the maximum fraction above `|z|=4` is **21.98% of flat-map samples**. BIO4 reaches 6.26 sigma and up to 4.78% of flat samples beyond four sigma. Height remains below 3.13 sigma in the inspected diagnostic reports. These numbers justify inspecting the pending actual polar NN outputs; they alone do not prove a particular visual failure or justify clipping the physical climate.

Below-datum/non-ocean flat area ranges are approximately 0.10–1.96% for Gondwana, 0.35–1.04% for continents, 0.47–1.43% for Earthlike and 0.69–0.91% for archipelago. The receipt correctly warns that the nearest-native diagnostic mask includes both inland basins and reconstruction/coast cells; these are not exact inland-basin areas.

### Completed noise studies

Read the complete 16-patch wide-coarse noise study and both completed three-site, 512 × 512 native studies under `terrestrial-bootstrap-final-nn`. The 512-pixel crops span **15.36 km** per side at 30 m. The native source-to-NN sign disagreement improves when changing altitude noise from 0.5 to 0.05:

| Native site | Noise 0.5 | Noise 0.05 |
| --- | --- | --- |
| Earthlike seed 0 coast | 42.25% | 18.77% |
| Gondwana seed 0 coast | 42.85% | 12.61% |
| Earthlike seed 42 coast | 49.68% | 26.20% |

This supports the stronger altitude conditioning choice on these measured sites. It does **not** show improvement in every metric: Earthlike seed 0's median parent-to-NN edge-linear zero-contour distance increases from **1.41 km to 1.82 km**, although its tail/reverse distances and sign agreement improve. Coast offsets remain material. Earthlike seed 42 regains a visible coastline in the crop at 0.05; at 0.5 that crop was entirely learned land. One 0.05 Earthlike seed 42 comparison board was opened and inspected; its visible shoreline displacement agrees with that limited metric interpretation. These earlier studies precede climate-v2 final validation and must be identified as supporting experiments rather than the completed final v2 result.

### Evidence still incomplete at review time

`terrestrial-bootstrap-v2-nn/final/neural-report.json` still had `complete=false` during this pass. No final conclusion is made about its planned polar/plain/mountain/coast crops or its sparse world sample. The validator's sparse world records still compute spherical-weighted summaries/disagreement; flat-map equivalents can be derived from the saved NPZ samples without rerunning GPU work. Sparse 8 × 4 points cannot establish NN global component topology or shoreline length.

Before the final report, inspect the completed boards and receipts, state observed remaining coast/polar differences, and retain actual Natural equivalence and browser timing evidence. Keep the source hashes attached to each run: completed experiments from before the final correction batch do not by themselves validate a later runtime identity.

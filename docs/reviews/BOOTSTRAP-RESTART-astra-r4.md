# Bootstrap restart review — Astra r4

Date: 2026-10-07. Focus: the real-browser coarse-admission failure, its correction, preserved rejection checks, and the newly available runtime receipts. No GPU work was performed by this reviewer. Three targeted CPU tests were run; no production files were changed.

## Proven failure and correction

The pre-fix real Chrome session failed to reach native detail. `browser-before-coarse-fix.json` records `passed=false` and `native view did not complete`; the parent traced its LOD4 HTTP 400 to `Coarse manifest ablation does not match pipeline`. The NN export tests had exercised generation without this server persistence-admission boundary. Therefore the previous source/NN reviews did **not** establish that terrestrial server navigation worked.

`terrain_coarse.py:install` now maps only Natural to canonical ablation `A0`; each terrestrial world retains its complete `terrestrial-*` profile name. It checks membership in the supported profile list and requires `manifest.world_profile` to equal the configured pipeline profile. Seed, SNR, precision, execution-profile, window geometry and persistent-identity collision checks remain in place. The change is an admission/identity correction; the network math, noise, window contribution ordering and LOD sampling arithmetic are unchanged.

Independently ran these CPU-only checks:

- Both tests in `test_terrain_server_world`: **passed**. The exact server `world_manifest`, `_create_world` and `get_world` functions run with the real WorldPipeline, inference configuration, manifest builder and CoarsePreparation. Checkpoint loading and native parent creation are mocked, network forwards are forbidden, and CUDA access is guarded. All four terrestrial profiles and Natural install correctly; a mismatched explicit world profile and all retired Earth/Macro labels are rejected.
- `test_terrain_windows.CoarsePreparationTests.test_identity_collision_rejected`: **passed**. Wrong seed, wrong conditioning SNR and conflicting persisted geometry are still rejected.

No active city-generator or rejected custom bootstrap path was introduced by this fix. Unsupported historical labels are rejected rather than translated to new terrestrial worlds.

## Actual browser correction evidence

Read the subsequently completed `E:/TerrainDiffusionRuntime/terrestrial-bootstrap-runtime/browser-cold.json`: **passed=true, no recorded errors**, every captured response status 200. It is a real Chrome/server Earthlike seed 42 run with automatic full-world preparation and predictive prefetch disabled for measurement.

Independently verified:

- Every harness `sourceHashes` entry matches current local bytes.
- The canonical Python world-manifest hash reconstructed from `worldResponses.rawUTF8` is valid, and every implementation-file digest in that manifest matches current local bytes, including the corrected coarse module.
- Both screenshot files match their recorded SHA256.
- Final visible coverage is **10/10 tiles ready, zero pending**, rendered exclusively at **LOD0, decoder source 30 m**.

The receipt contains 37 HTTP responses and 22 physical tile cache misses. World overview readiness was **3.942 s**. After the scripted native-view change, submitted refinement draws reached LOD4/3/2/1/0 at **4.619 / 7.045 / 10.896 / 12.516 / 12.682 s**, with final camera coverage complete at **12.925 s**. These are the harness's separate timing origins; they must not be mixed into a single first-open duration without following the harness timestamps. Its stage milestones measure submitted draws, not a GPU presentation fence.

Opened and inspected `browser-cold.png`. The visible map shows a continuous shoreline and terrain detail; the UI reports 10/10 ready and 30 m source/LOD0, consistent with the receipt. This successfully exercises the previously broken server integration path. It is evidence for this particular cold-view session, not a frame-rate or all-navigation guarantee. Warm revisit, prolonged panning and other styles are not certified by this one receipt.

## Natural preservation and remaining evidence boundaries

Read `terrestrial-bootstrap-runtime/natural-byte-reference.json`: it reports the real decoder elevation (304 × 304 float32) and climate (5 × 33 × 33 float32) as **byte exact**, zero maximum error, against the saved original reference; combined payload **391,444 bytes**. This GPU check predates the admission-only coarse correction. The targeted CPU regression confirms Natural still installs after that correction; no network arithmetic was changed. The reviewer did not rerun the GPU comparison.

The earlier matched-relief sites are now correctly described as **median high-elevation sites**, which can be plateaus or gentle slopes. The 0.05-vs-0.5 measurements remain valid for their actual footprints, including the plain's slope-p95 ratio **0.407** (about 59% reduction), but they do not establish behavior on all mountain types. The additional per-style high-local-relief and p99 selections are documented; their new GPU comparison receipts were not yet present when this report was written. Do not describe that planned experiment as completed evidence.

The previous sampled height/coast results remain evidence of generation behavior. The new coarse identity check changes the full implementation identity, so receipts from before it must retain their original provenance rather than being relabeled as current. No new inference rerun is required merely to claim that an admission-only correction leaves the inspected arithmetic unchanged; final runtime claims should use the current browser manifest above.

## Conclusion

The newly discovered **P1 terrestrial server-admission defect is closed** by source correction, targeted real-pipeline CPU regression, and a successful current-identity real-browser LOD0 session. No additional confirmed P1/P2 code defect was found in this pass. High-relief generalization and broader navigation performance remain bounded by the evidence available; learned physical climate remains explicitly unaccepted as documented in r3.

# Bootstrap restart review — Astra r5

Date: 2026-10-07. Read-only review of the final admission guards and completed stress evidence. No GPU execution or production edits by this reviewer.

## Decision

**No new proven P1/P2 implementation blocker.** Retain terrestrial altitude noise **0.05** for the requested heightmap/navigation default as a documented tradeoff favoring coast control. It is not a universal relief improvement: substantial fine-detail loss in one tested archipelago footprint and conspicuous rectilinear learned structures remain visible limitations. Learned physical climate is not accepted by this decision.

## Final guards: inspected and tested

Reviewed the frozen files:

- `terrain_coarse.py`: SHA256 `5e8b3ded29df32ce458f9d119b64b7d4148f665f82e91440310309b43d18ccb0`.
- `terrain_server.py`: SHA256 `f7d93ba3d1c8c4129a83c3519156a96df58157447d18ab4ff3e54dfd0f66b181`.

The upstream `_dtype=None` convention is interpreted as fp32. A terrestrial manifest requires an explicit supported pipeline profile and actual conditioning-factory height metadata whose `height_sha256` equals the manifest bootstrap hash. Existing seed, SNR, precision, profile and ablation mismatch checks remain. Server coarse-construction/admission `ValueError` becomes an internal `RuntimeError`, closes the failed world and reaches JSON HTTP503; malformed user tile coordinates remain HTTP400. These edits change admission/error handling, not NN arithmetic or its configuration.

Independently ran all **five `test_terrain_server_world` tests: passed**. They exercise real server construction functions and real WorldPipeline/CoarsePreparation on CPU, with network execution prohibited. Coverage includes Natural plus all four terrestrial profiles; mismatched/retired labels; missing profile/factory; different parent hash; incorrect precision; valid None/fp32; and actual height-endpoint 503 versus 400 responses. Earlier seed/SNR rejection checks passed in r4. No city-generator or retired initializer was reintroduced.

## Completed stress evidence and scope

Read `post-admission-relief-stress/relief-comparison.json`: **eight footprints × three noise settings = 24 completed variants**, all 100% land. Its CPU attestation reports 24 checked stages, and its recorded report SHA256 matches the report read here. The selection includes one high-local-relief site and one p99-height site per style, addressing the earlier altitude-only selection limitation.

For the archipelago high-local-relief footprint, noise 0.05 versus 0.5 retains **75.7% of p95−p05 relief, 71.6% of slope p95, and only 41.5% of 1 km high-pass standard deviation**. The last figure is a **58.5% loss of that fine-detail metric**. The full table now exposes this rather than implying all detail losses stay below 50%. Other sites differ, including increases in some measures; the result supports a policy choice, not blanket superiority.

Opened the Gondwana rectilinear stage-diagnosis image and read its JSON. Strong rectilinear forms appear in the learned output and are absent from the smooth parent. The saved-stage CPU analysis does not establish a cube-face seam or fusion bug: decoder-stride and eight-pixel expansion boundary gradients are ordinary; partial coincidence with some latent boundaries is insufficient causation. These images are an actual quality limitation. Their cause remains unresolved; this review neither assigns them to precision nor claims a proven implementation defect. No additional inference experiment is required for the limited default decision requested here.

## Runtime evidence and identity limits

Read the completed real-Chrome warm receipt: **passed, no errors, 22 physical cache hits, 552 ms to completed native coverage**. Its cold counterpart, reviewed in r4, had 22 misses and took 12.925 s. These are measured sessions with the harness's stated conditions, not universal latency guarantees. The actual Natural-byte receipt reports 391,444 bytes exactly equal to the original reference; a real-server persistence replay refresh is separately planned.

These completed GPU/browser receipts precede the last admission guards and retain that provenance. At this review snapshot `final-hardened/neural-report.json` was still `complete=false` with 14 sites recorded; it is **not certified as a completed fresh-identity run here**. The sole GPU validation agent is refreshing the final receipts. Their completion and the new Natural replay can supplement this review without changing its source findings or overstating older evidence.

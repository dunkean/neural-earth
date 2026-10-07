# B2 Astra review — revision 2

Date: 2026-10-07. Reviewer: Astra, high. Decision: **CHANGES REQUIRED**.

Reviewed the revised scheduler, persistence, inference profile, CPU tests, three GPU verifiers and their current reports, both first-round reviews, `WINDOW_IMPLEMENTATION.md`, and the relevant server integration/manifest code. No production code was edited and no GPU workload was run. The CPU suite `test_terrain_windows` plus `test_terrain_climate` passed **15/15**. Two additional small CPU reproductions exposed the shared-cache issue below.

## Remaining proven defect

### P2 — Separate foreground/background preparations disagree about their shared namespace

Locations: `terrain_coarse.py:182–184`, `:289–304`, `:321–337`, `:356–364`, `:504–514`; real integration at `terrain_server.py:179` and `:213`.

Each `CoarsePreparation` snapshots `_persisted_indices` and `_disk_bytes` on installation, then updates them only for its own writes. A successful `_load_window()` of a file created by another preparation does not adopt its completion or bytes. The server now deliberately creates separate foreground and background worlds, each with its own preparation for the same world-hash directory. The GPU lock serializes computation but does not synchronize these Python bookkeeping objects.

**Reproduction 1 (completion):** install two preparations A/B against the same temporary root and manifest before either computes. Use the existing `_world(100000)` fixture and bounds `(0,0,2*7680,2*7680)`. Run A's preparation to completion, then call `B.step(..., budget_windows=1)`. Observed:

```text
actual_files=144, planned=144,
B.reported_complete=0, B.disk_hits=144, B.network_windows=0
```

B scans and replays every saved window but continues to report no completed coverage. Since the quantum budget counts network generation, a nominal one-window step can do a whole-plan replay under the GPU lock, and subsequent steps can repeat it. `complete()` can see valid files individually, but does not repair the status/completion set either. This invalidates background progress and work accounting when foreground requests supply windows first.

**Reproduction 2 (quota):** install A/B with `budget_bytes=1152` (two fixture windows). Materialize one distinct window through A, then two more through B. Observed:

```text
actual_payload_bytes=1728, shared_budget_bytes=1152,
A.reported_bytes=576, B.reported_bytes=1152,
B.disk_budget_exhausted=False
```

The quota is described as per-world, but is enforced against incomplete per-instance counts. This is sequential, not a race requiring concurrent GPU calls.

**Fix:** share a namespace-level persistence index/byte ledger between preparations under the existing serialization mechanism, or reliably refresh and adopt peer writes before readiness/progress reporting and quota admission. Successful external replay must add planned membership; disk accounting must include peer files. Avoid solving this by scanning every payload in every quantum. Add two-instance regressions for foreground-first coverage, completion short-circuit, shared quota, and peer invalidation. Preserve demand's RAM fallback when persistence is full.

## First-round findings and corrections

- **Astra B2-R1-01 and B2-R1-02 resolved:** v3 handles EOF/truncated NumPy payloads, binds sidecars to world/index/shape, checks exact canonical weight bytes for readiness, and verifies full digest/finiteness on demand load. Tests cover empty/truncated files and zero/altered positive weight channels. Readiness still means header/identity/weight verification, not full-payload digest verification; documentation now states that distinction.
- **Sparse-climate recursion correction retained:** the climate tests pass.
- **Quota no longer breaks a single demand reader:** persistence returns false, marks exhaustion, and leaves computed RAM outputs usable. Background stops. The two-instance quota problem above remains.
- **RAM-only repair improved:** background can recompute the exact dependency-free scalar coarse window when it exists only in RAM. The CPU regression passes.
- **Free-VRAM numerical profile correction:** latent batch defaults to 16 regardless of free memory. The server removes the descriptive free-memory `name` from its numerical identity. GPU name/UUID and cuDNN version are now present through profile/runtime identity. Identity is still conservative and includes more than coarse arithmetic; unnecessary cache invalidation remains a reuse limitation, not evidence of wrong terrain.
- **Manifest/pipeline checks improved:** seed, SNR, precision, world profile/ablation and declared batch profile are checked on install. Actual legacy-earth provenance is rewritten before hashing in the server. These tests are not a general proof that every mutable pipeline configuration agrees with a manifest.
- **World-edge integration corrected in code:** high-LOD height/climate samples are clamped to the finite world, with a nine-cell coarse context; out-of-world tiles are rejected. CPU preparation tests include clipped LOD 7/12 requests. Actual rendered edge continuity and end-to-end B3 readiness remain B3 validation work.
- **Instrumentation corrected:** unique/regenerated history uses bounded 1 MiB Bloom storage per stage and labels estimates; timing is explicitly inclusive; coarse model wall time synchronizes before recording. The scheduler adds an output admission limit. This does not prove peak dependency VRAM bounds or exclusive GPU stage timings.
- **Rebuild lifecycle improved:** `_build_hierarchy()` removes stale world-level preparation; reinstalling the current manifest remains explicit and documented.
- **Background/foreground separation fixes the old active-seed/LRU issue**, while introducing the shared bookkeeping exposure described above. Coarse area sampling and learned overview integration are B3 work and should be reviewed there.

## Current real-model evidence

The current reports' source hashes match the reviewed `terrain_coarse.py`, `terrain_window_scheduler.py`, and `terrain_inference.py` bytes. These are the new v3 reports, not merely the old v2 evidence.

- `coarse-persistence-fidelity.json`: **passed**. Exact weighted-channel replay, finite positive denominator, zero replay network windows. Coverage is five saved windows out of 6,160, not a completed world.
- `window-scheduler-fidelity.json`: **passed**. Scheduler off/on is exactly equal for the whole crop and separately for the split request sequence. The baseline whole/split height discrepancy remains `0.210693359375 m`; maximum climate discrepancy is about `0.0001221`. The assertion gate now distinguishes zero added scheduler drift from a declared tolerance on the existing request-shape discrepancy. Timing order/warmup still prevents a causal speedup claim.
- `canonical-latent-fidelity.json`: **failed**. The optional batch-one profile has whole/split errors of `0.09616–0.32422 m`; baseline/profile errors are `2.04178–11.90234 m`, and the seed-42 climate difference reaches `0.119989`. Post-eviction whole-crop replay is exact in these cases. The profile remains opt-in and must not be called an accepted fidelity fix or promoted to production.

The canonical verifier compares each permutation to the whole crop, not the permutation outputs directly to one another. Equal reported maxima do **not** establish that all three outputs are identical. Documentation should say that the three orders had the same reported maximum discrepancy, unless pairwise array equality is separately checked. Whole/split DEM errors also do not by themselves localize the discrepancy to latent fusion; reconstruction and crop-shape arithmetic remain possible contributors. No particular cause is established by these reports.

## Documentation and outstanding gates

`WINDOW_IMPLEMENTATION.md` still says the scheduler/persistence reports precede v3; update that paragraph to describe the current hashed runs. Keep the failed optional-profile experiment and the remaining default request-shape limitation explicit. The original exact whole/subcrop product criterion is still not achieved; this is a declared limitation rather than a new scheduler-induced regression.

After the shared-cache bookkeeping correction, the conservative B2 persistence/observability implementation can be accepted with limitations. Remaining evidence needs are full-world preparation and lock-hold measurements, real-model corruption/cancellation/resume coverage, bounded peak memory for supported production requests, multi-case default-profile revisit/edge checks, and B3 navigation/visual validation. No current result establishes ultra-fast cold 30 m navigation, coast topology preservation, or complete E3/E4 acceptance.

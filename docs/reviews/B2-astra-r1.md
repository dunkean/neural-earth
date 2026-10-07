# B2 review — Astra r1

**Decision: CHANGES REQUIRED.** No P0 found. The P1 sparse-climate recursion found during this review was corrected by the coordinator and its regression test passes. Two reproducible P2 persistence-integrity defects remain at this review snapshot. No runtime files were edited by this reviewer.

Reviewed 7 October 2026: `ASTRA_TERRAIN_AUDIT.md`, `AUDIT_IMPLEMENTATION.md`, `WINDOW_IMPLEMENTATION.md`, the B2 scheduler/coarse/inference implementation, its unit and GPU verification scripts, and relevant server/climate hooks. Inspected installed `infinite_tensor` dependency traversal and memory-store fusion/eviction. Read both real reports under `E:/TerrainDiffusionRuntime/`. No GPU inference or server restart was performed. Coordinator changes during review are explicitly identified below.

## Findings requiring correction

### B2-R1-01 — P2: empty persisted files bypass damaged-window recovery

**Location:** `terrain_coarse.py:186–198`, especially the exception list at line 195; readiness at lines 267–270.

**Proven:** a zero-byte `.npy` raises `EOFError: No data left in file` in this environment. `_load_window` catches only `OSError` and `ValueError`. Both demand reads and preparation therefore fail instead of regenerating. File-membership readiness initially reports this corrupt contributor as complete. Atomic writes protect normal interrupted writes, but the code explicitly supports recovery from damaged persisted products, and its test currently exercises only `b'broken'` (a different NumPy exception).

**Reproduction:** using `_world()` and `_manifest(2)` from `test_terrain_windows`, install a temporary `CoarsePreparation` with bounds `(0,0,6*7680,6*7680)`; read coarse `(0,0,2,2)`; select one required index; replace its file with `b''`; clear the coarse RAM cache; repeat the read. Observed: `complete(0,0,2,2) == True`, followed by uncaught `EOFError`.

**Fix:** handle NumPy's empty-file exception as a corrupt cache miss, invalidate its readiness membership, and regenerate/atomically replace it. Add zero-byte and truncated-header recovery cases, including reopened preparation.

### B2-R1-02 — P2: invalid fusion weights are accepted as learned coverage

**Location:** `terrain_coarse.py:188–194`, with publication through `complete()` at 267–270 and the persisted wrapper at 149–171.

**Proven:** shape-correct FP32 arrays of all zeros pass validation: finite and `weight >= 0` are insufficient. Replace every contributor of coarse `(0,0,2,2)` in the same CPU fixture with `np.zeros((7,4,4), np.float32)`, clear RAM, and read again. Observed: readiness true, zero network recomputations, and fused denominator `[[0,0],[0,0]]`. This violates the complete-fusion invariant. The ordinary stage read does not perform the positive-denominator guard that `read_mip()` adds; downstream division may therefore produce NaNs or fabricated zero coarse heights depending on the caller.

**Fix:** validate the weight channel against the canonical stage weight window (including its legitimate zero taper boundaries), or use an integrity digest bound to the canonical window identity plus appropriate weight validation. A blanket strictly-positive check would incorrectly reject valid tapered edges. Add all-zero and altered-positive-weight corruption regressions. Document whether membership-only readiness means merely 'file present' or verified learned coverage; cache the verified state so checks do not repeatedly read the entire world.

## P1 corrected during review

### B2-R1-03 — P1: sparse climate recursion did not guarantee progress

**Original location:** `terrain_climate.py:21–33`; server caller `terrain_server.py:294–300`.

The old function recursively grouped points into the same 48-coarse-cell bucket whenever the local extent plus an 18-cell allowance exceeded 64. A bucket can itself require 66 cells, so recursion received the identical group forever. Real LOD 7/8 tiles at `tx=1`, `4`, `-2` and other ordinary positions produce such groups. A CPU reproduction before correction was:

```python
sample_coarse_climate(None,
    np.arange(-638976., -626856. + 1, 1212.), np.array([0.]))
# RecursionError: maximum recursion depth exceeded
```

The coordinator added an explicitly local child path so a grouped request executes directly, and a regression covering real positive/negative LOD 7 sample coordinates. This reviewer reran `test_terrain_climate` and `test_terrain_windows`: **all eight tests passed**. This finding is resolved for the reviewed correction; expanding the regression to LOD 8 is still useful.

## What the evidence does support

- `WorldWindowScheduler.observed()` calls the original full `_ensure_processed` traversal. Installed InfiniteTensor recursively materializes all upstream contributors, and only then invokes a consuming stage. Its store iterates canonical intersecting windows when summing weighted numerators and denominators. The wrapper does not publish partially fused pixels.
- Integer stage indices, row-first rectangles, negative coordinates, and independent per-world RAM stores are consistent in the inspected paths. Persistent coarse filenames represent canonical row/column indices under a manifest namespace. The batch-one restriction prevents persistence from repartitioning the BF16 coarse solver batch.
- Real scheduler report: seed 42, natural SNR 0.5, checkpoint `9ef8030cb805b433b98ec25c5dddefbac07a9e26`, BF16 RTX 3090, coarse/base/decoder batches 1/16/1. Scheduler off/on is exactly equal for the 256-square full crop, and separately for the tested subcrop order, in DEM and five climate outputs.
- Full versus split crops differ by **0.210693359375 m** max, also with the scheduler disabled. This is a demonstrated upstream request-shape/order numerical limitation, not a new B2 regression. The documentation correctly discloses it; exact request-order independence is not established.
- Real persistence report: four weighted coarse contributors replay exactly after rebuilding and explicitly reinstalling preparation, with zero coarse model calls. The report covers five persisted windows total out of a 6,160-window plan, not full-world preparation.
- Cancellation is checked before real model batches and completed coarse files remain reusable. The CPU suite verifies cancellation/resume and eviction, as well as full/split fusion, negative coordinates and mip provenance. The tile queue remains tile-level; the implementation and documentation do not claim a separately prioritised neural DAG.

## Limitations / missing evidence, not additional proven production failures

1. **Memory is a soft cache limit, not a peak allocation bound.** `read_rect` delegates a complete requested region; installed MemoryTileStore defers all eviction until the outermost access ends (`tilestore/__init__.py:198–203,257–266`). A large request pins its complete dependency working set. No admission check in `terrain_window_scheduler.py:140–171` estimates that peak. Server sparse blocking and `read_mip`'s one-million-cell limit help bound its present callers, but do not prove an arbitrary public rectangle is GPU-safe. Record allocated/reserved peak for the largest supported production tile and reject oversized API reads or implement numerically validated partitioning.
2. **Observability has unbounded history.** `StageCounters.unique` (`terrain_window_scheduler.py:23,83–85`) retains every visited index for the lifetime of the world, independently of window eviction. Long 30 m exploration can grow Python RAM substantially. Cap the detailed history or use bounded approximate counts; do not describe all scheduler metadata as bounded.
3. **Disk quota is per world only.** The 2 GiB quota covers window file payloads in one namespace; no aggregate world quota exists, as the documentation admits. The finite 6,160-window plan is approximately 707 MB using the observed 114,816-byte file size. Arbitrary seeds can accumulate indefinitely across namespaces. This needs an explicit product-level retention policy before broad deployment.
4. **Rebuild reuse requires reinstalling preparation.** `_build_hierarchy` reinstalls only the scheduler; the persistence wrapper is attached to the previous coarse tensor. The GPU verifier explicitly installs a fresh `CoarsePreparation` after rebuilding. Automatic persistence continuity through arbitrary `world.rebuild()`/parameter setters is not demonstrated; stale `world._terrain_coarse_preparation` must not be treated as an active installation.
5. **Manifest completeness is cross-block work.** The initial inspected `terrain_manifest._files()` did not include `terrain_coarse`, `terrain_window_scheduler`, `terrain_climate`, or the installed InfiniteTensor implementation/version; Torch/CUDA versions were also absent. A self-consistent JSON hash alone does not encode every implementation affecting fusion/replay. This was reported to the coordinator for B1 correction; this reviewer has not validated that subsequent patch.
6. **Timing and scope are insufficient for E3/E4 acceptance.** Scheduler stage seconds include recursive dependency work and are not additive exclusive stage costs. GPU work is asynchronous; `CoarsePreparation.network_seconds` stops timing before the following CPU serialization synchronizes the output. The reports contain no repeated p95 navigation timings, full-world elapsed time, or peak memory. The faster on-run versus off-run is not an isolated scheduler speedup: ordering/warmup differ. The verification scripts write metrics but do not fail on fidelity thresholds, nonfinite errors, or replay-model calls; convert them into actual assertions before using them as automated gates.
7. **Real-model coverage remains narrow.** Only one seed/crop and one split order are in the supplied GPU evidence. Real-model cancellation/resume, eviction/revisit, diverse coastal/mountain/flat corpus, seed zero and world-boundary cases remain unverified. CPU arithmetic fixtures cannot establish BF16 fidelity in those cases.

The implementation is a useful conservative foundation for credible learned terrain. It does not yet establish ultra-fast cold 30 m navigation or complete E3/E4, and the documentation mostly states that distinction accurately. Resolve the two persistence-integrity defects, retain the disclosed numerical limitation, and measure the remaining product gates rather than treating these narrow replay checks as their acceptance.

# Neural window scheduling and learned coarse persistence

The pipeline still uses the checkpoint's original `InfiniteTensor` graph. Its
coarse windows are 64²/stride 48, each latent pass is 64²/stride 32, and the
decoder is 512²/stride 384. The stage tensor store enumerates **every** window
intersecting a read and sums the original weighted numerator and denominator
in its original order. `terrain_window_scheduler.py` wraps the real stage
nodes for per-stage request, cache-hit, generated, regenerated, and unique
window counters. A thread-local interest check runs before each real model
batch. It leaves dependency traversal and batching unchanged, including the
accepted coarse/decoder batch one and latent batch 16 profile. Completed
windows remain reusable after cancellation; a partially fused region is never
returned as a finished pixel rectangle.

`TERRAIN_CANONICAL_LATENTS=1` selects an experimental, separate numerical
profile with latent batch one at every base pass. It is **NO-GO** for promotion:
the four-seed GPU replay still found 0.096–0.324 m whole/subcrop drift
in each of three tested request orders, and 2–11.9 m differences from the
accepted batch-16 baseline. The remaining drift has not been isolated to
fusion or reconstruction. This flag must remain opt-in.
The default latent batch remains 16 regardless of free startup VRAM. The
reference whole/split drift described below still applies to that default.
Per-stage counters use a fixed 1 MiB Bloom filter for estimated unique and
regenerated window counts; those estimates can undercount/overcount at scale.
Stage seconds include recursive dependency work and are not additive.

`read_rect(world, stage, i1, j1, i2, j2, check=...)` reads a fully fused stage
rectangle in stage cells. Direct `world.get()`, `coarse[]`, and `latents[]` also
pass through the instrumented graph after `configure_world()` and `bind()`.
`scheduler_status(world)` exposes counts. The server owns the HTTP job queue;
this lane does not add a separately prioritised, consumer-counted DAG of
individual neural nodes. An active model batch is the cancellation quantum.

`CoarsePreparation(root, manifest, bounds, budget_bytes).install(world)`
attaches after `world.bind()`, before coarse reads. It requires a complete,
content-hashed `terrain_manifest` identity. Both ordinary demand and the
background worker save the **weighted** model output for each canonical
window in `learned-coarse-windows-v3/<world_hash>/windows/`. Data, per-window
identity/digest sidecar, and progress cursor are replaced atomically. A cheap
readiness check verifies the NumPy header, byte count, world/index identity,
and the canonical linear taper bytes in the weight channel. Stable file stamps
cache that verification. A demand read also verifies the full SHA-256 digest,
FP32 shape, and finite values. Empty/truncated or altered-weight windows are
invalidated and recomputed. The stage geometry, dtype, batch profile, bounds,
weight digest and nine-cell climate context are recorded in a contract.
Mismatches refuse reuse. The default per-world window payload budget is 2 GiB;
`status()` reports used/budget bytes, total/completed windows, disk hits, model
windows, synchronized model wall time, and persistence time. Quota exhaustion
stops background preparation while demand continues in RAM, and status flags
the limit. There is no cross-world aggregate quota. After an explicit
`world.rebuild()`, install a new `CoarsePreparation` on the rebuilt tensor.
Live foreground and background preparations of the same world share one
in-process namespace ledger for completed indices, verified file stamps,
payload bytes, and quota exhaustion. A second install adopts peer writes
without rescanning every window on each quantum. A corrupt file found by
readiness or demand removes the shared completion entry and reconciles bytes.
The server serializes generation under its GPU lock; concurrent independent
processes writing the same namespace are not supported by this quota ledger.
The server freezes model, source-file and runtime-version hashes at startup for
all seeds in that process. Restart it after changing those files. The current
manifest includes some server and presentation sources, so edits to those
files can invalidate persistent neural windows more broadly than numerical
necessity; this is a declared cache cost, not a change in terrain values.

The background worker calls `step(world, budget_windows=1, check=...)` while
holding the existing GPU lock. One step computes at most the requested number
of missing coarse model windows and resumes from its saved cursor. The finite
plan covers the 40,000 × 20,000 km rectangle plus nine coarse cells of
climate context; the current geometry counts roughly 6,000 windows. Foreground
requests can fill any of them first. `ready_for_samples(xs_native, ys_native)`
tests all height and climate contributors through the same dense/sparse block
geometry used by the server, using file membership checks. It reports false
while any required contributor is absent, so the high LOD path retains its
conditioning preview until learned coarse coverage is ready.

`read_mip()` averages **decoded learned coarse metres** in aligned powers of
two after checking complete contributors. It returns `None` for partial
coverage unless generation is explicitly requested. Its provenance is
`source=learned-coarse`, `source_resolution=7680`,
`final_dem_mip=false`; it must not be labelled a mip of the 30 m final DEM.
These mip arrays are derived on request; canonical weighted source windows
are the persistent unit. Edge tiles that extend beyond the finite world need
the server's explicit no-data/clamping policy; full coarse coverage alone does
not make every unbounded external coordinate learned-ready.

Validation on the installed 30 m checkpoint, seed 42, natural SNR 0.5, BF16
CUDA, RTX 3090: scheduler disabled/enabled gave **0 m** maximum elevation
difference and zero difference in all five climate outputs for the same 256²
whole crop. The same comparison over four reversed subcrop requests was also
exact. The measured pre-B2 local runtime profile differed between one whole
read and those four subcrop reads by 0.210693 m maximum elevation, with climate
differences at most 0.0001221. This is one observation, not a bound on other
seeds, crops, profiles or hardware. A rebuild using persisted learned coarse
windows replayed with zero weighted-channel difference and zero coarse model
calls. Reports: `E:/TerrainDiffusionRuntime/window-scheduler-fidelity.json` and
`E:/TerrainDiffusionRuntime/coarse-persistence-fidelity.json`. Those numeric
claims apply only to the source hashes recorded in each report. Both were
replayed successfully after the shared-namespace fix and final manifest
dependency update. The verifier scripts fail
on nonfinite outputs, added scheduler drift, missing disk hits or replay model
calls, and record coast-sign changes and source hashes.
CPU unit tests in `test_terrain_windows.py` exercise negative coordinates,
full/split crops, scalar-base shape-sensitive order independence, eviction,
cancellation/resume, budgeted preparation, manifest/geometry checks,
incomplete mips, replay, quota fallback, and empty/truncated/altered-weight
window recovery. The completed 6160-window finite-world measurement and
remaining E3 throughput limits are reported in [final validation](VALIDATION_FINAL.md).

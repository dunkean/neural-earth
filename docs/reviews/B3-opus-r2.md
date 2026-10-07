I found no P0 or P1 issues. The fixes listed in B3-opus-r1 and B3-astra-r1 hold up on reading, but each of the four P2 findings below needs a fix or an explicit B5 limitation. As requested, I edited nothing and ran nothing, so every timing figure below is an estimate.

## Fixes confirmed by reading the code

| Earlier finding | Status | Evidence |
|---|---|---|
| Opus P1-1: preview tiles flickered on every progress tick | **Fixed.** The old tile stays on screen and is swapped only when its replacement arrives. Previews expire after 2 s, and a stale tag is set at admission if the coarse revision changed. | `index.html:85`, `:106`; `test_terrain_freshness.cjs:19-34` |
| Astra P2: a late preview could survive the final revision | **Fixed.** A preview whose response arrives after the revision changed is marked stale and requeued. | `index.html:81,85`; freshness test |
| Opus P1-2: cached previews waited for the GPU | **Fixed.** The readiness probe no longer waits for the GPU lock, never calls `get_world`, and checks each shared window once. | `terrain_server.py:330-339`, `:475-496`; tests at `:376`, `:418` |
| Opus P1-3: foreground and background disagreed on finished windows | **Fixed.** There is now one shared ledger per `world_hash`, and a window that passes validation from disk is counted as done. | `terrain_coarse.py:226-244`, `:319-322`, `:354-355`; `test_terrain_windows.py:176-215` |
| Opus P1-4: natural showed procedural input at regional zoom | **Resolved for the default path.** Background preparation now starts automatically after the overview loads; `prepare=0` turns it off, and the benchmark uses it. Visual approval is still pending. | `index.html:87,90`; `benchmark_browser.cjs:21` |
| Opus P2-1: priority test could not fail | **Fixed.** It now asserts `['visible','coarse','coarse']` without stopping first. | `test_terrain_background.py:30-58` |
| Opus P2-2 and P2-3: LOD 1–2 mip validity, unreachable LOD ≥7 mip branch | **Fixed.** Promotion uses the full mip validation, is limited to LOD 1–2, and a corrupt child is tested. | `terrain_server.py:499-502`; test `:393-416` |
| Opus P2-4: provenance labels | **Fixed.** The blur is now named in the label, and preview source resolution is 7680. | `terrain_server.py:737,746` |
| Opus P2-5: identity could change mid-run | **Fixed.** File hashes are snapshotted once at startup. The very broad identity is still a declared limitation. | `terrain_server.py:72-74,127-150`; test `:202` |
| Macro profiles skipped by the restart quota scan | **Fixed.** | `terrain_disk_cache.py:12,97`; test `:22-54` |

## P2 findings

**P2-A: preview promotion mostly stalls while preparation runs** (`terrain_server.py:480-488`)
- **Cause:** the probe gives up whenever the GPU lock is held. Background quanta are queued back to back, so the lock is held most of the time.
- **Effect:** cached previews whose windows are ready mostly stay previews until preparation ends (estimated minutes). Tiles that miss the cache still compute learned output correctly.
- **Second case:** the probe only looks at worlds already loaded in memory. After a restart with `prepare=0`, previews cached in an earlier session keep being served as hits even though their windows are on disk. Default auto-preparation hides this because it creates the background world.
- **Reproduce:** run preparation, view LOD 9–10 previews of an area that has finished, and record how often `X-Terrain-Stage` stays `conditioning-preview`.
- **Fix:**
  - Protect `_valid_metadata`, `_save_window` and `_invalidate` with a small per-namespace lock and drop `gpu_lock` from the probe. The preparation object only needs `output_window` and disk for this.
  - When no world is loaded, queue a low-priority promotion job in the jobs lane rather than on the HTTP thread.

**P2-B: previews and LOD 1–2 tiles are re-downloaded forever** (`index.html:74,80,85`)
- **Cause:** previews expire every 2 s and LOD 1–2 decoder tiles every 10 s, with no backoff even when nothing can change (`prepare=0`, a stopped job, or a corrupt mip child). Each check re-downloads the full tile (about 391 KB in GPU mode), re-uploads the texture and increments `counters.evictions`.
- **Second problem:** these refreshes have the same priority as genuinely missing tiles, so with 4 transfers at a time they can delay new coverage.
- **Fix:**
  - Rank refreshes after missing tiles (priority + 1000).
  - Back off the expiry, for example 2 s rising to 30 s while the revision is unchanged.
  - Count replacements separately from evictions.

**P2-C: automatic preparation has a first-quantum stall and no total disk cap** (`terrain_coarse.py:236-237`, `index.html:90`)
- **Stall:** on a fully prepared world, `install` validates every planned window (about 6,100 sidecars, 16 KB weight reads and SHA-256 each, roughly 100 MB) while holding `gpu_lock`. This happens in the first background quantum, which cannot be interrupted, right after the overview appears.
- **Disk:** each world visited now automatically takes about 0.70 GB (114,816 B × about 6,100 windows) under `OUTPUT/coarse-worlds`. That directory has a 2 GiB cap per world but no cap across worlds.
- **Fix:** measure the first-quantum time after a restart. If it is significant, do the scan outside `gpu_lock`. Either skip auto-start above a total coarse-worlds budget, or state the per-seed disk cost explicitly as a B5 limitation.

**P2-D: background status labels and a remaining failure mode** (`terrain_background.py:61-69`)
- A job that finishes the whole world reports `budget-complete`; it should report `complete`.
- A `QueueFull` error still marks the job permanently `failed` (Opus P2-8, not fixed). That matters more now that preparation starts automatically.

## Test and validation gaps
- No test shows a cached preview being promoted when the lock is free and the windows are ready; there is only the busy-lock test.
- "Parent revalidation" in the freshness test only checks the expiry flag (`test_terrain_freshness.cjs:35`). It does not show the tile being requeued or the server swapping a decoder tile for a final mip over HTTP.
- Benchmarks should record `/api/status.coarse_preparation.state` at the start. `prepare=0` only isolates the client; a job started by another tab can still be running.

## Verdict: **ACCEPT WITH LIMITATIONS**

Fix P2-A and P2-B now if cheap, or carry them into B5 explicitly along with P2-C and P2-D.

**Outstanding gates** (nothing here certifies performance or visuals):
- Final GPU, corpus and browser reruns on the frozen source snapshot, with source hashes.
- Measurements still needed:
  - cached-preview latency, and promotion delay while preparation runs;
  - readiness-probe cost at LOD 10–12;
  - time for one LOD-12 area-mean tile;
  - first-quantum time after a restart;
  - full-world preparation time and disk use;
  - peak GPU memory with three worlds loaded.
- On CUDA: decoder tiles vs final mips at LOD 1–2, and LOD 8 interpolated vs LOD 9 area mean.
- A browser time series of missing and approximate coverage during pans, plus preview-to-learned replacement in a real browser.
- Visual approval of natural at LOD 7–9, preview vs learned.
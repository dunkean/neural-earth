**Verdict: CHANGES REQUIRED.** The core geometry is sound, but three defects in this block affect navigation and background preparation. A fourth problem is a fidelity regression that needs a product decision.

This is a read-only review. I ran nothing, so every timing statement below comes from reading the code, not from measurement. I compared against the pre-B3 copy in `E:/TerrainDiffusionRuntime/audit-implementation/baseline` to tell which behaviours this block introduced.

## What checks out (by reading the code)
- **No hidden network work at LOD ≥7.** For a seed nobody has visited, `_available_world` returns `None` and the tile falls back to the labelled preview without loading the model (`terrain_server.py:400-410`).
- **Area mean (LOD 9–12).** Each display pixel averages exactly its own coarse cells, and averaging is done in metres, after squaring. Reads go in 32-pixel chunks; at LOD 12 the largest read is about 1.8 M values, well under the scheduler's 4 M cap. Readiness checks exactly the same rectangles that are later read, and an incomplete fusion raises an error instead of returning a partial value (`terrain_server.py:295-348`).
- **Exact mips at LOD 1–2.** Each child tile contributes only its own interior, values stay signed with 0 m as sea level, a missing child means no mip, and children with a different world identity or a non-decoder stage are rejected (`terrain_final_mips.py:35-55`, `terrain_server.py:438-454`).
- **World bounds.** Tiles entirely outside the world are refused (`_tile_coordinates`), the client clips drawing to the world edge, and the last partial 30 m column is handled.
- **Priority.** Background work runs in the lowest priority band (4000), so visible tiles and parent fallbacks are chosen first between background quanta (`terrain_jobs.py:168`).
- **Natural baseline and documentation.** `natural` is the default everywhere, including `benchmark_navigation.py`. The temperature lapse correction is applied exactly once, and the WebGPU no-go is stated in `README.md:49`.

## P1

**P1-1: every background progress tick makes the whole preview view flicker and reload** (`index.html:103`)
- **Cause:** Every 2 s, if `complete_windows` changed, the client drops every `conditioning-preview` tile at once. During preparation it changes on almost every poll.
- **Effect:** Tiles and their parents vanish, the overview underlay shows through, and everything is fetched again (the new `coarse_revision` parameter also bypasses the browser cache). Most refetches return the same preview, because readiness rarely changed for those tiles.
- **Reverse problem:** when the windows come from the user's own navigation rather than the background job, the counter never moves, so stale previews are never replaced.
- **Reproduce:** natural seed, world view, click "Préparer le monde NN", then watch `terrainDebug.snapshot().rendered.sources` and `counters.evictions` every 2 s. The mock test never covers this, because its `/api/status` has no `coarse_preparation`.
- **Fix:**
  - Mark preview tiles stale instead of deleting them, and let the existing replace-on-arrival in `fetchTile` swap them.
  - Have the server report a learned-ready marker per tile (for example a header), and refetch only tiles whose readiness can have changed.
  - Also refetch previews periodically when no background job is running.

**P1-2: serving a cached preview tile waits for the GPU** (`terrain_server.py:457-470`, called from `:479-481`, `:690` and `:793`)
- **Cause:** On every disk-cache hit for a preview tile, `valid_physical_report` calls `learned_tile_ready`, which takes `gpu_lock`.
- **Effect:**
  - A cached world-view tile waits behind any running decoder tile (seconds) or background quantum. That breaks the audit's target of useful coverage in under 250 ms after a teleport when a preview or cache exists.
  - Under that lock, `_available_world` can call `get_world`, which loads the model and can evict a foreground world, all on an HTTP thread (`:228-231`).
  - The readiness check itself scales badly: at LOD 10 one tile calls `complete()` for about 26×26 sample groups, each checking several windows with two `stat` calls apiece (an estimate, not measured).
- **Fix:**
  - Answer readiness from disk without `gpu_lock`: keep a per-`world_hash` set of persisted windows, or use a small separate lock for the `worlds` and background mutations.
  - Never call `get_world` from a readiness probe.
  - Collect the unique required window indices once per tile and check each only once.

**P1-3: background and foreground disagree about which windows exist, so the background job never finishes** (`terrain_server.py:201-216` and `:219-232`, with `terrain_coarse.py:200-207, 364, 379`)
- **Cause:** B3 creates a separate background world, so the same `world_hash` has two `CoarsePreparation` objects. `_persisted_indices` is filled only at install time and in `_save_window`. Loading a window from disk, or finding it valid, never adds it to the set.
- **Effect:** Windows the foreground persists after the background world was created are never counted. Once its own share is done, every background quantum walks the whole ring and re-reads (SHA-256 plus `np.load`) every window the foreground produced, while holding `gpu_lock`. `complete_windows` never reaches `total_windows`, so the job keeps going until the 8192-quantum budget runs out and then reports `budget-complete`, never actually complete. Progress in the UI is under-reported.
- **Reproduce:** start preparation, browse native tiles in a region the background has not reached yet, and let preparation wrap around. Disk hits keep rising while `complete_windows` stays below `total_windows`.
- **Fix:** in `step()`, treat an index as done when `_valid_metadata(index)` passes, add it to the set and continue *before* calling `ensure_window`. Do the same on a disk hit in the `persisted` wrapper. Better still, share one `CoarsePreparation` per `world_hash` between foreground and background.

**P1-4 (needs a product decision): the natural world now shows procedural input at regional zoom by default** (`terrain_server.py:403-410`)
- **Before:** baseline `terrain_server.py:438` ran the learned coarse network for natural at every zoom level.
- **Now:** from LOD 7 upward (3.84 km/pixel and coarser), natural shows the procedural network input until the whole required area has been prepared. Preparation only starts when the user clicks the button.
- **Why it matters:** the audit (§4) names showing this input when zoomed out as a cause of the soft, "potato" look of `earth`. The legend labels it honestly and the generator itself is unchanged.
- **Fix, either:**
  - Start a low-priority background preparation automatically when a natural world opens.
  - Or compute the learned coarse on demand at LOD 7–8 when the number of missing windows is small, keeping the preview only as a temporary placeholder.
- If speed is preferred, this needs explicit sign-off as a documented limitation.

## P2
1. **The background priority test can't detect broken priority** (`test_terrain_background.py:49-58`). `stop()` is called before the background job runs, so it returns without recording anything either way, and the result is `['visible']` even if priority were inverted. Fix: don't stop; assert `['visible','coarse',…]`.
2. **LOD 1–2 can recompute forever.** `valid_physical_report` only checks that the child files exist (`:474`), while `existing_final_mip` fully validates them. If a child is corrupt, has another identity, or is not a decoder tile, each request rejects the cached tile and reruns the decoder. Fix: base the check on `existing_final_mip(...) is not None`, or record that the mip is unavailable.
3. **A test covers code that can never run.** `plan_mip` stops at LOD 2, since `(304·step)² ≤ 2M`, so the LOD ≥7 final-mip branches (`:476-480`, `:721-723`) are unreachable, yet `test_final_mip_with_unprepared_climate_uses_preview` tests LOD 7. Either remove the branches or test LOD 1–2 with real child files.
4. **Provenance labels.**
   - Area-mean tiles get an extra Gaussian blur (σ = 0.65 px, `:415-416`), but `height_filter` reports only `coarse-physical-area-mean`.
   - Preview tiles report `source_resolution` equal to the display resolution (`:711`), even though their source is the 7.68 km grid.
5. **World identity covers too much** (`terrain_manifest.py:59-76`).
   - The hash includes `terrain_server.py`, the job, disk-cache and background modules, and `relief_map.py` (a colour-palette file). The audit (§8.1) says the palette must not be part of the heightmap identity.
   - Any edit to UI or scheduling code therefore orphans every prepared coarse world and the physical cache, and there is no global disk quota (`README.md:51`).
   - Because `world_manifest` is computed lazily and kept in an LRU cache, editing files while the server runs can also make a seed's identity disagree with the startup `PROFILE`.
   - Fix: hash only the code that affects numbers, and take one snapshot of the hashes at startup.
6. **The benchmark labels can mislead.**
   - `benchmark_browser.cjs` calls its first scenario "cold" but always uses seed 42 and the reference region, which are probably already cached.
   - It clicks `#native` before the minimap (`:24`, also `:28`), which can launch a decoder tile at the world centre that can't be cancelled, inside the measured time.
   - One minimap pixel is about 182 km, so the click can miss the reference coast. Use a debug camera setter instead.
   - `benchmark_navigation.py` names its cases `native-cold` even when they hit the cache, and it never requests LOD ≥7 learned or area-mean tiles.
7. **The 5760-pixel mock covers less than it may seem.** It never exercises preview → learned replacement, `coarse-area-mean`, `final-dem-mip` or preparation status. Passing it proves UX mechanics only.
8. **Minor.**
   - Edge tiles at LOD 0–6 are not clipped, so the network computes outside the world (cost only).
   - A `QueueFull` during a background quantum marks the job permanently `failed` (`terrain_background.py:66`) instead of retrying.
   - `docs/AUDIT_IMPLEMENTATION.md:25` still says "B3 en préparation".

## Outstanding gates (not certified)
- CUDA and browser benchmark reports. No speed claim until they exist, including:
  - waiting time on cached preview tiles;
  - cost of the readiness check at LOD 10–12;
  - time for one LOD-12 area-mean tile;
  - time and disk use for a full world preparation of about 6,200 windows (my estimate is around 0.7 GB).
- Numerical comparison of LOD 1–2 decoder tiles against the final mips on CUDA, given the known sensitivity to how requests are split.
- Measured difference between LOD 8 (interpolated) and LOD 9 (area mean), per audit §13.3.
- A browser test of preview → learned replacement without flicker.
- Visual approval of the natural world at LOD 7–9, preview against learned, which depends on the P1-4 decision.
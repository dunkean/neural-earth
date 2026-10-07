I've read the two audit docs, the block's code and its four tests, and checked how the renderer handles cropped parent tiles. I did not run anything; the PASS results below are as reported. **Verdict: CHANGES REQUIRED.** There are two proven P1 defects, both in partial coverage, and both are small to fix. The LOD 11 cap, hiding fine tiles on zoom-out, the stage ordering and cancellation look correct on reading.

## P1: proven defects

**P1-a. Cells that already have finer cached tiles still request every stage ancestor** (`index.html:75`, `terrain_lod.js:27-49`)
- **Cause:** `refinementLod` correctly ignores cells covered by a cached tile finer than the current stage when deciding the stage. But the request loop at `:75` asks for the stage ancestor of every cell whose exact target tile is missing (`!tiles.has(t.key)`). It never checks whether that cell already has a cached LOD between the target and the stage.
- **Reproduction:**
  1. Zoom to 60 m/px at A and let LOD 1 finish.
  2. Press "30 m / pixel", or pan half a screen and go native, so part of the view has no ancestor ≤4.
  3. The stage becomes 4, then 3, then 2. At each stage, cells already covered by the cached LOD 1 tiles also request their LOD 4, 3 and 2 ancestors.
  4. `plan()` picks the closest cached tile first, so those cells keep showing LOD 1 and the new tiles are never displayed.
- **Impact:**
  - It breaks the "3-2-1-0 as needed" requirement and the doc's claim that a covered revisit skips its ancestors (`docs/RUNTIME_OPTIMIZATION.md:21-22`).
  - The wasted LOD 3 and LOD 2 tasks (about 3.0 s and 4.6 s median each) have the same priority (2000+distance) as the needed ones. They compete for the 4 HTTP slots and the GPU lane.
  - The next stage also waits for them: `fetchTile` only schedules a refresh when `!queue.length` (`index.html:90`).
  - Some of this work may be reused on the server through stored NN windows. That is unmeasured.
- **Fix:**
  - Add `refinementTasks(tiles,targetLod,view,opts)` to `terrain_lod.js`, returning `{level, ancestors}`. A cell needs the stage-L ancestor only if it has no cached tile at any LOD from the target to L. Use only this result at `:75`.
  - Unit test: a cached `{lod:1,tx:-1,ty:-1}` over a view spanning two LOD 1 footprints. Assert that the covered footprint gets no LOD 4, 3 or 2 ancestor request.

**P1-b. Staged coarse tiles cover the natural 30 m initial image, which is ignored as coverage** (`index.html:50`, `:68`, `:75`; `terrain_lod.js:27-49`)
- **Cause:**
  - `hasCoverage` (`:68`) counts `initialImage` as LOD 0 coverage, but that check is only used for LOD ≥4 parents (`:76`).
  - `refinementLod` does not know about the initial image, so at target 0 over the initial region it returns 4.
  - The LOD 4 crops are then drawn after the initial image on the 2D canvas, or on `#gpuMap`, which sits above it.
- **Reproduction:** natural profile with a cold tile cache, click "Région initiale". In the mock, the 30×20 km initial area on a 1000 px view lands at LOD 0.
  - `snapshot().refinement.active===4`.
  - The native relief is replaced by a 7.68 km coarse crop, then by LOD 3, 2 and 1, before LOD 0 arrives.
  - Previously, `hasCoverage` prevented this.
- **Fix:**
  - Pass extra coverage rectangles to both `plan` and `refinementLod`: `{bounds: initial_bounds, lod: 0}`, when `viewMode==='relief'` and the image has loaded.
  - Cells fully inside those bounds count as covered and get no fallback patch.
  - Add a Chrome mock case: fit to the initial region, then assert there is no `/height/` request with lod>0 inside it and no fallback patch over it.

## P2: hardening and misleading validation

1. **Gating ignores the tile's stage** (`terrain_lod.js:30`). Any LOD ≤4 tile counts, including a `conditioning-preview`. The requirement that learned coarse is shown before LOD 3 therefore depends on the server never serving previews at LOD 4. The mock's `preview_min_lod: 7` suggests that, but the client never checks. Either require `tile.stage` to be one of the learned stages, or assert `world.preview_min_lod>4`.
2. **One failing tile can stop all refinement.** Stage advance is global, so a stage tile that keeps failing (3 s backoff loop) or keeps returning 409 holds the whole viewport at that stage. Let the stage advance past cells that are in failure backoff, or add a test showing it recovers.
3. **Hard-coded report flags** (`verify_progressive_zoom.cjs:22,26,70`).
   - `noLOD12:true` and `slowLOD3HasPresentedLOD4:true` are constants.
   - The `lod<=11` checks run inside Playwright route handlers, where a failure may surface as a hang or an unhandled rejection rather than a clean failure.
   - `/tiles/` (PNG) requests are never checked.
   - Fix: record LODs from every request and assert `requests.every(r=>r.lod<=11)` after the run; compute the report fields from data.
4. **The presentation check is too weak** (`verify_progressive_zoom.cjs:57`). `lods.some(l=>l<=lod+1)` passes if any single patch is coarse enough. Assert `Math.max(...lods)<=lod+1` and that the patches cover the clipped view.
5. **The "cancel" phase never checks cancellation** (`verify_progressive_zoom.cjs:63-68`). Requests tagged `cancel` are never inspected. Hold LOD 4 in that phase, click fit, then assert that `counters.aborts` went up and that no request with lod<9 was issued after the fit epoch.
6. **"Revisit" is only an immediate refresh** (`verify_progressive_zoom.cjs:60-62`).
   - Nothing tests pan-away-and-back, revisit after eviction, partial coverage (P1-a) or the initial image (P1-b).
   - The unit tests (`test_terrain_lod.cjs:27-39`) use one cell only.
   - The doc wording at `docs/RUNTIME_OPTIMIZATION.md:28-29` is broader than what is tested.
7. **Memory eviction is untested under pressure.** `evict()` (`index.html:48`) first removes the oldest tile not in `interests`. During refinement, `interests` excludes the displayed previous-stage parents and the cached target tiles. Only the touch-on-draw ordering protects them. There is no proven defect, but nothing tests stage progression near the 192 MiB limit. A drop-then-refetch loop of LOD 4 tiles is possible in theory.

## Checked and correct on reading
- **LOD 11 cap:** applies on every path (`zoom`, `fitBounds`, `nominalLod`, `lodForCamera`, the `lod+2` parent clamped by `MAX_LOD`, and prefetch). 61 440 m/px is exactly LOD 11, so the arithmetic is exact.
- **No fine tiles after zoom-out:** `plan` (`terrain_lod.js:8`) never uses tiles finer than the target, for both the 2D and GPU paths. The renderer applies the crop in its vertex stage (`terrain_renderer.js:107`).
- **Cropped parents:** the crop coordinates for cropped parents, negative coordinates and world-edge clipping are covered by unit tests.
- **Stage order:** LOD 4 is requested at priority 0+distance and finer stages wait during wheel zoom (`:78`). Prefetch waits until refinement ends. In-flight requests that leave `interests` are aborted (`:79`).
- **Freshness:** the cache-age and preview-revision logic is unchanged and still covered by `test_terrain_freshness.cjs`.
- **Cache identity:** the lookups in `plan` and `refinementLod` key on LOD/tx/ty only. This is safe today because world, mode and backend changes clear `tiles` and discard late responses by epoch, but it is fragile.

## Remaining gates (not certified here)
- An end-to-end run against the real server and NN, recording time to first LOD 4 presentation, the time of each later stage, and how many stage tiles were requested.
- Server-side effect of aborts and of the stage tiles listed in the posted view.
- GPU and presentation timing.
- The visual quality of each stage.
- Eviction under memory pressure.

Visual quality and performance are not measured by these mock tests and are not claimed here.

Separately, the Google Calendar, Google Drive and Sentry connectors need authorizing in your claude.ai connector settings before they can be used. None was needed for this review.
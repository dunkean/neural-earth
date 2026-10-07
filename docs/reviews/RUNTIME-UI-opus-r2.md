**Verdict: CHANGES REQUIRED.** The two re-review fixes (P1a, P1b) and the frame barrier are correct on reading. But I found a new P1 on the normal panning path at native zoom. It is not covered by the accepted "permanently failing stage" limit. The fix is small.

I only read files: no commands, no tests. The PASS results cited in `RUNTIME-UI-astra-r2.md` are as reported. `docs/RUNTIME_OPTIMIZATION.md` changed while I was reviewing (it now records the failure-stage and eviction limits). I re-checked the code and test line anchors afterwards and they had not moved.

## P1: proven defect

**A small pan at 30 m/px cancels native work already under way for cells that finished refining** (`index.html:75`, `:78-79`, `:83`; server side `terrain_jobs.py:81-86`, `:114-120`, `:188-196`)

- **Cause:** the refinement stage is one value for the whole view. When any new cell enters the view without ancestors, the stage drops back for every cell.
  - `:78` adds the target tile only when `refinementLevel===lod`.
  - `:79` adds only stage ancestors.
  - So target LOD 0 tiles that are in flight for cells already covered by LOD 1 drop out of `interests`. They are aborted at `:83` and removed from the posted view.
  - The server then cancels the queued job, and cancels a running job at its next quantum boundary (`check_current_interest`).
- **Reproduction (by reading):**
  1. Natural profile, `mpp=30`, view 1000×720. Let the stages finish so LOD 0 requests are in flight.
  2. Press ArrowRight a few times (3 km each) until a new LOD 1 footprint (15.36 km) enters the view.
  3. `refinementLod` returns 1. `hasRefinementCoverage(...,0,1)` checks no levels at all (`parent<level` is empty), so only LOD 1 ancestors are added.
  4. Every in-flight LOD 0 fetch is aborted (`counters.aborts` rises) and its server job is cancelled.
  5. LOD 0 restarts only once the new strip has its LOD 1 tile.
- **Impact:** during a sustained pan at native zoom, this repeats each time a new footprint at LOD 4, 3, 2 or 1 enters the view. The centre's native tiles keep being cancelled and requeued. Cached target tiles also leave `interests`, so they cannot be revalidated and become eviction candidates.
  - The cost has not been measured: B2's window store may let some of the NN work be reused.
  - It directly affects the product priority (fast navigation down to 30 m).
- **Minimal fix (keeps the global barrier for *starting* new work):** in the `else` branch at `:79`, keep the interest when the target fetch is already in flight, e.g. `else if(inflight.has(t.key))add(t)`.
  - Cleaner alternative: give each cell its own stage, one level finer than its finest presented coverage. Order the stages by priority band instead of excluding finer requests.
- **Test:**
  1. In `verify_progressive_zoom.cjs`, hold the LOD 0 responses.
  2. Set `cx += 16000` so a new LOD 1 footprint enters.
  3. Assert that `aborts` did not increase for LOD 0 keys still in view.
  4. Assert that `desiredView.tiles` still lists them.
  5. Assert that the new strip asks for its ancestor first.

## P2: misleading or weak validation

1. **The native-image overlay check proves nothing** (`verify_progressive_zoom.cjs:84-86`).
   - Line 86 runs after `pending===0`. By then every cell has its exact LOD 0 tile, so `plan` returns no fallback patches with or without `protectedBounds`.
   - Line 85's `.every` also passes on an empty array.
   - The request check itself does discriminate: without the fix, the stage would be 4.
   - The no-overlay property is currently only proven at unit level (`test_terrain_lod.cjs:42-46`).
   - **Fix:**
     - Hold the LOD 0 responses in this phase and wait for a drawn frame with `pending>0`.
     - Then assert that no fallback patch intersects `initial_bounds`, and that `initialRequests.length>0`.
     - Add a second view that straddles the image edge. Assert that LOD 4 requests are made only for cells not fully inside, and that the fallback pieces stay outside the image.
2. **The cancel phase's "no obsolete fine request after fit" check is likely empty and has a timing gap** (`:72-79`).
   - The LOD 10 tiles from the opening fit are still cached, so the fit probably issues no requests at all.
   - Requests sent between the click and `phase='after-fit'` are tagged `cancel` and escape line 79.
   - `counters.aborts` only counts calls to `abort()` in the client.
   - **Fix:**
     - Take `mark=requests.length` before the click and assert `requests.slice(mark).every(r=>r.lod>=9)`.
     - Assert that the key `/42/4/3/3` never enters the cache after release.
     - Optionally, check that Playwright's `requestfailed` event reports `net::ERR_ABORTED`.
3. **The partial-coverage VM test covers stage 4 only** (`test_terrain_refinement.cjs:11-17`). Astra's check of stages 3, 2 and 1 was not committed. **Fix:** seed LOD 4, then 3, then 2 for the cold half and assert at each step that `interests` holds only `[[L,1or…,0]]` for that half. Add a case for the P1 above in the same harness.
4. **"Presented" means "in the plan", not "drawn"** (`index.html:54-56`). Every `visiblePlan` entry is marked presented, even a GPU tile that failed `renderer.hasTile` and was left out of `gpuRects`. This is not a proven defect, because the tile's heights are kept and re-uploaded next frame. **Hardening:** mark presented only the entries actually pushed to `gpuRects` or drawn with `tile.image`.

## Checked and correct on reading

- **P1a:**
  - `hasRefinementCoverage` checks presented levels from `lod+1` up to the level just below the current stage (`:72`).
  - Cells covered at a finer level are skipped at `:79`.
  - `refinementLod` uses the same rule (`terrain_lod.js:48-56`).
  - Negative tile coordinates use `Math.floor` on both sides.
- **P1b:**
  - `nativeImageBounds` applies only in relief mode, only at LOD 0, and only once the image has loaded (`:31`).
  - `refinementLod` drops cells that lie fully inside the image (`terrain_lod.js:46`).
  - `plan` subtracts the image area only from coarser-level patches (`:25`), so new LOD 0 tiles still replace the image.
  - The WebGPU canvas is configured `premultiplied` and clears with alpha 0 (`terrain_renderer.js:126`, `:238`), so the 2D image underneath shows through.
- **Frame barrier:**
  - New tiles start as `presented:false` (`:93`) and are not counted as coverage.
  - `drawFrame` calls `schedule()` when a tile is presented while the stage is still coarser than the target (`:57`).
  - I found no way for it to get stuck: a tile received off-view stays unpresented but does not gate any visible cell.
  - In a hidden tab, refinement pauses, as designed.
- **Strict max ≤ previous level (`:64`):** it works as a full-coverage check here only because LOD 11 tiles cover the whole world. That dependency is implicit in the test.
- **LOD ≤ 11:** checked on every recorded `/height/` and `/tiles/` request (`:87`), on top of the checks inside the route handlers.
- **Preview tiles at LOD 4 (earlier P2-1):** not possible, because the server serves `conditioning-preview` only at LOD ≥ 7 (`terrain_server.py:410`, `:422`).
- **Accepted limits:** the permanently failing stage, memory pressure, warm refresh versus a pan revisit, and the default world profile not being promoted to A4 are now stated in `RUNTIME_OPTIMIZATION.md:35-41` or this block's scope.

## Outstanding gates (not certified)

- The P1 fix, with a test that holds native responses while panning.
- An end-to-end run on the real server and NN: time to first LOD 4, time of each later stage, number of stage tiles, and a sustained native-zoom pan measuring tiles cancelled versus completed.
- Server-side reuse after cancellation.
- GPU/compositor presentation timing.
- Visual quality of each stage.
- Eviction under real concurrent generation.

None of this review measures performance or visual quality.

The Google Calendar, Google Drive and Sentry connectors need authorizing in your claude.ai connector settings before they can be used; none was needed here.
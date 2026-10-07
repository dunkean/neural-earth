# B5 focused correction review (P2-1 / P2-2 from Opus r1)

I only read files and ran nothing. The report `E:/TerrainDiffusionRuntime/audit-implementation/browser-continuous-pan.json` does not exist yet. Only `browser-continuous-pan-native.png` is there. The harness writes that image at `benchmark_browser.cjs:149`, before the pans start and before the JSON is written. So I report no continuous-pan metrics, and P2-1 is fixed in the instrument but has not been measured.

## What the correction gets right

- **The per-frame driver is correct.** `index.html` is a classic script and `drawFrame` is a global function that can be reassigned (`index.html:49`). `draw()` looks up `drawFrame` by name each time it runs (`index.html:48`). So the wrapper at `benchmark_browser.cjs:59-63` catches every coalesced draw, and the `finally` at :92-94 restores the original function after the two trailing frames, so the last pending draw is also counted.
- **`renderer.draw` is fully synchronous**, ending with `queue.submit` (`terrain_renderer.js:220-245`). So `draw_frame_cpu_*` really is CPU recording plus submission. The notes at :167-168 correctly say it is not GPU execution or presentation time, and that the <16.7 ms gate is still unmeasured.
- **The pan is continuous and time-based.** The camera moves 6 m/ms × the actual frame gap (:72), so a dropped frame doesn't distort the speed. The report records actual displacement and cross-axis drift (:104-105), which would reveal clamping.
- **Cadence, draw work and coverage are separate fields** (:107-122). This matches the r1 fix list.
- **P2-2's report table is right.** Client hashes, historical labels for the pre-fix runs, the 180.948 s failure kept, the 25.443 s rerun described as partly warm cache, and the scenario renamed to `native-region-transport` (`VALIDATION_FINAL.md:40-48`). The served files come from the repo root (`terrain_app.py:5`; `terrain_server.py:559-569`), so the harness's hashes match what the browser loaded, as long as it runs with the repo as its working directory.

## Findings

**P2-A (proven by reading the code): the wait after each pan can hang until the 300 s timeout.**
- **Where:** `benchmark_browser.cjs:88-91, 97-99`.
- **Cause:** while panning, `refresh()` runs about every 50 ms (`index.html:64,68`). The last frame's `schedule()` leaves a timer that fires within 50 ms. The snapshot whose `cameraEpoch` becomes the wait threshold is taken about two frames (~33 ms) after the loop ends.
- **Failure path:**
  - If that final refresh fires before the snapshot (roughly 2 times in 3, by timing), then `settled()` needs yet another `cameraEpoch++`.
  - At LOD 0 with prefetch off, the only thing that would trigger one is the fetch `finally` at `index.html:86`. It only does so if `pending > 0` once the queue is empty.
  - Status polling does not trigger one either, because decoder tiles never need revalidation (`index.html:85,106`).
  - So on a warm run where hits arrive inside the 50 ms throttle, nothing triggers another refresh. The wait times out, the script throws, and no JSON is written.
- **Impact:** the failure is loud and does not produce false data. I can't tell whether this is why the report is missing.
- **Fix:** in the last `tick`, before `resolve()`, record `endEpoch = cameraEpoch` and pass that to `settled()`. The timer that `schedule()` left on that frame guarantees a later refresh that sees the final position.

**P2-B: coverage lag is measured with prefetch off, which is not the product default.**
- **Where:** `benchmark_browser.cjs:138`; `index.html:10` (prefetch is `checked` by default); `index.html:73`.
- **Impact:** `coverage_pending_*` and `coverage_settle_after_pan_ms` describe a configuration users don't run, and the notes (:165-170) don't say so.
- **There is also a definition gap:** `visible.pending` counts tiles at the target LOD that are not loaded yet (`index.html:61`). It does not count blank screen, because the parent fallback still draws something (`index.html:72`).
- **Fix:** add a note "prefetch disabled; pending = target-LOD tiles not yet received, not blank screen area". Either run a second pass with prefetch on, or also record each sample's `rendered.lods` / fallback share.

**P2-C: the harness never checks that WebGPU is actually the backend.**
- **Where:** :144-151, :168.
- **Risk:** if WebGPU fails to start, the client falls back silently with only a `console.warn` (`index.html:104`). The note would still say "WebGPU command submission", while `drawFrame` measured only Canvas 2D. The transport filter `/height/` (:26) would also see no tile traffic, because the fallback fetches `/tiles/`.
- **Fix:** `assert.equal(native.backend, 'webgpu')` before the pans. Also record adapter info (`renderer.adapter.info`) and the fact that the run is headless next to `display_cadence_estimate_*`. In headless Chrome there is no physical display, so this is the compositor's frame rhythm, not a screen refresh rate.

**P2-D: some per-frame main-thread work is not counted; this is a wording issue only.**
- **Where:** :55-75, :113-117.
- **What is missing:**
  - `uploadTile` runs inside `fetchTile` (`index.html:83`), outside `drawFrame`.
  - Style and layout work triggered by the `innerHTML` / `textContent` writes on every frame (`index.html:56-57`) happens after the measurement.
  - The `terrainDebug.snapshot()` sampling every 100 ms runs inside the frame callback but outside both timers.
- **Also:** the camera update and the draw are timed separately and land in different callbacks of the same frame. The r1 request was one per-frame span.
- **Fix:** call `draw_frame_cpu` a lower bound on main-thread work per frame. Optionally record `renderer.getStats().uploads` / `lastUploadSubmitMs` per sample.

**P2-E: the second pan is not a return trip, and the evidence files aren't tied to their run.**
- **Name:** "pan-return" (:152-156) starts after Fit → Native, so it starts from the world centre, not from the end of the first pan. It covers a different region, which may be cold. Either rename it (e.g. `pan-west-from-world-centre`) or move the camera back to `sample.endCx` before starting it.
- **Screenshots:** they are written to fixed file names and neither their paths nor hashes are stored in the JSON. A failed run leaves a `-native.png` behind that a later JSON could be wrongly paired with — exactly the state of the directory today. Store the screenshot paths and sha256 in the report.
- **Transport attribution:** `transport` entries are pushed after an `await` (:27). A response from the previous scenario can therefore land inside a pan's slice. Push synchronously and resolve the headers later, or tag each entry with its scenario.

**P2-F: the old keypress metric is still presented as "pan" in places, and E3 overclaims.**
- `VALIDATION_FINAL.md:29` ("pan p95 16,8–16,9 ms, handlers 0,3 ms") and `:31` call the old keypress run "pan" with no historical qualifier. So does `AUDIT_IMPLEMENTATION.md:29`. Rows `VALIDATION_FINAL.md:30` and `AUDIT_IMPLEMENTATION.md:41` qualify it correctly, so the documents contradict themselves.
  - **Fix:** in those three places, replace the figure with "intervalle rAF pendant sauts clavier (instrument historique)".
- `VALIDATION_FINAL.md:13` says "caméra réactive" (responsive camera). The only support is the timing of the synchronous key handler.
  - **Fix:** write "handler synchrone ≤0,3 ms ; coût de dessin par frame en mesure" until the continuous-pan report exists.
- `VALIDATION_FINAL.md:33` says the 1.746 s post-preparation world view "peut encore présenter" (may still show) the preview. That view was in fact 100% `conditioning-preview` (client `04ddf`, see the doc's own `:45`).
  - **Fix:** state that plainly and point to the 25.443 s corrected promotion.
- Measurement rows `:24-36` have no column saying which report each figure came from. The hashes exist only per report (`:42-48`), so P2-2's attribution is only partly done.
  - **Fix:** add a report/hash column to `:24-36`.

## Verdict: **ACCEPT WITH LIMITATIONS** for the narrow correction

- **P2-2 is accepted** once the three wording fixes in P2-F are made.
- **P2-1:** the instrument's design is correct and honestly labelled. It stays **open** because no report exists. Fix P2-A before the rerun, and P2-B/P2-C before citing its numbers.

## Gates still open (not regressions)

- Frame time below 16.7 ms end to end, including GPU and presentation: **unmeasured**.
- Main-thread work per frame during a continuous pan: **pending** `browser-continuous-pan.json`.
- Cold sustained pan throughput, and cold native view <250 ms: **not established**. The first native view takes 5.2–5.7 s; the fast figures are cache hits only.
- Coverage lag with the default prefetch: **unmeasured**.
- E1 visual approval, E2 NO-GO, E4 per-seed cost and LOD 12 reconstruction: unchanged.

The Google Calendar, Google Drive and Sentry connectors need to be authorized in your claude.ai connector settings before they can be used. None of them was needed for this review.
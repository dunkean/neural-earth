**Verdict: ACCEPT WITH LIMITATIONS.** I found no P0 or P1. The native-pan fix from my r2 review is correct on reading, and so are the earlier P2 hardenings. What remains is test evidence for parts of the claim, one unmeasured latency tradeoff, a minor priority quirk, and a doc gap.

I only read files. I ran no commands or tests and made no edits. "Mock PASS" is as reported in `RUNTIME-UI-astra-r3.md`; I did not reproduce it.

## Checked and correct on reading

- **Native flights keep their interest.** `index.html:78` keeps a target-LOD cell whenever its tile is cached or already in flight. It recomputes the task fresh with `taskFor`, so its priority does not stack. Abort (`:86`) and the posted view (`:87`) both read from `next`. So these flights are neither aborted on the client nor dropped from the server's view. The server's cancel checks (`terrain_jobs.py:81-86`, `:188-196`) then leave the running job alone.
- **Intermediate flights keep their interest.** `:83` keeps in-flight tasks that match all of: same `epoch`, not a prefetch, `lod ≤ t.lod < refinementLevel`, and overlapping the visible area clipped to the world.
  - Zooming out still aborts: `t.lod < lod` fails.
  - Panning a tile off-view still aborts: the overlap test fails.
  - Old-world flights are already cleared by `openWorld`/`resetBackend`.
- **No new finer work before the stage barrier.** While `refinementLevel > lod`, `next` can only hold four kinds of entry:
  - cached or in-flight target tiles (`:78`);
  - ancestors at the current stage (`:79`);
  - kept flights (`:83`);
  - nothing from `:80` or `:84`, which only run when `refinementLevel===lod`.

  The queue filter (`:85`) excludes cached tiles and in-flight keys, so no new native or intermediate fetch can start.
- **Revalidating a kept cached target tile costs no NN work.** Only LOD 1–2 tiles expire (after 10 s). Their server path at most swaps in a final MIP read on the CPU (`terrain_server.py:499-503`, `:419-421`).
- **"Presented" now means actually submitted** (`index.html:53-56`): a tile counts only if it was drawn with `ctx.drawImage` or pushed into `gpuRects` after `hasTile`. Physical on-screen presentation is still not certified (already a stated limit).
- **VM stages 4/3/2/1** (`test_terrain_refinement.cjs:18-24`): I worked through the tile indices. They give exactly the right-half queues `[3,2..3]`, `[2,4..7]` and `[1,8..15]`, and the left half is never requested.
- **VM kept flights** (`:27-30`): LOD 0 is kept by `:78`, and LOD 1/2/3 by `:83` while the stage is 4.
- **Browser pan** (`verify_progressive_zoom.cjs:82-92`): at `cx=900000 → 916000` the new strip lacks LOD 3 tile 15 (LOD 4 tile 7 is already present), so the stage becomes 3. Overlapping LOD 0 flights are checked as not aborted and still in `desiredView`.
- **Initial-image check** (`:97-104`): it runs while `pending>0` with requests held, and asserts the requests are non-empty and all LOD 0.
  - It does discriminate. Cells tx −2 and tx 1 produce fallback slivers outside `[-15000,15000]`. Without the subtraction, whole-cell parents would overlap the image and the check would fail.
- **Cancel phase:** the mark is now taken before the click (`:75`), and the obsolete key is checked absent (`:81`).

## P2: missing evidence / weak validation

**P2-1: The pan-retention tests check only one moment, and only the "kept" side** (`verify_progressive_zoom.cjs:85-94`, `test_terrain_refinement.cjs:27-30`). Three parts of the claim are untested:

- **"No new finer job before the barrier."** Neither test asserts it.
  - With the current code, the VM `queue` is only `[[4,0,0],[4,1,0]]`, but nothing checks that.
  - Fix: add `assert(queue.every(t=>t.lod===refinementLevel))` after the VM refresh. In the browser, record `u.pathname` in `requests` and assert that no `pan-native` request with `lod<3` targets new-strip tiles before release.
- **"Off-view abort still works" during a pan.** Nothing guards it: no test asserts that a non-overlapping flight is aborted. If the overlap test at `index.html:83` were dropped, every test would still pass. Zoom-out is covered by the cancel phase; panning is not.
  - Browser fix: assert `panFlights.filter(f=>!retained.includes(f.key)).every(f=>f.flight.controller.signal.aborted)`. The panned-away LOD 0 column (tx 116) should be among them.
  - VM fix: add one flight at `tx=100` and assert it is aborted.
- **"Release and settle."** The test does not show the kept flights complete without restarting. An abort followed by a re-request would reach the same final state, because `holdNative` is already false by then.
  - Fix: after settling, assert `counters.aborts` equals its value right after the pan, and that each kept key appears exactly once among the recorded `pan-native` paths.

**Minor test notes (non-blocking):**

- The fallback check at `:104` would pass vacuously if `visiblePlan` had no fallbacks. Add `visiblePlan.some(p=>p.fallback)`.
- The check at `:81` can run before the released response's 25 ms delay has elapsed. It is still sound only because the abort counter at `:76` can only come from that single LOD 4 flight.

## P2: unmeasured tradeoff

**P2-2: Kept flights can occupy all 4 transfer slots and delay the leading edge** (`index.html:91`, `terrain_jobs.py:168`).

- **Cause:**
  - `pump` caps transfers at 4. In the browser scenario, all 4 slots are held by native LOD 0 flights, so the new strip's LOD 3 ancestor, which gates the stage, cannot even be requested until one completes.
  - Once submitted, stage 3/2/1 ancestors sit in the same server priority band (2000) as the kept native jobs. Their distance from centre is larger, so they also queue behind the native jobs already waiting on the server.
  - Measured costs here are 4.6 s for LOD 2 and roughly 5.7 s for a cold native tile, so the leading edge may wait several native tiles.
- **Context:** this is the intended consequence of keeping work instead of cancelling and redoing it. It cannot starve, because no new finer jobs are admitted. It is not a correctness defect, and its cost is unknown.
- **Gate:** measure leading-edge time-to-stage and cancelled-versus-completed tiles during a sustained 30 m/px pan on the real server.
- **Optional fix, only if the measurement is bad:**
  - Allow one extra transfer when the head of the queue is an ancestor at `refinementLevel` and every in-flight task is kept finer work; or
  - Post stage-1–3 ancestors in band 1000 so the server runs them ahead of kept band-2 native jobs.

## P2: nit and documentation

**P2-3: Kept validation flights get their +1000 priority penalty twice** (`index.html:76`, `:83`). `flight.task` already carries the +1000 from queueing, and `add` applies it again. A kept LOD 1–2 revalidation is therefore posted to the server at around 4000+d. The cost is fixed, not growing, and these requests are a cheap CPU read, so the impact is negligible.

- Fix: at `:83`, either strip the earlier penalty before calling `add`, or skip the adjustment when `t.validation` is set.

**P2-4: The block's doc does not describe this change** (`docs/RUNTIME_OPTIMIZATION.md:15-38`).

- It does not state that native and intermediate work already admitted is kept while a new strip pulls the stage back, that only new work waits for the barrier, or that this test exists.
- Fix: add two sentences there and list the pan-retention check. Also state the slot-occupancy limit from P2-2.

## Outstanding gates (not certified here)

- **Real server with NN:**
  - a sustained 30 m/px pan: tiles cancelled versus completed, leading-edge time to each stage, server reuse after cancellation (P2-2);
  - time to first LOD 4 and to each later stage.
- GPU/compositor presentation timing, as opposed to submission.
- Visual quality of each stage.
- Eviction under real concurrent generation.
- Macro A4 is not promoted to default; the default remains `natural` (`index.html:99`, `:119`). This is correct for this block and needs no change here.

I measured no performance or visual quality.

The Google Calendar, Google Drive and Sentry connectors need authorizing in your claude.ai connector settings before they can be used; none was needed for this review.
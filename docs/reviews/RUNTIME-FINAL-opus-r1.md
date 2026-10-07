**Verdict: ACCEPT WITH LIMITATIONS.** I found no P0 or P1 issues. There are six P2s: three are docs that are now out of date or inconsistent, one is an over-precise comparison, one is a missing unit test and one is a provenance gap. A4 v2 is acceptable as the **experimental UI default** with `natural` still selectable. It is not a learned-planet claim.

This was a read-only review: no commands, no GPU, no edits. I did not run the tests, so test pass counts are as reported to me, not verified.

## Findings (all P2)

| # | Kind | Where | Problem | Fix |
|---|---|---|---|---|
| F1 | Misleading summary | `docs/RUNTIME_OPTIMIZATION.md:106-119`, `:95-104` | The A4 summary still reports only the two original coasts (37.80% / 20.02% water) and lists "faible pente côtière" (low coastal slope) as a limit. The measurement says the opposite: shores have a median step about 3× ETOPO's and only 1.89% / 1.81% of land is below 100 m. It also leaves out the final-run failure at the weakest seed0 site: 7.04% NN land against 45.84% input land, maximum height 18.5 m. The browser section stops at the historical v2 receipts. | Replace "faible pente" with "steep shores, few lowlands". Add the weakest seed0 failure and link `CONTINENTAL_BOOTSTRAP.md:184-203`. Add the final browser receipts (see F3). |
| F2 | Default not recorded | `index.html:26,119` default to `macro-a4`; `AUDIT_IMPLEMENTATION.md:5,27`; `VALIDATION_FINAL.md:76`; `CONTINENTAL_BOOTSTRAP.md:203` | The shipped UI opens A4 by default. The top-level tracker and final summary never mention A4 and still describe the macro as something "à examiner" (still to be examined). `CONTINENTAL_BOOTSTRAP.md:203` says the default still needs the root's review and browser gate. | Record the decision in both top-level docs: experimental default, natural one click away, the final browser receipts, the 8-site NN run and its limits. Otherwise revert the default until that is written. |
| F3 | Not yet documented; easy to misquote | `browser-continental-final.json`, `-warm.json`; `benchmark_runtime_navigation.cjs:18-29` | Cold run: world overview at **12.5 s**. The startup LOD 11 input-preview tiles each spent **9.70–10.65 s in the server queue** against 0.28–0.34 s of compute. First LOD 4 at 6.19 s, which includes 6.09 s for the first coarse tile (first model use). Native 30 m coverage at 13.54 s, 25/25 cache misses. `lod11ReadyMs = 886` is measured from the zoom after the overview was ready (`.cjs:26-29`), so it is **not** a cold first-preview time. Warm run: overview 1.091 s, native 0.547 s, 25/25 hits. | Document these with that distinction. The queue time suggests the CPU-only previews wait behind a cold-start job. That is not proven: record a server job timeline before attributing it. |
| F4 | Comparison more precise than the method | `terrain_bootstrap_diagnostics.py:20-48,51-78`; doc `:151-156` | The A4 inventory starts from the 1024-wide overview grid, whose cells are 39 km. It only inspects ±~11.5 km around each overview shore edge, but the real crossing can be up to ~19.5 km away. ETOPO starts from its own 18.5 km raster edges, and bilinear upsampling to 7.68 km smooths its steps. A4 being steeper is robust; the "about three times" factor is not a matched comparison and probably overstates the gap. `:108` says "complete … sampled inventory", which reads as contradictory. | Qualify the ratio in the doc and fix the wording at `:108`. Optional: scan the whole coarse lattice (~13.6M cells, cheap on CPU) so the inventory is exhaustive. |
| F5 | Missing test | `terrain_bootstrap_diagnostics.py` | There is no unit test for axis labels, pair centres (`:43-46`) or the ETOPO longitude/latitude and half-pixel mapping (`:59-67`). I checked them by hand against the report and they are correct now (see below), but nothing protects them. The guard test (`test_terrain_bootstrap.py:66-75`) does not alter the diagnostics file, though the full-inventory check does cover it. | Add a synthetic planar-elevation test (known crossing → expected centre and axis), plus a small synthetic raster for the mapping. |
| F6 | Provenance | `continental-bootstrap-v2-nn/historical-comparison.json` | It was written by a `terrain_bootstrap_compare.py` with SHA `0a7206…`. The current file (`1f73…`) has since changed. The report-hash chain is still intact. | Archive the earlier comparer source or note the change. |

## Checked and accepted
- **Diagnostics math:**
  - Pair centres reproduce exactly (e.g. seed0 weakest at (−8,578,560, 8,559,360)).
  - The weakest pairs' gradient components along their own axis are 0.018 and 0.021, so they are tangential crossings as the doc says. The median pairs cross with components 1.70 and 1.80.
  - The ETOPO mapping matches `terrain_world.latitude` (y increases southward) and GDAL's pixel-corner convention.
  - ETOPO's maximum step is 1,753 m, so no nodata values leaked in.
  - The numbers at doc `:141-162` and the 8-site table at `:187-192` match the receipts.
- **Final NN run:**
  - All 8 DEMs are finite and the 512-point probe agrees with the input land sign everywhere (33.40% land).
  - `source-only-array-comparison.json` shows all 28 arrays (4 sites × 7) byte-identical between the previous and final runs.
  - The previous report hash `e1c7d0…` matches in both comparison receipts.
  - `terrain_macro.py` has the same hash (`7b008f30…`) in the CPU report, the NN manifest and the browser manifest.
  - The weakest seed0 failure is now reported honestly in `CONTINENTAL_BOOTSTRAP.md:192-202`.
- **Statistics cache schema 2:** the identity now includes `source_order` and the GDAL version. The nine tests cover file order, GDAL version, an implementation change, four write-failure stages with temp-file cleanup, and the disable switch. S1–S3 from my previous review are closed.
- **Benchmark harnesses:**
  - R1 is closed. The stub prediction is now non-zero, and the AST mutation is a real check: with the old zero stub it would have passed.
  - R2 is partly closed: conditioning is compared to upstream, but the output arithmetic is still checked against a hand-written reference. `RUNTIME_LOD_OPTIMIZATION.md:56` describes this accurately.
  - R3, R4 and R6 are closed. The only `-O`-sensitive checks are now explicit raises.
- **UI:** the final screenshot shows "Profil MACRO-A4", "MACRO-A4 · expérimental" and "Aperçu des entrées du réseau". The runtime core SHAs `03fd…` / `2d24…` match in both browser receipts. The historical v2 browser receipts are not reattributed.

## A4 as experimental default
It is supportable:
- It is labelled experimental in three places and `natural` is one click away.
- LOD 11 is labelled as an input preview.
- 7 of 8 actual NN sites hold up numerically: finite DEMs, open sea and plausible water fractions at the coasts, structured ridges.
- The sparse global probe preserves the land/sea sign.

Accepted limitations:
- The shore inventory is sampled, not exhaustive.
- The weakest axial step is usually a tangential crossing, not the smallest gradient.
- Shore steps are about 3× ETOPO's median.
- Lowlands are scarce (1.9% of land below 100 m against 14.7% for natural). This is exactly the hypsometry risk the audit warns about (`ASTRA_TERRAIN_AUDIT.md:183`).
- Polar / weakest-crossing coasts are unreliable.
- Climate bounds depend on the seed.
- The input is geometric; there is no full NN planet.

## Outstanding gates
1. F1–F3 doc updates, before the default is described anywhere as accepted.
2. A cheap CPU scoping of the polar failure: the share of sampled A4 shore pairs above 60° latitude. This needs the latitude recorded per pair; the saved shore-pairs files keep only cell indices and heights. Optionally, run A0 at the weakest seed0 coordinates for attribution.
3. Your own visual review of the native and LOD 11 boards; I am not certifying visual quality.
4. A full coarse-planet preparation under the A4 identity, plus a cold-start server timeline.
5. Still open from earlier reviews: LOD 2 paired timing with at least 5 alternating pairs, and seed-independent climate bounds.

The Google Calendar, Google Drive and Sentry connectors need authorising in your claude.ai connector settings before they can be used; none was needed here.
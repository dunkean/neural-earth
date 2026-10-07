# Bootstrap restart review — Astra r3

Date: 2026-10-07. Final source and measured-height review following r2, the cache/error fixes, and the matched-relief experiment. This reviewer ran no GPU work and changed no production files. The report is the only file written. Browser/Natural-byte evidence was still pending when this version was written.

## Verdict

**No new confirmed P1/P2 source defect.** The corrected height-provider identity, disk/HTTP cache isolation, structured initialization errors, once-only encoding and preserved runtime geometry remain consistent in the inspected frozen source. The new lazy terrestrial snapshot guard compares the on-disk bootstrap Python digest against the imported implementation identity and refuses a mixed-source manifest.

The completed evidence supports retaining altitude noise **0.05 as an explicit coast/relief tradeoff for the requested heightmap and navigation deliverable**. It does not establish uniformly better mountains, exact coastline preservation, a complete learned planet, or physically valid learned climate. The user has authorized the terrestrial default; no additional approval is required by this review.

## Independent receipt checks

Read `E:/TerrainDiffusionRuntime/terrestrial-bootstrap-v2-nn/current-final/neural-report.json` after it reached `complete=true`: **21 native 512 × 512 DEMs**, **eight world probes of 32 points each**, and `numerical_failures=[]`.

Independently checked:

- Canonical SHA256 for all **16 distinct referenced neural manifests**.
- Every implementation-file digest in those manifests against current on-disk source files.
- All 21 comparison NPZ digests and eight global-probe NPZ digests against the report.

There were **zero mismatches** in those checks. This is a direct source/artifact audit; it does not duplicate checkpoint inference or verify every model-weight byte independently.

All eight ordinary coast crops retain a learned zero crossing. Changed land/sea signs cover **4.14–25.77% of the crop**; median parent-to-NN edge-linear contour distances are **274–1,927 m**. The Earthlike seed 42 polar coast has 18.19% sign changes and a 2,291 m median. These are 15.36 km crops at 30 m spacing, and the measures are crop-limited nearest-contour distances, not proof of matching geological coastline segments.

Sparse world results now include both flat and spherical metrics. Thirty-two points per world are useful sampled consistency evidence, but cannot establish learned continent topology or global shoreline detail.

The separately completed `earthlike-global512` receipt reports **512 points**, 0.390625% sign disagreement with equal weights and 0.437505% with spherical weights. At this review snapshot its manifest differed from current source **only in `terrain_server.py`**; the other implementation files and executed SNR were consistent. A current-hash refresh was announced by the parent agent. Until that receipt is replaced or supplemented, these numbers are prior generation-math evidence rather than proof of the final complete runtime identity.

## Matched relief: measured tradeoff

Read the completed `matched-relief/relief-comparison.json`, covering four identical 512 × 512 footprints at altitude noise 0.05, 0.1 and 0.5, plus Natural. All terrestrial variants in these four footprints are 100% land, so a changing land mask does not explain their differences.

| Footprint | Relief p95−p05 ratio, 0.05 / 0.5 | Slope p95 ratio | 1 km high-pass std ratio |
| --- | ---: | ---: | ---: |
| Earthlike mountain | 1.109 | 0.632 | 0.687 |
| Archipelago mountain | 0.813 | 0.657 | 0.611 |
| Continents mountain | 1.197 | 1.267 | 1.384 |
| Earthlike plain | 0.531 | 0.407 | 0.485 |

The Earthlike and archipelago mountains are smoother at 0.05, while the selected continents mountain has greater upper-tail slope and detail. The selected plain is also smoother. This supports a measured choice, not a universal quality-improvement claim. There is no basis here to extrapolate to all mountain forms or all seeds.

Opened and visually inspected the Earthlike and archipelago matched-relief boards. Earthlike's all-land mountain footprint remains relatively gently textured; archipelago retains recognizable ridges and dissection at 0.05 while its finer texture is reduced. Natural at the Earthlike footprint is mostly ocean, so its greater-looking coastal detail cannot serve as a matched mountain comparison. Numerically, Natural has only 2.8% land in that Earthlike footprint and no land in the continents mountain footprint. Natural is all land at the archipelago footprint and is a usable local land comparison there, but that one case is not a general ranking.

## Climate boundary and runtime evidence

The actual polar climate limitation is confirmed in the current final receipt: precipitation is negative throughout the selected Gondwana, Earthlike and archipelago seed 0 polar-land crops, and across 83.75% of the Earthlike seed 42 polar-coast crop. Their precipitation minima are approximately −406, −330, −210 and −66.6 mm/year. The finite-height result does not validate these climatic outputs. The documentation correctly excludes physical-climate acceptance from the current height/navigation deliverable and exposes the raw measurements.

Read `terrestrial-bootstrap-runtime/cpu-bootstrap.json`. For its one newly generated u64 seed across four styles, full parent creation is **1.18–2.34 seconds**, verified disk reload is **37–45 ms**, and resident lookup is approximately **2.5–3.7 microseconds**. Continents used three deterministic candidates. These are measured CPU parent timings under uncontrolled desktop load, excluding initial Rust compilation, NN inference, transport and rendering. They must not be presented as browser-ready or 30 m generation latency.

The parent reports 93 passing CPU tests; those tests were not rerun by this reviewer. Source-level preservation and finite sampled outputs do not replace the still-pending actual browser navigation timing or the final Natural byte comparison. No browser-performance certification is made here. Those receipts can complete the runtime record without changing this height-quality conclusion.

Minor reporting synchronization noted to the parent: the QA document still linked the preceding `final` NN directory instead of `current-final`, and the restart document still said 92 tests when the parent reported 93. These are documentation updates, not new terrain-source defects.

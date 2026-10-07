# B3 Astra review — navigation, preparation and LOD provenance

Date: 2026-10-07. Reviewer: Astra, high. Decision: **CHANGES REQUIRED** at this review snapshot.

Reviewed `terrain_server.py`, `terrain_background.py`, `terrain_final_mips.py`, `terrain_climate.py`, `terrain_jobs.py`, `terrain_disk_cache.py`, `index.html`, `terrain_lod.js`, the navigation/background/mip/LOD tests and benchmark scripts, and the real HTTP/browser reports under `E:/TerrainDiffusionRuntime/audit-implementation`. Read-only apart from this report; no GPU workload, server restart or real browser benchmark was run.

## Proven remaining defect

### P2 — An in-flight preview can outlive the last coarse promotion notification indefinitely

Locations: `index.html:79–84` (response admission), `:72` (cached tiles excluded from requests), `:103` (revision-based invalidation).

The client drops already cached `conditioning-preview` tiles when background `complete_windows` changes, but it does not tag an in-flight response with the coarse revision at admission or expire preview entries. A preview requested at revision R can arrive after the final R+1 status poll. The poll sees no tile yet; the late response is subsequently inserted into `tiles`. Later polls see the same revision and do nothing, while `refresh()` excludes `tiles.has(key)` entries. The server's two-second preview HTTP lifetime cannot refresh a tile held indefinitely in this separate JS cache.

**Reproduction performed without browser/GPU:** a Node VM executed the actual `index.html` function definitions with mocked DOM and deferred fetch. Start `fetchTile()` at revision 0, apply the exact revision-change eviction while the fetch is pending, then resolve the fetch with `X-Terrain-Stage: conditioning-preview`. Observed:

```json
{"revision":100,"stage":"conditioning-preview","cacheContainsPreview":true,"expiresAt":null}
```

The tile can remain a procedural preview after preparation is complete, until unrelated eviction, mode/world switching, or another progress change. The same design also lacks a notification when foreground demand alone makes learned coverage available: progress polling watches only the background task. Existing cached parent tiles likewise do not automatically revisit the server when a final DEM mip becomes available.

**Fix:** bind in-flight responses to their coarse revision and requeue stale previews; give previews bounded in-memory freshness or a server coverage revision independent of an actively running background task. Ensure the final-revision response race cannot suppress the retry. If automatic final-mip promotion is part of B3, provide a corresponding revalidation signal for cached approximate parents. Add a delayed-response regression that completes preparation before a preview response resolves, plus a foreground-only promotion case. Keep the currently displayed preview as fallback while fetching its replacement if desired.

## Defect corrected during review

**P2 — Macro-profile disk cache files were omitted by startup indexing.** The initially read `TerrainDiskCache._key()` accepted only `earth` and `natural`, so pre-existing `macro-a1/a2/a3` paths were ignored on restart and escaped the active-root quota until touched. The coordinator corrected this during review with the shared `WORLD_PROFILES` set and a startup-index/quota regression. The reviewed correction resolves all three macro path keys; the disk-cache test suite passes. This is distinct from the explicitly disclosed lack of a global quota across obsolete numerical profile roots.

## Validation and implementation assessment

- Re-ran **10 CPU tests** covering disk groups/quota/pins, queue cancellation/deduplication/priority, independent encoding, final-mip arithmetic/completeness, and background stop/priority. All passed. `node test_terrain_lod.cjs` also passed. The broader mocked Flask/browser suites were read, not rerun in this review.
- The LOD planner renders the target level or cropped coarser parents, excludes historical finer tiles when zooming out, uses floor-based negative indices, and clips world-edge geometry. The client bounds its resident tile budget and adjusts target LOD on wide views. A 30 m screen scale can therefore have a coarser selected source when the cache budget requires it; the UI discloses that distinction.
- Physical provenance is explicit: procedural conditioning previews, learned coarse/latent approximations, native decoder reductions, and exact block means of committed native children have separate stages. Interpolated coarse tiles are not asserted to be newly generated high-resolution terrain.
- Final DEM mips use canonical ownership of native child interiors, signed physical metres, complete-contributor checks, and a bounded fine buffer. They are exact means of those committed children, not a proof that independently generated request shapes yield identical native DEM values. The existing request-shape limitation remains relevant.
- Coarse area sampling averages decoded physical heights on the global tile lattice and has bounded chunks; it does not square an average signed-root height. Climate transport uses sea-level BIO1 plus lapse rate, and the preview lapse correction avoids applying elevation twice. Complete numerical/visual climate validation across all LOD boundaries is still not demonstrated by the supplied B3 benchmarks.
- Visible work has priority over the next background quantum. An already launched coarse window finishes and persists before stop takes effect. Separate background worlds prevent preparation from changing the active foreground seed/LRU. The shared-namespace progress/quota defect is recorded in **B2-astra-r2.md** and was assigned for correction separately; it must be resolved before accepting integrated preparation.
- Model, source and runtime identities are conservative and isolated. Cache IO leases protect active native child reads and streamed PNG responses from quota eviction. The current WebGPU probe/model routes require `TERRAIN_WEBGPU_PROBE=1` and return 404 by default; no failed browser inference backend is required or approved for production.

## What the real measurements establish

The supplied runs are historical measurements of their recorded state, not certification of the current edited manifest or final source snapshot. Runtime changes during this review invalidate any claim that they are the final post-fix benchmark.

- `final-natural-http.json`, seed `20261007092`, natural RTX 3090 path: first native tile `5.693527 s`, adjacent native tile `0.467899 s`; native cache revisits `9.393–33.415 ms`. The complete hit corpus also contains a coarse revisit at `44.17 ms`, so “all cache hits are 9–33 ms” would overstate the report. Nine total cache hits have median `23.54 ms` and nearest-rank p95 `44.17 ms`. Three misses are too few for a sustained-throughput conclusion. This measures receipt of physical fields, not browser presentation; model weights/download were not isolated as cold.
- `browser-navigation.json`: initial world preview settles in `1.48 s` and is explicitly `conditioning-preview`. The recorded native-region transition settles in `4.24 s` with decoder sources. The recorded camera location is approximately `(-181818.18, 0)` m, whereas the currently edited benchmark requests `(-103680, 80640)` m; this is another reason to keep the old report separate from the next run.
- The two short pan scenarios report frame-interval p95 near `16.8 ms`, handler p95 `0.3/0.2 ms`, no page errors, and decoder sources at their final snapshots. This supports responsive camera handlers on that desktop run. It does not satisfy a strict `<16.7 ms` frame target, `<250 ms` initial useful response, or establish continuous cold NN coverage at every instant. The pan helper sends discrete 3 km key steps roughly twice a second; its nominal 6 km/s label describes that cadence, not continuous motion at a precisely measured speed.
- The browser report does not record a time series of visible missing/approximate coverage throughout the pan. Final pending=0 and smooth frame intervals cannot establish p95 camera-to-complete-NN latency. Peak GPU memory, full-world learned preparation time, final-mip promotion in the actual browser, and coast/LOD visual approval remain unmeasured here.

**Required disposition:** fix client promotion freshness, finish the already assigned shared coarse bookkeeping correction, and retain the explicit performance/fidelity limitations. Rerun the relevant protocol/mocked-browser checks and the chosen real measurements only after the runtime source snapshot is fixed. No new procedural engine, sub-30 m generation, or failed WebGPU-port deployment is required for this block.

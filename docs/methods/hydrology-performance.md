# Hydrology delivery: fast default and optional work

This note covers `terrain_hydrology_service.py` and the option switches in
`terrain_hydrology_geometry.py`. The [integration note](hydrology-integration.md)
describes the full detailed chain; this note covers what runs by default and
what each option costs.

The current viewer's [hierarchy and lake delivery](hydrology-hierarchy-climate-lakes.md)
and [drawing controls](hydrology-rendering.md) extend this fast path. Generated
meanders and nested meanders are no longer exposed by the viewer, including
when loading older recipes. Their costs below describe the retained legacy
geometry API, not a current UI option.

## Why the default changed

The earlier delivery scheduled regional packages for every display LOD up to 3.
Each package was a fixed 61.44 km square. It needed a neural Base 576² window, a
Coarse downstream-profile closure, regional routing and the full curve
generator. A natural LOD2 screen at 120 m/px covers 230 × 130 km and needed about
15 packages. A LOD3 screen needed more than the 64-package cache, so it never
completed. CPU work was at most about 5 s per package. The live wall time was
30–160 s per package, because the neural jobs queued behind decoder tiles on the
GPU lane and the packages ran on one worker. A screen therefore took minutes.

Each source reach was also refined with valley fitting, generated meanders,
nested belts and bend-radius smoothing. That cost about 20–30 ms per reach, so
a cold LOD5/7 view took 4.6–6 s. The regional courses were different curves from
the source courses, so zooming swapped a straight overview for meanders.

## Default: source overview plus cached-Base vectors

When `detail=False` and `cosmetics=()`, which is what callers get if they pass
nothing, delivery has two layers. Neither one calls `read_detail` or
`read_coarse_points`, or creates neural work.

### Global source overview (immediate)

- Each global reach has one curve. A pinned Douglas–Peucker pass removes the D8
  staircase (tolerance: the larger of half the parent spacing and half the
  channel width). A bounded centripetal Hermite spline is then fitted through
  the result. Its endpoints and shared node tangents are exact, so confluences
  meet. There are no generated oscillations. Sampling is 1/64 of the parent
  spacing, and the tessellation tolerance is 7.5 m. `features()` only
  simplifies this stored curve per LOD.
- Source channels start at about 3,000 km² of drainage (8 source cells). A
  170 km view therefore often contains no overview river at all. Lowering that
  threshold would draw 20 km lines in the wrong valleys. Local rivers come from
  the vector packages below.
- Drainage, basins, annual discharge and the synthetic monthly budgets
  (`_properties`) are unchanged. They are stored in `world.npz`.

### Vector packages on already-generated Base (asynchronous)

`features()` subscribes the fixed 30.72 km cores of the requested view, centre
first. Only views at Base scale (LOD ≤ 3) subscribe new cores; ready packages
are kept at every LOD, with the coarse tributary filter. The server callback
`read_cached_base(seed, profile, bounds)` returns ordinary-chart Base only when
every latent window over core + 7.68 km has already been generated for the
displayed terrain. Otherwise it returns `None`:

- in canonical blend strips;
- for polar views;
- when the runtime is busy;
- when windows are missing.

A `None` core stays on the overview, reported as `unavailable`, and queues no
worker job. A later terrain revision can retry it immediately. The nonblocking
callback probes complete contributors before CPU work is admitted: repeatedly
queueing missing-window probes would keep wide views building forever. Fused
values are read directly from the pinned tensor store; the generation path is
never called. Existing stored plans load on the worker without a Base read.

The terrain job captures complete nearby Base grids before the GPU window LRU
evicts them while generating the view's edges. This reuses the job's reentrant
device lock and never generates missing contributors. Up to 64 immutable CPU
grids are retained per neural world (about 9 MiB). The asynchronous routing
worker can read these while the GPU continues producing terrain. Without this
short-lived CPU copy, only the last generated edge regions remained available.

`X-Terrain-Hydro-Base-Revision` advances when a new Base snapshot is captured.
The viewer coalesces refreshes of the vector network from this readiness signal.
Revalidating unchanged height tiles does not repeatedly recompute the network.

Each package runs the regional method (`refine_region`) with
`align_coarse=False` and source water levels (`water_profiles=None`):

- **Routing surface:** the plain 240 m Base, the decoder's own parent, over
  core + one block. There is no Coarse read and no coarse-mean correction.
- **Unchanged:** ownership labels, portals, unit links and the partition of
  parent discharge. Neighbours share portals exactly.
- **Skipped:** the downstream Coarse profile closure, parent composition, chart
  corrections, carving and local water grids. Plans go to a separate
  `vector-regions/` store; detailed water never reads it.
- **Kept:** `_regional_properties` (monthly flow, navigation), and the
  axis-only courses with optional cosmetics.

At the readiness frontier, `_bridge` extends an overview fragment cut at a
ready core edge to that package's portal for the **same parent reach on the
same edge** (valley portals may lie up to a corridor away from the parent
crossing).

The displayed vector stops at its first sea crossing in the captured Base.
The crossing is sampled at half a Base cell and interpolated to zero elevation,
including on long simplified segments. This changes neither the routing graph
nor its discharge. Coarse views also apply the source drainage-area threshold
to local tributaries, so revisiting a detailed area does not leave a rectangular
patch of small streams in an overview.

At most 3 CPU builds are in flight on the single hydrology worker, with 32
packages in memory. Only those builds count as `building`.

### Terrain and quality

Binary height delivery returns the raw terrain and no RGBA water raster; the
viewer draws the vectors once. It also accepts useful completed tiles across
subscription-epoch changes, while the client discards obsolete display tasks.
CPU PNG delivery retains source-water raster compositing (including the ≤2 m
ocean-shore adjustment). Reports carry `hydro_vector_preview = True`.

**Tradeoff:** vector packages follow the actual Base valleys (240 m), but their
water levels are the coarse source profile (approximate; used for colour and
information only). Rivers are drawn over the terrain, not cut into it. Cores
whose 7.68 km halo was not generated, and polar or seam strips, show only the
overview until the user pans there. Use `detail` for carved channels and
valley-consistent water levels.

Source ownership and shared portals still constrain routing when source and
Base relief disagree. Straight constrained sections and offsets from decoder
valley bottoms can remain. Disabling generated meanders removes the artificial
detail revealed on zoom; it does not solve this terrain/topology mismatch.

## Optional backend work (all off by default)

| Option | Effect | Implementation |
|---|---|---|
| `detail` (`terrainDetail`) | Regional Base packages at LOD ≤ 3: Coarse profiles, Base 240 m routing, parent composition and carved channels. At LOD > 3, packages that are already ready are reused. | `_regions`, `_build_region` (unchanged method) |
| `valleyFit` | Viterbi lateral fit to the valley floor | `_valley_fit` |
| `meanders` | Generated bends (angle integration), confinement sampling, far-view belt; the crossing repair includes the bend-damping steps | `_channel_offsets` and the remaining steps of `refine_reach` |
| `nestedMeanders` | Climate-gated nested belt; **only with `meanders`** | gate + belt noise |
| `smoothBanks` | Axis bend-radius limiter (≥ 1.5 W) | `_limit_axis_radius` |

When a switch is off, its helper is not called. Tests patch the helpers so that
any call raises. Source curves are cached in memory per `(reach, cosmetics)`.
Cosmetic variants of a regional package are computed on demand per option set,
and also kept in memory only. Stored worlds and plans never depend on display
options.

## Cache identities

Three chained digests, each extending the previous one:

- **world:** core, routing and balance code, plus `STORE_VERSION`. Keys `world.npz`.
- **region:** world plus the regions, profiles and sampling code. Keys `regions/*.npz`.
- **display:** region plus the geometry and service code. This is `hydro_id`.

An edit to geometry or delivery code therefore no longer forces the 30 s routing
rebuild or invalidates any neural-derived package. Bump `STORE_VERSION` when
`_properties` or the stored plan corrections change.

## Measured CPU costs

These were measured on one CPU, on a desktop shared with other work, using the
cached seed-42 world (2048 × 1024 source, 14,456 reaches) with a scalar climate
stub. No GPU and no server were involved. These are not end-to-end timings: the
neural terrain tiles themselves are not included.

| Stage | Before | Default now |
|---|---|---|
| Neural regional packages, natural LOD2 view | ~15 (30–160 s each, live) | 0 |
| Load stored world | 1.8–2.4 s | same, once |
| Seasonal properties on load | 1.0–1.3 s | skipped (stored) |
| Cold curves, LOD5 view (157 reaches) | 4.6 s | 0.17–0.25 s |
| Cold curves, LOD7 view (512 reaches) | 6.1 s | 0.67–0.83 s |
| Cached `features()` | 0.07–0.14 s | 0.01–0.12 s |
| Vertices, 512 reaches | 279k | 22k |
| Warm per-tile `physical()` (304² samples) | 64–85 ms | 43–80 ms |

The per-tile reach selection was a Python loop over every reach box (about
60 ms per tile). It is now vectorised. The rest of the per-tile time is the
ocean-shore sampling in `rasterize_water`. The server's binary-height path can
skip the per-tile raster entirely in default mode.

Cold curves for 512 reaches with each option alone: `valleyFit` and
`smoothBanks` about 1.1 s, `nestedMeanders` alone no extra cost (no effect),
`meanders` 4.5 s (279k vertices), all options together 8.1 s.

### Vector package cost

The hero core (−1872, 48) was run from its real captured Base, cropped to
core + 7.68 km (320² cells instead of 576²):

- cold build 1.0 s, of which `refine_region` is 0.78 s (316 records);
- reload from the vector store 0.09 s.

Stored detailed plans of other cores report 0.4–3.1 s of `refine_region` CPU.
Package counts depend on viewport size and land coverage; the measured 1440 ×
866 LOD2 view below used 28 ready packages.
The number of wet tributaries follows the parent lateral yield: the arid hero
core has 1–3 in both modes, while humid cores have about 300.

### Live viewer measurements

On the RTX 3090 workstation, seed 42, 1440 × 866 map viewport, render layer,
LOD2 (120 m/px), camera `(10123000, -4166300)`, all options off:

| Run | First terrain tile | Complete visible view |
|---|---:|---:|
| Cold terrain tiles | 29.6 s | 102.4 s |
| Cached terrain tiles | 5.2 s | 17.3 s |
| Final cached run, coastal clipping enabled | 5.6 s | 21.7 s |
| Cached tiles, server just restarted | 7.6 s | 47.9 s |

These include terrain generation, HTTP delivery, GPU upload and a three-second
settling period; they are not hydrology-only timings. The cold neural terrain
still takes substantial time. The completed view contained 28 vector packages
and 1,125 river records before coastal display clipping, with no browser errors.
Private captures and raw timings are in the local research directory
`hydrology-usability-2026-10-21`, runs `usable-cold-lod2-render-B` and
`usable-warm-lod2-render-B`, plus the final `final-warm-isolated-lod2-render-B`
and `final-warm-lod2-render-B` captures. The final run contains 1,065 displayed
records after clipping. Restarting the server reloads the world and rebuilds
display geometry even when terrain and regional plans are on disk.

Final cached requests for the same bounds took 285 ms at LOD2 and 39–41 ms at
LOD5/7. CUDA forward counters stayed unchanged across all three requests. All
28 visible vector packages were ready, with no pending work or errors. The
Atlas LOD5 and Render LOD7 live captures both showed rivers and had no browser
errors. The 75 focused Python tests, the WebGPU hydrology probe and the sharing
probe passed. These measurements concern one seed and view, not a performance
guarantee for arbitrary regions or hardware.

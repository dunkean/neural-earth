# Hydrology river overlay (viewer)

Module: `web/terrain_hydrology.js` (`TerrainHydrology`), wired in `web/index.html` (`hydrologyState`, `fetchTile`, `cacheIdentity`). Tests: the Node assertions at the top of `tests/browser/test_terrain_hydrology.cjs`. The physical water raster and its colours are described separately in [hydrology water appearance](hydrology-render.md). This note covers the vector overlay drawn on the 2D annotation canvas over every layer and the globe.

The overlay is a display aid. It never changes hydraulic quantities: reach width, flow and course come from `/api/hydrology/<seed>` unchanged.

## Which reaches are drawn

All filters below are display only. They read the stored reply and redraw it: no fetch, no `requestParams`/`tileParams`/`optionsIdentity` change. A routed flow, width or lake budget is never changed.

- **Threshold catchment.** `A = 250 km² · (m/px ÷ 120)^0.85 ÷ max(0.15, density)^1.5`. The density exponent follows Magna Urbis (`1.8 km² / density^1.5`). The base value and the zoom exponent are empirical, calibrated on cached captures.

  | m/px | 7.5 | 30 | 120 | 380 | 960 | 27,800 |
  |---|---|---|---|---|---|---|
  | A (km²), density 1 | 24 | 77 | 250 | 666 | 1,464 | 25,590 |

  A non-zero **Min catchment** (`minArea`, km²) replaces `A`, and the density slider is then disabled.
- **Visibility.** A reach group is visible when its mean flow is at least `Q_min = A · 300 mm/yr` *and* its `area_m2` is at least `A / 4`. A group is the set of records sharing an `id`: fragments and connectors.
  - With little runoff, a stream needs a larger catchment to reach `Q_min`.
  - Dry local weather does not mask a trunk's inherited flow.
- **Continental headwaters.** Above 1,000 m/px, in automatic catchment mode,
  source reaches of Strahler order below 3 seed the display only when their
  physical width exceeds `0.12 / max(0.15, density)^1.5` pixels. Downstream
  closure still retains a smaller-order course when needed for continuity.
  An explicit minimum catchment overrides this extra thinning.
- **Downstream closure.** Every group downstream of a visible group is also visible. A lake connector's `display_downstream_id` identifies the inflow course it merges into; otherwise the viewer follows `downstream_id`. This keeps a smaller-catchment merge owner visible even if its own display threshold fails. Hydraulic topology and budgets are unchanged. There is no random or hash thinning.
- **Measured counts.** At density 1:
  - seed 42 view B at LOD2: about 200 of 1,065 records;
  - the final user LOD3 view at 240 m/px: 327 of 3,251 records including lake connectors.
- **Drawing order.** Records are sorted by mean flow, so the largest are drawn last, on top.
- **Strahler scales stay separate.** Source order is used only for continental
  headwater thinning. Local orders are now computed on the Base network, with
  a partial flag at unit links, and are never compared with source orders.
  Width and opacity continue to use discharge.

## Stroke width and opacity

Let `d = log10(max(1, Q_month / Q_min))`, where Q_month is the annual mean in the "Annual mean" view.

- **Cartographic width:** `min(4.5, 0.7 + 0.8·d) · widthScale` px.
- **Opacity:** `min(1, 0.6 + 0.13·d)`. Width and opacity both grow monotonically with flow.
- **Physical width:** `width_m · min(1.8, (Q_month/Q_annual)^0.26)` divided by the metres per pixel. On the globe it is measured by projecting an offset point.
- **Drawn width:** `max(cartographic, physical)`. A resolved channel keeps its exact hydraulic width, and `widthScale` never inflates it.
- **Colours:** `#1a4d5c` when the river is drawn at its physical width, teal `#2d7894` when it is held at the cartographic width.
- **Diagnostic views** keep a 1.4 px floor and their own colours.
- **Line style:** canvas antialiasing with round caps and joins.
- **Screen continuity:** in the map view, a polyline is broken only where a projected step is longer than 1.5× its wrapped world length + 4 px, which happens at the longitude seam. The previous rule (break above 1.5 screens) dropped long coarse source segments at fine zoom. The globe keeps the 1.5-screen limit, and its far side projects to null.

## Flow regime

With 12 values in `monthly_flow_m3s`, the dry months are exactly those with `Q < 0.01 m³/s`. The threshold is absolute: a 700 m³/s river at 5 m³/s is still flowing.

| Dry months | Class |
|---|---|
| 0 | perennial |
| 1–5 | seasonal |
| 6 or more | ephemeral |

- Twelve equal months are a flat fallback, not evidence. The class is *unresolved*, and the reach is drawn like a perennial one.
- Without a series, a reach is ephemeral only if its annual mean is below 0.01 m³/s or the backend sends `regime: 'ephemeral'|'dry'`. Otherwise it is unresolved.
- `low_flow_months` reports the backend's relative 1 % low-flow count, separately from physical near-dry months. Older replies used `dry_months` for that count, which remains the compatibility fallback when `low_flow_months` is absent. No calendar months are derived from a count.

**Seasonal and dry streams** (`intermittent`) is off by default.

| | Annual view | Month view |
|---|---|---|
| Off | seasonal and ephemeral reaches hidden | reaches dry that month hidden |
| On | seasonal: dashed blue; ephemeral: faint dashed brown | reaches dry that month: faint dashed brown |

The brown dash is `#9b7a50`, drawn at the cartographic width.

## Lakes

`lakes[].rings` (world metres) are projected per vertex and filled even-odd, one path per lake, in `#215a6d`, with a 1 px shore line.

- **Drawing order.** Lakes are filled after all rivers, so inflow ends and connector courses beneath them are covered. A hidden lake leaves its connector visible, which keeps the river continuous.
- **Filters.**
  - `minLakeArea`: km², default 1. It hides the many 240 m Base pits.
  - `minLakeDepth`: m, default 0.
  - Each filter is compared with `area_m2` or `max_depth_m` only when that field exists.
  - Lakes smaller than about 1 px on screen are skipped.
- **Approximate coarse shorelines.** With `resolution: 'source'` at LOD ≤ 3, the shoreline is the approximate 19.5 km shape. It gets a dashed outline without a fill: unknown water must not flood the detailed terrain. The inspector labels it as an approximate shore. Base polygons stay solid.
- **Smooth shorelines** (off by default) draw quadratic curves through the edge midpoints, with no resampling.
- **Skipped rings.** A ring with an unprojectable vertex (globe far side) or a seam crossing is skipped for the frame.

## Inspector

Clicking a river shows:

- its class from mean flow: Main river ≥ 100, River ≥ 5, Stream ≥ 0.2 m³/s, otherwise Rill;
- "through a lake" for connectors;
- its regime and dry months;
- its mean flow, the selected month's flow, and the lowest month;
- its upstream area and basin;
- the backend navigation label, as an estimate (no safety claim);
- after the catchment query, the catchment area and outlet kind.

Clicking inside a drawn lake shows its area, surface level, maximum depth, and whether it is closed or drains through an outlet.

## Optional river work

The Water panel (`#hydrologySettings`) keeps all server work off by default:

- `Detailed channel carving (slow)` (`terrainDetail`): sends `hydro_detail=1`. This generates extra regional neural terrain and carves physical channels into it, as described in the routing and region notes.
- Server cosmetics: `valleyFit` ("Fit course to valley floor", a Viterbi lateral fit) and `smoothBanks` ("Limit bend radius to 1.5 widths"). They are sent as `hydro_cosmetics=<comma list>`.
- `meanders` and `nestedMeanders` were removed after user review. `normalize` drops them even when an old URL (`hydroCosmetics=meanders,…`) or share recipe sets them, so they never reach the server or a cache key.

`requestParams` builds these parameters. They are sent to `/api/hydrology` and, through the per-task snapshot `tile.hydro` (`tileParams()`), to every `/height/`, `/coarse/` and `/tiles/` request. A tile is therefore always fetched with the options of the cache key it is stored under. The defaults add no parameter and no cache-key suffix.

A changed server option:
- appends `/hydro_detail-1` or `/hydro_cosmetics-…` to `cacheIdentity()` and to the overlay's network key;
- calls `schedule()`, not `resetBackend`, so neural caches, the forced LOD and earlier option variants are kept.

Display settings in the URL, written only when they differ from the default:

| Setting | Default | URL parameter |
|---|---|---|
| `density` | 1 (range 0–2) | `hydroDensity` |
| `minArea` | 0 (automatic) | `hydroMinArea` |
| `widthScale` | 1 (range 0.5–2) | `hydroWidth` |
| `intermittent` | off | `hydroIntermittent=1` |
| `lakes` | on | `hydroLakes=0` |
| `minLakeArea` | 1 km² | `hydroLakeArea` |
| `minLakeDepth` | 0 m | `hydroLakeDepth` |
| `smoothShores` | off | `hydroShores=1` |

Malformed values fall back to the default or are clamped, never NaN. Share recipes store the whole settings object, and restoring one starts from the defaults.

**Not ported from Magna Urbis.** These have no controls because nothing implements them:
- braiding, anabranching, deltas, avulsions, estuary shapes and distributaries, which need one-to-many topology;
- `wetness`, which would falsify the climate budget;
- generated meanders, which were rejected;
- a coastal small-stream toggle, which waits for the reach `end_kind`/`outlet_kind` fields from the backend contract.

## Ready physical patches

A tile counts as *ready* (`tile.hydroReady`) only when its own pixels already show river water:

- **GPU physical tile:** uploaded with RGBA hydrology fields that contain authoritative river coverage (`TerrainHydrology.hasRiver`: class 3, coverage > 0, authority > 0.5).
- **Server-rendered image tile:** `X-Terrain-Hydro-State: ready`.
- **Never ready:**
  - tiles marked `X-Terrain-Hydro-Vector-Preview: 1`
  - native coarse tiles (`uploadCoarse` ignores hydrology)
  - fantasy styles, whose ground drawing has no water
  - the overview image

With the default fast path (`hydro_detail` off), the server sends raw heights, with no RGBA fields, and marks tiles as vector preview. No tile is ready, so the overlay draws every river. The patch logic below applies only to tiles that carry physical river water, which in practice means `terrainDetail`.

In the map view, the overlay draws the same reach geometry in two clip regions:

1. **Outside ready patches:** every river at the width and opacity above. Here the vector is the river.
2. **Inside ready patches:** only rivers whose physical width is under 2.5 px, at their cartographic width and opacity. Wider rivers are left to the raster, so no broad opaque ribbon is doubled over physical water.

Both passes stroke identical polylines, so a river crossing a patch edge is not cut. The globe has no per-tile screen rectangles: it uses pass 2 everywhere when every visible tile is ready, and pass 1 otherwise. Diagnostic views always use pass 1.

## Default network and terrain arrival

With every option off, the server is responsible for what `/api/hydrology` returns; this viewer does not check it. It returns the source overview network plus any local vector routes it can derive on the CPU from Base terrain the server has already generated, without new neural work. While those local routes are computing it may answer `building` for a short time. Where no Base exists yet, it answers `ready` with the source network alone. The viewer draws whatever reaches it receives, using the same geometry at every LOD. The panel says "Local rivers refine as terrain loads."

As Base becomes available, physical tile headers carry `X-Terrain-Hydro-Base-Revision`. The viewer exposes its latest value as `terrainRevision`. Each network request records that revision when it starts. Unchanged tile revalidations do not advance it; the reply stays associated with the captured value rather than a later revision.

A ready default network that covers the view at a fine enough LOD is reused only while its recorded revision equals the current one. When the revision changes:

- the view is refetched at most once every 500 ms after the previous ready reply;
- a burst of tile arrivals inside that window coalesces into one deferred redraw, which fetches with the latest revision;
- once the recorded revision matches, nothing is fetched or redrawn.

Pans outside the covered bounds and finer LODs still refetch immediately. With `terrainDetail` on, the revision is ignored and the existing polling applies (1.2 s retry while `building`). Resetting the network, after a world, month or option change, clears the recorded revision.

At LOD ≤ 3 the network query covers exactly the view. At coarser levels it is padded by 30% to limit refetching while panning.

## Limits

- The ready test is per tile, not per reach. If a ready tile shows some river water but lacks one particular wide reach, that reach is not drawn inside the tile. For example, this can happen under `terrainDetail` when a resolved bed is left dry as a blocked channel.
- Server-rendered image tiles cannot be scanned. They rely on the server's ready state.
- The two passes share one course only when the raster and the network use the same curve per option set. The backend states this, but it is not checked in the viewer.
- The Base revision is world-wide, so snapshots captured outside the query can trigger a refresh whose result is unchanged. The 500 ms limit bounds that cost.
- The threshold constants (250 km² at 120 m/px, exponent 0.85, 300 mm/yr) are calibrated on two humid captures and one world view. They are not derived from a cartographic law. Arid views were not measured in the viewer.
- Downstream closure follows the display continuation and `downstream_id` only within one reply. A continuation the server did not deliver is not drawn.
- The regime is only as good as the backend monthly series. Orogen half-year fields supply a synthetic rainfall phase when available; BIO-only fallbacks do not uniquely determine it. These are climatological monthly means, not individual storms.
- Lake rings crossing the seam or the globe horizon are skipped for the frame, not clipped.
- The Node assertions cover revision reuse and coalescing. They also cover:
  - defaults, legacy meander removal, the URL round trip and malformed values;
  - no refetch on display changes;
  - visibility at two zooms and downstream closure;
  - monotonic width and opacity, and the exact physical width once resolved;
  - regime styles, density 0 and a fixed minimum catchment;
  - lake order, filters, smoothing and the coarse source style;
  - inspector hit tests and regime classification.

  Live appearance needs captures.

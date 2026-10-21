# Hydrology display: hierarchy, climate regime and vector lakes

Implementation:
- `backend/terrain_hydrology_service.py`: `_prepare_courses`, `_prepare_lakes`,
  `_local_orders`, `_unit_fractions`, `_hierarchy`, `_vector_lakes`.
- `backend/terrain_hydrology_lakes.py`.

Tests: `tests/python/test_terrain_hydrology_lakes.py`.

All of this is display work on the default fast path: the global source
network, plus vector packages routed on Base that is already cached. It
changes no routing, annual discharge or budget, id, portal or stored plan,
and it needs no neural work. Monthly distributions of units that previously
used flat forcing are corrected. The code is in the display digest (`DISPLAY_FILES`). Editing it
changes `hydro_id` only; it does not change the stored world or the regional
and vector plans.

## Climate: what was wrong and what was not

Measured on the cached seed-42 world and the 28 vector packages of view B
(LOD2, 1,065 delivered records):

- **The forcing is consistent.** Source runoff around B is 78–644 mm/yr. The
  implied tributary runoff Q/area has 10–90 % quantiles of 195–570 mm/yr.
  Tributary water is the parent cell's own lateral yield, spread over its
  blocks (`_cell_density`), so it follows the source climate. Land with no
  runoff gets no tributaries. Exogenous trunks keep their routed flow across
  arid cores; one trunk carries 1,912 m³/s through a core whose median local
  runoff is 0.
- **The apparent indifference comes from display and channel initiation.**
  Tributaries start at 10 km² whatever the climate. They were drawn at a
  1.2 px floor, so 0.013 m³/s arid channels looked like 0.47 m³/s humid ones,
  although the medians differ by 35×.
- **Real defect 1: flat regimes.** About 35 % of wet tributaries (1,772 of
  5,136 in the two cached plan stores) belong to a lake unit or a coastal or
  closed fringe unit, which has no source reach. `_regional_properties` gave
  them a uniform `[1]*12` regime. They now take the synthetic monthly runoff
  weights of their unit's representative station:
  - lake unit: the lake's `xy`;
  - fringe unit: the basin root cell;
  - otherwise: the record's head.

  These weights come from the same `monthly_water_balance` and source-cell
  sampling as `_properties` (`_monthly_budget`), cached per source cell.
  Every package derives the same station for the same unit. The annual flow
  is unchanged; only its monthly distribution changes. A record's own earlier
  monthly output is no longer read back as forcing, so a re-saved detailed
  plan cannot keep a stale regime.
- **Real defect 2: inherited Strahler.** Tributaries had their unit's
  *source* Strahler order, so a 10 km² rill could read as order 5. Trunks keep
  the source order. Tributaries now get a Strahler order on the package's own
  240 m network (Kahn traversal over `downstream_id`, through local lakes).
  `strahler_partial` marks records fed through a unit link, whose order is a
  lower bound. The two scales are labelled by `strahler_scale` and are not
  comparable; rank across them by Q and area.

## Hierarchy and regime metadata

Each delivered record gains:

- `strahler_scale`;
- `end_kind` (immediate downstream end);
- `outlet_kind` (kind of the source basin: sea, closed, edge or unresolved);
- `catchment_runoff_mm_year` (1000·Q·year/area);
- `local_runoff_mm_year` (the source cell's net land yield at the record
  midpoint);
- `dry_months`, `low_flow_months` and `regime`.

`dry_months` counts months below 0.01 m³/s, the `flow_statistics` default.
`low_flow_months` counts months below 1 % of the annual mean. A large river
with a 5 m³/s month under a 700 m³/s mean has one low-flow month, no dry
month, and stays `perennial`. `regime` is `perennial`, `intermittent` or
`dry` (all 12 months dry). The monthly series is synthetic: unit forcing,
fixed annual lateral budgets and groundwater baseflow. It is not a measured
hydrograph.

This metadata is cached per stored course.

## Local Base lakes

A vector package already contains the flow-through ponds its Priority-Flood
found (`plan.local_lakes`, 4+ cells, 1+ m deep). They lie wholly inside the
core, because components touching the core edge are rejected. Before this
work they were routed but never delivered. A river entering one stopped on
its shore, and its outlet started on the other shore. In view B, 366 of the
free downstream ends and 285 of the 301 records joining nothing came from
this.

**Screening rule (not a solved lake balance).** A pond is shown when:

- it is wet (through-flow > 1e-6 m³/s, the detailed-path rule); and
- its routed annual through-flow is at least the open-water loss
  `max(E − (P − R), 0)·A`. E is Oudin PET; P and R come from the source
  balance at the pond's source cell. This is the net rule of the source lake
  balance.

Otherwise it is hidden (`dry` or `evaporative`). Local routing does **not**
subtract this loss: water still passes the pond downstream unchanged. The
screen only avoids drawing water where inflow could not keep it. It is not a
calibrated equilibrium shoreline. Ponds continuing a parent lake are not
screened. The shoreline is marching squares at the pond level on the 240 m
routing surface (`lake_shore_rings`).

## Lake connectors

For every displayed river ending at a local pond, the connector follows
`plan.receivers` from the receiver of its last cell to the pond's
`outlet_cell`. It ends on one of:
- the first vertex of the record starting there;
- the nearest point of the trunk owning that cell;
- the cell centre.

Larger inflows go first. A smaller inflow stops on the first cell owned by an
earlier course. That cell centre is an exact vertex of the earlier course,
and `display_downstream_id` names it. Elsewhere `display_downstream_id` is the
real downstream id. `downstream_id` itself stays the hydraulic one.

Rejected: closed ponds, failed walks, and steps longer than 3 cells. Courses
of vector packages are clipped at the first Base sea crossing, like rivers.
Source connectors (see the connectors note) also carry `lake_id`, `lake_shown`
and `display_downstream_id`.

Clients draw connectors in the river pass and fill lakes afterwards. A shown
lake therefore covers its connector, and also the sub-cell gap between a
river end and the marching-squares shore. A lake hidden by the client keeps
its river continuous.

## Source lakes

- **Coarse fallback.** The source lake's 19.5 km shoreline
  (`lake_shore_rings` on the source grid), cached per lake. Delivered as
  `resolution: 'source'`. It is honest at overview scale and blocky at Base
  zoom, and may cover Base mountains.
- **Base constraint inside ready cores.** On the package's halo grid (core +
  one 240 m cell of the piece-independent routing surface), a cell is wet if
  Base < lake level and it lies in the lake's 7680 m blocks or in the land
  blocks 8-adjacent to them in its basin. Kept components contain a
  lake-block cell below level, or touch the halo edge (their anchor may lie
  next door). The contour is clipped at the exact core line, snapped to 1 mm,
  and the pieces of all ready cores are unioned (shapely). Neighbouring cores
  see the same values on the shared columns, so a lake shows no core seam.
  Islands become holes. Where Base stands above the level, no water is drawn.
  Example: lake 442 (379 km², level 610 m) has only 3 % of its core's Base
  cells below level.
- **Delivery.** Up to two records with the same `id`: `base-240m` (the union)
  and `source` (the coarse shape minus every ready core). Cached per (lake,
  ready-core set) in a 512-entry LRU, simplified per LOD with the river
  tolerance. Source lake budgets (closed/open, evaporation) are the routed
  ones.
- **Incomplete shores.** If Base water meets the outer edge of the ready
  coverage inside the coarse lake, the whole lake stays on the coarse
  fallback. Adjacent ready cores are unioned before this check; their shared
  edge is not a frontier. This prevents an opaque lake from ending in a
  rectangle at a calculation boundary. The actual coarse shoreline and point
  tangencies are excluded from the check. The fine viewer draws the fallback
  as a dashed outline without flooding the detailed terrain, until the finer
  shore can be closed.

Polygons whose bounding box is smaller than one display pixel are left out.
Their connectors remain.

## Costs (CPU, cached seed-42 data, stub climate, no server)

| Item | Cost |
|---|---|
| Preparing 28 packages, including lakes and connectors | ~5–6 s (≈ 2.9 s before). About 0.15 s per package is lake rings and connectors. It runs once, on the worker. |
| Warm `features()`, LOD2 view B | 0.27–0.43 s (0.21 s before). Output: 1,434 records (369 connectors) and 1,200 local lakes. |
| Warm `features()`, LOD5 | 0.05 s, 342 lakes |
| JSON payload, LOD2 | 3.8 MB |

`_plain` now converts numeric arrays and plain scalars directly.

## Live verification, 2026-10-22

Seed `8261288290247486956`, the user's retained Orogen world, 1440 × 866 view,
LOD3 at 240 m/px, camera `(-7477474.424359638, 691298.2643638747)`, default
settings:

- 32 ready vector cores; no pending cores, river errors or lake errors.
- 3,251 delivered records including lake connectors; 327 drawn after the
  hierarchy/regime filter. 115 lakes or approximate outlines drawn.
- 48 dry and 115 evaporative local ponds suppressed.
- 933 previously flat monthly series became seasonal. The annual discharge
  difference is exactly zero for the 2,386 common physical records compared
  with the preceding capture.
- 16 records with Q > 100 m³/s remain in locally dry source cells (<10 mm/yr
  runoff): exogenous river flow is preserved.
- Full view: 20.7 s with the server and terrain caches warm; 95.5 s after a
  server restart, rebuilding display packages. Times include a 3 s settling
  period. They do not guarantee performance on other views or hardware.
- Cached HTTP network requests: 1.47 s at LOD3 (including the larger JSON
  payload), 0.88 s at LOD5, 0.68 s at LOD7 for the same bounds. CUDA forward
  counters stayed unchanged across these requests.
- Real captures: Render LOD3 on the user world, Atlas LOD5 and Render LOD7 on
  seed42; no browser errors. The coarse tests confirm river visibility.
- The final continental headwater selection draws 289 of 580 records at
  LOD7 on seed42 (575 before that extra selection), with 44 lake surfaces.
  The final repeat captured a ready river network while terrain tiles were
  still loading under concurrent GPU work; the preceding full terrain
  capture completed in 19.4 s. The headwater rule changes only client display.
- 83 focused Python tests, the real WebGPU hydrology probe and the portable
  sharing probe passed. Opus5.5 reviewed the backend/frontend blocks and the
  incomplete-shore guard.

Private captures and raw timings live outside Git in
`hydrology-hierarchy-2026-10-22` under the local runtime research directory.

## Limits

- Local lakes are DEM depressions of the 240 m neural Base. In mountains
  they are dense (often 50–190 per 30 km core) and mostly 0.2–0.6 km². Their
  depth comes from the Base, not from the 30 m display terrain. Clients
  should apply a minimum area.
- Pond screening is annual, not seasonal. Routed flow is not reduced by
  pond evaporation.
- Partial Strahler orders across unit links; source and local orders are
  not comparable.
- A large lake can keep its coarse fallback while its finer shore extends
  outside the ready coverage. Its representation can change as cached cores
  enter or leave the current view.
- The source-lake envelope rule may keep a below-level valley touching a
  halo edge that connects to nothing (bounded to one block around the lake).
- The physical lake raster (`DISPLAY_LAKES`) remains off.

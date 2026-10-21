# Global hydrology routing (core block)

Implementation: `backend/terrain_hydrology.py` (records, identity, queries) and
`backend/terrain_hydrology_routing.py` (Numba kernels). Tests:
`tests/python/test_terrain_hydrology.py`. Design context:
[multiscale hydrology](../design/multiscale-hydrology.md). Status: built and
cached by `backend/terrain_hydrology_service.py`.

## Inputs, outputs and units

`build_hydrology(height_m, *, bounds, seed=0, topology='sphere', runoff_mm_year=None, evaporation_mm_year=None, ...)`

- `height_m`: one final physical source raster, metres, row 0 at the northern
  edge. `bounds = (x0, y0, x1, y1)` in map metres, y increasing southward.
  A sphere must be a full equirectangular world (height = width / 2, even width).
- `runoff_mm_year`: net land yield (precipitation minus land
  evapotranspiration), default 300. `evaporation_mm_year`: open-water
  evaporation, default 800. Both can be scalars or full arrays; negative values
  are clamped to zero and counted in `metadata['inputs']`. The optional
  `precipitation_mm_year` gives lake surfaces their precipitation instead of
  the land yield, so a lake cell nets `P - E`. This requires `E >= P - R` on
  every cell, which holds when `R = P - AET` with `AET <= E` (the service
  passes Thornthwaite–Mather AET and Oudin PET). Where `E < P - R`, the net
  loss is clamped to zero and the lake keeps the land yield `R`.
- Sea: `ocean_mask`, or `ocean_seeds` (map points selecting below-sea
  components), or every below-`sea_level_m` component of at least
  `min_ocean_area_m2` (default min(10^12 m², 2 % of the domain)). All other
  below-sea components stay inland depressions. A world with no qualifying
  component has no sea: all drainage ends in closed basins.
- `plane_edges`: `'outlet'` (an edge cell with no lower neighbour is a declared
  world-edge outlet) or `'closed'`. Ignored on a sphere.

`HydroWorld` exposes `height_m`, `filled_m` (routing surface), `receivers`
(flat index, -1 terminal), `order` (every receiver before its donors),
`area_m2` (contributing area), `discharge_m3s` (annual mean), `basin_ids`,
`lake_ids`, `ocean`, `reaches`, `lakes`, `bounds` and `metadata`. Extras are
`nodes`, `basins` (table of arrays), `reach_ids`, `owner_reach`, `owner_lake`,
`terminal_kind`, `step_m`, `cell_area_m2` and `hierarchy` (depression forest,
per-cell `leaf` and `flood_node`). Per-cell arrays have the raster shape.
Graph arrays are int32. `area_m2`, `discharge_m3s` and `step_m` are stored as
float32; routing and budgets are computed in float64.

## Geometry

Sphere: `R = (x1 - x0) / (2 pi)`. A row between latitudes `phi_n` and `phi_s`
has the exact cell area

`A = R^2 * delta_lambda * (sin(phi_n) - sin(phi_s))`  [m²],

whose total is `4 pi R^2` (tested to 1e-12). Neighbour lengths are haversine
distances between cell centres,
`d = 2 R asin(sqrt(sin^2(dphi/2) + cos(phi1) cos(phi2) sin^2(dlambda/2)))`.
Longitude wraps. The three missing neighbours of a polar row are the cells of
the same row at `j + W/2 - 1 .. j + W/2 + 1`, across the pole, at their true
geodesic distance. The pole is therefore a shared node, never an outlet or a
wall. Plane: Cartesian spacing, `A = dx * dy`, no wrap.

## Algorithm

1. **Sea.** 8-connected, topology-aware components of `height < sea_level`.
2. **Depression forest** (adapted from Barnes, Callaghan and Wickert 2020,
   [Computing water flow through complex landscapes, part 2](https://esurf.copernicus.org/articles/8/431/2020/)).
   Land cells are visited once in ascending `(height, flat index)` order. A
   cell takes the label of its steepest strictly lower neighbour (sea counts as
   lower); a cell without one is a pit and a new leaf. Every already visited
   neighbour in another union-find set is a contact at the current height.
   Because heights arrive sorted, the first contact between two sets is their
   pass, so no edge list or sort is needed (Kruskal order). Two depressions
   merge into a parent whose children have spill level = pass height; a set
   meeting the sea set (or a declared edge) records its outlet and the leaf on
   the far side of its pass. Each cell is attached to the lowest node whose
   spill exceeds its height, which gives exact flooded area, volume and
   evaporative loss at each spill.
3. **Annual Fill–Spill–Merge equilibrium** (simplified from Barnes, Callaghan
   and Wickert 2021, [Fill–Spill–Merge](https://esurf.copernicus.org/articles/9/105/2021/)).
   Volumes are m³/year. A leaf watershed receives `V_i = runoff_i / 1000 * A_i`.
   Net open-water loss is `L_i = e_i / 1000 * A_i`, with `e_i = E_i`, or
   `max(E_i - (P_i - R_i), 0)` when precipitation is given. A node is full when
   the gross water reaching it satisfies `G_d >= sum of L_i over its flooded
   cells at spill`. Sweep events are replayed in order. A full child spills its
   surplus into its sibling at the pass. When both siblings are full, the
   parent is the active lake. A full depression that met the sea-connected
   set spills into the depression on the far side of its pass, which may be an
   earlier closed basin, and that link chain is followed to absorption or the
   sea. A closed lake floods its candidate cells in height order while
   cumulative loss stays at or below `G_d`. Its level is the midpoint of the
   next height step. `residual = G_d - evaporation` is reported (water held
   below cell resolution; zero inflow means dry). Open lakes stand at their
   spill, with `outflow = inflow - evaporation`.
4. **Lake topology.** Every final water body is one connected flooded set.
   Pass cells of submerged merges are added as zero-depth connectors, with
   topology but no evaporation. When an open lake's pass cell lies on the far
   side, the lake cell beside it is also a connector, even at an exact height
   tie. Water reaches a single sink (closed) or a single sill (open) through a
   BFS tree. On tied rasters the sill can also be the lake's internal pass. The
   lake then touches its sill from both sides, and each part drains straight
   to it. The sill link is pinned to the side the cascade credited, also when
   both sides stand at the same level. When several lakes share one sill cell,
   the lake that contains the cell sets its receiver. Significant bodies
   (default depth >= 2 m) get lake ids. Smaller ones keep their balance as
   flow-through water without a lake record.
5. **Receivers** on `filled = max(height, water level)`. Dry cells first keep
   the raw steepest neighbour that defined their depression label, unless it
   is higher on the routing surface. A dry cell never lies below the level of
   the lake that owns its label, so this also holds where a dry cell stands
   exactly at that lake's level. Routed catchments then equal the cascade's,
   and the lake-to-lake routing follows the cascade's acyclic spill order.
   Next comes steepest geodesic descent. Remaining flats use
   the combined towards-lower / away-from-higher gradient
   `mask = 2 * towards + (max_away - away)` of Barnes, Lehman and Mulla 2014,
   [An efficient assignment of drainage direction over flat surfaces](https://arxiv.org/abs/1511.04463).
   Lake cells are low edges for a flat only if it has no dry low edge, so a
   sill never drains back into its lake. The stack `order` is a BFS from
   terminals. A cycle raises an error.
6. **Accumulation.** `area_m2 = sum of upstream A_i`. The annual mean
   discharge is

   `Q = (sum of upstream V_i - sum of upstream flooded L_i) / 31,536,000 s`  [m³/s]
   (a 365-day year, the same as the monthly balance).

   The signed sum is conserved exactly: runoff = evaporation + sea + plane edge
   + closed-sink residual (`metadata['budget']`). `discharge_m3s` clamps
   negative values inside evaporating lakes to zero.
7. **Basins and reaches.** A basin is the set of cells sharing a last land
   cell: a mouth, a closed sink or an edge outlet. Basin ids are sorted by
   area. Channel cells meet `area >= channel_min_area_m2` and
   `Q >= channel_min_discharge_m3s`, and exclude lake cells. Reaches run
   between nodes: sources, confluences, lake exits, lakes, mouths, sinks,
   edges, and dry-out where a channel ceases to qualify. A reach entering a
   lake ends on the shore midpoint with `target` = the lake node and
   `downstream_id` = the lake's exit reach, so no ribbon crosses a lake. Exit
   reaches start on the shore at the lake level. Confluence vertices are
   shared exactly. `xy` is the source-grid skeleton (cell centres plus
   shore/coast points). It is longitude-unwrapped, and cross-pole links become
   two pole vertices (`crosses_pole`). `water_m` is non-increasing. Strahler
   order follows the reach graph, including links through lakes.
8. **Ownership and queries.** `owner_reach` / `owner_lake` give the network
   unit where each cell's water first joins (incremental catchment); the two
   are mutually exclusive. Coastal fringe cells that never join the network
   have neither, only a basin id. Queries never re-route anything:
   `sample(x, y)` returns the authoritative parent record for fine samples;
   `region(bounds)` returns window labels plus every receiver edge crossing
   the window, with direction, parent reach, area and discharge (antimeridian
   windows wrap); `catchment_cells`, `lake_catchment_cells`,
   `corridor_cells(reach, radius)` (dilation confined to the reach's basin),
   `upstream_cells` and `downstream_path` cover the rest.

## Deterministic identity

There are no random draws; `seed` only salts identities. Ties are broken by
height, flat index and a fixed neighbour order. `metadata['hydro_id']` is a
BLAKE2b-128 digest of the algorithm version, shape, bounds, parameters, sea
rule and the height/runoff/evaporation/sea arrays. Record `id`s are compact
deterministic indices (array values, list positions). Each record also has a
63-bit `key`, a splitmix64 chain salted by version, kind and seed over rounded
map-metre anchors (reach end points, lake sink/sill, basin root, node
position). Keys never depend on request order or display LOD. Combine a key
with `hydro_id` for cache identity.

## Scale behaviour and adaptation limits

- One canonical source grid is the authority. Finer levels inherit basins,
  ownership, receivers and portals through the queries; nothing here depends
  on a viewport or LOD. Regional 240 m refinement, cut/node artifacts, chart
  reconciliation and conditioned heights are later blocks.
- D8 on an equirectangular grid: near the poles, east–west neighbour
  distances shrink (about 30 m on the polar rows of 2048×1024, against 19.5 km
  north–south). Lengths and areas are exact, but polar routing is strongly
  anisotropic. It still needs checking on real polar-chart terrain.
- Divides are raster divides of the source grid. The skeleton `xy` exposes
  D8 steps by design; smoothing belongs to the geometry block.
- The FSM is an annual steady state on a binary merge forest. It has no
  seasonal storage, groundwater, infiltration or transmission loss, and dry
  land never loses flow. Closed levels are quantized to the source height
  steps. Equal heights create zero-volume nodes, which are harmless but
  counted. On quantized or clamped rasters, sills, lake levels and dry cells
  often share one height. Routing then relies on the tie rules of steps 4
  and 5, not on strict descent, and the order still follows the cascade.
- The default sea rule is a size threshold. Narrow straits closed at source
  resolution can turn a sea into an inland lake. Integration should supply
  the source water mask or explicit seeds.
- Channel thresholds are global constants, not climate- or slope-dependent.
- No Magna Urbis, RichDEM, pysheds or PyFlwDir code was copied. The methods
  were implemented from the cited papers, so no third-party licence applies
  to these files.

## Evidence (2026-10-21, local development machine, CPU only)

`.venv/Scripts/python.exe -m unittest tests.python.test_terrain_hydrology -v`:
6 tests OK. Every test also applies a shared invariant check: complete
acyclic stack, budget closure, cascade/routing agreement for closed lakes,
lake water leaving only at its sill, non-increasing reach water and no reach
cell inside a lake.

1. Ranked flats drain one ring per step toward the coast, away from raised
   rows. A replay of a rough 48×96 sphere gives identical arrays, `hydro_id`
   and keys. `seed` changes identities only.
2. Sphere area equals `4 pi R^2`. On a plane with spatial runoff, discharge
   and area at every basin root equal the analytic sums (rtol 1e-6).
3. An inland below-sea bowl stays inland. Arid: a closed terminal basin; an
   upper non-evaporating lake that met the sea-connected set spills into it.
   Humid: an open lake at its spill drains to the sea. A dry world has no sea.
4. A lake node with entry and exit reaches, a two-branch confluence (Strahler
   ≥ 2) and a one-component lake. Incremental ownership areas of all upstream
   units sum to the mouth reach area (1e-6). The corridor stays in its basin.
5. A river crosses the antimeridian with continuous `xy`. A seam-straddling
   region returns its wrapped columns and consistent in/out portals. A tilted
   cap routes over the pole to the sea, with no polar outlet. Plane edges are
   outlets when declared and closed otherwise.

6. Tied integer rasters. The review fixture, where a spilling lake's sill
   drains into a lake at the same level, used to raise a receiver cycle. It
   is checked at evaporation 0, 300 and 3000 mm/year. Also covered: a lake
   whose two parts meet only at its own sill; two open lakes sharing one sill;
   dry cells at a lake's level, where each open lake's outflow equals the
   discharge at its sill; and six 10 m-quantized, sea-clamped sphere worlds.

Mutation check: reverting any of three fixes makes the suite fail (sea-set
overflow sent to the sea instead of the far-side depression; no sill
connectors; no raw-label receivers). The tie test fails if any one of its
five rules is reverted: equal-level sill pin, equal-level raw-label
receiver, far-side sill connector, per-part sill roots, or the containing
lake ordered first.

Tie fuzz (2026-10-21 integration review, FINDING 1). These are research
scripts, not repository tests.
- Before: the reviewer's `fuzz2.py` failed on 10 m-rounded terrain with 11/66
  receiver cycles, 10/66 closed-lake mismatches and one reach-graph cycle.
  A 3,000-case integer-grid fuzz also found lake water leaving away from its
  sill, split lakes, and open lakes bypassed by dry cells at their level.
- After: `fuzz2.py` passes 200/200. A 20,000-case fuzz passes every case,
  with all invariants above plus open-lake sill discharge. It covers 3×3 to
  8×8 integer planes with closed edges, open edges or sea; small integer
  spheres; 20 m-quantized fields; and 50 m plateaus with a sea-level clamp.
- Continuous terrain is unchanged. Receivers, lake ids and discharge are
  bit-identical before and after the fix on the 133 continuous `fuzz2` worlds
  and on a 2048×1024 fractal sphere. That sphere's warm build takes 1.19 s
  before and 1.24 s after (timing noise). `ALGORITHM_VERSION` is now
  `hydro-core-v2`, because tied inputs route differently, so `hydro_id` and
  record keys change.

Synthetic fractal benchmark (`beta = 2.2`, 35 % land, default parameters; the
synthetic grid has far more pits than real eroded terrain):

| Grid | First build, empty JIT cache | First build, fresh process with cache | Warm build | Peak traced NumPy memory |
| --- | --- | --- | --- | --- |
| 2048×1024 sphere | 15.8 s | 2.5 s | 1.8 s | 330 MiB |
| 1024×1024 plane | — | 1.1 s | 0.59 s | 146 MiB |
| 1024×512 sphere / 512² plane | — | — | 0.52 s / 0.14 s | — |

On the 2048×1024 sphere, the warm stages take: inputs 0.27 s, depressions
0.6 s, routing 0.15 s, accumulation 0.16 s, records 0.6 s. That build has
67k depression nodes, 22.8k lakes, 19.6k reaches and 146k basins. Its budget
closes to 2e-10 m³/s and closed lakes agree with routing to 2e-15. Queries:
a 10k-cell region contract takes 2 ms, 10k samples 0.6 ms, and the first
catchment query (index build) 30 ms, then 10 µs each. Peak memory exceeds
the design's 256 MiB goal. Import time (about 0.5 s) is separate.

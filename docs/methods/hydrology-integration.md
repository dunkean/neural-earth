# Hydrology delivery, neural parents and rendering

This is the implementation note for `terrain_hydrology_service.py`,
`terrain_hydrology_sampling.py`, the server endpoints and the viewer integration.
The [literature review](../design/hydrology-literature.md) and
[design](../design/multiscale-hydrology.md) distinguish the research sources from
the choices made here. The other method notes cover [routing and lakes](hydrology-routing.md),
[water balance](hydrology-balance.md), [regional routing](hydrology-regions.md),
[canonical profiles](hydrology-profiles.md), [geometry](hydrology-geometry.md)
and [water appearance](hydrology-render.md).

## Interactive default and optional detail

The default first draws the global drainage network, then adds local routing
on the 240 m Base already generated for the displayed terrain. Fixed 30.72 km
packages reuse complete cached windows with a 7.68 km halo. They retain the
global catchments, portals and water budgets. No extra neural inference,
canonical downstream profile closure or channel carving runs in this mode.
The same ready courses feed every display LOD; zoom simplifies their vertices.
Basins, confluences, annual flow and synthetic monthly flow remain available.
Missing Base and chart blend strips keep the global overview. River levels
remain approximate source profiles; the vectors do not modify physical relief.

For binary height tiles this path returns the raw neural terrain and marks
water as vector-owned (`X-Terrain-Hydro-Vector-Preview: 1`). It sends no extra
RGBA water raster: the viewer draws the network once over the assembled view,
including native coarse tiles. CPU PNG requests retain raster compositing,
with the viewer overlay responsible for continuous, readable source rivers.

**Water → Detailed channel carving (slow)** opts into the Coarse/Base/decoder chain
described below. Valley fitting, generated meanders, nested tropical meanders
and bank smoothing are separate options, all off by default. Source routing
and water budgets are independent of these switches. See the
[performance method](hydrology-performance.md) for costs and limits, and the
[viewer rendering method](hydrology-rendering.md) for minimum screen widths.
When lakes are hidden, [display connectors](hydrology-connectors.md) retain
verified wet receiver paths through open lakes; closed and dry ends remain ends.

## Authority and data flow

1. Sample the **selected final physical source** and its climate, after the
   selected Orogen/native erosion stage. The default hydrology raster is
   2048 × 1024 on a sphere (1024² on a plane). `TERRAIN_HYDRO_WIDTH` permits a
   smaller even raster for development. The source's 100,000-point Orogen
   simulation is not confused with either this raster or the neural lattice.
2. Compute periodic annual land runoff and open-water potential evaporation.
   The global graph supplies sea/lake ownership, basins, reaches, disjoint
   incremental catchments, areas and annual discharge. Nothing in a display
   request re-routes an isolated viewport or tile.
3. When base detail is needed, acquire canonical **7680 m coarse** and
   **240 m base** arrays through the existing neural GPU scheduler. First,
   sample the requested downstream closure in the Coarse field to establish
   immutable descending water profiles. Nodes and portals use these profiles;
   an independently chosen regional ground height never becomes a water level.
   Regional
   packages occupy a uniform lattice of 8 × 8 coarse cells (61.44 km square),
   with the method's fixed input halo. These are cache/work packages; drainage
   ownership still comes from the global graph.
4. The 30 m decoder supplies residual terrain detail. The same regional graph
   supplies every finer display LOD; sub-30 m detail does not solve another
   watershed. River curves are generated once in physical station coordinates,
   then simplified or averaged for display.

All positions and geometry calculations use float64 map metres, y south.
The renderer subtracts a local origin before using float32. Physical fields
use cell-centred, globally aligned samples and include the existing tile halo.

## A geographic parent field, independent of the camera

The ordinary NN and the rotated polar NN are **different learned realizations**.
Switching one on because the camera changed cannot define a drainage authority.
`canonical_field` instead chooses their contribution from geographic position:

```
w_pole = smoothstep(54 degrees, 64 degrees, abs(latitude))
w_seam = smoothstep(0.45, 0.49, abs(x - world_center_x) / world_width)
w = max(w_pole, w_seam)
H_parent = (1 - w) H_ordinary + w H_rotated
```

Blend **physical metres**, not signed-square-root latent values. The coarse and
base fields use this same rule. The source-window reader groups points into
bounded 128-cell blocks; grouping and query order do not change the samples.

Both learned charts have their own longitude seam. In particular, the rotated
chart's seam crosses part of the geographic equator. A camera-independent
choice of chart alone therefore does not fix longitude continuity.
`_periodic_chart` defines an explicit **two-sided canonical strip** for each
chart, with width 8 × 7680 m (capped at 5% of a small world). Let `s` be distance
to that seam and `b` the strip width. Mirror the two chart ends, sample at
`q(s) = s² (2b - s) / b²`, and cross-fade with
`a(s) = 0.5 (1 - smoothstep(0, b, s))`. Both sides have the same average at
the seam, zero first derivative there, and the original value/derivative at
the outside edge. No NN samples outside the finite chart are assumed valid.
This strip is a procedural construction; the raw NN is not claimed periodic.

Canonical **parent** geography is shared by map and globe. The decoder's small
scale residual still comes from the requested NN chart. This is not a new
globally reprojected decoder model or a claim that its residuals are identical
in the two charts.

## Keep coarse structure without drawing terraces

The regional method adds base residual detail to coarse height using a fixed
local mean-correction stencil. It reports the remaining coarse-block error;
channel constraints can also alter means. Exact block-mean utilities remain
available for data processing, but an exact piecewise-constant remainder can
draw artificial steps in the terrain.

For ordinary 30/60/120/240 m display grids covering complete 240 m blocks,
the delivery bridge computes the deficit between the **uncarved regional base**
and the decoder's block means. Two Richardson steps give a smooth bilinear
correction, without the final constant block remainder. Its three-base-cell
support fits the 24-sample native halo. The remaining mean error is measured.
Fine river carving follows this composition; the 240 m channel trench is not
upsampled into 30 m terraces.
Parent-mean diagnostics measure this reconciled terrain before water
conditioning. The subsequent narrow channel cuts intentionally change those
means locally; they do not replace the parent's broad landforms.

Rotated display grids, partial ordinary grids and experimental sub-30 m grids
(negative LOD) cannot form 240 m block means. Their decoder belongs to **one raw
chart**, so its baseline is that chart's own Base, not the canonical blend:

```
z_out = z_decoder + bilinear(H_routing - Base_chart)     chart in {ordinary, polar}
```

`Base_chart` is the raw chart's Base sampled at the package's geographic
240 m centres (polar: through `to_chart`), without chart blending or the
periodic seam strip. Only where one chart is the sole canonical contributor
does it equal the canonical Base. In the 54-64 degree and longitude blend
strips the two learned realizations can differ substantially; correcting a
rotated decoder against the canonical Base would then shift the parent by that
difference even for a decoder with no residual.

The correction covers **core + one 240 m cell**. Bilinear samples within half
a cell of a core edge then use the true neighbouring centres instead of
clamping to the core's last centre, which would draw a step between packages.
`H_routing` there is the region's uncarved routing terrain halo
(`routing_halo_m` and `routing_halo_bounds`, cropped from the piece-independent
`_Region.Hh`), so both sides of an edge see the same values. `RegionPlan.bounds`
and the routing/flow arrays still cover only the core. The canonical-Base
correction uses the same extent.

`read_hydrology_detail(..., charts=(name,))` reads this reference through the
same GPU lane, once per package and chart, and only over that core + 240 m
extent: unlike routing, it needs no coarse smoothing halo. It is acquired lazily, the first
time a fallback tile of that chart needs the package; the package reports
pending for that chart meanwhile. The correction is stored in the package as
`chart_correction_m[chart]`; raw NN caches are unchanged. Aligned ordinary
grids do not use it. A `read_detail` callback without a `charts` parameter
keeps the canonical-Base approximation, and the tile report says so.

Approximations: the decoder's local mean is assumed to follow bilinear Base,
which a learned decoder only approximately does; that residual mean passes
through. For the rotated chart, `Base_chart` is resampled on the geographic
lattice and re-interpolated, a sub-240 m smoothing of the reference. No exact
block means are promised on a rotated or partial block.
The graph, portals, lake levels and meander seed remain unchanged.

## Progressive delivery and bounded conditioning

The global build and regional packages run on one background worker. Parent
NN acquisition uses the existing GPU lane and device locks. Ready world and
regional packages persist as NumPy arrays plus JSON in the runtime hydrology
directory, never as pickle and never in Git. Loading reconstructs `HydroWorld`,
including its ownership/query methods. Cache identity includes source/world
identity, implementation hashes and hydrology raster width.

Packages use a standard NPZ container with UTF-8 JSON metadata and fast
DEFLATE compression. Older Unicode-metadata packages remain readable. This
keeps persistence simple and avoids spending the regional worker's time on
maximum compression of thousands of small reach arrays.

World and geometry caches are bounded. Broad requests select rivers by flow
and resolvable width. Regional pieces have fixed bounds, shared portal inputs
and retry transient queue/cancellation failures. A source-network preview
remains available while finer packages are pending.

Before the first base package is ready, fine display tiles deliver still
water and the viewer draws the connected source vectors once as an overlay.
`X-Terrain-Hydro-Vector-Preview` distinguishes this state from physical river
coverage. This avoids rasterizing all long source curves separately in every
fine tile while the worker is trying to prepare their replacements.

Source-network ribbons carry coverage; they cannot excavate the decoder to a
height measured on a 19.5 km source cell. Only ready regional channels resolve
a bed. The preferred profile incision is 30 m. On an incompatible barrier,
the bounded conflict allowance is **60 m of valley lowering plus the hydraulic
bed depth**, with the latter capped at 30 m (total at most 90 m). The profile,
regional route and decoder share this allowance, so a finer stage cannot
silently mask a cut accepted by its parent. Bank raising
is capped at **2 m**, and
still-water bed correction at **2 m** above its level. Higher neural islands
remain dry. These are explicit procedural limits, not erosion predictions.
If bounded excavation leaves a resolved river bed at or above its water level,
the physical water raster keeps that sample dry and reports
`hydro_unresolved_river_cells`; it does not conceal the conflict with blue paint.

### Decoder relief is not charged to the parent allowance

The shared allowance `P = 60 m + min(D, 30 m)` is charged to the **canonical
routing parent `S`** at the display stage, as the profile and regional route
charge it: `S` is the plan's routing halo, sampled bilinearly. It is not charged
to decoder relief above `S`. Within ready cores the display cut budget is:

```
B = max(P, clip(S - U, 0, P) + R),    R = DECODER_RESIDUAL_CUT_M = 30 m
```

Here `U` is the channel's own upper envelope (refined course, `surface_envelopes`),
so `B = P` outside cores and in coarse views. A resolved channel may cut what its
parent must lose to reach the envelope (at most `P`), plus at most `R` of
remaining relief: decoder residual above `S` and any parent excess beyond `P`.
Total excavation is therefore at most `P + R` (≤ 120 m).

- **Why.** Previously `P` was measured from the composed decoder surface `S + r`.
  On a deep but accepted parent cut, every positive decoder bump became a dry
  "spot" inside the channel, a physically blocked bed rather than an island.
- **Scope.** The budget acts only where the envelope is finite (the channel and
  its bank slope). Terrain outside the finite river support is bit-identical,
  and the parent composition and the 7680 m cells of coarse views are unchanged.
- **Monotone.** `B >= P`, so no cell can become blocked that was not blocked before.
- **Real tile.** On tile -1871,53 (key c45dd…), blocked cells inside the bankfull
  half-width fell from 507 to 8. All 528 axis samples are wet, max cut 99.9 m,
  and 240 m blocks without river support are unchanged.
- **Reporting.** `hydro_channel.blocked_bed_cells` counts the cells that remain
  dry inside the bankfull half-width; these are the genuine conflicts.
  `parent_excess_cells` counts cells whose parent is more than `P` above the
  envelope.
- **Rejected alternatives.** Composing the decoder against `conditioned_base_m`
  smears the 240 m D8 trench across land, by up to 27.8 m outside the river
  support on this tile. Charging `P` again below the conditioned Base would
  silently remove arbitrarily high decoder hills.

Limits:

- A decoder hill more than `R` above its parent, or a refined course whose parent
  ground stands more than `P + R` above the bed, stays dry and reported. On this
  tile, 8 cells need 2–8 m more.
- `R` is calibrated on one real tile, where the decoder residual reached a p99 of
  18.8 m and a maximum of 29.5 m.
- `hydro_unresolved_river_cells` (the geometry definition) also counts
  antialiased edge pixels at or beyond `W/2` whose envelope prescribes bank
  ground (284 on this tile). These are correct dry bank, not blockers.
Regional quality also records water above routing ground and conservative link
fill. Descending profiles alone cannot guarantee plausible banks in a learned
terrain that contradicts its source network.
The admissible-envelope projection is idempotent for envelopes derived from
the raw inputs. Raw neural caches remain independent of these display copies.

The water raster carries level in metres, coverage, class (dry/ocean/lake/river)
and validity. CPU PNGs and WebGPU consume the same contract. The GPU interpolates
inland levels weighted by wet coverage, so a dry neighbour's zero level does
not pull a lake edge down to sea level. Dry endorheic ground below sea level
shades as land. Ocean appearance retains the existing material. On the polar
display, rivers are reprojected and still-water samples are transformed back
to the **same geographic shore field**, rather than re-routing a rotated DEM.

### River support across package edges

Package cores own parent terrain corrections and local still-water fields.
They do not clip river geometry: a river generated in one core can meander
into the next. Delivery combines the neighbouring river rasters by coverage,
then distance, in sorted package order. Ready cores replace the source-network
preview, and regional river/bank influence also replaces it beyond those cores;
uncovered areas keep their preview. Local lakes remain restricted to their
owner core.

Regional geometry uses an optional `support_bounds`: the core grown by
**3840 map metres**. The complete channel and belt displacement, including
axis smoothing and valley fitting, is smoothly limited relative to the original
parent station before arc lengths are recomputed. Displacements below half
the available room are unchanged; larger ones approach the envelope through
a C1 saturation. Shared pins remain exact. No curve is cut at its core edge.
The geometry API's default `support_bounds=None` retains its existing behavior.
Bounded regional belts have effective width at most four bankfull widths.

Acquisition uses the same support guarantee before rendering, even if the tile
lies wholly inside the neighbouring core. With maximum parent bankfull width
`W`, target map cell size `c`, and seasonal width multiplier at most 1.8, the
conservative representation width is `1.8 * max(W, 4 * min(W, c / 2.75))`.
The second term covers only narrow rivers whose belt representation can be
active at this cell size. Three such widths plus one target cell cover the
raster's wet and bank kernel. Add 3840 map metres in y, and divide the physical
kernel by the minimum longitude cosine over that expanded latitude interval
before adding 3840 in x. Thus high-latitude neighbours are acquired without
assuming that physical and map east-west metres agree. Small interior tiles
need only their own package; tiles near an edge subscribe to relevant
neighbours through the existing bounded cache and progressive worker.

### Coarse views reuse ready courses

Once a package is ready, its course is the river for every LOD. Views coarser
than the 240 m Base (`lod > 3`: network queries and raster tiles) reuse cached
ready packages that touch the query. They never schedule Base, neural or chart
reference work, so a broad zoom-out does not start regional generation. Without
a ready package, the source preview remains, as on a cold start. Vector
records are the same regional curves, simplified with the LOD tolerance. Outside
ready cores, the source remainder is kept.

These coarse views split parent composition from water geometry:

- **Parent terrain is not composed.** The display keeps its own cells, such as
  the 7680 m Coarse stage, and no 240 m routing parent or decoder correction is
  applied to them.
- **Regional water is coverage and level only.** `resolved` is zero, so a coarse
  pixel is neither excavated nor bank-filled to a 240 m-scale bed. This is the
  same rule as for source-network ribbons.
- **Local still-water fields are not composed.**
- **Density matches the source network.** A coarse view shows a regional record
  only at or above the source channel threshold (`channel_min_discharge_m3s`).
  Rasters also apply the source network's LOD width cut. Small new tributaries
  therefore appear only at Base LOD and finer.

At Base LOD and finer (`lod <= 3`), every piece a request requires is accounted
for: it is either delivered ready, building, or counted pending. A fine tile or
network is never reported ready with required pieces omitted. Builds stay
bounded at four in flight, closest pieces first.

- **Support size.** One tile's support needs at most about 30 pieces. That case
  is a polar LOD3 tile near 79° latitude.
- **Cache.** The package cache holds 64 packages (`REGION_CACHE`). A request
  needing more pieces than that is served up to the cache size. It delivers any
  other cached ready package and stays pending until a narrower view fits.
- **Memory.** A ready package holds about 7 MB, with 19 MB measured at most.
- **Eviction.** It removes least recently used packages that no request has
  touched within the 30 s supersede window. Concurrent tiles of one broad view
  may overflow the soft size, but never the hard ceiling of 96 packages
  (`REGION_CEILING`, about 0.6 GB at the median package size and 1.8 GB at the
  measured maximum). The 64-package soft size covers roughly a 1440-pixel
  Base view with its query margin. This prevents
  tiles from evicting each other's packages while they are still being
  displayed.

Tile headers expose `X-Terrain-Hydro-Blocked-Bed` and
`X-Terrain-Hydro-Max-Cut-M` for live inspection of channel conflicts and
conditioning depth.

Limits:

- A coarse view shows regional courses only for the packages still in that
  cache. An evicted package falls back to the preview until a fine view reloads
  it from disk. There is no global regional atlas.
- Reloading a package from disk still re-runs display geometry, because
  geometry is not persisted.

## Seasonal properties and watershed queries

The annual graph remains authoritative. Annual per-reach lateral yield is
`Q_reach - sum(Q_upstream)`. Positive yield follows the local monthly runoff
fractions; annual open-water loss follows potential-evaporation fractions.
An availability-constrained allocation moves any unpayable monthly loss to
months with water while retaining the annual loss. It is a seasonal proxy,
**not** a lake-storage or backwater model. Invisible gaps in the rendered
channel threshold retain their hydraulic downstream continuation. The monthly
mass residual and error against annual discharge are recorded.

New regional tributaries use their incremental unit's monthly runoff fractions,
scaled to their annual flow. Trunk inflows retain the parent's routed regime;
unit links carry their own incremental unit's regime on both sides of the cut.
The existing monthly solver routes these inputs and new lateral yield through
the regional graph together, so a confluence retains its upstream seasonality.
Negative increments use the same bounded annual-loss allocation. Monthly mass
residuals and annual discharge errors are recorded for each region. This keeps
distant upstream snowmelt from assigning the same low-flow season to every new
local stream. Forcing samples wrap in longitude on spherical worlds.

Display geometry skips reaches with zero annual flow: their drainage and
budgets remain in the routing arrays, but there is no water to tessellate or
include in bend-crossing repairs. Intermittent streams with a positive annual
mean keep their geometry even when the selected month is dry.

After this filter, unbranched constant-flow trunk records are joined into
display courses. Wet confluences, portals, lake terminals and flow changes
remain cuts; routing and monthly budgets keep the original records. This
removes artificial bend pins at dry tributary junctions. Each course exposes
its original `members`. Climate is sampled at the joined course's midpoint.
Joined courses may explore up to eight bankfull widths, limited by physical
room inside the package support; the original 2.5-width minimum corridor is
retained. Terrain confinement, slope and climate still decide whether any
nested bends can form. This does not force meanders in confined valleys.
Regional curve tessellation is capped at 7.5 m error (one quarter of a native
30 m decoder cell); narrow streams keep their smaller width-based tolerance.
Changing this cap changes vertices, not the underlying physical curve.

Their navigation estimate
combines their own width/depth/current/slope and possible rapid/fall proxy
with the inherited downstream route, month by month. A small tributary cannot
inherit ship access merely because the parent river is wide. Direct sea
terminals provide access; closed pits/lakes do not, and a parent lake uses its
actual outlet reach's route. Local lake chains retain their exit type (sea,
pit, parent lake or regional continuation), so passing through a lake cannot
invent sea access. The lowest of
twelve synthetic monthly means is the displayed low flow; it is not Q95 or a
measured drought return level. Vessels and thresholds are explicit worldbuilding
heuristics. No tides, locks or ice navigation are modelled.

`GET /api/hydrology/<seed>` supplies the network (`bounds`, `lod`, `month`,
`world_profile`, `world_identity`). `reach_id` requests that reach's full
upstream watershed; `basin_id` requests a terminal basin. Areas use spherical
cell areas. Boundaries are source-grid contours, not invented fine divides.
The Water inspector uses this query; the basin diagnostic displays the
selected watershed. For a local tributary the inspector explicitly labels this
as its parent river's watershed, rather than claiming a fine tributary divide.
Hydrology settings are included in portable view recipes.

Changing the selected month or water visibility preserves an explicitly
forced display LOD. Repainting newly ready water tiles does not issue a new
camera epoch: the viewer updates its subscription only when the actual view
changes, with a 15-second keep-alive. This avoids cancelling the same physical
requests repeatedly during progressive delivery.

## Persistence cost

The existing atomic NPZ cache stores large numerical arrays normally and
inlines arrays up to 2048 bytes into typed JSON metadata (dtype and shape
included). Loading still disables pickle and accepts the earlier array-member
format. This avoids tens of thousands of ZIP members for short reach records.
On the real 14,456-reach source world, the member count fell from 61,407 to 88,
loading from 9.47 to 2.27 seconds and saving to 3.46 seconds. The new file was
31.6 MB instead of 42.6 MB; receivers and sampled geometry were identical.

## Supported scope

This implementation targets the Python server and its map/globe viewer.
Lake rendering is temporarily disabled (`DISPLAY_LAKES = False`), following
the user's close-zoom feedback. Global and local lake polygons and lake water
fields are omitted; drainage nodes, levels and water budgets remain active.
Existing shoreline helpers are retained for later work and are not accepted
as realistic fine shorelines by this delivery.
It does not add hydrology execution to the standalone WASM inference engine.
Global geometry remains the authority in the final longitude-edge pieces
whose zero-aligned neural lattice would cross the finite world extent, and
above approximately 84.6 degrees latitude. It supplies connected periodic
water geometry there; these guard regions do not yet add the same 240 m
tributary density as ordinary regional packages.

Cutoffs, oxbow lakes, sediment bars, braided/distributary networks, flood-wave
routing and monthly lake-level dynamics are not represented by the current
single-receiver graph. The optional nested tropical bends reproduce detail
inside a broader belt, not a full anabranching Amazon floodplain simulation.

## Verification

Focused Python cases cover routing/lakes, seasonal balance, parent composition,
regional ownership/portals, geometry, water fields and persistence. The browser
probe checks finer requests inside cached bounds, real WebGPU colours, missing
authority and dry land below sea level. The existing Render3D WebGPU probe is
also retained. Live evidence and the integration review are recorded during
development; measured performance is distinguished from algorithmic guarantees.

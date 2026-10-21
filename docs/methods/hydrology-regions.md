# Regional 240 m hydrology refinement

Module: `backend/terrain_hydrology_regions.py` (`REGIONS_VERSION = 'hydro-regions-v2'`). CPU only: NumPy, SciPy (`ndimage`, `sparse.csgraph.dijkstra`) and three small Numba kernels. No server, Torch or CUDA import. Tests: `tests/python/test_terrain_hydrology_regions.py`. Design context: [multiscale hydrology](../design/multiscale-hydrology.md), section 2 and the user's coarse/base/decoder preservation refinement.

The module refines the global `HydroWorld` from `terrain_hydrology.build_hydrology` over one fixed piece of terrain. The global network stays authoritative. The piece adds valley-following trunk geometry, 240 m tributaries and small flow-through lakes, together with a conditioned surface. It never solves a new macro drainage graph on the base DEM.

## Root integration API

```python
from terrain_hydrology_regions import refine_region, piece_at, piece_windows, compose_decoder, refit_reach_30m

core = piece_at(x, y, piece_blocks=16)            # uniform aligned grid; the service uses 8 blocks
base_bounds, coarse_bounds = piece_windows(core)  # core + 5 coarse blocks (smoothing_steps=3)
plan = refine_region(world, coarse_7680m, coarse_bounds, base_240m, base_bounds, core,
                     seed=seed, water_profiles=profiles)
```

- `world`: the `HydroWorld` returned by the core. The module reads `sample()`/`cell_at()`, `owner_reach`, `owner_lake`, `lake_ids`, `reach_ids`, `filled_m`, `ocean`, `basin_ids`, `basins['kind']`, `discharge_m3s`, `area_m2`, `receivers`, `nodes`, `metadata['parameters']['sea_level_m']` and the reach/lake records (`cells`, `xy`, `source`, `target`, `enters_lake`, `leaves_lake`, `downstream_id`, `exit_reach`, `level_m`, `basin_id`, `closed`).
- `coarse_7680m` and `base_240m` must be canonical neural fields on the global lattices: bounds are multiples of the cell size from origin 0, and spacing is exactly 7680 m and 240 m. Larger windows are allowed; the result depends only on the fixed neighbourhood of the core. The fields must come from the final physical generation settings, never from whichever preview is cached.
- `core` must be a multiple of 7680 m. Neighbouring pieces must come from one uniform grid with this core's size and alignment (`piece_at`, or the service's fixed 8-block grid) and share all options: a cut is a whole shared piece edge, and unit links infer the grid from the core.
- `water_profiles` implements `level(reach_id, station)`, where station is the fractional original reach vertex index. Build the [canonical profiles](hydrology-profiles.md) before refinement, including `required_profile_reaches(world, core)` and the visible trunks. That pure helper includes own/absorber references of entire connected units touching the core. Without the optional object, descending original source water supplies all levels. Local neural ground never becomes independent water authority.
- `RegionPlan` (all arrays cover the core, row 0 north, `origin` = global 240 m (row, col) of cell [0, 0]):
  - `reaches`: trunk segments (`kind='trunk'`, `parent_id` = global reach id) and tributaries (`kind='tributary'`). Each record has the shared contract fields `id, source, target, basin_id, xy (N, 2), water_m (N), flow_m3s, area_m2, downstream_id`. Records also carry `cells` (local flat), `pins`, `flow_profile_m3s` (trunks), `start_portal`/`end_portal` and portal tangents (trunks), and `start_link`/`end_link` (tributaries crossing a cut).
  - `local_lakes`: owned flow-through lakes (`id, level_m, closed=False, flow_through=True, area_m2, cells, max_depth_m, on_trunk, entry_reaches, exit_reach, outlet_cell`) and parent-lake shore records (`id='P{lake}@...'`, `parent_lake_id`, the parent `level_m` and `closed`). A local lake filled to a parent lake's level from that lake also carries `parent_lake_id`.
  - `portals`: the shared trunk crossing contracts. `links`: the shared unit-link records touching this core (section 4).
  - Fields: `routing_height_m` (H), `conditioned_base_m` (min(H, U)), `upper_m` (U), `water_m` (NaN where dry), `unit_key`, `basin_ids`, `receivers` (local flat; -1 terminal, -2 exit portal or unit link), `area_m2`, `flow_m3s`, `drain_class` (1 network, 4 local pit; 3 is no longer produced).
  - `metadata`: `region_id` (hash of version, `hydro_id`, core, seed, options and the composed inputs), `quality` and `timings_s`.
- `compose_decoder(plan, decoder_30m, decoder_bounds, rect_bounds)` composes the 30 m decoder detail onto the conditioned base and re-applies U. Its default stays one Richardson step; the decoder input must cover `rect` plus two base cells, inside the plan core.
- `refit_reach_30m(reach, decoder_30m, decoder_bounds)` moves the interior vertices of one plan reach by at most 105 m toward the 30 m valley floor. Ids, links, pins and levels are unchanged. Detailed bends remain `terrain_hydrology_geometry.refine_reach`, which can consume these records (`pins`, tangents, `corridor_m`).

## Method

### 1. Aligned residual composition (coarse -> base -> decoder)

The coarse and base fields can disagree by hundreds of metres at macro scale. The routing height therefore keeps the parent's block means and adds only the child's aligned residual:

`H = F + up(C)`, with `D = P - blockmean(F)` and `C_{k+1} = C_k + omega (D - S C_k)`, `C_0 = D`, three steps by default, `omega = 1.6`.

Here `P` is the parent field (7680 m), `F` is the child (240 m, factor 32; or 30 m, factor 8), and `up` is cell-centred bilinear upsampling. `S` is the exact block mean of `up`, a separable `[1/8, 3/4, 1/8]` stencil for even factors. The block-mean error `(S C - D)` shrinks by a factor of at most 0.6 per step for every frequency; linear deficits are reproduced exactly, and if `P` equals the child's own block means, `H = F`. The result over a block rectangle depends only on data within `1 + steps` blocks of it. The core, its one-block halo and every portal or link strip therefore get bit-identical values in every window that contains them.

There is no final piecewise-constant remainder. An exact block residual makes parent means exact but adds a 240 m (or 7.68 km) terrace for any rough deficit. The remaining mean error is smooth, measured and reported (`block_mean_error_max_m`). For a white 30 m rms deficit it is 2.9 m max, 1.3 m rms; the Nyquist checkerboard worst case is 4.9 m. One step gave 15 m and 13.5 m. Height jumps at block edges equal those inside blocks (no terraces), see Evidence. This is an independent helper rather than `terrain_hydrology_geometry.preserve_parent`, whose iterated correction depends on the window extent (stated in its own docstring). The idea is comparable to Tobler's pycnophylactic interpolation (1979). It is a constrained downscaling, not learned terrain.

### 2. Inherited ownership and parent water

Every 7680 m block takes the parent record of the source cell that contains its centre: basin, ocean, lake and incremental unit. A unit is a reach (`owner_reach`), a lake (lake cells and `owner_lake`) or, for cells without either, their basin, which covers coastal fringes and small closed basins. Base cells inherit their block's labels. The labels are a pure function of global position, aligned at 0. They are an explicit approximation: a 19.5 km parent staircase snapped to 7680 m blocks, not the exact fine watershed. The disagreement is measured, not hidden (see quality).

Parent water seeds the local Priority-Flood at its water level, never at the neural ground height: parent-lake cells at the lake's `level_m`, parent-sea cells at `min(H, sea_level_m)`. Trunk node levels come from canonical water profiles, including fixed lake entry/exit levels and original terminal mouth water; anchors select positions only.

**Parent shore envelope.** Every land block that is 8-adjacent to a block of parent lake L, in L's basin, belongs to L's shore envelope. Its cells below L's level that connect (8-connected) to L's blocks inside the 3 x 3 block window centred on that block join L at L's level. They are lake seeds (`unit_key` = the lake unit) and are returned as a `P{L}@...` water record at the parent level. The rule is a pure function of that window (H covers the core plus one block), so both sides of a cut agree. Farther land that floods to exactly L's level from L becomes an owned local lake tagged `parent_lake_id`: still the parent level, never a separate pond hundreds of metres below the parent lake. If such a component touches the core edge it stays dry and is reported as `parent_lake_conflict_area_m2`.

### 3. Shared portals and node anchors

Each parent polyline gets a fixed sub-centimetre offset, identical for all pieces, which breaks exact vertex/corner coincidences with the cut lattice. It is then intersected exactly with the core edges using half-open rules. Inside spans run between consecutive events: the source node, entries, exits and the target node. A reach leaving the piece therefore continues through a portal; it never ends at an artificial mouth.

A portal is a pure function of the parent reach, the cut line and a fixed strip: one block on each side of the cut, and along it over the crossing +- corridor, clipped to the shared edge. Candidate stations are the cell pairs adjacent to the cut. Pairs with the reach basin on both sides rank first (tier 0). Pairs with it on one side rank next (tier 1), which happens when a parent diagonal passes a source-cell corner. A downstream parent-sea side is allowed for mouth reaches. Among the best tier, the chosen station minimises the detrended level plus a deviation penalty:

`cost(t) = z(t) - trend(t) + w_p ((t - t_P) / corridor)^2`, with `z = (H_left + H_right)/2`, `trend` the least-squares line of `z` over the admissible candidates, and `w_p = 20 m`.

Detrending makes the portal select a valley, not the downslope end of the corridor. The record freezes the canonical profile level at the original crossing station, a tangent (the lowest strip points 8 cells either side of the cut), and parent flow and area interpolated at that station. Local `z` remains diagnostic `ground_m`. Both neighbours compute identical records. No first-writer-wins mutation is possible, because nothing is written outside the call.

Nodes inside the core (sources, confluences, lake entries/exits, mouths, sinks) get an anchor. The anchor is the cell, within one block (widened once to two), that minimises `H - plane + w_p (d / r)^2`, where `plane` is the local least-squares plane. Mouth anchors prefer cells next to the parent sea; lake anchors prefer cells next to their (shore-extended) lake. The anchor belongs to the piece that contains the node.

### 4. Unit fragments across cuts

A parent unit can cross a cut while its own reach (or lake, or coast) lies only in the neighbouring piece. Version 1 drained such fragments to the core edge and never drew them; their tributaries stopped at the cut. Version 2 resolves every fragment with a small block-level contract that is a pure function of the parent world and the piece grid:

- The unit block graph is 8-connected, like the fine flood: diagonal blocks meet through a corner cell pair. A **fragment** is a connected set of one unit's blocks inside one piece. The unit's component is found by a breadth-first search over block centres from the core's blocks, capped at 20 000 blocks; a larger unit is reported as stranded, which is still pure.
- A fragment is **own** if it holds its unit's sink: blocks crossed by the reach polyline, the lake's blocks, or a coastal block 8-adjacent to parent sea in the same piece. Parent sea across a cut is a terminal node of the graph: it is the parent outlet itself.
- Otherwise it is **absorbed** if the polyline of a parent reach that delivers its water to the unit crosses it, and its canonical first-entry water in this piece is at most the unit's own canonical source level plus `strong_incision_m` (60 m by default). A lake or sea unit uses its fixed level instead of a source. Such a reach lies upstream of the unit's reach in the parent network, enters the unit's lake, or, for a coastal fringe, belongs to the same basin. Among eligible candidates, parent flow then id decides; a rejected high candidate does not hide a valid lower one. The 60 m allowance preserves modest upstream confluences while rejecting remote high trunks. It bounds their additional lift, not absolute fill over the local terrain. The fragment's flood key becomes the selected reach, so it drains to that trunk upstream of the parent junction. Downstream topology is unchanged, and trunk flows stay pinned to parent values, so nothing is counted twice.
- Otherwise it is **linked**. `delta` is the graph distance to the nearest sink block. The fragment leaves through the cross-piece neighbour pair with the smallest `(delta, corner before edge, parent filled height, block indices)`. A minimum-`delta` block always has a neighbour with `delta - 1` outside its fragment, so links strictly descend in `delta`: there are no cycles, and every chain ends at a sink.
- **Flow.** A link carries `Q = fsum(block yields of the fragment) + fsum(inflowing links)`, with the same per-block yield `density(cell) * fine block area` that the fine yields use, and the matching area. `math.fsum` makes it order-independent and bit-identical in both pieces. Fragments are evaluated upstream first, in descending minimum `delta`.
- **Site.** On a block edge, the site is the lowest `z = (H_up + H_down)/2` of the 32 cell pairs, avoiding cells within 2.5 cells of a trunk portal on that edge. On a corner, the site is the corner cell pair. `ground_m` records this ground estimate. The record freezes the site, shared water level, both cells and `downstream_tributary`: the id that the downstream piece gives the tributary starting at the inlet cell, set when the link area reaches `min_channel_area_m2`. Links into parent sea are `downstream_sea` and terminal.
- **Water authority.** Each fragment has exactly one exit, so the links of a unit form trees, each rooted at one sink fragment. Every link of a tree takes that sink's level: the canonical water at its trunk's first entry into the sink's piece (source station 0 if it starts there), the fixed lake level for a lake unit, or sea level. Every rerouted span in that piece starts at this station or later, so the inlet ceiling never lies below the trunk it joins. The unit-wide maximum is only a fallback for a sink without a valid entry station. Version 2 used that maximum for every link, which put links up to 60 m above a lower own sink. High-absorber rejection happens before sink-distance BFS, so links reorient towards the retained sinks and still strictly descend in graph distance. A component without an own or eligible absorber sink emits no invented links; `fragment_blocks.unresolved_authority_blocks` and `stranded` expose it for the existing flagged local fallback. `rejected_high_absorber_candidates` records the selection heuristic. This uses actual selected absorbers rather than remote source levels of all candidates. It is pure in the world and fixed piece grid. It is still conservative: it can impound water above the local valley; `unit_link_fill_max_m` measures the largest link level above its site ground. It does not prove physical feasibility.

In the fine routing, a linked fragment of this core gets its own flood key and a `LINK_SEED` exit at its link cell (at the shared level). The core's inlet cells inject `Q` and the area into the local accumulation, and they start a tributary. That tributary begins exactly on the link point. The upstream tributary ends on the same point, with `downstream_id` set to the downstream piece's tributary. Remnants that are still unreached, for example a fragment cut off by a foreign trunk, join an adjacent connected area of the same basin; this is flagged as `orphan_area_m2`. Anything left after that becomes a local pit. No outlet is invented: every exit is the parent's path to the unit's sink.

After flooding, inlet ceilings propagate downstream along the receiver DAG. Water on those paths is capped at the inlet level, retaining monotonicity and exact joins rather than clipping a final polyline and changing its endpoint. This can require additional incision: `inlet_ceiling_cells`, `inlet_ceiling_lowering_max_m`, and `inlet_ceiling_incision_max_m` expose it. A contradictory higher fixed downstream authority raises an explicit error. If a rerouted trunk or lake already occupies an inlet cell, a short connector preserves the shared link point/level and its downstream id.

### 5. Valley-following trunks

Trunk spans are processed in a pure order: parent flow descending, then id. Each is a Dijkstra least-cost path (SciPy) on the 8-connected 240 m grid between its endpoints. The path is restricted to the reach basin label, non-sea and non-parent-lake cells, cells not already used by an earlier trunk, and the corridor around the clipped parent span. The corridor half-width defaults to 0.75 source cell, about 14.6 km on the 2048 x 1024 atlas. The edge cost is

`L_ij (c_i + c_j)/2 + w_c max(0, H_j - H_i)`, with `c = 1 + (H - minfilter_r(H)) / h_r + w_d min(d / corridor, 4)^2`.

The defaults are `r = 8` cells, `h_r = 10 m`, `w_d = 2` and `w_c = 24`. These terms penalise height above the local valley floor (an incision proxy), deviation from the parent and climbing. Curvature is not penalised at 240 m; the geometry module smooths it at 30 m.

The fallback sequence is fixed: corridor, doubled corridor, the whole basin label, then foreign cells allowed within one block of the endpoints (a label-staircase artefact), then a raster of the parent line. Each stage is recorded.

The whole water profile comes from canonical Coarse authority. Fractional original station advances monotonically from the span's first to last station in proportion to rerouted path length. Thus source/target nodes, neighboring portals and interior vertices belong to one descending profile; no regional ground bump can introduce a rise. The retained `max_incision_m` regional option no longer re-solves local trunk water: set the incision preference when building canonical profiles.

**Fine trunk water.** The canonical level at each path cell is then bounded by the 240 m path ground (`_bounded_water`): `w = min(c, max(max(E, reverse_cummax(G - strong_incision_m)), cummax_min(G + bank_fill_m)))`, where `cummax_min` is the running minimum from upstream and `E` is the pinned end level. Both endpoints keep their shared portal/node levels exactly. Each envelope descends, so `w` descends. Where the canonical line already lies within 2 m of the ground upstream, it is unchanged. Otherwise the water may cut up to 60 m rather than float over a lower fine valley. `fine_lowering_m` reports the largest change.

Incision above `strong_incision_m = 60 m`, the parent-line stage, or more than 2 foreign-label cells sets `strong_correction`. `profile_rise_m` measures the largest actual vertex-to-vertex rise (zero for a valid canonical profile). Required incision and fill are reported even if they exceed physical conditioning limits; canonical monotonicity does not justify unlimited terrain carving or floating water.

### 6. Tributaries, lakes and water accounting

A keyed Priority-Flood (Numba binary heap with FIFO ties, as in Barnes, Lehman & Mulla 2014) starts from trunk cells at their water level, parent-lake cells (including shore extensions) and parent-sea cells at their water level, and unit-link exits. A cell is only reached from a neighbour with the same flood key: its unit, its absorbing trunk or its linked fragment. Parent-sea seeds may also enter fringe units of sea-terminating basins. Equal levels are processed FIFO, so flats drain toward their entry by breadth-first rank. No tributary can cross a parent label except through the absorption and link rules above, so it cannot steal a basin or invent an outlet.

Yield partitions the parent lateral yield; it never multiplies accumulated parent discharge:

`y_f = a_f * max(0, Q_c - sum_{d -> c} Q_d) / A_assigned(c)`.

Here `a_f` is the fine cell's physical area (`cos(latitude)` of its row on the sphere), and `A_assigned(c)` is the exact sum of the fine-cell areas of the blocks whose centres lie in source cell `c`. Summed over all pieces, the fine yields therefore reproduce each source cell's non-negative lateral yield. Tributary flow is the downstream sum of `y_f` and of injected link flow.

Trunk flows keep the parent values at both span ends, i.e. the portal interpolation or the node values. In between, they distribute the parent increment by the cumulative fine lateral inflow:

`Q(s) = Q_in + (Q_out - Q_in) F(s)`.

Nothing is added twice, and the flow on either side of a portal is identical. Tributaries start at `min_channel_area_m2 = 10 km^2` and at link inlets. They split at heads, fine confluences and lake exits. A tributary joining a trunk splits the parent trunk into segments, which keep `parent_id`. Trunk segment ids are `"{parent}#{start}"`, where `start` is `src`, `p{portal id}` or a global 240 m cell. Downstream links to neighbouring pieces use the same ids. Tributary ids are `"{unit}~g{row}_{col}"`, from the head's global cell. Lakes are flow-through, with zero extra evaporation, as the design specifies for v1.

**Local lakes** (`_local_lakes`) are filled components at least 1 m deep, labelled **per flood key**: the keyed Priority-Flood gives one level only inside a key. Version 2 merged neighbouring keys at the maximum level, which drew ponds up to 11.5 m above lower ground across a unit-block edge. A component touching the core edge is never drawn. The rim surface is seed water or bare ground (a neighbouring flood is not assumed drawn).
- A component within `min_lake_depth_m` of its rim is drawn at its routed level.
- Otherwise it is impounded by the unit-block ownership (a key wall) or by a lower river, and is counted in `key_wall_*`. Drawn channels breach its sill down to the lowest water its routed outlet can reach (`floor`, below). It is drawn only as the sub-ponds below that level, each checked against its own rim; a failure counts in `undrawn_ponds`.
- An inlet-ceiling path through a pond fixes the pond at that lower level.
- Continuations of a parent lake keep the parent level.
- `routed_level_m` keeps the routing level. Routing, receivers and flows never change.

A drawn pond must still drain downhill. `_route_local` recomputes the channel water with all drawn lakes fixed, drops any re-levelled pond whose outlet floor exceeds its level, and repeats until nothing changes. Contained lakes can never be dropped: their floor never exceeds their fill.

**Drawn channel water** (`_channel_water`, Numba) is the receiver-DAG form of the trunk rule: `w = min(filled, max(low, up))`.
- `up` is the minimum over all upstream cells of ground + `bank_fill_m`, pushed downstream in reverse flood order.
- `low` is the maximum over the downstream path of ground − `strong_incision_m` and of the fixed water it reaches (seeds, drawn lakes).
- Fixed cells keep their level. Every bound descends along receivers, so the tributary records descend and their confluence ends agree by construction.
- Water filled over an undrawn flood therefore runs in its own channel and cuts the sill (at most 60 m) instead of floating over the hidden pond. The tributary records, beds and `water_m` use this drawn water. The routing `filled` stays the flood surface used for accounting.

### 7. Conditioned surface and the shared edge strip

Channel cells get a cell-mean bed of `w - d(Q) min(1, W(Q)/240 m)`, using the practical bankfull priors `W = 7.2 Q^0.5`, `d = 0.27 Q^0.39`. Their provenance and annual-mean discharge limitation are documented in the [geometry method](hydrology-geometry.md). The carve is bounded and local:

`D = max(0, min(H - bed_n - s d, cut_n (1 - d / r)))`, `U = H - D` where `D > 0` (else +inf), `conditioned = min(H, U)`.

Here `bed_n` and `cut_n = H - bed` belong to the nearest channel cell (Euclidean distance transform), `s = bank_slope = 0.1` per metre and `r = bank_radius_cells = 3`. Version 1 used `U = bed + s d` everywhere. That lowered every hillside steeper than 10 % into a cone around each channel, without bound. The carve now never exceeds the channel's own cut, and it vanishes within 3 cells.

**Edge contract.** The own-channel carve is multiplied by a fade from 0, at cells adjacent to any core edge, to 1 at `edge_strip_cells = 4` cells. Shared records, the trunk portals and unit links of this core's edges, carve the same function around their point on the cut. That carve uses their shared level and flow, and it fades only towards the perpendicular piece lines (corner links towards both). Both pieces see the same records, and H is pure, so at a cut both sides apply one function of position: the conditioning cannot create a step. The price is a weaker carve of own channels within 4 cells of an edge; channel water left below the conditioned ground is reported as `hidden_water_max_m`. Root's display conditioning is a separate, later projection: its budget is the shared 60 m conflict valley cut (`strong_incision_m`, which the service also passes to `build_profiles`) plus `min(bankfull_depth_m, 30 m)` for the hydraulic bed, at most 90 m. Shared portal/link carves reach only the bankfull depth below their level, so a strongly incised portal can leave Base-level hidden water near a cut that the display projection must remove. Unresolved conflicts remain dry and are reported; this contract only covers `conditioned_base_m`/`upper_m`. `compose_decoder` still needs its window inside one plan.

The projection is idempotent; raw inputs are never modified.

## Determinism and LOD

- There are no random draws; `seed` only enters `region_id`. Ties break by lowest index or FIFO insertion; candidate orders are fixed.
- Portals, links, anchors, H, labels and shore extensions are pure functions of the parent world, global coordinates, the piece grid and fixed neighbourhoods. Two neighbours agree whatever the request order or window size. The tests check bit equality.
- The topology ends at 240 m. The 30 m helpers never change ids, links or levels. Screen LOD should select and tessellate these records; nothing here depends on the viewport.

## Quality record

- **Composition:** `block_mean_error_max_m`/`_rms_m` (H vs coarse) and `coarse_base_offset_rms_m` (H vs raw base).
- **Ownership:**
  - `cross_basin_links` counts routed cells draining into another basin. It is 0 by construction, except where an explicitly counted foreign trunk cell carries a different label.
  - `divide_disagreement` is the fraction of label-boundary land cells whose steepest H descent points into another unit.
  - `fragment_blocks` counts own, absorbed, linked and stranded blocks.
  - `orphan_area_m2` is the area that needed the same-basin remnant fallback.
- **Links:** `links_out`, `links_in`, `link_exit_area_m2`, `link_inflow_m3s` and `link_flow_mismatch_m3s` (declared minus fine exit flow).
- **Water:**
  - `parent_lake_extension_area_m2` and `parent_lake_conflict_area_m2`.
  - `local_pits` and `local_pit_area_m2`.
  - `yield_destination_m3s` (trunk, parent lakes, sea, link exits, pits) and `yield_residual_m3s` (yield + link inflow − destinations).
- **Surface:** `max_conditioning_m` and `hidden_water_max_m`.
- **Containment:**
  - `lateral_containment_max_m` and `uncontained_cells` are computed per drawn wet cell as the lowest water in its 3 x 3 neighbourhood (itself included) minus the lowest adjacent dry conditioned ground. `uncontained_cells` counts cells above `bank_fill_m`.
  - `key_wall_impoundments`, `key_wall_area_m2` and `key_wall_excess_max_m` describe the re-levelled ponds; `undrawn_ponds` counts the dropped ones.
  - `channel_water_above_ground_max_m` and `unit_link_fill_max_m` are also reported.
- **Per span:** stage, cells, foreign cells, incision/fill, deviation, portal rise, anchor fallback, parent increment vs fine lateral, and `strong_correction`.

## Initial evidence snapshot (2026-10-21, local CPU, synthetic fields only)

- `.venv/Scripts/python.exe -m unittest tests.python.test_terrain_hydrology_regions`: 7 tests pass (1.9 s warm). The hydrology set, `discover -s tests/python -p "test_terrain_hydrology*.py"`, is 34 tests, including core and service, and passes.
  - **Shifted base valley.** Base macro tilt opposite to the coarse field (RMS offset > 1000 m), with a narrow valley 2.4 km off the parent/coarse valley. The trunk follows the base valley within half a cell, stays within the corridor and keeps the coarse eastward fall. Block-mean error is under 0.7 m on this deficit, with no strong correction. A rough white 30 m deficit gives 2.9 m max error, and block-edge jumps equal interior jumps.
  - **Neighbouring pieces.** Shared portals are identical in `xy`, level, tangent, flow, area and cells. The upstream exit links to the downstream piece's first segment at the same point, level and flow. The same core from a window 2 blocks larger gives an identical package and `region_id`.
  - **Unit crossing a cut.** Source column 19 straddles the cut at block 48. Its tributary reaches 19 and 83 lie wholly in the eastern piece. In the western piece, the old 470 km^2 of undrawn orphan area became 0: two blocks are absorbed by reach 50 (upstream of 51), and six blocks link east. The link records are bit-identical in both pieces and in a different window/order. The western tributaries end on the link point with `downstream_id` = the eastern inlet tributary, which starts on the same point and carries at least the link flow (1.169 m^3/s, 177 km^2). Link mismatch is about 1e-15 m^3/s, and there is no conditioning step across the cut.
  - **Parent lake.** The neural floor lies up to about 250 m below the parent lake level (1052 m), inside a bowl wider than the parent cells. v1 produced an owned local lake at 932 m beside the parent lake. Now 695 km^2 of shore extension and two farther components carry the parent level, and no owned pond lies below it.
  - **Flow partition and decoder.** As in v1: yield residual about 1e-14 m^3/s, trunk flows equal parent portal flows, the closed bowl becomes a linked flow-through lake, and the 30 m refit keeps ids and endpoints.
- **Ad hoc tiling** (outside the test suite; script saved with the research report). Each case is a 3 x 2 grid of 8-block pieces: three random plane worlds, and a 128 x 256 sphere world with coasts. Results: 0 mismatched link fields over 85 links, 0 dangling tributary ids, 0 m seam step, 0 stranded or orphan area, 0 local pits, and residuals ≤ 1.3e-13. v1 left 9,038 km^2 of undrawn edge-drained orphan area in the same sphere tiling. In two cases an inlet tributary had no upstream tributary ending at its exit: the upstream side was a flow-through lake impounded right behind the exit, whose `exit_reach` is the inlet tributary.
- **Speed:** 0.12 to 0.15 s per 8-block (256^2) piece, and 0.7 s per 16-block (512^2) piece; trunk Dijkstra dominates.

## Limitations

- The initial evidence above used synthetic arrays. Subsequent seed42 cached canonical neural plans exposed lake-condensation cycles and remote high-absorber water lifts (451.757 m at a link; 111.313 m at a channel). The cycle regression and the bounded absorber policy address those identified defects.
- Version 3 (2026-10-21) fixed a canonical dam: 5860/5807 sat flat at 109.87 m, up to 37 m above both banks. Measured on a CPU rebuild of the actual seed42 piece -1872_48_-1864_56 from the original NN Coarse/Base:
  - 34 annual-positive channel reaches and 537 wet cells.
  - 22 cells stand more than 2 m above a lower adjacent bank (maximum 6.2 m); trunk 5860 peaks at 1.8 m and 5807 at 4.4 m.
  - No record rises and no confluence mismatches.
  - The largest trunk incision is 52.7 m.
  - Annual-zero drainage routes still float in the routing surface; they are omitted from displayed geometry by the service.
  - Pinned canonical portal levels can still stand a few metres above a fine valley next to the portal. Passing monotonicity and conservation tests does not establish plausible river heights, divide placement or incision on real terrain.
- Labels are block-snapped parent units. Absorption and links follow parent lineage at block level, not the unknown fine divide. A linked fragment's water leaves through one block edge, even when the fine relief would split it. A trunk passing through a fragment can cut it; the cut-off part then joins a neighbouring unit of the same basin (flagged).
- A link's flow is the exact parent partition. The upstream fine accumulation can differ when foreign trunk cells or remnants sit inside the fragment (`link_flow_mismatch_m3s`; about 1e-15 m^3/s in the tests). When several sub-threshold flows converge exactly at an exit, or a lake sits right behind it, the downstream tributary starts at the cut without an upstream tributary record.
- Edge strips fade own-channel carving within 4 cells (960 m) of a cut. An incised trunk there can leave water up to `hidden_water_max_m` below the conditioned ground (3.1 m in one random case). The 30 m `compose_decoder` cannot cross piece edges yet.
- The parent shore envelope reaches one block beyond the parent lake blocks. A bowl that extends farther and touches a core edge stays dry (`parent_lake_conflict_area_m2`), which can show as a level step at the envelope. Corner links route through a single corner cell pair.
- Antimeridian-crossing pieces raise an error, because the canonical seam strip is not implemented. The unit graph does not wrap longitude, cross-pole donors are ignored in the lateral yield, and polar-chart blending belongs to the root sampler.
- Fine coastlines are not reconciled: sea seeds are parent-ocean blocks, not a coastal strip matched to the learned shoreline.
- Tributary flow uses non-negative parent lateral yield (losing parent cells contribute 0), and fine climate weights are not yet applied. Trunk flows still carry the parent's losses. Local lakes are flow-through with one level per per-key component. Ponds impounded across key walls are drawn at their breach level, not drained across the wall: their routing still uses the in-key outlet. Closed local lakes and their evaporation are deliberately not modelled.

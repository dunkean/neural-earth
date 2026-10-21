# Hydrology display connectors

Implementation: `backend/terrain_hydrology_connectors.py` (`display_connectors`).
Tests: `tests/python/test_terrain_hydrology_connectors.py`.

## Why connectors exist

Lakes are routed but not displayed (`DISPLAY_LAKES = False`). A reach that
enters a lake ends on the shore, and the lake's exit reach starts on the
opposite shore. In the routing graph the two are already linked:
`downstream_id` of the entering reach is the exit reach, and the water's
cell-by-cell path through the lake is already in `world.receivers`. Only the drawn course across
the lake is missing, so a river that continues looks cut.

A similar but rare case is a channel that fades below the channel threshold
and reaches another channel through unchannelled cells. `HydrologyService._properties`
records that continuation as `hydraulic_downstream_id`.

A connector draws that existing path. It is neither a lake polygon nor new flow:

- no routing step is added;
- no flow, budget, id, basin, lake record or `downstream_id` is changed;
- no terrain is carved.

## Algorithm

It runs once per world, on the stored graph. It uses NumPy and an internal graph walk.

1. **Candidates.** A reach qualifies if its annual flow is above 1e-9 m³/s and either:
   - `end == 'lake'` and it has a `downstream_id`, or
   - it has a `hydraulic_downstream_id`.
2. **Ordering.** Candidates are sorted by decreasing flow, with ties broken by reach id.
3. **Walk.** From the receiver of the reach's last channel cell, follow `world.receivers` while the
   cell has no channel owner (`reach_ids < 0`). The walk stops when it reaches a channel cell or a
   cell already owned by a larger connector. It is capped at 512 cells, and a revisited cell aborts it.
4. **Checks.** A link is accepted only if:
   - **Lake.** Every path cell belongs to the entered lake. The lake is open (`closed` false and
     `outflow_m3s > 0`), and the walk stops on the exit reach's **head cell**. Inside a lake the
     routed cell discharge is not the flow measure (the lake balance is), so it is not tested per cell.
   - **Hydraulic.** Every path cell has routed discharge `> 0`, and the walk stops on a cell owned
     by `hydraulic_downstream_id`. If water is lost on the way, the end is genuinely dry and gets no connector.
   - **Merge.** The owning connector ends on the same downstream reach.
   - **Geometry.** All vertices are finite, and no raw step is longer than two cell diagonals.
     This rejects pole crossings and corrupt data.

   A failed check leaves the gap visible rather than inventing a course.

   No link is made for closed lakes, dry ends, `terminal` ends or unverified paths.
5. **Ownership: one trunk per lake.** The cells of an accepted path belong to that connector.
   - A smaller inflow stops at the first owned cell.
   - Its end is projected onto the owner's *smoothed* polyline and inserted there as a shared
     vertex. The owner lists that vertex index in `pins`.
   - The branch therefore ends exactly on a vertex of the trunk, and the trunk is drawn once.
   - Targets point either to a real reach (`target_kind='reach'`) or to an earlier, larger connector
     (`'connector'`). Ownership only flows to earlier links, so the connector graph has no cycles.
6. **Geometry.**
   - The raw polyline is the parent's last vertex, the path cell centres, then the join point:
     - lake: the exit reach's first vertex, its shore link point;
     - hydraulic: the target cell centre, which is a vertex of the target reach;
     - merge: the inserted trunk vertex.
   - On a sphere, longitudes continue from the parent's last vertex. Each step takes the shortest
     representative, `x[i] -= round((x[i]-x[i-1])/P)·P`, where P is the world width. The join is
     moved by whole periods only, so it stays the same place on the planet.
   - The polyline is smoothed by two rounds of Chaikin corner cutting with pinned ends. Each segment
     (P, Q) becomes the two points (3P+Q)/4 and (P+3Q)/4, and the first and last points are kept.
   - `water_m` falls linearly with length from the parent's end level to the join level, clamped
     non-increasing.

Output: `{parent reach id: link}`. Each link has:

- `xy`: vertices in the parent's longitude frame;
- `water_m`;
- `kind`: `'lake'` or `'hydraulic'`;
- `target` and `target_kind`;
- `reach_target`: the real downstream reach;
- `cells`: the receiver path;
- `pins`: branch joins on this link.

## Limits

- **Coarse geometry.** The course follows the 19.5 km source grid, softened by corner cutting. It
  is display geometry: it has no hydraulic width of its own and does no carving. Across very large
  lakes (up to 129 cells on the seed-42 world) it is a long line through what is physically open water.
- **Width.** A connector shows the entering reach's attributes. A lake trunk therefore has the
  largest inflow's width, which can be narrower than the exit reach.
- **Do not simplify across joins.** Deliver connector vertices unsimplified, or keep `pins`, so that
  a branch end stays on the trunk at every LOD. The paths are short.
- **Source network only.** In `terrainDetail`, ready regional packages carry their own routed
  courses. Connectors belong to the source overview, so the caller clips them to outside regional
  cores, as for other source pieces. No lake physics on the Base grid is attempted.
- **Not drawn.** Closed (endorheic) lakes and dry fade-outs keep a visible end. That end is real.

## Measured (seed 42 cached world, CPU)

- 1,912 of 1,912 open-lake entries are linked. Of those, 670 are trunks and 1,242 merge into a larger inflow.
- 2 of 13 hydraulic links are wet. The other 11 lose all discharge on the way, so they get no link.
- Path length: median 2 cells, p90 11, maximum 129.
- Size: 37k vertices in total.
- Build time: about 0.44 s per world.

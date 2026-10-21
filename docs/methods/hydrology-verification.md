# Hydrology integration evidence

Development log, 2026-10-21. This separates measured results from the
[design](../design/multiscale-hydrology.md) and the individual method notes.
The river stack is integrated and the reference scene has been checked live. Lake rendering was
deferred at the user's request: polygons and water fields are disabled while
drainage and budgets retain their lake nodes. These results do not claim that
every neural terrain can fit the prescribed network.

## Focused checks

- `python -m unittest discover -s tests/python -p 'test_terrain_hydrology*.py'`:
  **62 tests passed in 21.037 s** after final integration. Cases cover
  source routing/lakes, monthly conservation, shared regional ownership and
  levels, decoder parent composition, geometry, rendering and persistence.
- `node tests/browser/test_terrain_hydrology.cjs`: actual WebGPU shader output
  for a 10 m-deep lake was RGB `(12,44,71)`. Missing water authority matched
  the legacy frame; dry land below zero differed from ocean. Minified 25%
  wet stripes gave `(10,140,83)`, matching area-weighted water coverage.
- `node tests/browser/test_terrain_share.cjs`: portable recipes replayed
  across fresh/conflicting preferences, viewport sizes and globe cameras.
  The existing Render3D WebGPU probe also passed with the water bindings.

The share regression also checks that changing the flow month preserves a
forced LOD and that an unchanged viewport does not churn camera epochs.
The latest replay was 1570 portable characters versus 3283 in encoded JSON.
The delivered applied-view link was also reopened on the live server: seed,
world identity, camera, annual flow setting and forced LOD 0 matched, with
20/20 physical 30 m tiles, no finer-history fallback and zero displayed lakes.

These checks use the local Python 3.12 environment and RTX 3090. They are
focused regressions, not a new CI or comprehensive hardware campaign.

## Geometry cost

The same 172-reach regional package, including a deliberately shifted Base
field, was profiled before and after replacing repeated spatial-bin crossing
tests with a bounding-box sweep:

| CPU operation, under cProfile | Before | After |
| --- | ---: | ---: |
| Complete `refine_network` | 92.053 s | 21.697 s |
| Crossing detection, 10 calls | 77.235 s | 10.764 s |

Both runs made 282 reach refinements and 110 repairs. The change preserved
sampling, seeds, physical profiles and repair decisions. The previously slow
service composition case plus all eight geometry cases then passed in
25.826 s. These numbers describe that package, not a bound for every network.

## Real neural case

The working case uses seed 42, GPU Orogen profile
`orogen--g46261a320a644f43189ea849`, map centre
`(-14365234.375, 400390.625)` metres, 24 m per screen pixel, forced LOD 0.
The source network has 14,456 reaches. Coarse, Base and decoded terrain are
the actual 7680 m, 240 m and 30 m neural fields, rather than synthetic ramps.

This case exposed defects that the initial synthetic cases missed:

- Independently chosen portal ground levels made one trunk rise 33.234 m.
  Shared immutable Coarse water profiles replaced those local water levels.
- Local-lake condensation could select an apparent outlet whose receiver
  returned into the same lake. Choosing by receiver topological rank removed
  the cycles in all six archived regional drainage graphs.
- An upstream absorber at about 595 m raised a different unit whose own
  river was about 113 m. Absorber eligibility now has an explicit bounded
  level allowance before the unit's link graph is built.
- Per-core river masking discarded 308 of 364 wet pixels in a reproduced
  tile. Neighbour acquisition and river union now retain those overruns;
  terrain and local lake ownership remain core-based.

The first regenerated regional package had no ascending reach, no orphan
area, zero declared link-flow mismatch, and a maximum Coarse block-mean error
of 0.473 m. It still reported 31.06 m maximum link fill and 23.45 m hidden
water before final decoder conditioning. These diagnostics measure remaining
terrain conflict; a monotone graph alone does not validate realistic banks.

Further physical review replaced a trench-relative wall test, removed dry
tributary display pins, and joined unbranched constant-flow trunk records.
The current 66.9 km course is a continuous valley-following spline; real
confinement is retained instead of forcing free meanders. On that package,
compacted geometry takes 0.04-0.09 s, compared with 0.47 s for the separate wet
records. The historical 280-reach package's Numba kernels reduced 267.6 s to
about 22.5 s with bit-identical output; neither figure is a universal bound.

The delivered real 30 m frame uses numerical profile `d380d0b5b981860c`, world
`895aae6c02ba28598e00511dfdee7de3eeff059a23e740e6d6edf7598689bfe2`, hydro
`abf7c6078277286711b036316aa9ace04a0ed84c48db8c74e28d507de8f3c147`.
All 20 visible tiles were physical-water-ready, with no fine-history fallback.
The same camera at Base LOD 3 displayed two physical 240 m tiles. The main
course now has two navigation-equivalent records, with 70 and 84 fine vertices.
Their 25/29 vertices at LOD 3 and 15/19 at LOD 4 are exact members of the fine
course (maximum vertex distance 0 m).

All four water
channels matched exactly over 31 horizontal and vertical tile overlaps, with no class
mismatch and no lake pixels. Resolved wet river samples had positive depth
(minimum 0.0552 m in that sampling). Terrain halo heights differed by up to
2.0476 m; this is not claimed as exact terrain overlap. The October low-flow
display reduced fully wet river samples from 34,277 to 27,469 (19.9%) and kept
forced LOD 0. The inspector fetched 2,604,192 km² of upstream catchment,
annual flow 10,178 m³/s and minimum monthly mean 4,487 m³/s. It correctly kept
the year-round navigation label conditional on downstream access.

The overview previously dropped ready regional courses at LOD 4 and returned
the source preview, displaced by up to 11.97 km on this case. It now reuses
ready courses without scheduling new NN work. The final LOD 4 query retained
both refined mainstem records, with 2 ready regions, 0 pending and 0 errors.
Coarse terrain itself stays uncarved and retains its own large cells.

Every one of the 1,576 mainstem-axis samples, at approximately 15 m intervals
inside the final viewport, was wet; the closest bed sample remained 2.026 m
below its profile water. The actual tile -1871,53 returned 8 residual side
conflicts and maximum cut 99.871 m through its new diagnostic headers.

The ordinary/polar correction now uses each decoder's actual raw chart Base,
with a true producer halo instead of clamping a synthetic edge. Focused tests
cover that mixed-chart contract; a planetary visual campaign has not been run.
Transient obsolete-view HTTP 409 responses occur during progressive stages.
The final live run reported no page exception or WebGPU shader error.

## Final Opus corrections

Opus 5.5 in high effort was used through its CLI for coding and independent
reviews. Real-package checks found and corrected axis curvature tighter than
1.5 channel widths, navigation metadata mixed by display compaction, and
decoder relief incorrectly charged against the parent's valley allowance.
The axis limiter preserves portal positions and tangents; navigation changes
now retain record cuts, and merged length/slope cover the complete course.

On the captured real tile -1871,53, the final parent-attributed budget reduced
blocked cells inside the channel from 507 to 8. All 528 sampled mainstem axis
points were wet, including the 39 formerly dry live points. Remaining eight
side cells exceeded the bounded cut by 2-8 m and stayed explicitly dry.
Maximum cut was 99.9 m, terrain outside support was bit-identical, and the
extra channel excavation changed the tile's mean by only -0.13 m. These are
CPU reproductions of captured neural data; the [integration method](hydrology-integration.md)
gives the precise budget and its calibrated 30 m residual allowance.

The [integration note](hydrology-integration.md) records supported chart guard
strips, source-grid watershed boundaries, seasonal proxies and deferred
dynamic lake, flood-wave and distributary modelling.

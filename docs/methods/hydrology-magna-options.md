# Magna Urbis and Neural Earth: hydrology options

This comparison uses Magna Urbis's `rust/bridge/hydrology.ts` and
`web/src/ui/vectorWater.ts`, and Neural Earth's `web/terrain_hydrology.js`.
They are different pipelines: a bounded local map versus a persistent source
network refined over independently generated neural terrain. The original
Neural Earth integration did not port all of Magna Urbis's methods.

| Magna Urbis control | Neural Earth treatment |
|---|---|
| Density / minimum river catchment | Zoom-dependent stream selection, adjustable density and minimum catchment in km². Uses existing drainage and flow; changes the display without rerouting the world. |
| Width multiplier | Multiplies the cartographic river width. Physical discharge, depth and navigation estimates stay unchanged. |
| Terrain influence | Base routing is part of the fast path. Additional valley-floor fitting remains optional. |
| Minimum lake area / depth | Display filters over existing lake geometry and attributes. They do not create water or change lake budgets. |
| Lake appearance | Simple vector fill for resolved shores; incomplete coarse shores are outlines at fine zoom. Optional rounded shores are separate from terrain carving. |
| Conditioning, breach and incision | The detailed regional/carving path is optional and slower. Its settings are not a one-to-one port of Magna's local conditioning controls. |
| Wetness | No cosmetic water multiplier: runoff comes from the selected world's precipitation and evapotranspiration balance. |
| Meanders / sinuosity | The generated meander and nested-meander controls have been removed. The retained drainage axis is simplified and interpolated consistently across LODs. |
| Braiding, anabranching, avulsions, deltas | Not implemented. These require additional channel topology and water partitioning, not an inactive checkbox. |
| Estuary modes | Not implemented as separate funnel/tidal estuary models. |

The fast viewer deliberately exposes only functioning options. Its display
thresholds do not change the underlying catchments or annual discharge. The
climatological monthly regime is synthetic, not a prediction of individual
storms or a calibrated perennial-flow assessment.

See [fast delivery](hydrology-performance.md),
[river rendering](hydrology-rendering.md) and
[water balance](hydrology-balance.md) for the implemented methods and limits.

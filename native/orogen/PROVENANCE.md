# Original Orogen pipeline

The JavaScript modules in `vendor/` were copied from the local
`../world_builder/orogen/js` snapshot on 2026-10-08. That checkout has no HEAD
commit; no upstream revision is claimed. The upstream GPL-3.0 license is
retained as `vendor/OROGEN-LICENSE`. `sphere-mesh.js` credits Red Blob Games.
Delaunator 5.0.1 and its ISC license are bundled for offline execution.

`pipeline.mjs` runs the original worker headlessly with Node.js. Original
superplates, Euler-pole collisions, stress propagation, uplift, hotspots,
terrain processing, wind, ocean currents, precipitation, temperature,
Koppen classification and biome colors are retained.

Adaptations in `planet-worker.js`: local Delaunator import; optional bypass
of all postprocessing; direct float32 metre heightmap import using the inverse
of the calibrated elevation curve; pixel-centre coordinates and the
application's atan2(z,x) longitude convention; configurable propagation and
plate angular-speed multiplier (original defaults 5 and 1).
`elevation.js` exposes boundary types and makes the original 0.75 collision
threshold configurable. `color-map.js` adds a finite coastal slope to the
original quartic physical-altitude mapping (see below).

For seed 42, a 20K mesh, and original defaults, graph elevation, both temperature
seasons, both precipitation seasons, and Koppen classes were compared against
a separately executed original worker: all six arrays were bit-identical.
That comparison describes the historical mapping; `coastal-slope-v1` deliberately changes
physical coastal altitudes and consequently climate. Dense spherical triangle interpolation is performed with CUDA in
`terrain_orogen_cuda.py`; it is an adaptation of the representation, not a
replacement tectonic formula. Categorical fields use the nearest graph vertex.
Source hashes and runtime identity participate in persistent atlas identity.

The optional City GPU engine adds a deferred-climate adapter hook,
`setExternalElevation`, to the worker. The headless adapter exports unprocessed
relief, waits while Python applies City WebGPU erosion, then resumes climate
on the same retained spherical mesh, original plates and noise state.
Eroded metre heights use the matching calibrated inverse convention. The source
atlas remains the GPU engine's exact output; graph climate samples are limited
by mesh/raster resolution and the original 6000 m climate altitude ceiling.
The original Orogen postprocessing path remains the default.

With `coastal-slope-v1`, positive elevation `t=min(e,1)` maps to
`6000*t^4*(5-4*t)+1000*t*(1-t)^4` metres. The added bounded term provides a
1000 m/unit slope at sea level (about 9.6 m at t=0.01), rather than making
large terrestrial regions sub-metre plains. It adds at most 81.92 m and leaves
the 6 km peak unchanged. This is an explicit coastal calibration, not an
empirical DEM distribution. Ocean depth stays `10000*e`, retaining the existing
bathymetry and exact zero crossing; the land/ocean slope discontinuity already
existed and is now smaller. Python, Node climate/import and CUDA helpers share
the law. Imported physical metres use its inverse, so imports are not raised.
No smoothing, coastline mask, height floor or neural-noise suppression is used.
Stage artifacts record the physical-height convention. Historical snapshots
without that annotation have their retained pre/post graph elevations rebased
through the old physical metres before resuming, keeping climate-only commands
consistent with the visible historical height; immutable old bytes are retained.

The optional CUDA adapter under `gpu/` creates temporary module copies; it
never modifies these reference modules in place. All GPU switches default to
off. Its experimental graph/erosion algorithms, retained CPU stages, bundled
Acorn license and runtime fallback are detailed in `gpu/PROVENANCE.md`.

The independent-stage adapter adds worker exports `exportRetainedState`,
`restoreRetainedState` and `applyRetainedErosion`. Snapshots retain the original
mesh/halfedges, plate state and raw/final elevations. The noise generator is
recreated from its original seed; wind/current caches are cleared before each
climate command. Retained erosion invokes the original postprocessing with the
original hotspot field. Imported sources rederive synthetic regions after
erosion, matching the original import pipeline. These exports avoid rebuilding
relief for climate-only or erosion-only commands. The original CPU full pipeline
remains the numerical reference and is checked against the staged CPU chain.

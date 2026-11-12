# Orogen v3 integration

The active generator is World Orogen v3.0.1, vendored from
[raguilar011095/planet_heightmap_generation](https://github.com/raguilar011095/planet_heightmap_generation)
at commit `ed466f9db71292e912d8d0160e689a67a994f2d9` (2026-11-09).
The source dependency closure of `js/planet-worker.js` is in `upstream/`.
Its GPL-3.0 license and COPYRIGHT are retained there. No upstream render shaders,
application UI, model weights or generated worlds are used by this integration.

## Upstream adaptations

`upstream/js/planet-worker.js` has two local changes: the CDN Delaunator import
uses the existing ISC-licensed bundle in `vendor/`, and `retainedTerrain()`
exports the graph triangles and ground-shaping inputs for independent stages.
`upstream/package.json` declares ES modules for headless Node execution.
`upstream/v2/js/present/belts.js` additionally records the actual back-arc
footprint while the existing algorithm builds it. This adds a data channel,
without changing random draws, terrain calculations or crust evolution.
The terrain, finish, climate, geological and agricultural algorithms are upstream.
The remaining files in `vendor/` preserve our configurable biome palette and the
existing two-season display classifier. Their original local snapshot revision
was unknown; they are not presented as part of the pinned v3 upstream source.

## Data adapter

`pipeline.mjs` runs Generate, retains its geological frame, reapplies sculpting
to that frame, and runs the common Evolution finish. Climate-only commands run
the new monthly climate on the retained final heights without reshaping land.
The finish itself depends on climate (runoff, ice and sediment); its stage
identity includes those climate settings. A later climate-only edit marks those
erosion settings pending, while keeping the explicit independent-stage behavior.

The upstream z-north coordinates `(x,y,z)` become application y-north `(x,z,y)`;
longitude remains `atan2(z,x)` in the application. Heights are physical kilometres
converted linearly to metres, with no 6 km cap. Imported metre rasters use
pixel-centre bilinear sampling and longitude wrapping, with no nonlinear inverse.
The original imported raster remains exact when erosion is disabled.

The old `coastal-slope-v1` correction compensated for the former quartic
normalized-height conversion. V3's generated heights already have physical units;
applying that curve again would distort them. The new conversion has constant
slope through sea level and preserves centimetre inputs and high mountains.
The upstream coast morphology is otherwise retained, including its known high
coastal ranges; this integration does not smooth or lower those ranges.

`climate.mjs` exports all twelve monthly temperature/rain fields and physical
BIO1, BIO4, BIO12 and BIO15. BIO4 uses sample SD × 100 and BIO15 uses sample SD
divided by mean monthly rain + 1 mm, × 100. Seasonal fields remain available to
the existing diagnostics. All neural inputs are finite, including over water.
The v3 temperature control is radiative forcing (W/m²), not a Celsius offset.
Climate parameters are mapped explicitly to upstream PARAMS; the old physical
climate tuning knobs and V1 graph GPU transformations have been removed.

Existing resource inputs receive volcanic relief (`volcWornKm / 6`) as `hotspot`
and the recorded back-arc footprint as `backArc` (`-0.06` for full membership).
The latter is a compatibility scale for resource-context membership, not a v1
elevation contribution. The unscaled `geology_backarc` field is also retained.
Imported heightmaps have no invented tectonic history: these context fields
are zero and crust follows the input's land/ocean mask.

CUDA still rasterizes the graph's original triangles. Categories use the nearest
vertex, continuous physical fields use barycentric weights. Optional exact CUDA
spatial search and external City WebGPU erosion remain separate choices.
Source and Node-runtime hashes participate in atlas/stage identities. V3 uses a
fresh cache namespace and rejects V1 binary snapshots; no old-world migration
or parallel legacy generator is provided.

The WASM build uses the same vendored v3 engine and monthly-climate adapter;
its existing inference and rendering shaders are unchanged.

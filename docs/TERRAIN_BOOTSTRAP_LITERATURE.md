# Terrestrial coarse bootstrap: primary literature and implementation options

Research date: 2026-10-07. Scope: a finite 40,000,000 × 20,000,000 metre plane, sampled by the NN at approximately 7,680 metres. This report does not inspect or reuse `city_generator`, the rejected A4 capsule construction, or canned initial worlds. Private `world-builder-rs` and `map_denoise` inspection belongs to the other source audits; no claim about their contents is made here.

## Recommendation

First evaluate the user's existing **world-builder-rs generation itself** as the bootstrap provider. If its audit confirms that it already supplies coherent continental structure, sufficient style controls, deterministic seeds, and CPU export, adapting its height and climate output is preferable to replacing it with another invented continent heuristic. Preserve its real generation stages and expose their parameters. Compare it with a bounded hybrid alternative described below; do not assume that recognizing the repository name proves its suitability.

The practical alternative is **global continental structure → plate-informed relief → constrained correlated detail → optional drainage erosion → physical field cache**. Treat continent organization and local surface roughness as different controls. This follows demonstrated public implementations and published multiscale terrain workflows. A whole-world CPU build followed by cheap coordinate sampling fits this application's finite planet much better than independently normalizing arbitrary requested windows. This recommendation is an engineering inference, not a measured performance result.

## Credible alternatives

| Approach | Evidence and actual mechanism | Useful here | Main limitation |
|---|---|---|---|
| Existing world-builder-rs generation | Requires the private source audit | Reuse established generation, seed controls, exports and parameter semantics | Suitability and output units must be demonstrated |
| Static tectonic and distance-field hybrid | World Orogen: graph-grown continents/plates, boundary classification, coast and mountain distances, correlated detail | Controllable large landmasses, coherent mountain belts, broad ocean basins; straightforward CPU rasterization | Geological plausibility rather than physical evolution |
| Dynamic procedural tectonics | Cortial et al. (2019); platec/Viitanen (2012) | Collisions, accretion and rifting give coherent histories and varied continental arrangements | Iterative initialization, tuning, integration cost; style guarantees are indirect |
| Uplift + fluvial erosion | Cordonnier et al. (2016), Braun–Willett (2013), Fastscapelib | Drainage networks and relief tied to uplift instead of disconnected noise ridges | Needs a separately credible continental/ocean substrate |
| Correlated scalar fields with topology constraints | Perlin's primary noise implementation; FastNoiseLite; own proposed global topology layer | Fast, deterministic fallback and continuous style controls | Noise/warping alone cannot guarantee continents, ocean connectivity or geology |
| Graph hydrology with authoring | Red Blob Games Mapgen4 | Useful drainage/climate architecture and manual world controls | Regional wilderness generator; not a demonstrated planetary tectonic bootstrap |

### World Orogen: useful reference, with clear scope

[World Orogen](https://github.com/raguilar011095/planet_heightmap_generation) describes itself as concept art with scientifically informed plausibility. Its README identifies graph-grown plates and continents, a land-area control, broad tectonic units, coast/mountain/ocean distance fields, shelf/abyss structure, domain warping, erosion and climate. These are relevant stages, but its screenshots and claims do not establish runtime or geological accuracy in this project. The browser renderer/exporter is also separate from the terrain-generation algorithm.

The inspected [coarse-plates.js](https://github.com/raguilar011095/planet_heightmap_generation/blob/cc2662b4edd52231c4f65d8765f3ef12cd82d9b7/js/coarse-plates.js) builds structural data on a fixed approximately 20K-region sphere mesh with an isolated RNG. It projects assignments onto higher resolution with noise-perturbed nearest-region lookup. This is concrete evidence for keeping continent topology independent of requested output resolution.

The inspected [ocean-land.js](https://github.com/raguilar011095/planet_heightmap_generation/blob/cc2662b4edd52231c4f65d8765f3ef12cd82d9b7/js/ocean-land.js) builds plate adjacency, chooses continent seeds, and grows adjacent plates toward separate area budgets while protecting separation. Continent size variety changes those budgets. This is a meaningful structural model rather than a list of imposed capsule silhouettes.

The inspected [elevation.js](https://github.com/raguilar011095/planet_heightmap_generation/blob/cc2662b4edd52231c4f65d8765f3ef12cd82d9b7/js/elevation.js) separates tectonic inputs, spatial fields, terrain classification, a relief skeleton and detail. It mixes ordinary and ridged fBm according to fold-belt influence, and warps coordinates. Adopt the principle that ridged detail is spatially conditioned by relief structure, rather than raising ridges indiscriminately across every plain and ocean.

Orogen's sphere mesh should not silently change the application's plane geometry. Either explicitly import an equirectangular procedural raster with known mapping, or reproduce the graph/raster architecture on the prescribed plane. Broad geographical analogy is appropriate; latitude-dependent metric changes require an explicit design decision.

### Dynamic tectonics and physical terminology

[Cortial, Peytavie, Galin and Guérin, *Procedural Tectonic Planets* (2019)](https://doi.org/10.1111/cgf.13614) models moving plates and approximates subduction, collision, crust creation and rifting, then amplifies crust data into detailed terrain. The authors explicitly avoid computationally demanding physically based simulation. Thus **dynamic procedural tectonics** is accurate terminology; mantle-convection simulation or validated geophysics is not. [Author-hosted paper](https://perso.liris.cnrs.fr/eric.galin/Articles/2019-planets.pdf).

[Viitanen's 2012 thesis](https://urn.fi/URN:NBN:fi:amk-201204023993) underlies [Mindwerks/plate-tectonics](https://github.com/Mindwerks/plate-tectonics), a C++ library with Python bindings. Its API exposes seed, dimensions, plate count, folding, aggregation, erosion period and simulation cycles. [WorldEngine's plates.py](https://github.com/Mindwerks/worldengine/blob/b2adbb674afb381232097f126e548a8c133e4633/worldengine/plates.py) actually steps platec to completion and obtains height and plate maps; its source comment suggests multiplying heights by roughly 2,000 for Earth scale. That conversion is an implementation convention, not a guarantee of metre-calibrated outputs. Benchmark and calibrate before feeding the checkpoint.

These are stronger alternatives than labeling Voronoi noise as a simulation. Neither assures a desired supercontinent or archipelago solely by changing plate count. Topology/style constraints remain necessary, or the generation must expose history and continental-crust controls.

### Erosion and noise

[Cordonnier et al., *Large Scale Terrain Generation from Tectonic Uplift and Fluvial Erosion* (2016)](https://doi.org/10.1111/cgf.12820) combines an input uplift map, stream-power erosion and a stream graph before terrain reconstruction. It supports relief/drainage coherence; it is not a complete continent generator. [Author-hosted paper](https://www.cs.purdue.edu/cgvlab/www/resources/papers/Cordonnier-Computer_Graphics_Forum-2016-Large_Scale_Terrain_Generation_from_Tectonic_Uplift_and_Fluvial_.pdf).

[Braun and Willett (2013)](https://doi.org/10.1016/j.geomorph.2012.10.008) provide an implicit stream-power solver with linear complexity in discretization points. [Fastscapelib's SPLEroder](https://fastscapelib.readthedocs.io/en/latest/api_python/_api_generated/fastscapelib.SPLEroder.html) implements this family of schemes; [its documented examples](https://fastscapelib.readthedocs.io/en/latest/examples/index.html) include raster and periodic-domain configurations. This is a CPU-compatible alternative to simulating huge numbers of water droplets. A routing/reconstruction stage still has its own cost.

[Barnes, Lehman and Mulla, Priority-Flood (2014)](https://doi.org/10.1016/j.cageo.2013.04.024) resolves raster depressions; the [authors' reference code](https://github.com/r-barnes/Barnes2013-Depressions) distinguishes filling, flow directions and related variants. Filling a routing surface is not sediment simulation. Keep intentionally endorheic basins when the style requires them; do not blindly require every real continent cell to drain into an ocean.

[Perlin's original improved-noise implementation](https://mrl.cs.nyu.edu/~perlin/noise/) supplies a continuous coherent primitive. [FastNoiseLite](https://github.com/Auburn/FastNoiseLite) provides Perlin/OpenSimplex, fractal variants and domain warping across native CPU languages. Neither creates geological structure automatically. Domain warping is a coordinate displacement; excessive amplitude can fragment coastlines and fold narrow ridges. Use it after establishing structural correlation scales and connectivity constraints. [Mapgen4](https://www.redblobgames.com/maps/mapgen4/) is an inspectable example connecting user-defined terrain with rainfall and river behavior.

## Proposed CPU architecture and continuous controls

The following is a proposed implementation plan, not a claim that any cited project already provides this exact API.

1. Generate a fixed global structural grid and adjacency graph in world metres. Use deterministic plate/crust growth or the verified world-builder provider. Continental crust may span multiple plates; a plate should not automatically equal a continent.
2. Expose a continuous fragmentation/concentration axis plus land fraction, size inequality, spacing, rift intensity, shoreline correlation and island-arc density. A Gondwana-style endpoint concentrates most land in one connected component; an Earth-like interior gives a few large components; the archipelago endpoint distributes many substantial islands over coherent shallow regions and arcs. Integer plate counts can coexist with continuous area/connectivity budgets. Topological changes at mergers/splits are unavoidable, so continuity means smoothly changing parameters/statistics, not mathematically smooth coastline topology.
3. Use physical-scale correlation: broad continental structure on thousand-kilometre wavelengths; belts/basins on hundreds; shoreline and regional detail on tens to hundreds. These are starting bands to tune, not geological constants. Restrict ridged fBm to uplift/belt masks and ordinary fBm to interiors. Avoid unresolved frequencies below the coarse raster's Nyquist wavelength (15.36 km at 7.68 km spacing); approximately 4–8 samples per visible feature is safer for conditioning.
4. Build land elevations, continental shelves/slopes and ocean floors with region-specific continuous profiles. The shoreline is an explicit zero crossing. Keep spatially variable relief amplitudes, lowland area and broad basins. Use a global sea-level quantile if enforcing area fraction, then check connectivity; a quantile alone can still produce hundreds of noise islands. Do not use local-window quantiles, local min/max normalization, universal ridge uplift or global hard clipping.
5. Optionally perform a bounded number of native CPU drainage/uplift iterations on a smaller fixed grid, then reconstruct and sample into the canonical raster. Scale erosion with grid spacing and physical units. Preserve oceans, shelf geometry and meaningful basins instead of smoothing the whole world indiscriminately.
6. Derive/cache physical climate from latitude convention, metres, distance to ocean, rain-shadow/orographic structure and selected world climate parameters. Map these to the checkpoint's actual expected channels, units and normalization from its source/config audit. Do not substitute arbitrary biome categories or renderer colors for conditioning channels.

At 7,680 m spacing, covering this extent needs roughly 5,209 × 2,605 cells, or 13.57 million samples, depending on the established lattice origin and endpoint convention. One float32 field is approximately 54.3 MB (51.8 MiB); six fields are approximately 326 MB before auxiliary arrays. A structural 1,024 × 512 grid is only 524,288 cells and approximately 2 MB per float32 field. This makes fixed structural generation followed by rasterization plausible on CPU; it does **not** establish a runtime target. Measure cold generation, memory peak, cache reload and coordinate-window sampling independently.

## Determinism and checkpoint boundary

Cache identity must include seed, all style/physical parameters, algorithm/source version, canonical extent, spacing, lattice origin, boundary conditions, field units and climate mapping version. Use independently derived RNG substreams for continents, plates, relief and climate so adding a detail octave does not silently redraw the continents. No per-view random seeds or changes based on tile traversal.

Generate and calibrate globally once. A narrow request, a wide request and an overlapping request must sample the same cached metres and physical climate at identical world coordinates. Bilinear interpolation is suitable for continuous physical fields; categorical plate/continent identifiers require separate discrete handling. Check shoreline zero crossings after interpolation and avoid spline overshoot near shelves/cliffs.

Keep height in **physical metres** through calibration, erosion, persistence and spatial interpolation. Apply the signed square root **once at the NN interface**, then apply exactly the checkpoint's verified channel normalization. Do not interpolate encoded square roots and call the result an interpolation of physical height: for example, equal weighting of 0 and 1,000 metres gives 500 metres before encoding, but encoding first and then decoding the midpoint gives 250 metres. Existing app source confirms the 7,680 m coarse spacing and signed-square decoding (`terrain_coarse.py`), but the complete climate/checkpoint contract belongs to the source audit.

## Acceptance evidence

Evaluate several procedurally sampled seeds per style, with no curated seed acceptance. Compare total land fraction, largest land component's share of land, counts of substantial components (area cutoff in km²), ocean connectivity, shoreline complexity at fixed scale, continental shelf area, land/ocean elevation quantiles, hill/belt distribution and spectral/correlation scale. Show maps in metres plus plate/crust/uplift diagnostics. Use the same seed and relief parameters while varying fragmentation, and test intermediate settings as well as endpoints.

Require exact repeatability for cached fields and tolerance-bounded agreement for sampled physical fields at matching coordinates. Verify overlapping/nested windows, cache reload, request order, resolution changes and signed-square-root encoding. Benchmark CPU initialization separately from NN work. A successful height histogram or attractive overview is insufficient evidence of continental organization.

## Reproducible public source snapshots

Commit timestamps below are GitHub default-branch commit dates, not release dates. Metadata and selected source files were inspected on 2026-10-07. No source was copied into production files.

| Project | Inspected commit / date | Declared repository license |
|---|---|---|
| [World Orogen](https://github.com/raguilar011095/planet_heightmap_generation/tree/cc2662b4edd52231c4f65d8765f3ef12cd82d9b7) | `cc2662b4edd52231c4f65d8765f3ef12cd82d9b7`, 2026-07-03 | [GPL-3.0](https://github.com/raguilar011095/planet_heightmap_generation/blob/cc2662b4edd52231c4f65d8765f3ef12cd82d9b7/LICENSE) |
| [plate-tectonics](https://github.com/Mindwerks/plate-tectonics/tree/2a27c4fb137c657517bca62122b9de80e6b8c255) | `2a27c4fb137c657517bca62122b9de80e6b8c255`, 2025-11-23 | [LGPL-3.0](https://github.com/Mindwerks/plate-tectonics/blob/2a27c4fb137c657517bca62122b9de80e6b8c255/LICENSE) |
| [WorldEngine](https://github.com/Mindwerks/worldengine/tree/b2adbb674afb381232097f126e548a8c133e4633) | `b2adbb674afb381232097f126e548a8c133e4633`, 2026-08-27; README stable version 0.20.0 | [MIT](https://github.com/Mindwerks/worldengine/blob/b2adbb674afb381232097f126e548a8c133e4633/LICENSE.txt); platec dependency has its own license |
| [FastNoiseLite](https://github.com/Auburn/FastNoiseLite/tree/785f37a9ad76e283586a379675085f2063ae03f7) | `785f37a9ad76e283586a379675085f2063ae03f7`, 2026-06-21 | [MIT](https://github.com/Auburn/FastNoiseLite/blob/785f37a9ad76e283586a379675085f2063ae03f7/LICENSE) |
| [Fastscapelib C++/Python](https://github.com/fastscape-lem/fastscapelib/tree/b85cc6b1341241cbe4a1d3458ed7e3fa5a857f48) | `b85cc6b1341241cbe4a1d3458ed7e3fa5a857f48`, 2025-06-25 | [GPL-3.0](https://github.com/fastscape-lem/fastscapelib/blob/b85cc6b1341241cbe4a1d3458ed7e3fa5a857f48/LICENSE) |
| [Mapgen4](https://github.com/redblobgames/mapgen4/tree/c1d8cb018a11a8b9e17d59233c36c176429d37eb) | `c1d8cb018a11a8b9e17d59233c36c176429d37eb`, 2026-03-29; author's page dates July 2018 / April 2023 | [Apache-2.0](https://github.com/redblobgames/mapgen4/blob/c1d8cb018a11a8b9e17d59233c36c176429d37eb/LICENSE) |

Repository licenses describe code reuse, not a license grant for all dependencies or paper text. Fastscapelib C++/Python is a distinct repository from its Fortran sibling; do not conflate their versions or APIs. Claims here rely on primary papers, authors' sites and actual repositories, not search-result commentary.

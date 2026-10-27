# Multiscale hydrology: focused literature and project review

Date: October 21, 2026. This review accompanies the [stack design](multiscale-hydrology.md). It selects methods useful to Neural Earth; it is neither an exhaustive systematic review nor a reproduction of published results. No external performance figure is presented as a measurement of our integration. [French original](hydrology-literature_FR.md).

The problem is specific: the world is finite, spherical or planar, but detailed neural terrain arrives on demand. Upstream–downstream relationships must be preserved without knowing the entire fine DEM. Methods assuming an existing global DEM and those building all terrain around rivers provide components; none integrates unchanged.

## Drainage, basins and spatial continuity

| Primary reference | Contribution | Decision for Neural Earth |
| --- | --- | --- |
| [Barnes, Lehman and Mulla, 2014 — Priority-Flood](https://arxiv.org/abs/1511.04463), *Computers & Geosciences*, DOI 10.1016/j.cageo.2013.04.024 | Ordered flooding from outlets, depression handling, variants for flow directions and basins. A simple queue inside depressions. | Retain a deterministic Numba-compiled kernel with a total tie order. Keep raw ground separate from the routing surface. Filling does not mean all depressions become lakes. |
| [Barnes, 2016 — Parallel Priority-Flood](https://arxiv.org/html/1606.06204v2), DOI 10.1016/j.cageo.2016.07.001 | Each domain produces labels and thresholds; a wider graph then determines the common solution. | Retain the principle of shared boundary conditions. The paper does not show that a halo suffices or that unknown domains can be ignored. Our hierarchical solution will be an adaptation constrained by parents, not exact drainage of an entire planet decoded at 30 m. |
| [Eilander et al., 2021 — Iterative Hydrography Upscaling](https://hess.copernicus.org/articles/25/5287/2021/), DOI 10.5194/hess-25-5287-2021 | Scaling known hydrography while preserving connections and channel parameters. | Retain connectivity and sub-grid quantities as LOD contracts. Reject the idea that IHU can invent our fine network: it proceeds from fine to coarse. |

Design consequence: compute an inexpensive global authority on the final source, then canonical regions with shared boundary crossings. A display tile cannot decide its incoming flow, outlet or lake level on its own. Unvisited areas must not contribute less water than already loaded areas.

## Lakes, depressions and endorheism

| Primary reference | Contribution | Decision for Neural Earth |
| --- | --- | --- |
| [Barnes, Callaghan and Wickert, 2020 — Depression Hierarchy](https://esurf.copernicus.org/articles/8/431/2020/), DOI 10.5194/esurf-8-431-2020 | Representation of nested depressions, their spill thresholds and connections. | Retain an explicit hierarchy. Avoid confusing an “area below an elevation” with a connected lake. |
| [Barnes, Callaghan and Wickert, 2021 — Fill–Spill–Merge](https://esurf.copernicus.org/articles/9/105/2021/), DOI 10.5194/esurf-9-105-2021 | Distribution of volumes in that hierarchy, spilling and merging filled depressions. | Adapt to a parsimonious annual, then seasonal balance. This is neither a complete rainfall–runoff model nor a river hydraulics solver; evaporation, infiltration and climate are additional choices in our model. |

Open lakes, closed lakes and dry basins are needed. Magna Urbis's local implementation is a practical reference, but its simplified equilibrium calculation and fixed iterations do not prove general convergence. One lake identity must control level, outline, routing and rendering, including when its footprint crosses regions. An inland basin below zero does not automatically become sea.

## River forms, terrain and LOD

| Primary reference | Contribution | Decision for Neural Earth |
| --- | --- | --- |
| [Génevaux et al., 2013 — Terrain Generation Using Procedural Models Based on Hydrology](https://perso.liris.cnrs.fr/eric.galin/Articles/2013-river-networks.pdf), DOI 10.1145/2461912.2461996 | Hierarchical geometric network and terrain composed around hydrological primitives. | Retain the separation of structure, path and surface. Replacing the entire neural generator with terrain built from rivers would fall outside our objective. |
| [Cordonnier et al., 2016 — Large Scale Terrain Generation from Tectonic Uplift and Fluvial Erosion](https://doi.org/10.1111/cgf.12820) | Coupling terrain and drainage at large scale. | Reference for geomorphological consistency and Orogen's context. Do not add a second global tectonic and erosion simulation after inference. |
| [Peytavie et al., 2019 — Procedural Riverscapes](https://diglib.eg.org/items/555e713d-c93f-4c7c-987e-674f7153d34d), DOI 10.1111/cgf.13814 | Composition of bed and water over terrain, with river forms and procedural animation. | Retain bed cross-sections and water effects dependent on morphology. Our adaptation does not claim the full implementation of their animation system. |
| [Dou et al., 2024 — A unified LOD model for river network and DEM](https://www.frontiersin.org/journals/environmental-science/articles/10.3389/fenvs.2024.1397798/full), DOI 10.3389/fenvs.2024.1397798 | Joint simplification of vector data and terrain to avoid disagreement. | Retain the requirement for river/terrain agreement during simplification; keep current renderers without introducing their TIN representation. |
| [Davis, 2026 — Plate-Local River Generation](https://arxiv.org/html/2610.00301v1), preprint arXiv:2610.00301 | On-demand regional computation and composition of a descending channel profile with terrain. | Retain the idea of bounded computation domains. The network explicitly does not cross plates: this restriction rules it out as the basin architecture for Neural Earth. Integration with a learned generator remains prospective there. |

The remaining challenge is the conflict between retained topology, unconstrained neural terrain and limited incision. None of the selected papers guarantees all three properties simultaneously in Neural Earth. The plan proposes corridors, shared profiles and a physical bed layer, detecting incompatibilities before publication. An integrated trial must measure the actual quality of that compromise.

For meanders, the first solution reuses continuous offsets and valley constraints from Magna's local Rust code, with a phase attached to a stable reach. Repeated historical migration will not be rerun on zoom. Sinuosity must depend on slope, confinement and stream size; noise or a spline applied uniformly to a D8 trace is insufficient.

## Climate, discharge, low flow and navigability

- **[Oudin et al., 2005, method presentation by INRAE](https://webgr.inrae.fr/eng/tools/hydrological-models/evapotranspiration-model)**: parsimonious estimation of potential evapotranspiration. Possible adaptation to temperature data and an explicit astronomical convention. This does not replace a soil balance or basin calibration.
- **[Hock, 2003 — Temperature index melt modelling in mountain areas](https://doi.org/10.1016/S0022-1694(03)00257-9)**: support for simplified melt related to positive temperatures. Retain for the seasonal snow/discharge lag without claiming to simulate glacier evolution.
- **[USGS — indirect measurements and Manning relation](https://wwwrcamnl.wr.usgs.gov/sws/fieldmethods/Indirects/rantz.htm)**: flow estimation under uniform-flow assumptions. Useful for approximating wetted cross-sections on suitable reaches; insufficient for backwater, lakes, tides or navigation guarantees.

Neural Earth supplies BIO statistics and, with Orogen, two half-year periods. These do not determine a weather time series. Mean discharge, climatological monthly minimum and dry months will therefore be distinguished from true low-flow quantiles or return periods, which these data cannot estimate. Navigability will be a seasonal worldbuilding classification including obstacles and route continuity.

## Available projects and reuse strategy

| Project | Relevant components | Choice |
| --- | --- | --- |
| **Local Magna Urbis**, `../city_generator`, files at HEAD `1d245aa`; embedded submodule `08f3e61` | Rust: flood, flats, lake balance, profiles, continuous offsets, Hermite curves; TS: migration and morphology; bridge: beds and rendering | Adapt targeted methods with GPL provenance, without copying the finite n×n orchestrator, edge outlets or cell IDs as world contracts. |
| [PyFlwDir](https://github.com/Deltares/pyflwdir), MIT | NumPy/Numba arrays, basins, accumulation, orders, vectors and IHU | Lightweight implementation reference; adapt attributed functions if useful. Outlet conventions must be replaced. |
| [RichDEM](https://github.com/r-barnes/richdem), GPL-3.0, and [FillSpillMerge code](https://github.com/r-barnes/Barnes2020-FillSpillMerge) | Barnes references for drainage and depressions | Algorithmic reference and small comparison cases; avoid imposing a new C++ build on the runtime. Check the precise license of each reused file. |
| [pysheds](https://github.com/pysheds/pysheds), GPL-3.0 | Drainage and basin delineation | Independent reference for a few finite cases; no need to install a second production stack. |
| [meanderpy](https://github.com/zsylvester/meanderpy), Apache-2.0 | Kinematic migration model based on upstream curvature, meanders and cutoffs | Morphological comparison source. Its iterative solver and visualization dependencies do not enter the interactive path. |
| [mapgen4 / Red Blob Games](https://github.com/redblobgames/mapgen4), Apache-2.0 | Compact organization of graphs, meshes and procedural rendering | Structural and readability inspiration. Its terrain construction differs from our neural runtime. |
| [Plate-local river generation](https://github.com/Davismk3/plate-local-river-generation) | Three C++17 demonstrators and their preprint | Study bounded computation; no direct integration or automatic attribution of a code license from the paper's license. |

Any actual reuse will record the commit, notices, functions concerned and their modifications. A project's existence does not demonstrate its performance on our worlds, compatibility with our seams or the quality of our lakes.

## Retained synthesis

The stack is a specific hybrid: global drainage and depression hierarchy on the final source; inherited spatial contracts; thalweg refinement on canonical neural fields; continuous geometry inspired by Magna; jointly conditioned terrain and water; conservative climate balance and derived properties. Guarantees apply to this constructed model. Exact drainage of an unknown global neural DEM is not claimed.

The methods review, reading local sources and Opus critiques are three distinct forms of evidence. The theoretical model is documented; visual quality, neural dependency costs and frequency of terrain conflicts must be established during the authorized implementation.

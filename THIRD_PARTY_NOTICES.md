# Neural Earth third-party credits

Neural Earth integrates upstream research and procedural generators. Their project names, attribution and licenses are retained.

| Component | Source and role | License / provenance |
| --- | --- | --- |
| InfiniteDiffusion / Terrain Diffusion | [Alexander Goslin's code](https://github.com/xandergos/terrain-diffusion) and [paper](https://arxiv.org/abs/2512.08309); pretrained terrain hierarchy and lazy spatial generation | Upstream [license](terrain-diffusion/LICENSE); checkpoint terms on [Hugging Face](https://huggingface.co/xandergos/terrain-diffusion-30m) |
| World Orogen | [Planetary generator](https://github.com/raguilar011095/planet_heightmap_generation); tectonic relief, erosion and climate | GPL-3.0, bundled [license](native/orogen/vendor/OROGEN-LICENSE); [adaptation provenance](native/orogen/PROVENANCE.md) |
| MagnaUrbis / City erosion | Adapted surface WGSL erosion from `city_generator`; [current repository](https://github.com/dunkean/burgmap), planned [magna-urbis repository](https://github.com/dunkean/magna-urbis) and [website](https://dunkean.github.io/magna-urbis/) | [GPL-3.0-only license](native/city_erosion/LICENSE) and [provenance](native/city_erosion/PROVENANCE.md) |
| Magna Urbis river geometry | River-bend phase modulation and surface-envelope ideas adapted from local `city_generator` revision `1d245aa`; new multi-scale implementation in `backend/terrain_hydrology_geometry.py` | GPL-3.0-only, retained [license](native/city_erosion/LICENSE); [method and adaptation details](docs/methods/hydrology-geometry.md) |
| Magna Urbis neural Rust port | Pinned `native/magna-urbis` submodule; existing Terrain Diffusion EDM, noise, window planning, reconstruction and decoder implementations compiled for the browser | GPL-3.0-only; [browser integration](docs/wasm.md) |
| Burn / CubeCL / wgpu | Rust tensor execution and browser WebGPU compute | Upstream MIT/Apache-2.0 license texts and Cargo metadata; pinned by the WASM crate's Cargo.lock |
| World Builder | [world-builder-rs](https://github.com/dunkean/world-builder-rs), optional CPU atlas bridge | MIT upstream; pinned commit `009efe18c457757da12ffc8a6215806efb87f012` |
| Cartographic style tokens | Adapted from City Generator style/rendering code | GPL-3.0; see `terrain_styles.json` and source comments |
| ETOPO / WorldClim | Elevation distribution and climate statistics | [NOAA ETOPO](https://www.ncei.noaa.gov/products/etopo-global-relief-model), [WorldClim](https://www.worldclim.org/data/worldclim21.html) |

The bundled Orogen snapshot was copied from a local checkout without a HEAD commit; no upstream revision is invented. Its source hashes and changes are documented in the native provenance. Delaunator's ISC license and Acorn's license are retained alongside their bundled code. The current integration includes GPL-covered components; this document does not replace their license texts or assign a new license to every file in the repository.

Scientific method references used by Orogen's erosion implementations include [Braun & Willett (2013), stream-power incision](https://doi.org/10.1016/j.geomorph.2012.10.008) and [Barnes, Lehman & Mulla (2014), Priority-Flood](https://doi.org/10.1016/j.cageo.2013.04.024). These are method references, rather than a standalone Orogen publication.

The long-term integration target is MagnaUrbis for procedural roads, settlements and cities. Existing City erosion adapters are separate from that planned settlement-generation integration. Public `magna-urbis` links anticipate the upstream repository and website rename.

| Component | Source and role | License / provenance |
| --- | --- | --- |
| Multi-Scale Erosion | [Schott et al., Terrain Amplification using Multi-Scale Erosion (TOG 2024)](https://h-schott.github.io/publications/mserosion/publi_mserosion.html), [reference source](https://github.com/H-Schott/MultiScaleErosion); stream-power incision, thermal transport, sediment deposition and scale sequencing adapted into new PyTorch algorithms in `backend/terrain_amplification.py` | MIT; upstream copyright and permission text retained below. Local reference sources were read only, never built or executed; this prototype uses bounded windows, inherited drainage and shorter iteration budgets. |

MIT License (MultiScaleErosion adaptation)

Copyright (c) 2023 Schott

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The MultiScaleErosion adaptation also retains three unmodified reference
heightfields in `tests/fixtures/amplification/` for comparison benchmarks;
their source, scaling and MIT license are recorded in that directory's
`README.md` and `LICENSE`.

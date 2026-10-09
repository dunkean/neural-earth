# Neural Earth third-party credits

Neural Earth integrates upstream research and procedural generators. Their project names, attribution and licenses are retained.

| Component | Source and role | License / provenance |
| --- | --- | --- |
| InfiniteDiffusion / Terrain Diffusion | [Alexander Goslin's code](https://github.com/xandergos/terrain-diffusion) and [paper](https://arxiv.org/abs/2512.08309); pretrained terrain hierarchy and lazy spatial generation | Upstream [license](terrain-diffusion/LICENSE); checkpoint terms on [Hugging Face](https://huggingface.co/xandergos/terrain-diffusion-30m) |
| World Orogen | [Planetary generator](https://github.com/raguilar011095/planet_heightmap_generation); tectonic relief, erosion and climate | GPL-3.0, bundled [license](native/orogen/vendor/OROGEN-LICENSE); [adaptation provenance](native/orogen/PROVENANCE.md) |
| City erosion | Adapted surface WGSL erosion from `city_generator` | [GPL-3.0-only license](native/city_erosion/LICENSE) and [provenance](native/city_erosion/PROVENANCE.md) |
| World Builder | [world-builder-rs](https://github.com/dunkean/world-builder-rs), optional CPU atlas bridge | MIT upstream; pinned commit `009efe18c457757da12ffc8a6215806efb87f012` |
| Cartographic style tokens | Adapted from City Generator style/rendering code | GPL-3.0; see `terrain_styles.json` and source comments |
| ETOPO / WorldClim | Elevation distribution and climate statistics | [NOAA ETOPO](https://www.ncei.noaa.gov/products/etopo-global-relief-model), [WorldClim](https://www.worldclim.org/data/worldclim21.html) |

The bundled Orogen snapshot was copied from a local checkout without a HEAD commit; no upstream revision is invented. Its source hashes and changes are documented in the native provenance. Delaunator's ISC license and Acorn's license are retained alongside their bundled code. The current integration includes GPL-covered components; this document does not replace their license texts or assign a new license to every file in the repository.

Scientific method references used by Orogen's erosion implementations include [Braun & Willett (2013), stream-power incision](https://doi.org/10.1016/j.geomorph.2012.10.008) and [Barnes, Lehman & Mulla (2014), Priority-Flood](https://doi.org/10.1016/j.cageo.2013.04.024). These are method references, rather than a standalone Orogen publication.

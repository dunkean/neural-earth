# Render v2 soils: references and shader application

October 23, 2026. Research requested by the user during the real-tile audit. The references guide corrections; their complete algorithms are not presented as already implemented. [French original](render-v2-soil-literature_FR.md).

## Design decision

Ground must remain a detailed material beneath grass, shrubs and crowns. Vegetation masks the portion actually occluded; gaps reveal soil, litter, gravel or moss. At a distance these elements must retain a consistent mean color and roughness. A green color veil over smooth ground is insufficient.

For Render v2, the chosen ground material is continuous, composed before vegetation, with explicitly filtered spatial bands. Geological and ecological structures provide large-scale variation; material details provide resolved grain. Neither should replace the other during zoom.

## Primary references consulted

| Reference | Documented contribution | Proposed application here |
| --- | --- | --- |
| Cook and DeRose, **Wavelet Noise**, SIGGRAPH 2005 ([Pixar paper](https://graphics.pixar.com/people/derose/publications/WaveletNoise/paper.pdf)) | Analysis of detail loss and aliasing in common noise, especially when 3D noise textures a 2D surface; construction of better-bounded bands. | Check band frequency content and projection onto terrain. Detail appearing late during zoom is directly relevant to our captures. |
| Lagae, Lefebvre, Drettakis and Dutré, **Procedural Noise using Sparse Gabor Convolution**, SIGGRAPH 2009 ([authors' publication and code](https://graphics.cs.kuleuven.be/publications/LLDD09PNSGC/)) | Control of frequency, bandwidth and orientation; surface noise and anisotropic filtering. | Candidate for oriented details: sand ripples, strata or fibers. The spectrum must prevent large unwanted patches from noise intended for grain. |
| Heitz and Neyret, **High-Performance By-Example Noise**, HPG 2018 ([authors' paper](https://eheitzresearch.wordpress.com/722-2/)) | Stochastic texture synthesis from a small example, with histogram-preserving blending. | Option for sand, humus, moss or granite when analytical noise fails to reproduce their visual distribution. Preserve material colors and contrast during blending. |
| Deliot and Heitz, **Procedural Stochastic Textures by Tiling and Blending**, GPU Zen 2, 2019 ([chapter and demonstration](https://eheitzresearch.wordpress.com/738-2/), [Unity Labs prototype](https://github.com/UnityLabs/procedural-stochastic-texturing)) | Practical variant and prefiltered tables to limit color drift with mipmaps. | Implementation reference for procedural sampling without obvious repetition. The Unity prototype is a historical reference, not a dependency to install in our WGSL renderer. |
| Olano and Baker, **LEAN Mapping**, I3D 2010 ([paper](https://userpages.cs.umbc.edu/olano/papers/lean/)) | Filtering normal distributions to retain the specular response of bumps that become subpixel. | Unresolved detail should contribute to roughness instead of disappearing without compensation or shimmering. Adapting this principle to our BRDF needs independent validation; the current formula is not automatically an implementation of LEAN. |
| Barré-Brisebois and Hill, **Blending in Detail**, 2012 ([technical article](https://blog.selfshadow.com/publications/blending-in-detail/)) | Comparison of normal blending and presentation of Reoriented Normal Mapping. | If detail normals are introduced, orient them relative to displayed terrain without flattening its slopes. The current sum of derivatives remains an option to compare, not a bug proven by this article. |
| Pharr, Jakob and Humphreys, **Physically Based Rendering**, 4th edition, texture chapter ([sampling](https://www.pbr-book.org/4ed/Textures_and_Materials/Texture_Sampling_and_Antialiasing), [image filtering](https://www.pbr-book.org/4ed/Textures_and_Materials/Image_Texture)) | Pixel-footprint calculation and elimination of unrepresentable frequencies; anisotropic texture filtering. | Derive filtering from projected-surface derivatives, accounting for pattern orientation. Requested LOD alone does not define visible bands. |
| Jensen, Legakis and Dorsey, **Rendering of Wet Materials**, 1999 ([authors' publication](https://graphics.cs.yale.edu/publications/rendering-wet-materials)) | Water on and within a material changes its appearance through several mechanisms beyond a simple dark tint. | Distinguish soil moisture from free water; vary material parameters by type. A real-time approximation must be explicitly calibrated on our scenes. |
| Khronos, **PBR Lighting Implementation** ([Vulkan documentation](https://docs.vulkan.org/tutorial/latest/Building_a_Simple_Engine/Lighting_Materials/04_lighting_implementation.html)); Epic, **Color Grading and the Filmic Tonemapper** ([documentation](https://dev.epicgames.com/documentation/unreal-engine/color-grading-and-the-filmic-tonemapper-in-unreal-engine)) | Lighting can exceed display range; highlight compression must preserve its shape before final clipping. | Remove the premature ceiling in the V2 BRDF path and apply a continuous shoulder after lighting calibration. Our simple curve in the legacy display space is neither ACES nor a complete HDR radiometric pipeline. |

These connections to Render v2 are engineering proposals. The papers demonstrate their methods under their own conditions, not the quality of our integration.

## Integration point in existing code

The initial defect was not merely the absence of a new noise function. In `web/terrain_render_v2.js`, `v2Fragment` added `form` and `swardTone` to `m.albedo`, then called `applyForest` with a `guide.open` lacking that detail. Recoloring ground and litter could therefore restart from a smooth color. The crown-free path also reblended open ground.

The correction now passes explicit ground modulation into open-ground and litter composition in NE and Magna. Crown normals follow macro terrain without inheriting mineral ground noise. The isolated GPU check verifies that detail changes bare ground and gaps while leaving an opaque crown unchanged; real-tile captures remain necessary to assess appearance. Unresolved slope variance already contributes to ground roughness: this is a shader approximation, not a claim of complete LEAN implementation.

Real recapture then revealed another defect: directly clamping `detail.x * 10` to ±0.08 saturated relief at 8 mm although the first band has 17 cm amplitude. A continuous material became almost two-tone. The correction first bounds normalized form to ±1, then applies a small material-dependent albedo amplitude. It retains the spatial field and filtering; it adds no new noise to conceal the previous one.

Mineral provinces now enrich the inherited mean through a ratio bounded relative to a neutral province: at most ±14% on fully exposed rock, less than ±5% on mineral soil without grass, attenuated by organic matter and snow. Sediments and beach are composed afterward; canopy remains independent. This variation no longer replaces the same soil palette between 1.5 and 6 m/pixel.

Integration should carry a complete ground material—linear albedo, normal, roughness and occlusion—through crown composition. Areas without a vegetation hit must retain that material. Shadows, litter and moss must modify it through explicit masks without replacing all ground with a constant color. Rock detail must not be reused as small-grass silhouettes.

## Proposed construction

1. **Stable physical support.** Anchor fields to coordinates of the terrain actually displayed, independent of tile origins and zoom. Share this contract between Neural Earth and Magna Urbis. Treat projection, terrain and texture together to avoid another isolated polar distortion.
2. **Material blending.** Separate rock, mineral soil, sand, organic soil/litter, moss and snow. Derive weights from existing pedology, moisture, cover, slopes and deposits. Normalization must retain a consistent mean across distant and close views.
3. **Known detail scales.** Use narrow bands for analytical detail or stochastic sampling of material examples. Orient patterns only when a relevant physical direction is available. Measure GPU cost before choosing full Gabor at every pixel.
4. **Explicit filtering.** Retain a band while resolved, then integrate its mean and variance. At 0.625 m/pixel, centimeter grains cannot be represented individually: they contribute to aggregate appearance. Large resolved details can remain visible in vegetation gaps.
5. **Vegetation above ground.** Compose grass, shrubs and canopy from actual coverage. Preserve material and normals in visible ground portions; restrict ground-grain contribution to the surfaces it represents, without projecting it onto crowns.
6. **Continuous seasons.** Keep the same positions and random seeds through the year. Evolve moisture, litter, senescence and snow with continuous parameters. Patterns should change state without moving or regenerating arbitrarily.

By-example synthesis remains compatible with real-time procedural rendering but requires small examples and preprocessing. A fully analytical path remains possible. Choose using real tiles and a measured GPU budget; neither technique justifies replacing the generated DEM with artificial terrain for the audit.

## Expected validation

Revisit the same generated sites: dense tropical forest with gaps, temperate forest and litter, grassy steppe, sandy desert, incised arid rock and partial snow. Compare identical framing at 40 km/10 km/2.56 km/640 m, then run through the zoom. Together check visible ground under discontinuous cover, stable mean colors, absence of camouflage at intermediate zoom, retained relief and rendering cost. Old synthetic grids are not part of this visual validation.

# Forest gaps: distinguish the continuous field from its classification

Real-render review identified another source of camouflage: the inherited population field turns a 100–400 m FBM into almost binary openings. Correct radiometric blending of its surfaces makes the defect visible; it does not correct the gap generator. Global blur or a color change is therefore not a correction to their distribution.

[Asner et al. (2013)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0060875) analyze continuous canopy-height distribution using several vertical thresholds and distinguish floodplain and upland contexts. [ForestGapR (Silva et al., 2019)](https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13211) separates detection, size, spatial aggregation and gap evolution. [Central Amazon UAV/field observations (2023)](https://bg.copernicus.org/articles/20/3651/2023/bg-20-3651-2023.html) also show that measured distribution depends on the minimum threshold chosen. These works do not prescribe our shader: they motivate distinguishing continuous density, structural gaps and their filtered appearance.

The progressive V2 profile under test retains spatial field, climate and physical exclusions while widening its coverage transition. This is a bounded appearance correction without mortality, succession or storm simulation, and without claiming to reproduce these studies' empirical distributions. Older Render3D profiles remain separate controls.

## Saturation of dense stands

The next preflight revealed another source of large gaps: directly multiplying cover percentage by stand noise. Symmetric variation clips on the full-cover side but digs strongly into the negative side. Constant-cover ablation removes the DFC pattern; removing bog does not. Initial hypotheses of lighting or snow alone are not confirmed.

The candidate under evaluation carries variation in effective density `tau=-log(1-cover)`, then reconstructs `cover=1-exp(-tau)`. The exponential relation to gap fractions appears in [Chianucci et al. (2014), canopy photography analysis](https://iforest.sisef.org/contents/?id=ifor0939-007). That publication distinguishes crown cover, foliage cover and porosity: we do not equate them or claim to estimate LAI. The proposed transfer is an appearance policy inspired by density saturation, without empirical forest calibration or succession simulation.

## Coastal water: depth and clarity are distinct

Real preflight revealed another loss of variation: uniform extinction of 0.15 per meter removes almost all seabed at 30 m, even in clear sea. The 150–3,000 m bathymetric gradient does not restore shallows. [NASA's MODIS Kd490 product](https://modis.gsfc.nasa.gov/data/dataprod/kd_490.php) and [NOAA CoastWatch's explanation](https://eastcoast.coastwatch.noaa.gov/cw_k490.php) distinguish depth, attenuation and water-column clarity. Exponential attenuation is therefore an appropriate mechanism; their spectral Kd490 product is not the directly measured coefficient of our RGB shader.

The marine candidate at 0.035 per meter, with inland water unchanged at 0.15, was compared on real 40 and 10 km coastal squares. It retains a graduated shallow-water band without recoloring land or replacing the whole sea with turquoise. This remains an appearance policy by water class, without chlorophyll or suspended-matter data or local optical calibration. Evidence and exact override: `output/render-v2-floor/marine-optics-causal/clear-water/`.

# Render v2 — multiscale procedural satellite landscape

[Français](render-v2_FR.md)

Date: 2026-10-22. Implementation specification for Neural Earth and Magna Urbis.

## Objective and authorized scope

Create a new **Render v2** layer (`render-v2`, with explicit compatibility if an API requires `Render_v2`) representing a natural satellite landscape from the first Orogen overview to close views. Deliver it in Neural Earth and the neighboring `city_generator` project, in Terrain Bench/the Rust bridge and the existing perspective view. Retain current layers and settings as alternatives; do not change the default engine or style.

User clarification: separate **Orogen**, **Coarse** and **Base** views/processes are permitted, with fine refinement from decoders. The latest instruction leaves the technical choice to the implementation (“do what works best”): separation is an option, not a requirement to create three new views. Choose transitions that yield the best result with available data; do not force the same shader or definition across stages, or confuse procedural detail with learned physical detail.

Agriculture is excluded: no generation of fields, crops, parcels or agricultural calendar. Existing building and water footprints remain vegetation exclusion constraints. Do not modify neural weights, physical DEM, hydrological routing, upstream revisions or the Magna Urbis submodule to achieve an appearance effect.

The desired result is the spatial variety of a satellite image: mineral provinces, slopes, valleys, deposits, vegetation mosaics, moisture gradients, ecotones and contextualized water. Fine relief must remain restrained and legible. Extra noise alone is not an improvement.

## Problems to address explicitly

1. Unconvincing small grasses, vegetation over water, boundaries cut by filtering and blur.
2. Climate/soil fields too coarse and bleeding on zoom; graphical detail without consistent local properties.
3. Snow, sand and other covers becoming high-contrast Dalmatian-like patches at some scales.
4. Palettes too uniform within a biome; arid rocks all alike instead of black, brown, beige and red provinces; the same limits for sand and grassland.
5. Insufficiently natural forest stands, clearings, density and boundary transitions.
6. Missing structured sea ice, lake ice and plausibly located icebergs.
7. Water too uniform and too prevalent in seasonal/dry streams; lack of cross-channel variation.
8. Poorly defined first Orogen level: enrich the distance, native blocks and initial overview without waiting for fine tiles.

## Shared surface contract

Define a deterministic evaluator of environmental properties at a world position and date, used by rendering and its diagnostics. The source grid supplies regional conditions; it must not be the only source of detail after interpolation.

Inputs in explicit units: position in meters, seed, physical elevation and gradient, stable regional relief, reference-season climate and thermal gradients, regional substrate, authoritative water and any river segments, normalized annual date. In Magna Urbis, an annual preset and existing continentality form a declared adapter, not a reconstructed planetary climate.

Properties: local seasonal temperature, potential and current moisture, exposure, bedrock/minerality, sand/clay/organic fractions, apparent soil, gravel/scree, herbs, woody cover, height and functional type, snow/ice, water class and fraction. Competing covers in a given layer are bounded and normalized; substrate, cover and canopy are distinct layers.

Refinement is an appearance and microhabitat model, not a new weather simulation or geological measurement. Historical physical climate layers retain their meaning. Provide at least Render v2 diagnostics for substrate, local moisture, vegetation and snow/ice, derived from the same evaluator as the final pixels. No separately calculated decorative diagnostic.

## Spatial and multiscale consistency

Separate environmental identity from display filtering: a position retains its material and vegetation candidates when tile, camera or LOD changes. Level of detail controls visible frequencies and their integration, not a new seed or ecosystem.

Use canonical world coordinates, separating integer cell and local fraction to retain precision. On the planet, province fields must be continuous on the sphere; local detail must avoid antimeridian seams and document its metric toward the poles. Do not use a tile origin as a seed.

Refine regional climate with bounded, terrain-correlated perturbations: exposed/sheltered slopes, damp bottoms, drained ridges and regional variability. Do not invent fine rain shadows with white noise. Preserve regional averages and limit amplitudes.

Prepare and cache stable regional fields; evaluate resolved patterns in the shader. Filter according to the actual pixel footprint: unresolved frequencies go to average color/cover and roughness; resolved frequencies go to texture. No renormalization that amplifies the last octaves and creates patches on zoom.

Indicative scales, to be adapted to world units: provinces of tens to hundreds of km; slopes and mosaics of 100 m to several km; stands/ecotones from a few meters to hundreds of meters; ground elements from a few cm to a few meters. Physical amplitudes remain modest close up.

## Orogen overview and transition to close views

The new layer must appear in server/PNG overviews, native/coarse blocks, map and globe. Distinct Orogen, Coarse and Base treatments are desirable where their data differ; their small structures need not reproduce those of the decoders. Enrich initial images with province and slope variation, alluvial networks where available, vegetation gradients and correlated mineral structures. Share broad environmental identities and the calendar; reserve actual refinement of small details for decoder outputs.

Do not draw unresolved pebbles, individual trees or fake ravines in the distance. If sub-grid normal variation is added, declare it as filtered procedural appearance without DEM displacement. Improving the first level is a deliverable to verify, not an assumed consequence of the close shader.

## Rock, soils, sand and grasses

Rock: continuous, spatially coherent mineral families, including grey/black, brown, beige/limestone and iron red. Preserve this diversity while avoiding the current global neutralization. Correlate outcrops, fractures, strata and debris with the same parent material. Orient patterns with relief and province; avoid a universal horizontal sine wave.

Soils: continuous mixtures, texture and moisture, moderate contrast. Lower-slope deposits and alluvium only with available relief/water factors. Declare proxies when there is no sediment transport.

Sand: mineral variety and restrained oriented traces/ripples; no large high-frequency color islands. Large dunes only when their shape is compatible with the data or explicitly procedural. Transition to gravel and soil according to environment.

Grasses: fine mats and low tufts only when resolved; variations of green, straw, dryness and visible soil according to moisture, temperature, substrate and season. Avoid shrunken shrub crowns as grass blades. Normal detail must remain soft.

## Vegetation and ecotones

Build a mosaic of functional communities: broadleaf, conifer, tropical, dry forest, savanna/shrubs, grassland/heath, wetlands where inputs justify them. Vary height, density, hue and grouping at multiple correlated scales.

Keep deterministic candidates and stable average cover. Define boundaries through ecological suitability and irregular ecotones, with changes in stature and density rather than binary cuts or spread-out blur. Do not artificially open a humid tropical forest using generic clearing noise.

Crowns retain their attributes at their fixed center. Grasses, trunks/candidates and ground shadows must respect water and occupancy masks; exclude submerged centers and composite water cover last. A possible overhanging crown cannot become a tree rooted in water. Banks use true hydrological geometry/width, not the minimum cartographic width.

Connect Magna Urbis's existing riparian galleries and adapt their envelope to the Neural Earth network. Control width, height, density, transition and wet ground by context; no uniform halo around all water.

## Shared season

Add a visible **Season** slider for Render v2, with readable month/period, a continuous annual cycle and persistence in URL/recipe/session. One season state drives all v2 components and, where appropriate, the displayed hydrology month; historical modes retain their own settings.

Changing it does not restart terrain generation or inference. Separate stable fields and seasonal state to avoid general reconstruction at each step. Ensure December/January continuity and opposite seasons in the hemispheres.

Effects: grass green-up/senescence, leaf-out and broadleaf colors, conifer persistence, tropical dry season if climate indicates it, soil moisture, snow, ice and seasonal water. Trees do not redistribute when the slider moves. Reduced leaf cover must affect opacity/structure and shadows, not just hue.

Snow: cover consistent with temperature, supply and retention, exposed rock on steep slopes and gentle depth/appearance variation. Prefer relief-related sheets and fine edges to high-contrast isotropic patches. A deterministic seasonal response with approximate lag is acceptable, but do not claim to simulate historical accumulation.

## Water and cryosphere

Respect hydrological authority and its surface fraction. Keep dry depressions below sea level as land. Lake, river and ocean water share a family of materials contextualized by depth, sediment/substrate, environment and ice cover.

In a resolved channel, vary shallow water, visible bed, deeper water and damp edge across its width. Exposed banks remain land. Use available depth and cover; a visual approximation does not alter bathymetry.

A wadi can be dry and show its bed: use monthly discharge/regime where available. An absent series or uniform fallback discharge does not prove a dry season. V2 must not subsequently cover this dry channel with a historical blue overlay. Preserve permanent rivers supplied from upstream, even in a local desert.

Lake ice: mask consistent with season/temperature and water type, shoreline transition, gentle variations, cracks only when resolved. River ice requires more caution with flowing water. Sea ice: seasonal cover, plates/floes and channels at resolved scale, white/bluish average at a distance. Do not apply freshwater thresholds to the ocean without distinction.

Icebergs: stable waterborne elements near plausible coastal glacier sources, bounded density and physical size, satellite-view-appropriate material silhouettes/shadows. Do not seed icebergs in every cold ocean or equate sea-ice fragments with icebergs. Without simulated calving or currents, present this distribution as a localized procedural proxy; do not claim physical dynamics. Their display must exist in the implementation, not merely in this document.

## Technical integration

Neural Earth: separate v2 module, explicit connection to selectors, overview, CPU/PNG and WebGPU modes, coarse blocks, globe, links and sharing. Reuse existing transport and data; add only necessary channels. V2 diagnostics must also work at close zoom. Preserve Render/Render3D/Brush and their outputs.

Magna Urbis: adapter to the same contract, WebGL2 shader and Terrain Bench/Rust bridge integration; reuse in perspective and export/raster with declared behavior. Do not introduce a competing engine or runtime network dependency on the other repository. Preserve applied footprints and water even if the corresponding overlays are hidden. Add the English method note and help link as requested by AGENTS.md.

Seek to share reproducible constants/fixtures/contracts where possible. WGSL/GLSL/Python differences must be checked against common inputs; do not copy three diverging models without verification. Do not promise exact parity if the climate adapter does not supply the same data.

## Validation and delivery criteria

Checks must exercise the actual compiled shader, not just search strings in its source. Reuse existing infrastructure; no large testing framework.

- New layer selectable in both applications; slider and sharing restore date and rendering; old modes remain accessible and defaults preserved.
- Captures at multiple scales of an arid mineral region, temperate grassland/forest, humid forest, sandy coast, snowy mountain and cold lake/sea; global Orogen samples and close views required.
- Zoom series at critical scales: no sudden high-contrast snow/sand patches, no blurred stand cutoffs and no unjustified loss of mean color.
- Water: river/lake/coast, dry depression, permanent desert river and dry seasonal bed; vegetation excluded from water, final water not reintroduced by historical overlay.
- Winter/summer: different grass and broadleaf foliage, responsive snow/ice; sea ice and icebergs visible in appropriate contexts and absent from warm contexts.
- Seed, tile, LOD and December/January continuity; no drift when the camera moves; antimeridian/pole checks or explicit limits if unresolved.
- Diagnostics: cover, moisture, substrate and snow/ice values match rendering decisions at the same point and date.
- Run targeted tests, shader compilation, Magna typecheck/build and browser console checks. Measure preparation, transfer, bake and frame separately; do not infer real-time performance from isolated shader cost.
- Document commands, captures, results, measurements and remaining limits in a report. Name every unmet requirement; do not present a palette change or Render3D alias as a complete delivery.

## References and nature of adaptations

References guide rendering adaptations; they do not imply implementation of their complete physical models.

- [PBRT, Noise](https://pbr-book.org/3ed-2018/Texture/Noise): deterministic consistency and frequency control of procedural textures. Adapted to footprint filtering and material patterns.
- [Filament, Physically Based Rendering](https://google.github.io/filament/main/filament.html): separation of material and lighting, normals and roughness. Reuse existing shader principles without imposing an engine.
- [NSIDC, Sea Ice](https://nsidc.org/learn/parts-cryosphere/sea-ice): sea ice as frozen seawater, seasonal extent; distinguish sea ice from land ice. V2 remains a seasonal appearance without coupled ocean/thermodynamics.
- [NSIDC, Science of Sea Ice](https://nsidc.org/learn/parts-cryosphere/sea-ice/science-sea-ice): morphology and freezing differences between fresh and salt water; do not treat lakes and ocean as one environment.
- [NOAA, What is an iceberg?](https://oceanservice.noaa.gov/facts/iceberg.html): glacial origin of icebergs. Sea ice alone is not a calving source.
- Existing code: `web/terrain_renderer.js`, `web/terrain_render3d.js`, `backend/terrain_render.py`, `backend/terrain_pedology.py`, `backend/terrain_hydrology_render.py`; in Magna Urbis: `rust/bridge/terrainGround.ts`, `terrainForest.ts`, `terrainForestRiparian.ts`, `terrainForestWater.ts`, `web/src/ui/terrain3d.ts`.

## Implementation status

This document specifies the authorized target. At its creation, Render v2 had not yet been implemented. The implementation report must distinguish features actually delivered, validation performed and limitations. The “Google Earth” visual target describes an aesthetic direction, not measured equivalence to real satellite imagery.

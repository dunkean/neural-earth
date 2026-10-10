# Neural Earth surface materials

**Render** combines soil, vegetation, exposed rock, water and snow on the current physical DEM. **Soils** shows shaded substrate and exposed rock. **Pedology** shows the regional soil composition of the source world, without vegetation, snow, lighting or contours.

Regional provinces are evaluated at half the source atlas resolution (about 40 km on the default world) from climate (moisture index, warmth, frost), regional relief (ruggedness, altitude, lowland basins) and parent-material noise, then resampled bilinearly. Soils use the same mixtures for sand, clay and humus; exposed-rock tint varies with parent noise and climate. Sandy, calcareous, clay-rich, ferrallitic, organic, podzolic and mineral mixtures provide smooth substrate colors. This is a procedural appearance model, rather than a soil survey or geochemical simulation.

Vegetation responds to aridity, season and the temperature-dependent tree line. Valleys favor moisture and small stands; ridges tend to remain exposed. Slope and terrain position control rock exposure. Snow responds to temperature, precipitation, sun exposure and slope retention; water color follows depth and seasonal sea ice.

Deterministic CPU/WGSL noise uses shared geographic coordinates and seed. Octaves fade as their wavelength approaches the sampling footprint and disappear when unresolved. Local variations diminish at distance, leaving climate and relief to define broad structures. Fine strata and vegetation texture are appearance detail, without a claim of additional trained DEM resolution.

The annual-cycle control runs from northern winter to summer and back; southern appearance follows its own climate. Forest cover, variation, moisture, rock slope, snow amount and color controls apply immediately and persist in the URL. Reset restores exact RGB defaults.

Coarse material colors are baked per block, mode and settings; frames reuse filtered albedo with terrain lighting. Fine tiles also cache GPU colors. Appearance sources have a separate identity from neural heights, so material-only edits do not rerun the networks. The full appearance transport contains 50 planes, including physical climate/biomes, substrate, metadata and pedology; older 46-plane transport remains readable.

Implementations: [CPU reference](../backend/terrain_render.py), [CUDA mirror](../backend/terrain_render_torch.py) for server PNG tiles and overviews (`TERRAIN_RENDER_DEVICE=cpu` forces NumPy), [WebGPU renderer](../web/terrain_renderer.js), [soil model](../backend/terrain_soil.py), [pedology](../backend/terrain_pedology.py) and [material controls](../web/terrain_render_controls.js).

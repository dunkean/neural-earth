# Neural Earth biome layer


### What the visible layer represents

The **Biomes** entry in the current layer menu selects `orogen-biomes`. It combines a Köppen climate class with an altitude-dependent color treatment. There are **30 land classes plus the ocean**, with additional rock and snow colors.

Orogen supplies the shared climate for the Noise, Tectonic and Custom relief generators. The layer represents climate-associated landscape colors. It does not independently simulate vegetation, forests, grasslands, wetlands, soils, glaciers or hydrology.

### Construction pipeline

1. **Generate relief and apply the selected erosion.** The resulting source terrain feeds the climate stage. Stages can also be generated independently; generating the complete chain updates relief, erosion and climate together.
2. **Calculate climate on a spherical mesh.** Orogen computes winds, ocean currents, precipitation and temperature for two seasons. Latitude, altitude, ocean influence and rain shadows contribute to these fields.
3. **Assign a Köppen class to each mesh region.** Temperature and precipitation determine the class. The two simulated seasons act as proxies for monthly climate; this is an approximation of Köppen classification, not a classification from a complete monthly time series. Local warm and cold seasons are identified from temperature so precipitation subtypes work in both hemispheres.
4. **Sample the class and appearance settings for each physical tile.** The class supplies a base color and alpine/snow thresholds. Newly exposed neural land inherits the nearest source land class; ocean pixels follow the current DEM sign. The nearest-land extension respects longitude wrapping on spherical worlds.
5. **Calculate the color from the current relief at every LOD.** Darkening, alpine rock and snow transitions are evaluated on the displayed physical DEM. At LOD 0 and finer, a slope override is applied after snow and vegetation. The class atlas remains **2,048 × 1,024 pixels**, in an equirectangular projection; compact sampled base colors and altitude lines are interpolated for rendering. The separate **Köppen** layer keeps categorical sampling and its diagnostic palette. The historical precomputed RGB atlas is retained, but the visible Biomes layer no longer renders it directly.

The main classification rules, with default thresholds, are:

| Group | Meaning | Main criterion |
|---|---|---|
| A | Tropical | Cold-season temperature ≥ 18 °C, unless classified as arid |
| B | Arid | Annual precipitation below a temperature- and seasonality-dependent threshold |
| C | Temperate | Cold season ≥ 0 °C, warm season ≥ 10 °C; outside tropical and arid classes |
| D | Continental | Cold season < 0 °C and warm season ≥ 10 °C; outside arid classes |
| ET | Tundra | Warm-season temperature ≥ 0 °C and < 10 °C |
| EF | Ice cap | Warm-season temperature < 0 °C |

Polar classes are assigned before the aridity test. For the other bands, the aridity threshold is `max(0, 20 × annual_mean_temperature + seasonal_addition)` in mm/year. The addition is 280 mm when at least 70% of rain falls in the local warm half-year, 0 mm when at most 30% does, and 140 mm otherwise. Below half this threshold, the class is desert; otherwise it is steppe. Arid climates are hot or cold according to whether annual mean temperature reaches 18 °C.

Within C and D, `f` means no dry season, `s` a dry summer and `w` a dry winter. Temperature letters distinguish hot summers (`a`), warm summers (`b`), short/cool summers (`c`) and extreme continental winters (`d`). These distinctions use seasonal proxies and an interpolated shoulder-month temperature.

### Default base colors

These are the **Biomes** palette colors before elevation adjustments and spatial interpolation. Hex values are rounded from the configured floating-point RGB values; they are not guaranteed to equal the final rendered pixel values.

| Code | Category | Base color |
|---|---|---|
| Af | Tropical rainforest | Deep green `#0D4D0D` |
| Am | Tropical monsoon | Dense green `#145412` |
| Aw | Tropical savanna | Yellow-green `#6B802E` |
| BWh | Hot desert | Sandy tan `#D1B880` |
| BWk | Cold desert | Gray-brown `#998C7A` |
| BSh | Hot steppe | Dry gold `#B89E4D` |
| BSk | Cold steppe | Olive-tan `#8C8552` |
| Cfa | Humid subtropical, hot summer | Medium green `#2E6B1F` |
| Cfb | Oceanic, warm summer | Rich green `#1F611A` |
| Cfc | Subpolar oceanic | Dark green `#1A471A` |
| Csa | Mediterranean, hot summer | Khaki `#737A38` |
| Csb | Mediterranean, warm summer | Olive green `#667333` |
| Csc | Mediterranean, cold summer | Dark olive `#596633` |
| Cwa | Subtropical, dry winter, hot summer | Medium green `#337024` |
| Cwb | Subtropical highland, dry winter | Green `#26661F` |
| Cwc | Cold subtropical highland, dry winter | Dark green `#1F521A` |
| Dfa | Humid continental, hot summer | Forest green `#1F5C14` |
| Dfb | Humid continental, warm summer | Dark forest green `#1A5214` |
| Dfc | Humid subarctic | Spruce green `#0F3814` |
| Dfd | Subarctic, extreme winter | Very dark green `#0D2E12` |
| Dsa | Continental, hot dry summer | Olive-brown `#61612E` |
| Dsb | Continental, warm dry summer | Dark olive-brown `#59592B` |
| Dsc | Subarctic, dry summer | Dark green `#143814` |
| Dsd | Extreme subarctic, dry summer | Very dark green `#0F2E12` |
| Dwa | Continental, dry winter, hot summer | Forest green `#245C1A` |
| Dwb | Continental, dry winter, warm summer | Dark forest green `#1F5217` |
| Dwc | Subarctic, dry winter | Spruce green `#123814` |
| Dwd | Extreme subarctic, dry winter | Very dark green `#0D2E12` |
| ET | Tundra | Earthy brown `#595238` |
| EF | Ice cap | Blue-gray white `#C7CCD6` |

Additional elevation colors:

- **Alpine rock:** brown-gray `#6B6152`.
- **Snow:** bluish white `#EBEDF5`.
- **Ocean:** a depth gradient, from approximately `#0A0F4D` in deep water to `#4D6B99` just below sea level. It is handled separately from the land palette.

### Elevation adjustments

| Climate group | Alpine rock transition starts | Snow transition starts |
|---|---:|---:|
| Tropical | 3,500 m | 5,500 m |
| Arid | 3,000 m | 5,000 m |
| Temperate | 2,000 m | 3,500 m |
| Continental with hot or warm summers | 1,500 m | 3,000 m |
| Subarctic | 800 m | 2,000 m |
| Tundra | 400 m | 1,500 m |
| Ice cap | Disabled | 500 m |

Below 200 m, the base color darkens by up to 7%, with maximum darkening at sea level. Between 200 m and the alpine line, it progressively darkens by up to 15%.

Above the alpine line, the color blends toward rock using a squared transition factor. The transition spans the interval between the alpine and snow lines, or a 2 km fallback interval when the snow line is not above the alpine line. Above the snow line, a separate squared blend adds snow. By default, the full snow color is reached **2,500 m above the snow threshold**. The threshold marks the start of the transition, not immediate complete snow cover.

### Slope rule and LOD behavior

The visible Biomes layer now follows the same physical terrain LODs as Relief, including neural coarse cells, latent previews, the 30 m DEM and experimental LODs −1 to −3. Its continuous refinement and forced LOD controls work like those of Relief. Other climate diagnostics retain their fixed LOD 9.

At **LOD 0 and finer** (sample spacing ≤ 30 m), terrain steeper than **40° by default** is rendered as rock, even in snowy or vegetated areas. Slope is measured with central differences of the current DEM in physical metres, using the tile halo. It does not use blurred hillshade or lighting exaggeration. The ocean mask takes precedence over the slope rule.

The threshold can be changed from **Rendering → Biomes → Pente rocheuse (°)**, between 1° and 89°. It is persisted as `biome_slope` in the URL. GPU tiles recolor without a neural inference or DEM upload; PNG rendering reuses the physical terrain cache. CPU and WebGPU use the same palette and altitude settings.

Biome colors, including snow, are multiplied directly by the terrain's hillshade intensity. This applies to physical tiles, native coarse rendering and the initial worldwide overview. Light is independent of the altitude palette: the former conversion from relief RGB luminance, multiplied by 1.8 and clipped, saturated on white summits and erased their shadows. The direct hillshade preserves the distinction between illuminated and shaded snow slopes while respecting the existing strength, ambient light, contrast and sunlight controls. The default snow base color remains `#EBEDF5`.

The transport retains the original five physical climate channels and appends sixteen display planes: base RGB, alpine/snow lines and appearance constants. The response advertises `X-Terrain-Climate-Layers: 21`. These display planes do not alter neural conditioning or the stored five-channel physical climate cache.

### Remaining limitations relevant to future improvements

- **Climate classes remain global.** Local terrain changes drive the surface appearance and coastline, but do not run a new climate simulation or assign a new Köppen class at every fine pixel.
- **Climate classes are used as landscape proxies.** Vegetation coverage, soil, wetlands and local ecological variation have no separate simulation in this layer.
- **Many categories use similar greens.** This suits the satellite-style palette, but makes climate classes difficult to distinguish visually.
- **Two biome implementations coexist.** The older `biomes` mode is hidden in the visible menu. It blends temperature, precipitation, seasonality, elevation, rock and snow continuously and follows the current neural terrain. It does not use the 30-class Orogen biome palette described above.

Colors, classification thresholds, altitude lines and transition settings are exposed through the climate controls. They take effect when the climate stage is regenerated; independently retained stages can still reflect previous inputs.


## Implementation

See [terrain_biomes.py](../backend/terrain_biomes.py), [Köppen classification](../native/orogen/vendor/koppen.js), and [climate defaults](../native/orogen/climate-parameters.json).

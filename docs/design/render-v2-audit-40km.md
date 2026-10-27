# Render v2 audit on generated tiles

[Français](render-v2-audit-40km_FR.md)

Date: October 23, 2026. Status: baseline before corrections, 24 real sites captured; further validation in preparation.

The rendering has not passed visual validation. Baseline comparisons show a loss of brightness when zooming in, covered beaches, and excessively fragmented snow at intermediate levels. The observations and figures below describe that baseline, not measurements after correction. The shaders have since changed and require another paired inspection.

## Acceptance threshold

Render and Render3D set the quality threshold for existing features. Render v2 must preserve physical beaches, sediment banks, terrain shadows, readable slopes and visible ground detail, then improve materials, seasons and transitions. A new effect does not compensate for a missing beach or erased relief. Each comparison uses the same terrain, camera and lighting settings.

The final gallery of 50–100 situations at five zoom levels will be produced after satisfactory corrections. Intermediate captures support correction work and are not that final delivery.

## Validation material

Only captures of the world generated in the application are used in this audit. The first analytical-terrain attempt was discarded following user feedback: it could not assess the requested landscape. Its images and figures are excluded from the coverage below.

The real-world matrix crosses six climate families—tropical forest, temperate forest, steppe, desert, taiga and tundra/polar—with four landforms—plain, hills, plateau/escarpment and mountains. Actual codes and elevations are measured on decoded tiles; a filename alone cannot validate a biome or landform.

Each site is framed at 40 × 40 km and then at the same coordinates at three closer levels. The 1024 × 1024 pixel captures give:

| Extent | Display scale | Terrain data |
|---|---:|---|
| 40 km | 39.0625 m/px | LOD0 decoder, physical samples at 30 m |
| 10 km | 9.765625 m/px | Same LOD0 tiles |
| 2.56 km | 2.5 m/px | Same tiles; procedural details visible |
| 640 m | 0.625 m/px | Same tiles; fine procedural texture |

Zooming enlarges the real decoded data and reveals shader detail. It does not produce a physical 62.5 cm DEM. This check is supplemented by actual intermediate levels of the generation pipeline.

For each zoom: Render v2 in July, Render3D in July, and Render v2 in January. The camera, terrain and lighting are identical, with no HUD or minimap. July and January are dates; seasons are reversed between hemispheres. Checking both dates does not validate the full annual animation.

Provenance: seed `7985403534069300490`, profile `orogen--g46261a320a644f43189ea849`, world `f5cddc89751589636d1052bfa46529f6f0a400b1fd1b176a35dbb14ce3590a1a`. The generation share is `gHY4ZFzncjegep8JTpf`. Reports retain the URL, camera, resolution and stage actually displayed, world identity, browser errors and rendering-source fingerprints.

## Defects already confirmed on the generated world

### Darkening when zooming in

The real tropical forest loses substantial brightness in V2. On the same close 1.28 km framing at 2.5 m/px, the forest texture proxy gives mean display luminance of 31.56/255 versus 57.89 in Render3D; 83.86% of proxy pixels are below 40/255 versus 12.18%. Temperate forest gives 74.41 versus 90.68. These measurements use paired real PNGs without HUD; the proxy selected on Render3D is not a semantic mask and does not separate albedo from shadow.

The real matrix also shows sand darkening on approach. Exposure and color consistency between distant and close materials must be addressed before adding more textures. Render3D is a comparison, not satellite ground truth.

Evidence: [real canopy measurements](../../output/render-v2-40km/real-detail/audit-real-canopy.json), [measurements at four zooms](../../output/render-v2-40km/real-matrix/readonly-image-metrics.json).

### Beaches still covered

On the real `tropical-rainforest-low` and `warm-coast` plates, Render3D shows sandy strips that V2 reduces to a narrow edge or covers with green. Relief and coast are identical. The code blends beach sediment into open ground, then blends the canopy back in without removing its cover over the beach: this is consistent with the observation but requires an isolated correction and a new capture for confirmation.

There is no reliable beach-width metric from a simple RGB threshold at this scale: shallow water, sand and light ridges can be confused. The finding rests on paired images.

Evidence: `output/render-v2-40km/real/{tropical-rainforest-low,warm-coast}-{v2-july,render3d-july}.png`; `web/terrain_render_v2.js`, function `v2Fragment`.

### Dalmatian effect at intermediate levels

In real `alpine-tundra`, `taiga-low` and shared `user-compact-pack` coastal views, V2 produces broken snow sheets and grey-green/white patches that compete with the relief. The defect is especially visible at levels 3 and 4. It should not be confused with normal DEM shadows or reduced to rock grain alone.

The decoded matrix reveals a second related effect even without changing tiles: at the 10 km framing, `dfc-rolling` taiga and the temperate `cfb-rolling` site show large cover/clearing patches. On approach these resolve into forest stands and gaps. The readability defect therefore also occurs during canopy reveal; removing ground-grain noise alone would not fix it. Real steppe also shows this behavior, while inspected deserts mainly show hue and exposure drift.

Counting black or red pixels is insufficient: a render can contain no absolute black pixels and still be too dark, and noise can create distracting patches with otherwise natural colors.

The intermediate views are actual application outputs at the following stages:

| Requested LOD | Stage actually displayed | Physical source resolution | Wide check extent |
|---|---|---:|---:|
| 3 | latent, quality 3 | 240 m | 153.6 km at 300 m/px |
| 4 | coarse, quality 4 | 7,680 m | 307.2 km at 600 m/px |
| 6 | coarse, quality 4 reused | 7,680 m | 1,228.8 km at 2,400 m/px |

These wide views are not presented as 40 km plates. A second check at positions provided by the user compares levels at a constant 40 km framing to distinguish a data change from a scale change.

The coarse path evaluates climate directly; it does not use the same RGBA8 auxiliary buffer as the detailed path. It would therefore be wrong to attribute all large white clusters at levels 4/6 to 8-bit thermal quantization. Snow retention on an enlarged 7.68 km DEM, its slopes and interpolated climate need examination on this path. Further analysis appears in the open-ground supplement.

On real PNGs, the standard deviation of the luminance band `Gauss(2 px) − Gauss(8 px)` reaches 10.30 in V2 versus 5.43 in Render3D for the alpine site at level 3; 4.29 versus 2.79 for the shared coast. At level 4 that coast gives 2.71 versus 0.61. The hot desert control gives 2.00 versus 1.93 at level 3 and 0.45 versus 0.46 at level 6. This supports a cover/snow defect in the observed cases; the image metric alone cannot distinguish shadow, noise and material. [Measurements of real intermediate views](../../output/render-v2-40km/audit-real-mid-metrics.json).

Evidence: [real intermediate sheets](../../output/render-v2-40km/mid-sheets/), reports `mid-lod3`, `mid-lod4`, `mid-lod6` and `transitions-lod*`.

### Sea ice still looks artificial

On the exact shared coast, channels still form a very uniform network of small cells. At intermediate distance the whole resembles a graphic pattern; at the coarse level channels disappear. Neither scale yet produces a credible, continuous material. Large, nearly uniform marine surfaces reinforce the plastic appearance. The winter 40 km capture confirms the defect; the presence of an ice layer does not validate its quality.

Evidence: `output/render-v2-40km/real/user-compact-pack-v2-july.png` and [comparison of its intermediate levels](../../output/render-v2-40km/mid-sheets/user-compact-pack.png).

## Coverage and cross-analysis

The 24 sites were captured at four zooms and in three styles/dates, for 288 real images. After documented reassignment of the two arid landforms, 15 combinations meet the strict biome and landform thresholds; two other mountain sites represent mixed natural transitions with at least 70% of the requested climate family. The remaining seven categories need replacements. This classification result does not imply rendering acceptance. Selection on the 19.53 km source atlas only identifies candidates; it does not validate their landforms or biomes after decoding.

Gallery: [real tiles at four zooms](../../output/render-v2-40km/real-matrix/index.html). Machine report: [captures, provenance and DEM measurements](../../output/render-v2-40km/real-matrix/report.json). Cells whose measurements contradict the requested category remain marked as such and are not counted as validated combinations.

## Proposed correction order

| Priority | Change to check | Criterion on the same real tiles |
|---|---|---|
| 1 | Harmonize exposure and color between distant material, close ground and crowns | No massive darkening from 10 km to 2.56 km; relief remains legible under forest and on snow |
| 1 | Revisit reveal of canopy and snow fields | No camouflage patches appearing at an intermediate zoom before details; average cover and color preserved |
| 1 | Adapt coarse snow to the resolution actually available | No large clusters from enlarged multi-kilometer slopes/TPI; continuity across coarse → latent → decoder substitution |
| 1 | Remove vegetation cover from beaches retained by the renderer | Sandy strips on both real coasts remain visible, including beneath close crowns |
| 2 | Rework sea-ice structure and its scale transition | Credible channel size and contrast, less uniform material, coherent transition to coarse |
| 2 | Check mineral palettes and open ground after these fixes | No ochre-to-grey-brown jump caused solely by zoom; restrained fine grain and preserved relief |

Each correction must be recaptured on the same real selection and inspected while scrolling. Adding extra noise before fixing these transitions could amplify the observed defects.

## Limitations of this audit

This audit covers Neural Earth/WebGPU. It does not by itself revalidate the `city_generator` integration. Baseline captures force LOD0 and use 30 m physical resolution. Intermediate checks show the actual latent/coarse paths, but static images are insufficient to assess all automatic scrolling, flicker or temporal transitions. Hydrology is disabled in the reference share: the land matrix therefore validates neither all seasonal lakes/rivers nor all dry beds. Agricultural fields remain excluded.

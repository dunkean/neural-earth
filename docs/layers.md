# Neural Earth layers

Choose a layer from the menu next to the map controls. Map and globe share the selected layer and physical terrain cache.

| View | What it shows |
| --- | --- |
| Relief | Height-colored shaded terrain |
| Render | Climate-informed soil, vegetation, rock, snow and water materials |
| Soils | Exposed soil and rock, with terrain shading |
| Pedology | Regional soil-mixture provinces on an approximately 200 km source grid |
| Biomes | Source Köppen class colors adjusted to current DEM altitude, slopes and coastlines |
| Köppen | Categorical climate classification |
| Annual temperature / precipitation | Climate diagnostics with physical units |
| Seasonal climate | Northern summer and winter temperature and half-year rainfall |
| Atmospheric layers | Pressure, wind direction/speed, rain shadows and ocean influence |
| Ocean layers | Seasonal surface current direction and speed |
| Orogen layers | Plates, superplates, crust, boundaries and convergence |
| Relief formation | Initial relief, uplift, orogeny, hotspots, back arcs, folded ridges and erosion changes |
| Map styles | Parchment, Atlas, Classic, Engraving, Cadastral, Blueprint, Illuminated, Topographic, Night and Copernicus DEM |

Some diagnostics describe the source atlas rather than every fine neural pixel. Imported Custom/Noise sources have diagnostic land/water regions rather than simulated moving tectonic plates. The visible Biomes layer follows the current DEM; source climate classes do not get recomputed for every neural tile.

Rendering settings apply immediately: global or per-LOD lighting, sun direction, contours, materials and styles. Every fifth contour is major; automatic spacing follows zoom, while manual spacing uses meters. These controls reuse physical heights without neural generation.

The globe supports orbit and progressive terrain detail. A planar world disables globe view. The scale bar, cursor altitude, measurement tool and diagnostics expose physical scale. Cache and refinement controls determine when parent terrain is replaced by finer tiles.

Open **Share** and choose **Copy code** to send a single portable string. Paste it under **Open a code or link** on another installation and choose **Open view**. **Copy link** creates a link for the current server; its portable recipe can also be opened on a different installation through the same panel. Existing seed/profile/generation URLs still work.

Codes use `NE1:<generation UID>:<rendering UID> layer=relief view=map zoom=1600 position=123456,-78901 size=1280,800 lod=auto recipe=<compressed recipe>`. The 19-character generation UID identifies the seed, applied generation settings and retained stages. It stays the same when only the camera or appearance changes. The rendering UID identifies materials, global/per-LOD lighting, contours, rendering backend, interpolation, sea rules, refinement and cache controls. `zoom` is metres per logical pixel, `position` is the map center in world metres, `size` is the logical viewport in pixels, and `lod` is `auto` or a forced LOD from -3 to 11. Globe codes also include `orbit=yaw,pitch,altitude`, with angles in radians.

Keep `recipe` when sharing across installations: a UID alone cannot describe arbitrary settings to a server that has never seen them. The compressed recipe contains full settings and the dependency graph needed to rebuild retained relief, erosion and climate stages, including older parents. Saved UID recipes persist under `generated/share-recipes/` (or the configured output root). Links keep the portable recipe in the URL fragment. Newly opened codes override local generation and rendering preferences. An imported view keeps its original viewport footprint, scaling down to fit the window; **Use full window** releases that size constraint.

**Alt+click** on the map or globe adds a pin. **Click a pin** to remove it. Pins have only coordinates, stored visibly as `pins=x,y;x,y` in the URL and share code. Their positions survive reloads and portable sharing.

Sharing restores the generation recipe and view, rather than freezing an in-progress frame. Missing tiles regenerate and refinement progresses normally. Use a forced Render LOD when sharing a specific detail level. Different model versions or GPU runtimes can produce different terrain values; the Share panel reports a world-identity mismatch after import. Pixel-identical output across different hardware is not guaranteed.

See [materials](materials.md), [biomes](biomes.md) and the [architecture](architecture.md) for the scientific limits of these views.

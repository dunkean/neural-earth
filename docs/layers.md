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
| Orogent layers | Plates, superplates, crust, boundaries and convergence |
| Relief formation | Initial relief, uplift, orogeny, hotspots, back arcs, folded ridges and erosion changes |
| Map styles | Parchment, Atlas, Classic, Engraving, Cadastral, Blueprint, Illuminated, Topographic, Night and Copernicus DEM |

Some diagnostics describe the source atlas rather than every fine neural pixel. Imported Custom/Noise sources have diagnostic land/water regions rather than simulated moving tectonic plates. The visible Biomes layer follows the current DEM; source climate classes do not get recomputed for every neural tile.

Rendering settings apply immediately: global or per-LOD lighting, sun direction, contours, materials and styles. Every fifth contour is major; automatic spacing follows zoom, while manual spacing uses meters. These controls reuse physical heights without neural generation.

The globe supports orbit and progressive terrain detail. A planar world disables globe view. The scale bar, cursor altitude, measurement tool and diagnostics expose physical scale. Cache and refinement controls determine when parent terrain is replaced by finer tiles.

See [materials](materials.md), [biomes](biomes.md) and the [architecture](architecture.md) for the scientific limits of these views.

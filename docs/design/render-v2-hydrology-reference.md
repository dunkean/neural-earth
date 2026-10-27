# Render v2 real hydrology supplement

[French original](render-v2-hydrology-reference_FR.md).

This supplement remains separate from the 60 non-hydrology situations and their 900 images. It comprises six centers selected from the same world's source network: humid river, temperate river, intermittent arid river, cold continental lake with temperate summer, polar lake and closed arid lake. Each is planned at 153.6 km, 40 km, 10 km, 2.56 km and 160 m, in July V2/Render3D and January V2. The DEM remains 240 m latent at LOD3, then 30 m decoder at LOD0; the last zoom enlarges that DEM and tests the material without creating submeter geometry.

Selection is retained in `output/render-v2-hydrology-reference/manifest.json`. Network levels, areas and paths derive from the 19.53 km atlas: they are references, not validation of neural DEM shores. Monthly hydrographs are modeled, not observed. The currently selected intermittent case is BSh, with July flow and dry January in its hydrograph; it must not be presented as Aw savanna.

## Lake display and limitations

`DISPLAY_LAKES` remains disabled for historical layers. Render v2 has a separate source display grid requested by `v2_lakes=1` and `v2_lake_min_area_m2`. It selects lakes by area and retains only samples whose raw terrain is below lake level. It does not enable small Base regional basins or perform carving: the returned DEM must remain strictly identical. Emerged cells remain dry and the physical mask takes precedence over vectors, including in a lake-only view.

This bounded choice does not correct any discrepancy between source levels and neural terrain. Shores, islands and transitions must be visually reviewed at all five zooms. From native coarse LOD4, the current path retains historical vectors; V2 seasonal lake freezing must not be claimed on that path.

## Protocol

The selector refuses to build an absent network. Any builds are separately triggered once through the existing server API. The network for current identity `bd36737b…` was built in 9.6 seconds; inference counters remained unchanged. Earlier `f5cddc89…` results are historical evidence only, without receipt migration or presumed neural pixel identity.

```powershell
.venv/Scripts/python.exe tools/verification/select_render_v2_hydrology_reference.py --world-info output/render-v2-reference/world-info-bd36737b.json
$env:REVIEW_CASES='lake-temperate'
node tools/verification/review_render_v2_hydrology_reference.cjs --freeze
node tools/verification/review_render_v2_hydrology_reference.cjs --run
```

These captures use the existing server and may populate neural caches for this new identity. The runner checks frozen sources and protocol, PNG dimensions, exact camera, actual DEM source and each key's identity. It also requires `hydrology.status=ready`, `fetching=false` and `X-Terrain-Hydro-State: ready` headers on buffers actually sent to the renderer; `visible.pending=0` alone is insufficient. V2 views require four physical channels, the effective month and zero maximum cut. The Dfc lake close-up must contain real lake samples; a lake-only view must hide vector painting. Captures compose the real viewport to retain historical vectors in the Render3D control.

After the core is frozen and the supplementary campaign authorized, remove the first trial's filter to produce 90 views of six centers:

```powershell
Remove-Item Env:REVIEW_CASES -ErrorAction SilentlyContinue
node tools/verification/review_render_v2_hydrology_reference.cjs --plan
node tools/verification/review_render_v2_hydrology_reference.cjs --freeze
node tools/verification/review_render_v2_hydrology_reference.cjs --run
```

`--plan` must report six sites and 90 views. This procedure is prepared; the complete series awaits the core freeze. Global coarse preparation must finish before opening another viewer page, because opening it can interrupt that background work.

Passing these assertions verifies the protocol, not visual quality. Shore and terrain readability, water and ice appearance, and season must be reviewed separately on the images.

## First evidence for the new identity

Working Dfc pass `291ff73314ce-efa1c53c-b2e60d57` produced all 15 views and met the protocol without constituting the final frozen series. Images and report are under that ID in `output/render-v2-hydrology-reference/captures-reference/captures/`. The 160 m square contains 20 lake samples from the 30 m DEM; the V2 raster is ready for both months, without cut or overlaid vector painting. July water and a pale frozen surface in January are visible. Magnification exposes a diffuse shore at DEM scale, dark crowns and weak canopy relief under snow: these limits remain for visual review.

`real-lake-raster-bd36737b.json` independently checks this identity's real native tile and receipt: 26,395 lake pixels out of 92,416, authority everywhere, DEM and historical output strictly unchanged. The targeted point is at 278.44 m below a 281.88 m level; the neighboring source center at 284.17 m remains dry. This tile's pixels differ from the historical version; no neural equality between versions is claimed.

The Render3D control uses the composite `#viewport` capture, but its overlay here has zero painted pixels and zero lakes drawn although the network contains a lake. Historical culling uses only the source lake's first ring, outside view for this center; that source path is also outline-only at LOD3 or finer. Absence of the lake in this image therefore reflects current historical product behavior, not a capture that omitted its canvas. It cannot serve as a realistic filled-lake reference. Historical layers were not modified in this V2 adaptation.

## Regional rivers and screen edges

Real review revealed two separate problems: the fast raster could retain the source course while Render3D already displayed its regional replacement; its coverage fraction was integrated over 30 m then bilinearly interpolated. This double filtering stretched the bank over 20–30 m in close-up. The hydrology controller now exposes to the renderer exactly the same selected and projected segments as the vector control, with physical widths multiplied by the existing monthly factor. V2 water adds no minimum cartographic width.

A GPU pass rasterizes these segment quads into a screen mask. Hardware rasterization limits work to relevant ribbons; no fragment traverses the entire network. Coverage follows a box slice and preserves streams thinner than a pixel. A depth test chooses maximum coverage at unions without holes or transparency accumulation. Signed level is carried in two half-precision components (high part and residual), then reconstructed in meters. RGBA16float and 24plus depth represent approximately 12 bytes per active pixel plus the segment buffer; a ready empty mask occupies only 1×1 pixel. Projected geometry and its Float32Array are cached while network, camera and month are unchanged.

This contract replaces only fast V2 river appearance on the map at LOD 0–3. It changes neither DEM, routing, four physical channels, server nor tile identity. Lakes, sea and dry authority below sea level remain higher priority. The old population exclusion and bank supplement on the source ribbon are removed in this path; resolved roots sample the new regional mask. While waiting for network or mask compilation, when a fine V2 GPU surface is available, Render3D vectors serve as an identified fallback without duplicate old raster ribbon. A `surfaceKey` acknowledgment distinguishes that surface from the actually ready mask (`frameKey`). Without a consuming GPU surface—GPU disabled, CPU PNG or V2 shader initially unavailable—the previous raster and overlay policy remains unchanged; no new regional overprint is added to PNG. New regional precision is therefore not claimed for CPU exports. Detailed carving, globe and native coarse retain their previous paths.

The snapshot exposes `hydrology.geometryCurrent`, `queryRevision` and `currentTerrainRevision`, and `renderer.riverDisplay`: ready/fallback state, camera/network/month key, revision, actually loaded segment count, format, dimensions, memory and error. Capture waits for current geometry after the latest terrain arrivals, then the submitted and ready mask. Readback distinguishes actually wet screen-mask pixels from older class-3 samples in the physical payload. The two forms of evidence are not interchangeable.

## Targeted evidence for three rivers

Real pass `15f91fbd07b9-d1820dd7-cec2659e` contains 36 images: three rivers and a control lake, at 40 km/2.56 km/160 m, July V2/Render3D and January V2. It is retained in `output/render-v2-river-display/reference/`. All four cases pass the protocol and bounded fix review: same bends as Render3D, sharp banks, confluence without holes, dry wadi in January and unchanged lake. It does not replace the final 990-image campaign or universal validation of the water material.

River centers now lie on a real regional segment and remain fixed at all zooms: humid [-4985167.1720028585,1063774.1584098355], temperate [4915129.518241145,-5336995.5385306], intermittent arid [2978506.019704413,2705087.4193684384]. Distances from source centers are 5112.49 m, 5672.22 m and 13.37 m respectively. Fragment and parent IDs, segment, monthly flows, width, ready status and network SHA are retained in the manifest and `regional-focus-selection.json`. Previous manifest 024388ad is archived byte-for-byte without relabeling older captures. Measured temperate landscape is mixed DFB/DFC with a CFB margin; it is not evidence of pure CFB biome.

At the humid center, July 160 m readback contains 256058 wet pixels versus 468306 in January; the January wadi has zero wet pixels in a ready 1×1 mask. The control lake retains 20 lake samples and an empty river mask in both months. Standalone tests also check unions, subpixel 0.02 → coverage 0.019989, signed levels, absence of old-ribbon gaps, movement, resizing, fallback, reset and dispose.

The water material remains simple; the regional network retains its own resolution and neural-terrain connection limits. A sharp edge does not prove physical bed incision or observed levels or flows. Polar lakes in particular remain to be reviewed for angular source-ownership footprints and small islands arising from the DEM.

## Archived hydrology pass 89516, rejected version

Production version 89516 was subsequently rejected: the real wet EF plain `polar-plain-candidate-01` in the main corpus exposes too much gray soil in summer while the warm month remains below 0 °C. V2 lacks annual/permanent snow retention. The hydrology pass remains archived technical evidence; it must be recaptured after correction and no verdict transfers to a future run.

Run `89516e9f2c58-e4b29729-cec2659e` retains 90 real images, six situations and five extents with July V2 / July Render3D / January V2 views. All six protocols pass; `--sheets --require-complete` assembly confirms the 90 files and native measurements. The human verdict linked to this run in `reviews.json` is separate: four cases accepted for targeted contracts, two rejected as sufficient evidence of lake quality.

The three rivers retain matched regional traces and sharp banks; the wadi becomes dry in January. The DFC lake shows its shore at 160 m, water in July and ice in January. Small tributaries are less visible at a distance than the Render3D cartographic stroke: physical subpixel coverage is integrated without artificial minimum width. Water material remains uniform and source-network bends sometimes remain angular.

The polar lake remains a counterexample: straight boundaries and striped or square islets at the intersection of lake plane and DEM; Render3D itself displays a very wide angular connector in that region. Correct seasonal freezing does not validate that geometry. The arid lake is visible at 153.6 and 40 km extents, but its source center is dry on native terrain; 10 km, 2.56 km and 160 m views are therefore not shore controls. Dry authority is retained and the frozen manifest is not moved to hide this framing failure. Both cases remain in the gallery with explicit rejection, without claiming universal photorealistic validation.

## Arid framing prepared for the next pass

Historical manifest `e4b29729ec6c9b732faac75c4da55e6c4263829ae1eea215cf7d2464d76f2d3d` is now archived byte-for-byte in `captures-reference/manifests/`. The active manifest prepares a new **six-site / 90-image** campaign: `lake-arid-closed-wet-shore`, centered at `[-6764655,3340995]`, replaces only the dry framing of the same source lake 194. A separate 15-image supplement is no longer required. The 89516 run's report, captures and verdicts remain attached to the old manifest and are not transferred.

Selection reads only existing tiles and receipts: native depth 0.356 m, lake ownership 194, 14 samples below level and 11 emerged in a 150 m window; 208338 wet samples are counted in the initial 40 km window. Evidence is `output/render-v2-hydrology-reference/arid-lake-wet-focus.json`. Source statistics were recomputed on CPU at the new center by `select_render_v2_reference.source_stats`: 100% BWK dominance, mean source elevation 1518.82 m. Original center, record and statistics are retained in `original_source_record`; these 19.53 km statistics do not prove the shore at 30 m.

This replacement is not acceptance: a future independent review must see water and shore in the 160 m view. No new captures were made; protocol `cec2659e` is unchanged, awaiting the new production freeze and the coordinator's explicit go-ahead.

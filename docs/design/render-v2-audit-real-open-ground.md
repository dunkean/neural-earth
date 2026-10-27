# Audit of real open ground and MID views

[Français](render-v2-audit-real-open-ground_FR.md)

Reading status as of October 23, 2026. This audit makes no production changes. Earlier synthetic matrices are excluded from visual evidence and do not validate the user-facing rendering.

## Provenance and resolution

The captures in `output/render-v2-40km/real-matrix/` come from the production WebGPU renderer, the same world and genuinely generated DEMs. The `tools/verification/review_render_v2_real_matrix.cjs` runner forces the LOD0 decoder, a **30 m** source resolution, no pending visible tiles, the same center and the same lighting across styles. Each complete row has twelve PNGs: four zooms, Render v2 July/January and Render3D July. Dates are calendar dates; July is not summer in the southern hemisphere.

| Extent | Image | Screen spacing | DEM source |
| --- | --- | --- | --- |
| 40 km | 1024² | 39.0625 m/px | 30 m decoder |
| 10 km | 1024² | 9.765625 m/px | same 30 m decoder |
| 2.56 km | 1024² | 2.5 m/px | same 30 m decoder enlarged |
| 640 m | 1024² | 0.625 m/px | same 30 m decoder enlarged |

The last image therefore does not prove terrain generated at 0.625 m: its window contains only about 21 DEM intervals. It allows inspection of procedural appearance on that terrain. Classifications and landforms are reconstructed on the native DEM; initial identification at 19.53 km is insufficient to accept a cell.

## Twelve requested combinations

All twelve candidates have their twelve PNGs and were inspected, with no artificial terrain used to fill a cell. **Five landform labels are not confirmed at the 30 m decoder**: this table therefore does not validate twelve distinct combinations. The native report remains authoritative for any replacements.

| Case | Native climate/landform | 40 km | 10 km | 2.56 km | 640 m |
| --- | --- | --- | --- | --- | --- |
| BSK plain | BSH 100%; plain; 508–700 m | Low relief readable, no distinct mineral mottling | Rounded cover patches, barely visible in Render3D | Ground darker/duller and blurry masses | Scattered dark tree dots, already present but lighter in Render3D |
| BSK rolling | BSH 65.95%, BSK 34.05%; rolling; 556–909 m | Relief readable, close to Render3D | Added rounded olive patches, much less visible in Render3D | Cover in large blurry masses | Dark crown dots appear; strong contrast with earlier blurry masses |
| BSK plateau | **Label unconfirmed: native mountain**; BSK 82.55%, BWK 17.45%; 1784–2793 m | Ridges and incisions readable, styles close | Same valley network, v2 darker | Slopes in very dark/blurry masses | Crowns much paler in July than January; very smooth ground |
| BSK mountains | BSK 95.54%, DSC 4.46%; mountain; 1228–4032 m | Strong relief; January snow localized at higher elevations | Incisions retained; cover differences more visible | Large light cover/ground areas sharper and blurrier than Render3D | Dark crowns and uniform ground, visible transition from earlier masses |
| ARID plain | BWH 100%; plain; 315–526 m | Relief/grain close to comparator | Soft view without distinct mineral Dalmatian effect | Material greyer, less ochre | Almost uniform ground, close grain lost; enlarged DEM |
| ARID rolling | BWH 100%; rolling; 322–573 m | Readable relief, close to comparator | Same central incision, no strong new mineral patches | Shift from ochre to grey-brown, attenuated texture | Very uniform grey-brown; clear hue difference from Render3D |
| ARID plateau | **Label unconfirmed: native mountain**; BWK 99.93%; 1956–3230 m | Dark rocky relief close to comparator | Incisions readable, no obvious specific mineral Dalmatian effect | Browner, very smooth ground | Uniform brown; Render3D retains grain |
| ARID mountains | **Label unconfirmed: native plateau**; BWK 100%; 818–2499 m | High surfaces and incised edges, visually plausible plateau | Canyon network retained | Incisions turn reddish brown and visually flatten; Render3D keeps them grey | Little brown variation and blurry enlarged relief, no crowns |
| POLAR plain | **Label unconfirmed: native mountain**; EF 100%; 5338–7649 m | White snow/dark outcrop contrast; Render3D much browner | Same slope retention, dark valley | Slope snow with very smooth center; enlarged DEM texture | Almost uniform light-grey snow; July/January identical, no visible crowns |
| POLAR rolling | ET 100%; rolling; 145–593 m | Snowy July, relief retained; January dark ground | Same seasonal contrast; enlarged relief | July greyer than Render3D, no strong isolated mottling at center | Small cover dots visible, especially in January; do not call them EF trees: this case is ET |
| POLAR plateau | **Label unconfirmed: native mountain**; EF 100%; 3071–5694 m | Highly contrasted grey slopes/white snow, Render3D more uniformly white | Same geometry, broad grey valleys | Enlarged relief very smooth, little snow grain | Nearly uniform mid-grey; July/January identical, no visible crowns |
| POLAR mountains | EF 100%; mountain; 4581–6489 m | White snow/ridges and grey slopes; Render3D much browner | Dark valleys and slope snow distinct | Large grey surfaces, blurry enlarged relief | Very dark grey, almost uniform center; July/January visually identical |

BSK rolling confirms a cover-transition problem without demonstrating a mineral-noise error. Neither BWH case shows an obvious specific rocky “Dalmatian” effect at 40/10 km; their separate problem is material/hue change at close zoom. EF mountain snow is not uniformly white: retention depends on slope, so EF alone does not guarantee snow cover equal to one.

Paired review sheets are in `output/render-v2-40km/real-matrix/audit-sheets/`. They are reductions of real PNGs, not reconstructed terrain renders.

## Different sources at constant extent, user positions

The four directories `transitions-lod0`, `transitions-lod3`, `transitions-lod4` and `transitions-lod6` retain **exactly 40 km, the same center and 39.0625 m/px** for each of three user positions. They separate data substitution from zoom change. Effective sources are decoder30m, latent240m, then coarse7680m. The LOD6 request here returns the same quality 4 and image as LOD4; it is not a fourth independent detail level.

`output/render-v2-40km/audit-real-transitions.json` contains 36 measurements, SHA256 hashes and snapshots. The green proxy G/R>1.05 and G/B>1.15 identifies only a color, not semantic vegetation cover. The Y band is measured at the same physical sigmas of 100 and 400 m for all sources.

| Position `user-green`, July | decoder30m | latent240m | coarse7680m |
| --- | --- | --- | --- |
| Mean Y v2 | 61.16 | 62.82 | 52.70 |
| Mean G/R v2 | 1.125 | 1.121 | 1.347 |
| Green proxy fraction v2 | 71.5% | 70.0% | 94.4% |
| Y-band RMS v2, 100–400 m | 3.413 | 1.146 | 0.128 |
| Y-band RMS Render3D, 100–400 m | 3.436 | 0.862 | 0.123 |

Coarse therefore stays green: it becomes **more uniformly green and darker**. It would be wrong to say greenness disappears. Loss of structure at this scale occurs in both styles and strongly follows DEM resolution; v2 colored masses then add a further cover difference. The numbers do not prove that actual cover would be higher in coarse.

At `user-pack`, the same geographic band gives v2/Render3D **5.241/2.686** (decoder), **4.264/0.442** (latent) and **2.245/0.253** (coarse). Extra v2 cover contrast therefore persists at constant center and extent; resolution alone cannot explain the difference between styles. At `user-desert`, the pairs are instead very close: 6.173/5.986, then 0.845/0.829, then 0.112/0.112. Its coarse v2 is darker (Y87.38 versus Y110.16), but does not show comparable excess contrast at 100–400 m.

## Hue measurement at identical geography

`output/render-v2-40km/real-matrix/audit-real-open-metrics.json` retains the measurement, center, resolution, files and SHA256 hashes. The comparison uses **the same central 640 m square**, extracted with fractional bounds at each zoom; it does not compare averages over different geographic windows. At 40 km this square is only 16.384 pixels wide: its average informs hue, not detail. Y values are Rec.709 display luminances on 8-bit RGB, not physical radiance.

| Case, v2 July | Mean RGB at 40 km | Mean RGB at 640 m | Y 40 km → 640 m |
| --- | --- | --- | --- |
| BSK rolling | 127.3 / 113.3 / 75.0 | 93.8 / 88.2 / 57.4 | 113.5 → 87.2 |
| ARID plain | 147.1 / 122.2 / 85.1 | 151.0 / 134.5 / 101.0 | 124.8 → 135.6 |
| ARID rolling | 143.7 / 119.6 / 82.9 | 111.0 / 98.5 / 73.3 | 122.0 → 99.3 |

The direction of change is not universal: the arid plain becomes lighter, the arid rolling site darker. Chromatic change and disappearance of detail are observations; these averages do not separate material from its lighting response.

## Real MID views: different data and shader path

| Directory | Extent | Screen spacing | Declared source |
| --- | --- | --- | --- |
| `mid-lod3` | 153.6 km | 300 m/px | latent, 240 m |
| `mid-lod4` | 307.2 km | 600 m/px | coarse, **7680 m** |
| `mid-lod6` | 1228.8 km | 2400 m/px | coarse, **7680 m** |

These views are not 40 km plates. A full-frame MID cannot be statistically compared to a full-frame 40 km decoder view. Resolutions come from frame snapshots, not directory names.

Paired MID3 reading: hot/cold deserts and polar ice show little visual difference between v2 and Render3D; alpine tundra shows much more fragmented white snow in v2. At the 40 km decoder, the same alpine tundra is actually 78.12% ET and 21.88% EF, at 3171–4391 m: nearly snow-free Render3D is therefore not an absolute reference for climate correctness. MID4 shows much larger snow clusters and diffuse relief. Hot desert MID6 retains fine diagonal grain very similar between styles, which does not allow attributing it specifically to v2.

`output/render-v2-40km/audit-real-mid-metrics.json` measures style pairs on the same window, with SHA256 hashes and source resolutions. The band is the difference between Gaussian blurs of sigma 2 and 8 pixels; physical sigmas therefore change between LODs. It measures spatial contrast, not “erroneous noise”, and includes snow and shading variations.

| Real view, July | MAE Y v2/Render3D | Band RMS v2 / Render3D | Band sigmas |
| --- | --- | --- | --- |
| MID3 hot desert | 3.22 | 2.00 / 1.93 | 600 / 2400 m |
| MID3 high cold desert | 2.48 | 3.35 / 3.33 | 600 / 2400 m |
| MID3 alpine tundra | 46.81 | 10.30 / 5.43 | 600 / 2400 m |
| MID3 user coast | 21.94 | 4.29 / 2.79 | 600 / 2400 m |
| MID4 alpine tundra | 84.86 | 2.87 / 1.46 | 1200 / 4800 m |
| MID4 user coast | 34.94 | 2.71 / 0.61 | 1200 / 4800 m |
| MID6 hot desert | 9.33 | 0.45 / 0.46 | 4800 / 19200 m |

These differences locate extra contrast in cover/snow cases rather than the desert control. They do not say whether the right result would be adding snow to Render3D or softening v2 snow.

## Verifiable causes and hypotheses to distinguish

1. **Separate coarse path.** In `web/terrain_renderer.js:457`, then `coarseColor` construction near line 654, coarse rendering calls `v2Evaluate` directly with `r32float` climate and source spacing as footprint. It uses neither `aux3 rgba8unorm` nor `v2Fragment`. MID4 clusters therefore cannot be explained solely by aux3 temperature quantization.
2. **No terrestrial microtexture at 7680 m.** `web/terrain_render_v2.js:29` filters 2400/670 m mosaics; at a 7680 m footprint they are disabled. `v2Appearance` also neutralizes displayed tone and mineral. Snow, however, still depends on temperature, potential, slope and TPI (`:30–54`). A hypothesis consistent with the PNGs is that multi-km climate/retention fields on an enlarged 7.68 km DEM produce the large MID4 clusters. Targeted ablation is needed to assign each contour to a specific term.
3. **Partial-snow contrast.** Snow blends colored ground with RGB(.88,.91,.91), retaining macro lighting. Even with a continuous function, intermediate cover ranges may dominate the weak variation of gentle relief. Formula continuity does not imply low visual contrast. Images alone do not prove a quantized thermal boundary.
4. **Explicit close material change.** `v2Fragment` (`:271–273`) replaces inherited ground with `v2Open` between 6 and 1.5 m/px. The 2.56 km and 640 m views cross this transition; it is a concrete cause of the observed hue change, without proving that all its contrast comes from this term alone.
5. **Crown climate contract verified in assembly.** Reading legacy `forestClimate` alone suggests a mismatch with v2 aux3, but `web/terrain_render_v2.js:310` explicitly replaces its preamble during `drawShader`: annual=(Ts+Tw)/2, humid=c.g×5 and coldest=min(Ts,Tw). The assembled shader has appropriate decoding. This lead is ruled out; it does not explain dark crowns.
6. **Fine/latent climate transport precision.** Aux3 is an `rgba8unorm` texture (`web/terrain_renderer.js:940`) and encodes 100 °C in 255 intervals: stored temperature steps are 0.39216 °C. Bilinear filtering interpolates values but cannot recover precision lost at bake. This property can contribute to fine/latent cover steps; no quantitative attribution of a real PNG boundary to this term is claimed here. It does not affect the coarse path described above.
7. **Different rainfall normalization at bake and draw.** Aux3.g (`web/terrain_renderer.js:562`) divides rainfall by a threshold calculated with fixed cooling of 0.0065 °C/m, while `v2EnvironmentAt` (`web/terrain_render_v2.js:240`) reconstructs rainfall with the threshold from actually transported Ts/Tw, whose altitude gradients can differ. Before quantization/saturation, decoded/source rainfall equals threshold(actual Ts,Tw)/threshold(annual temperature with fixed lapse). This is a transport inconsistency visible in the fine/latent path, not proof that every capture patch follows from it, and not an explanation for coarse MID4.

## Real canopy and limits of beach counting

The supplementary analysis `output/render-v2-40km/real-detail/audit-real-canopy.json` compares the same centers, 1.28 km windows, lighting and 30 m decoder DEM. The first 400 rows of the 512² images exclude HUD and minimap. A crown-texture mask is defined once on Render3D and applied to both images; it is a visual proxy, not semantic shader segmentation.

| Real site, July | Mean Y v2 / Render3D in proxy | Fraction Y<40 v2 / Render3D | High-frequency RMS v2 / Render3D, sigma 1 px |
| --- | --- | --- | --- |
| Low tropical forest | 31.56 / 57.89 | 83.86% / 12.18% | 4.04 / 6.97 |
| Low humid temperate | 74.41 / 90.68 | 0% / 0.17% | 1.90 / 9.22 |

The temperate site is at latitude −50.01°: July is winter there. These data corroborate AF darkening and the loss of temperate crown contrast; they do not establish physically correct comparator luminance.

No reliable “beach” pixel count is claimed for AF/warm-coast at 39.0625 m/px. Narrow strips are generally subpixel or one pixel, mixing water/land, beach and ridge lighting. A color threshold would invent semantic precision; a production beach diagnostic or genuinely closer coastal capture would be needed.

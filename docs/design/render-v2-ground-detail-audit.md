# Detailed ground under vegetation: audit and integration points

[Français](render-v2-ground-detail-audit_FR.md)

Initial code audit of October 23, 2026, compared exclusively with captures of real DEMs, followed by the composition correction described below. The defects analyzed first correspond to the **pre-correction** state. Literature research and the final spectrum/filter choice are handled separately: this note describes concrete code constraints for their integration.

## Two different causes of smooth ground

Filtering effectively eliminates every `groundFieldWithClasts` band once the footprint reaches 1.176 m/px. Then, even when a band is resolved, ground reconstruction in `applyForest` loses a substantial part of its detailed albedo. Adding noise or increasing gain does not fix this second cause.

The main sources are `web/terrain_render3d.js` (`BANDS`, `groundVisibility`, `groundFieldWithClasts`, `applyForest`) and `web/terrain_render_v2.js` (`v2Open`, `v2Fragment`, `drawShader` substitutions). Final shader assembly matters: v2 substitutions must be included in the reading.

## Bands actually visible

The filter applies to `wavelength × aspect × filterWidth`, not nominal wavelength alone. Visibility starts falling at 0.18 times that width and reaches zero at 0.42 times it. This is a conservative cutoff; relaxing it without further filtering would reintroduce undersampled detail.

| Band | Current form | Nominal wavelength | Filtered width | Fade begins | Extinction |
| --- | --- | --- | --- | --- | --- |
| 0 | Rounded anisotropic ridges | 8 m | 2.8 m | 0.504 m/px | 1.176 m/px |
| 1 | Narrow grooves | 2.2 m | 0.264 m | 0.04752 m/px | 0.11088 m/px |
| 2 | Elliptical clasts, irregular grid | 0.32 m | 0.136 m | 0.02448 m/px | 0.05712 m/px |
| 3 | Fine grain | 0.055 m | 0.0396 m | 0.007128 m/px | 0.016632 m/px |

At 2.5 m/px (real 2.56 km window), **all four bands are zero**. At 0.625 m/px (real 640 m window), only the 8 m ridge band remains, at 95.63% amplitude. Clasts are therefore not small visible dots at these two zooms. At 0.1 m/px, the 8 m band remains full, the 2.2 m band retains only 3.85%, and the other two are still zero.

`groundSlopeVariance` transfers a scalar approximation of unresolved variance to GGX roughness. This roughness does not create visible spatial texture. In v2, the BRDF reference uses the same roughness: when no band is resolved and there is no crown, the detail/reference ratio cannot invent absent detail.

## What grass currently changes

`v2Open` mixes a ground color and a grass color; it produces neither blade structure nor litter material. The same geometric ridge/groove/clast field is then used for all covers.

In `v2Fragment`, outside snow, micro-normal gain is:

- bare rock: up to 0.52;
- ground without grass or rock: 0.09;
- full grass without rock: 0.02;
- full snow: 0.008.

Grass therefore strongly attenuates the rock field's relief without replacing it with its own spectrum. Base roughness remains 0.88 for grass, earth and sand when rock fraction is zero; moisture and organic matter do not define a different BRDF material here.

`swardTone` is the **same** `detail.x`, multiplied by a grass mask and `groundVisibility(.32, fp)`. It dies out at 0.1344 m/px: it contributes nothing to captures at 2.5 or 0.625 m/px. It is not an independent grass field. `form` retains at most ±8% albedo outside snow but then encounters the composition problem below.

## Where detail is replaced in forest

The v2 path is:

1. `open` contains reference ground without `form/swardTone`; `distantColor` mixes canopy color into it.
2. `m.albedo` receives `distantColor × (1 + form + swardTone)`; `m.normal` receives the detailed ground gradient.
3. `applyForest` receives the **undetailed** `open`, `closed` and `distantColor` guides.
4. The v2 `drawShader` substitution replaces `groundRecolor(guide.open)` with `guide.open`. Reconstructed ground therefore does not receive the detail factor calculated at step 2.
5. Even in the **no-tree** branch (`st.cover`, neighboring cover and mip all almost zero), `applyForest` remixes albedo toward this smooth guide. At full reveal, the `form/swardTone` factor is entirely lost.
6. Other branches reconstruct `litter`, then `nearAlbedo`, from the same smooth ground and litter colors. `nGround` normals survive in gaps; detailed albedo is replaced.

This explains why inspecting `m.albedo` alone cannot establish that detail remains visible under vegetation. The opaque relief of a crown should naturally hide ground: the problem concerns gaps, ground between crowns and actually visible understory, not ground behind an opaque crown.

The sibling Cg already has a different strategy in `rust/bridge/terrainForest.ts:649`: `k = groundToDisplay(m.albedo) / renderGuideColor`, then `openGround = renderGuideOpen × k`. This passes detail to its open guide. NE v2 does not use that transfer. An explicit factor would be safer than division by a dark color and avoids accidentally transferring canopy tint to rock.

## Proposed integration in two bounded steps

### 1. Preserve ground material during composition

Explicitly separate macro reference, visible ground material and crown material:

```text
macroReference = macro color without detail
floor = detailed ground: albedo, normal, roughness, occlusion
floor = apply litter / low grass while preserving filtered detail
visible = compose floor and crown according to their actually resolved cover
final = macro lighting × detail response / reference response
```

In current code, the first change can reuse the already computed field: explicitly pass the ground-albedo factor to `applyForest`, applying it to no-tree branches and ground/litter albedos **after** litter-color selection. The factor must not modify crown color. Do not reconstruct albedo from the smooth guide alone in early returns.

Normals and roughness must follow the same separation: retain the ground normal in gaps; use the macro normal and crown-specific slopes on its surface. Currently `nCrown` obtains `macroSlope` from `nGround`, which already includes ground micro-slopes: this transport should be separated when understory material is enriched.

This step adds mainly parameters and blending operations, not a second noise evaluation. It must preserve low snow gain, marine domains, DEM relief and crown thermal masks.

### 2. Give ground, low grass and litter their own bands

Introduce per-material detail descriptions rather than one rock field whose amplitude is reduced. An evaluation must at least provide height/gradient, mean-controlled albedo modulation and unresolved slope variance. Rock fraction retains its ridges and clasts; earth, sand, low grass and litter choose other amplitudes, anisotropies and physical bands. Coefficients must be evaluated on real sites, not fixed by another series of artificial terrains.

For variation to be visible at 0.625 m/px with the current filter, its width must exceed 1.49 m; at 2.5 m/px it must exceed 5.95 m. At these spacings represent weak resolved surface/aggregate variations, not several-meter gravel or grass blades. Finer detail continues to feed filtered roughness. Surface bands are also missing between large ecological mosaics hundreds of meters wide and ground bands cut off below 1.176 m/px.

Integrate an oriented filter or controlled-spectrum noise into the shared evaluator using the footprint transformed into the detail domain, preserving unresolved variance. Replacing only the generator while retaining smooth `applyForest` reconstructions would leave the main problem intact. Avoid arbitrary contrast renormalization of remaining bands at each LOD.

Retain integer/fractional ground coordinates (`r3.cells`, `r3.fracs`) for new scales; do not pass a world f32 position of millions of meters directly to a centimeter-scale pattern. Material weights and masks must be filtered before composition; rapid transitions need coherent treatment of their derivatives or composition of responses, rather than a false height gradient.

## Scope and checks against existing evidence

The BWH `arid-plain/rolling` captures show almost uniform ground in the 640 m window, and BSK shows the transition from blurry masses to crowns. Real AF/CFB canopy loses local contrast in v2; these findings are recorded in the [real audit](render-v2-audit-real-open-ground.md), with methods and limits. They are consistent with this composition path without constituting ablation of every shader term.

For step 1, reuse the **same real positions and extents**, particularly a crown-free BWH area, an AF gap, a BSK area between trees and an EF control. Check that ground modulation survives the no-tree return and gaps, crowns remain unchanged, and detail disappears cleanly when subpixel. Also check a slight camera movement for flicker; one sharp image is insufficient.

The Cg GLSL port must follow the same composition without accumulating its existing ratio and a new explicit factor. `tools/assets/sync_render_v2.cjs` currently shares the environment, not this entire composition. The NumPy backend produces environmental/macro albedo and relief lighting, but does not reproduce GPU micro-BRDF/crown bands: environmental parity alone will not validate this detail change.

## First composition correction implemented

In `web/terrain_render_v2.js`, the fine v2 path now passes the ground-albedo factor to no-tree, average-litter and near-litter branches. Crowns receive their own albedo and macro/crown normal without ground modulation. Reference roughness without unresolved variance follows the same cover mixes as the final material: ground variance retains its BRDF response in gaps and does not rise onto an opaque crown.

The exposure reference is calculated after composition from the final material and a **filtered** macro slope distinct from physical DEM slope. Final correction is `shaded × (baked.a / flatHillshade) × Y(flatShade) / Y(macroReferenceShade)`. `flatHillshade` uses the same strength, ambient, contrast and sun direction as the bake. It removes double base exposure and excess slope shading without normalizing flat-light level to bare albedo. Directional DEM shadows remain applied.

The sixth v2 layer carries `aux5.rgb=canopy leaf-on` before phenology/snow and `aux5.a=macroGradient.x`; `aux0.b=macroGradient.y`. The bake macro gradient is six times filtered physical slope. The BRDF reference uses it divided by three to match the ×2 physical-slope gain retained in the material. Thus the reference removes large relief already present in hillshade but leaves small DEM-relief contribution; this is not replaced by ground noise. Crowns use the ×2 physical normal without ground micro-slopes. Render3D's five byte layers remain distinct from v2's six half-float layers.

Inherited leaf-on color is multiplied by the seasonal ratio `v2Canopy(snowFree) / v2Canopy(leafOn)`, then receives v2 snow. There is no global palette gain. The cover contract was later replaced by filtered effective aux2 population: see the three surfaces below. `applyForest` remains responsible for revealing crowns.

The inherited beach mask, whose storage is multiplied by 0.8, excludes forest and grass cover even when snow hides sand. New transport `aux0.g=r3Flats` retains sediment bars/flats beyond the strict beach. This sediment acts after **average** canopy in the distant guide, as in Render; it does not recolor resolved crowns. Flats do not prohibit estuary trees. Average ground remains `macroColor` at every footprint, enriched by `v2GroundTint` before overlays; this weak mineral correction is not a second close palette. Palette replacement between 1.5 and 6 m/px was removed.

`tests/browser/test_terrain_render_v2_floor.cjs` compiles the production shader and ablates only its detailed ground response on small unit material inputs. These inputs **are not visual terrain evidence**. Chrome/ANGLE GPU result with six aux16float layers, after saturated-form correction: average difference with/without detail 0.617/255 on bare ground, 0.414/255 in 1,284 gap/litter pixels, **exactly zero in 14,875 opaque-crown pixels**, and zero forest cover on the tested beach. The test also bounds bare-ground contrast to prevent its return as a binary mask. Color error for expected flat sediment: 0.358/255; all 14,875 crown pixels remain in the estuary control. Difference from directly lit flat-ground BRDF response, with the same display curve on both sides, is only 0.093/255, at most one quantization level: flat-ground double exposure is eliminated. Report: `output/render-v2-floor/report.json`.

Exposure isolation on those crowns now gives Y55.941 for albedo and Y57.152 with final correction; Y46.712 would result from albedo multiplied only by raw hillshade. Initial normalization to bare albedo gave Y38.042 on the old palette. These stages were isolated separately; the change is not a global gain added to the palette.

Low visible reflectance of cover alone, however, does not prove incorrect albedo: [Wu et al., 2018](https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.14939) measure mean visible Amazon leaf reflectance of 0.05 for mature and 0.04 for old leaves. These spectral measurements are not directly crown RGB colors; they justify separating albedo diagnosis, lighting and structure before blindly brightening a palette.

This first correction does not replace the four-band spectrum and does not claim to complete the realistic material. Before/after comparison on real sites and camera-motion checking remain necessary before final visual validation.

The intermediate real pass `output/render-v2-floor/final-candidate-real/AUDIT.fr.md` then exposed two defects that this unit test could not visually accept: overly contrasted arid flats and snow clipped after exposure correction. Ground form is now bounded in [-1,1] **before** multiplying by its amplitude 0.04 soil → 0.06 rock → 0.006 snow; the old `clamp(detail.x*10,-.08,.08)` saturated the 8 m band at only 8 mm height. Sward follows the same form/amplitude separation. The V2 path retains high BRDF values until final `v2DisplayShoulder` compression on peak RGB before water; Render3D keeps its existing clamp. Exposure references remain uncompressed. This curve is a display adaptation to the hybrid architecture, not a claimed ACES implementation or complete radiometric simulation.

An isolated real ablation of AF average blending is recorded in `output/render-v2-floor/mean-display-candidate/AUDIT.fr.md`. The distant material now decodes the inherited displayRGB average rather than remixing two linear albedos; modulation remains applied only to the open fraction. The GPU test verifies this mean equality on mixed cover 0.761 with exactly zero error. On the real AF 640 m square at four zooms, however, this correction lowers distant Y only 1.15% and does not change the close view. The average gap as crowns resolve remains: normal/shadow distribution and occlusion must be isolated before concluding a palette error.

## Unresolved crown response

`TerrainRenderV2.canopyMeanShader` exports a LUT of 19 solar elevations for diffuse and specular coefficients of the two analytical forms already used by `forestCrowns`. Quadrature is uniform in projected-disc area; broadleaf slope 1.1r and needle slope 1.6r/sqrt(r²+0.015), intrinsic AO 1−0.4r² and roughness 0.9 come from the existing shader. `v2CanopyMeanRatio(rgb,needle,light,light2,strength)` divides integrated response by that of the same flat material. This ratio applies only to the average closed guide, before sediment/beaches and fading under snow. Actual resolved crown color remains distinct. No quadrature runs per fragment: the LUT contains 76 coefficients, interpolated by the elevations of both lights.

`tools/verification/verify_render_v2_canopy_mean.cjs` checks 60 combinations against the actual GPU-compiled BRDF: maximum ratio error 0.000266. The LUT preserves a **linear** response average, not an exact mean of RGB codes. Maximum bias in the latter reaches 0.099 at grazing light; at 45° it is about 0.02 for broadleaf and 0.03–0.05 for conifers. Overlap, inter-crown shadows, clumps and ground under cover are not preintegrated. This limited scope must remain visible in quality claims.

The historical real pass `output/render-v2-floor/canopy-mean-af/AUDIT.fr.md` showed a 6.6% reduction of distant AF Y without changing the fully resolved close view: partial improvement, with a 29.5% residual between the two views of the same 640 m square. A separate test of aux2 cover averaged at mip4 was **removed**: insufficient improvement and unresolved coarse/fine contract. At that point the guide retained ecology.r. The f5 archives do not validate the new bd367 identity after server reload: the new campaign retains its own provenance.

## Population and average of three lighting responses

Real bd367 ablation `output/render-v2-floor/bd-tone-fixed/tone-comparison.json` isolated a contract error: aux2.a contains riverbank moisture, whereas `ForestStand.tone` expected a residual centered at 0.5. Replacing bank 0 with canonical tone corrects an undue 0.83 factor without globally brightening the palette. On the same central 640 m square, close AF rises from Y53.241 to 59.097; Render3D captures remain identical. The GPU test now changes bank moisture from 0 to 1 and requires exactly zero difference in the 14,875 opaque-crown pixels.

Aux2 population filtered at the physical footprint then replaces the ecological guide as authority for fine cover and its diagnostic. For effective cover `c` and exact internal density `d=forestDensity(c,kappa)`, stand fraction is `share=c/d`. The three areas are `(1-share, share-c*leaf, c*leaf)`, excluding physical beach. Each surface is lit separately: open ground, litter from `forestFloor` with AO 0.72, average crown from `forestCrownColor` then the intrinsic LUT, roughness 0.9. **Linear lighting responses** are added; their AO or normals are not mixed into an artificial material. Flats and beach follow, before display compression. Original reveal blends this average with the close response; fully resolved photons remain unchanged.

Ablation `output/render-v2-floor/bd-three-state/` gives AF Y62.401→59.097 between 40 km and 640 m views of the same square, or 5.59%, and CFB 88.328→84.839, or 4.11%. Direct activation of close reveal produced promising averages but unacceptable binary patches; it is rejected. The progressive clearing profile keeps the same noise/positions and widens the threshold using `sigmoid(4*(score-1.07*threshold))`. Real pass `bd-progressive-three-state/` retains similar gaps (AF 5.61%, CFB 4.06%) with softer transitions. A cartographic mosaic remains visible: these measurements do not constitute photorealistic validation. The 1.07 compensation approximates variance; its reference Gaussian quadrature is retained in `output/render-v2-floor/clearing-profile/gaussian-reference.json`, separately from visual evidence.

## Native coarse average

`terrain_render3d.js` exports exact pure climate, density, litter and crown-color functions, as well as `r3ForestExpected`. The latter evaluates the deterministic part of the same field: climate at elevation, slope, valley/ridge, bog, rock, perennial snow, height and density; lattice residuals are centered at zero. Clearing fraction is integrated with terrain bias instead of evaluating noise at one point. This proxy is **not** the exact mean of the fine field: climate nonlinearities, noise covariance and rare disturbances are not integrated. It does not guarantee coarse/fine identity or create new physical data.

The V2 coarse bake lights the same three surfaces on a plane, retains their linear sum and then encodes it as HDR display RGB **RGBA16float**. The fragment applies only directional hillshade normalized by its flat-ground value, then the shoulder once before water. Response depends on light: the key includes vectors for the actually displayed LOD, season and forest settings. Render3D retains RGBA8 storage and its historical shader. No additional statistical field or texture is added; the V2 color buffer costs eight bytes per texel versus four for Render3D, or 819,200 versus 409,600 bytes for a doubled 160² block.

`test_terrain_render_v2_climate.cjs` verifies on GPU the calendar, temperature/rain reconstruction, LOD-specific light, forest invalidation, both formats and memory accounting. Switching V2→Render3D→V2 restores the exact memory total, and Render3D bytes read before/after are identical. This technical check does not replace real LOD4/6 captures needed to accept the coarse proxy.

## Next causal correction: stand thinning

The progressive stages above remain archives: the reduced humid rate had not removed the last CFB blobs. Real audit `output/render-v2-floor/cfb-causal/AUDIT.fr.md` isolates their origin. Zero clearings removes them, zero outcrops retains them; near-zero population pixels have grass close to 1 and rock fraction close to 0. The problem is therefore disturbance distribution, not missing ground or relief.

The shared profile is now `thinning=min(.55,2*c*sigmoid(score/1.15))`, where `c` is the natural climate rate; cover receives `cover*=1-thinning`. The historical signature `v2ClearingFraction(score,threshold)` decodes `c` from its quantile. A 100–400 m event no longer empties its cell; small gaps come from tree candidates, physical openings remain independent. The coarse proxy retains average `min(.55,c)` instead of the old threshold CDF. No preservation of the previous large-opening average beyond the cap is imposed.

On real AF/CFB candidates at 10 km and 640 m, the targeted CFB 10 km defect is removed while retaining relief. Mean Y of the entire CFB plate remains 83.7598→83.7273, and 20–160 m chromatic contrast drops 26%: this is not a global gain. Render3D controls are pixel-identical. This correction does not constitute global photorealistic acceptance; crowns and broad gradients remain those of the existing engine and must be revisited in the final real gallery.

## Rejected DFC preflight, then optical density and snow on litter

Real db083 preflight still shows an invasive white network in DFC at LOD3. The verified path is latent 240 m → uploadTile 304² → nine aux16float tiles → r3Pipeline/v2Fragment; native-coarse is only used at LOD≥4. The first three coarse-helper probes were therefore uninformative and explicitly discarded. Full causal audit: `output/render-v2-floor/dfc-snow-causal/AUDIT.fr.md`.

Useful ablations isolate fine population: constant cover removes the network, whereas snow 0, lighting 0, bog 0 and neutralized eligibility do not. Direct cover modulation by the `stand` residual is replaced by `v2StandCoverage`, applying the same residuals to effective optical density. Noise, positions, height/tone and physical exclusions are preserved; no DEM blur or palette gain is involved. This density is not measured LAI. DFC 153.6/40 km images show readable relief again; AF/CFB and their close 160 m views remain consistent. Crowns remain those of the existing engine and still stylized.

A separate defect reconstructed brown litter after snow. V2 blending now follows the CG contract `litter=mix(litter,ground,v2GroundSnow)`, applied to statistical/close paths and the pure native-coarse helper. The full-snow GPU gate verifies exactly zero albedo difference on 2,098 understory/gap pixels. Both real post-port Render3D DFC PNGs, 153.6 km and 40 km, are pixel-identical to reference preflight. Final sources for this stage: `790e34695a49…`; 25 Python tests and both floor/climate GPU regressions pass. This does not turn the rejected preflight into final validation: a new campaign version must retain its own receipts and reviews.

## Opaque crowns at beach edges

Preflight `790e34695a49` is also a partial archive: the real 160 m `warm-coast-beach-detail` frame reveals translucent crowns on sand. The cause is identified in `drawShader`: `reveal *= 1-v2BeachCoverage` modulated blending of the entire close response by the per-pixel beach mask, including inside crowns. The earlier “full beach without trees” test did not cover this error at intermediate fractions.

Beach becomes root eligibility, stable for each candidate. Physical mask `B=clamp(aux4.a/.8,0,1)` is read at tree center, bake mip 0: a dedicated deterministic random value rejects the candidate if below B. This decision depends neither on fragment position within the crown nor camera footprint. Surface availability `1-B` remains applied to the unresolved average and unresolved stand/shadow masks; it no longer multiplies explicit-tree reveal. This area mix is a local approximation of coastal transition, not exact integration of crown intersections. Estuary flats are not root exclusions.

The GPU gate for partial beach B=0.5 retains 10,932 fully opaque crown pixels, with exactly zero ground-detail leakage. Full beach has no trees; flats alone retain the forest control's 14,875 opaque pixels. The real 160 m / 2.56 km beach pair is captured in July V2, Render3D and January V2: six PNGs, protocol passed, ghost crowns removed and sand retained. Both Render3D controls are strictly pixel-identical to preflight 790. Receipts: `output/render-v2-floor/beach-root-final/captures/eea7a9b2aa83-f870d039-a911f822/report.json` and `legacy-pixel-equality.json`. This targeted correction is accepted; the global campaign remains to be repeated after separate hydrological work.

The same stage carries the validated marine candidate on two real coastal frames: artistic attenuation 0.035 for sea, 0.15 unchanged for inland water. Shallow bottoms remain readable at 20–30 m without claiming calibration of a measured Kd coefficient. The 150–3000 m bathymetric palette remains distinct from this attenuation. The NumPy mirror and shared GLSL are synchronized; 26 Python V2 tests pass, including exact reconstruction of old inland absorption. Primary optical references and model limits remain in the soil-literature document.

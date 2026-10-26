# Neural Earth architecture

## Source world and geometry

`backend/terrain_generation.py` normalizes settings. `backend/terrain_orogen_stages.py` retains relief, erosion and climate as immutable stage artifacts. Orogen's bundled graph algorithms run through Node; CUDA rasterizes spherical triangles into a 2,048 × 1,024 atlas. Optional CUDA acceleration and City wgpu erosion are separate choices.

A stage's raster projection, hashing and saving run on a worker while Node computes the next stage. The server writes stage and atlas files after responding (`backend/terrain_array_io.py`), serves them from memory meanwhile, and flushes before exiting. Recent stage artifacts stay in memory while their files are unchanged.

The spherical world uses an equirectangular map: width `pi × diameter`, height `pi × diameter / 2`. The default diameter is `40,000 / pi` km. Longitude wraps. A planar world is a finite square whose side is the diameter, with Cartesian distances and no globe view. Selecting planar geometry does not replace Orogen's spherical tectonic construction with a new planar tectonic model.

The optional World Builder bridge samples a native cube-sphere CPU atlas, then produces a 1,024 × 512 raster. Separate area-weighted quantile transforms match positive land heights and ocean depths to ETOPO distributions. They preserve sign and rank without copying Earth's geographic locations. Native candidates are selected deterministically for the requested continent layout; receipts distinguish requested and selected seeds.

## Conditioning and networks

The five physical inputs are elevation, BIO1 temperature, BIO4 temperature variability, BIO12 rainfall and BIO15 rainfall variability. `backend/terrain_conditioning.py` adapts each source; checkpoint normalization constants remain unchanged. For Orogen, two seasonal temperature proxies define a 12-month harmonic. Seasonal rainfall totals are split equally across six months each. Sample standard deviations use the `sqrt(12/11)` correction; BIO15 uses the monthly-rain convention documented in the adapter. This matches a statistic contract without recovering actual monthly weather.

The pretrained `EDMUnet2D` hierarchy runs on CUDA in BF16: coarse statistics at 7,680 m, base latents at 240 m, and the Laplacian decoder at 30 m. Coarse sampling retains 20 solver steps; the base stage retains two fusion steps. `backend/terrain_inference.py` adds cached constants, preparation, batching and exact kernels around the unchanged networks. CUDA Graphs reduce launch overhead and fall back when admission fails.

The UI calls its noise-to-signal amplitude **SNR**, following the existing checkpoint interface. Here, a larger value permits more learned correction; a smaller value follows source inputs more closely. This naming is not the conventional signal-to-noise power ratio. Adaptive rules use altitude, climate and latitude. Per-LOD settings scale reconstructed local relief residuals after inference; they are not separate networks.

## Lazy spatial generation

InfiniteDiffusion uses deterministic spatial noise and overlapping dependency windows. `backend/terrain_window_scheduler.py` coordinates reusable coarse, base and decoder requests. Work scales with requested area; random access does not mean a free full-resolution planet. Coarse windows can run on 1, 2, 4, 8 or 16 streams, with four by default and network batch one.

`backend/terrain_jobs.py` prioritizes visible coverage and detail. Camera sessions and epochs cancel obsolete queued work; a submitted GPU block finishes at its cancellation boundary. Persistence and image encoding run outside the serialized GPU compute lane. Whole-world coarse preparation is optional and disabled by default.

| Display LOD | Sample spacing | Typical source |
| --- | ---: | --- |
| 11–7 | 61.44 km–3.84 km | Source overview, then learned coarse where available |
| 6–4 | 1,920–480 m | Learned coarse at 7,680 m |
| 3 | 240 m | Base latent reconstruction |
| 2–0 | 120–30 m | Decoder at 30 m |
| -1 | 15 m | Experimental decoder refinement below model resolution |
| -2–mode floor (currently -6) | 7.5 m–47 cm | Experimental procedural detail |
| -7–-12 | 23 cm–7.32 mm | C2 interpolation of the selected mode's last generated relief |

The source resolution and display spacing are different quantities. Parent tiles remain visible while target coverage arrives. Negative LODs add optional detail below the trained resolution; they are not a higher-resolution checkpoint. `backend/terrain_refinement.py` works in the checkpoint's signed-sqrt height space, where coasts cross zero nearly linearly. It samples a C2 cubic B-spline of the native DEM directly at the requested spacing and adds one decoder octave at LOD -1, also a C2 spline. Finer decoder octaves would need latents stretched beyond the checkpoint's training distribution, so LOD -2 to -6 add deterministic sparse-convolution octaves instead. Their amplitude continues the local roughness of the decoder and native octaves, decays more slowly on steep ground, is smaller in valleys and is capped in slope; the two coarsest octaves elongate down steep slopes. A low-frequency noise and the relief gather the finer octaves into rocky patches, with heavy-tailed impulse weights, so metre-scale grain does not cover every slope evenly. Within a few cells of sea level the base comes from a smoothed spline, so the coastline ignores the 30 m raster and stays the same curve at every level, up to sub-metre bends of about 2° at most. Height detail never changes land or sea. Hillshading therefore shows no 30 m grid or coastal staircase. Optional CuPy kernels fuse these steps on CUDA and match the PyTorch fallback bit for bit. This detail is plausible texture, not learned or measured terrain. The glacier and hydrology content of learned tiles has no feature-specific guarantee.

## Rendering and polar sampling

`backend/terrain_lod_interpolation.py::RELIEF_FLOOR` is the single per-mode policy table. The seed-42 high/coastal mountain captures and height-normal detector in `tmp/e2e/interp/diagnosis` did not reproduce a repeating lattice through -6 in classic, hybrid, Phasor or MSE; all four currently use their generator limit of -6. The detector reviews lattice-frequency/harmonic power against neighbouring frequencies (>8) together with lattice-line normal-gradient jumps against interiors (>1.8), plus visual inspection. These are empirical defaults for the tested versions, not a guarantee for every terrain. `TERRAIN_RELIEF_FLOORS='{"classic":-3,"hybrid":-4}'` overrides individual floors for retesting (integers -6..-1; restart the server). Generator modules and their own minimum LOD remain unchanged.

Below a mode's floor, no finer generator is called. Fixed globally anchored 128-sample parent blocks cache heights and surface channels per world/mode/version/intensity (up to 128 blocks, seeded from already generated floor tiles); overlapping reads use the same source values. Biquintic Hermite patches share all nodal derivatives through order (2,2), giving C2 continuity and exact parent-knot values. A shared nodal limiter bounds the Bezier control net to each cell's corner range plus at most 2 cm for heights (numerical tolerance only for surface channels); this avoids cardinal-cubic cliff ringing without post-clamping the interpolated surface. Integer division precedes coordinate conversion, including negative positions. Evaluation and gradients remain float64 until transport. The local limiter optionally runs as a fixed-order FP64 CUDA kernel (FMA disabled), with bit-exact CPU parity and a CPU fallback if CuPy is unavailable; no reductions depend on the requested window shape. Keys add `/interp-from-<floor>-v1`, and capability metadata publishes `relief_floors`. Pure climate inputs are unchanged; height-derived classification, soil appearance, hydrology display and contours use the interpolated DEM.

At millimetre sample spacing, differencing absolute FP32 mountain heights would amplify their rounding into shading bands. The viewer requests `gradients=1`: interpolated binary tiles then append an optional full-resolution interleaved `rg32float` (dH/dx,dH/dy) plane after surface channels, declared by `X-Terrain-Gradient-Format`. WebGPU uses these float64-derived slopes for Relief/Render and the Render3D material bake. Heights retain their original FP32 format and clients that omit the option receive the original payload layout. PNG rendering consumes float64 interpolated heights. `X-Terrain-Relief-Floor-LOD` and `X-Terrain-Relief-Interpolation` identify interpolation. WASM retains its runtime minimum LOD.

The Flask server returns FP32 heights and compact climate/appearance planes. `web/terrain_renderer.js` shades them in WebGPU. `backend/terrain_render.py` is the independent NumPy material reference; `backend/terrain_render_torch.py` runs it on CUDA for PNG tiles and overviews. `web/terrain_render3d.js` adds the Render3D layers: a per-tile GPU bake of Render's guide, forest field and climate, then per-pixel ground and crowns ([method](render3d.md)). `web/terrain_globe.js` projects the same physical terrain onto a sphere. Above 60° latitude at local zoom, the globe uses a 90° rotated chart so the geographic poles lie near its equator. This improves sampling geometry without retraining the model on polar terrain.

Biomes sample source Köppen classes and recompute altitude/slope appearance on the visible DEM. Climate diagnostic layers keep a stable coarse view. Render materials combine climate, slope, terrain position and filtered deterministic noise; pedology remains a regional source-world diagnostic.

## Identity and persistence

World identities include seeds, settings, stage references, generator source hashes, native binary/source fingerprints, model revision and calibration rules. Heightmaps are persisted, verified and read-only. Neural tile storage and image/appearance caches have different identities. Appearance-only edits avoid neural regeneration; changes to generator/server transport sources can still produce a new world identity.

Browser cache budget and disk tile quota do not bound every server cache. Coarse and source atlas storage can accumulate across worlds. See `backend/terrain_manifest.py`, `backend/terrain_disk_cache.py`, `backend/terrain_final_mips.py` and `/api/status` for runtime behavior.

## Scientific scope

The system combines scientifically informed procedural models with Earth-trained learned terrain. Seasonal climate, Köppen proxies, soil mixtures and surface colors target plausible worldbuilding. They are not validated climate forecasts, ecosystem dynamics or a globally consistent fine-resolution hydrological simulation. Neural refinement may move coastlines and change source erosion forms. The overview is the conditioning atlas, rather than a fully decoded planet.

References: [InfiniteDiffusion paper](https://arxiv.org/abs/2512.08309), [checkpoint](https://huggingface.co/xandergos/terrain-diffusion-30m), [Orogen source and adaptation](../native/orogen/PROVENANCE.md), [City adaptation](../native/city_erosion/PROVENANCE.md).

# Neural Earth architecture

## Source world and geometry

`backend/terrain_generation.py` normalizes settings. `backend/terrain_orogen_stages.py` retains relief, erosion and climate as immutable stage artifacts. Orogen's bundled graph algorithms run through Node; CUDA rasterizes spherical triangles into a 2,048 × 1,024 atlas. Optional CUDA acceleration and City wgpu erosion are separate choices.

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
| -1–-3 | 15–3.75 m | Experimental refinement below model resolution |

The source resolution and display spacing are different quantities. Parent tiles remain visible while target coverage arrives. Negative LODs add optional detail below the trained resolution; they are not a higher-resolution checkpoint. The glacier and hydrology content of learned tiles has no feature-specific guarantee.

## Rendering and polar sampling

The Flask server returns FP32 heights and compact climate/appearance planes. `web/terrain_renderer.js` shades them in WebGPU. `backend/terrain_render.py` is the independent NumPy material reference; `backend/terrain_render_torch.py` runs it on CUDA for PNG tiles and overviews. `web/terrain_globe.js` projects the same physical terrain onto a sphere. Above 60° latitude at local zoom, the globe uses a 90° rotated chart so the geographic poles lie near its equator. This improves sampling geometry without retraining the model on polar terrain.

Biomes sample source Köppen classes and recompute altitude/slope appearance on the visible DEM. Climate diagnostic layers keep a stable coarse view. Render materials combine climate, slope, terrain position and filtered deterministic noise; pedology remains a regional source-world diagnostic.

## Identity and persistence

World identities include seeds, settings, stage references, generator source hashes, native binary/source fingerprints, model revision and calibration rules. Heightmaps are persisted, verified and read-only. Neural tile storage and image/appearance caches have different identities. Appearance-only edits avoid neural regeneration; changes to generator/server transport sources can still produce a new world identity.

Browser cache budget and disk tile quota do not bound every server cache. Coarse and source atlas storage can accumulate across worlds. See `backend/terrain_manifest.py`, `backend/terrain_disk_cache.py`, `backend/terrain_final_mips.py` and `/api/status` for runtime behavior.

## Scientific scope

The system combines scientifically informed procedural models with Earth-trained learned terrain. Seasonal climate, Köppen proxies, soil mixtures and surface colors target plausible worldbuilding. They are not validated climate forecasts, ecosystem dynamics or a globally consistent fine-resolution hydrological simulation. Neural refinement may move coastlines and change source erosion forms. The overview is the conditioning atlas, rather than a fully decoded planet.

References: [InfiniteDiffusion paper](https://arxiv.org/abs/2512.08309), [checkpoint](https://huggingface.co/xandergos/terrain-diffusion-30m), [Orogen source and adaptation](../native/orogen/PROVENANCE.md), [City adaptation](../native/city_erosion/PROVENANCE.md).

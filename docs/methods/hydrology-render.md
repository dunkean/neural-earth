# Hydrology water appearance (CPU)

Module: `backend/terrain_hydrology_render.py`. Pure NumPy; no server, Torch or CUDA import. Tests: `tests/python/test_terrain_hydrology_render.py`. It mirrors the WGSL `hydrologyShader` in `web/terrain_renderer.js` (`hydroGround`, `hydroBlend`). Design context: [multiscale hydrology](../design/multiscale-hydrology.md), section 5, "single documented water-material contract".

This is an appearance rule, not a physical water solver. It paints water that the hydrology authority already decided exists; it never infers water from the height field.

## Inputs and units

| Array | Shape | Contents |
| --- | --- | --- |
| `elevation` | `(H, W)` | Raw physical height in metres above sea level, halo included. |
| `hydrology` | `(H, W, 4)` float | R: water surface level, metres. G: wet coverage of the sample, 0..1. B: class, 0 dry, 1 ocean, 2 lake, 3 river. A: 1 where hydrology is authoritative, 0 otherwise. |
| `rgb` | `(H - 2c, W - 2c, 3)` | Shaded colour, uint8 or float 0..1, after cropping `c` samples on every side. |

The hydrology raster must be sampled on exactly the elevation's sample centres: same origin, resolution and halo. The module does not resample, shift or guess an offset. `composite_water` takes the crop explicitly (`crop=HALO`, usually 24) and rejects any other `rgb` shape.

## Rules

1. **Authority.** A sample is used only if A > 0.5 and all four channels are finite. Otherwise, and for class 1 (ocean), the legacy appearance is unchanged: the CPU and GPU renderers still classify sea as elevation below zero. `hydrology=None` returns the inputs untouched.
2. **Ground for shading.** `ground_for_shading(elevation, hydrology)` returns `max(h, 0.5 m)` for authoritative dry samples (class < 0.5) and the raw height elsewhere. An endorheic basin or dry depression below sea level then shades as low land, not as sea. Use it only as the renderer's input height; depths always use raw elevation.
3. **Lake and river blend.** For authoritative samples with class >= 1.5 and coverage > 0:
   - `depth = max(level - elevation, 0)`, in metres;
   - `t = smoothstep(0.15, 12, depth)`;
   - `water = mix(shallow, deep, t)`, shallow `(0.1019608, 0.3019608, 0.3607843)`, deep `(0.0431373, 0.1607843, 0.2705882)` (the `#1a4d5c` / `#0b2945` palette of `web/terrain_hydrology.js`);
   - `rgb = mix(rgb, water, clamp(coverage, 0, 1))`.

   Other samples keep their exact input bytes. uint8 output is rounded, as an `rgba8unorm` store.

Server call order around the existing CPU renderer:

```python
from terrain_hydrology_render import ground_for_shading, composite_water
rgb = render_elevation(ground_for_shading(elevation, water), lod, ...)
rgb = composite_water(rgb, elevation, water, crop=HALO)
```

## Deviations from the GPU path

- The CPU composite is evaluated at sample centres. The GPU interpolates the bed bilinearly and inland level weighted by wet coverage, retaining the dominant wet class. It blends after the colour mip. Dry neighbours therefore do not pull an inland level toward zero. At one sample per pixel, the two agree within colour quantization.
- The GPU builds six water mip levels. Coverage is area-averaged; level is averaged with wet coverage as its weight, and the dominant wet class is retained. Screen-footprint interpolation between these levels keeps a distant channel's wet fraction rather than painting a full-width ribbon or sampling alternating wet/dry texels. The same filtered field is used by the map and Render3D shaders.
- Non-finite hydrology channels count as no authority. A non-finite bed under valid water counts as zero depth. WGSL behaviour for NaN is undefined, so the GPU path makes no such promise.
- Float `rgb` outside 0..1 is not clipped where no water is painted.

## Limitations

- Ground for shading changes only the visual height, never the physical bed. CPU shading receives this ground field; the WebGPU shore-distance and openness probes also consult hydrology authority. Render3D's land-cover branch uses the same dry-land rule.
- Lake and river samples are not lifted. Below sea level, their base is still the legacy sea colour, so a lake with coverage below 1 mixes the lake palette with sea blue rather than with land.
- There is no resolved foam, sediment bar or explicit flow-animation appearance. CPU compositing follows the existing CPU/Torch material path; there is no additional Torch water kernel.
- Depth colour uses the delivered elevation. If the delivered bed differs from the hydrology's conditioned bed, the depth tint follows the delivered field.

## Evidence

`python -m unittest tests.python.test_terrain_hydrology_render`: 4 tests. They cover legacy appearance (no hydrology, invalid authority, ocean class), dry endorheic land below sea level, depth/coverage clamping and crop alignment, and finiteness with NaN/inf inputs. A 304 x 304 tile with ~60 % wet samples takes about 8 ms for both calls on the development machine.

`node tests/browser/test_terrain_hydrology.cjs` compiles the actual WebGPU shaders and reads screen pixels: a fully covered 10 m-deep lake yields RGB (12, 44, 71), within one unit of the CPU contract. Missing authority matches the legacy frame exactly; a dry basin below sea level differs from the ocean. The existing Render3D GPU probe also passes with the water bindings present.

The same probe minifies alternating water/land stripes with a 25% wet fraction. Its measured colour is within quantization of the area-weighted blend, verifying the water mip path independently of the solid-lake case. These are rendering-contract checks; they do not establish that a procedural river's physical path is plausible.

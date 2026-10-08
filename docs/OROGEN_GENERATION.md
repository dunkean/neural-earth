# Orogen experiments

The Orogen profile uses the original global Orogen mechanisms, including
superplates. It replaces the earlier approximate CUDA tectonic formula.
Large chains are driven primarily by superplate motion (original weights:
95% superplates, 5% fine plates), with collision threshold, density asymmetry,
stress propagation, subduction, back arcs, folded ridges and hotspots.

The `coastal-slope-v1` physical-altitude calibration keeps the original quartic mountain
curve and adds a bounded coastal slope term: `1000*t*(1-t)^4` metres, with
`t=min(elevation,1)` on land. This gives about 9.6 m at native elevation 0.01
and adds at most 81.92 m, keeping sea level and the 6 km summit unchanged.
It prevents extensive sub-metre plains from becoming many tiny islands under
neural refinement, while preserving fine detail. Ocean depth remains linear.
Climate, CUDA and physical-height import use the same conversion/inverse;
existing imported metre heights are not raised. This is a chosen calibration,
not a measured Earth elevation distribution. Source identities invalidate
newly requested atlas/stage caches. The v7 storage namespace remains readable
for historical saved stage references; generate the complete chain to apply
the calibration consistently to relief, erosion and climate.
Resuming historical erosion/climate stages rebases their native graph units
through physical metres before processing; a climate-only command preserves
the historical visible height and evaluates its original physical altitude.

The `orogen-worldclim-v2` adapter supplies physical BIO1 (C), BIO4 (monthly
temperature standard deviation ×100), BIO12 (mm/year) and BIO15 (monthly
precipitation CV in percent), as required by the checkpoints. Their fixed
checkpoint means and standard deviations are unchanged. Temperature uses a
12-month harmonic between the two seasonal month proxies. Rain is divided
equally into six months per half-year, preserving both seasonal totals; this
is the minimum monthly variability compatible with those totals, not measured
monthly climatology. Both standard deviations use the sample correction
`sqrt(12/11)`; BIO15 adds 1 mm to monthly rain as in WorldClim's reference
[dismo::biovars](https://github.com/rspatial/dismo/blob/master/R/biovars.R).

New climate generations preserve precipitation above the global p95 anchor
instead of clipping each half-year at 1,000 mm. The existing 1,000 mm/p95
conversion is a conventional physical scale, not a measured absolute rainfall
calibration for every seed. Only the moisture proxy used for temperature lapse
is bounded to [0,1]. Rain diagnostics use a fixed logarithmic 0–6,000 mm
half-year scale. Historical cached climates retain their previously clipped
values: regenerate climate to obtain the wet tail. Two seasons cannot recover
short monsoons or identify all Earth-like joint climate distributions; this
adapter corrects the unit/statistic contract without imposing a histogram.

The **Génération** panel offers three initial-relief generators: **NN**,
**Tectonique** (Orogen plates), and **Custom** (continental atlas or procedural
continents). The **Érosion** selector offers **Désactivée**, **Orogen · GPU**,
**Orogen · CPU** (the default for tectonic worlds), and **City · GPU**.
Settings has three tabs: **Relief**, **Érosion**, and **Climat**, with shared seasonal climate,
biome thresholds and palette. New UI generations always use Orogen climate;
historical API profiles and saved receipts remain readable for comparison.

Continent layouts are presets, not separate generators. In Tectonique they
initialize plate / continent counts and land coverage. In NN they initialize
height-noise frequency, octaves and land / sea bias, without promising tectonic
continent shapes. In Custom they choose the atlas layout while preserving the
selected atlas / procedural-continent mode.

On imported sources, the original Orogen import pipeline derives diagnostic
regions from connected land and water; these are synthetic regions, not moving
tectonic plates. Only Tectonique creates tectonic mountain chains.

The controls expose plate / continent counts, land coverage, continent-size
variety, angular-motion strength, convergence threshold, graph detail, tectonic
propagation, noise amplitude, warp, smoothing, each erosion type, sharpening,
temperature offset in degrees C and precipitation offset. Defaults match the
original page: 204000 vertices, 80 plates, 4 continents, land coverage 0.3,
variety 0.35, noise 0.4, warp 0.75, smoothing 0.1, hydraulic 0.5, thermal 0.1,
glacial 0.5 and sharpening 0.5. Motion=1, threshold=0.75, propagation=5.
20K vertices is useful for quick experiments; geometry and climate change
with mesh detail. Plate controls affect only the Orogen initial source.
Save A / Show A preserves configurations for comparisons; variant URLs and
cache identities include normalized settings.

The icon menus overlay the map without resizing it. **Settings** groups Relief,
Érosion and Climat. **Rendu** groups Rendu and SNR. They share one unapplied draft.
Switching initial generators preserves common settings. Sliders and exact
numeric entries are linked; palettes use color pickers. Apply regenerates at
the current camera. SNR presets affect only SNR settings. Save A / Show A
compares complete applied configurations, including retained stage artifacts.

## Independent stage commands

**Générer toute la chaîne**, the general menubar Generate button, and **Random**
execute relief → erosion → climate. Random chooses a new seed and starts this
command immediately. Unchanged stage inputs can reuse their immutable cache.

Each Settings tab also has its own generation button:

- **Relief** builds only the new raw source. It becomes the visible height;
  previous erosion and climate artifacts are retained and marked stale.
- **Érosion** starts from the latest raw source, avoiding cumulative erosion.
  Its output becomes the visible height. The climate is retained and marked stale.
- **Climat** restores the graph corresponding to the visible height and computes
  only the seasonal climate. Source and eroded heights remain byte-identical.

A stale artifact is never automatically recomputed by a single-stage command.
Each tab reports its state; the complete-chain command updates all dependencies.
Unapplied parameters in another tab stay in the shared UI draft. The API applies
only the requested stage's parameter group. **Appliquer SNR** applies refinement
settings without running physical generation. A/B switches restore the exact
saved composition, including a deliberately stale climate.

The original mesh/plates/elevation are persisted as validated Node snapshots.
Erosion and climate restore these objects directly; they do not recreate the
sphere, plate motion or initial heightmap. Independent content-addressed stage
artifacts contain input identities, checksummed arrays and backend/settings
identities. The final atlas composes retained fields. Invalid or missing retained
data fails explicitly rather than secretly generating another stage.

`POST /api/generation/run` accepts `seed`, base `profile`, `settings`, `stage`
(`all`, `relief`, `erosion`, `climate`, or `settings`), and `source_profile` for
individual commands. Its response includes the resulting generation profile,
settings, `generation_stages`, and `generation_execution` (true for computation,
false for cache reuse). Immutable references are stored as `orogen_relief_stage`,
`orogen_erosion_stage`, `orogen_climate_stage`, and `orogen_height_stage`;
historical receipts receive neutral reference defaults.

**Maillage · sommets** belongs to Relief and controls the retained graph's
resolution. Changing it requires generating Relief, or the complete chain.
The visible LOD and neural refinement depth belong to Rendu; they are independent
of source mesh detail.

**Bruit du relief · SNR** exposes only elevation conditioning noise. Adaptive
ramps use mean land altitude, temperature (or another existing climate input)
and absolute latitude at the inference window centre. Multipliers of 1 disable
a rule. LOD -3 through 11 have independent experimental relief-noise overrides:
0 inherits the global value, and a positive override scales the local residual
of the reconstructed physical height relative to the global elevation SNR.
This operates after the shared NN generation, before shading, and uses the tile
halo; it does not change the climate or run an independent checkpoint per LOD.
**Activer le SNR adaptatif** enables/disables altitude, climate and latitude
ramps while preserving their values. Disabled rules use the global SNR and skip
regional input sampling. Existing enabled behavior is retained for old receipts;
neutral default gains remain inactive numerically.
**Réglage des détails** selects **Global** or **Par LOD**. Global uses the existing
neural setting; Par LOD additionally scales the reconstructed local height
residual. A zero LOD override inherits Global. This is a display-height operation,
not an independent neural model execution per LOD. Latent previews at LOD 1/2
use their requested LOD setting, rather than the source LOD 3 setting.
Neutral overrides preserve the prior numerical path. The diagnostic SNR layer
continues to show the coarse conditioning policy, including latitude modulation,
rather than measuring the reconstructed image noise.

The shared catalog in `native/orogen/climate-parameters.json` includes 258
settings for pressure, ITCZ, winds, ocean currents, temperature, moisture
advection, orographic rain, rain shadows, zonal rain, Köppen thresholds,
alpine/snow lines and the 30 land-biome colors. The Python schema and Node
pipeline consume this same catalog. Settings are validated, saved in generation
receipts and included in atlas/cache identity; earlier generation links receive
the original climate defaults. Palette, alpine and Köppen settings affect the
**Original Orogen biomes** and **Original Köppen classes** source layers. The
menubar **Biomes · Orogen** entry uses the original source palette and thresholds.

Diagnostic layers include plates, superplates, crust, boundary types,
propagated stress, tectonic contribution, hotspots, orogenic power, back arcs,
folded ridges, postprocessing delta, original biomes and Koppen classes,
seasonal temperature, rain, rain shadow, pressure, continentality, winds and
currents. Wind/current hue represents direction and brightness represents
speed. No neural computation is needed for these source layers.
The original Orogen biome layer follows the initial source atlas. The legacy
physical-terrain Biomes rendering API remains available for old links, but is
not offered as a second biome model in the new layer menu.

## Physical fields and performance

### Optional Orogen CUDA pipeline

The historical CPU graph and CPU erosion remain the default. **Settings →
Relief → Orogen · accélération GPU optionnelle** selects GPU relief, propagation
and spatial lookup for tectonic sources. Érosion exposes **Orogen · GPU** in its
engine selector, plus optional GPU local postprocessing. Climat has its own GPU
checkbox for every source. The six backend flags remain independent and default
off. They persist in generation receipts, A/B and cache identity.

| UI option | Settings/API key | GPU work |
| --- | --- | --- |
| Relief · plaques, collisions, bruit | `orogen_gpu_relief` | Coarse plate projection, collisions, elevation, simplex noise and hotspots |
| Propagation · stress et distances | `orogen_gpu_propagation` | Experimental Jacobi stress and converged shortest-hop distance fields |
| Post-traitement Orogen · passes locales | `orogen_gpu_post` | Warp, bilateral smoothing, sharpening and other independent local passes |
| Érosion → Orogen · GPU | `orogen_gpu_erosion` with `relief_pipeline=orogen` | Experimental graph hydraulic, thermal and glacial erosion |
| Climat · vents, courants, pluie, température | `orogen_gpu_climate` | Pressure, winds, ocean currents, moisture advection, rain/shadow, temperature diffusion and Köppen classification |
| Projection · recherche spatiale GPU | `orogen_gpu_raster` | Exact four-nearest-region spatial lookup using a GPU grid; triangle rasterization was already CUDA |

Install the additional optional runtime in the same Python environment:

```powershell
python -m pip install -r requirements-orogen-gpu.txt
```

This backend uses NVIDIA CUDA and CuPy/NVRTC. CPU defaults never initialize
CuPy. If this optional runtime is unavailable, requested stages use the original
CPU graph and the manifest records the fallback reason. A compilation/execution
failure after the availability probe fails generation explicitly. City uses
its independent WGSL runtime and remains a separately selectable erosion engine.
The complete application still needs CUDA for atlas projection and the NN.

The adapter parses bundled upstream JavaScript and creates temporary copies,
preserving the historical modules and shared climate settings. It translates
independent region loops to CUDA with double intermediates and original typed
array stores. Unsupported loops keep their CPU implementation. Geometry and
unchanged arrays are uploaded once per generation; iterative smoothing,
advection, rain shadow and temperature passes run in batches on resident GPU
buffers. CPU orchestration, triangulation, seeded plate control structures,
ITCZ spline preparation, some ordered traversals/reductions and debug palette
export remain. This is an accelerated hybrid chain, with all six stages optional.

Propagation and erosion deliberately change numerical algorithms. GPU stress
uses simultaneous gather updates, and distance uses shortest-hop relaxation
instead of randomized first visits. GPU erosion uses minimax basin filling,
converged upstream area and implicit incision, followed by simultaneous
sediment/talus and glacier passes. It preserves ocean values and clamps land
nonnegative, checks convergence, and retains the existing strength sliders.
Coarse plate projection starts its walk independently per vertex. These
variants can change relief and therefore climate; identical results across
hardware/drivers are not promised. See [GPU provenance](../native/orogen/gpu/PROVENANCE.md).

Local benchmark, RTX 3090, seed 42, 204K mesh and 2048×1024 atlas. Values are
medians of two repeated generations after the first generation; they include
all source fields, lookup, transfers and atlas rasterization, excluding NN and
cache persistence. First generations measured 8.75, 7.72, 7.73, 5.97 and 7.44 s
respectively. Initial runtime probing/kernel compilation can cost extra.
These measurements use the single-pass benchmark entry point. Interactive
generation also saves independent stage snapshots and builds world metadata;
its end-to-end request time includes that additional work.

| Variant | GPU switches | Median | Relative to CPU |
| --- | --- | ---: | ---: |
| Original CPU graph | None | 8.52 s | 1× |
| Climate and lookup | climate, raster | 7.89 s | 1.08× |
| Local passes | relief, post, climate, raster | 6.79 s | 1.25× |
| All Orogen GPU stages | All six | 5.89 s | 1.45× |
| City erosion with GPU Orogen stages | relief, propagation, post, climate, raster + City | 6.72 s | 1.27× |

On this case, the climate/lookup and local variants reproduced the CPU height,
coast and Köppen fields exactly. All-GPU Orogen changed land/water membership
on 0.41% of pixels, had 149 m mean absolute height difference across the whole
atlas and retained 94.2% of Köppen labels. City changed 2.27% of land/water
membership, with 383 m mean absolute height difference and 89.4% identical
Köppen labels. These are differences from a reference, not quality scores.
Every variant repeated bit-for-bit across the exported physical fields on
this device. Relief previews retain coherent continents and mountain chains;
the resulting erosion appearance is left for visual comparison.

To compare a conservative variant, enable relief, postprocessing, climate
and lookup while retaining CPU propagation/erosion. All six switches exercise
the fastest measured experimental variant. The alternative City engine can
also run with GPU climate and lookup; its own dose/iterations control erosion.

```powershell
python benchmark_orogen_gpu.py --repeats 2
python benchmark_orogen.py --gpu-stages relief propagation post erosion climate raster --no-persistence
```

The matrix benchmark writes timings, receipts and physical comparisons to
`output/orogen/gpu-comparison/comparison.json`, plus per-variant relief PNGs
and float fields. It checks every exported field on repeat. The focused
`test_terrain_orogen_gpu.py` suite also checks climate against CPU, exact lookup
against SciPy at seams/poles, immutable cache replay, missing-runtime fallback,
City integration, thermal mass conservation and zero-motion/zero-erosion controls.

### Optional City surface erosion

City is a different erosion engine, adapted from the GPU surface pass in the
neighbouring `city_generator/rust` prototype. Its GPU code is WGSL, executed
here with native `wgpu-py`; the bundled snapshot runs without the sibling
checkout, a browser, or a new Rust build. Install the optional runtime with
`python -m pip install -r requirements-city-gpu.txt`. Orogen CPU erosion does
not import or initialize it. GPU absence or nonconvergence produces an explicit
error with Orogen as the available alternative.

City exposes dose 0–2, 1–64 iterations, talus slope and a physical drainage
scale in km. Defaults are 1, 12, 0.6 and 300 km. These are City settings;
Orogen hydraulic/thermal/glacial sliders do not control this engine.
The settings, GPU/runtime identity and source hashes participate in atlas
and world identity. Existing configuration links receive neutral new defaults
while keeping their original canonical token. Save A / Show A includes the
chosen engine and its settings.

The pipeline is original source relief → CUDA atlas projection → City GPU
erosion → eroded heights sampled onto the retained original sphere → original
Orogen climate → final source fields / NN conditioning. Tectonic plates and
superplates remain original; imported sources retain their synthetic regions.
No Orogen postprocessing is applied on top of City. Dose zero bypasses the
City device and preserves the existing unprocessed source path exactly.

City retains split D-infinity flow, upstream accumulation, implicit stream
power incision, thermal relaxation and diffusion. It operates on a rectangular
atlas with periodic longitude and latitude-dependent distances/areas. Sea
samples remain exact and land remains nonnegative; polar boundary rows are
retained. It does not implement Orogen glaciers or the City mountain-generation
and hydrology stages. Closed bowls remain closed in this surface variant.
It is an adaptation of the planar engine, not CPU/GPU numerical parity with
the City bench. The source atlas keeps the exact eroded raster; climate remains
limited by graph sampling and the original 6000 m altitude convention.
The existing erosion diagnostic displays City deltas in metres when selected
(blue incision / red deposition, saturated at 1000 m).

Measured on the local RTX 3090, seed 42, 204K mesh, 2048×1024 atlas, 12 City
iterations: native erosion including encoding/upload/readback about **0.57 s**,
of which submit/readback about **0.27 s**, with **240 MiB** of native WebGPU
buffers. A 128-pass tiled budget converged (maximum 86 steps). Two complete
generations were bit-identical on this device. Separate CUDA projection
workspace is about 246 MiB; these are distinct allocations, excluding NN.
First native shader initialization costs roughly two additional seconds.

The initial integration, with CPU Orogen climate, measured full creation
without persistence at **9.67 s for City** and **8.91 s for Orogen** in single
cases. These historical timings predate the optional Orogen CUDA backend above.
City adds an intermediate relief projection and resampling before climate;
the new matrix reports its combination with GPU Orogen passes separately.
The algorithms, erosion schedules and resulting terrain differ.
The original Orogen composite erosion alone measured 0.57 s, with total
postprocessing 0.78 s including warp/smoothing/sharpening/creep.

Reproduce without writing the large persistent cache:

```powershell
python benchmark_orogen.py --seed 42 --erosion-engine city-gpu --no-persistence
python benchmark_orogen.py --seed 42 --output output/orogen/gpu-feasibility --no-persistence
```

The first writes `output/orogen/benchmark-city-gpu.json`; the second keeps a
separate Orogen baseline. Both measure the real pipeline, check a full repeat
and export preview/float fields. See `native/city_erosion/PROVENANCE.md` for
source attribution and exact adaptations.

### Initial feasibility audit and remaining CPU work

The audit below guided implementation of the optional backend above. CPU
measurements use original Orogen, seed 42, 204K vertices; timings are illustrative
single runs, excluding NN and persistence.
Wind/ocean/precipitation/temperature rows are top-level totals, not a sum of
their nested timing records.

| Stage | Current time | GPU feasibility and required change |
| --- | ---: | --- |
| Sphere mesh / Delaunay | 0.26 s | Keep triangulation on CPU initially; transfer reusable geometry once. Building identical Delaunay topology entirely on GPU is a separate substantial project. |
| Coarse plates / superplates | 0.14 s | Small seeded graph/control structures; CPU is inexpensive. Per-vertex plate aggregation can later move to GPU. |
| Project coarse plates | 0.74 s | Strong target: simplex noise plus nearest coarse-region walks per vertex. Preserve permutation tables/seed and replace the shared warm-start walk with independent lookup. |
| Elevation / tectonics | 1.18 s | Collisions and relief/noise assignment parallelize well; graph propagation and randomized traversal need explicit semantic decisions. |
| Orogen postprocessing | 0.78 s | Warp, smoothing, thermal transfer, sharpening and creep are good kernels. Priority flood, sorted downstream solves and in-place glacier widening require graph algorithms or a changed numerical scheme. City already provides an alternative surface engine. |
| Wind | 0.45 s | Pressure, gradients, geographic reductions and smoothing are good GPU work; retain the tiny ITCZ spline control step on CPU initially. |
| Ocean currents | 0.32 s | Vector rules/smoothing parallelize; coast distances use frontier traversal and normalization uses selection/reduction. |
| Precipitation | 1.40 s | Priority target: advection already uses ping-pong buffers; incoming moisture is gathered from neighbours, avoiding floating atomics. Gradient/mechanism/smoothing passes follow. |
| Temperature | 0.38 s | Per-vertex formulas and coastal diffusion are direct kernels. |
| Köppen / palette | 0.02 s | Straight per-vertex classification; low absolute cost but keeps fields resident. |
| Atlas nearest lookup | 0.91 s | Current `cKDTree` is CPU. Cache geometry-dependent lookup for climate/erosion variants; later use a spherical spatial index on GPU. Avoid an all-pixels × all-vertices dense search. |
| Final triangle rasterization | 1.41 s | Already CUDA, but repeatedly uploads/query-chunks and reads each field. Retain geometry/data, fuse kernels, group readback and export only required layers. |

Two important correctness constraints come directly from `elevation.js`:
`assignDistanceField` randomly selects the next queue item, so it is **not**
ordinary shortest-path BFS. `propagateStress` updates stress and the associated
subduction factor in place, so a parallel ping-pong maximum is not identical.
Treating either replacement as an exact GPU port would silently change relief.
Similarly, original erosion sorts by height and deposits/updates recipients in
place; those dependencies cannot be removed just by changing array libraries.

Seasonal climate, projection/noise/collisions, independent local passes, graph
alternatives and the spatial index are now implemented as optional CUDA stages.
Triangulation, small plate structures, ITCZ setup, remaining ordered traversals
and reductions still run on CPU. Transfers and the final multi-field atlas
projection remain useful optimization targets. Stage-specific reuse is now
implemented: new climate settings restore the retained height graph and unchanged
physical stages reuse their independent immutable caches. Atlas exports and
palette work still incur projection and persistence cost.

For a future portable Orogen backend, native WebGPU/WGSL is consistent with City
and could share shaders with a browser. This implementation chose CUDA for the
existing NVIDIA neural server and retains explicitly selectable CPU stages. WGSL
concrete floats are f32/f16 and atomic types are integers, so use gather passes
and deterministic reductions rather than assuming JavaScript double precision
or portable floating-point atomic adds. See the
[WGSL type specification](https://www.w3.org/TR/WGSL/#types).
Repeatability on one GPU does not imply byte parity across CPU/GPU/drivers;
backend and algorithm version must remain explicit in persistent identity.

The application currently selects CUDA before building the atlas and loading
the NN (`terrain_device.py`, `terrain_app.py`). A CPU-only standalone Orogen
generator therefore also needs an optional atlas rasterizer/device selection;
supporting the complete NN application without CUDA is a separate scope.

### Original Orogen reference

The global original graph simulation runs once on CPU with bundled Node.js,
without CDN/network dependencies. CUDA projects original spherical triangles
to a 2048 by 1024 atlas. The existing CUDA neural generator samples those
physical fields at arbitrary coordinates. Sea heights are 10000*e metres;
land heights follow `6000*t^4*(5-4*t)+1000*t*(1-t)^4`. No quantile remap.
The atlas wraps longitude; the existing neural map's edge behavior is unchanged.
The atlas's native pixel spacing is about 19.53 km; it cannot promise
tectonic features finer than its source mesh. Neural detail remains at 30 m.

Annual temperature is the mean of original seasons; annual precipitation is
their sum using Orogen's 1000 mm/p95 half-year anchor. BIO4 uses a harmonic
monthly cycle and BIO15 equal monthly rain within each half-year, using the
WorldClim sample-standard-deviation and rain+1 conventions described above.
The seasonal arrays and original Koppen classification
are retained separately. Imported land heights use the matching calibrated inverse and saturate at 6000 m
inside the Orogen graph; preserving the original source keeps the metre
heightmap unchanged but does not remove that climatic altitude limit.
Climate is evaluated on the processed initial relief,
not re-simulated against every neural fine-scale height correction.

Seed 42 on RTX 3090, 204K vertices: global pipeline about 6.5 s, spatial lookup
about 0.9 s, CUDA rasterization about 1.7 s, workspace about 253 MiB, excluding
NN inference. Two complete generations were bit-identical. The original graph
worker and adapter were independently compared at 20K vertices: elevation,
four seasonal climate fields and Koppen arrays were all bit-identical.
Atlas files store float32 fields without compression for fast experiments;
allow several hundred MB per cached variant. Measured full creation including
writing is about 13 s; disk replay is about 1.5 s (in-memory reuse is immediate). Fields are hashed, validated,
immutable and reused in memory. Run `python benchmark_orogen.py --seed 42`
for local creation/replay measurements, preview PNGs and a JSON report under
`output/orogen` (ignored by Git).

Original reference page: http://127.0.0.1:8770 . Small seeds up to 2^31-1 use
that same original numeric seed; larger u64 seeds are hashed to the original
numeric RNG range. See `native/orogen/PROVENANCE.md` for source adaptations.

The World menu selects the source family (original NN inputs, custom continents,
or Orogen). Settings groups source parameters separately from NN refinement;
the NN refines every source. Independent elevation/climate/erosion combinations
remain available under Advanced source overrides.

The five SNR map layers display the allowed conditioning noise amplitude used
by coarse NN windows, including the quantized altitude/climate multipliers.
They use the same `window_snr` calculation as inference, without running a model.
Cells select the nearest window centre (64 samples, stride 48, 7680 m/sample);
they show a window's policy, not a pixel SNR measured from generated terrain.
The color scale is logarithmic and follows the selected channel's baseline and
configured gain range. Neutral multipliers produce a uniform layer. The scale
remains in the compact HUD; the LOD stays in the toolbar, and loading uses a
spinner. The map's explanatory banner is reserved for errors.

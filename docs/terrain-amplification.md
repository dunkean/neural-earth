# Terrain amplification

Experimental geometric detail below the 30 m neural DEM, from LOD -1 (15 m) down to LOD -12 (7.32 mm).
Four modes share one tile pipeline: the existing **classic** refinement, and three amplification modes,
**hybrid** (recommended), **Phasor** and **MSE**. All are deterministic by seed and seamless across tiles.
This page explains the generation path and how to use it. Reference details (headers, payload layout,
cache keys) are in [Layers](layers.md#experimental-terrain-amplification) and
[Architecture](architecture.md#rendering-and-polar-sampling).

## Generation path

### LOD scale

LOD `k` has a cell size of `30·2^k` m. Tiles are 256 samples plus a 24-sample halo (304² requests).

| LOD | Cell | What produces the heights |
| --- | --- | --- |
| ≥ 0 | 30 m and coarser | Neural DEM (latent + decoder); unchanged by amplification |
| -1 … floor | 15 m … 0.47 m | Selected mode: classic refinement or an amplification sampler (MSE keeps classic at -1) |
| floor+1 … -12 | 23 cm … 7.32 mm | C2 interpolation of the floor LOD, no new geometry |

The **relief floor** is the finest LOD a mode is allowed to generate. It is set per mode in
`backend/terrain_lod_interpolation.py::RELIEF_FLOOR` and is currently **-6 for all modes** (the current
limit of the generators). It is an empirical policy: a mode generates relief as long as no grid appears,
then the server interpolates. The seed-42 mountain captures and the lattice detector
(`tmp/e2e/interp/detector-by-mode-lod.csv`) found no repeating lattice down to -6 in any mode.
Fine detail below the floor is meant to come from ground textures, not geometry.

### Common rules

- **Parent consistency.** A finer level adds detail without moving coarser features: each mode's delta is
  zero-mean and band-limited above the parent grid's frequencies, so a LOD -k tile averaged down matches
  LOD -(k-1). Measured adjacent-LOD mismatches are below 0.12 m RMS (Phasor/hybrid) and a few mm (MSE).
- **Seams and determinism.** Every sampler reads a window larger than its stencil reach and is anchored to
  global coordinates, so any two windows agree bit for bit on their overlap, including negative
  coordinates. Results depend only on seed, world profile, mode, version and intensity.
- **Cache identity.** Amplified tiles append `/amp-<mode>-<version>-<intensity>`; interpolated tiles
  append `/interp-from-<floor>-v1`. Old caches are never reused after a version change.

### Classic (existing refinement)

`backend/terrain_refinement.py`, unchanged: a C2 cubic B-spline of the native DEM in signed-sqrt height,
the decoder octave at LOD -1, and hashed sparse-convolution procedural octaves from LOD -2 to -6, with
coast handling. This is the default and the "before" of the A/B comparison.

### Phasor (Grenier et al. 2024)

`backend/terrain_phasor.py` (`phasor-v3`). Oriented phasor noise, after *Real-time Terrain Enhancement
with Controlled Procedural Patterns*:

1. Base: C2 spline + decoder octave (procedural octaves off).
2. Orientation from the smoothed gradient of the parent relief.
3. Cascade of three levels, wavelengths **60 / 20 / 6.67 m** (Horton ratio 3), each below the native
   60 m Nyquist so it never rewrites relief the neural DEM already resolves. Rounded-triangle profile,
   analytic Jacobian per level, `a2 = 0.2·r1` so narrow gullies vanish in wide valley floors, crest fBm.
4. Irregularity: per-impulse frequency (0.8–1.25×) and orientation (±18°) jitter, patchy amplitude,
   gully heads faded below crests.
5. Zero-mean displacement, amplitude caps 12 / 2.4 / 0.48 m, bounded by local relief.

Each level fades in once the grid resolves it, so the 6.67 m level only appears from about LOD -4.

### Hybrid (recommended)

`backend/terrain_hybrid.py` = Phasor heights driven by real drainage + MSE surface channels.

- **Drainage.** `terrain_amplification.sample_drainage` routes flow on the native 30 m grid
  (multiple-flow-direction accumulation, contributing area in m²).
- **Heights.** `sample_phasor(..., drainage=True)` (`phasor-drainage-v1`): ravine amplitude is strongest
  along and near channels above an area threshold and fades with distance (Grenier fig. 6); gully heads
  start below a contributing-area threshold; interfluves, flats and plateaus stay almost unchanged;
  orientation blends toward the drainage direction near channels.
- **Surface.** `terrain_amplification.sample_surface` (`surface-v1`) computes four guide channels from the
  final heights, without changing them:

  | Channel | Meaning |
  | --- | --- |
  | flow | Narrow channel lines (native drainage + fine downhill concavity) |
  | deposit | Valley floors, fans and slope feet (low slope, concave, high drainage) |
  | scree | Talus aprons below rock, 25–38° slope band |
  | rock | Steep and sharp faces with low drainage |

### MSE (Schott et al. 2024, experimental)

`backend/terrain_amplification.py` (`mse-v5`), a port of *Terrain Amplification using Multi-Scale
Erosion*: a native drainage stage, then stream-power erosion, thermal erosion (noisy repose angle) and
deposition at 7.5 m (LOD -2), 1.875 m (-4) and 0.47 m (-6), each stage inheriting the parent's drainage.
Run physically inside finite tile windows, it only carves **centimetres** of relief: the method needs
whole-massif simulations with many iterations, which on-demand tiles cannot afford. Earlier versions
that amplified the result produced artificial patterns (bubbles, fingerprints). The module is kept as the
drainage and surface provider for hybrid, and as a mode for comparison.

### Below the floor

`backend/terrain_lod_interpolation.py`: the server reads 128-sample parent blocks at the floor LOD with the
selected mode and evaluates biquintic Hermite patches that share derivatives up to order (2,2). Heights
are C2, pass exactly through the floor samples, and cliff overshoot is bounded to 2 cm. Surface channels
are upsampled the same way. Evaluation stays in float64; the viewer also requests float64-derived slopes
(`gradients=1`, `rg32float`), which removes shading bands caused by FP32 height rounding at millimetre
spacing.

### Effects on other layers

Relief, Render (materials, snow, vegetation), Soils, biome and Köppen altitude corrections, contours,
cursor height and hydrology display all read the selected heights. Render also uses the surface
channels: rock raises bare outcrop, deposit tints sediment, scree adds pale talus, flow darkens wet
channels and thins vegetation. Pure climate fields (temperature and precipitation sources, regional
pedology, river routing) are unchanged: amplification does not simulate new climate.

## Usage

### Starting the server

Follow [Installation](installation.md), then `python launch_terrain.py`. On machines where the system CUDA
toolkit is older than the NVRTC bundled with PyTorch (e.g. CUDA 12.4 headers with a cu128 wheel), CuPy's
CUB kernels fail to compile when opening an Orogen world (`incomplete type "__nv_fp8_e8m0"`). Start the
server with CuPy accelerators disabled:

```sh
CUPY_ACCELERATORS= python launch_terrain.py
```

or install headers matching the PyTorch CUDA version (`pip install "cupy-cuda12x[ctk]"`).

### Viewer

Open **Rendering**. The amplification selector offers:

- **Classique (raffinement actuel)**: the existing refinement (default);
- **Hybride (recommandé)**;
- **Phasor (Grenier 2024)**;
- **MSE (expérimental, Schott 2024)**.

The intensity slider goes from 0 to 2 (default 1). Phasor and hybrid depth grows sublinearly above 1:
intensity 2 gives 4/3 of the depth. Zoom past 30 m/pixel (LOD -1 and finer) to see the effect; use
**Render LOD** to force LOD -1 … -12 in the current view. Layers **Relief** and **Render** show it best.

| Key | Action |
| --- | --- |
| `B` | Before/after for the selected mode |
| `Shift+B` | Cycle before → hybrid → Phasor → MSE → before |

The badge at the top shows which modes are displayed. Settings persist in the browser and travel with
share links. The WASM edition does not offer amplification.

### URL and API parameters

| Parameter | Values |
| --- | --- |
| `amplify` | Intensity 0–2; 0 or absent = classic |
| `amplify_mode` | `hybrid` (default when `amplify` > 0), `phasor`, `mse` |
| `amplify_before=1` | Records that the A/B toggle shows the "before" side |

The same parameters apply to `/height/...bin`, `/tiles/...png` and `/api/view`. World metadata publishes
`amplification.modes` (availability, versions) and `relief_floors`. Responses carry
`X-Terrain-Amplification*`, `X-Terrain-Relief-Floor-LOD` and, for surface data,
`X-Terrain-Surface-Format: rgba8unorm` with channels `flow,deposit,scree,rock`.

### Changing the relief floor

```sh
TERRAIN_RELIEF_FLOORS='{"classic":-3,"hybrid":-4}' python launch_terrain.py
```

Integers from -6 to -1; restart the server. Use this to retest a mode when a grid appears.

### Benchmarks and verification

| Command | Output |
| --- | --- |
| `python tools/benchmarks/amplification_bench.py --seed 42` | MSE and surface sheets, spectra, timings in `tmp/amplification/` |
| `python tools/benchmarks/phasor_bench.py --seed 42 --real-only --hybrid` | Classic / Phasor / hybrid sheets and zoom strips in `tmp/phasor/` |
| `node tools/verification/verify_amplification_live.cjs` | Live captures, seam ratios, timings, contact sheets in `tmp/e2e/` |

The live script reads `BASE_URL` (e.g. `http://[::1]:8765`), `E2E_SITES`, `E2E_MODES`, `E2E_LODS`,
`E2E_LAYERS`, `E2E_GRID_DETECTOR=1` and `E2E_OUT`. Regressions:
`tests/python/test_terrain_{amplification,phasor,hybrid,amplification_delivery,lod_interpolation}.py`
and `tests/browser/test_terrain_amplification.cjs`; CUDA gates need `TERRAIN_TEST_CUDA=1`.

## Performance

RTX 4090/5090, 304² requests:

| Step | Cold | Warm |
| --- | --- | --- |
| Hybrid tile, LOD -2 … -6 | 0.4–1.5 s (first drainage of the area) | 48–92 ms |
| Phasor tile | ~0.6–0.8 s per view | 21–31 ms |
| MSE tile | 0.4–3.4 s | 11–88 ms |
| Surface channels (extra) | — | 20–34 ms |
| Interpolated tile, LOD -7 … -12 | 28–80 ms median | — |

A complete view at LOD -12 loads in about 0.5–0.7 s.

## Limitations

- Detail is plausible procedural geometry, not measured terrain.
- Drainage is computed in finite windows, not as a global river simulation.
- MSE heights are centimetre-scale; MSE is useful for its channels, not its relief.
- At LOD -5/-6, hybrid and Phasor look smoother than classic, which keeps its procedural octaves.
- Below the floor, zooming only magnifies the floor relief; textures are expected to carry that scale.
- Across separate server runs, coastal LOD -6 heights were observed to differ by up to 0.285 m in every
  mode, including classic, upstream of interpolation. This is under investigation.

## Porting notes

Phasor and the interpolation are closed-form per sample and need no simulation, so they port directly to
WGSL or Godot shaders. Hybrid additionally needs the native 30 m drainage field, which can be precomputed
per region and shipped as a texture.

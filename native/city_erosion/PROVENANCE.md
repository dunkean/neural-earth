# City surface erosion

`surface.wgsl` adapts the GPU surface pass from
`../city_generator/rust/bridge/terrainErosion.wgsl`, captured on 2026-10-08.
The exact source SHA-256 is recorded in `source.json`. The original GPL-3.0-only
license is retained in `LICENSE`. There is no runtime dependency on that checkout.

The original CPU counterpart is `rust/crates/core/src/erosion.rs`,
`erode_surface` / `erode_shape_coarse`; the GPU schedule is
`rust/bridge/terrainErosion.ts`, the `coast` / surface path. The GPU implementation
is WGSL, not a Rust native CUDA library. This adapter executes those GPU kernels
with the optional native Rust-backed wgpu runtime exposed by `wgpu-py`.

Retained mechanisms: D-infinity split flow, upstream accumulation, downstream
implicit incision, bounded catchment response, thermal transfer, channel-aware
diffusion, 16 by 16 tiled fixed-point solves, active tile flags, convergence
validation and budget retries. Dispatches remain synchronization boundaries.
The engine is distinct from Orogen's graph-based hydraulic/glacial erosion.

Adaptations for the worldwide atlas:

- Import physical metre heights directly; remove generation, uplift, noise,
  mountain/canyon construction, reconstruction and percentile normalization.
- Rectangular 2:1 raster instead of a square regional raster; longitude wraps
  in neighbour lookup, shared tile halos and tile wakeups.
- East/west distances and cell contributing areas use cosine of latitude.
  The outermost polar rows are retained boundaries; the metric uses a 0.01
  minimum latitude scale. This is a spherical atlas adaptation, not exact
  parity with the planar City bench.
- Original ocean/zero samples remain exact; erosion of land drains at sea
  level rather than the ocean floor. Land remains nonnegative. Thermal and
  diffusion stencils exclude sea samples.
- Dose, iterations, talus and drainage scale are independent of Orogen sliders.
  Closed basins remain closed in this surface pass. It does not import City
  terrain generation, basin-flood mountain evolution, hydrology, or glaciers.

The native runtime and GPU information participate in the generation identity.
The complete bundled source snapshot and adapter are hashed by Orogen. GPU
absence, initialization failure or nonconvergence are reported; the selected
engine is never silently replaced. Orogen remains the default CPU erosion.
The existing atlas projection and neural pipeline still require CUDA/NVIDIA.

Runtime API: [wgpu-py GPUDevice](https://wgpu-py.readthedocs.io/en/stable/generated/wgpu.GPUDevice.html).
Shader semantics: [WGSL specification](https://www.w3.org/TR/WGSL/).

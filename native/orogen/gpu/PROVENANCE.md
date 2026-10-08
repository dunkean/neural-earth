# Optional Orogen CUDA adapter

The historical `../vendor` JavaScript is the reference. It is not rewritten in
place. `build.mjs` parses a temporary copy and identifies independent region
loops in plate projection, collisions, elevation/noise, postprocessing and all
five physical climate stages. It keeps the original formulas and shared
climate catalog. The generated copy exists only for a requested GPU variant.

Acorn 8.15.0 is bundled from the npm release (`dist/acorn.mjs`), with its MIT
license in `ACORN-LICENSE`; no npm install or download is required at runtime.
`runtime.mjs` emits CUDA C++ for the supported numeric subset of those modules.
Numeric intermediates use double precision; array stores preserve original
typed-array rounding. The original SimplexNoise permutation is uploaded.
Unsupported loops stay on CPU and are listed in generation metadata. Compilation
or execution errors fail the generation rather than silently accepting bad fields.

The optional Python dependency is CuPy 13.6.0 (`requirements-orogen-gpu.txt`).
NVRTC compilation and CUDA availability are probed only when a GPU option is
requested. If that runtime is absent, the original CPU graph is used; the
requested settings and fallback identity/reason remain in the cache receipt.
Kernels are cached within the process and by CuPy on disk. The primary app still
requires CUDA for atlas projection and the neural model.

Transfers use a framed local subprocess pipe. Array identities and digests avoid
reuploading unchanged geometry. Iterative smoothing, moisture advection, shadow
propagation and coastal diffusion execute as batches on resident buffers, with
readback at the batch boundary. Orogen's Node orchestration, mesh construction,
small seeded structures, ITCZ spline preparation, some graph traversals/reductions,
and debug-palette export remain CPU work. This is not a claim of zero CPU work.

`terrain_orogen_gpu.py` and `erosion.cu` additionally implement explicitly
experimental alternatives:

- stress propagation uses simultaneous gather/Jacobi passes;
- distance fields use converged shortest-hop relaxation instead of the original
  randomized first-visit traversal;
- erosion uses minimax basin fill, descending receivers, converged upstream-area
  accumulation and implicit incision, followed by simultaneous sediment/talus
  and latitude/elevation glacier passes. It replaces the CPU priority-flood
  canyon carving and order-dependent glacier widening/deposition. Oceans stay
  unchanged; land is clamped nonnegative. All iterative graph solves check
  convergence. Water-free worlds skip basin filling and retain closed minima;
- coarse plate lookup starts independently per vertex rather than sharing a
  warm-start walk across vertices.

These change terrain, even with the same seed. Every stage is disabled by
default and has its own persistent switch. The CPU erosion engine remains the
default; the separately attributed City surface engine is another option.

`nearest.cu` replaces the CPU tree with a GPU uniform grid and expanding cube
shells. It stops only when a distance bound excludes any closer unseen vertex.
Distances use double precision and ties use vertex ID. This lookup is checked
against SciPy's independent tree, including seam and polar samples.

The adapter and kernel algorithms are adaptations of GPL-3.0 Orogen and retain
the license bundled in `../vendor/OROGEN-LICENSE`. No upstream GPU revision or
cross-device bit parity is claimed. Sources, runtime/device identity, flags and
physical field hashes participate in persistent world/cache identity.

API references: [CuPy RawKernel](https://docs.cupy.dev/en/stable/reference/generated/cupy.RawKernel.html),
[CuPy installation](https://docs.cupy.dev/en/stable/install.html).

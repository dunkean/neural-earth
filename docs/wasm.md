# Neural Earth in the browser

The browser edition is a static application: Orogen, inference, caches, sharing
and rendering run on the user's machine. There is no Python process, PyTorch,
CUDA service or application API at runtime. HTTPS hosting or a local static
HTTP server supplies the HTML, JavaScript, WASM and weight files.

The existing Python/PyTorch edition remains available through `launch_terrain.py`.

## Build and run

Install Node.js 20+, Rust 1.99+, the `wasm32-unknown-unknown` target and `wasm-pack`:

```sh
rustup target add wasm32-unknown-unknown
cargo install wasm-pack --locked
npm ci
git submodule update --init native/magna-urbis
npm run build:wasm
npm run serve:wasm
```

Open **http://127.0.0.1:8780**. `PORT` changes the static server port.
`start-neural-earth-wasm.cmd` / `.sh` builds on first use and starts that server.
The output directory, **`wasm/dist/`**, can also be uploaded to static hosting.
Opening the HTML with `file://` does not provide the browser security context
required by module workers and WebGPU.

Use a browser with WebGPU and graphics acceleration enabled. **`shader-f16`**
is optional: the acquired GPU device determines whether activations and
convolution weights are stored in FP16 or FP32. Every kernel computes and
accumulates in FP32. Subgroups are not used. The application distinguishes
insecure hosting, a missing WebGPU API and an inaccessible GPU adapter before
loading weights. This is not a CPU-only inference build.

The first download includes about **162 MB of prepared weights** and the inference
WASM module. Weights have SHA-256 checksums and use the browser Cache Storage when
space is available. Models also occupy GPU memory. No minimum hardware or speed
is certified. The NF4 base reduces storage and download size while retaining
the native network architecture and computation cost. FP32 fallback uses more
GPU memory than FP16.

## Prepared weights

The browser consumes the existing Rust coarse/decoder bundles and the **same
distilled NF4 B75 base artifact as Python**:

| Bundle | Model |
| --- | --- |
| `coarse.safetensors` | Coarse4 distilled solver, four network evaluations |
| `base.nf4` | Distilled NF4 B75, native two-step solver |
| `decoder.safetensors` | Distilled decoder |

For coarse/decoder, the builder reuses `../city_generator/rust/pkg/neural-weights`
when available. Set `NEURAL_EARTH_WASM_WEIGHTS` to another prepared bundle
directory. Otherwise the small `native/neural-earth-weights` tool reuses the
pinned Rust exporter's readers with this repository's `distill/weights/coarse4.pt`
and `decoder.pt`.

For base, `NEURAL_EARTH_WASM_BASE` selects an installed packed release directory;
the default is `DISTILL_ROOT/releases/base-nf4-b75-v1` (`~/data/distill` by default).
Missing release files are downloaded with the same integrity pins as Python's
`distill/compact_release.py`. The builder preserves the packed payload, adds a
manifest header, and the browser decodes it in Rust. NF4 kernels already include
normalization; sensitive layers preserve the anchored output gain contract.
The original base checkpoint is never loaded, exported or shipped. Export needs
no Python. Weights and generated builds are ignored by Git.

## Shared implementation

* **Viewer:** `web/index.html` and the existing toolbar, generation, camera, LOD,
  measuring, pins and sharing modules are used by both editions. The builder
  changes only the bootstrap. `web/terrain_transport.js` supplies normal HTTP
  requests in Python mode; `wasm/client.mjs` supplies worker RPC in browser mode.
* **Inference:** `native/magna-urbis` pins the existing upstream Rust port. The
  build compiles its noise, window plan, solvers, reconstruction and safetensors
  implementation. `tools/wasm/prepare-engine.mjs` applies a checked integration
  overlay: external planetary conditioning, independent X/Y origins, staged
  outputs, window caching, rectangular reconstruction and the native two-step
  base solver. Generated source is never edited or maintained separately. The
  overlay fails if an expected upstream integration point changes.
* **Networks:** the coarse/base EDM UNets and the decoder student run on a small
  direct WebGPU runtime, `native/neural-earth/src/engine` (Rust, `wgpu`, WGSL),
  with the same graph as the upstream Burn implementation. Convolutions are
  implicit GEMMs over NHWC activations; `mp_silu`, embedding modulation, pixel
  norms, biases, residual sums, clamps and skip concatenations are fused into
  them, and each readback is a single queue submission. Burn's portable WebGPU
  convolution is a naive direct kernel (autotuning only selects among CUDA/Vulkan
  matrix-unit kernels), which made a 30 m tile take minutes. The GEMM tile can
  change without changing outputs: the accumulation order is fixed.
* **World generation:** the browser runs the same maintained
  `native/orogen/vendor/` sources as Python's Node adapter. The static build only
  converts the existing Delaunator UMD wrapper to an ES module. Raster projection
  is Rust/WASM, with a spherical spatial index and bilinear atlas reads.
* **Control schema:** `wasm/generation-schema.json` is an export of Python's
  canonical settings. After changing Python defaults or climate controls, use
  `npm run build:wasm -- --refresh-config` with the existing Python environment.
* **Rendering:** terrain shading and isolines run in the inference worker's WASM;
  perspective globe projection and lighting run in the small separate
  `native/neural-earth-render` WASM module. The globe's navigation and coordinate
  handlers are taken from the shared viewer at build time. Canvas presents the
  RGBA images. Orogen still runs as JavaScript, as it does in the Python edition.

The network pipelines are compiled by one tiny pass of each network while
Orogen builds the world in its own worker, so the first neural tile does not
pay shader compilation.

## Current capabilities and differences

The browser edition supports spherical Orogen worlds, full world regeneration,
map/globe navigation, neural coarse → latent → decoder refinement down to **30 m**,
signed sea heights, the source elevation layer, relief, biomes, temperature,
precipitation, WASM terrain shading, isolines, measuring, pins and portable
generation/camera recipes. Physical tiles have the same 24-cell halo convention
as the Python viewer. Overlapping network windows use global coordinates and
deterministic noise; window caches are bounded to 128 MiB and DEM tiles to 64 MiB.

This is a browser runtime, not certified numerical or visual parity with CUDA.
Python inference uses BF16; the browser stores FP16 or FP32 and computes in FP32.
Its FP32 path matches the upstream Burn implementation in FP32 to 0.002 m on
test tiles; FP16 storage stays within a few metres of it. The browser atlas currently
uses four-nearest spherical interpolation rather than CUDA's barycentric raster.
The WASM material compositor is a simpler climate/slope palette than the Python
and WGSL surface compositor. World identities are therefore runtime-specific.

The following Python features are not yet available in this edition: separate
retained Orogen stages, non-Orogen source profiles, rotated polar neural charts,
sub-30 m refinement, soil/pedology and the extended climate/geology diagnostic
layers, selectable checkpoint variants, multi-GPU scheduling and global coarse
background preparation. Unsupported controls/layers are disabled or return an
explicit local error; they do not call a backend.

## Verification

```sh
npm run test:wasm
cargo test --manifest-path native/neural-earth-render/Cargo.toml
cargo test --manifest-path native/neural-earth-weights/Cargo.toml
npm test
# With the static server running, exercise the actual checkpoints:
npm run verify:wasm
# Repeat with shader-f16 and subgroups hidden from the real GPU adapter:
# PowerShell: $env:NEURAL_EARTH_WEBGPU_PROFILE='portable'; npm run verify:wasm
NEURAL_EARTH_WEBGPU_PROFILE=portable npm run verify:wasm
```

The browser check runs actual coarse and decoder tiles on high land at unequal
global X/Y origins, checks overlapping halos and exact repeated outputs after
clearing all neural caches, the globe and the absence of network API requests. It writes evidence under
`tmp/wasm-browser.*`. This is deliberately separate from the fast unit tests.

The verification uses normal browser startup without experimental WebGPU flags.
The portable profile hides both optional features, checks that no device request
or compiled shader requires them, and exercises real FP32 GPU inference. Requests
include a 304² raster with its halo; uncached fine exploration remains expensive
because the distilled base retains the native network architecture.

Local measurements (normal Chrome, RTX 3090, not a portable speed guarantee):
an uncached 30 m mountain tile takes about 2.0–2.2 s with FP16 storage and 2.3 s
in the portable FP32 profile (previously 104 s and 295 s with Burn); its
neighbour, sharing cached base windows, about 0.8 s; the first coarse tile
0.3 s. A 30 m tile is about 15–18 TFLOP of convolutions, mostly the two-step
base. The inference WASM module is 0.7 MB.

`npm run bench:wasm` (with the static server running) times uncached 30 m tiles
in Chrome and reports GPU wait and dispatch counts. `NEURAL_EARTH_TILES` lists
GEMM tiles to compare (`bm,bn,bk,tm,tn;…`, empty for the default). The native
benchmark runs the same engine on the local GPU with a synthetic atlas:

```sh
cd native/neural-earth
cargo run --release --example bench                   # coarse and 30 m tiles
NEURAL_EARTH_PROFILE=1 cargo run --release --example bench -- 0 1600 -640   # per-kernel times
NEURAL_EARTH_FP32=1 cargo run --release --example bench -- --dump out.bin  # FP32 storage
``` The separate CPU reconstruction regression verifies
exact east/south halo overlaps with the expanded 56-cell internal filter margin.

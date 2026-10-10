# Install Neural Earth

## Platform and hardware

Neural Earth supports Linux and Windows. The original tested reference is Windows, Python 3.12, NVIDIA RTX 3090 **24 GB VRAM**, PyTorch 2.11 and CUDA 12.8. Inference uses BF16. Use an NVIDIA CUDA GPU with BF16 support and a driver compatible with the locked Torch build. A 24 GB card is the tested target for comfortable exploration; smaller-card minimums and frame-rate guarantees have not been established.

Node.js 20 or newer runs bundled Orogen code. Chrome or Edge with WebGPU accelerates rendering. PNG fallback still requires the CUDA inference server. AMD/Intel browser rendering or optional City erosion does not provide an alternative neural backend.

Linux stores runtime caches in `${XDG_CACHE_HOME:-$HOME/.cache}/neural-earth`. Windows keeps the existing `E:/TerrainDiffusionRuntime` default. Set `TERRAIN_RUNTIME_ROOT` before launching to select another writable directory on either platform.

## Environment

Linux (install Python 3.12, Node.js 20+ and an NVIDIA driver first):

```bash
git clone --recurse-submodules https://github.com/dunkean/neural-earth.git
cd neural-earth
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

On Debian/Ubuntu, Python's matching `venv` package may be required. The CUDA Torch wheels supply the CUDA runtime and NVRTC compiler; a separate CUDA Toolkit is not required for the default backend. Native kernels load Linux shared libraries or Windows DLLs and retain their numerical admission checks.

Windows:

```powershell
git clone --recurse-submodules https://github.com/dunkean/neural-earth.git
cd neural-earth
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

For an existing checkout, use `git submodule update --init --recursive`. The lock file records the tested environment, including CUDA Torch wheels; it is larger than a minimal inference dependency list.

## Data and storage

The Terrain Diffusion submodule includes `data/global/etopo_10m.tif`. The adapter also needs WorldClim 2.1 **10 arc-minute** bioclimatic data. Extract the [official archive](https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip) into `terrain-diffusion/data/global/`:

Linux:

```bash
curl -fL https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip -o /tmp/neural-earth-worldclim.zip
python -m zipfile -e /tmp/neural-earth-worldclim.zip terrain-diffusion/data/global
```

Windows:

```powershell
Invoke-WebRequest 'https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip' -OutFile "$env:TEMP\neural-earth-worldclim.zip"
Expand-Archive -LiteralPath "$env:TEMP\neural-earth-worldclim.zip" -DestinationPath terrain-diffusion/data/global -Force
New-Item -ItemType Directory -Force E:/TerrainDiffusionRuntime | Out-Null
```

Required rasters are `wc2.1_10m_bio_1.tif`, `wc2.1_10m_bio_4.tif`, `wc2.1_10m_bio_12.tif` and `wc2.1_10m_bio_15.tif`. Upstream creates `synthetic_map_stats.json` on first use. [WorldClim data and terms](https://www.worldclim.org/data/worldclim21.html).

The runtime directory is created as needed. If Windows has no writable `E:` drive, configure a different directory; Linux requires no drive letters. Model/Hugging Face storage, source statistics, generation settings, atlas caches and verification reports follow `TERRAIN_RUNTIME_ROOT`. `generated/` holds application outputs.

Configure paths before starting, for example:

```bash
export TERRAIN_RUNTIME_ROOT="$HOME/neural-earth-runtime"
# Optional: keep large generated world/tile caches outside the checkout too.
export TERRAIN_OUTPUT_ROOT="$HOME/neural-earth-output"
```

```powershell
$env:TERRAIN_RUNTIME_ROOT = "$env:LOCALAPPDATA/NeuralEarth"
$env:TERRAIN_OUTPUT_ROOT = "$env:LOCALAPPDATA/NeuralEarth/generated"
```

Existing `HF_HOME`, `HF_HUB_CACHE` (or the legacy `HUGGINGFACE_HUB_CACHE`), `TERRAIN_GENERATION_ROOT` and `TERRAIN_BOOTSTRAP_CACHE` settings override their respective defaults. Both model loading and manifest hashing use the same Hub snapshot directory. Relative overrides resolve against the launch working directory before the runtime enters the upstream submodule. Developer junctions for `.venv` and `generated` are optional; fresh clones can use ordinary directories. Large assets and generated results stay outside Git.

The three neural models download automatically from `xandergos/terrain-diffusion-30m`, pinned at `9ef8030cb805b433b98ec25c5dddefbac07a9e26`. First installation needs internet access. A complete pinned local snapshot avoids Hub validation on later starts.

## Startup

Linux:

```bash
./start-neural-earth.sh
./start-neural-earth.sh --no-open
# Foreground server:
.venv/bin/python backend/terrain_server.py
```

Windows:

```powershell
.\.venv\Scripts\python.exe launch_terrain.py
# Without opening a browser:
.\.venv\Scripts\python.exe launch_terrain.py --no-open
# Foreground server for troubleshooting:
.\.venv\Scripts\python.exe backend/terrain_server.py
```

The Linux launcher also works from outside the checkout. On Windows, use `start-neural-earth.cmd`. Open **http://127.0.0.1:8765**. Logs are `server.log` and `server-error.log`; the server binds to localhost. Startup loads all three models and prepares CUDA Graphs. Source terrain can appear before neural warmup finishes. `/api/status` reports preload state and GPU selection.

## Optional dependencies

The default Orogen source needs Node.js, without npm installation. **Custom → Continental atlas** additionally needs Rust/Cargo and this pinned sibling source:

```powershell
git clone https://github.com/dunkean/world-builder-rs.git ../world-builder-rs
git -C ../world-builder-rs checkout 009efe18c457757da12ffc8a6215806efb87f012
```

The bridge checks the pinned clean source and compiles with `cargo build --locked` on first use. Its local crate dependency expects the sibling layout. Noise and Orogen sources do not require it.

```powershell
# Experimental Orogen NVIDIA acceleration:
.\.venv\Scripts\python.exe -m pip install -r requirements-orogen-gpu.txt
# Independent City surface erosion through native wgpu:
.\.venv\Scripts\python.exe -m pip install -r requirements-city-gpu.txt
```

On Linux, use `python -m pip install -r requirements-orogen-gpu.txt` or `python -m pip install -r requirements-city-gpu.txt` in the activated environment.

The first-launch GPU choice enables or disables all optional Orogen GPU switches and map rendering before generating the world. Individual settings are then saved locally. Orogen stages can fall back to CPU. Parallel propagation and erosion can change source terrain. City is an independent surface erosion engine. [Provenance](../THIRD_PARTY_NOTICES.md).

## Troubleshooting

- **CUDA unavailable:** check the driver and CUDA Torch build. `$env:TERRAIN_CUDA_DEVICE='0'` selects a visible device; automatic selection is a hardware heuristic, without measured throughput ranking.
- **Missing climate data:** extract WorldClim before background startup, so an upstream download prompt cannot wait invisibly.
- **Graph memory pressure:** `$env:TERRAIN_PREWARM='0'` skips warmup; `TERRAIN_PREWARM_BASE=0` skips base warmup only. These do not certify a lower VRAM minimum.
- **Slow navigation:** reduce visible extent, refinement depth and streams. Begin with four coarse streams. Browser cache budget limits tile/texture storage, rather than all server VRAM.
- **Source changes during runtime:** restart the server. Generator identities deliberately detect modifications to imported sources.

Run GPU verification with the server stopped to avoid device contention.

## Verification

With the virtual environment active and Node.js 20+ on `PATH`:

```bash
python -m unittest discover -s tests/python
npm ci
npx playwright install chromium
npm test
npm run test:webgpu
```

Browser tools use an installed Chrome/Chromium when available, otherwise Playwright's Chromium. `CHROME_PATH` selects an explicit browser; `PLAYWRIGHT_PATH` selects an existing Playwright installation. Python fixture tools select `.venv/bin/python` on Linux and `.venv/Scripts/python.exe` on Windows, with `TERRAIN_PYTHON` as an override. Real WebGPU tests require a working browser GPU adapter. On Linux, Playwright's `npx playwright install-deps chromium` can install missing browser system libraries.

`TERRAIN_BROWSER_ARGS` accepts additional Chromium flags as a JSON array. For shader correctness checks in a Linux/WSL headless environment whose GPU cannot present WebGPU textures, explicitly select SwiftShader:

```bash
export TERRAIN_BROWSER_ARGS='["--enable-unsafe-webgpu","--use-angle=swiftshader","--use-vulkan=swiftshader","--enable-features=Vulkan","--disable-vulkan-surface"]'
npm test
```

This runs browser shaders in software and does not measure GPU rendering performance. Neural inference still runs on CUDA. Unset this variable before hardware benchmarks.

The native atlas integration tests require the optional pinned `world-builder-rs` sibling checkout. ONNX quantization tests additionally require `python -m pip install onnx onnxruntime onnxscript`; these packages are independent of the default inference server. Stop the server before enabling `TERRAIN_TEST_CUDA=1` for exclusive kernel checks.

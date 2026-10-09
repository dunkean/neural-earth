# Install Neural Earth

## Platform and hardware

The tested reference is Windows, Python 3.12, NVIDIA RTX 3090 **24 GB VRAM**, PyTorch 2.11 and CUDA 12.8. Inference uses BF16. Use an NVIDIA CUDA GPU with BF16 support and a driver compatible with the locked Torch build. A 24 GB card is the tested target for comfortable exploration; smaller-card minimums and frame-rate guarantees have not been established.

Node.js runs bundled Orogen code. Chrome or Edge with WebGPU accelerates rendering. PNG fallback still requires the CUDA inference server. AMD/Intel browser rendering or optional City erosion does not provide an alternative neural backend.

The current runtime is Windows-oriented. Several cache paths use `E:/TerrainDiffusionRuntime`; Linux and alternate storage layouts have not been validated end to end.

## Environment

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

```powershell
Invoke-WebRequest 'https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_10m_bio.zip' -OutFile "$env:TEMP\neural-earth-worldclim.zip"
Expand-Archive -LiteralPath "$env:TEMP\neural-earth-worldclim.zip" -DestinationPath terrain-diffusion/data/global -Force
New-Item -ItemType Directory -Force E:/TerrainDiffusionRuntime | Out-Null
```

Required rasters are `wc2.1_10m_bio_1.tif`, `wc2.1_10m_bio_4.tif`, `wc2.1_10m_bio_12.tif` and `wc2.1_10m_bio_15.tif`. Upstream creates `synthetic_map_stats.json` on first use. [WorldClim data and terms](https://www.worldclim.org/data/worldclim21.html).

A writable `E:` drive is required by the current defaults. Model/Hugging Face storage, source statistics and most world caches use `E:/TerrainDiffusionRuntime`. `generated/` holds application outputs. Developer junctions for `.venv` and `generated` are optional; fresh clones can use ordinary directories. Large assets and generated results stay outside Git.

The three neural models download automatically from `xandergos/terrain-diffusion-30m`, pinned at `9ef8030cb805b433b98ec25c5dddefbac07a9e26`. First installation needs internet access. A complete pinned local snapshot avoids Hub validation on later starts.

## Startup

```powershell
.\.venv\Scripts\python.exe launch_terrain.py
# Without opening a browser:
.\.venv\Scripts\python.exe launch_terrain.py --no-open
# Foreground server for troubleshooting:
.\.venv\Scripts\python.exe terrain_server.py
```

Or use `start-neural-earth.cmd`. Open **http://127.0.0.1:8765**. Logs are `server.log` and `server-error.log`; the server binds to localhost. Startup loads all three models and prepares CUDA Graphs. Source terrain can appear before neural warmup finishes. `/api/status` reports preload state and GPU selection.

## Optional dependencies

The default Tectonic source needs Node.js, without npm installation. **Custom â†’ Continental atlas** additionally needs Rust/Cargo and this pinned sibling source:

```powershell
git clone https://github.com/dunkean/world-builder-rs.git ../world-builder-rs
git -C ../world-builder-rs checkout 009efe18c457757da12ffc8a6215806efb87f012
```

The bridge checks the pinned clean source and compiles with `cargo build --locked` on first use. Its local crate dependency expects the sibling layout. Noise and Tectonic sources do not require it.

```powershell
# Experimental Orogen NVIDIA acceleration:
.\.venv\Scripts\python.exe -m pip install -r requirements-orogen-gpu.txt
# Independent City surface erosion through native wgpu:
.\.venv\Scripts\python.exe -m pip install -r requirements-city-gpu.txt
```

Orogen GPU switches are off by default and can fall back to CPU. Parallel propagation and erosion can change source terrain. City is an independent surface erosion engine. [Provenance](../THIRD_PARTY_NOTICES.md).

## Troubleshooting

- **CUDA unavailable:** check the driver and CUDA Torch build. `$env:TERRAIN_CUDA_DEVICE='0'` selects a visible device; automatic selection is a hardware heuristic, without measured throughput ranking.
- **Missing climate data:** extract WorldClim before background startup, so an upstream download prompt cannot wait invisibly.
- **Graph memory pressure:** `$env:TERRAIN_PREWARM='0'` skips warmup; `TERRAIN_PREWARM_BASE=0` skips base warmup only. These do not certify a lower VRAM minimum.
- **Slow navigation:** reduce visible extent, refinement depth and streams. Begin with four coarse streams. Browser cache budget limits tile/texture storage, rather than all server VRAM.
- **Source changes during runtime:** restart the server. Generator identities deliberately detect modifications to imported sources.

Run GPU verification with the server stopped to avoid device contention.

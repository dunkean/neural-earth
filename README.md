# Neural Earth

**Procedural worlds, from tectonic plates to neural terrain.**

Neural Earth builds on **[World Orogen](https://github.com/raguilar011095/planet_heightmap_generation)** ([project](https://orogen.studio/)) and **[InfiniteDiffusion / Terrain Diffusion](https://github.com/xandergos/terrain-diffusion)** ([scientific paper](https://arxiv.org/abs/2512.08309), [project](https://xandergos.github.io/terrain-diffusion/)). We combine their planetary generation and learned refinement into a complete procedural world workflow: continents, tectonic relief, erosion, seasonal climate, biomes, soils, and continuous exploration in a map or globe.

Orogen's erosion methods draw on [stream-power incision](https://doi.org/10.1016/j.geomorph.2012.10.008) and [Priority-Flood drainage](https://doi.org/10.1016/j.cageo.2013.04.024). These are scientific method references, rather than a dedicated Orogen paper.

![Neural Earth procedural planet](docs/images/world.png)

## Explore a world at five scales

Actual application captures centered on the same location within each series. Left to right: **LOD 4 â†’ 3 â†’ 2 â†’ 1 â†’ 0**, ending at the model's **30 m** native terrain resolution. Neural detail is generated as you explore. The Render layer uses climate-informed surface materials.

**Snow-covered mountains**

![Snow-covered mountains at LOD 4 through LOD 0](docs/images/zoom-snow.png)

**Green mountains meeting the sea**

![Green coastal mountains at LOD 4 through LOD 0](docs/images/zoom-coast.png)

**Desert terrain**

![Desert terrain at LOD 4 through LOD 0](docs/images/zoom-desert.png)

Full-size captures, seeds, coordinates, settings and source resolutions are recorded in the [screenshot manifest](docs/images/screenshots.json).

## Read the world through its layers

![Terrain, natural materials, biomes and climate classification](docs/images/layers-surface.png)

![Soils, pedology, temperature and precipitation](docs/images/layers-environment.png)

![Tectonic plates, uplift, winds and ocean currents](docs/images/layers-dynamics.png)

The menu also includes seasonal climate, crust types, plate boundaries, convergence, hotspots, rain shadows, atmospheric pressure and cartographic styles. [Layer guide](docs/layers.md).

## Install and run

This is a **local Windows/Python application**. Neural inference requires an **NVIDIA GPU with CUDA and BF16 support**. A **24 GB GPU such as the RTX 3090** is the tested reference for comfortable exploration; smaller cards have no certified minimum configuration. Browser WebGPU accelerates rendering; the neural networks run on the Python server. PNG fallback still needs CUDA.

Install Python 3.12, Node.js and an NVIDIA driver compatible with the locked CUDA 12.8 build:

```powershell
git clone --recurse-submodules https://github.com/dunkean/neural-earth.git
cd neural-earth
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu128
```

Prepare the geography inputs and runtime directory using the [installation guide](docs/installation.md), then launch:

```powershell
.\.venv\Scripts\python.exe launch_terrain.py
```

Or double-click **start-neural-earth.cmd**. Open **http://127.0.0.1:8765**. Pinned model weights download on first use; loading all three networks and preparing CUDA Graphs takes time.

The default **Tectonic** generator uses bundled Orogen code. **Custom â†’ Continental atlas** additionally requires Rust/Cargo and a pinned sibling `world-builder-rs` checkout. Optional GPU erosion has separate dependencies. The installation guide documents these and the current `E:/TerrainDiffusionRuntime` storage convention.

## How it works

### Planetary structure and physical conditioning

Orogen constructs plates on a spherical mesh, derives boundary motion and stress, builds mountain belts and ocean basins, and applies the selected erosion. Its seasonal climate uses latitude, elevation, atmospheric circulation, ocean currents and moisture transport. Neural Earth rasterizes the source into a persistent atlas and adapts it to five checkpoint conditioning fields:

| Field | Physical convention |
| --- | --- |
| Elevation | Signed meters; negative values are below sea level |
| BIO1 | Annual mean temperature, Â°C |
| BIO4 | Monthly temperature standard deviation Ã— 100 |
| BIO12 | Annual precipitation, mm/year |
| BIO15 | Monthly precipitation coefficient of variation, % |

Two seasonal states become monthly proxies to match these statistics. This approximates a climatology, rather than a full atmospheric simulation. Relief, erosion and climate are independent retained stages.

```mermaid
flowchart LR
    A[Seed + settings] --> B[Orogen / noise / custom atlas]
    B --> C[Optional erosion]
    C --> D[Orogen seasonal climate]
    D --> E[Five conditioning fields]
    E --> F[Coarse â†’ latent â†’ decoder]
    F --> G[Physical elevation tiles]
    D --> H[Biomes + soils + diagnostics]
    G --> I[WebGPU map / globe]
    H --> I
```

### Three pretrained neural networks

We use the unchanged **[xandergos/terrain-diffusion-30m](https://huggingface.co/xandergos/terrain-diffusion-30m)** checkpoint, pinned to revision `9ef8030cb805b433b98ec25c5dddefbac07a9e26`. Its `EDMUnet2D` networks use magnitude-preserving layers. Neural Earth runs them in BF16 on CUDA; it does not train a new model.

| Network | Role | Physical sampling scale |
| --- | --- | ---: |
| `coarse_model` | Regional elevation statistics and climate; 20 diffusion solver steps | 7,680 m |
| `base_model` | Conditioned latent terrain generation | 240 m |
| `decoder_model` | Reconstruction of Laplacian-encoded elevation detail | 30 m |

InfiniteDiffusion evaluates overlapping windows of a deterministic noise field on demand. Shared dependencies and context allow consistent spatial access without materializing the entire high-resolution planet. Learned detail is reconstructed with coarse elevation constraints. Coastlines can move during refinement: a land mask does not force the neural DEM into the initial silhouette.

### Scale, navigation and rendering

Display sampling follows `r(LOD) = 30 Ã— 2^LOD` meters. At LOD 4, the screen samples at 480 m while the neural coarse source is 7,680 m: interpolation smooths display, without adding source resolution. LOD 3 uses 240 m latents; LOD 2â€“0 use the 30 m decoder with downsampling.

```mermaid
flowchart LR
    A[Procedural overview] --> B[Learned coarse Â· 7.68 km]
    B --> C[Latents Â· 240 m]
    C --> D[Decoded DEM Â· 30 m]
    D --> E[Optional experimental sub-30 m refinement]
```

The camera remains responsive while visible regions receive priority. Parent tiles stay visible until finer coverage arrives. World identities include settings, weights and source fingerprints. Physical heights are cached separately from appearance: lighting, contours and materials recolor existing terrain without rerunning the networks. Near the poles, the globe uses a rotated spherical chart for local neural sampling.

[Technical architecture and limitations](docs/architecture.md).

## Use the viewer

- **Scroll / + / âˆ’** zoom; drag or use arrows to move. **Map / Globe** changes the view.
- **Settings** controls source relief, erosion and climate. Each tab generates its stage; **Generate full pipeline** runs all three.
- **Layer menu** selects surface, climate, geology and cartographic views.
- **Rendering** controls lighting, contours, materials, cache budget and refinement depth.
- **SNR** controls allowed conditioning noise: lower values follow the source more closely. Per-LOD controls separately scale displayed relief detail.
- **Tools â†’ Prepare neural world** enables optional global coarse preparation. Visible regions remain prioritized.
- Share the URL to preserve seed, generation settings, camera and display preferences. **Save A / Show A** compares applied variants.

## Scope and performance

Neural Earth produces a complete geographic world to explore, with finite spherical or planar geometry. It does not generate an entire planet at 30 m on startup. Larger visible regions, deeper refinement and more streams increase computation and memory. Four coarse streams are the default scheduling balance; the coarse network batch remains one.

Terrain and climate target worldbuilding plausibility. Biomes and materials are visual models, without independent ecosystem simulation. Neural detail does not guarantee globally connected rivers or preservation of every source erosion feature. Optional sub-30 m refinement is experimental.

## Documentation and credits

- [Installation and troubleshooting](docs/installation.md)
- [Architecture and neural generation](docs/architecture.md)
- [Layers](docs/layers.md), [biomes](docs/biomes.md) and [materials](docs/materials.md)
- [Third-party provenance and licenses](THIRD_PARTY_NOTICES.md)

Upstream projects retain their names, licenses and pinned revisions. Native adaptations have explicit provenance. Historical audits, reviews and implementation work reports are archived outside this repository.

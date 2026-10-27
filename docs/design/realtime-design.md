# Neural Earth — region engine design

This note records prospective directions. The shipped runtime is described in [architecture.md](../architecture.md). Historical audits, benchmarks, reviews and reports are archived outside the repository. The [French original](realtime-design_FR.md) remains available.

## Unit of computation

The engine should treat a region as a shared graph of coarse, latent and decoded dependencies. Multiple cameras requesting the same terrain should reuse this work. Physical data and its provenance should remain independent of appearance.

The camera publishes a spatial requirement, a level of detail and an epoch. The scheduler prioritizes visible coverage, then its refinement, then neighboring areas. Parents remain visible until their replacements arrive. Cancellation removes work that is no longer needed at the designated computation boundaries.

## Potential portability

A future ONNX Runtime Web/WebGPU inference path could move the networks into a browser worker. Beyond exporting the U-Nets, it would need to carry deterministic noise, spatial dependencies, the solver, post-processing and caches. The prototype in `webgpu/` remains a feasibility study.

An optional native agent could use ONNX Runtime CUDA as a porting reference, then TensorRT if measurements justify that dependency. The current public runtime remains Python/PyTorch CUDA; these proposals do not describe compatibility already delivered.

## Contracts to preserve

- The same seed, coordinates and parameters produce a reproducible field within the validated numerical contract.
- No random reset per request or camera.
- Separate the physical resolution of the source from displayed sampling.
- Preserve the units of the five conditioning fields.
- Keep distinct identities for sources, weights, physical settings and appearance.
- Verify halos, window boundaries, the meridian and polar regions.
- Maintain an independent CPU reference for ported palettes, materials and methods.

## Research directions

Consistent hydrology across scales, controlled preservation of erosion structures during neural refinement, and adaptation to polar climates remain research directions. A biome palette or visual snow is not enough to validate these properties.

Detail below 30 m must be assessed as experimental refinement. It must not be presented as a learned or measured resolution higher than that of the checkpoint.

## Validating future ports

Compare identical regions, the same windows and their overlaps. Measure startup, the first region, regions with reused dependencies, transfers, memory and rendering separately. Average inference throughput does not replace a measurement of complete navigation latency.

References: [InfiniteDiffusion](https://arxiv.org/abs/2512.08309), [upstream](https://github.com/xandergos/terrain-diffusion), [Orogen](https://github.com/raguilar011095/planet_heightmap_generation).

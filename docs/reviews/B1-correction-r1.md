# B1 review correction — final GPU attestation

**Status:** Code corrections and E1 provenance checks complete. `natural`
remains the default. A1–A3 remain experimental and await human visual approval.

## Corrections to Astra B1 r1

- The CPU conditioning manifest is explicitly `cpu-conditioning` and no longer
  presents default CUDA settings as executed settings. After the NN world is
  configured, `neural_manifest()` records the actual
  `terrain_inference.VERSION`, batch/graph options, checkpoint kwargs,
  precision, device capability, checkpoint file hashes and source hashes. It
  excludes the free-memory label from numerical identity.
- Artifacts live under `worlds/<world_hash>/<site>/`. Every NPZ embeds its
  world hash, seed, ablation, exporter hash and coordinate bounds. Metrics bind
  the NPZ SHA-256 and configured numerical profile. The index lists CPU and
  NN artifacts separately. `write_e1_report()` rejects mismatched manifests,
  changed source/model files, altered NPZs, wrong profiles/focuses and mixed
  DEM bounds.
- E1 now labels full conditioning/coarse/latent/DEM summaries as **context
  distributions over different footprints**. Each includes bounds and cell
  size. A separate diagnostic bilinearly samples conditioning/coarse/latent
  elevations at the exact native DEM pixel centers. Cross-ablation DEM metrics
  require the same native bounds and dimensions.

## Corrections to Opus B1 r1

- A1 substitutes the macro elevation into the *raw* natural five-field stack,
  then applies natural `finalize`, so BIO1 lapse and BIO4 coupling reflect the
  new elevation. A2 still changes climate only. A3 combines macro elevation
  and the latitude climate experiment.
- A2/A3 climate channels are clipped to per-seed A0 physical support measured
  on the fixed global 256×128 lattice. The export emits each channel's
  pre/post-clip fraction and p99/max checkpoint z-score. Seed-zero A3 raw BIO1
  has about **20.6%** outside A0 support before clipping. This remains a
  climate quality limitation, especially at high latitudes.
- A3 adds a medium regional relief band and allows ridge influence to reach
  margins. The new profile version is `finite-geography-v2`; golden sample
  tests pin its fields. This is exploratory and has not passed visual review.
- The corpus default native crop is 256². Three selected sites request 1024²
  (30.72 km). Fixed palette now reaches 9,000 m. Exports include fixed-light
  hillshade, shaded DEM, central profiles and separate coarse/latent boards.
  The 1024² latent context grows to cover the full DEM footprint plus halo.
- The next NN corpus selects eight sites across seeds zero and 42, including
  negative coordinates and a coarse-window border. All fourteen sites remain
  in the CPU conditioning corpus.

## Validation and evidence boundary

`test_terrain_macro.py`: **9/9 passed**; `py_compile` passed. The tests include
upstream A0 agreement, A1 climate recoupling, support bounds, crop invariance,
portable manifest identity, profile-specific artifact paths, altered NPZ/source
rejection and mixed-report rejection.

An initial v2 checkpoint run produced all **32** intended NN arrays and the
regional visual boards under `E:/TerrainDiffusionRuntime/reference-b1r1`.
`write_e1_report()` correctly refused to attest that run: `terrain_server.py`
and `terrain_background.py` changed between CPU manifest creation and the
final check while B3 was in progress. Those arrays are **historical,
unattested evidence**, not a completed E1 baseline. In that run the A3-selected
coast at 1024² spans roughly −12 to 21 m; the broader shoreline is visible but
still too flat to approve A3. The A3-selected mountain shows learned ridges
and valleys across the regional crop. These observations do not establish a
passed visual gate.

After the B3 source freeze, the command below ran into a **new output
directory** so no historical products entered the index:

```powershell
.venv/Scripts/python.exe terrain_reference.py --output E:/TerrainDiffusionRuntime/reference-b1r1-final --neural all --neural-site coast_seed0 --neural-site mountain_seed0 --neural-site plain_seed0 --neural-site macro_coast_seed0 --neural-site macro_mountain_seed0 --neural-site macro_plain_seed0 --neural-site coast_negative --neural-site coarse_window_border
```

It completed with **14 CPU conditioning sites, eight NN sites, 32 neural
artifacts and an attested `e1_report.json`**. Current-file source/model hashes,
actual `cuda-resident-window-v2` settings (latent batch 16, CUDA graphs on),
NPZ SHA-256, seed/profile/focus, and same-native-pixel bounds were checked.
The report keeps `context_distributions` separate from
`matched_native_footprint`; the latter explicitly interpolates each lower
resolution stage. DEM, shaded, coarse, latent and profile boards are in
`comparisons/<site>/`.

On the 1024² A3-selected coastal crop, A3 spans **−12 to 21 m** with 49% land:
the 30.72 km shore remains too flat for visual approval. Its selected mountain
crop spans **5,315 to 7,017 m** and shows learned ridge detail. At the protected
A0 coast, A0 spans −2,263 to 1,675 m at the same 1024² footprint; A2 changes
its sea/land layout substantially. These are quality observations, not an
accepted new generator. The human visual gate remains open.

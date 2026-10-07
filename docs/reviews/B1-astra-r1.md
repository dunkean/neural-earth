# B1 review — Astra r1

**Decision: CHANGES REQUIRED for reference/provenance tooling.** The macro generator can remain available as an explicitly experimental profile; **natural must remain the default**. No P0 or proven physical-unit/runtime terrain defect was found. One P1 provenance defect and one P2 comparison-report defect prevent treating the current export as a fully reproducible E1 baseline. Neither finding requires changing the terrain generator itself.

Reviewed 7 October 2026: `terrain_macro.py`, `terrain_manifest.py`, `terrain_reference.py`, `docs/reference_corpus.json`, `test_terrain_macro.py`, `docs/MACRO_IMPLEMENTATION.md`, relevant inference/server factory and preview integration, and source-distribution derivation in `terrain_world.py`. Read the supplied `reference-l0-l4/e1_report.json`, `index.json`, complete A0 manifest, and actual NPZ products; visually inspected the actual `macro_coast_seed0/neural-comparison.png`. Ran CPU tests only: **7/7 `test_terrain_macro` tests pass**. No runtime edits, GPU inference, or server restart.

## Findings

### B1-R1-01 — P1: neural exports are not bound to their actual numerical profile

**Locations:** `terrain_reference.py:214–219`, `244–260`, `266–268`; artifact serialization at `134–147`. Defaults originate in `terrain_manifest.py:88–92`.

`export_corpus()` creates the supposedly complete manifests before loading the neural world. It supplies neither the configured `inference_profile` nor the actual numerical version. The later neural path configures the world but never replaces/binds that identity to the actual execution profile. Neither `stages.npz` nor its metrics records a world hash, actual graph/batch profile, or a reference to an immutable generation manifest.

**Concrete supplied evidence:** `E:/TerrainDiffusionRuntime/reference-l0-l4/manifests/seed-0-A0.json` reports `complete=true`, `generation.inference_profile={}`, and `numerical_profile="cuda-resident-v1"`. Current inference declares `VERSION="cuda-resident-window-v2"`. The production profile includes coarse/base/decoder batch sizes, graph settings, and other numerical options; BF16 batch shape already has demonstrated terrain effects in B2. Environment changes such as `TERRAIN_LATENT_BATCH` or `TERRAIN_CUDA_GRAPHS` can therefore produce different NN settings while this export manifest remains identical. This is a provenance defect, not evidence that the saved DEM itself is incorrect.

**Reproduction without GPU:** inspect the supplied manifest's `generation` mapping and the NPZ/metrics keys. The code path is unconditional: `build_manifest(seed, ablation, file_hashes=...)` uses defaults, and no later call records `asdict(world._terrain_profile)`. Reusing an output directory can also overwrite its manifests/conditioning while leaving old neural products present; the report reads any existing `neural/stages.npz` without checking their generation identity.

**Fix:** after configuring each world, build a manifest from that actual execution state and the actual `terrain_inference.VERSION`; bind each neural NPZ/metadata to its hash. Record source/config hashes at generation time, sample bounds, and actual profile. Verify these identities when composing an E1 report; reject stale or mixed products. Keep a separate explicitly conditioning-only manifest if CPU-only export is desired. Exclude free-memory labels from the stable numerical profile as production already does. Do not relabel old neural arrays as having been generated under newly hashed source files without verifying the operative generation state.

**Validation:** CPU tests should simulate two distinct inference profiles and stale directory contents, proving that artifact identities differ and mixed reports are refused. A future actual NN rerun should carry that recorded profile; the current supplied numerical results can remain historical evidence with their provenance limitation stated.

### B1-R1-02 — P2: stage summaries describe different spatial footprints as an input-to-output comparison

**Locations:** `terrain_reference.py:119–127`, `164–187`, especially the docstring at 165 and summaries at 183–186.

`write_e1_report()` says it measures input→coarse→latent→DEM changes at identical native pixels, but it summarizes entire differently sized arrays. The conditioning/coarse arrays span **491.52 km**, the latent array **15.36 km**, and the supplied 128-square DEM **3.84 km**. The coarse rectangle is centered on `center_ci/cj`, while latent/DEM can use an offset `focus_native_i/j`. Stage distribution differences therefore combine resolution, footprint and focus changes; they cannot establish what the same piece of terrain did through the stages. The report has no metric-local bounds/resolution to expose this distinction.

**Reproduction:** open `coast_seed0/A0/conditioning.npz` and `neural/stages.npz`. Observed conditioning bounds `[235,227,299,291]`, coarse shape `[7,64,64]`, latent shape `[6,64,64]`, and native DEM bounds `[68561,66674,68689,66802]`. The E1 report summarizes all of each array. The cross-ablation **DEM-vs-A0** comparisons do use equal native pixels and are not affected by this finding.

**Fix:** label existing summaries as contextual distributions and include explicit footprint, resolution and coordinate convention per stage. Add a separate comparison sampled on the same native footprint if stage-to-stage causal claims are needed. Save coarse/latent bounds directly in the NPZ instead of requiring reconstruction from corpus conventions. Preserve the larger context arrays for useful visual inspection.

## Contracts that look correct

- Five model input fields have the intended semantics: signed metres before one signed-square-root transform; physical BIO1 Celsius; BIO4 standard deviation ×100; BIO12 mm/year; BIO15 percent. Macro `finalize` is identity because macro outputs are already physical. The default pipeline path invokes the matching factory and does not double-finalize those fields.
- `source_distributions()` lifts source BIO1 to sea level using ETOPO and the precipitation-dependent lapse convention before forming latitude quantiles; A2/A3 restore the local positive-elevation lapse once. It uses physical BIO4 source values, not the natural factory's regression residuals. The server's preview converts local physical BIO1 into the sea-level transport representation before the shader restores the displayed altitude lapse.
- A0 reproduces the installed factory on tested square windows, including the seed-zero workaround. The documented sub-0.001 model-unit tolerance is consistent with the test, not a claim of bitwise equality. Production `natural` keeps the upstream factory rather than silently replacing it with this approximation.
- A1 leaves A0 physical climate intact, and A2 leaves A0 elevation intact. That is a meaningful channel ablation, although A1 climate then remains tied to the original terrain rather than being a physically coupled new climate. A3 deliberately combines effects and is not a clean independent ridge-only ablation.
- Global objects and calibration lattice are seed/version dependent, not request dependent. Negative coordinates, split crops and sampling order pass the CPU contract tests. Half-cell positions in the arbitrary-point preview agree with the macro cell-center convention. The world remains flat and nonperiodic.
- The updated manifest now includes 28 local source hashes, installed InfiniteTensor source hashes, and relevant package/Torch/CUDA/cuDNN versions. Weight/config/source/statistics hashes are path independent in the tested relocated-checkpoint case. Incomplete manifests and tampered payloads are rejected. This resolves the implementation-coverage concern raised during B2 review, subject to R1-01's requirement that the actual execution options be supplied.
- Server profile selection defaults to `natural`; macro profiles are explicit. Documentation labels A3 experimental and acknowledges that its visual gate is not passed.

## Evidence limits and quality assessment

The index records fourteen CPU conditioning sites but only **six neural sites, all seed zero**, for all four ablations. Seed 42, negative-coordinate named cases and the coarse-window-boundary case exist in the corpus but do not have supplied neural coverage. `manifest_complete=true` describes the manifest-building mode; it does not mean the whole corpus was run through NN or that R1-01 is satisfied.

The supplied A0 coastal crop spans approximately **−201.9 to 355.7 m**, with 50% land; its mountain crop spans **4,967.4 to 6,229.7 m**. At the A3-selected coastal location, A3 spans approximately **−0.919 to 6.901 m**, 49.44% land. The actual comparison board shows the reported nearly flat shore and smooth fields. At the A3-selected mountain location, A3 spans **6,115.2 to 6,878.2 m**. These are actual decoder results, not merely successful input statistics.

The measurements support the documentation's refusal to approve A3. They do not prove that the new macro improves geographic credibility. Same-coordinate large changes between A0 and a new geography are expected and are not alone a defect; the right next evidence is matched regional structure, coast detail, mountain variety and protected-site review. Fixed elevation colors saturate above 6,000 m and expose little mountain shape: add fixed-light hillshade/normals and height profiles alongside the neutral palette before making a visual selection.

The global marginal test checks only a few altitude quantiles, land fraction and an equator-to-pole temperature contrast on the calibration lattice. It does not validate the five-field joint distribution, spectral/coastal structure, out-of-distribution cold inputs, or multi-seed tail behavior. A2/A3 climate deliberately draws from a narrow range of percentile ranks; using empirical source quantiles is not equivalent to reproducing their full distribution. These are experimental-design limitations, not proven unit-conversion bugs.

No measured navigation latency, throughput or performance improvement follows from B1. Natural remains the credible reference while B2/B3/B4 establish execution feasibility. Do not trade that reference for the speed of the CPU conditioning preview or claim visual acceptance from the seven passing contract tests.

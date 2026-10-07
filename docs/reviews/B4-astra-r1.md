# B4 Astra review — feasibility experiment

Date: 2026-10-07. Reviewer: Astra, high. Decision: **CHANGES REQUIRED**.

This reviews the experiment, not a proposal to deploy the failed browser backend. Production CUDA should remain in place. The documented **L1 NO-GO is justified**. Fixing the experiment defects below does not require making WebGPU pass, rewriting the networks, or deploying a failed port.

## Proven defects

### P2 — Base conditioning omits the normalized noise-level scalar

Location: `webgpu/crop.mjs:78`, compared with `terrain-diffusion/terrain_diffusion/inference/world_pipeline.py:1144` and `:1154`.

`baseConditioning()` builds its final group as `[0]`. Upstream sets `NOISE_LEVEL=0` and then applies `(noise_level - 0.5) * sqrt(12)` before magnitude-preserving concatenation. Consequently the browser passes zero in conditioning element 57, where upstream passes approximately `-5.385165` after group scaling. This is directly corroborated by `crop64-coast-base_model-cond_0.bin`: float32 element 57 is `-5.385165214538574`; elements 52–56 are zero as expected for the histogram group.

This changes every base forward in the integrated crop, independently of the demonstrated decoder operator discrepancy. The current 5,583.79 m crop error cannot be attributed exclusively to the backend. The real-feed operator comparisons remain valid because those use captured upstream feeds rather than this JS conditioning function.

Fix: use the upstream normalized scalar and add an independent fixture comparison for all 58 base-conditioning values, including the final group. The existing seven Node tests do not exercise this function. Preserve the old failed report as historical evidence; label it as including a known pipeline error and record any later corrected experiment separately.

### P2 — The documented export recipe cannot produce the base artifact consumed by the runners

Locations: `webgpu/export_models.py:109–110`, `webgpu/crop.mjs:24–27`, `webgpu/verify_real_feeds.py:20`, `docs/WEBGPU_IMPLEMENTATION.md:109`.

The exporter produces `base_model.onnx`, while the crop, standalone base probe, and native real-feed comparison all require `base_model_external.onnx` and `base_model_external.data`. No checked-in Python conversion creates these required files, and the reproduction recipe provides no conversion command. The actual external files exist in the artifact directory, explaining why the recorded run could succeed in loading, but a clean reproduction of the documented steps cannot reach that run. In addition, the export report hashes the inline base file, not the external pair actually consumed in the browser.

Fix: check in the deterministic inline-to-external conversion, invoke it from the recipe/export workflow, validate the converted graph, and record the graph plus external-weight checksums and conversion options. This is a reproducibility/provenance correction, not a requirement to rerun heavy exports during review.

### P2 — Diagnostic no-capture mode can emit an undifferentiated PASS

Location: `webgpu/crop.mjs:259`.

The reported `status` depends only on height/climate errors, even when `requireGraphCapture=false`. Thus numerically acceptable output from `runCrop(..., false)` would return `PASS` and the CLI would exit successfully, although the documented L1 gate explicitly requires graph capture. The current measured run fails numerically, so this has not falsely certified the saved evidence.

Fix: distinguish `numericalStatus` from `l1Status` or emit a clearly diagnostic status for no-capture runs. A numerical pass without required capture must not be represented as an L1 pass.

## Evidence accepted and limits

- Read the audit, implementation tracker, WebGPU implementation notes, JS/Python runners and math, corresponding upstream formulas, and real reports/fixture metadata under `E:/TerrainDiffusionRuntime/webgpu-models`.
- Re-ran only the lightweight Node fixture suite: **7/7 passed**. No GPU workloads, model exports, or native ORT model executions were run during this review.
- `browser-crop.json` records the coarse step-2 bind-group failure. `browser-crop-nocapture.json` records 40 coarse + 34 base + 1 decoder forwards, height max/RMSE `5583.7875 / 3101.1063` m and climate max/RMSE `14.36427 / 5.69164`, with 25.73 s wall time. The implementation notes correctly reject these as a terrain port or speedup.
- Real-feed CPU comparisons distinguish native CPU/CUDA variation from much larger base/decoder browser divergence. Decoder prefixes match through the six-channel input concatenation exactly and normalized weights within `3.58e-7`; the first Conv differs by max `16.8906`, RMSE `1.44249`. Frozen weights retain the discrepancy. This is strong localized evidence independent of the integrated crop conditioning bug.
- The isolated single-Conv model exists and its CPU equivalence is recorded. The saved browser diagnostic report does not contain a single-Conv result; documentation explicitly acknowledges that pending run. This is **missing evidence**, not a false claim that the isolated model already failed in-browser. The observed prefix discrepancy is sufficient for the present NO-GO.
- The noise, scheduler, fusion, Laplacian reconstruction, climate channel ordering, negative-coordinate treatment, and fixed-crop dependency path broadly follow the upstream formulas. Passing synthetic component tests does not certify the complete JS pipeline; the conditioning defect demonstrates this distinction. Fusion trusts the supplied reference window inventory: its positive-weight check does not independently prove all overlapping windows are present. For this fixed experiment, a captured complete dependency inventory is an acceptable scope limitation, not proof of arbitrary-crop scheduling.
- Browser device sharing and per-node GPU residency remain unverified and are correctly disclosed. The saved adapter object is empty and reports do not independently pin browser version, driver, or the actual ORT adapter. Future runs should record those along with consumed model/runtime hashes. This limits reproducibility of hardware attribution, but does not invalidate the observed numerical failure.
- Full pipeline fidelity, memory/latency acceptance, coast topology, visual quality, arbitrary-crop behavior, and browser/world-scale throughput remain unpassed or unmeasured. No speed or production readiness is certified.

**Required disposition:** correct the three experiment defects, keep the failed evidence and its limitations explicit, and retain L1 NO-GO unless a later independently recorded run passes its actual gates. No deployment of this backend is required to complete B4 as a feasibility experiment.

I'm giving **B4 CHANGES REQUIRED**, but the changes are small. The main negative result holds up: ONNX Runtime Web 1.30's WebGPU backend gives wrong base and decoder outputs on this machine, and the pipeline code isn't the cause. But the experiment can't yet separate backend errors from pipeline errors, it contains one proven port defect, and the docs don't record the outcome. I didn't run anything; everything below comes from reading the code and the reports on E:.

## What the evidence shows

| Check | Result | Source |
|---|---|---|
| ONNX export vs PyTorch, random inputs, CPU | Max error 3.6e-6 to 1.1e-5 | `export-report.json` |
| ONNX Runtime CPU vs captured CUDA forwards (real inputs) | coarse 0.0066, base **0.0121**, decoder 0.0018 | `real-feed-ort-cpu-comparison.json` |
| Browser coarse, one forward | 0.0054, passes | `browser-coarse_model-reference.json` |
| Browser base, one forward | **15.02**, with or without graph capture | `browser-base_model-reference*.json` |
| Browser decoder, one forward | **30.2158** in all four variants (capture on/off, fixed batch, Resize instead of Tile) | `browser-decoder_model-reference*.json` |
| Decoder prefix diagnosis | Concat exact, weight Cast 3.6e-7, **first Conv off by 16.9** even with frozen weights | `decoder-diagnostic-browser.json` |
| Full crop, graph capture off | height **5,583.8 m**, climate **14.36**, base stage max 19–22 | `browser-crop-nocapture.json` |
| Full crop, graph capture on | Crashes at the first replay (`createBindGroup … buffer undefined`) after coarse step 2 | `browser-crop.json` |

The base and decoder failures appear with the exact captured inputs, outside the pipeline code. They are deterministic and identical across capture/static/resize variants, so the backend fault is attributed correctly. One more file to note: `browser-crop64-report.json` doesn't exist; the crop reports are `browser-crop.json` and `browser-crop-nocapture.json`.

## P1 findings

**P1-1 — Proven port defect: the base model's noise-level input is 0 instead of −√3.** `webgpu/crop.mjs:78` (comment at :61–62).
- **Upstream:** `noise_level_norm = (0 − 0.5)·√12 = −1.7320508` (`world_pipeline.py:1144`). After the `mp_concat` scaling it becomes −√29 ≈ **−5.385165**.
- **Browser:** `new Float32Array([0])`.
- **Reproduction:** the captured `crop64-coast-base_model-cond_0.bin` has minimum −5.385165. No other group in the 58-value vector can produce that value. Every one of the 34 base forwards gets the wrong value.
- **Why it went unnoticed:** the backend failure hides it. Even a correct WebGPU backend would fail the crop gate because of this.
- **Fix:** `new Float32Array([f32((0 - 0.5) * Math.sqrt(12))])`. Add a Node test that checks `baseConditioning(...)` for the first base window (7, −17), against the captured `cond_0`.

**P1-2 — No way to check a stage in isolation, so pipeline correctness is unproven.** `crop.mjs:215–222`
- Each stage consumes the browser's own output from the previous stage, so stage errors are mixed together.
- Nothing compares the model inputs the browser builds against the captured inputs. P1-1 slipped through because of this.
- The base, decoder, Laplacian and climate integration has therefore never run correctly end to end.
- **Fix:**
  1. Compare the browser-built inputs (coarse step 1, first base window, decoder) against the captured first-forward inputs.
  2. Add a mode where each stage reads the reference tiles of the previous stage, which the manifest already contains.
  3. Run the same `crop.mjs` once with a diagnostic, non-product CPU backend (e.g. onnxruntime-node). That proves the pipeline separately from WebGPU.

**P1-3 — Gates calibrated against a reference that isn't true FP32.** `probe.js:95–97`, `crop.mjs:259`, `capture_crop.py:53,114`
- **Forward gate:** ONNX Runtime CPU itself scores 0.0121 on the base model, which fails the 0.01 browser forward gate. A correct backend would be rejected.
- **Likely cause:** `capture_crop.py` never disables PyTorch's default `cudnn.allow_tf32=True`, yet the manifest says `"precision": "fp32"`. The 1e-2 CUDA-vs-CPU gap against 1e-5 CPU-vs-CPU fits that. This is an inference, not proven.
- **Crop gate:** no error floor was measured for the ≤1 m gate. The audit (§7.2) recorded 3.2 m and 209 m differences from batch-size changes alone.
- **Fix:**
  - Re-capture with TF32 disabled and record the flags in the manifest.
  - Measure the CPU-backend crop error (from P1-2) to get the floor.
  - Set the per-stage and final WebGPU gates relative to that floor, fixed before re-running.

**P1-4 — The graph-capture evidence is misleading.**
- The one-forward probes call `run` only once (`probe.js:79`). The first run is the capture; replay happens on later runs, so `graphCaptureRequired: true` shows nothing about replay.
- The only run that replays (the crop) crashes on its first replay.
- The doc (`WEBGPU_IMPLEMENTATION.md:39–41,53–55`) still treats graph capture as a working requirement.
- **Fix:** run each forward at least 3 times and compare replay outputs with the first. Record the crash with a minimal two-run coarse reproduction.

**P1-5 — The documented status doesn't match the outcome.**
- `WEBGPU_IMPLEMENTATION.md:13–56` lists only the CPU export checks and says the crop "must still pass".
- `AUDIT_IMPLEMENTATION.md:25` still says B4 is in progress.
- **Fix:** add a results section with the table above, the first-Conv localization, the capture crash, P1-1, and an explicit status for audit gate E2 (WebGPU feasibility): **NO-GO for ONNX Runtime Web 1.30 WebGPU on this adapter; pipeline not yet validated.**

## P2 findings

- **ONNX Runtime ignores the GPU device the code hands it.**
  - Every probe report says `ORT device matches requested device: false`: 128 MiB requested limit vs 2 GiB at runtime.
  - The stack traces show the native WebGPU backend (`asyncify` build), not JSEP. Yet the doc says "explicit GPUDevice" and cites the JSEP operator table (`WEBGPU_IMPLEMENTATION.md:32–41`).
  - `crop.mjs:170–172` creates a device that is never used or destroyed.
  - **Fix:** correct the doc and the cited operator coverage, assert device equality or drop the injection, and delete the unused device.
- **Decoder diagnostic reports misleading min/max.** `verify_decoder_diag.cjs:34` takes min/max over the first 1,000 elements only. That is why Concat shows error 0 alongside min −3.07, while the reference min is −4.84. **Fix:** use the full array and record where the maximum error occurs.
- **Artifacts with no traceable source.**
  - `base_model_external.onnx/.data`, `decoder_model_static.onnx` and `decoder_first_conv_repro.onnx` have no producing script and no checksum in `export-report.json`. The first-Conv reproduction has no report at all.
  - The browser reports were made by earlier versions of `probe.js`: the error line numbers 95/96/97 and the status text differ from the current code.
  - The reports don't record ONNX Runtime, Chrome or driver versions, the GPU (`adapter: {}`), or model/fixture checksums.
  - **Fix:** have `export_models.py` produce and checksum every variant, and stamp each browser report with versions, adapter and checksums.
- **Climate gate ignores physical units.** `crop.mjs:259,264` applies ±0.1 to every channel. Channel 4 (lapse rate) is clamped to [−0.012, 0], so it can never fail. **Fix:** per-channel limits in °C, °C×100, mm/yr, %, and °C/m.
- **The "coast" crop has no coast.** The reference height is 1,190–1,475 m, so the sign/zero-crossing path and coastline topology (audit §13.3) are untested. **Fix:** rename it, and add a crop with min < 0 < max.
- **Test fixtures don't match the real case.** `make_math_fixture.py:72–95` hand-copies the `_compute_climate` resampling instead of calling it, and uses only i1 = j1 = 0 (the real crop has j1 = −3584). There are no tests for coarse/base/decoder input assembly. **Fix:** call the upstream functions directly and include negative coordinates.
- **Fusion completeness check is weak.** `fusion.mjs:38` only catches pixels with zero weight. A missing overlapping tile silently gives a partial average. The window list comes from the fixture (`crop.mjs:215`); the browser doesn't derive it. **Fix:** derive the required windows from size and stride and assert they are all present.
- **Timings mix in other work.** `stageMs` includes HTTP fixture fetches and comparisons (`crop.mjs:218,222`), and nothing measures memory. Don't quote these numbers; separate the timers.
- **Experiment leaks into the product.**
  - `terrain_server.py:543–550` serves the whole `webgpu/` folder (including `node_modules` and scripts) and every model artifact, without `serve_probe.py`'s file filter.
  - The reproduce steps `pip install` into the app's own Python environment.
  - **Fix:** remove or flag-gate the routes, and use a separate environment.

## Hypothesis (not proven)

The two failing models' first convolution has 6 input channels (5 inputs plus 1 constant, confirmed by the `[64,6,3,3]` weight). The passing coarse model has 12. A cheap test: a single Conv with 4/6/8/12 input channels at 64² and 512², or zero-padding the decoder's Concat to 8 channels. If that confirms it, the next steps are layer-by-layer frozen-input checks for the base model and filing the bug with ONNX Runtime.

## Verdict: CHANGES REQUIRED

None of this requires shipping or fixing the WebGPU backend. Outstanding gates:
1. Fix P1-1, then add the captured-input checks and the per-stage isolation mode.
2. Run the crop with a CPU backend, against a reference captured with TF32 off, to set the error floor and fix gates (including per-channel climate limits) before any re-run.
3. Record the results and the E2 status in both docs, including the capture crash and device findings.
4. Give every tested artifact a producing script and a checksum, and stamp reports with versions.
5. Still unproven: memory, timing, and coastal behaviour. Nothing here certifies visual quality or performance.
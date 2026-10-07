# Terrain Diffusion 30 m: WebGPU L1 probe

This experiment uses the immutable `xandergos/terrain-diffusion-30m` snapshot
`9ef8030cb805b433b98ec25c5dddefbac07a9e26`. The world profile is
`natural`, with checkpoint values (`native_resolution=30`, latent compression
8, `cond_snr=0.5` per channel, `residual_std=0.7`). The fixed physical crop is
seed 42, rows 2560–2624 and columns −3584–−3520 (64 × 64 native cells). The
reference fixture is produced by the upstream `WorldPipeline.get` and records
every neural window requested, all five climate fields, and final height in
metres. The fixture supplies the five raw geographic fields to the browser;
network states and final physical fields must be recomputed there.
Despite the historical artifact name `crop64-coast`, this is an upland crop:
its reference elevations range from 1,190 to 1,475 m. It does not test a coast.

## Decision: L1 NO-GO

The three neural graphs load and execute with the browser's WebGPU execution
provider, and all 75 forwards complete for the fixed crop. **Audit E2 remains
NO-GO: the browser crop fails the numerical gate.** With explicit NCHW layout
and the corrected base conditioning, Chromium 154 on Windows/RTX 3090 differs
from the upstream crop by at most 4.7927 m in height (RMSE 1.4043 m), against
a provisional 1 m limit. All five provisional per-channel climate limits pass.
The result is reproducible with and without `enableGraphCapture:true`; the
previous bind-group replay crash no longer occurs. Reports are
`browser-crop.json` and `browser-crop-nocapture.json` on E:. Their wall times
include fixture downloads and comparisons, so they are not pure NN benchmarks.
The earlier default-layout run, before the conditioning fix, failed by
5,583.79 m and crashed on capture replay. It is historical evidence only and
cannot be used to attribute the corrected crop's error.

## Export and forward verification

The three FP32 ONNX exports passed dummy-input forward comparisons against
PyTorch with pinned `onnx==1.23.2` and `onnxruntime==1.30.0` CPU. Thresholds
are maximum absolute difference ≤0.003 and RMSE ≤0.0003, enforced by
`webgpu/export_models.py` (no optional skip or print-only verification).

| Graph | Input shape | ONNX bytes | Nodes | Observed maximum forward difference |
|---|---:|---:|---:|---:|
| Coarse | 1×11×64×64 + noise scalar + five conditioning scalars | 22,497,125 | 741 | 3.58×10⁻⁶ |
| Base | 1×5×64×64 + noise scalar + 58 conditioning scalars | 2,029,994,361 | 3,328 | 1.07×10⁻⁵ |
| Decoder | 1×5×512×512 + noise scalar | 223,854,125 | 3,213 | 8.82×10⁻⁶ |

All graph nodes are in `ai.onnx`, opset 17. `export-report.json` in
`E:/TerrainDiffusionRuntime/webgpu-models/` contains the exact per-model
operator counts, tensor dimensions, checksums and timings. The union of actual
operators is Add, Cast, Clip, Concat, Constant, ConstantOfShape, Conv, Cos,
Div, Einsum, Gather, MatMul, Mul, Pow, ReduceL2, ReduceMean, ReduceSum,
Reshape, Shape, Sigmoid, Sin, Slice, Softmax, Split, Sqrt, Squeeze, Tile,
Transpose and Unsqueeze. The [ORT WebGPU operator table for v1.30.0](https://github.com/microsoft/onnxruntime/blob/v1.30.0/js/web/docs/webgpu-operators.md)
lists these. It marks Shape and Reshape as metadata operations without GPU
kernels. This is the JSEP operator table, while the observed browser stack
uses the native WebGPU asyncify runtime; the table is an inventory hint, not
proof of native backend coverage or GPU residency.

The local `webgpu` package pins ONNX Runtime Web 1.30.0 and vendors its WebGPU
script plus matching JSEP module/WASM. The [official WebGPU guide](https://onnxruntime.ai/docs/tutorials/web/ep-webgpu.html)
documents the WebGPU-only provider, GPU tensors and conditional graph capture.
The probe requires `executionProviders: [{ name: 'webgpu', preferredLayout: 'NCHW' }]`. A failure is shown as a
failure; there is no WASM execution provider in the session options. The
earlier device injection was ignored by this runtime (ORT's actual device did
not match); it was removed. Device sharing with the renderer is unverified.
Provider selection and successful capture replay still do not independently
trace every node's placement; GPU-only residency remains unproven. The [large-model guidance](https://onnxruntime.ai/docs/tutorials/web/large-models.html)
warns about browser ArrayBuffer limits, 2 GB protobuf files and 4 GB WASM
memory. The 2.03 GB base graph is near these limits, before initialization,
weights, activations, other graphs, textures or caching. The original inline
base graph caused browser `std::bad_alloc`. Splitting it into a 511 KB ONNX
file and a 2.03 GB external-weight file allowed the browser to load it. The
exporter now produces and verifies this pair when exporting base. Its
`--materialize-base-external-only` mode backfills an already verified inline
export; `export-report.json` records SHA-256 hashes of both files and the
source inline graph. Reuse verifies graph topology and stream-hashes every
initializer against the inline
graph: all 304 match (266 external, 38 embedded), with 4 MiB read chunks.
The actual pair was additionally tested through native ORT real-feed and
full-crop runs.

The first real production forward for each model was captured from upstream
PyTorch CUDA and rerun through native ORT CPU with identical feeds. The CPU
max/RMSE differences against CUDA were coarse 0.00656/0.000564, base
0.01212/0.000617, and decoder 0.00176/0.000210. The larger real-feed
differences for coarse/base than the dummy-input export checks mean those
dummy checks do not establish a production tolerance; CUDA arithmetic may
contribute. In the historical provider-string run, browser WebGPU differed
from captured CUDA by coarse 0.00536/0.000540, base 15.024/3.255, and
decoder 30.216/4.949. With the provider explicitly set to NCHW, first real
forward comparisons against native ORT CPU are coarse max 0.001734/RMSE
0.000168, decoder 0.001752/0.000187, and base **0.254596/0.009950**. Three
repeated no-capture forwards were bit-identical for each model. Base still
fails the forward gate; the other two pass.
Historical browser forward reports came from a one-run harness; they showed
the first capture but not replay. The current harness makes three identical
calls and checks replay, using stored native ORT CPU outputs as its numerical
oracle and reporting CUDA differences separately. Its no-capture mode has
been rerun on the GPU. Reports generated by the revised harness stamp source,
fixture and model hashes plus ORT and Chrome versions.

The historical decoder fault was localized on the exact production input.
The browser matched native ORT CPU **exactly** through the 1×6×512×512
`Concat`; computed 64×6×3×3 weights matched within 3.6×10⁻⁷. At the first
`Conv`, WebGPU differed by max 16.89 and RMSE 1.44. Freezing the weights or
replacing nearest-neighbor `Tile` upsampling with `Resize` did not fix it.
The isolated repro is
`decoder_first_conv_repro.onnx` with input
`decoder_diag_39_reference.bin` (1×6×512×512) and CPU output
`decoder_diag_49_reference.bin` (1×64×512×512), generated by
`webgpu/decoder_diagnostic.py`. It is a single Conv with the exact production
weights, padding and shapes. The 49-node decoder prefix demonstrated the
original browser divergence; the single-Conv browser matrix below exercised
the same operation in isolation.

A follow-up single-Conv matrix isolated its cause to the layout path. For
the exact 1×6×512×512 first Conv, explicit NCHW yields WebGPU/ORT CPU max
3.34×10⁻⁶, while explicit NHWC reproduces max 16.89058/RMSE 1.44249.
Zero-padding input and fixed weights from six to eight channels leaves the
CPU result within 3.34×10⁻⁶ and makes NHWC accurate (max 1.91×10⁻⁶). The
64×64 variants with 4, 6, 8 and 12 channels all work **under explicit NCHW**;
NHWC was tested only at 512×512. This demonstrates a
specific ORT WebGPU layout/channel bug for this Conv; it does not explain the
remaining base model divergence. The pinned artifacts and results are
`conv-channel-repro.json` and `conv-channel-browser.json`, produced by
`webgpu/conv_channel_repro.py` and `webgpu/verify_conv_channel_repro.cjs`.

Source review found a separate JavaScript port defect: the base model's final
conditioning component was zero, while upstream converts `noise_level=0` to
`(0−0.5)√12`, which becomes −5.385165 after magnitude-preserving concatenation.
The implementation now matches all 58 values of the captured first production
base conditioning vector in a Node test. Fusion also asserts every overlapping
tile is present. `verify_capture_inputs.mjs` compares all feeds for the first
coarse, base and decoder production forwards: coarse `x` differs by at most
4.77×10⁻⁷, base `cond_0` by 4.77×10⁻⁷, and every other feed is exact.

The same `crop.mjs` orchestration was run with an injected **diagnostic-only**
native ORT CPU backend (`onnxruntime-node@1.30.0`), loading one model at a time.
All 75 forwards completed and the current crop gate passed: height maximum
error 0.51013 m (RMSE 0.23279 m), and per-channel climate maxima 0.00547 °C,
0.02411 °C, 0.12030 mm/year, 0.00183%, and 1.09×10⁻⁶ °C/m. The largest
weighted-tile errors were coarse 0.537, base pass 1 0.0331, base pass 2
0.0437, decoder 0.00788. This is evidence for the JavaScript scheduler,
fusion and reconstruction on this one upland crop, independently of WebGPU.
`cpu-crop-diagnostic.json` stores the result. It is not a browser CPU fallback
and does not satisfy WebGPU E2. The current CUDA capture records FP32 tensors
but not TF32 flags, so the source of residual CPU/CUDA differences is
unresolved. A TF32-disabled reference is still needed before fixing general
acceptance thresholds.

The [ORT provider API](https://github.com/microsoft/onnxruntime/blob/v1.30.0/js/common/lib/inference-session.ts)
documents NCHW as the default, yet the provider-string run behaved like the
failing layout path for this Conv. The runtime's effective layout choice is
not directly inspected, so this is an inference. The probe now requests
NCHW explicitly. The next operator-level task is to localize the base model's
first divergence on its real feed under NCHW, then correct the model or
runtime path and rerun the full crop. Current numerical failure still blocks
E2 even though capture replay now completes.

The browser runner in `webgpu/crop.mjs` implements the specific upstream
schedule and dependency path: portable PCG/Marsaglia tile noise, 20 coarse EDM
DPM-Solver++ steps, two base passes, decoder windows, weighted overlap fusion,
Laplacian denoise/decode and signed-square elevation, and coarse-dependent
temperature regression plus the remaining four climate fields. These math
components pass independent PyTorch fixture tests in Node. The integrated crop
does **not** pass in Chromium with WebGPU, so E2 remains NO-GO on this adapter
and ORT Web 1.30. The corrected pipeline passes one CPU diagnostic crop; other
sites, batches and coastline topology remain unproven. The provisional
full-crop gate requires ≤1 m maximum height difference and per-channel climate
limits of 0.1 °C temperature, 0.1 °C temperature variability, 1 mm/year
precipitation, 0.1% precipitation variability, and 1×10⁻⁵ °C/m lapse rate,
as well as graph capture and all three model stages. They passed one
diagnostic CPU crop but need broader calibration, including a reference with
recorded TF32 settings. Coastal topology and visual quality require separate
inspection. `crop.mjs` reports `numericalStatus` separately from `l1Status`:
the CPU diagnostic is `NOT_APPLICABLE` for L1, and a no-capture browser run
cannot return L1 PASS even if its numbers pass. The saved browser crop reports
predate these structured status fields; the source and tests enforce them for
future runs. No new GPU measurement was made for this reporting-only change.
Future browser reports hash the actual requested model and runtime files and
verify every tensor declared by the fixture manifest against its size/SHA-256.
These added provenance fields are not retroactively attributed to old reports.

## Reproduce

From the workspace root, using the application Python environment:

```powershell
& E:/TerrainDiffusionRuntime/venv/Scripts/python.exe -m pip install onnx==1.23.2 onnxruntime==1.30.0
npm ci --prefix webgpu
& E:/TerrainDiffusionRuntime/venv/Scripts/python.exe webgpu/export_models.py --device cuda
& E:/TerrainDiffusionRuntime/venv/Scripts/python.exe webgpu/export_models.py --materialize-base-external-only
& E:/TerrainDiffusionRuntime/venv/Scripts/python.exe webgpu/verify_external_data.py
& E:/TerrainDiffusionRuntime/venv/Scripts/python.exe webgpu/make_math_fixture.py
node --test webgpu/*.test.mjs
& E:/TerrainDiffusionRuntime/venv/Scripts/python.exe webgpu/capture_crop.py
& E:/TerrainDiffusionRuntime/venv/Scripts/python.exe webgpu/make_conditioning_fixture.py
& E:/TerrainDiffusionRuntime/venv/Scripts/python.exe webgpu/verify_real_feeds.py
node webgpu/verify_capture_inputs.mjs
node webgpu/run_cpu_crop.mjs
& E:/TerrainDiffusionRuntime/venv/Scripts/python.exe webgpu/serve_probe.py
node webgpu/verify_browser.cjs coarse_model --reference
node webgpu/verify_browser.cjs base_model --reference
node webgpu/verify_browser.cjs decoder_model --reference
node webgpu/verify_browser.cjs crop
node webgpu/verify_browser.cjs crop --no-capture
& E:/TerrainDiffusionRuntime/venv/Scripts/python.exe webgpu/decoder_diagnostic.py
node webgpu/verify_decoder_diag.cjs
& E:/TerrainDiffusionRuntime/venv/Scripts/python.exe webgpu/conv_channel_repro.py
node webgpu/verify_conv_channel_repro.cjs
```

`capture_crop.py` and `--device cuda` need an exclusive GPU window. Artifacts
live on E: and are not copied into the source tree. `probe.html` can also be
opened through `/webgpu/probe.html` on the application server, which exposes
the same model artifact directory at `/webgpu-models/`.

The benchmark path currently reads tensors back to JavaScript between neural
steps and runs fusion/reconstruction on CPU. That is a correctness probe, not
an ultra-fast implementation. Latency and peak memory must be measured after
the full crop passes. GPU-resident tensor flow and sharing the device with the
renderer remain future work; no speedup or world-scale throughput is claimed.

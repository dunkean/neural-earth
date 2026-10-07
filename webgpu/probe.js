// A placement and memory probe. Full terrain generation lives in a later crop
// runner; an isolated forward must never be labeled a completed world pipeline.
import { makeGpuRunner } from './io.mjs';
const status = document.querySelector('#status');
const SHAPES = {
  coarse_model: { x: [1, 11, 64, 64], noise_labels: [1], cond_0: [1], cond_1: [1], cond_2: [1], cond_3: [1], cond_4: [1] },
  base_model: { x: [1, 5, 64, 64], noise_labels: [1], cond_0: [1, 58] },
  decoder_model: { x: [1, 5, 512, 512], noise_labels: [1] },
};
const OUTPUT_SHAPES = { coarse_model: [1, 6, 64, 64], base_model: [1, 5, 64, 64], decoder_model: [1, 1, 512, 512] };

function makeInput(shape, salt) {
  const length = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(length);
  for (let i = 0; i < length; i++) data[i] = Math.fround(Math.sin((i + salt) * 0.017) * 0.125);
  return data;
}

function log(message) {
  status.textContent += `${message}\n`;
}

async function loadReference(model) {
  const response = await fetch('/webgpu-models/crop64-coast-manifest.json');
  if (!response.ok) throw new Error(`Reference manifest HTTP ${response.status}`);
  const manifest = await response.json();
  const record = manifest.first_real_forward[model];
  if (!record) throw new Error(`No captured ${model} forward in native crop`);
  const tensors = {};
  for (const [name, spec] of Object.entries(record)) {
    const value = await fetch(`/webgpu-models/${spec.file}`);
    if (!value.ok) throw new Error(`Reference tensor HTTP ${value.status}: ${spec.file}`);
    tensors[name] = { shape: spec.shape, data: new Float32Array(await value.arrayBuffer()) };
  }
  const cpuReportResponse = await fetch('/webgpu-models/real-feed-ort-cpu-comparison.json');
  if (!cpuReportResponse.ok) throw new Error('ORT CPU real-feed oracle missing');
  const cpuReport = await cpuReportResponse.json();
  const cpuSpec = cpuReport[model].cpu_output;
  const cpuResponse = await fetch(`/webgpu-models/${cpuSpec.file}`);
  if (!cpuResponse.ok) throw new Error(`ORT CPU output HTTP ${cpuResponse.status}`);
  const cpuOutput = new Float32Array(await cpuResponse.arrayBuffer());
  return { manifest, tensors, cpuOutput, cpuSpec };
}

function difference(actual, expected) {
  if (actual.length !== expected.length) throw new Error('Output size mismatch');
  let maxAbs = 0, squares = 0;
  for (let i = 0; i < actual.length; i++) {
    const d = actual[i] - expected[i];
    maxAbs = Math.max(maxAbs, Math.abs(d)); squares += d * d;
  }
  return { maxAbs, rmse: Math.sqrt(squares / actual.length) };
}

async function run(model, useReference = false, requireGraphCapture = true, staticBatch = false, resizeUp = false) {
  if (!(model in SHAPES)) throw new Error(`Unknown model ${model}`);
  status.textContent = '';
  if (!navigator.gpu) throw new Error('navigator.gpu unavailable; WebGPU is required.');
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter available.');
  const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : {});
  log(`GPU adapter: ${info.vendor || '?'} / ${info.device || info.description || '?'}; adapter maxStorageBufferBindingSize=${adapter.limits.maxStorageBufferBindingSize}`);
  ort.env.wasm.wasmPaths = new URL('./vendor/', import.meta.url).href;
  ort.env.wasm.numThreads = 1;
  const url = `/webgpu-models/${model === 'base_model' ? 'base_model_external' : model === 'decoder_model' && resizeUp ? 'decoder_model_resize' : model === 'decoder_model' && staticBatch ? 'decoder_model_static' : model}.onnx`;
  const began = performance.now();
  log(`Loading ${url} with WebGPU-only provider; graph capture=${requireGraphCapture}…`);
  const reference = useReference ? await loadReference(model) : null;
  const session = await ort.InferenceSession.create(url, {
    executionProviders: [{ name: 'webgpu', preferredLayout: 'NCHW' }],
    enableGraphCapture: requireGraphCapture,
    freeDimensionOverrides: { batch: 1 },
    graphOptimizationLevel: 'all',
    ...(model === 'base_model' ? { externalData: [{ path: 'base_model_external.data', data: '/webgpu-models/base_model_external.data' }] } : {}),
  });
  const loadedMs = performance.now() - began;
  log(`Session loaded in ${loadedMs.toFixed(1)} ms`);
  const runtimeDevice = await ort.env.webgpu.device;
  log(`ORT device maxStorageBufferBindingSize=${runtimeDevice.limits.maxStorageBufferBindingSize}`);
  const inputs = {};
  for (const [index, name] of session.inputNames.entries()) {
    const shape = SHAPES[model][name];
    if (!shape) throw new Error(`Unexpected input ${name}`);
    if (reference) {
      const tensor = reference.tensors[name];
      if (!tensor) throw new Error(`Captured feed ${name} missing`);
      inputs[name] = tensor.data;
    } else {
      inputs[name] = makeInput(shape, index + 23);
    }
  }
  const runner = makeGpuRunner(ort, runtimeDevice, session, SHAPES[model], OUTPUT_SHAPES[model]);
  let data;
  const runMs = [];
  const replay = [];
  try {
    for (let index = 0; index < 3; index++) {
      log(`Forward ${index + 1}/3`);
      const startRun = performance.now();
      const current = await runner.run(inputs);
      runMs.push(performance.now() - startRun);
      if (index === 0) data = current;
      else replay.push(difference(current, data));
    }
  let min = Infinity, max = -Infinity, sum = 0;
  for (const value of data) {
    if (!Number.isFinite(value)) throw new Error('Non-finite output');
    min = Math.min(min, value); max = Math.max(max, value); sum += value;
  }
  const result = { model, adapter: info, limits: { maxStorageBufferBindingSize: runtimeDevice.limits.maxStorageBufferBindingSize }, loadedMs, runMs, replay, elements: data.length, min, max, mean: sum / data.length, graphCaptureRequired: requireGraphCapture, capturedRealForward: useReference, staticBatch, resizeUp };
  if (replay.some(item => item.maxAbs > 1e-5)) throw new Error(`Graph replay changed output ${JSON.stringify(replay)}`);
  if (reference) {
    result.cudaComparison = difference(data, reference.tensors.output.data);
    result.forwardComparison = { ...difference(data, reference.cpuOutput), reference: 'pinned ORT CPU 1.30.0 real feed', maxAbsLimit: 0.01, rmseLimit: 0.001 };
    log(`Comparison ${JSON.stringify(result)}`);
    if (result.forwardComparison.maxAbs > 0.01 || result.forwardComparison.rmse > 0.001) throw new Error(`Captured ${model} forward mismatch ${JSON.stringify(result.forwardComparison)}`);
  }
  log(`PASS ${JSON.stringify(result)}`);
  return result;
  } finally {
    runner.dispose();
    await session.release();
  }
}

window.terrainWebGpuProbe = { run };
for (const button of document.querySelectorAll('[data-model]')) {
  button.addEventListener('click', () => run(button.dataset.model).catch(error => {
    log(`FAIL ${error.stack || error}`);
  }));
}
for (const button of document.querySelectorAll('[data-reference]')) {
  button.addEventListener('click', () => run(button.dataset.reference, true).catch(error => {
    log(`FAIL ${error.stack || error}`);
  }));
}
document.querySelector('#run-crop').addEventListener('click', async () => {
  status.textContent = '';
  try {
    const { runCrop } = await import('./crop.mjs');
    const report = await runCrop(ort, log);
    log(`${report.l1Status === 'PASS' ? 'PASS' : 'NO-GO'} ${JSON.stringify(report)}`);
  } catch (error) {
    log(`FAIL ${error.stack || error}`);
  }
});

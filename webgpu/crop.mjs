import { gaussianNoisePatch } from './noise.mjs';
import { createSolver, noiseLabel, preconditionInput } from './scheduler.mjs';
import { linearWeightWindow, fuseWeightedTiles, assertFullContributors } from './fusion.mjs';
import { laplacianReconstruct } from './laplacian.mjs';
import { climateFromCoarse } from './climate.mjs';
import { makeGpuRunner } from './io.mjs';

const f32 = Math.fround;
const MEANS = [-37.70000792952155, 1.1403065255556186, 18.102486588653473, 332.8342598198454, 1332.2078969994473, 52.660088206981435];
const STDS = [39.741999742263, 1.7681844104569366, 8.92146918789914, 321.7660336396054, 842.9293648884745, 31.079985318715785];
const COND_MEAN = [14.99, 11.65, 15.87, 619.26, 833.12, 69.40, 0.66];
const COND_STD = [21.72, 21.78, 10.40, 452.29, 738.09, 34.59, 0.47];
const MODEL_ROOT = '/webgpu-models/';
// Provisional physical-unit limits; recalibrate against a TF32-disabled CPU crop.
const CLIMATE_LIMITS = [0.1, 0.1, 1, 0.1, 1e-5]; // °C, °C, mm/yr, %, °C/m

async function loadArray(spec) {
  const response = await fetch(`${MODEL_ROOT}${spec.file}`);
  if (!response.ok) throw new Error(`Fixture HTTP ${response.status}: ${spec.file}`);
  const data = new Float32Array(await response.arrayBuffer());
  if (data.length !== spec.shape.reduce((a, b) => a * b, 1)) throw new Error(`Fixture shape mismatch: ${spec.file}`);
  return data;
}

async function sessionFor(ort, name, requireGraphCapture) {
  return ort.InferenceSession.create(`${MODEL_ROOT}${name === 'base_model' ? 'base_model_external' : name}.onnx`, {
    executionProviders: [{ name: 'webgpu', preferredLayout: 'NCHW' }], enableGraphCapture: requireGraphCapture,
    freeDimensionOverrides: { batch: 1 }, graphOptimizationLevel: 'all',
    ...(name === 'base_model' ? { externalData: [{ path: 'base_model_external.data', data: `${MODEL_ROOT}base_model_external.data` }] } : {}),
  });
}

async function infer(runner, feeds) {
  return runner.run(Object.fromEntries(Object.entries(feeds).map(([name, [data]]) => [name, data])));
}

function weightedTile(data, channels, size) {
  const plane = size * size;
  const weight = linearWeightWindow(size);
  const out = new Float32Array((channels + 1) * plane);
  for (let c = 0; c < channels; c++) {
    for (let p = 0; p < plane; p++) out[c * plane + p] = f32(data[c * plane + p] * weight[p]);
  }
  out.set(weight, channels * plane);
  return out;
}

function maxDifference(a, b) {
  if (a.length !== b.length) throw new Error(`Comparison length mismatch ${a.length} != ${b.length}`);
  let maxAbs = 0, square = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    maxAbs = Math.max(maxAbs, Math.abs(d)); square += d * d;
  }
  return { maxAbs, rmse: Math.sqrt(square / a.length) };
}

export function classifyCropStatus(numericallyPass, diagnosticBackend, requireGraphCapture) {
  const numericalStatus = numericallyPass ? 'PASS' : 'FAIL';
  const l1Status = diagnosticBackend
    ? 'NOT_APPLICABLE'
    : !numericallyPass
      ? 'FAIL'
      : requireGraphCapture ? 'NOT_ESTABLISHED' : 'NOT_ESTABLISHED_NO_CAPTURE';
  return { numericalStatus, l1Status };
}

function tilesInRegion(stageTiles, channels, size, stride, y, x, height, width) {
  assertFullContributors(stageTiles, size, stride, y, x, height, width);
  return fuseWeightedTiles(stageTiles, channels, size, y, x, height, width);
}

export function baseConditioning(coarse4) {
  // Six physical coarse channels plus an all-one mask, normalized and grouped
  // exactly like _process_latent_conditioning (zero noise, histogram_raw=0).
  const plane = 16;
  const normalized = new Float32Array(7 * plane);
  for (let c = 0; c < 7; c++) {
    for (let p = 0; p < plane; p++) {
      const value = c === 6 ? 1 : coarse4[c * plane + p];
      normalized[c * plane + p] = f32((value - COND_MEAN[c]) / COND_STD[c]);
    }
  }
  const groups = [
    normalized.subarray(0, 16), normalized.subarray(16, 32),
    Float32Array.from({ length: 4 }, (_, index) => {
      const c = index + 2;
      return f32((normalized[c * plane + 5] + normalized[c * plane + 6] + normalized[c * plane + 9] + normalized[c * plane + 10]) / 4);
    }),
    normalized.subarray(6 * plane, 7 * plane),
    new Float32Array(5), new Float32Array([f32(-0.5 * Math.sqrt(12))]),
  ];
  const output = new Float32Array(58);
  const C = Math.sqrt(58 / (6 * (1 / 6) ** 2));
  let offset = 0;
  for (const group of groups) {
    const scale = C / Math.sqrt(group.length) / 6;
    for (let i = 0; i < group.length; i++) output[offset + i] = f32(group[i] * scale);
    offset += group.length;
  }
  return output;
}

export function coarseInputs(inputMap, seed, ty, tx) {
  const plane = 64 * 64;
  const cond = new Float32Array(5 * plane);
  const condNoise = gaussianNoisePatch(seed, ty * 48, tx * 48, 64, 64, 5, 64, 64);
  const c = f32(Math.cos(Math.atan(0.5)));
  const s = f32(Math.sin(Math.atan(0.5)));
  const indices = [0, 2, 3, 4, 5];
  for (let k = 0; k < 5; k++) {
    for (let p = 0; p < plane; p++) {
      const normalized = f32((inputMap[k * plane + p] - MEANS[indices[k]]) / STDS[indices[k]]);
      cond[k * plane + p] = f32(f32(c * normalized) + f32(s * condNoise[k * plane + p]));
    }
  }
  const startNoise = gaussianNoisePatch(seed + 1n, ty * 48, tx * 48, 64, 64, 6, 64, 64);
  return { cond, startNoise };
}

async function generateCoarse(runner, inputMap, seed, ty, tx, onProgress) {
  const { cond, startNoise } = coarseInputs(inputMap, seed, ty, tx);
  const solver = createSolver(20);
  let sample = Float32Array.from(startNoise, value => f32(value * solver.sigmas[0]));
  const scalar = new Float32Array([f32(Math.log(0.5 / 8))]);
  for (let step = 0; step < 20; step++) {
    onProgress(`coarse ${ty},${tx} step ${step + 1}`);
    const x = new Float32Array(11 * 64 * 64);
    x.set(preconditionInput(sample, solver.sigmas[step]));
    x.set(cond, 6 * 64 * 64);
    const feeds = { x: [x, [1, 11, 64, 64]], noise_labels: [new Float32Array([noiseLabel(solver.sigmas[step])]), [1]] };
    for (let k = 0; k < 5; k++) feeds[`cond_${k}`] = [scalar, [1]];
    const modelOutput = await infer(runner, feeds);
    sample = solver.step(sample, modelOutput);
  }
  const plane = 64 * 64;
  const physical = new Float32Array(sample.length);
  for (let c = 0; c < 6; c++) for (let p = 0; p < plane; p++) {
    physical[c * plane + p] = f32(f32(sample[c * plane + p] / 0.5) * STDS[c] + MEANS[c]);
  }
  for (let p = 0; p < plane; p++) physical[plane + p] = f32(physical[p] - physical[plane + p]);
  return weightedTile(physical, 6, 64);
}

export function baseFeeds(coarseTiles, previousTiles, seed, ty, tx, pass) {
  const coarse4 = tilesInRegion(coarseTiles, 6, 64, 48, ty - 1, tx - 1, 4, 4);
  const condition = baseConditioning(coarse4);
  const t = pass === 1 ? Math.atan(80 / 0.5) : Math.atan(0.35 / 0.5);
  const c = f32(Math.cos(t)), s = f32(Math.sin(t));
  const noise = gaussianNoisePatch(seed + BigInt(pass === 1 ? 5819 : 5820), ty * 32, tx * 32, 64, 64, 5, 64, 64);
  const previous = pass === 1 ? new Float32Array(5 * 64 * 64) : tilesInRegion(previousTiles, 5, 64, 32, ty * 32, tx * 32, 64, 64);
  const xT = new Float32Array(previous.length);
  const modelInput = new Float32Array(previous.length);
  for (let i = 0; i < previous.length; i++) {
    xT[i] = f32(f32(c * f32(previous[i] * 0.5)) + f32(s * f32(noise[i] * 0.5)));
    modelInput[i] = f32(xT[i] / 0.5);
  }
  return { modelInput, noiseLabels: new Float32Array([f32(t)]), condition, xT, c, s };
}

async function generateBase(runner, coarseTiles, previousTiles, seed, ty, tx, pass) {
  const { modelInput, noiseLabels, condition, xT, c, s } = baseFeeds(coarseTiles, previousTiles, seed, ty, tx, pass);
  const output = await infer(runner, { x: [modelInput, [1, 5, 64, 64]], noise_labels: [noiseLabels, [1]], cond_0: [condition, [1, 58]] });
  const sample = new Float32Array(output.length);
  for (let i = 0; i < output.length; i++) sample[i] = f32(f32(f32(c * xT[i]) + f32(s * f32(0.5 * output[i]))) / 0.5);
  return weightedTile(sample, 5, 64);
}

export function decoderFeeds(latentTiles, seed, ty, tx) {
  const latents = tilesInRegion(latentTiles, 5, 64, 32, ty * 48, tx * 48, 64, 64);
  const size = 512, plane = size * size, smallPlane = 64 * 64;
  const t = Math.atan(80 / 0.5), c = f32(Math.cos(t)), s = f32(Math.sin(t));
  const noise = gaussianNoisePatch(seed + 5819n, ty * 384, tx * 384, size, size, 1, size, size);
  const x = new Float32Array(5 * plane);
  for (let p = 0; p < plane; p++) {
    x[p] = f32(s * noise[p]);
    const src = Math.floor(Math.floor(p / size) / 8) * 64 + Math.floor((p % size) / 8);
    for (let channel = 0; channel < 4; channel++) x[(channel + 1) * plane + p] = latents[channel * smallPlane + src];
  }
  return { x, noiseLabels: new Float32Array([f32(t)]), noise, c, s };
}

async function generateDecoder(runner, latentTiles, seed, ty, tx) {
  const { x, noiseLabels, noise, c, s } = decoderFeeds(latentTiles, seed, ty, tx);
  const size = 512, plane = size * size;
  const output = await infer(runner, { x: [x, [1, 5, size, size]], noise_labels: [noiseLabels, [1]] });
  const sample = new Float32Array(plane);
  for (let p = 0; p < plane; p++) sample[p] = f32(f32(f32(c * f32(s * f32(noise[p] * 0.5))) + f32(s * f32(0.5 * output[p]))) / 0.5);
  return weightedTile(sample, 1, size);
}

export async function runCrop(ort, onProgress = () => {}, requireGraphCapture = true, diagnosticBackend = null) {
  if (!diagnosticBackend) {
    if (!navigator.gpu) throw new Error('WebGPU unavailable');
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) throw new Error('No WebGPU adapter');
    ort.env.wasm.wasmPaths = new URL('./vendor/', import.meta.url).href;
    ort.env.wasm.numThreads = 1;
  } else if (requireGraphCapture) {
    throw new Error('Graph capture applies only to the WebGPU probe');
  }
  const started = performance.now();
  const fixture = diagnosticBackend ? diagnosticBackend.manifest : await (async () => {
    const response = await fetch(`${MODEL_ROOT}crop64-coast-manifest.json`);
    if (!response.ok) throw new Error(`Crop fixture missing HTTP ${response.status}`);
    return response.json();
  })();
  const read = diagnosticBackend ? diagnosticBackend.loadArray : loadArray;
  if (fixture.checkpoint_revision !== '9ef8030cb805b433b98ec25c5dddefbac07a9e26' || fixture.seed !== '42' || fixture.profile !== 'natural') throw new Error('Crop identity mismatch');
  const seed = BigInt(fixture.seed);
  let runtimeDevice;
  const discrepancies = {};
  const times = {};
  const stages = [
    ['coarse', 'coarse_model', 6, 64, 48],
    ['base_pass1', 'base_model', 5, 64, 32],
    ['base_pass2', 'base_model', 5, 64, 32],
    ['decoder', 'decoder_model', 1, 512, 384],
  ];
  const results = {};
  let session;
  let runner;
  let currentModel;
  try {
    for (const [stage, model, channels, size, stride] of stages) {
      if (model !== currentModel) {
        if (runner) runner.dispose();
        if (session) await session.release();
        onProgress(`Loading ${model}`);
        session = diagnosticBackend ? await diagnosticBackend.createSession(model) : await sessionFor(ort, model, requireGraphCapture);
        if (!diagnosticBackend) runtimeDevice = await ort.env.webgpu.device;
        const shapes = model === 'coarse_model'
          ? { x: [1, 11, 64, 64], noise_labels: [1], cond_0: [1], cond_1: [1], cond_2: [1], cond_3: [1], cond_4: [1] }
          : model === 'base_model'
            ? { x: [1, 5, 64, 64], noise_labels: [1], cond_0: [1, 58] }
            : { x: [1, 5, 512, 512], noise_labels: [1] };
        const outputShape = model === 'coarse_model' ? [1, 6, 64, 64] : model === 'base_model' ? [1, 5, 64, 64] : [1, 1, 512, 512];
        runner = diagnosticBackend
          ? diagnosticBackend.makeRunner(session, shapes, outputShape)
          : makeGpuRunner(ort, runtimeDevice, session, shapes, outputShape);
        currentModel = model;
      }
      const t0 = performance.now();
      const tiles = [];
      let stageMax = 0;
      for (const entry of fixture.windows[stage]) {
        const [, ty, tx] = entry.ctx;
        let data;
        if (stage === 'coarse') data = await generateCoarse(runner, await read(entry.input_map), seed, ty, tx, onProgress);
        else if (stage === 'base_pass1') data = await generateBase(runner, results.coarse, [], seed, ty, tx, 1);
        else if (stage === 'base_pass2') data = await generateBase(runner, results.coarse, results.base_pass1, seed, ty, tx, 2);
        else data = await generateDecoder(runner, results.base_pass2, seed, ty, tx);
        const expected = await read(entry.tile);
        stageMax = Math.max(stageMax, maxDifference(data, expected).maxAbs);
        tiles.push({ y: ty * stride, x: tx * stride, data });
        onProgress(`${stage}: ${tiles.length}/${fixture.windows[stage].length}`);
      }
      results[stage] = tiles;
      discrepancies[stage] = stageMax;
      times[stage] = performance.now() - t0;
    }
  } finally {
    if (runner) runner.dispose();
    if (session) await session.release();
  }
  const [i1, j1, i2, j2] = fixture.bbox_row_col;
  const scale = 8, padHr = 48;
  const pi1 = Math.floor((i1 - padHr) / scale) * scale;
  const pj1 = Math.floor((j1 - padHr) / scale) * scale;
  const pi2 = Math.ceil((i2 + padHr) / scale) * scale;
  const pj2 = Math.ceil((j2 + padHr) / scale) * scale;
  const highH = pi2 - pi1, highW = pj2 - pj1;
  const residual = tilesInRegion(results.decoder, 1, 512, 384, pi1, pj1, highH, highW);
  for (let i = 0; i < residual.length; i++) residual[i] = f32(residual[i] * 0.7);
  const lowH = highH / scale, lowW = highW / scale;
  const latentLow = tilesInRegion(results.base_pass2, 5, 64, 32, pi1 / scale, pj1 / scale, lowH, lowW);
  const lowres = Float32Array.from(latentLow.subarray(4 * lowH * lowW), value => f32(f32(value * 38.6) - 31.4));
  const fullHeight = laplacianReconstruct(residual, lowres, highH, highW, lowH, lowW);
  const cropH = i2 - i1, cropW = j2 - j1;
  const height = new Float32Array(cropH * cropW);
  for (let y = 0; y < cropH; y++) height.set(fullHeight.subarray((i1 - pi1 + y) * highW + j1 - pj1, (i1 - pi1 + y) * highW + j1 - pj1 + cropW), y * cropW);
  const ci1 = Math.floor(i1 / 256), cj1 = Math.floor(j1 / 256);
  const ci2 = Math.ceil(i2 / 256), cj2 = Math.ceil(j2 / 256);
  const coarseH = ci2 - ci1 + 16, coarseW = cj2 - cj1 + 16;
  const coarse = tilesInRegion(results.coarse, 6, 64, 48, ci1 - 8, cj1 - 8, coarseH, coarseW);
  const climate = climateFromCoarse(coarse, coarseH, coarseW, height, i1, j1, cropH, cropW, scale);
  const heightComparison = maxDifference(height, await read(fixture.elevation));
  const climateExpected = await read(fixture.climate);
  const climateComparison = maxDifference(climate, climateExpected);
  const climatePlane = cropH * cropW;
  const climateByChannel = CLIMATE_LIMITS.map((limit, channel) => ({
    channel, limit, ...maxDifference(
      climate.subarray(channel * climatePlane, (channel + 1) * climatePlane),
      climateExpected.subarray(channel * climatePlane, (channel + 1) * climatePlane),
    ),
  }));
  const report = {
    ...classifyCropStatus(heightComparison.maxAbs <= 1 && climateByChannel.every(item => item.maxAbs <= item.limit), !!diagnosticBackend, requireGraphCapture),
    crop: fixture.bbox_row_col, seed: fixture.seed, profile: fixture.profile,
    heightComparison, climateComparison, climateByChannel, stageMaxAbs: discrepancies,
    stageMs: times, totalMs: performance.now() - started,
    modelCalls: { coarse: fixture.windows.coarse.length * 20, base: fixture.windows.base_pass1.length + fixture.windows.base_pass2.length, decoder: fixture.windows.decoder.length },
    limits: { heightMaxAbsM: 1, climateByChannel: CLIMATE_LIMITS },
    placement: diagnosticBackend ? 'diagnostic native ORT CPU; never used in product' : `WebGPU executionProviders only; graph capture ${requireGraphCapture ? 'required' : 'disabled for diagnosis'}`,
  };
  return report;
}

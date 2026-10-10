// Diagnose browser input assembly against first real forwards, without a GPU.
import fs from 'node:fs';
import path from 'node:path';
import platform from '../tools/platform.cjs';
import { coarseInputs, baseFeeds, decoderFeeds } from './crop.mjs';
import { createSolver, noiseLabel, preconditionInput } from './scheduler.mjs';

const root = process.env.TERRAIN_WEBGPU_MODELS || platform.runtimePath('webgpu-models');
const manifest = JSON.parse(fs.readFileSync(path.join(root, 'crop64-coast-manifest.json')));
const f32 = Math.fround;

function load(spec) {
  const bytes = fs.readFileSync(path.join(root, spec.file));
  const raw = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
  const array = new Float32Array(raw);
  if (array.length !== spec.shape.reduce((a, b) => a * b, 1)) throw new Error(`Shape mismatch: ${spec.file}`);
  return array;
}

function compare(actual, spec) {
  const expected = load(spec);
  if (actual.length !== expected.length) throw new Error(`Length mismatch: ${spec.file}`);
  let maxAbs = 0, squares = 0;
  for (let i = 0; i < actual.length; i++) {
    const error = actual[i] - expected[i];
    maxAbs = Math.max(maxAbs, Math.abs(error));
    squares += error * error;
  }
  return { maxAbs, rmse: Math.sqrt(squares / actual.length) };
}

function tiles(stage, stride) {
  return manifest.windows[stage].map(entry => ({
    y: entry.ctx[1] * stride, x: entry.ctx[2] * stride, data: load(entry.tile),
  }));
}

const seed = BigInt(manifest.seed);
const coarseEntry = manifest.windows.coarse[0];
const [, coarseTy, coarseTx] = coarseEntry.ctx;
const coarse = coarseInputs(load(coarseEntry.input_map), seed, coarseTy, coarseTx);
const solver = createSolver(20);
const sample = Float32Array.from(coarse.startNoise, value => f32(value * solver.sigmas[0]));
const coarseX = new Float32Array(11 * 64 * 64);
coarseX.set(preconditionInput(sample, solver.sigmas[0]));
coarseX.set(coarse.cond, 6 * 64 * 64);
const coarseReference = manifest.first_real_forward.coarse_model;

const baseEntry = manifest.windows.base_pass1[0];
const [, baseTy, baseTx] = baseEntry.ctx;
const base = baseFeeds(tiles('coarse', 48), [], seed, baseTy, baseTx, 1);
const baseReference = manifest.first_real_forward.base_model;

const decoderEntry = manifest.windows.decoder[0];
const [, decoderTy, decoderTx] = decoderEntry.ctx;
const decoder = decoderFeeds(tiles('base_pass2', 32), seed, decoderTy, decoderTx);
const decoderReference = manifest.first_real_forward.decoder_model;

const report = {
  coarse_model: {
    x: compare(coarseX, coarseReference.x),
    noise_labels: compare(new Float32Array([noiseLabel(solver.sigmas[0])]), coarseReference.noise_labels),
    ...Object.fromEntries(Array.from({ length: 5 }, (_, index) => [
      `cond_${index}`,
      compare(new Float32Array([f32(Math.log(0.5 / 8))]), coarseReference[`cond_${index}`]),
    ])),
  },
  base_model: {
    x: compare(base.modelInput, baseReference.x),
    noise_labels: compare(base.noiseLabels, baseReference.noise_labels),
    cond_0: compare(base.condition, baseReference.cond_0),
  },
  decoder_model: {
    x: compare(decoder.x, decoderReference.x),
    noise_labels: compare(decoder.noiseLabels, decoderReference.noise_labels),
  },
};
fs.writeFileSync(path.join(root, 'capture-inputs-js-comparison.json'), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report, null, 2));
if (Object.values(report).some(model => Object.values(model).some(metric => metric.maxAbs > 1e-3))) process.exitCode = 1;

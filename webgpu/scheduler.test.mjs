import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { createSolver, karrasSigmas } from './scheduler.mjs';
import { gaussianNoisePatch } from './noise.mjs';
import { fuseWeightedTiles, linearWeightWindow, assertFullContributors } from './fusion.mjs';
import { laplacianReconstruct } from './laplacian.mjs';
import { climateFromCoarse } from './climate.mjs';

const fixture = JSON.parse(fs.readFileSync(new URL('./math-fixture.json', import.meta.url)));

function maxDiff(a, b) {
  return Math.max(...a.map((value, i) => Math.abs(value - b[i])));
}

test('Karras sigma schedule tracks upstream FP32 values', () => {
  assert.ok(maxDiff(Array.from(karrasSigmas()), fixture.sigmas) < 3e-5);
});

test('twenty EDM solver steps track upstream deterministic tensor', () => {
  const solver = createSolver();
  let sample = Float32Array.from(fixture.initial);
  for (let i = 0; i < 20; i++) {
    const output = Float32Array.from({ length: 8 }, (_, j) => Math.fround(Math.sin((i + 1) * 0.19 + j * 0.37) * 0.25));
    sample = solver.step(sample, output);
    const expected = fixture.solver_snapshots[String(i + 1)];
    if (expected) assert.ok(maxDiff(Array.from(sample), expected) < 1e-4, `step ${i + 1}: ${maxDiff(Array.from(sample), expected)}`);
  }
});

test('full patch matches upstream portable RNG', () => {
  const actual = gaussianNoisePatch(42n, -3, 5, 3, 4, 2, 8, 8);
  const expected = fixture.noise_patch.flat(2);
  assert.deepEqual(Array.from(actual), expected);
});

test('weight window and two-tile fusion match PyTorch reference', () => {
  assert.ok(maxDiff(Array.from(linearWeightWindow(8)), fixture.weight_8.flat()) < 1e-7);
  const f = fixture.fusion;
  const actual = fuseWeightedTiles(f.tiles.map(tile => ({ ...tile, data: Float32Array.from(tile.data) })), f.channels, f.tile_size, f.y0, f.x0, f.height, f.width);
  assert.ok(maxDiff(Array.from(actual), f.expected) < 1e-5);
});

test('fusion rejects a missing overlapping contributor even when coverage is positive', () => {
  const tiles = [{ y: -48, x: -48 }, { y: 0, x: -48 }];
  assert.doesNotThrow(() => assertFullContributors(tiles, 64, 48, 6, -18, 4, 4));
  assert.throws(() => assertFullContributors(tiles.slice(0, 1), 64, 48, 6, -18, 4, 4), /Missing contributing tile/);
});

test('Laplacian denoise/decode and signed square match PyTorch reference', () => {
  const f = fixture.laplacian;
  const actual = laplacianReconstruct(Float32Array.from(f.residual), Float32Array.from(f.lowres), f.high_h, f.high_w, f.low_h, f.low_w);
  assert.ok(maxDiff(Array.from(actual), f.expected_height) < 0.001, `max=${maxDiff(Array.from(actual), f.expected_height)}`);
});

test('coarse temperature regression and all five climate fields match PyTorch reference', () => {
  const f = fixture.climate;
  const actual = climateFromCoarse(Float32Array.from(f.coarse), f.coarse_h, f.coarse_w, Float32Array.from(f.elevation), f.i1, f.j1, f.height, f.width, f.scale);
  assert.ok(maxDiff(Array.from(actual), f.expected) < 0.01, `max=${maxDiff(Array.from(actual), f.expected)}`);
});

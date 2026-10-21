import test from 'node:test';
import assert from 'node:assert/strict';
import {sampleSketch, resizeBilinear, resizeDown8, gaussian, bf16, finiteDecoderInputs} from './tile40_runtime.mjs';
import {gaussianNoisePatch} from './noise.mjs';

test('finite decoder retains world512 noise lattice and8px latent alignment', () => {
  const latents = Float32Array.from({length:5*320*320}, (_, i) => i);
  const inputs = finiteDecoderInputs(latents, 751, 5), n = 1856*1856;
  const noise = gaussianNoisePatch(751n+5819n, 0, 0, 16, 16, 1, 512, 512);
  for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
    assert.equal(inputs[(y+256)*1856+x+256], noise[y*16+x]);
  }
  for (const c of [0, 3]) for (const y of [0, 7, 8, 1855]) for (const x of [0, 7, 8, 1855]) {
    assert.equal(inputs[(c+1)*n+y*1856+x], latents[c*320*320+(32+Math.floor(y/8))*320+32+Math.floor(x/8)]);
  }
});

test('PCHIP matches SciPy physical sketch fixture without amplitude normalization', () => {
  const values = [[0, 100, 0], [100, 300, 100], [0, 100, 0]];
  const axis = [0, 10000, 20000, 30000, 40000];
  assert.deepEqual(Array.from(sampleSketch(values, axis, axis)), [
    0, 75, 100, 75, 0, 75, 206.25, 250, 206.25, 75, 100, 250, 300, 250, 100,
    75, 206.25, 250, 206.25, 75, 0, 75, 100, 75, 0]);
});

test('sketch heights outside40km clamp to physical boundary values', () => {
  assert.deepEqual(Array.from(sampleSketch([[100, 300], [500, 700]], [-40000, 80000], [-1, 50000])),
                   [100, 300, 500, 700]);
});

test('antialiased downsampling matches Torchvision fixture including outer cells', () => {
  const input = Float32Array.from({length:256}, (_, i) => i);
  const actual = resizeDown8(input, 16);
  const expected = [72.55357360839844, 79.01785278320312, 175.98211669921875, 182.44642639160156];
  for (let i = 0; i < 4; i++) assert.ok(Math.abs(actual[i]-expected[i]) < .00004);
});

test('linear extrapolation preserves a physical plane at outer reconstruction boundary', () => {
  const input = Float32Array.from({length:16}, (_, i) => 10+Math.floor(i/4)*2+i%4);
  const actual = resizeBilinear(input, 4, 32, true);
  for (let y = 0; y < 32; y++) for (let x = 0; x < 32; x++) {
    assert.ok(Math.abs(actual[y*32+x]-(10+2*((y+.5)/8-.5)+(x+.5)/8-.5)) < .00001);
  }
});

test('whole-sample and half-sample reflection are distinct and preserve constants', () => {
  const v = new Float32Array(64).fill(10), kernel = [.25, .5, .25];
  assert.deepEqual(gaussian(v, 8, kernel, false), v);
  assert.deepEqual(gaussian(v, 8, kernel, true), v);
  v[0] = 0;
  assert.notEqual(gaussian(v, 8, kernel, true)[0], gaussian(v, 8, kernel, false)[0]);
});

test('BF16 conversion implements tie-to-even and preserves sign', () => {
  assert.equal(bf16(1+1/256), 1);
  assert.equal(bf16(1+3/256), 1+2/128);
  assert.equal(bf16(-1-1/256), -1);
});

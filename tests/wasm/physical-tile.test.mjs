import { test } from 'node:test';
import assert from 'node:assert/strict';
import { physicalTile } from '../../wasm/physical-tile.mjs';
test('physical transport preserves height bytes and climate layer order', () => {
  const n = 304 * 304, fields = new Float32Array(n * 5), heights = new Float32Array(n).fill(123);
  for (let layer = 0; layer < 5; layer++) fields.fill(layer * 100, layer * n, (layer + 1) * n);
  const result = physicalTile(heights, fields, 304, 24, true), values = new Float32Array(result.body);
  assert.equal(values.length, n + 33 * 33 * 5);
  assert.equal(values[0], 123); assert.equal(values[n - 1], 123);
  for (let layer = 0; layer < 4; layer++) assert.equal(values[n + layer * 33 * 33], (layer + 1) * 100);
  assert.ok(Math.abs(values.at(-1) + .006) < 1e-8);
  assert.equal(result.headers['X-Terrain-Climate-Layers'], '5');
  assert.deepEqual(new Float32Array(physicalTile(heights, fields).body), heights);
});

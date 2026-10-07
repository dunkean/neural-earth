import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import { baseConditioning } from './crop.mjs';

const fixture = JSON.parse(fs.readFileSync(new URL('./conditioning-fixture.json', import.meta.url)));

test('first production base conditioning matches captured upstream 58-vector', () => {
  assert.deepEqual(fixture.ctx, [0, 7, -17]);
  const actual = baseConditioning(Float32Array.from(fixture.coarse4));
  const expected = fixture.expected_cond_0;
  assert.equal(actual.length, 58);
  assert.ok(Math.abs(actual[57] + Math.sqrt(29)) < 1e-6);
  let maxAbs = 0;
  for (let i = 0; i < actual.length; i++) maxAbs = Math.max(maxAbs, Math.abs(actual[i] - expected[i]));
  assert.ok(maxAbs < 1e-4, `production conditioning max difference ${maxAbs}`);
});

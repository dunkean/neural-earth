import test from 'node:test';
import assert from 'node:assert/strict';
import { gaussianNoisePatch } from './noise.mjs';

test('portable Gaussian noise matches Python fixture across negative tile boundary', () => {
  const actual = gaussianNoisePatch(42n, -3, 5, 3, 4, 2, 8, 8);
  const expected = [
    -1.2839975357055664, -0.5033348798751831, 2.7349488735198975, -0.8726141452789307,
    -0.9949352145195007, -1.3967865705490112, -0.2543424069881439, 1.636263370513916,
    0.28738662600517273, 0.982470691204071, -0.7470621466636658, 0.46035727858543396,
    -2.518589735031128, 1.370664358139038, -1.1704463958740234, -0.20020613074302673,
  ];
  assert.deepEqual(Array.from(actual.slice(0, expected.length)), expected);
});

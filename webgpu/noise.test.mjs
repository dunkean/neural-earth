import test from 'node:test';
import assert from 'node:assert/strict';
import { gaussianNoisePatch, standardNormal, tileSeed } from './noise.mjs';

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

function bigIntReference(seed, count) {
  const mask = (1n << 64n)-1n;
  let state = BigInt(seed) & mask;
  const next = () => {
    state = (state*6364136223846793005n+1442695040888963407n) & mask;
    const x = (((state >> 18n)^state) >> 27n) & 0xffffffffn;
    const rot = state >> 59n;
    return Number(((x >> rot) | (x << ((32n-rot)&31n))) & 0xffffffffn);
  };
  const values = new Float32Array(count);
  let index = 0;
  while (index < count) {
    const a = 2*(next()+1)/4294967296-1, b = 2*(next()+1)/4294967296-1;
    const s = a*a+b*b;
    if (s <= 0 || s >= 1) continue;
    const f = Math.sqrt(-2*Math.log(s)/s);
    values[index++] = a*f;
    if (index < count) values[index++] = b*f;
  }
  return values;
}

test('word arithmetic preserves reference streams through carries and rotations', () => {
  for (const seed of [0n, 1n, -1n, 0xffffffffffffffffn, 0x8000000080000000n, 8105819n,
                     tileSeed(8105819n, -1, -1), tileSeed(8105819n, 4000, -9000)]) {
    assert.deepEqual(standardNormal(seed, 4097), bigIntReference(seed, 4097));
  }
});

test('overlapping full-size decoder windows share the same Gaussian samples', () => {
  const wide = gaussianNoisePatch(8105819n, -384, -384, 896, 896, 1, 512, 512);
  const part = gaussianNoisePatch(8105819n, 0, 0, 512, 512, 1, 512, 512);
  for (let y = 0; y < 512; y++) {
    assert.deepEqual(wide.slice((y+384)*896+384, (y+384)*896+896), part.slice(y*512, (y+1)*512));
  }
});

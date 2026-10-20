import test from 'node:test';
import assert from 'node:assert/strict';
import {presets,sketchPreset,heightSummary,reliefPixels} from './tile40_demo_helpers.mjs';

test('six physical presets preserve altitude scale and mountain/plain contrast',() => {
  const summaries=presets.map(name => heightSummary(sketchPreset(name).flat()));
  assert.equal(presets.length,6);
  assert.ok(summaries.every(s => s.minM >= 0 && s.maxM < 5000));
  assert.ok(summaries[2].reliefM > summaries[0].reliefM*50);
  assert.ok(summaries[1].reliefM > summaries[0].reliefM*10);
});

test('relief shading retains physical slopes,not per-image normalized contrast',() => {
  const flat=Float32Array.from({length:16},() => 100);
  const slope=Float32Array.from({length:16},(_,i) => 100+20*(i%4));
  const a=reliefPixels(flat,4,30),b=reliefPixels(slope,4,30);
  assert.equal(a.length,64);assert.notDeepEqual(a,b);
  assert.deepEqual(b,reliefPixels(slope.map(v => v+500),4,30));
  assert.throws(() => heightSummary([NaN]),/Nonfinite/);
});

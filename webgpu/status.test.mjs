import test from 'node:test';
import assert from 'node:assert/strict';
import { classifyCropStatus } from './crop.mjs';

test('numerical success is separate from WebGPU L1 proof', () => {
  assert.deepEqual(classifyCropStatus(true, true, false), {
    numericalStatus: 'PASS', l1Status: 'NOT_APPLICABLE',
  });
  assert.deepEqual(classifyCropStatus(true, false, false), {
    numericalStatus: 'PASS', l1Status: 'NOT_ESTABLISHED_NO_CAPTURE',
  });
  assert.deepEqual(classifyCropStatus(true, false, true), {
    numericalStatus: 'PASS', l1Status: 'NOT_ESTABLISHED',
  });
  assert.deepEqual(classifyCropStatus(false, false, true), {
    numericalStatus: 'FAIL', l1Status: 'FAIL',
  });
});

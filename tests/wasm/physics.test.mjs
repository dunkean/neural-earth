import {test} from 'node:test';
import assert from 'node:assert/strict';
import {physicalFields,orogenRequest,orogenSeed} from '../../wasm/physics.mjs';
test('Orogen physical units: sea level, land peaks, harmonic BIO4 and half-year BIO12/BIO15',()=>{
  const result={numRegions:4,r_elevation:[-1,0,.5,1],r_temperature_summer:[1,.5,.7,0],r_temperature_winter:[0,.5,.3,1],r_precip_summer:[1,0,2,.2],r_precip_winter:[1,0,1,.8]};
  const f=physicalFields(result);
  assert.deepEqual(Array.from(f.slice(0,4)),[-10000,0,1156.25,6000]);
  assert.equal(f[4],0);assert.equal(f[5],0);assert.equal(f[9],0);
  assert.ok(Math.abs(f[8]-90*100*Math.sqrt(12/11)/(2*Math.sqrt(2)))<.001);
  assert.equal(f[12],2000);assert.equal(f[13],0);assert.equal(f[14],3000);
  assert.equal(f[16],0);assert.equal(f[17],0);
  assert.ok(Math.abs(f[18]-1000*100*Math.sqrt(12/11)/3012)<.00001);
});
test('All u64 seed bits enter Orogen through the same SHA-256 mixing as Python',async()=>{
  const {createHash}=await import('node:crypto');
  const expected=createHash('sha256').update('orogen-tectonic-cuda-v1:18446744073709551615').digest().readUInt32LE(0)&0x7fffffff;
  assert.equal(await orogenSeed('18446744073709551615'),expected);
  assert.equal(await orogenSeed('42'),42);
  assert.equal(orogenRequest('42',{orogen_detail:20000,orogen_plate_count:12,relief_pipeline:'original'}).N,20000);
  assert.equal(orogenRequest('42',{relief_pipeline:'original'}).skipPost,true);
});

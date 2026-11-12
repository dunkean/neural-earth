import {test} from 'node:test';
import assert from 'node:assert/strict';
import {physicalFields,orogenRequest,orogenSeed} from '../../wasm/physics.mjs';
test('Orogen v3 physical km and monthly BIO fields are passed without remapping',()=>{
  const result={numRegions:4,r_elevation:[-10,0,.01,8.5],r_bio1:[0,1,2,3],r_bio4:[100,200,300,400],r_bio12:[0,1000,3000,8000],r_bio15:[0,10,50,150]};
  const f=physicalFields(result);
  assert.deepEqual(Array.from(f),[-10000,0,10,8500,0,1,2,3,100,200,300,400,0,1000,3000,8000,0,10,50,150]);
});
test('All u64 seed bits enter Orogen through the same SHA-256 mixing as Python',async()=>{
  const {createHash}=await import('node:crypto');
  const expected=createHash('sha256').update('orogen-tectonic-cuda-v1:18446744073709551615').digest().readUInt32LE(0)&0x7fffffff;
  assert.equal(await orogenSeed('18446744073709551615'),expected);
  assert.equal(await orogenSeed('42'),42);
  assert.equal(orogenRequest('42',{orogen_detail:20000,orogen_plate_count:12,relief_pipeline:'original'}).N,20000);
  assert.equal(orogenRequest('42',{relief_pipeline:'original'}).skipPost,true);
});

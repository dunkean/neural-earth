import {test} from 'node:test';
import assert from 'node:assert/strict';
import {probeWebGPU} from '../../wasm/webgpu.mjs';

test('WebGPU accepts a baseline adapter without optional neural extensions',async()=>{
  assert.deepEqual(await probeWebGPU({requestAdapter:async()=>({features:new Set()})},true),{shaderF16:false,subgroups:false});
});
test('WebGPU distinguishes a missing API, insecure hosting and an inaccessible GPU',async()=>{
  await assert.rejects(probeWebGPU(undefined,false),/HTTPS or localhost/);
  await assert.rejects(probeWebGPU(undefined,true),/unavailable in this browser/);
  await assert.rejects(probeWebGPU({requestAdapter:async()=>null},true),/no GPU adapter is accessible/);
});
test('WebGPU retries default adapter selection when high performance is unavailable',async()=>{
  const preferences=[];
  const features=await probeWebGPU({requestAdapter:async options=>{preferences.push(options);return options?null:{features:new Set(['shader-f16'])};}},true);
  assert.deepEqual(preferences,[{powerPreference:'high-performance'},undefined]);
  assert.deepEqual(features,{shaderF16:true,subgroups:false});
});

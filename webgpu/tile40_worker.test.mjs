import test from 'node:test';
import assert from 'node:assert/strict';
import vm from 'node:vm';
import {readFileSync} from 'node:fs';

function worker({adapterError=false,gpuError=false}={}) {
  const messages=[],providers=[],requests=[],released=[];
  let rejectDecoder=null;
  const manifest={models:{coarse:{file:'coarse.onnx'},encoder:{file:'encoder.onnx'},decoder:{file:'decoder.onnx'}}};
  const context=vm.createContext({URL,performance,Float32Array,console,
    self:{location:'http://example.test/tile40/worker.js',crossOriginIsolated:true},
    navigator:{hardwareConcurrency:8,gpu:{requestAdapter:async()=>{if(adapterError)throw Error('adapter unavailable');return {info:{vendor:'nvidia'}};}}},
    importScripts:()=>{},postMessage:value=>messages.push(value),
    fetch:async url=>{requests.push(url);return {ok:!rejectDecoder||!url.includes(rejectDecoder),json:async()=>manifest};},
    generateFake:async()=>({height:new Float32Array([100,200]),timings:{totalMs:1}}),
    ort:{env:{wasm:{}},InferenceSession:{create:async(url,options)=>{
      providers.push(options.executionProviders[0]);
      if(gpuError&&options.executionProviders[0]==='webgpu')throw Error('GPU initialization failed');
      return {release:async()=>released.push(url)};
    }}},
  });
  const original=readFileSync(new URL('./tile40_demo_worker.js',import.meta.url),'utf8');
  const source=original.replace("const runtime=import('./runtime/tile40_runtime.mjs');",'const runtime=Promise.resolve({generateTile:generateFake});');
  assert.notEqual(source,original);
  vm.runInContext(source,context);
  return {messages,providers,requests,released,failDecoder:value=>rejectDecoder=value,
    run:(provider='auto',decoder='decoder16')=>context.onmessage({data:{provider,decoder,input:{}}})};
}

test('automatic mode falls back to CPU if adapter probing rejects',async()=>{
  const w=worker({adapterError:true});await w.run();
  assert.equal(w.messages[0].provider,'wasm');assert.match(w.messages[0].fallback,/adapter unavailable/);
  assert.deepEqual(w.providers,['wasm','wasm','wasm']);
});

test('automatic GPU initialization failure falls back, while forced GPU reports the failure',async()=>{
  const automatic=worker({gpuError:true});await automatic.run();
  assert.equal(automatic.messages[0].provider,'wasm');assert.match(automatic.messages[0].fallback,/GPU initialization failed/);
  const forced=worker({gpuError:true});await forced.run('webgpu');
  assert.match(forced.messages[0].error,/GPU initialization failed/);assert.deepEqual(forced.providers,['webgpu']);
});

test('decoder changes release old sessions and reload the selected scoped model paths',async()=>{
  const w=worker();await w.run('wasm','decoder16');await w.run('wasm','decoder64');
  assert.equal(w.released.length,3);
  assert(w.requests[0].endsWith('/tile40/model/decoder16/manifest.json'));
  assert(w.requests[1].endsWith('/tile40/model/decoder64/manifest.json'));
  assert.equal(w.messages[1].provider,'wasm');
});

test('a failed decoder switch can recover to the previous decoder without stale manifest state',async()=>{
  const w=worker();await w.run('wasm','decoder16');w.failDecoder('decoder64');await w.run('wasm','decoder64');
  assert.match(w.messages[1].error,/Modèles indisponibles/);
  w.failDecoder(null);await w.run('wasm','decoder16');
  assert.equal(w.messages[2].provider,'wasm');assert.equal(w.requests.length,3);
});

// Optional WebGPU extensions improve precision/memory usage; they are not a
// prerequisite. Rust chooses its precision from the device it actually opens.
export async function probeWebGPU(gpu=globalThis.navigator?.gpu,secure=globalThis.isSecureContext) {
  if(secure===false)throw Error('WebGPU requires HTTPS or localhost. Open the static site through a secure URL.');
  if(!gpu)throw Error('WebGPU is unavailable in this browser. Enable browser graphics acceleration and use a browser with WebGPU support.');
  let adapter=await gpu.requestAdapter({powerPreference:'high-performance'});
  if(!adapter)adapter=await gpu.requestAdapter();
  if(!adapter)throw Error('WebGPU is enabled but no GPU adapter is accessible. Check browser graphics acceleration and the GPU driver.');
  return {shaderF16:adapter.features.has('shader-f16'),subgroups:adapter.features.has('subgroups')};
}

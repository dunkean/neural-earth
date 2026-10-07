// ORT WebGPU graph capture requires stable external GPU input/output buffers.
// Upload/readback is deliberate for this correctness probe; production should
// run solver and fusion on the same device and avoid these per-forward copies.
export function makeGpuRunner(ort, device, session, shapes, outputShape) {
  const inputs = {};
  const gpuBuffers = [];
  const shapeMap = {};
  for (const name of session.inputNames) {
    const shape = shapes[name];
    if (!shape) throw new Error(`Missing input shape ${name}`);
    const size = shape.reduce((a, b) => a * b, 1) * 4;
    const buffer = device.createBuffer({
      size: Math.ceil(size / 16) * 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
    });
    gpuBuffers.push(buffer);
    inputs[name] = ort.Tensor.fromGpuBuffer(buffer, { dataType: 'float32', dims: shape });
    shapeMap[name] = { shape, buffer, size };
  }
  const outputSize = outputShape.reduce((a, b) => a * b, 1) * 4;
  const outputBuffer = device.createBuffer({
    size: Math.ceil(outputSize / 16) * 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | GPUBufferUsage.STORAGE,
  });
  const readback = device.createBuffer({
    size: Math.ceil(outputSize / 16) * 16,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });
  const outputTensor = ort.Tensor.fromGpuBuffer(outputBuffer, { dataType: 'float32', dims: outputShape });
  return {
    async run(arrays) {
      for (const name of session.inputNames) {
        const spec = shapeMap[name];
        const data = arrays[name];
        if (!(data instanceof Float32Array) || data.byteLength !== spec.size) throw new Error(`Input ${name} length/type mismatch`);
        device.queue.writeBuffer(spec.buffer, 0, data);
      }
      await session.run(inputs, { [session.outputNames[0]]: outputTensor });
      const encoder = device.createCommandEncoder();
      encoder.copyBufferToBuffer(outputBuffer, 0, readback, 0, outputSize);
      device.queue.submit([encoder.finish()]);
      await readback.mapAsync(GPUMapMode.READ);
      const result = new Float32Array(readback.getMappedRange().slice(0, outputSize));
      readback.unmap();
      return result;
    },
    dispose() {
      for (const buffer of gpuBuffers) buffer.destroy();
      outputBuffer.destroy();
      readback.destroy();
    },
  };
}

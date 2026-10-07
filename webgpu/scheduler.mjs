// EDM DPM-Solver++ midpoint order 2 as configured by WorldPipeline.
// Outputs are FP32 arrays in channel-first order. The final sigma is zero,
// hence the last step uses the first-order formula, matching upstream.
const f32 = Math.fround;
const SIGMA_DATA = 0.5;

export function karrasSigmas(count = 20) {
  const out = new Float32Array(count + 1);
  const a = 80 ** (1 / 7);
  const b = 0.002 ** (1 / 7);
  for (let i = 0; i < count; i++) {
    const ramp = i / (count - 1);
    out[i] = f32((a + ramp * (b - a)) ** 7);
  }
  out[count] = 0;
  return out;
}

export function preconditionInput(sample, sigma, sigmaData = SIGMA_DATA) {
  const scale = 1 / Math.sqrt(sigma * sigma + sigmaData * sigmaData);
  return Float32Array.from(sample, value => f32(value * scale));
}

export function noiseLabel(sigma, sigmaData = SIGMA_DATA) {
  return f32(Math.atan(sigma / sigmaData));
}

export function preconditionOutput(sample, networkOutput, sigma, sigmaData = SIGMA_DATA) {
  if (sample.length !== networkOutput.length) throw new Error('EDM shape mismatch');
  const denominator = sigma * sigma + sigmaData * sigmaData;
  const skip = sigmaData * sigmaData / denominator;
  const outScale = sigma * sigmaData / Math.sqrt(denominator);
  const out = new Float32Array(sample.length);
  for (let i = 0; i < out.length; i++) out[i] = f32(f32(sample[i] * skip) + f32(networkOutput[i] * outScale));
  return out;
}

export function createSolver(count = 20) {
  const sigmas = karrasSigmas(count);
  let stepIndex = 0;
  let previousDenoised = null;
  const lambda = sigma => sigma === 0 ? Infinity : -Math.log(sigma);
  return {
    sigmas,
    get stepIndex() { return stepIndex; },
    step(sample, networkOutput) {
      if (stepIndex >= count) throw new Error('Solver exhausted');
      const sigmaS = sigmas[stepIndex];
      const sigmaT = sigmas[stepIndex + 1];
      const denoised = preconditionOutput(sample, networkOutput, sigmaS);
      const next = new Float32Array(sample.length);
      const ratio = sigmaT / sigmaS;
      // alpha=1 for EDM sigma parameterization. exp(-h)=sigmaT/sigmaS.
      const b = 1 - ratio;
      const firstOrder = stepIndex === 0 || sigmaT === 0;
      let correction = 0;
      if (!firstOrder) {
        const h = lambda(sigmaT) - lambda(sigmaS);
        const h0 = lambda(sigmaS) - lambda(sigmas[stepIndex - 1]);
        correction = 0.5 * b * h / h0;
      }
      for (let i = 0; i < next.length; i++) {
        let value = f32(f32(ratio * sample[i]) + f32(b * denoised[i]));
        if (!firstOrder) value = f32(value + f32(correction * f32(denoised[i] - previousDenoised[i])));
        next[i] = value;
      }
      previousDenoised = denoised;
      stepIndex++;
      return next;
    },
  };
}

// CPU reference implementation of torchvision's bilinear resize and Gaussian
// blur used by WorldPipeline._compute_elev. Kept explicit for crop validation.
const f32 = Math.fround;

function clamp(index, length) { return Math.max(0, Math.min(length - 1, index)); }

function resizeAxis(input, srcH, srcW, dstLength, axis, antialias) {
  const srcLength = axis === 0 ? srcH : srcW;
  const dstH = axis === 0 ? dstLength : srcH;
  const dstW = axis === 0 ? srcW : dstLength;
  const output = new Float32Array(dstH * dstW);
  const scale = srcLength / dstLength;
  const support = antialias && scale > 1 ? scale : 1;
  for (let dst = 0; dst < dstLength; dst++) {
    const center = (dst + 0.5) * scale;
    const start = Math.max(0, Math.floor(center - support - 0.5));
    const end = Math.min(srcLength, Math.ceil(center + support - 0.5));
    const weights = [];
    let sum = 0;
    for (let src = start; src < end; src++) {
      const weight = Math.max(0, 1 - Math.abs((src + 0.5 - center) / support));
      weights.push(weight); sum += weight;
    }
    for (let other = 0; other < (axis === 0 ? srcW : srcH); other++) {
      let value = 0;
      for (let src = start; src < end; src++) {
        const index = axis === 0 ? src * srcW + other : other * srcW + src;
        value += input[index] * weights[src - start] / sum;
      }
      output[axis === 0 ? dst * dstW + other : other * dstW + dst] = f32(value);
    }
  }
  return [output, dstH, dstW];
}

export function resizeBilinear(input, srcH, srcW, dstH, dstW, antialias = true) {
  if (input.length !== srcH * srcW) throw new Error('Resize shape mismatch');
  const [vertical] = resizeAxis(input, srcH, srcW, dstH, 0, antialias);
  const [horizontal] = resizeAxis(vertical, dstH, srcW, dstW, 1, antialias);
  return horizontal;
}

function extrapolatePad(input, height, width) {
  const outH = height + 2, outW = width + 2;
  const out = new Float32Array(outH * outW);
  for (let y = 0; y < outH; y++) {
    const sy = clamp(y - 1, height);
    const sy2 = y === 0 ? Math.min(1, height - 1) : y === outH - 1 ? Math.max(0, height - 2) : sy;
    const yFactor = (y === 0 || y === outH - 1) && height > 1;
    for (let x = 0; x < outW; x++) {
      const sx = clamp(x - 1, width);
      const sx2 = x === 0 ? Math.min(1, width - 1) : x === outW - 1 ? Math.max(0, width - 2) : sx;
      const xFactor = (x === 0 || x === outW - 1) && width > 1;
      const base = input[sy * width + sx];
      const across = xFactor ? 2 * base - input[sy * width + sx2] : base;
      const row2 = input[sy2 * width + sx];
      const across2 = xFactor ? 2 * row2 - input[sy2 * width + sx2] : row2;
      out[y * outW + x] = f32(yFactor ? 2 * across - across2 : across);
    }
  }
  return out;
}

export function resizeExtrapolated(input, srcH, srcW, dstH, dstW) {
  const scaleH = dstH / srcH, scaleW = dstW / srcW;
  const padH = Math.round(scaleH), padW = Math.round(scaleW);
  const fullH = Math.round(dstH + 2 * scaleH), fullW = Math.round(dstW + 2 * scaleW);
  const full = resizeBilinear(extrapolatePad(input, srcH, srcW), srcH + 2, srcW + 2, fullH, fullW);
  const out = new Float32Array(dstH * dstW);
  for (let y = 0; y < dstH; y++) out.set(full.subarray((y + padH) * fullW + padW, (y + padH) * fullW + padW + dstW), y * dstW);
  return out;
}

function reflect(index, length) {
  if (length === 1) return 0;
  while (index < 0 || index >= length) index = index < 0 ? -index : 2 * length - index - 2;
  return index;
}

export function gaussianBlur(input, height, width, sigma = 5) {
  const radius = Math.floor((Math.floor(sigma * 2) | 1) / 2);
  const kernel = Float64Array.from({ length: radius * 2 + 1 }, (_, i) => Math.exp(-0.5 * ((i - radius) / sigma) ** 2));
  const total = kernel.reduce((a, b) => a + b, 0);
  for (let i = 0; i < kernel.length; i++) kernel[i] /= total;
  const tmp = new Float32Array(input.length), out = new Float32Array(input.length);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    let sum = 0;
    for (let k = -radius; k <= radius; k++) sum += input[y * width + reflect(x + k, width)] * kernel[k + radius];
    tmp[y * width + x] = f32(sum);
  }
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    let sum = 0;
    for (let k = -radius; k <= radius; k++) sum += tmp[reflect(y + k, height) * width + x] * kernel[k + radius];
    out[y * width + x] = f32(sum);
  }
  return out;
}

export function laplacianReconstruct(residual, lowres, highH, highW, lowH, lowW, sigma = 5) {
  if (residual.length !== highH * highW || lowres.length !== lowH * lowW) throw new Error('Laplacian shape mismatch');
  const extrapolated = resizeExtrapolated(lowres, lowH, lowW, highH, highW);
  const decoded = Float32Array.from(residual, (value, i) => f32(value + extrapolated[i]));
  const refreshed = gaussianBlur(resizeBilinear(decoded, highH, highW, lowH, lowW), lowH, lowW, sigma);
  const up = resizeBilinear(refreshed, lowH, lowW, highH, highW);
  return Float32Array.from(residual, (value, i) => {
    const sqrtHeight = f32(value + up[i]);
    return f32(Math.sign(sqrtHeight) * sqrtHeight * sqrtHeight);
  });
}

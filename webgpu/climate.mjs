// Port of WorldPipeline._compute_climate for a supplied, fully fused coarse
// context. Coordinates and scale are native 30 m pixel indices.
const f32 = Math.fround;

function bilinear(field, height, width, y, x) {
  y = Math.max(0, Math.min(height - 1, y));
  x = Math.max(0, Math.min(width - 1, x));
  const y0 = Math.floor(y), y1 = Math.min(height - 1, y0 + 1);
  const x0 = Math.floor(x), x1 = Math.min(width - 1, x0 + 1);
  const a = y - y0, b = x - x0;
  return f32((1 - a) * ((1 - b) * field[y0 * width + x0] + b * field[y0 * width + x1]) + a * ((1 - b) * field[y1 * width + x0] + b * field[y1 * width + x1]));
}

export function climateFromCoarse(coarse, coarseH, coarseW, elevation, i1, j1, height, width, scale = 8) {
  if (coarse.length !== 6 * coarseH * coarseW || elevation.length !== height * width) throw new Error('Climate shape mismatch');
  const win = 15, pad = 7;
  const sourceH = coarseH - 2 * pad, sourceW = coarseW - 2 * pad;
  if (sourceH < 1 || sourceW < 1) throw new Error('Coarse climate context too small');
  const plane = coarseH * coarseW;
  const field = (channel, y, x) => coarse[channel * plane + y * coarseW + x];
  const baseline = new Float32Array(sourceH * sourceW);
  const betas = new Float32Array(sourceH * sourceW);
  const features = Array.from({ length: 4 }, () => new Float32Array(sourceH * sourceW));
  for (let cy = 0; cy < sourceH; cy++) for (let cx = 0; cx < sourceW; cx++) {
    let n = 0, sumT = 0, sumE = 0, sumE2 = 0, sumET = 0;
    for (let wy = 0; wy < win; wy++) for (let wx = 0; wx < win; wx++) {
      const raw = field(0, cy + wy, cx + wx);
      const e = Math.max(0, raw) ** 2;
      if (e > 0) {
        const T = field(2, cy + wy, cx + wx);
        n++; sumT += T; sumE += e; sumE2 += e * e; sumET += e * T;
      }
    }
    const coverage = n / (win * win);
    const denom = coverage + 1e-6;
    const avg = value => f32(f32(value / (win * win)) / denom);
    const muT = avg(sumT), muE = avg(sumE), muE2 = avg(sumE2), muET = avg(sumET);
    const variance = f32(muE2 - f32(muE * muE));
    const covariance = f32(muET - f32(muE * muT));
    let beta = variance < 1 || coverage < 0.02 ? -0.0065 : covariance / (variance + 1e-6);
    beta = f32(Math.max(-0.012, Math.min(0, beta)));
    const y = cy + pad, x = cx + pad, dst = cy * sourceW + cx;
    const raw = field(0, y, x);
    const e = Math.max(0, raw) ** 2;
    baseline[dst] = f32(field(2, y, x) - f32(beta * e));
    betas[dst] = beta;
    for (let c = 0; c < 4; c++) features[c][dst] = field(c + 2, y, x);
  }
  const ci1 = Math.floor(i1 / (32 * scale)), cj1 = Math.floor(j1 / (32 * scale));
  const S = 32 * scale;
  const outPlane = height * width;
  const output = new Float32Array(5 * outPlane);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) {
    const u = (i1 + y + 0.5) / S - ci1 + 0.5;
    const v = (j1 + x + 0.5) / S - cj1 + 0.5;
    const p = y * width + x;
    const beta = bilinear(betas, sourceH, sourceW, u, v);
    output[p] = f32(bilinear(baseline, sourceH, sourceW, u, v) + f32(beta * Math.max(0, elevation[p])));
    for (let c = 1; c <= 3; c++) output[c * outPlane + p] = bilinear(features[c], sourceH, sourceW, u, v);
    output[4 * outPlane + p] = beta;
  }
  return output;
}

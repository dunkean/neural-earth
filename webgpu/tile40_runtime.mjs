// Full40km inference from a physical sketch,climate and seed. Hidden fields are
// produced by the three small models;there are no teacher fixtures in this API.
import {gaussianNoisePatch} from './noise.mjs';

const f32 = Math.fround;

export function finiteDecoderInputs(latents, seed, channels = 5) {
  if (latents.length !== 5*320*320 || ![5, 16].includes(channels)) throw new Error('Invalid finite decoder interface.');
  const size = 1856, n = size*size, inputs = new Float32Array(channels*n);
  inputs.set(gaussianNoisePatch(BigInt(seed)+5819n, -256, -256, size, size, 1, 512, 512));
  for (let c = 0; c < 4; c++) for (let y = 0; y < size; y++) {
    const row = c*320*320+(32+Math.floor(y/8))*320+32;
    for (let x = 0; x < size; x++) inputs[(c+1)*n+y*size+x] = latents[row+Math.floor(x/8)];
  }
  return inputs;
}
const bits = new Uint32Array(1), floats = new Float32Array(bits.buffer);
export function bf16(value) {
  floats[0] = value;
  bits[0] = (bits[0]+0x7fff+((bits[0] >>> 16)&1)) & 0xffff0000;
  return floats[0];
}

function derivatives(values) {
  const n = values.length, slopes = new Float64Array(n-1), d = new Float64Array(n);
  for (let i = 0; i < n-1; i++) slopes[i] = values[i+1]-values[i];
  if (n === 2) {d.fill(slopes[0]); return d;}
  for (let i = 1; i < n-1; i++) {
    const a = slopes[i-1], b = slopes[i];
    d[i] = a && b && Math.sign(a) === Math.sign(b) ? 2/(1/a+1/b) : 0;
  }
  const edge = (a, b) => {
    let v = (3*a-b)/2;
    if (Math.sign(v) !== Math.sign(a)) v = 0;
    else if (Math.sign(a) !== Math.sign(b) && Math.abs(v) > Math.abs(3*a)) v = 3*a;
    return v;
  };
  d[0] = edge(slopes[0], slopes[1]);
  d[n-1] = edge(slopes[n-2], slopes[n-3]);
  return d;
}

function pchip(values, d, position) {
  const p = Math.max(0, Math.min(values.length-1, position));
  const i = Math.min(values.length-2, Math.floor(p)), t = p-i;
  const a = values[i], b = values[i+1], delta = b-a;
  return a+t*(d[i]+t*(3*delta-2*d[i]-d[i+1]+t*(d[i]+d[i+1]-2*delta)));
}

export function sampleSketch(sketch, xs, ys, sideM = 40000) {
  const n = sketch.length;
  if (n < 2 || sketch.some(row => row.length !== n || row.some(v => !Number.isFinite(v)))) throw new Error('Finite square physical sketch required.');
  const width = xs.length, height = ys.length, across = new Float64Array(n*width);
  for (let y = 0; y < n; y++) {
    const d = derivatives(sketch[y]);
    for (let x = 0; x < width; x++) across[y*width+x] = pchip(sketch[y], d, xs[x]/sideM*(n-1));
  }
  const out = new Float32Array(width*height), column = new Float64Array(n);
  for (let x = 0; x < width; x++) {
    for (let y = 0; y < n; y++) column[y] = across[y*width+x];
    const d = derivatives(column);
    for (let y = 0; y < height; y++) out[y*width+x] = pchip(column, d, ys[y]/sideM*(n-1));
  }
  return out;
}

function coordinates(origin, size, spacing) {
  return Array.from({length:size}, (_, i) => (origin+i+.5)*spacing);
}

function weight(size) {
  const axis = new Float32Array(size), middle = (size-1)/2;
  for (let i = 0; i < size; i++) axis[i] = f32(1-f32(f32(1-.001)*f32(Math.abs(i-middle)/middle)));
  return axis;
}

function blend(canvas, sums, prediction, axis, origin, size, windowY, windowX, channels) {
  const tile = axis.length;
  const y0 = Math.max(origin, windowY), x0 = Math.max(origin, windowX);
  const y1 = Math.min(origin+size, windowY+tile), x1 = Math.min(origin+size, windowX+tile);
  for (let y = y0; y < y1; y++) {
    for (let x = x0; x < x1; x++) {
      const src = (y-windowY)*tile+x-windowX, dst = (y-origin)*size+x-origin;
      const w = f32(axis[y-windowY]*axis[x-windowX]);
      sums[dst] += w;
      for (let c = 0; c < channels; c++) canvas[c*size*size+dst] += f32(prediction[c*tile*tile+src]*w);
    }
  }
}

function divide(canvas, sums, channels) {
  for (let i = 0; i < sums.length; i++) {
    if (!(sums[i] > 0)) throw new Error('Incomplete neural-window coverage.');
    for (let c = 0; c < channels; c++) canvas[c*sums.length+i] /= sums[i];
  }
  return canvas;
}

export function resizeBilinear(values, sourceSize, outputSize, extrapolate = false) {
  const out = new Float32Array(outputSize*outputSize);
  const read = (y, x) => {
    const row = yy => {
      if (x < 0) return f32(f32(2*values[yy*sourceSize])-values[yy*sourceSize+1]);
      if (x >= sourceSize) return f32(f32(2*values[yy*sourceSize+sourceSize-1])-values[yy*sourceSize+sourceSize-2]);
      return values[yy*sourceSize+x];
    };
    if (y < 0) return f32(f32(2*row(0))-row(1));
    if (y >= sourceSize) return f32(f32(2*row(sourceSize-1))-row(sourceSize-2));
    return row(y);
  };
  for (let y = 0; y < outputSize; y++) {
    const yy = (y+.5)*sourceSize/outputSize-.5, y0 = Math.floor(yy), fy = yy-y0;
    for (let x = 0; x < outputSize; x++) {
      const xx = (x+.5)*sourceSize/outputSize-.5, x0 = Math.floor(xx), fx = xx-x0;
      const at = (dy, dx) => extrapolate ? read(y0+dy, x0+dx) :
        values[Math.max(0, Math.min(sourceSize-1, y0+dy))*sourceSize+Math.max(0, Math.min(sourceSize-1, x0+dx))];
      out[y*outputSize+x] = (1-fy)*((1-fx)*at(0, 0)+fx*at(0, 1))+fy*((1-fx)*at(1, 0)+fx*at(1, 1));
    }
  }
  return out;
}

export function resizeDown8(values, size) {
  const down = size/8;
  if (!Number.isInteger(down)) throw new Error('Antialiased grid must align to8.');
  const horizontal = new Float32Array(size*down), out = new Float32Array(down*down);
  const taps = Array.from({length:down}, (_, i) => {
    const center = (i+.5)*8-.5, items = []; let total = 0;
    for (let x = Math.ceil(center-8); x < center+8; x++) {
      if (x < 0 || x >= size) continue;
      const w = 1-Math.abs(x-center)/8; items.push([x, w]); total += w;
    }
    return items.map(([x, w]) => [x, w/total]);
  });
  for (let y = 0; y < size; y++) for (let x = 0; x < down; x++) {
    let sum = 0; for (const [sx, w] of taps[x]) sum += values[y*size+sx]*w;
    horizontal[y*down+x] = sum;
  }
  for (let y = 0; y < down; y++) for (let x = 0; x < down; x++) {
    let sum = 0; for (const [sy, w] of taps[y]) sum += horizontal[sy*down+x]*w;
    out[y*down+x] = sum;
  }
  return out;
}

export function gaussian(values, size, kernel, halfSample = false) {
  const radius = (kernel.length-1)/2;
  const at = position => position < 0 ? -position-(halfSample ? 1 : 0) :
    position >= size ? 2*size-position-(halfSample ? 1 : 2) : position;
  const temporary = new Float32Array(values.length), out = new Float32Array(values.length);
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    let sum = 0; for (let k = 0; k < kernel.length; k++) sum += values[y*size+at(x+k-radius)]*kernel[k];
    temporary[y*size+x] = sum;
  }
  for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
    let sum = 0; for (let k = 0; k < kernel.length; k++) sum += temporary[at(y+k-radius)*size+x]*kernel[k];
    out[y*size+x] = sum;
  }
  return out;
}

export function reconstructHeight(residual, latents, sketch, kernels) {
  const size = 1856, lowSize = 232, low = new Float32Array(lowSize*lowSize);
  for (let y = 0; y < lowSize; y++) for (let x = 0; x < lowSize; x++) {
    low[y*lowSize+x] = f32(f32(latents[4*320*320+(y+32)*320+x+32]*f32(38.6))-f32(31.4));
  }
  const up = resizeBilinear(low, lowSize, size, true), decoded = new Float32Array(size*size);
  for (let i = 0; i < decoded.length; i++) decoded[i] = f32(f32(residual[i]*f32(.7))+up[i]);
  const revised = gaussian(resizeDown8(decoded, size), lowSize, kernels.laplacian, false);
  const revisedUp = resizeBilinear(revised, lowSize, size);
  const root = new Float32Array(decoded.length), reduced = new Float32Array(low.length);
  for (let i = 0; i < root.length; i++) {
    const r = f32(f32(residual[i]*f32(.7))+revisedUp[i]);
    const h = f32(Math.sign(r)*f32(r*r));
    root[i] = f32(Math.sign(h)*Math.sqrt(Math.abs(h)));
  }
  for (let y = 0; y < lowSize; y++) for (let x = 0; x < lowSize; x++) {
    let sum = 0;
    for (let dy = 0; dy < 8; dy++) for (let dx = 0; dx < 8; dx++) sum += root[(y*8+dy)*size+x*8+dx];
    reduced[y*lowSize+x] = sum/64;
  }
  const smooth = resizeBilinear(gaussian(reduced, lowSize, kernels.guidance, true), lowSize, size);
  const nativeSize = 1344, axis = coordinates(0, nativeSize, 30), background = sampleSketch(sketch, axis, axis);
  const native = new Float32Array(nativeSize*nativeSize);
  for (let y = 0; y < nativeSize; y++) for (let x = 0; x < nativeSize; x++) {
    const i = y*nativeSize+x, p = (y+256)*size+x+256;
    const r = f32(f32(Math.sign(background[i])*Math.sqrt(Math.abs(background[i])))+f32(root[p]-smooth[p]));
    native[i] = Math.sign(r)*f32(r*r);
  }
  const outputSize = 1334, out = new Float32Array(outputSize*outputSize);
  for (let y = 0; y < outputSize; y++) {
    const yy = Math.max(0, (y+.5)*40000/outputSize/30-.5), y0 = Math.floor(yy), fy = yy-y0;
    for (let x = 0; x < outputSize; x++) {
      const xx = Math.max(0, (x+.5)*40000/outputSize/30-.5), x0 = Math.floor(xx), fx = xx-x0;
      const at = (dy, dx) => native[(y0+dy)*nativeSize+x0+dx];
      out[y*outputSize+x] = (1-fy)*((1-fx)*at(0, 0)+fx*at(0, 1))+fy*((1-fx)*at(1, 0)+fx*at(1, 1));
    }
  }
  return out;
}

export async function generateTile(ort, sessions, manifest, input) {
  const params = manifest.parameters, climate = manifest.climates[input.climate], seed = BigInt(input.seed);
  if (!climate) throw new Error('Unknown climate.');
  const timings = {}, started = performance.now();
  const infer = async (stage, data, shape) => {
    const tensor = new ort.Tensor('float32', data, shape);
    const result = await sessions[stage].run({inputs:tensor});
    const name = manifest.models[stage].output_name;
    const values = new Float32Array(await result[name].getData());
    result[name].dispose(); tensor.dispose(); return values;
  };
  const coarseOrigin=manifest.coarse_origin ?? -8, coarseSize=manifest.coarse_size ?? 24;
  const coarse = new Float32Array(6*coarseSize*coarseSize), coarseSums = new Float32Array(coarseSize*coarseSize), coarseWeight = weight(64);
  for (const y of [-48, 0]) for (const x of [-48, 0]) {
    const elevation = sampleSketch(input.sketch_m, coordinates(x, 64, 7680), coordinates(y, 64, 7680));
    const n = 64*64, condNoise = gaussianNoisePatch(seed, y, x, 64, 64, 5, 64, 64);
    const features = new Float32Array(16*n);
    features.set(gaussianNoisePatch(seed+1n, y, x, 64, 64, 6, 64, 64));
    for (let c = 0; c < 5; c++) for (let i = 0; i < n; i++) {
      const z = elevation[i];
      const raw = c === 0 ? f32(Math.sign(z)*Math.sqrt(Math.abs(z))) : c === 1 ?
        f32(f32(climate[0])-f32(f32(.0065)*Math.max(z, 0))) : climate[c-1];
      const channel = [0, 2, 3, 4, 5][c];
      const synthetic = bf16(f32(f32(raw-params.coarse_means[channel])/params.coarse_stds[channel]));
      features[(6+c)*n+i] = bf16(bf16(params.cos_theta[c]*synthetic)+bf16(params.sin_theta[c]*bf16(condNoise[c*n+i])));
      features[(11+c)*n+i] = params.labels[c];
    }
    const prediction = await infer('coarse', features, [1, 16, 64, 64]);
    for (let c = 0; c < 6; c++) for (let i = 0; i < n; i++) {
      prediction[c*n+i] = f32(f32(prediction[c*n+i]*params.physical_stds[c])+params.physical_means[c]);
    }
    blend(coarse, coarseSums, prediction, coarseWeight, coarseOrigin, coarseSize, y, x, 6);
  }
  divide(coarse, coarseSums, 6);
  const coarseFinished = performance.now(); timings.coarseMs = coarseFinished-started;
  const halo = manifest.encoder_halo, encoderSize = 320+2*halo, encoderOrigin = -64-halo, nn = encoderSize*encoderSize;
  const features = new Float32Array(32*nn);
  features.set(gaussianNoisePatch(seed+5819n, encoderOrigin, encoderOrigin, encoderSize, encoderSize, 5, 64, 64));
  features.set(gaussianNoisePatch(seed+5820n, encoderOrigin, encoderOrigin, encoderSize, encoderSize, 5, 64, 64), 5*nn);
  for (let y = 0; y < encoderSize; y++) for (let x = 0; x < encoderSize; x++) {
    const i = y*encoderSize+x, cy = Math.floor((encoderOrigin+y)/32)-coarseOrigin, cx = Math.floor((encoderOrigin+x)/32)-coarseOrigin;
    if (cy < 0 || cx < 0 || cy >= coarseSize || cx >= coarseSize) throw new Error('Encoder exceeds coarse conditioning halo.');
    for (let c = 0; c < 6; c++) features[(10+c)*nn+i] = f32(f32(coarse[c*coarseSize*coarseSize+cy*coarseSize+cx]-manifest.condition_means[c])/manifest.condition_stds[c]);
    features[16*nn+i] = (1-.66)/.47;
    const ay = f32((encoderOrigin+y)*f32(2*Math.PI/64)), ax = f32((encoderOrigin+x)*f32(2*Math.PI/64));
    features[17*nn+i] = Math.sin(ay); features[18*nn+i] = Math.cos(ay);
    features[19*nn+i] = Math.sin(ax); features[20*nn+i] = Math.cos(ax);
    for (let c = 0; c < 5; c++) features[(21+c)*nn+i] = params.histogram_raw[c];
  }
  const latents = await infer('encoder', features, [1, 32, encoderSize, encoderSize]);
  const encoderFinished = performance.now(); timings.encoderMs = encoderFinished-coarseFinished;
  let canvas;
  const decoderChannels = manifest.models.decoder.input_channels || 16;
  if (manifest.models.decoder.layout === 'finite-root') {
    const inputs = finiteDecoderInputs(latents, seed, decoderChannels);
    canvas = await infer('decoder', inputs, [1, decoderChannels, 1856, 1856]);
  } else {
    canvas = new Float32Array(1856*1856);
    const sums = new Float32Array(canvas.length), decoderWeight = weight(512);
    const noiseOrigin = -384, noiseSize = 2432;
    const decoderNoise = gaussianNoisePatch(seed+5819n, noiseOrigin, noiseOrigin, noiseSize, noiseSize, 1, 512, 512);
    const decoderInputs = new Float32Array(decoderChannels*512*512), dn = 512*512;
    for (const y of [-384, 0, 384, 768, 1152, 1536]) for (const x of [-384, 0, 384, 768, 1152, 1536]) {
      for (let iy = 0; iy < 512; iy++) {
        const ni = (y-noiseOrigin+iy)*noiseSize+x-noiseOrigin;
        decoderInputs.set(decoderNoise.subarray(ni, ni+512), iy*512);
        for (let ix = 0; ix < 512; ix++) for (let c = 0; c < 4; c++) {
          decoderInputs[(c+1)*dn+iy*512+ix] = latents[c*320*320+(y/8+64+Math.floor(iy/8))*320+x/8+64+Math.floor(ix/8)];
        }
      }
      const prediction = await infer('decoder', decoderInputs, [1, decoderChannels, 512, 512]);
      blend(canvas, sums, prediction, decoderWeight, -256, 1856, y, x, 1);
    }
    divide(canvas, sums, 1);
  }
  const decoderFinished = performance.now(); timings.decoderMs = decoderFinished-encoderFinished;
  const height = reconstructHeight(canvas, latents, input.sketch_m, manifest.kernels);
  const finished = performance.now(); timings.reconstructionMs = finished-decoderFinished;
  timings.totalMs = finished-started;
  return {height, timings, sideM:40000, pixels:1334, spacingM:40000/1334};
}

// Integration overlay over Magna Urbis's pinned pure Rust implementation.
// Fail closed when upstream changes: every replacement has to match exactly.
import {readFile, writeFile, mkdir} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {twoStepBase} from './base-solver.mjs';
const root = fileURLToPath(new URL('../../', import.meta.url));
const source = `${root}native/magna-urbis/rust/crates/neural/src/`;
const output = `${root}native/neural-earth/generated-engine/`;
await mkdir(output, {recursive:true});
function replace(text, from, to) {
  if (!text.includes(from)) throw Error(`Neural engine integration no longer matches: ${from.slice(0,100)}`);
  return text.replace(from,to);
}
// Networks run on the local direct WebGPU runtime (native/neural-earth/src/engine):
// Burn's portable WebGPU convolution is a naive direct kernel.
for (const name of ['noise','pipeline','plan','reconstruct','sketch','weights']) {
  let code = (await readFile(`${source}${name}.rs`,'utf8')).replace(/\r\n/g,'\n');
  if (name === 'weights') {
    code = replace(code,'        let data = &bytes[8 + header_len..];','        let data = &bytes[8 + header_len..];\n        if header["architecture"] == "native-weight-quant-scalar-packed" {\n            return crate::packed_base::parse(&header, data);\n        }');
  }
  if (name === 'plan') {
    // AA downsampling needs one extra low cell beyond the blur/upscale halo.
    // Six cells leak its truncated edge weights into the first/last 4 samples.
    code = replace(code,'pub const ELEVATION_PAD: i64 = 6 * LATENT;','pub const ELEVATION_PAD: i64 = 7 * LATENT;');
    // Refinement's initial windows extend one stride on both sides.
    code = replace(code,'start: base.start - 1,','start: base.start - 2,');
    code = replace(code,'end: base.end - 1 + 3,','end: base.end - 1 + 4,');
  }
  if (name === 'sketch') {
    code = replace(code,'pub enum Sketch {','pub enum Sketch {\n    Planet(std::sync::Arc<crate::atlas::Atlas>),');
    code = replace(code,'Sketch::Validation(case) => validation(*case, x, y),','Sketch::Planet(atlas) => atlas.sample(x, y)[0] as f64,\n            Sketch::Validation(case) => validation(*case, x, y),');
    code = replace(code,'Sketch::Validation(_) => Climate::REFERENCE,','Sketch::Planet(_) => Climate::REFERENCE,\n            Sketch::Validation(_) => Climate::REFERENCE,');
    code = replace(code,'        let plane = size * size;\n        let climate = self.climate();','        if let Sketch::Planet(atlas) = self { return atlas.conditioning(y0, x0, size); }\n        let plane = size * size;\n        let climate = self.climate();');
  }
  if (name === 'reconstruct') {
    code = replace(code,'offset: usize, size: usize) -> Vec<f32>', 'offset: (usize, usize), size: usize) -> Vec<f32>');
    code = replace(code,'for y in offset..offset + size {\n        for x in offset..offset + size {','for y in offset.0..offset.0 + size {\n        for x in offset.1..offset.1 + size {');
  }
  if (name === 'pipeline') {
    code = replace(code,'use burn::tensor::Tensor;','use crate::gpu::Tensor;');
    code = replace(code,'    start: i64,\n    n: usize,','    start: i64,\n    xstart: i64,\n    n: usize,');
    code = replace(code,'fn new(channels: usize, start: i64, n: usize)', 'fn new(channels: usize, start: i64, xstart: i64, n: usize)');
    code = replace(code,'            start,\n            n,','            start,\n            xstart,\n            n,');
    // Cache hits must not change floating-point blending order. Assemble every
    // requested field in canonical global row-major window order, even when
    // cached and newly evaluated windows are interleaved.
    code = replace(code,'struct Field {','type PendingWindow = (Vec<f32>, usize, Vec<f32>);\nstruct Field {\n    pending: std::collections::BTreeMap<(i64,i64), PendingWindow>,');
    code = replace(code,'            channels,\n            start,','            pending: Default::default(),\n            channels,\n            start,');
    code = replace(code,'    fn add(&mut self, window:', '    fn accumulate(&mut self, window:');
    code = replace(code,'impl Field {',`impl Field {
    fn add(&mut self, window:&[f32], y0:i64, x0:i64, size:usize, weight:&[f32]) {
        self.pending.insert((y0,x0),(window.to_vec(),size,weight.to_vec()));
    }
    fn finalize(&mut self) {
        for ((y,x),(window,size,weight)) in std::mem::take(&mut self.pending) {
            self.accumulate(&window,y,x,size,&weight);
        }
    }
`);
    code = replace(code,'let x = x0 + wx as i64 - self.start;', 'let x = x0 + wx as i64 - self.xstart;');
    code = replace(code,'+ (x0 - self.start + x as i64)', '+ (x0 - self.xstart + x as i64)');
    code = replace(code,'    gpu: Gpu,\n    coarse: Edm,','    cache: std::cell::RefCell<std::collections::BTreeMap<(u8, i64, i64), Vec<f32>>>,\n    gpu: Gpu,\n    coarse: Edm,');
    code = replace(code,'            gpu,\n        })','            gpu,\n            cache: Default::default(),\n        })');
    code = replace(code,'    pub origin: Option<i64>,','    pub origin: Option<i64>,\n    pub origin_x: Option<i64>,\n    pub output_stage: u8,');
    code = replace(code,'            origin: None,','            origin: None,\n            origin_x: None,\n            output_stage: 0,');
    code = replace(code,'let plan = Plan::new(origin, size, request.exact);','let plan = Plan::new(origin, size, request.exact);\n        let plan_x = Plan::new(request.origin_x.unwrap_or(origin), size, request.exact);');
    code = replace(code,'self.coarse_stage(request, &plan)', 'self.coarse_stage(request, &plan, &plan_x)');
    code = replace(code,'Self::tune_relief(request, &plan, size, &mut coarse)', 'Self::tune_relief(request, &plan, size, &mut coarse)');
    // Planetary tuning is applied only with the default relief scale. Diagnostic means aren't used.
    code = replace(code,'let coarse_stats = Self::tune_relief(request, &plan, size, &mut coarse);','let coarse_stats = if matches!(request.sketch, Sketch::Planet(_)) { CoarseStats::default() } else { Self::tune_relief(request, &plan, size, &mut coarse) };');
    code = replace(code,'self.base_stage(request, &plan, &coarse)', 'self.base_stage(request, &plan, &plan_x, &coarse)');
    code = replace(code,'self.decoder_stage(request, &plan, &latents)', 'self.decoder_stage(request, &plan, &plan_x, &latents)');
    code = replace(code,'r.start, r.start, r.len()', 'r.start, plan_x.residual.start, r.len()');
    code = replace(code,'latents.normalized(4, low, low, r.len() / LATENT as usize)', 'latents.normalized(4, low, plan_x.residual.start / LATENT, r.len() / LATENT as usize)');
    code = replace(code,'(origin - r.start) as usize, size)', '((origin - r.start) as usize, (plan_x.origin - plan_x.residual.start) as usize), size)');
    code = replace(code,'f.normalized(c, f.start, f.start, f.n)', 'f.normalized(c, f.start, f.xstart, f.n)');
    code = replace(code,'async fn coarse_stage(&self, request: &Request, plan: &Plan)', 'async fn coarse_stage(&self, request: &Request, plan: &Plan, plan_x: &Plan)');
    code = code.replaceAll('        plan: &Plan,\n        coarse: &Field,', '        plan: &Plan,\n        plan_x: &Plan,\n        coarse: &Field,');
    code = code.replaceAll('        plan: &Plan,\n        latents: &Field,', '        plan: &Plan,\n        plan_x: &Plan,\n        latents: &Field,');
    for (const [stage, span, channels, stride, id] of [['coarse','coarse_cells',6,'COARSE_STRIDE',0],['base','latent',5,'BASE_STRIDE',1],['decoder','residual',1,'DECODER_STRIDE',2]]) {
      code = replace(code,`plan.${stage}.tiles().map(move |j| (i, j))`, `plan_x.${stage}.tiles().map(move |j| (i, j))`);
      const initial = `        let mut field = Field::new(${channels}, plan.${span}.start, plan.${span}.len());\n`;
      code = replace(code,initial,'');
      const match = `            .flat_map(|i| plan_x.${stage}.tiles().map(move |j| (i, j)))\n            .collect();`;
      code = replace(code,match,`${match}\n        let mut field = Field::new(${channels}, plan.${span}.start, plan_x.${span}.start, plan.${span}.len().max(plan_x.${span}.len()));\n        let weight = weight_window(${stage === 'coarse'?'COARSE_SIZE':stage === 'base'?'BASE_SIZE':'DECODER_SIZE'} as usize);\n        let tiles = self.cached(${id}, tiles, &mut field, ${stride}, &weight);\n        if tiles.is_empty() { return Ok(field); }`);
    }
    code = replace(code,'field.add(&physical, i * COARSE_STRIDE, j * COARSE_STRIDE, s, &weight);', 'self.remember(0, i, j, &physical);\n            field.add(&physical, i * COARSE_STRIDE, j * COARSE_STRIDE, s, &weight);');
    for (const [id, stride, channels] of [[1,'BASE_STRIDE',5],[2,'DECODER_STRIDE',1]]) {
      const slice = channels === 5 ? '&values[k * 5 * plane..(k + 1) * 5 * plane]' : '&values[k * s * s..(k + 1) * s * s]';
      code = replace(code,`                field.add(\n                    ${slice},\n                    i * ${stride}`, `                self.remember(${id}, i, j, ${slice});\n                field.add(\n                    ${slice},\n                    i * ${stride}`);
    }
    const fastOutput = (field, stride, band, scale, bias) => `        if request.output_stage >= ${field==='coarse'?4:3} {\n            let heights = crate::atlas::sample_field(&${field}.v, ${field}.channels, ${field}.start, ${field}.xstart, ${field}.n, ${band}, ${stride}, origin, plan_x.origin, request.extent_m, request.resolution, ${scale}, ${bias});\n            return Ok(Output {stages:None, coarse:coarse_stats, native:Vec::new(),native_n:0, heights,resolution:request.resolution,evaluations:(0,0,0),timings});\n        }\n`;
    code = replace(code,'        timings.coarse_ms = clock.now_ms() - started;','        timings.coarse_ms = clock.now_ms() - started;\n'+fastOutput('coarse',256,0,'1.0','0.0'));
    code = replace(code,'        timings.base_ms = clock.now_ms() - started;','        timings.base_ms = clock.now_ms() - started;\n'+fastOutput('latents',8,4,'38.6','-31.4'));
    code = replace(code,'impl Networks {',`impl Networks {
    pub fn clear_cache(&self) { self.cache.borrow_mut().clear(); }
    pub fn precision(&self) -> &'static str { self.gpu.precision() }
    pub fn gpu(&self) -> &Gpu { &self.gpu }
    /// Compiles every network pipeline with one tiny pass of each network.
    pub async fn warm(&self) -> Result<(), String> {
        let gpu = &self.gpu;
        let emb = self.coarse.embedding(&[0.0], &vec![vec![0.0]; 5], None);
        read(self.coarse.forward(gpu.image(vec![0.0; 11 * 16 * 16], [1, 11, 16, 16]), &emb)).await?;
        let emb = self.base.embedding(&[0.0], &[], Some((&[0.0; 58], 58)));
        read(self.base.forward(gpu.image(vec![0.0; 5 * 64 * 64], [1, 5, 64, 64]), &emb)).await?;
        read(self.decoder.forward(gpu.image(vec![0.0; 16 * 64 * 64], [1, 16, 64, 64]))).await?;
        Ok(())
    }
    fn cached(&self, stage:u8, tiles:Vec<(i64,i64)>, field:&mut Field, stride:i64, weight:&[f32]) -> Vec<(i64,i64)> {
        let cache = self.cache.borrow();
        tiles.into_iter().filter(|&(y,x)| {
            if let Some(v) = cache.get(&(stage,y,x)) {field.add(v,y*stride,x*stride,(weight.len() as f64).sqrt() as usize,weight);false} else {true}
        }).collect()
    }
    fn remember(&self, stage:u8, y:i64, x:i64, value:&[f32]) {
        let mut cache=self.cache.borrow_mut();
        // Bounded independent window cache (approximately 128 MiB).
        while cache.values().map(|v|v.len()*4).sum::<usize>() + value.len()*4 > 128*1024*1024 {
            if let Some(key)=cache.keys().next().copied() {cache.remove(&key);} else {break;}
        }
        cache.insert((stage,y,x),value.to_vec());
    }
`);
    code = code.replaceAll('return Ok(field);','field.finalize(); return Ok(field);');
    code = code.replaceAll('        Ok(field)','        field.finalize();\n        Ok(field)');
    code = twoStepBase(code, replace);
  }
  await writeFile(`${output}${name}.rs`,code);
}

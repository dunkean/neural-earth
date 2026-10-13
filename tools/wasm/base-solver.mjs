// Adapt the shared one-pass Rust operator to Python's native T=2 base schedule.
// The EDM forward itself remains the pinned upstream implementation.
export function twoStepBase(code, replace) {
  const begin=code.indexOf('    async fn base_stage('),end=code.indexOf('    async fn decoder_stage(',begin);
  if(begin<0||end<0)throw Error('Shared base operator was not found');
  const source=code.slice(begin,end);
  const loop=source.indexOf('        let s = BASE_SIZE as usize;');
  if(loop<0)throw Error('Shared base operator loop changed');
  let body=source.slice(loop);
  body=replace(body,'let t = round((80.0f32 / SIGMA_DATA).atan());','let t = round((if initial.is_some() { 0.35f32 } else { 80.0f32 } / SIGMA_DATA).atan());');
  body=replace(body,'request.seed.wrapping_add(5819)','request.seed.wrapping_add(if initial.is_some() { 5820 } else { 5819 })');
  body=replace(body,'            let noise_gpu = self.gpu.image(noise.clone(), [n, 5, s, s]);',`            let noise_gpu = self.gpu.image(noise, [n, 5, s, s]);
            let mut input = noise_gpu.mul_scalar(sin);
            if let Some(initial) = initial {
                let mut prior = Vec::with_capacity(n * 5 * plane);
                for &(i,j) in batch {
                    for channel in 0..5 {
                        prior.extend(initial.normalized(channel, i*BASE_STRIDE, j*BASE_STRIDE, s));
                    }
                }
                input = self.gpu.image(prior, [n,5,s,s]).mul_scalar(cos).add(input);
            }`);
  body=replace(body,'let out = self.base.forward(noise_gpu.clone().mul_scalar(sin), &emb);\n            let sample = noise_gpu.mul_scalar(cos * sin).add(out.mul_scalar(sin));','let out = self.base.forward(input.clone(), &emb);\n            let sample = input.mul_scalar(cos).add(out.mul_scalar(sin));');
  body=replace(body,'self.remember(1, i, j,','self.remember(stage, i, j,');
  const implementation=`    async fn base_stage(
        &self, request:&Request, plan:&Plan, plan_x:&Plan, coarse:&Field,
    ) -> Result<Field,String> {
        let mut field=Field::new(5,plan.latent.start,plan_x.latent.start,plan.latent.len().max(plan_x.latent.len()));
        let weight=weight_window(BASE_SIZE as usize);
        let tiles=plan.base.tiles().flat_map(|i|plan_x.base.tiles().map(move |j|(i,j))).collect();
        let tiles=self.cached(1,tiles,&mut field,BASE_STRIDE,&weight);
        if tiles.is_empty() { field.finalize(); return Ok(field); }
        // Every refinement window reads the blended first pass, including
        // all three overlapping initialization windows on each axis.
        let mut dependencies=std::collections::BTreeSet::new();
        for &(i,j) in &tiles { for y in i-1..=i+1 { for x in j-1..=j+1 { dependencies.insert((y,x)); } } }
        let initial=Field::new(5,plan.base.start*BASE_STRIDE,plan_x.base.start*BASE_STRIDE,
            (plan.base.len().max(plan_x.base.len())+1)*BASE_STRIDE as usize);
        let initial=self.base_pass(request,coarse,dependencies.into_iter().collect(),initial,None).await?;
        self.base_pass(request,coarse,tiles,field,Some(&initial)).await
    }

    async fn base_pass(
        &self, request:&Request, coarse:&Field, tiles:Vec<(i64,i64)>, mut field:Field, initial:Option<&Field>,
    ) -> Result<Field,String> {
        let stage=if initial.is_some() { 1 } else { 3 };
        let weight=weight_window(BASE_SIZE as usize);
        let tiles=self.cached(stage,tiles,&mut field,BASE_STRIDE,&weight);
        if tiles.is_empty() { field.finalize(); return Ok(field); }
${body}`;
  return code.slice(0,begin)+implementation+code.slice(end);
}

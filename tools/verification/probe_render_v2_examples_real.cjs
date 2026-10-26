// Independent CC0 by-example material experiment. All modifications are
// intercepted HTTP responses; production files/world identity stay unchanged.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),base=path.join(root,'output/render-v2-floor/by-example-causal');
const variant=process.env.EXAMPLE_PROBE||'stochastic';assert(['production','production-witness-v2','stochastic','stochastic-detail'].includes(variant));
const production=variant.startsWith('production');
const out=path.join(base,variant);fs.mkdirSync(out,{recursive:true});
const hash=s=>crypto.createHash('sha256').update(s).digest('hex');
const assetBase=variant==='stochastic-detail'?path.join(base,'soil-highpass'):base;
const meta=JSON.parse(fs.readFileSync(path.join(assetBase,'material-array.json'),'utf8'));
const sigma=meta.materials.flatMap(m=>m.gaussianSigmaByMip).map(v=>Number.isInteger(v)?v+'.':String(v)).join(',');
const helper=fs.readFileSync(path.join(__dirname,'render_v2_example_material.wgsl'),'utf8')
 .replace(/const EXAMPLE_SIGMA=array<f32,18>\([\s\S]*?\);/,`const EXAMPLE_SIGMA=array<f32,18>(${sigma});`);
const changes=[['      fn v2Fragment(',helper+'\n      fn v2Fragment('],
 ['forestResolvedCover*=e.leaf;}',
  'forestResolvedCover*=e.leaf;exampleOpaqueWitness=forestResolvedCover>=.99999;}'],
 ['return vec4<f32>(v2Final(v2DisplayShoulder(color),s,p,fp,e),1.);',
  'if(params.style.z>.5){return vec4<f32>(vec3<f32>(select(0.,1.,exampleOpaqueWitness)),1.);}return vec4<f32>(v2Final(v2DisplayShoulder(color),s,p,fp,e),1.);']];
if(!production)changes.push(
 ['var clasts=0.;let detail=groundFieldWithClasts(p,r3Seed(),fp,&clasts);',
  'var clasts=0.;let detail=groundFieldWithClasts(p,r3Seed(),fp,&clasts);\n        let exampleKeep=groundVisibility(1.28,fp)*(1.-sstep(.05,.35,e.snow))*(1.-.6*e.organic);var example:ExampleSurface;example.ratio=1.;example.gradient=vec3<f32>(0.);example.rough=0.;if(exampleKeep>0.){example=exampleAt(p,s,fp,e.rock,r3Seed());}let exampleGradient=vec2<f32>(example.gradient.x+example.gradient.y*s.y,example.gradient.z+example.gradient.y*s.z)*.28*exampleKeep;'],
 ['m.normal=normalize(vec3<f32>(-s.y*2.*params.lighting.w-detail.y*gain,1.,-s.z*2.*params.lighting.w-detail.w*gain));',
  'm.normal=normalize(vec3<f32>(-s.y*2.*params.lighting.w-detail.y*gain-exampleGradient.x,1.,-s.z*2.*params.lighting.w-detail.w*gain-exampleGradient.y));'],
 ['v2FloorModulation=1.+form+swardTone;',
  'v2FloorModulation=(1.+form+swardTone)*mix(1.,example.ratio,.65*exampleKeep);'],
 ['m.roughness=clamp(pow(pow(v2ReferenceRoughness,4.)+gain*gain*groundSlopeVariance(fp),.25),.4,.98);',
  'm.roughness=clamp(pow(pow(v2ReferenceRoughness,4.)+gain*gain*groundSlopeVariance(fp),.25)+example.rough*.12*exampleKeep,.4,.98);']);
const injection=`\n;(()=>{const changes=${JSON.stringify(changes)};const original=TerrainRenderV2.drawShader;TerrainRenderV2.drawShader=(...args)=>{let source=original(...args);for(const [before,after] of changes){if(!source.includes(before))throw Error('By-example hook missing: '+before);source=source.replace(before,after);}return source;};})();`;
const loader=`
    initializeExampleMaterial(){
      if(this.examplePromise)return this.examplePromise;
      this.examplePromise=(async()=>{
        const meta=await (await fetch('/__v2_byexample/material-array.json')).json();
        this.exampleTexture=this.device.createTexture({label:'CC0 by-example material prototype',size:[meta.size,meta.size,meta.layers],format:meta.format,mipLevelCount:meta.mipLevels,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
        for(const level of meta.levels){const response=await fetch('/__v2_byexample/'+level.file);if(!response.ok)throw Error('Missing prototype material mip');const bytes=await response.arrayBuffer();this.device.queue.writeTexture({texture:this.exampleTexture,mipLevel:level.level},bytes,{bytesPerRow:level.size*8,rowsPerImage:level.size},[level.size,level.size,meta.layers]);}
        this.exampleSampler=this.device.createSampler({addressModeU:'repeat',addressModeV:'repeat',minFilter:'linear',magFilter:'linear',mipmapFilter:'linear'});
        this.exampleBytes=meta.allocationBytes;
      })();return this.examplePromise;
    }
`;
const rendererChanges=[
 ['    ensureRender3d() {',loader+'\n    ensureRender3d() {'],
 ["      this.r3Promise=(async()=>{\n        device.pushErrorScope('validation');","      this.r3Promise=(async()=>{\n        await this.initializeExampleMaterial();\n        device.pushErrorScope('validation');"],
 ["          this.r3DrawLayout=device.createBindGroupLayout({entries:[","          this.r3DrawLayout=device.createBindGroupLayout({entries:[\n            {binding:10,visibility:fragment,texture:{viewDimension:'2d-array'}},\n            {binding:11,visibility:fragment,sampler:{type:'filtering'}},"],
 ['const uniform=r3.tileUniform(origin,tile.materialSeed);',
  'const uniform=r3.tileUniform(origin,tile.materialSeed);{const cells=new Int32Array(uniform,0,128),cellBits=new Float32Array(uniform,0,128),fracs=new Float32Array(uniform,512,128);for(let i=0;i<2;i++){const q=origin[i]/2.56,cell=Math.floor(q),skew=q*Math.sqrt(12),skewCell=Math.floor(skew);cells[124+i]=cell;fracs[121+i]=q-cell;cells[120+i]=skewCell;cellBits[122+i]=skew-skewCell;}}'],
 [":kind==='render3d'?{layout:this.r3DrawLayout,entries:[",":kind==='render3d'?{layout:this.r3DrawLayout,entries:[{binding:10,resource:this.exampleTexture.createView({dimension:'2d-array'})},{binding:11,resource:this.exampleSampler},"],
 ['...(tile.origin||[0,0]),0,0,...(city?.vectors()', '...(tile.origin||[0,0]),window.__exampleMaskMode||0,0,...(city?.vectors()'],
 ['this.available=false;this.clear();this.noHydrology?.destroy();', 'this.available=false;this.clear();this.exampleTexture?.destroy();this.noHydrology?.destroy();']
];
// Fail before launching Chrome if a source hook or JS response is invalid.
let rendererPreflight=fs.readFileSync(path.join(root,'web/terrain_renderer.js'),'utf8').split(String.fromCharCode(13,10)).join(String.fromCharCode(10));
for(const [a,b] of rendererChanges){assert(rendererPreflight.includes(a),'Renderer hook missing: '+a);rendererPreflight=rendererPreflight.replace(a,b);}
new Function(rendererPreflight);
const override={variant,drawChanges:changes,rendererChanges,assetMeta:meta,
 assetProvenance:JSON.parse(fs.readFileSync(path.join(base,'asset-provenance.json'),'utf8'))};
const overrideHash=hash(JSON.stringify(override));
fs.writeFileSync(path.join(out,'ABLATION.json'),JSON.stringify({kind:'EXPERIMENTAL ROUTE-ONLY stochastic CC0 material prototype, pending root visual review',productionCodeHash:'2b6b939d8e0e1c1eaeff4ecb1a0f314d73b202da41c3adc8fe673157d8ecbed3',overrideHash,...override},null,2));
const manifest=JSON.parse(fs.readFileSync(path.join(root,'output/render-v2-floor/meso-clasts-causal/manifest.json'),'utf8'));
const main=JSON.parse(fs.readFileSync(path.join(root,'output/render-v2-reference/manifest.json'),'utf8'));
manifest.sites.push({...main.sites.find(s=>s.id==='polar-plain-candidate-01')});
manifest.sites.push({...main.sites.find(s=>s.id==='af-rolling')});
manifest.sites.forEach(s=>delete s.featureProvenance);
manifest.label='EXPERIMENTAL ROUTE-ONLY CC0 by-example material: real terrain, not final acceptance';
manifest.expectedFrames=45;manifest.routeOnlyExperiment={variant,overrideHash,assetArrayBytes:meta.allocationBytes};
fs.writeFileSync(path.join(out,'manifest.json'),JSON.stringify(manifest,null,2));process.env.REVIEW_REFERENCE_BASE=out;
const runner=path.join(__dirname,'review_render_v2_reference.cjs');let code=fs.readFileSync(runner,'utf8');
const anchor="   page.on('pageerror',e=>entry.errors.push(e.stack));";assert(code.includes(anchor));
const rendererTail='\n;{const original=createTerrainRenderer;window.createTerrainRenderer=(...args)=>original(...args).then(r=>{window.__exampleRenderer=r;return r;});}';
code=code.replace(anchor,anchor+`
   await page.route('**/__v2_byexample/*',async route=>{const file=path.basename(new URL(route.request().url()).pathname);assert(file==='material-array.json'||meta.levels.some(l=>l.file===file));await route.fulfill({status:200,contentType:file.endsWith('.json')?'application/json':'application/octet-stream',body:fs.readFileSync(path.join(${JSON.stringify(assetBase)},file))});});
   await page.route('**/terrain_render_v2.js*',async route=>{await route.fulfill({status:200,contentType:'application/javascript',body:fs.readFileSync(path.join(root,'web/terrain_render_v2.js'),'utf8')+${JSON.stringify(injection)}});});
   await page.route('**/terrain_renderer.js*',async route=>{let source=fs.readFileSync(path.join(root,'web/terrain_renderer.js'),'utf8').split(String.fromCharCode(13,10)).join(String.fromCharCode(10));for(const [a,b] of ${JSON.stringify(rendererChanges)}){if(!source.includes(a))throw Error('Example renderer hook missing: '+a);source=source.replace(a,b);}source+=${JSON.stringify(rendererTail)};await route.fulfill({status:200,contentType:'application/javascript',body:source});});
`);
// Keep low-level shader errors visible rather than silently awaiting a fallback.
code=code.replace("await page.waitForFunction(()=>window.terrainDebug?.snapshot().renderer?.render3d?.ready,null,{timeout:120000});", "await page.waitForFunction(()=>{const s=window.terrainDebug?.snapshot();if(s?.renderer?.render3d?.error)throw Error(s.renderer.render3d.error);return s?.renderer?.render3d?.ready;},null,{timeout:120000});");
code=code.replace('entry.frames.push({zoom:z.id,tag,file,sha256:digest(png),dimensions,imageDimensions,',
 `entry.frames.push({routeOverride:${JSON.stringify({variant,sha256:overrideHash})},zoom:z.id,tag,file,sha256:digest(png),dimensions,imageDimensions,`);
const maskAnchor="     const png=fs.readFileSync(path.join(base,file));assert.equal(png.subarray(1,4).toString('ascii'),'PNG');";
assert(code.includes(maskAnchor));
code=code.replace(maskAnchor,maskAnchor+`
     if(['bsk-rolling','af-rolling'].includes(site.id)&&['160m','40m'].includes(z.id)&&tag==='v2-july'){
      await page.evaluate(()=>{window.__exampleMaskMode=1;const r=window.__exampleRenderer,c=document.getElementById('gpuMap');r.draw(r.lastDrawnRects,{width:c.width,height:c.height,dpr:1});});
      const maskName=site.id+'-'+z.id+'-actual-opaque-mask.png',maskFile=path.join(out,maskName);
      await page.locator('#gpuMap').screenshot({path:maskFile});
      (entry.controlMasks??=[]).push({zoom:z.id,file:path.relative(base,maskFile).split(path.sep).join('/'),sha256:digest(fs.readFileSync(maskFile)),authority:'actual forestResolvedCover>=.99999 after leaf/crown visibility'});
      await page.evaluate(()=>{window.__exampleMaskMode=0;const r=window.__exampleRenderer,c=document.getElementById('gpuMap');r.draw(r.lastDrawnRects,{width:c.width,height:c.height,dpr:1});});
     }
`);
const identityAnchor='const codeIdentity=identity(),codeHash=digest(JSON.stringify(codeIdentity));';assert(code.includes(identityAnchor));
code=code.replace(identityAnchor,identityAnchor+"assert.equal(codeHash,'2b6b939d8e0e1c1eaeff4ecb1a0f314d73b202da41c3adc8fe673157d8ecbed3','Production differs from2b6');");
// Expose only wrapper-local immutable configuration to the compiled runner.
const virtual=path.join(out,'capture-wrapper.cjs');
fs.writeFileSync(virtual,`const meta=${JSON.stringify(meta)};\n`+code.replace("const root=path.resolve(__dirname,'../..'),base=",`const root=${JSON.stringify(root)},base=`));
fs.writeFileSync(path.join(out,'replay.cjs'),`// Replay the exact archived response overrides against their frozen production.\nconst fs=require('node:fs'),path=require('node:path'),Module=require('node:module');\nconst runner=${JSON.stringify(runner)};process.env.REVIEW_REFERENCE_BASE=__dirname;const child=new Module(runner,module);child.filename=runner;child.paths=Module._nodeModulePaths(path.dirname(runner));child._compile(fs.readFileSync(path.join(__dirname,'capture-wrapper.cjs'),'utf8'),runner);\n`);
function run(){const child=new Module(runner,module);child.filename=runner;child.paths=Module._nodeModulePaths(__dirname);child._compile(fs.readFileSync(virtual,'utf8'),runner);}
// Use runner's own module path for ../platform.cjs resolution.
process.argv=process.argv.filter(a=>!['--run','--freeze'].includes(a));process.argv.push('--freeze');run();process.argv.pop();process.argv.push('--run');run();

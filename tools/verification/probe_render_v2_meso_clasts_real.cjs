// Explicit route-only material experiment on unchanged real cached DEMs.
// Production source files are never edited. Not a final reference corpus.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..');
const variant=process.env.MESO_PROBE||'clasts-only';
assert(['production','clasts-only'].includes(variant));
const base=path.resolve(root,process.env.MESO_PROBE_BASE||'output/render-v2-floor/meso-clasts-causal');
const output=path.join(base,variant);fs.mkdirSync(output,{recursive:true});
const hash=s=>crypto.createHash('sha256').update(s).digest('hex');
const helper=`
      // 4x the inherited .32m lattice. Keep the quotient in integer space:
      // WGSL integer division truncates, so normalize signed remainders first.
      // Only small local coordinates become floats; world cells stay exact.
      fn v2MesoClasts(p:vec2<f32>,seed:u32,fp:f32)->vec4<f32>{
        let keep=groundVisibility(1.28*.85*.5,max(fp,0.));
        if(keep<=0.){return vec4<f32>(0.);}
        let cell=r3.cells[29].xy;
        let remainder=((cell%vec2<i32>(4))+vec2<i32>(4))%vec2<i32>(4);
        let quotient=(cell-remainder)/vec2<i32>(4);
        let cs=groundRotation(2,seed);let c=cs.x;let sn=cs.y;
        let local=(vec2<f32>(remainder)+r3.fracs[29].xy+
          vec2<f32>(c*p.x+sn*p.y,-sn*p.x+c*p.y)/vec2<f32>(.32,.32*.85))*.25;
        let fl=floor(local);
        let cap=groundGravel(quotient+vec2<i32>(fl),local-fl,seed+3u*0x6c8e9cf5u);
        let g=cap.yw/vec2<f32>(1.28,1.28*.85);
        return vec4<f32>((cap.x-.1)*keep,(c*g.x-sn*g.y)*keep,0.,(sn*g.x+c*g.y)*keep);
      }
`;
const changes=[
 ['      fn v2Fragment(',helper+'\n      fn v2Fragment('],
 ['var clasts=0.;let detail=groundFieldWithClasts(p,r3Seed(),fp,&clasts);',
  'var clasts=0.;let detail=groundFieldWithClasts(p,r3Seed(),fp,&clasts);\n        let meso=v2MesoClasts(p,r3Seed(),fp);let mesoWeight=(e.rock+(1.-e.rock)*(1.-e.grass)*(1.-.8*e.sand)*(1.-e.organic))*(1.-e.snow);'],
 ['m.normal=normalize(vec3<f32>(-s.y*2.*params.lighting.w-detail.y*gain,1.,-s.z*2.*params.lighting.w-detail.w*gain));',
  'm.normal=normalize(vec3<f32>(-s.y*2.*params.lighting.w-detail.y*gain-meso.y*.025*mesoWeight,1.,-s.z*2.*params.lighting.w-detail.w*gain-meso.w*.025*mesoWeight));'],
 ['v2FloorModulation=1.+form+swardTone;',
  'v2FloorModulation=(1.+form+swardTone)*(1.+meso.x*.13*mesoWeight);']
];
const actualChanges=variant==='production'?[]:changes;
const injection=`\n;(()=>{const changes=${JSON.stringify(actualChanges)};const production=TerrainRenderV2.drawShader;TerrainRenderV2.drawShader=(...args)=>{let source=production(...args);for(const [before,after] of changes){if(!source.includes(before))throw Error('Meso-clast override hook missing: '+before);source=source.replace(before,after);}return source;};})();`;
const receipt={kind:'EXPERIMENTAL ROUTE-ONLY real cached terrain material ablation; not accepted production',variant,
 productionCodeHash:'2b6b939d8e0e1c1eaeff4ecb1a0f314d73b202da41c3adc8fe673157d8ecbed3',
 physicalSourcesUnchanged:true,syntheticTerrain:false,negativeLodForbidden:true,
 override:{route:'/terrain_render_v2.js',changes:actualChanges,injection,sha256:hash(injection)},
 policy:'Mesoscopic1.28m inherited clasts only. Low-amplitude exposed mineral/rock albedo and2.5cm bump. Suppressed by snow/grass/sand/organic weights; composed beneath true crowns. No strata yet.',
 acceptance:'pending root visual review; reject clouds, blobs or plastic response'};
fs.writeFileSync(path.join(output,'ABLATION.json'),JSON.stringify(receipt,null,2));
const manifest=JSON.parse(fs.readFileSync(path.join(base,'manifest.json'),'utf8'));
manifest.routeOnlyExperiment={variant,overrideSha256:hash(injection),productionCodeHash:receipt.productionCodeHash};
fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(manifest,null,2));
process.env.REVIEW_REFERENCE_BASE=output;
const runner=path.join(__dirname,'review_render_v2_reference.cjs');let code=fs.readFileSync(runner,'utf8');
const anchor="   page.on('pageerror',e=>entry.errors.push(e.stack));";
assert(code.includes(anchor),'Canonical capture page hook changed');
code=code.replace(anchor,anchor+`\n   await page.route('**/terrain_render_v2.js*',async route=>{await route.fulfill({status:200,contentType:'application/javascript',body:fs.readFileSync(path.join(root,'web/terrain_render_v2.js'),'utf8')+${JSON.stringify(injection)}});});\n`);
// Each frame names the route override rather than masquerading as production.
code=code.replace('entry.frames.push({zoom:z.id,tag,file,sha256:digest(png),dimensions,imageDimensions,',
 `entry.frames.push({routeOverride:${JSON.stringify({variant,sha256:hash(injection)})},zoom:z.id,tag,file,sha256:digest(png),dimensions,imageDimensions,`);
assert(code.includes("const codeIdentity=identity(),codeHash=digest(JSON.stringify(codeIdentity));"));
code=code.replace("const codeIdentity=identity(),codeHash=digest(JSON.stringify(codeIdentity));",`const codeIdentity=identity(),codeHash=digest(JSON.stringify(codeIdentity));assert.equal(codeHash,${JSON.stringify(receipt.productionCodeHash)},'Production changed from2b6; experiment needs review');`);
function run(){const child=new Module(runner,module);child.filename=runner;child.paths=Module._nodeModulePaths(__dirname);child._compile(code,runner);}
process.argv=process.argv.filter(a=>!['--run','--freeze'].includes(a));process.argv.push('--freeze');run();process.argv.pop();process.argv.push('--run');run();

// Route-only stochastic broadleaf crown-material experiment. Production stays unchanged.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),variant=process.env.CROWN_PROBE||'stochastic-masses';
assert(['production','stochastic-masses'].includes(variant));
const base=path.resolve(root,process.env.CROWN_PROBE_BASE||'output/render-v2-floor/crown-masses-causal'),output=path.join(base,variant);fs.mkdirSync(output,{recursive:true});
const hash=x=>crypto.createHash('sha256').update(x).digest('hex');
const actual=fs.readFileSync(path.join(root,'web/terrain_render3d.js'),'utf8').replace(/\r\n/g,'\n');
const begin=actual.indexOf('          let a=hit.local*6.5;let ph='),end=actual.indexOf('\n        }\n        let slope=',begin);
assert(begin>=0&&end>begin,'Broadleaf clump production section missing');
const before=actual.slice(begin,end);
const helper=`
    // Independent per-crown rotations, phases and gradient-noise cells. All
    // floats stay in the small crown-local frame; ID and roots stay canonical.
    fn v2CrownMass(local:vec2<f32>,frequency:f32,id:u32,phase:f32)->vec3<f32>{
      let c=cos(phase);let sn=sin(phase);
      let q=vec3<f32>(vec2<f32>(c*local.x+sn*local.y,-sn*local.x+c*local.y)*frequency+vec2<f32>(7.31,19.73),.37+53.*forestU8(id,16u));
      let fl=floor(q);let n=groundNoise(vec3<i32>(fl),q-fl,id^0x7f4a7c15u);
      return vec3<f32>(n.x,frequency*vec2<f32>(c*n.y-sn*n.z,sn*n.y+c*n.z));
    }
`;
const replacement=`          // Irregular foliar masses; no fixed three-direction wave lattice.
          // Fixed band energies disappear under the physical pixel footprint;
          // neither the last remaining band nor its contrast is renormalized.
          let primaryKeep=groundVisibility(hit.r/2.3,fp);
          let secondaryKeep=groundVisibility(hit.r/5.3,fp);
          let primary=v2CrownMass(hit.local,2.3,hit.id,phase);
          var secondary=vec3<f32>(0.);
          if(secondaryKeep>0.){secondary=v2CrownMass(hit.local,5.3,hit.id^0x5bd1e995u,phase+1.173);}
          clump=1.6*(.8*primary.x*primaryKeep+.35*secondary.x*secondaryKeep);
          dClump=1.6*(.8*primary.yz*primaryKeep+.35*secondary.yz*secondaryKeep)/hit.r;`;
const changes=variant==='production'?[]:[['    fn applyForest(',helper+'\n    fn applyForest('],[before,'          if(params.field.z<=-8.){\n'+replacement+'\n          }else{\n'+before+'\n          }']];
const injection=`\n;(()=>{const changes=${JSON.stringify(changes)};const production=TerrainRenderV2.drawShader;TerrainRenderV2.drawShader=(...args)=>{let source=production(...args);for(const [a,b] of changes){if(!source.includes(a))throw Error('Crown mass override hook missing: '+a);source=source.replace(a,b);}return source;};})();`;
const receipt={kind:'EXPERIMENTAL ROUTE-ONLY broadleaf material ablation on real cached terrain; not final acceptance',variant,productionCodeHash:'2b6b939d8e0e1c1eaeff4ecb1a0f314d73b202da41c3adc8fe673157d8ecbed3',physicalSourcesUnchanged:true,syntheticTerrain:false,negativeLodForbidden:true,override:{route:'/terrain_render_v2.js',changes,injection,sha256:hash(injection)},policy:'Broadleaf clump and analytic derivatives only. Two crown-local rotated 3D-gradient-noise slices at2.3 and5.3 cells/radius, per-crown IDs and phases, fixed energies and physical footprint filtering. Unchanged population, roots, crown outline, conifers, palette, ecology, projected shadows and beach selection.',acceptance:'Pending root visual and cost review; no production or City Generator port authorized by this probe'};
fs.writeFileSync(path.join(output,'ABLATION.json'),JSON.stringify(receipt,null,2));
const manifest=JSON.parse(fs.readFileSync(path.join(base,'manifest.json'),'utf8'));manifest.routeOnlyExperiment={variant,overrideSha256:hash(injection),productionCodeHash:receipt.productionCodeHash};fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(manifest,null,2));process.env.REVIEW_REFERENCE_BASE=output;
const runner=path.join(__dirname,'review_render_v2_reference.cjs');let code=fs.readFileSync(runner,'utf8');const anchor="   page.on('pageerror',e=>entry.errors.push(e.stack));";assert(code.includes(anchor));
code=code.replace(anchor,anchor+`\n   await page.route('**/terrain_render_v2.js*',async route=>{await route.fulfill({status:200,contentType:'application/javascript',body:fs.readFileSync(path.join(root,'web/terrain_render_v2.js'),'utf8')+${JSON.stringify(injection)}});});\n`);
code=code.replace('entry.frames.push({zoom:z.id,tag,file,sha256:digest(png),dimensions,imageDimensions,',`entry.frames.push({routeOverride:${JSON.stringify({variant,sha256:hash(injection)})},zoom:z.id,tag,file,sha256:digest(png),dimensions,imageDimensions,`);
assert(code.includes('const codeIdentity=identity(),codeHash=digest(JSON.stringify(codeIdentity));'));code=code.replace('const codeIdentity=identity(),codeHash=digest(JSON.stringify(codeIdentity));',`const codeIdentity=identity(),codeHash=digest(JSON.stringify(codeIdentity));assert.equal(codeHash,${JSON.stringify(receipt.productionCodeHash)},'Production changed from2b6; stop');`);
function run(){const child=new Module(runner,module);child.filename=runner;child.paths=Module._nodeModulePaths(__dirname);child._compile(code,runner);}
process.argv=process.argv.filter(a=>!['--run','--freeze'].includes(a));process.argv.push('--freeze');run();process.argv.pop();process.argv.push('--run');run();

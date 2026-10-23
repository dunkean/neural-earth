// ROUTE-ONLY structural canopy z-profile experiment. No production edits.
const fs=require('node:fs'),path=require('node:path'),Module=require('node:module'),crypto=require('node:crypto'),assert=require('node:assert/strict');
const root=path.resolve(__dirname,'../..'),variant=process.env.CANOPY_PROBE||'hierarchical-lobes';assert(['production','hierarchical-lobes'].includes(variant));
const base=path.resolve(root,process.env.CANOPY_PROBE_BASE||'output/render-v2-floor/canopy-hierarchy-causal'),output=path.join(base,variant);fs.mkdirSync(output,{recursive:true});const hash=x=>crypto.createHash('sha256').update(x).digest('hex');
const actual=fs.readFileSync(path.join(root,'web/terrain_render3d.js'),'utf8').replace(/\r\n/g,'\n');
function section(a,b){const i=actual.indexOf(a),j=actual.indexOf(b,i);assert(i>=0&&j>i,a);return actual.slice(i,j);}

// Bound by the next function rather than relying on formatting of nested loops.
const crownBegin=actual.indexOf('        var r=k.r;var a=0.;var ink=.5;');const crownEnd=actual.indexOf('\n      }}\n    }\n    // Soft cast shadow',crownBegin);assert(crownEnd>crownBegin);const oldCrown=actual.slice(crownBegin,crownEnd);
const shadowBegin=actual.indexOf('        let caster=k.h-.5*k.depth;'),shadowEnd=actual.indexOf('\n      }}\n      return shade;',shadowBegin);assert(shadowEnd>shadowBegin);const oldShadow=actual.slice(shadowBegin,shadowEnd);
const helper=`
    struct V2CanopyLobe { centre:vec2<f32>,axes:vec2<f32>,rotation:vec2<f32>,z:f32,depth:f32 }
    struct V2CanopyProfile { z:f32,slope:vec2<f32>,cover:f32,rel:f32,near:f32 }
    fn v2CanopyDirection(h:u32)->vec2<f32>{let d=vec2<f32>(forestU8(h,0u),forestU8(h,8u))*2.-1.;return normalize(d+vec2<f32>(.013,-.019));}
    fn v2CanopyLobe(k:ForestCrown,index:i32)->V2CanopyLobe{
      let parent=index/3;let child=index%3;
      let hp=groundHash(vec3<i32>(parent,71,13),k.id^0xa511e9b3u);
      let hc=groundHash(vec3<i32>(index,43,97),k.id^0x63d83595u);
      let direction=v2CanopyDirection(hp);let side=vec2<f32>(-direction.y,direction.x);
      var l:V2CanopyLobe;
      if(k.needle){
        // Three vertically separated branch families. Every branch has its
        // own azimuth, lateral offset, size and crest; no repeated radial fan.
        let tier=f32(parent);let branch=v2CanopyDirection(hc);
        let size=(.60-.17*tier)*(.92+.16*forestU8(hc,16u));
        l.centre=k.r*(.30-.075*tier)*branch;
        l.axes=k.r*size*vec2<f32>(1.05+.10*forestU8(hp,16u),.89+.10*forestU8(hp,24u));
        l.rotation=branch;l.depth=k.r*(.40-.08*tier);
        let crest=k.h-k.depth*(.70-.28*tier)+l.depth;
        l.z=crest-l.depth+.06*k.r*(forestU8(hc,24u)-.5);
      }else{
        // Three primary scaffold branches, each supporting three overlapping
        // foliar volumes. Child offsets are correlated within a scaffold.
        let along=.27+.12*forestU8(hp,16u);
        let lateral=(f32(child)-1.)*.23+.09*(forestU8(hc,0u)-.5);
        l.centre=k.r*(direction*(along+.12*(forestU8(hc,8u)-.5))+side*lateral);
        let size=.51+.12*forestU8(hc,16u);
        l.axes=k.r*size*vec2<f32>(1.03+.12*forestU8(hp,24u),.88+.12*forestU8(hc,24u));
        l.rotation=direction;l.depth=k.r*(.35+.12*forestU8(hc,8u));
        l.z=k.h-k.r*(.05+.28*forestU8(hc,24u))-l.depth;
      }
      let horizontalScale=select(1.096,1.261,k.needle);l.centre*=horizontalScale;l.axes*=horizontalScale;
      return l;
    }
    fn v2CanopyProfile(k:ForestCrown,d:vec2<f32>,aa:f32)->V2CanopyProfile{
      var p:V2CanopyProfile;p.z=-1e9;p.slope=vec2<f32>(0.);p.cover=0.;p.rel=1.;p.near=9.;
      for(var i=0;i<9;i++){
        let l=v2CanopyLobe(k,i);let delta=d-l.centre;let c=l.rotation.x;let sn=l.rotation.y;
        let local=vec2<f32>(c*delta.x+sn*delta.y,-sn*delta.x+c*delta.y);
        let q=local/l.axes;let e=dot(q,q);let radius=sqrt(e);
        let signedDistance=(1.-radius)*min(l.axes.x,l.axes.y);
        p.near=min(p.near,radius);let a=clamp(signedDistance/max(aa,.001)+.5,0.,1.);
        p.cover=max(p.cover,a);if(a<=0.){continue;}
        // Analytic upper ellipsoid surface and its derivative. At the AA
        // silhouette the vertical tangent is regularized over1%of a lobe.
        let root=sqrt(max(1.-e,.0001));let z=l.z+l.depth*root;
        let g=-l.depth*q/(l.axes*root);let slope=vec2<f32>(c*g.x-sn*g.y,sn*g.x+c*g.y);
        if(p.z< -1e8){p.z=z;p.slope=slope;p.rel=min(radius,1.);}
        else{
          // C1 smooth maximum of height fields. Its derivative is the same
          // blend of the input derivatives; no unrelated normal noise.
          let width=.10*k.r;let h=clamp(.5+.5*(z-p.z)/max(width,.001),0.,1.);
          p.z=mix(p.z,z,h)+width*h*(1.-h);p.slope=mix(p.slope,slope,h);p.rel=mix(p.rel,min(radius,1.),h);
        }
      }
      return p;
    }
    fn v2CanopyProjectedShadow(k:ForestCrown,p:vec2<f32>,zp:f32,sun:vec2<f32>,cot:f32,aa:f32)->f32{
      var shade=0.;
      for(var i=0;i<9;i++){
        let l=v2CanopyLobe(k,i);if(l.z+l.depth<=zp){continue;}
        let offset=(l.z-zp)*cot;
        let d=p-(k.c+l.centre-sun*offset);let c=l.rotation.x;let sn=l.rotation.y;
        let rx=l.axes.x*l.axes.x;let ry=l.axes.y*l.axes.y;let height=l.depth*l.depth*cot*cot;
        // Affine projection of an ellipsoid onto the receiver plane. This
        // changes only the caster geometry; source candidates and softness
        // follow the inherited stand shadow policy.
        let xx=c*c*rx+sn*sn*ry+height*sun.x*sun.x;
        let yy=sn*sn*rx+c*c*ry+height*sun.y*sun.y;
        let xy=c*sn*(rx-ry)+height*sun.x*sun.y;
        let determinant=max(xx*yy-xy*xy,.0001);
        let q=sqrt(max((yy*d.x*d.x-2.*xy*d.x*d.y+xx*d.y*d.y)/determinant,0.));
        let radius=min(l.axes.x,l.axes.y);
        let penumbra=max(aa,.22*k.r+.055*max(offset,0.));
        shade=max(shade,clamp((1.-q)*radius/penumbra+.5,0.,1.));
      }
      return shade;
    }
`;
const crownReplacement=`        if(params.field.z<=-8.&&!brush){
          let profile=v2CanopyProfile(k,d,aa);(*hit).near=min((*hit).near,profile.near);if(profile.cover<=0.){continue;}
          (*hit).cover=1.-(1.-(*hit).cover)*(1.-profile.cover);
          if(profile.z>(*hit).top){
            (*hit).second=(*hit).top;(*hit).top=profile.z;(*hit).slope=profile.slope;
            (*hit).rel=profile.rel;(*hit).local=d/k.r;(*hit).r=k.r;(*hit).needle=k.needle;
            (*hit).edge=k.edge;(*hit).id=k.id;(*hit).level=level;(*hit).ink=.5;
          }else{(*hit).second=max((*hit).second,profile.z);}
        }else{
${oldCrown}
        }`;
const shadowReplacement=`        if(params.field.z<=-8.){shade=max(shade,v2CanopyProjectedShadow(k,st.p,zp,sun,cot,aa));}
        else{
${oldShadow}
        }`;
const structureChanges=variant==='production'?[]:[['    fn forestCrowns(',helper+'\n    fn forestCrowns('],[oldCrown,crownReplacement],[oldShadow,shadowReplacement],['        if(brush){clump=(hit.ink-.5)*2.;}', '        if(params.field.z<=-8.&&!brush){}else if(brush){clump=(hit.ink-.5)*2.;}']];
// The diagnostic flag is zero for every delivered image. A read-only probe
// temporarily renders the actual final cover into the canvas, reads it, then
// restores the original material before the next frame. No synthetic DEM.
const coverageChanges=[['        return vec4<f32>(v2Final(v2DisplayShoulder(color),s,p,fp,e),1.);','        if(params.render4.w>900.){return vec4<f32>(vec3<f32>(forestResolvedCover),1.);}\n        return vec4<f32>(v2Final(v2DisplayShoulder(color),s,p,fp,e),1.);']];
const changes=[...structureChanges,...coverageChanges];
const injection=`\n;(()=>{const changes=${JSON.stringify(changes)};const production=TerrainRenderV2.drawShader;TerrainRenderV2.drawShader=(...args)=>{let source=production(...args);for(const [a,b]of changes){if(!source.includes(a))throw Error('Canopy hierarchy hook missing: '+a);source=source.replace(a,b);}return source;};let create;Object.defineProperty(window,'createTerrainRenderer',{configurable:true,get(){return create;},set(fn){create=async(...args)=>{const r=await fn(...args);if(!r)return r;window.__canopyProbe={renderer:r,flag:0,last:null};const draw=r.draw;r.draw=function(...a){window.__canopyProbe.last=a;return draw.apply(this,a);};const vectors=r.renderVectors;r.renderVectors=function(...a){const v=vectors.apply(this,a);v[19]=window.__canopyProbe.flag;return v;};return r;};}});})();`;
const receipt={kind:'EXPERIMENTAL ROUTE-ONLY structural canopy height-field ablation; not production',variant,productionCodeHash:'2b6b939d8e0e1c1eaeff4ecb1a0f314d73b202da41c3adc8fe673157d8ecbed3',syntheticTerrain:false,negativeLodForbidden:true,override:{route:'/terrain_render_v2.js',changes,injection,sha256:hash(injection)},policy:'V2 only;9ellipsoidal lobes on3scaffolds or3conifer tiers; deterministic geometry, upper height/slope/silhouette and projected shadows from same volumes. Fixed horizontal geometry normalization1.096broadleaf/1.261needle from64unit-crown area measurements; not a fit to imagery. Canonical candidates, roots, ecology, genus, beach selection and palettes unchanged. Old normal-only clump disabled. No population/Tau refit or colour-mean flattening.',limitations:'AA-distance is elliptical approximation; upper-surface tangent is regularized in silhouette AA region. Projected ellipsoids are not clipped at the receiver plane. No individual leaves, transmission, full z-field/light-field atlas or calibrated mean BRDF yet.',references:[{url:'https://onlinelibrary.wiley.com/doi/full/10.1111/j.1467-8659.2012.03016.x',use:'Reference distinguishes per-tree z-fields from distant shader-maps and scale-consistent reflectance; this experiment is only a small analytic upper-height-volume prototype, not their complete algorithm.'},{url:'https://developer.nvidia.com/gpugems/gpugems3/part-iii-rendering/chapter-16-vegetation-procedural-animation-and-shading-crysis',use:'Reference motivates foliage-specific hemisphere/AO and translucency. Existing renderer shading is retained here; those full foliage effects are not newly implemented.'}],acceptance:'Pending root visual, coverage and isolated runtime review; no production port.'};
fs.writeFileSync(path.join(output,'ABLATION.json'),JSON.stringify(receipt,null,2));
// Validate every replacement in the actual generated shader without a browser.
require(path.join(root,'web/terrain_render3d.js'));require(path.join(root,'web/terrain_render_v2.js'));let shader=globalThis.TerrainRenderV2.drawShader(globalThis.TerrainRender3D.shaders.draw,'    fn surfaceHash(\n    // (unit-variance pattern');for(const [a,b]of changes){assert(shader.includes(a),'Prepared shader hook missing: '+a);shader=shader.replace(a,b);}fs.writeFileSync(path.join(output,'prepared-shader.wgsl'),shader);
if(process.argv.includes('--prepare-only')){console.log(JSON.stringify({prepared:true,variant,override:hash(injection),output}));process.exit(0);}
const manifest=JSON.parse(fs.readFileSync(path.join(base,'manifest.json'),'utf8'));manifest.routeOnlyExperiment={variant,overrideSha256:hash(injection),productionCodeHash:receipt.productionCodeHash};fs.writeFileSync(path.join(output,'manifest.json'),JSON.stringify(manifest,null,2));process.env.REVIEW_REFERENCE_BASE=output;
const runner=path.join(__dirname,'review_render_v2_reference.cjs');let code=fs.readFileSync(runner,'utf8');const anchor="   page.on('pageerror',e=>entry.errors.push(e.stack));";assert(code.includes(anchor));code=code.replace(anchor,anchor+`\n   await page.route('**/terrain_render_v2.js*',async route=>route.fulfill({status:200,contentType:'application/javascript',body:fs.readFileSync(path.join(root,'web/terrain_render_v2.js'),'utf8')+${JSON.stringify(injection)}}));\n`);
const measure=`let coverageMeasurement=null;if(tag!=='render3d-july')coverageMeasurement=await page.evaluate(async()=>{const p=window.__canopyProbe,r=p.renderer,last=p.last;if(!last)throw Error('No real renderer frame');p.flag=901;r.draw(...last);await r.device.queue.onSubmittedWorkDone();const c=new OffscreenCanvas(r.canvas.width,r.canvas.height),ctx=c.getContext('2d');ctx.drawImage(r.canvas,0,0);const a=ctx.getImageData(0,0,c.width,c.height).data;let sum=0,nonzero=0,opaque=0;for(let i=0;i<a.length;i+=4){sum+=a[i]/255;if(a[i])nonzero++;if(a[i]===255)opaque++;}p.flag=0;r.draw(...last);await r.device.queue.onSubmittedWorkDone();return{mean:sum/(a.length/4),nonzero,opaque,pixels:a.length/4,method:'Actual forestResolvedCover rendered under temporary test-only uniform901, restored after readback'};});\n     `;
code=code.replace('entry.frames.push({zoom:z.id,tag,file,sha256:digest(png),dimensions,imageDimensions,',measure+`entry.frames.push({coverageMeasurement,routeOverride:${JSON.stringify({variant,sha256:hash(injection)})},zoom:z.id,tag,file,sha256:digest(png),dimensions,imageDimensions,`);
code=code.replace('const codeIdentity=identity(),codeHash=digest(JSON.stringify(codeIdentity));',`const codeIdentity=identity(),codeHash=digest(JSON.stringify(codeIdentity));assert.equal(codeHash,${JSON.stringify(receipt.productionCodeHash)},'Production changed from2b6; stop');`);
function run(){const child=new Module(runner,module);child.filename=runner;child.paths=Module._nodeModulePaths(__dirname);child._compile(code,runner);}process.argv=process.argv.filter(a=>!['--run','--freeze'].includes(a));process.argv.push('--freeze');run();process.argv.pop();process.argv.push('--run');run();



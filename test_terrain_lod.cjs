const assert=require('node:assert/strict');
const {plan,refinementLod,MAX_LOD,MAX_METRES_PER_PIXEL}=require('./terrain_lod.js');
const parent={key:'parent',lod:3,tx:-1,ty:-1};
const fine={key:'historic',lod:0,tx:-1,ty:-1};
const exact={key:'exact',lod:2,tx:-1,ty:-1};
const span=256*30*4,bounds=[-2*span,-2*span,0,0];
const before=plan([parent,fine],2,bounds);
assert.equal(before.length,4);assert.ok(before.every(x=>x.tile===parent&&x.fallback));
assert.deepEqual(before.at(-1).uv,[.5,.5,.5,.5]);
const after=plan([parent,fine,exact],2,bounds);
assert.equal(after.length,4);assert.equal(after.filter(x=>x.tile===exact).length,1);
assert.ok(!after.some(x=>x.tile===fine));
assert.deepEqual(after.at(-1).uv,[0,0,1,1]);
assert.equal(plan([fine],2,bounds).length,0);
const last={key:'last',lod:0,tx:2604,ty:0};
const clipped=plan([last],0,[19998000,0,20002000,7680],{worldBounds:[-20000000,-10000000,20000000,10000000]});
assert.equal(clipped.length,1);
assert.equal(clipped[0].bounds[2],20000000);
assert.equal(clipped[0].bounds[0],2604*7680);
assert.equal(clipped[0].uv[2],(20000000-2604*7680)/7680);
assert.deepEqual(plan([last],0,[20000001,0,20010000,7680],{worldBounds:[-20000000,-10000000,20000000,10000000]}),[]);
assert.equal(MAX_LOD,11);
assert.equal(MAX_METRES_PER_PIXEL,61440);
assert.equal(plan([{lod:12,tx:0,ty:0}],11,[0,0,7680,7680]).length,0);
// A teleport refines the missing region in order; a hot native revisit skips
// every ancestor. Negative coordinates must map to the same parent footprint.
const tiny=[-7600,-7600,-100,-100],levels=[];
assert.equal(refinementLod(levels,0,tiny),4);
for(let lod=4;lod>=1;lod--){
  levels.push({lod,tx:-1,ty:-1});
  assert.equal(refinementLod(levels,0,tiny),lod===4?3:0);
}
levels.push({lod:0,tx:-1,ty:-1});
assert.equal(refinementLod([levels.at(-1)],0,tiny),0);
assert.equal(refinementLod([{lod:3,tx:-1,ty:-1}],2,tiny),2);
assert.equal(refinementLod([{lod:3,tx:0,ty:0}],2,tiny),4);
assert.equal(refinementLod([],3,tiny),4);
assert.equal(refinementLod([{lod:4,tx:-1,ty:-1,presented:false}],3,tiny),3);
assert.equal(refinementLod([{lod:4,tx:-1,ty:-1,presented:true}],3,tiny),3);
assert.equal(refinementLod([],5,tiny),5);
assert.equal(refinementLod([],0,[20e6,0,21e6,7680],{worldBounds:[-20e6,-10e6,20e6,10e6]}),0);
const protectedBounds=[-7000,-7000,-200,-200];
assert.equal(refinementLod([],0,protectedBounds,{protectedBounds}),0);
const protectedPlan=plan([parent],0,tiny,{protectedBounds});
assert(protectedPlan.every(p=>p.bounds[2]<=protectedBounds[0]||p.bounds[0]>=protectedBounds[2]||p.bounds[3]<=protectedBounds[1]||p.bounds[1]>=protectedBounds[3]));
assert(plan([fine],0,tiny,{protectedBounds}).some(p=>p.tile===fine),'new native tiles must replace the protected image');
console.log('LOD uniforme, substitution parent et coordonnées négatives : OK');

const latentPatch={lod:1,source_lod:3,tx:-1,ty:-1,presented:true};
const decoderParent={lod:2,tx:-1,ty:-1,presented:true};
assert.equal(plan([latentPatch,decoderParent],0,tiny)[0].tile,decoderParent,'better source wins over finer display geometry');
assert.equal(refinementLod([latentPatch],1,tiny),1,'source3 satisfies intermediate barrier, never final cache');
assert(plan([latentPatch],1,tiny)[0].fallback);
assert.equal(refinementLod([{...latentPatch,presented:false}],1,tiny),1);
assert.equal(plan([latentPatch],2,tiny).length,0,'fine historical geometry cannot leak when zooming out');
assert(plan([latentPatch],1,tiny,{protectedBounds}).every(p=>p.bounds[2]<=protectedBounds[0]||p.bounds[0]>=protectedBounds[2]||p.bounds[3]<=protectedBounds[1]||p.bounds[1]>=protectedBounds[3]));

assert.equal(refinementLod([{lod:4,tx:-1,ty:-1,presented:true}],0,tiny,{latentPreview:false}),0,'default coarse coverage schedules native without duplicate latent inference');
assert.equal(refinementLod([],0,tiny,{latentPreview:false}),4);
assert.equal(refinementLod([{lod:4,tx:-1,ty:-1,presented:false}],0,tiny,{latentPreview:false}),0);

// Continuous LOD displays each ready descendant while retaining parent coverage
// in the unfinished regions. The patches must cover the area exactly once.
const adaptiveParent={lod:2,tx:-1,ty:-1},child={lod:1,tx:-1,ty:-1},grandchild={lod:0,tx:-1,ty:-1};
const adaptiveBounds=[-span,-span,0,0];
const adaptive=plan([adaptiveParent,child,grandchild],2,adaptiveBounds,{minLod:0});
assert(adaptive.some(p=>p.tile===adaptiveParent));
assert(adaptive.some(p=>p.tile===child));
assert(adaptive.some(p=>p.tile===grandchild));
assert.equal(adaptive.reduce((area,p)=>area+(p.bounds[2]-p.bounds[0])*(p.bounds[3]-p.bounds[1]),0),span*span);
for(let i=0;i<adaptive.length;i++)for(let j=i+1;j<adaptive.length;j++){
  const a=adaptive[i].bounds,b=adaptive[j].bounds;
  assert(!(a[0]<b[2]&&a[2]>b[0]&&a[1]<b[3]&&a[3]>b[1]),'adaptive patches must not overlap');
}
assert.deepEqual(adaptive.find(p=>p.tile===grandchild).uv,[0,0,1,1]);
assert.deepEqual(adaptive.find(p=>p.tile===adaptiveParent).uv,[0,0,.5,.5]);
assert(plan([adaptiveParent,child,grandchild],2,adaptiveBounds,{minLod:1}).every(p=>p.tile!==grandchild),'depth limits visible refinement');
assert(plan([adaptiveParent,child,grandchild],2,adaptiveBounds).every(p=>p.tile===adaptiveParent),'disabling continuous LOD restores camera resolution');
const adaptiveClipped=plan([adaptiveParent,child,grandchild],2,adaptiveBounds,{minLod:0,worldBounds:[-span,-span,-100,-100]});
assert(adaptiveClipped.every(p=>p.bounds[2]<=-100&&p.bounds[3]<=-100));
console.log('Continuous LOD displays mixed ready depths with complete non-overlapping coverage: OK');

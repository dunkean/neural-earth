const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
const {readRepositoryFile} = require(NEURAL_EARTH_ROOT + '/tools/repository-files.cjs');
const assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm');
const context=vm.createContext({window:{TerrainStyles:JSON.parse(readRepositoryFile('terrain_styles.json','utf8'))},Float32Array,Math,Number});
vm.runInContext(readRepositoryFile('terrain_style_rendering.js','utf8'),context);
const {buildContours}=context.window.TerrainStyleRendering;
const rendering=context.window.TerrainStyleRendering;
assert.equal(rendering.get().density,25);
assert.equal(rendering.densityFromSlider(0),3.125);
assert.equal(rendering.densityFromSlider(3),25,'previous minimum is at the slider midpoint');
assert.equal(rendering.densityFromSlider(6),200);
for(const density of [3.125,12.5,25,50,100,200])assert.equal(rendering.densityFromSlider(rendering.densityToSlider(density)),density);
for(const [mpp,oldInterval]of [[30,20],[100,50],[600,100],[3000,200],[15000,500]]){
  assert.equal(rendering.autoInterval(mpp),4*oldInterval,'default density uses the previous minimum at every zoom');
  assert.equal(rendering.autoInterval(mpp,3.125),Math.min(10000,32*oldInterval),'left endpoint spaces contours eight times farther, within the interval limit');
  assert.equal(rendering.autoInterval(mpp,100),oldInterval);
  assert.equal(rendering.autoInterval(mpp,25),4*oldInterval);
  assert.equal(rendering.autoInterval(mpp,200),oldInterval/2);
}
const field=(w,h,fn)=>Float32Array.from({length:w*h},(_,i)=>fn(i%w,Math.floor(i/w)));
const segments=data=>Array.from({length:data.length/5},(_,i)=>Array.from(data.slice(i*5,i*5+5)));

// An exact grid-edge level must exist once, including on a steep slope.
for(const slope of [2,80]){
  const w=40,h=30,halo=2,values=field(w,h,x=>100+(x-10)*slope);
  const lines=segments(buildContours(values,{width:w,height:h,halo},100));
  assert(lines.length>0);
  const at100=lines.filter(l=>Math.abs(l[0]-(10+.5-halo)/(w-2*halo))<1e-6);
  assert.equal(at100.length,h-2*halo+1,'complete contour, even when gradient exceeds half the interval');
  assert(at100.every(l=>l[0]===l[2]));
}
// Closed hill contours form loops, with no missing or branching vertices.
const hill=field(65,65,(x,y)=>2000-5*((x-32)**2+(y-32)**2));
const loops=segments(buildContours(hill,{width:65,height:65},100)),degree=new Map();
for(const line of loops)for(const [x,y]of [[line[0],line[1]],[line[2],line[3]]]){const key=x.toFixed(6)+','+y.toFixed(6);degree.set(key,(degree.get(key)||0)+1);}
assert(loops.length>100);assert([...degree.values()].every(n=>n===2),'closed rings have no gaps');
for(const height of [-100,0,100,500])assert.equal(buildContours(new Float32Array(64).fill(height),{width:8,height:8},100).length,0,'constant plateau stays clear');

// Shared halos produce exactly the same world positions at a tile seam.
const width=20,halo=2,inner=width-2*halo;
const seamPoints=[];
for(const tx of [0,1]){
  const values=field(width,width,(x,y)=>(tx*inner+x-halo)*3+(y-halo)*7+51);
  const lines=segments(buildContours(values,{width,height:width,halo},50));
  const pts=[];
  for(const l of lines)for(const [u,v]of [[l[0],l[1]],[l[2],l[3]]]){
    const x=tx*inner+u*inner,y=v*inner;
    if(Math.abs(x-inner)<=.5001){pts.push(x.toFixed(4)+','+y.toFixed(4));}
  }
  seamPoints.push([...new Set(pts)].sort());
}
assert.deepEqual(seamPoints[0],seamPoints[1]);
const saddle=segments(buildContours(Float32Array.from([2,0,0,2]),{width:2,height:2},1));
assert.equal(saddle.length,2);assert(saddle.every(l=>l.slice(0,4).every(Number.isFinite)));
const ramp=field(101,8,x=>x*10+1);
assert.equal(buildContours(ramp,{width:101,height:8},rendering.autoInterval(600,25)).length,
  buildContours(ramp,{width:101,height:8},rendering.autoInterval(600)).length,'default preserves the previous minimum on actual terrain');
assert(buildContours(ramp,{width:101,height:8},rendering.autoInterval(600,3.125)).length<
  buildContours(ramp,{width:101,height:8},rendering.autoInterval(600)).length,'left half permits fewer actual lines');
console.log('Physical isolines: exact grid levels, steep slopes, closed rings, flat fields, saddles and shared halos: OK');

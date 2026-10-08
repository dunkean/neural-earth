// Adapted from city_generator/rust/bridge/terrainErosion.wgsl (GPL-3.0-only).
// Rectangular periodic atlas, latitude metrics, fixed ocean; see PROVENANCE.md.
// Full FP32 regional erosion. Dispatches are global synchronization boundaries.
struct Params { values: array<vec4<f32>, 12> }
struct Node {
  h: f32, uplift: f32, acc: f32, receiver: i32,
  flood: array<vec2<f32>,2>, flow: vec4<f32>, state: array<vec2<f32>,2>,
  thermal: vec2<f32>, nextH: f32, routingPriority: f32,
  parent: atomic<u32>, bottom: atomic<u32>, spill: atomic<u32>, pour: atomic<u32>,
  breached: atomic<u32>,
}
struct Control {
  flags: array<atomic<u32>,2>, errors: atomic<u32>, padding: u32,
  histogram: array<atomic<u32>,256>,
  prefix: u32, quantileIndex: u32, selected: u32, pad: u32,
  p99: f32, p1: f32,
  solveSteps:u32, maxSolveSteps:array<u32,4>,
}
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> sourceHeight: array<f32>;
@group(0) @binding(3) var<storage, read_write> nodes: array<Node>;
@group(0) @binding(5) var<storage, read_write> control: Control;
@group(0) @binding(6) var<storage, read_write> outputHeight: array<f32>;
@group(0) @binding(7) var<storage, read_write> tileFlags: array<atomic<u32>>;
const directions = array<vec2<i32>,8>(vec2<i32>(1,0),vec2<i32>(1,1),vec2<i32>(0,1),vec2<i32>(-1,1),vec2<i32>(-1,0),vec2<i32>(-1,-1),vec2<i32>(0,-1),vec2<i32>(1,-1));
const distances = array<f32,8>(1.0,1.4142135623730951,1.0,1.4142135623730951,1.0,1.4142135623730951,1.0,1.4142135623730951);
fn size() -> u32 {return u32(p.values[11].x);}
fn height() -> u32 {return u32(p.values[11].y);}
fn count() -> u32 {return size()*height();}
fn tileCount() -> u32 {return ((size()+15u)/16u)*((height()+15u)/16u);}
fn wrapX(x:i32)->i32 {let w=i32(size());return ((x%w)+w)%w;}
fn offsetX(x:i32)->i32 {let w=i32(size());return select(select(x,x-w,x>w/2),x+w,x< -w/2);}
fn edge(i:u32)->bool {return i/size()==0u || i/size()==height()-1u || sourceHeight[i]<=0.0;}
fn neighbor(i:u32,d:u32)->i32 {
 let q=vec2<i32>(i32(i%size()),i32(i/size()))+directions[d];
 if(q.y<0 || q.y>=i32(height())){return -1;}return q.y*i32(size())+wrapX(q.x);
}
fn latitudeScale(i:u32)->f32 {return max(0.01,sin((f32(i/size())+0.5)*3.141592653589793/f32(height())));}
fn direction(i:u32,d:u32)->vec2<f32>{return vec2<f32>(f32(directions[d].x)*latitudeScale(i),f32(directions[d].y));}
fn distance(i:u32,d:u32)->f32{return length(direction(i,d));}
fn receiverDistance(i:u32,a:u32)->f32 {return length(vec2<f32>(f32(offsetX(i32(a%size())-i32(i%size())))*latitudeScale(i),f32(i32(a/size())-i32(i/size()))));}
fn lower(a:vec2<f32>,b:vec2<f32>)->bool {return a.x<b.x || (a.x==b.x && a.y<b.y);}
@compute @workgroup_size(1)
fn start(){atomicStore(&control.flags[0],1u);atomicStore(&control.flags[1],1u);control.solveSteps=0u;}
@compute @workgroup_size(1)
fn clearNext(){let phase=u32(p.values[6].x);if(atomicLoad(&control.flags[phase])!=0u){control.solveSteps++;}atomicStore(&control.flags[1u-phase],0u);}
@compute @workgroup_size(1)
fn solveStats(){let mode=u32(p.values[6].y);control.maxSolveSteps[mode]=max(control.maxSolveSteps[mode],control.solveSteps);}
var<workgroup> tileA:array<vec2<f32>,324>;
var<workgroup> tileB:array<vec2<f32>,324>;
var<workgroup> tileH:array<f32,324>;
var<workgroup> tileCost:array<f32,324>;
var<workgroup> tileFlow:array<vec4<f32>,324>;
var<workgroup> tileActive:u32;
var<workgroup> tileChanged:atomic<u32>;
var<workgroup> roundChanged:atomic<u32>;
override SOLVE_MODE:u32=0u;
fn tileState(k:u32,step:u32)->vec2<f32>{if((step&1u)==0u){return tileA[k];}return tileB[k];}
// 16x16 interior plus a one-cell halo. Halo is immutable for one dispatch.
@compute @workgroup_size(16,16)
fn solveTile(@builtin(local_invocation_id) local:vec3<u32>,@builtin(workgroup_id) group:vec3<u32>){
  let lane=local.y*16u+local.x;let phase=u32(p.values[6].x);let mode=SOLVE_MODE;
  let tiles=(size()+15u)/16u;let tileId=group.y*tiles+group.x;let epoch=control.solveSteps;
  if(lane==0u){
    tileActive=atomicLoad(&control.flags[phase]);atomicStore(&tileChanged,0u);
    if(tileActive!=0u){
      tileActive=select(0u,1u,atomicLoad(&tileFlags[phase*tileCount()+tileId])==epoch);
    }
  }
  let run=workgroupUniformLoad(&tileActive);if(run==0u){return;}
  if(mode!=0u){
    let xy=group.xy*16u+local.xy;let valid=all(xy<vec2<u32>(size(),height()));let i=xy.y*size()+xy.x;
    var state=vec2<f32>(0.0,1.0);if(valid){state=nodes[i].state[phase];}
    if(state.y==0.0){atomicStore(&tileChanged,1u);}workgroupBarrier();
    if(lane==0u){tileActive=atomicLoad(&tileChanged);}
    let unresolved=workgroupUniformLoad(&tileActive);
    if(unresolved==0u){
      // A completed graph cell never changes again. Synchronize the opposite
      // global phase before skipping this tile's halo loads and local rounds.
      if(valid){nodes[i].state[1u-phase]=state;}return;
    }
    if(lane==0u){atomicStore(&tileChanged,0u);}workgroupBarrier();
  }
  let base=vec2<i32>(group.xy*16u)-1;let n=i32(size());
  for(var k=lane;k<324u;k+=256u){
    let xy=base+vec2<i32>(i32(k%18u),i32(k/18u));
    var state=vec2<f32>(3.4e38,1e9);var h=3.4e38;var cost=1.0;var flow=vec4<f32>(-1.0);
    if(xy.y>=0 && xy.y<i32(height())){let i=u32(xy.y*n+wrapX(xy.x));if(mode==0u){h=nodes[i].h;cost=nodes[i].routingPriority;state=nodes[i].flood[phase];}else{if(mode==1u){flow=nodes[i].flow;}state=nodes[i].state[phase];}}
    tileA[k]=state;tileB[k]=state;
    if(mode==0u){tileH[k]=h;tileCost[k]=cost;}else if(mode==1u){tileFlow[k]=flow;}
  }
  workgroupBarrier();
  let xy=group.xy*16u+local.xy;let valid=all(xy<vec2<u32>(size(),height()));let i=xy.y*size()+xy.x;let k=(local.y+1u)*18u+local.x+1u;
  let original=tileA[k];
  var incisionFactor=0.0;var ownFlow=vec4<f32>(-1.0);var ownHeight=0.0;
  if(valid && mode>=2u){incisionFactor=nodes[i].nextH;ownFlow=nodes[i].flow;ownHeight=nodes[i].h;}
  var value=original;
  for(var step=0u;step<16u;step++){
    if((step&3u)==0u){if(lane==0u){atomicStore(&roundChanged,0u);}workgroupBarrier();}
    let previous=value;
    if(valid){
      if(mode==0u){
        if(!edge(i)){
          for(var d=0u;d<8u;d++){let j=u32(i32(k)+directions[d].y*18+directions[d].x);let a=tileState(j,step);let level=max(tileH[k],a.x);
            if(level>value.x){continue;}
            // Prefer existing low terrain inside a filled basin. A positive
            // seeded cost breaks equal-level routing ties without raising h.
            var distance=0.0;
            if(tileH[k]<=a.x){
              if(level==value.x && a.y>=value.y){continue;}
              let depthCost=1.0+8.0*exp(-max(0.0,level-(tileH[k]+tileH[j])*0.5)/max(0.1,p.values[1].x*0.1));
              let tieCost=(tileCost[k]+tileCost[j])*0.5;
              distance=a.y+distances[d]*depthCost*tieCost;
            }
            let candidate=vec2<f32>(level,distance);if(lower(candidate,value)){value=candidate;}}
        }
      } else if(value.y==0.0){
        var ready=true;var result=latitudeScale(i);
        if(mode==1u){
          for(var d=0u;d<8u;d++){let j=u32(i32(k)+directions[d].y*18+directions[d].x);let f=tileFlow[j];var weight=0.0;if(f.x==f32(i)){weight+=1.0-f.z;}if(f.y==f32(i)){weight+=f.z;}if(weight>0.0){let state=tileState(j,step);if(state.y==0.0){ready=false;break;}result+=state.x*weight;}}
        } else {
          let f=ownFlow;result=ownHeight;
          if(f.x>=0.0){
            let a=u32(f.x);let offset=vec2<i32>(offsetX(i32(a%size())-i32(xy.x)),i32(a/size())-i32(xy.y));let j=u32(i32(k)+offset.y*18+offset.x);let state=tileState(j,step);var low=state.x;ready=state.y!=0.0;
            if(f.y>=0.0 && f.z>0.0){let b=u32(f.y);let off=vec2<i32>(offsetX(i32(b%size())-i32(xy.x)),i32(b/size())-i32(xy.y));let s=u32(i32(k)+off.y*18+off.x);let other=tileState(s,step);low=(1.0-f.z)*low+f.z*other.x;ready=ready && other.y!=0.0;}
            // Marine receivers drain land at the sea datum, not at the
            // (possibly very deep) ocean floor. Preserve signed sea samples.
            if(sourceHeight[i]>0.0){low=max(0.0,low);}
            result=min(result,(result+incisionFactor*low)/(1.0+incisionFactor));
          }
        }
        if(ready){value=vec2<f32>(result,1.0);}
      }
    }
    // Identical Jacobi rounds, alternating the two shared arrays. Both halos
    // remain immutable. A fixed point over four rounds needs no further work.
    if(any(value!=previous)){atomicStore(&roundChanged,1u);}
    if((step&1u)==0u){tileB[k]=value;}else{tileA[k]=value;}workgroupBarrier();
    if((step&3u)==3u){
      if(lane==0u){tileActive=atomicLoad(&roundChanged);}
      let changed=workgroupUniformLoad(&tileActive);if(changed==0u){break;}
    }
  }
  if(valid){
    if(mode==0u){nodes[i].flood[1u-phase]=value;}else{nodes[i].state[1u-phase]=value;}
    if(any(value!=original)){
      // Wake self to continue propagation and synchronize both global phases.
      // Only changed halo cells can affect another tile: edge/corner bits name
      // the corresponding destinations in a row-major 3x3 neighborhood.
      var mask=16u;
      if(local.x==0u){mask|=8u;}if(local.x==15u || xy.x==size()-1u){mask|=32u;}
      if(local.y==0u){mask|=2u;}if(local.y==15u){mask|=128u;}
      if(local.x==0u && local.y==0u){mask|=1u;}if((local.x==15u || xy.x==size()-1u) && local.y==0u){mask|=4u;}
      if(local.x==0u && local.y==15u){mask|=64u;}if((local.x==15u || xy.x==size()-1u) && local.y==15u){mask|=256u;}
      atomicOr(&tileChanged,mask);
    }
  }
  // Convergence is a Boolean OR: one device-wide write per changed tile avoids
  // hundreds of thousands of contenders on the same global flag.
  workgroupBarrier();
  if(lane==0u && atomicLoad(&tileChanged)!=0u){
    atomicStore(&control.flags[1u-phase],1u);
    let mask=atomicLoad(&tileChanged);
    for(var dy=-1;dy<=1;dy++){for(var dx=-1;dx<=1;dx++){
      let xy=vec2<i32>(group.xy)+vec2<i32>(dx,dy);let bit=u32((dy+1)*3+dx+1);
      if((mask&(1u<<bit))!=0u && xy.y>=0 && xy.y<i32((height()+15u)/16u)){
        atomicStore(&tileFlags[(1u-phase)*tileCount()+u32(xy.y)*tiles+u32(((xy.x%i32(tiles))+i32(tiles))%i32(tiles))],epoch+1u);
      }
    }}
  }
}
@compute @workgroup_size(256)
fn receivers(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=count()){return;}let current=nodes[i].flood[0];
  var best=0.0;var bestFlat=0.0;var receiver=-1;var lowest=3.4e38;
  if(!edge(i)){for(var d=0u;d<8u;d++){
    let r=neighbor(i,d);if(r<0){continue;}let other=nodes[u32(r)].flood[0];if(!lower(other,current)){continue;}
    let drop=(current.x-other.x)/distance(i,d);let flatDrop=(current.y-other.y)/distance(i,d);let h=nodes[u32(r)].h;
    if(drop>best || (drop==0.0 && best==0.0 && (flatDrop>bestFlat || (flatDrop==bestFlat && h<lowest)))){
      receiver=r;best=drop;bestFlat=flatDrop;lowest=h;
    }
  }}
  nodes[i].receiver=receiver;
}
@compute @workgroup_size(256)
fn flow(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;if(i>=count()){return;}let r=nodes[i].receiver;var f=vec4<f32>(f32(r),-1.0,0.0,1.0);
  if(r>=0){let a=u32(r);f.w=receiverDistance(i,a);let center=nodes[i].flood[0];let outlet=nodes[a].flood[0];let flat=center.x==outlet.x;
    var best=select(center.x-outlet.x,center.y-outlet.y,flat)/f.w;
    for(var d=0u;d<8u;d++){let ar=neighbor(i,d);let br=neighbor(i,(d+1u)&7u);if(ar<0 || br<0){continue;}
      let af=nodes[u32(ar)].flood[0];let bf=nodes[u32(br)].flood[0];
      if(!lower(af,center) || !lower(bf,center)){continue;}
      if(flat && (af.x!=center.x || bf.x!=center.x)){continue;}
      let u=direction(i,d);let v=direction(i,(d+1u)&7u);
      let da=select(center.x-af.x,center.y-af.y,flat);let db=select(center.x-bf.x,center.y-bf.y,flat);let det=u.x*v.y-u.y*v.x;
      let g=vec2<f32>((da*v.y-db*u.y)/det,(db*u.x-da*v.x)/det);let ca=u.x*g.y-u.y*g.x;let cb=g.x*v.y-g.y*v.x;if(ca<0.0 || cb<0.0){continue;}let slope=length(g);if(slope<=best){continue;}let t=clamp(atan2(ca,dot(u,g))/atan2(det,dot(u,v)),0.0,1.0);best=slope;f=vec4<f32>(f32(ar),f32(br),t,(1.0-t)*distance(i,d)+t*distance(i,(d+1u)&7u));
    }
  }nodes[i].flow=f;
}
@compute @workgroup_size(256)
fn graphInit(@builtin(global_invocation_id) id:vec3<u32>){
  let i=id.x;let tiles=(size()+15u)/16u;let tileCount=tileCount();
  if(i<tileCount){atomicStore(&tileFlags[i],1u);atomicStore(&tileFlags[tileCount+i],0u);}
  if(i>=count()){return;}let mode=u32(p.values[6].y);
  let value=select(latitudeScale(i),nodes[i].h,mode>=2u);nodes[i].state[0]=vec2<f32>(value,0.0);nodes[i].state[1]=nodes[i].state[0];
  // Heights, accumulation and receivers remain fixed throughout this solve.
  // Cache the coefficient in scratch consumed only later by thermal/diffusion.
  var factor=0.0;let f=nodes[i].flow;
  if(mode>=2u && f.x>=0.0){
    factor=min(6.0,p.values[5].z*min(sqrt(nodes[i].acc),4.0*sqrt(0.004)*p.values[0].w*2400.0/p.values[1].x)/f.w);
    var originalLow=nodes[u32(f.x)].h;
    if(f.y>=0.0 && f.z>0.0){originalLow=mix(originalLow,nodes[u32(f.y)].h,f.z);}
    let slope=max(0.0,(nodes[i].h-originalLow)/(p.values[1].x*f.w));
    let response=slope*slope/(slope*slope+0.04);
    if(mode==3u){
      let threshold=p.values[3].w*0.10;
      let localResponse=0.08+0.92*slope*slope/(slope*slope+threshold*threshold);
      factor=min(4.0,p.values[5].z*min(sqrt(nodes[i].acc),4.0*sqrt(0.004)*p.values[0].w*2400.0/p.values[1].x)*localResponse/f.w);
    }else{factor*=response;}
  }
  nodes[i].nextH=factor;
}
@compute @workgroup_size(256)
fn graphFinish(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=count()){return;}let s=nodes[i].state[0];if(s.y==0.0){atomicAdd(&control.errors,1u);}if(p.values[6].y==1.0){nodes[i].acc=s.x;}else{nodes[i].h=select(s.x,0.0,edge(i) && p.values[6].y==2.0);}}
@compute @workgroup_size(256)
fn thermalFlux(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=count()){return;}var total=0.0;var maximum=0.0;if(!edge(i)){for(var d=0u;d<8u;d++){let r=neighbor(i,d);if(r<0 || sourceHeight[u32(r)]<=0.0){continue;}let excess=max(0.0,nodes[i].h-nodes[u32(r)].h-p.values[3].w*p.values[6].z*p.values[1].x*distance(i,d));total+=excess;maximum=max(maximum,excess);}}nodes[i].thermal=vec2<f32>(0.1*p.values[4].z*maximum,total);}
@compute @workgroup_size(256)
fn thermalApply(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i>=count()){return;}var change=-nodes[i].thermal.x;for(var d=0u;d<8u;d++){let r=neighbor(i,d);if(r<0){continue;}let f=nodes[u32(r)].thermal;if(f.y>0.0){let excess=max(0.0,nodes[u32(r)].h-nodes[i].h-p.values[3].w*p.values[6].z*p.values[1].x*distance(i,d));change+=f.x*excess/f.y;}}nodes[i].nextH=select(max(0.0,nodes[i].h+change),nodes[i].h,edge(i));}
@compute @workgroup_size(256)
fn copyHeight(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i<count()){nodes[i].h=nodes[i].nextH;}}
@compute @workgroup_size(256)
fn diffuse(@builtin(global_invocation_id) id:vec3<u32>){
 let i=id.x;if(i>=count()){return;}var value=nodes[i].h;
 if(!edge(i)){var average=0.0;var total=0.0;for(var d=0u;d<8u;d++){let r=neighbor(i,d);if(r<0 || sourceHeight[u32(r)]<=0.0){continue;}let w=select(0.2,0.05,(d&1u)!=0u);average+=nodes[u32(r)].h*w;total+=w;}
 if(total>0.0){let weight=p.values[6].w*select(1.0,0.75,nodes[i].acc>=f32(count())*0.001);value=mix(value,average/total,weight);}}
 nodes[i].nextH=value;
}
@compute @workgroup_size(256)
fn loadHeight(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i<count()){nodes[i].h=sourceHeight[i];nodes[i].uplift=sourceHeight[i];nodes[i].acc=latitudeScale(i);}}
@compute @workgroup_size(256)
fn surfaceFlowInit(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i<count()){nodes[i].flood[0]=vec2<f32>(nodes[i].h,0.0);}}
@compute @workgroup_size(256)
fn packHeight(@builtin(global_invocation_id) id:vec3<u32>){let i=id.x;if(i<count()){outputHeight[i]=select(max(0.0,nodes[i].h),sourceHeight[i],sourceHeight[i]<=0.0);}}

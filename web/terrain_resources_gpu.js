/* Resource layers evaluated per pixel in the renderer's shade pass (WGSL).
 * Mirror: backend/terrain_resource_gpu.py (operation for operation, float32,
 * uint32 wrapping hashes). Inputs: the tile DEM, its Gaussian pair and water
 * already on the GPU, plus one package per world and mode (LOD-10 fields as
 * half floats, zone thresholds, deposits sorted in LOD-10 buckets).
 * Method and tolerances: docs/RESOURCE_LAYERS.md section 16. */
(function(root){
  const PREFIX='resources-';
  const KINDS={ramp:0,log:1,mix:2,classes:3,cyclic:4,points:5,stock:6};
  const PARAM_BYTES=(8+3+20)*16;
  let order=[];
  const isMode=mode=>typeof mode==='string'&&mode.startsWith(PREFIX);
  // Stable numbers below the existing ranges (Render v2 uses -8..-12).
  const number=mode=>{let i=order.indexOf(mode);if(i<0){order.push(mode);i=order.length-1;}return -100-i;};
  const isNumber=n=>n<=-100;
  const core=`
    struct ResParams { grid:vec4<f32>, info:vec4<f32>, range:vec4<f32>, prof0:vec4<f32>, prof1:vec4<f32>, zones:vec4<f32>, zones2:vec4<f32>, seeds:vec4<u32>, order:array<vec4<u32>,3>, palette:array<vec4<f32>,20> }
    struct ResDeposit { ij:vec4<i32>, dim:vec4<f32>, extra:vec4<f32> }
    @group(0) @binding(11) var resField:texture_2d_array<f32>;
    @group(0) @binding(12) var resTau:texture_2d_array<f32>;
    @group(0) @binding(13) var<uniform> res:ResParams;
    @group(0) @binding(14) var<storage,read> resDeposits:array<ResDeposit>;
    @group(0) @binding(15) var<storage,read> resBuckets:array<vec2<u32>>;
    struct RLat { i:i32, f:f32 }
    fn rLowbias(v:u32)->u32{var x=v;x^=x>>16u;x*=0x7feb352du;x^=x>>15u;x*=0x846ca68bu;x^=x>>16u;return x;}
    fn rHash(i:i32,j:i32,salt:u32)->u32{return rLowbias((bitcast<u32>(i)*0x9e3779b1u)^rLowbias(bitcast<u32>(j)+0x85ebca77u)^salt);}
    fn rSalt(layer:u32,o:i32,comp:u32)->u32{return rLowbias(res.seeds.x+layer*0x632be5abu+u32(o+64)*0x27d4eb2du+comp*0x165667b1u);}
    fn rNode(i:i32,j:i32,salt:u32)->f32{return f32(rHash(i,j,salt)>>8u)*(1./16777216.)*2.-1.;}
    fn rFade(t:f32)->f32{return t*t*t*(t*(t*6.-15.)+10.);}
    fn rLattice(g:i32,lod:i32,o:i32)->RLat{
      let k=o-lod;if(k<=0){return RLat(g,.5);}
      return RLat(g>>u32(k),(f32(g&((1<<u32(k))-1))+.5)*(1./f32(1<<u32(k))));
    }
    fn rOctave(gi:i32,gj:i32,lod:i32,o:i32,salt:u32,w:vec2<f32>)->f32{
      let a=rLattice(gi,lod,o);let b=rLattice(gj,lod,o);
      let q=vec2<f32>(a.f,b.f)+w;let fl=floor(q);
      let i=a.i+i32(fl.x);let j=b.i+i32(fl.y);let f=q-fl;
      let u=vec2<f32>(rFade(f.x),rFade(f.y));
      let n0=rNode(i,j,salt)*(1.-u.x)+rNode(i+1,j,salt)*u.x;
      let n1=rNode(i,j+1,salt)*(1.-u.x)+rNode(i+1,j+1,salt)*u.x;
      return n0*(1.-u.y)+n1*u.y;
    }
    fn rLam(o:i32)->f32{return 30.*exp2(f32(o));}
    fn rWarp(gi:i32,gj:i32,lod:i32,layer:u32,lo:i32,hi:i32,amp:f32)->vec2<f32>{
      var w=vec2<f32>(0.);
      for(var o=max(lo,lod+2);o<=hi;o++){let a=amp*rLam(o);w+=a*vec2<f32>(rOctave(gi,gj,lod,o,rSalt(layer,o,1u),vec2<f32>(0.)),rOctave(gi,gj,lod,o,rSalt(layer,o,2u),vec2<f32>(0.)));}
      return w;
    }
    fn rFbm(gi:i32,gj:i32,lod:i32,layer:u32,lo:i32,hi:i32,decay:f32,w:vec2<f32>)->f32{
      var total=0.;var norm=0.;
      for(var o=max(lo,lod+2);o<=hi;o++){let a=exp2(decay*f32(o-hi));total+=a*rOctave(gi,gj,lod,o,rSalt(layer,o,0u),w/rLam(o));norm+=a*a;}
      return total/(max(sqrt(norm),1e-6)*.45);
    }
    // Strike-aligned fbm: four integer orientations, a axis stretched x2, blended by strike.
    fn rOctaveS(a:i32,b:i32,lod:i32,o:i32,salt:u32,w:vec2<f32>)->f32{
      let la=rLattice(a,lod,o+1);let lb=rLattice(b,lod,o);
      let q=vec2<f32>(la.f+w.x/rLam(o+1),lb.f+w.y/rLam(o));let fl=floor(q);
      let i=la.i+i32(fl.x);let j=lb.i+i32(fl.y);let f=q-fl;
      let u=vec2<f32>(rFade(f.x),rFade(f.y));
      let n0=rNode(i,j,salt)*(1.-u.x)+rNode(i+1,j,salt)*u.x;
      let n1=rNode(i,j+1,salt)*(1.-u.x)+rNode(i+1,j+1,salt)*u.x;
      return n0*(1.-u.y)+n1*u.y;
    }
    fn rFbmAligned(gi:i32,gj:i32,lod:i32,layer:u32,lo:i32,hi:i32,decay:f32,w:vec2<f32>,theta:f32)->f32{
      var dirs=array<vec4<i32>,4>(vec4<i32>(1,0,0,1),vec4<i32>(0,1,-1,0),vec4<i32>(1,1,-1,1),vec4<i32>(1,-1,1,1));
      var offs=array<i32,4>(0,-1,0,0);
      var phi=array<f32,4>(0.,1.5707964,.7853982,2.3561945);
      var norm=0.;for(var o=max(lo,lod+2);o<=hi;o++){let a=exp2(decay*f32(o-hi));norm+=a*a;}
      let scale=max(sqrt(norm),1e-6)*.45;
      var total=0.;var weight2=0.;
      for(var d=0;d<4;d++){
        let wd=max(cos(2.*(theta-phi[d])),0.);
        if(wd<=0.){continue;}
        let c=dirs[d];let a=c.x*gi+c.y*gj;let b=c.z*gi+c.w*gj+offs[d];
        let wv=vec2<f32>(f32(c.x)*w.x+f32(c.y)*w.y,f32(c.z)*w.x+f32(c.w)*w.y);
        var n=0.;
        for(var o=max(lo,lod+2);o<=hi;o++){n+=exp2(decay*f32(o-hi))*rOctaveS(a,b,lod,o,rSalt(layer,o,3u+u32(d)),wv);}
        total+=wd*(n/scale);weight2+=wd*wd;
      }
      return total/sqrt(max(weight2,1e-6));
    }
    fn rCellPos(g:i32,lod:i32,w:f32)->RLat{
      var base:i32;var frac:f32;
      if(lod<=10){let k=10-lod;base=g>>u32(k);frac=(f32(g&((1<<u32(k))-1))+.5)*(1./f32(1<<u32(k)))-.5;}
      else{let pos=(f32(g)+.5)*exp2(f32(lod-10))-.5;base=i32(floor(pos));frac=pos-floor(pos);}
      let q=frac+w*(1./30720.);let fl=floor(q);return RLat(base+i32(fl),q-fl);
    }
    fn rColRow(c:i32,r:i32)->vec2<i32>{let W=i32(res.grid.z);let H=i32(res.grid.w);return vec2<i32>(((c-i32(res.grid.x))%W+W)%W,clamp(r-i32(res.grid.y),0,H-1));}
    fn rFieldAt(layer:i32,c:i32,r:i32)->f32{return textureLoad(resField,rColRow(c,r),layer,0).r;}
    fn rField(layer:i32,gi:i32,gj:i32,lod:i32,w:vec2<f32>)->f32{
      let x=rCellPos(gi,lod,w.x);let y=rCellPos(gj,lod,w.y);
      let a=rFieldAt(layer,x.i,y.i)*(1.-x.f)+rFieldAt(layer,x.i+1,y.i)*x.f;
      let b=rFieldAt(layer,x.i,y.i+1)*(1.-x.f)+rFieldAt(layer,x.i+1,y.i+1)*x.f;
      return a*(1.-y.f)+b*y.f;
    }
    fn rNearest(layer:i32,gi:i32,gj:i32,lod:i32,w:vec2<f32>)->f32{
      let x=rCellPos(gi,lod,w.x);let y=rCellPos(gj,lod,w.y);
      return rFieldAt(layer,x.i+select(0,1,x.f>=.5),y.i+select(0,1,y.f>=.5));
    }
    fn rSmooth(a:f32,b:f32,x:f32)->f32{let t=clamp((x-a)/(b-a),0.,1.);return t*t*(3.-2.*t);}
    // Terrain features: slope (high-passed, degrees), valley, ridge, north, dz, river.
    struct RFeat { slope:f32, valley:f32, ridge:f32, north:f32, dz:f32, river:f32 }
    fn rFeatures(height:f32,g:vec2<f32>,gl:vec2<f32>,small:f32,large:f32,r:f32,river:f32)->RFeat{
      let slope=degrees(atan(length(g)));let broad=degrees(atan(length(gl)));
      let tpi=(small-large)/(6.*r);
      let north=-g.y/max(length(g),1e-6)*rSmooth(1.,6.,slope);
      return RFeat(slope-broad,rSmooth(0.,-.10,tpi),rSmooth(0.,.10,tpi),north,height-large,river);
    }
    fn rTerms(f:RFeat)->f32{
      return clamp(res.prof0.x*f.slope+res.prof0.y*f.valley+res.prof0.z*f.ridge+res.prof0.w*f.north+res.prof1.x*f.dz+res.prof1.y*f.river,-3.,3.);
    }
    fn rContinuous(gi:i32,gj:i32,lod:i32,terms:f32)->f32{
      let layer=res.seeds.y;let w=rWarp(gi,gj,lod,layer*4u+1u,8,10,.25);
      var p:f32;
      if(res.info.w>.5){
        let K=i32(res.info.y);let m=res.range.w*12.-.5;let m0=floor(m);let t=m-m0;
        let a=((i32(m0)%K)+K)%K;let b=(((i32(m0)+1)%K)+K)%K;
        p=rField(a,gi,gj,lod,w)*(1.-t)+rField(b,gi,gj,lod,w)*t;
      }else{p=rField(0,gi,gj,lod,w);}
      let sigma=res.prof1.z;
      let n=rFbm(gi,gj,lod,layer*4u+2u,7,9,.8,vec2<f32>(0.));
      let z=terms*(1.-.75*rSmooth(1.,5.,f32(lod)))+sigma*n;
      if(res.range.z>.5){
        let lo=bitcast<f32>(res.seeds.z);let hi=bitcast<f32>(res.seeds.w);   // physical bounds
        let q=clamp((p-lo)/(hi-lo),1e-4,1.-1e-4);
        return select(lo+(hi-lo)/(1.+exp(-(log(q/(1.-q))+z))),lo,p<=lo);
      }
      return max(p,0.)*exp(z-.5*sigma*sigma);
    }
    // Month-like layers: nearest LOD-10 value after a warp (no blending of months).
    fn rUniform(n:f32)->f32{
      var knots=array<f32,17>(-3.5,-1.5528,-1.1966,-.9383,-.7218,-.5275,-.346,-.1713,0.,.1713,.346,.5275,.7218,.9383,1.1966,1.5528,3.5);
      var k=0;for(var i=1;i<16;i++){if(n>=knots[i]){k=i;}}
      return clamp((f32(k)+(n-knots[k])/(knots[k+1]-knots[k]))/16.,0.,1.);
    }
    // One of the four cells around the sample, with probability = bilinear weight (organic blobs, no blending).
    fn rSelect(gi:i32,gj:i32,lod:i32,w:vec2<f32>)->vec2<i32>{
      let x=rCellPos(gi,lod,0.);let y=rCellPos(gj,lod,0.);
      let u=rUniform(rFbm(gi,gj,lod,res.seeds.y*16u+8u,8,10,.7,w));
      let w00=(1.-x.f)*(1.-y.f);let w10=x.f*(1.-y.f);let w01=(1.-x.f)*y.f;
      let a=w00;let b=w00+w10;let c=w00+w10+w01;
      let dx=select(select(select(1,0,u<c),1,u<b),0,u<a);let dy=select(1,0,u<b);
      return vec2<i32>(x.i+dx,y.i+dy);
    }
    fn rCyclic(gi:i32,gj:i32,lod:i32)->f32{
      let w=rWarp(gi,gj,lod,res.seeds.y*4u+1u,max(lod+1,-5),10,.25);
      let c=rSelect(gi,gj,lod,w);
      return rFieldAt(0,c.x,c.y);
    }
    // Generalised class at small scales: largest parent share at the warped position.
    fn rDominant(gi:i32,gj:i32,lod:i32)->u32{
      let w=rWarp(gi,gj,lod,res.seeds.y*4u+1u,lod+1,11,.15);
      var best=-1.;var out=0u;
      for(var k=0;k<i32(res.info.z);k++){let v=rField(k,gi,gj,lod,w);if(v>best){best=v;out=u32(k);}}
      return out;
    }
    // Realized geology: argmax_k (log E_k + b1 x province noise of its group + b2 x class noise).
    // res.order: per class (belt flag << 4 | group), then per group belt flag; res.zones.yz = (b1, b2).
    fn rRealized(gi:i32,gj:i32,lod:i32)->u32{
      let layer=res.seeds.y;let K=i32(res.info.z);
      let w=rWarp(gi,gj,lod,layer*4u+1u,lod+1,11,.15);
      let theta=.5*atan2(rField(K+1,gi,gj,lod,w),rField(K,gi,gj,lod,w));
      var province:array<f32,4>;
      for(var g=0u;g<4u;g++){
        let salt=layer*64u+48u+g;
        if(res.order[(8u+g)/4u][(8u+g)%4u]>0u){province[g]=rFbmAligned(gi,gj,lod,salt,12,15,1.,w,theta);}
        else{province[g]=rFbm(gi,gj,lod,salt,12,15,1.,w);}
      }
      var best=-1e30;var out=0u;
      for(var k=0;k<K;k++){
        let code=res.order[u32(k)/4u][u32(k)%4u];let salt=layer*64u+32u+u32(k);
        var n:f32;
        if((code>>4u)>0u){n=rFbmAligned(gi,gj,lod,salt,11,14,1.,w,theta);}
        else{n=rFbm(gi,gj,lod,salt,11,14,1.,w);}
        let v=rField(k,gi,gj,lod,w)+res.zones.y*province[code&15u]+res.zones.z*n;
        if(v>best){best=v;out=u32(k);}
      }
      return out;
    }
    fn rRamp(t:f32)->vec3<f32>{
      let n=max(i32(res.zones2.w),2);
      let x=clamp(t,0.,1.)*f32(n-1);let k=min(i32(x),n-2);let f=x-f32(k);
      return mix(res.palette[k].rgb,res.palette[k+1].rgb,f);
    }
    fn rDelta(g:i32,d:i32,lod:i32,r:f32)->f32{
      if(lod>=0){return f32((g<<u32(lod))-d)*30.+(15.*exp2(f32(lod))-15.);}
      return f32(g-(d<<u32(-lod)))*r+r*.5-15.;
    }
    // Deposit footprints (points) or kernel density (stock) around the sample.
    fn rDeposits(gi:i32,gj:i32,lod:i32,r:f32,valley:f32,base:vec3<f32>,density:ptr<function,f32>)->vec3<f32>{
      var rgb=base;
      if(res.zones2.y<.5){return rgb;}
      var cx:i32;var cy:i32;
      if(lod<=10){cx=gi>>u32(10-lod);cy=gj>>u32(10-lod);}else{cx=i32(floor((f32(gi)+.5)*exp2(f32(lod-10))));cy=i32(floor((f32(gj)+.5)*exp2(f32(lod-10))));}
      let R=clamp(i32(ceil(max(12000.,8.*r)/30720.)),1,3);
      let W=i32(res.grid.z);let stock=res.info.x>5.5;
      for(var oy=-R;oy<=R;oy++){for(var ox=-R;ox<=R;ox++){
        let cr=rColRow(cx+ox,cy+oy);let bucket=resBuckets[u32(cr.y*W+cr.x)];
        for(var k=bucket.x;k<bucket.x+bucket.y;k++){
          let d=resDeposits[k];
          let dx=rDelta(gi,d.ij.x,lod,r);let dy=rDelta(gj,d.ij.y,lod,r);
          let dist=sqrt(dx*dx+dy*dy);
          if(stock){let sg=max(d.extra.z,.75*r);*density+=d.extra.w*1e6/(6.2831853*sg*sg)*exp(-(dx*dx+dy*dy)/(2.*sg*sg));continue;}
          let a=d.dim.x;let b=d.dim.y;let c=cos(d.dim.z);let s=sin(d.dim.z);
          let u=dx*c+dy*s;let v=-dx*s+dy*c;
          let color=res.palette[min(d.ij.w,19)].rgb*(.45+.55*d.dim.w);
          if((min(a,b)*2./r<1.5&&d.ij.z!=3)||a*2./r<1.5){
            if(d.ij.z==0&&a*2./r>=3.){let body=clamp(1.2-abs(v)/r,0.,1.)*clamp((a-abs(u))/r,0.,1.);rgb=mix(rgb,color,body);continue;}
            let rpx=d.extra.y*2.5*clamp(4000./r,.4,1.)+1.5;let q=dist/r;
            let disc=clamp(rpx+.5-q,0.,1.);let ring=clamp(rpx+1.5-q,0.,1.)-disc;
            rgb=mix(rgb,color,disc);rgb=rgb*(1.-ring);continue;
          }
          let q=sqrt((u/a)*(u/a)+(v/b)*(v/b));
          var body:f32;
          if(d.ij.z==0){body=clamp((b+.5*r-abs(v))/r,0.,1.)*clamp((a-abs(u))/(r+.05*a),0.,1.);}
          else if(d.ij.z==3){body=.8*clamp(1.2-dist/a,0.,1.)*valley;}
          else{body=clamp((1.-q)*b/r,0.,1.);}
          let haloR=min(max(a,2.*r),4000.);let halo=.2*exp(-(dist/haloR)*(dist/haloR));
          let rim=clamp((1.6-q)/.6,0.,1.)*(1.-body)*d.extra.x;
          rgb=mix(rgb,vec3<f32>(.788,.651,.290),halo);rgb=mix(rgb,vec3<f32>(.710,.325,.165),.7*rim);rgb=mix(rgb,color,body);
        }
      }}
      return rgb;
    }
    `;
  // Colour of one sample from its global sample index and terrain features;
  // shared by the shade pass (physical tiles) and the native coarse blocks.
  const shared=`
    fn resourceShade(gi:i32,gj:i32,lod:i32,r:f32,f:RFeat,light:f32)->vec3<f32>{
      let kind=res.info.x;
      let neutral=vec3<f32>(.60,.58,.53)*light;
      if(res.zones2.z<.5){return neutral;}
      if(kind>1.5&&kind<2.5){
        // Rocks: realized geology at every LOD (same function as the canonical
        // shares). Contexts: dominant soft membership.
        var c:u32;
        if(res.zones.x>2.5){c=rRealized(gi,gj,lod);}else{c=rDominant(gi,gj,lod);}
        let rgb=res.palette[min(c,19u)].rgb;
        return rgb*light;
      }
      if(kind>3.5&&kind<4.5){
        let v=rCyclic(gi,gj,lod);
        if(v<0.){return neutral;}
        return (.5+.5*cos(6.2831853*(v/12.+vec3<f32>(0.,1./3.,2./3.))))*.9*light;
      }
      if(kind>5.5){
        var density=0.;_=rDeposits(gi,gj,lod,r,f.valley,neutral,&density);
        let lo=res.range.x;let hi=res.range.y;
        if(density<=lo){return neutral;}
        let t=(log(max(density,lo))-log(lo))/(log(hi)-log(lo));
        return mix(neutral,rRamp(t)*light,rSmooth(0.,.06,t));
      }
      let v=rContinuous(gi,gj,lod,rTerms(f));
      if(kind>4.5){
        var density=0.;let bg=mix(neutral,vec3<f32>(.353,.231,.110)*light,.55*v);
        return rDeposits(gi,gj,lod,r,max(f.valley,f.river),bg,&density);
      }
      if(kind>2.5){
        let c=select(select(select(select(select(0,1,v>0.),2,v>=.2),3,v>=.4),4,v>=.6),5,v>=.8);
        return select(res.palette[c].rgb*light,neutral,c==0);
      }
      let lo=res.range.x;let hi=res.range.y;
      var t:f32;
      if(kind>.5){if(v<=lo){return neutral;}t=(log(max(v,lo))-log(lo))/(log(hi)-log(lo));}
      else{if(v<=lo+1e-6*(hi-lo)){return neutral;}t=(v-lo)/(hi-lo);}
      // Fade into the neutral base over the first 6 % of the scale: no hard edge where a field reaches 0.
      return mix(neutral,rRamp(t)*light,rSmooth(0.,.06,t));
    }`;
  // Shade-pass front: reads the tile textures and params of terrain_renderer.js.
  const color=`${shared}
    // Land base keeps the relief readable where a layer is empty.
    fn resourceColor(p:vec2<i32>,height:f32,light:f32)->vec3<f32>{
      let r=params.resolution;
      let gi=bitcast<i32>(params.style.z)+p.x;let gj=bitcast<i32>(params.style.w)+p.y;
      let e=textureLoad(elevation,p,0).r;
      let g=vec2<f32>(textureLoad(elevation,p+vec2<i32>(1,0),0).r-textureLoad(elevation,p-vec2<i32>(1,0),0).r,textureLoad(elevation,p+vec2<i32>(0,1),0).r-textureLoad(elevation,p-vec2<i32>(0,1),0).r)*.5/r;
      let gl=vec2<f32>(textureLoad(blurred,p+vec2<i32>(1,0),0).r-textureLoad(blurred,p-vec2<i32>(1,0),0).r,textureLoad(blurred,p+vec2<i32>(0,1),0).r-textureLoad(blurred,p-vec2<i32>(0,1),0).r)*.5/r;
      let bl=textureLoad(blurred,p,0).rg;
      var river=0.;if(textureNumLevels(hydrology)>2u){river=textureLoad(hydrology,p>>vec2<u32>(2u),2).g;}
      return resourceShade(gi,gj,i32(round(log2(r/30.))),r,rFeatures(e,g,gl,bl.g,bl.r,r,river),light);
    }`;
  // Native coarse front (LOD 4-7 views drawn directly from LOD-8 blocks): the
  // sample index is that of the screen pixel's LOD (fragment footprint), so
  // zones and deposits keep their detail; terrain terms use the block heights.
  const coarse=`${shared}
    fn resourceCoarseColor(p:vec2<f32>,height:f32,light:f32,footprint:f32)->vec3<f32>{
      let r=params.viewport.w;
      // +0.02: a view exactly at a LOD boundary must not flip between 2x2 quads.
      let lod=clamp(i32(floor(log2(max(footprint*r,30.)/30.)+.02))-1,-2,i32(round(log2(r/30.))));
      let s=30.*exp2(f32(lod));
      let x=params.style.xy+(p+.5)*r;
      let gi=i32(floor(x.x/s));let gj=i32(floor(x.y/s));
      let g=vec2<f32>(heightAt(p+vec2<f32>(1.,0.),0.)-heightAt(p-vec2<f32>(1.,0.),0.),heightAt(p+vec2<f32>(0.,1.),0.)-heightAt(p-vec2<f32>(0.,1.),0.))*.5/r;
      let gl=vec2<f32>(heightAt(p+vec2<f32>(4.,0.),3.)-heightAt(p-vec2<f32>(4.,0.),3.),heightAt(p+vec2<f32>(0.,4.),3.)-heightAt(p-vec2<f32>(0.,4.),3.))*.125/r;
      return resourceShade(gi,gj,lod,s,rFeatures(height,g,gl,heightAt(p,1.),heightAt(p,3.),r,0.),light);
    }`;
  // Uniform bytes from a package header (backend/terrain_resource_layers.py:gpu_package).
  function pack(header,{season=.5,active=true,marine=false}={}){
    const buffer=new ArrayBuffer(PARAM_BYTES),f=new Float32Array(buffer),u=new Uint32Array(buffer);
    const [i0,j0,W,H]=header.lattice;f.set([i0,j0,W,H],0);
    f.set([header.kind,header.layers,header.classes,header.seasonal?1:0],4);
    f.set([header.range[0],header.range[1],header.bounded?1:0,season],8);
    f.set(header.coef.slice(0,4),12);f.set([header.coef[4],header.coef[5],header.sigma,header.thresholds],16);
    f.set([header.realized?3:0,...(header.beta||[0,0]),0],20);
    f.set([0,header.deposits?1:0,active?(marine?2:1):0,Math.min(20,header.palette.length)],24);
    u.set([header.seed32>>>0,header.layer>>>0],28);f.set(header.bounds||header.range.slice(0,2),30);
    (header.order||[]).slice(0,12).forEach((v,i)=>{u[32+i]=v;});
    header.palette.slice(0,20).forEach((c,i)=>f.set([c[0],c[1],c[2],1],44+i*4));
    return buffer;
  }
  // Binary package: field (f16), thresholds (f16), deposits (12 words each), buckets (2 u32 per cell).
  function parse(header,bytes){
    const [K,H,W]=header.field_shape;let at=0;
    const field=new Uint16Array(bytes,at,K*H*W);at+=K*H*W*2;
    let tau=null;if(header.thresholds){tau=new Uint16Array(bytes,at,header.thresholds*H*W);at+=header.thresholds*H*W*2;}
    let deposits=null,buckets=null;
    if(header.deposits){at=Math.ceil(at/4)*4;deposits=new Uint32Array(bytes.slice(at,at+header.deposits*48));at+=header.deposits*48;buckets=new Uint32Array(bytes.slice(at,at+header.lattice[2]*header.lattice[3]*8));}
    return {field,fieldShape:[W,H,K],tau,deposits,buckets};
  }
  root.TerrainResourcesGPU={PREFIX,KINDS,PARAM_BYTES,wgsl:core+color,coarse:core+coarse,core,isMode,number,isNumber,pack,parse};
})(typeof window!=='undefined'?window:globalThis);

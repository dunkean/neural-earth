/* Render3D and Render3DBrush: a multiscale ground material and a canonical
 * forest over the Render layer. WGSL adaptation of Magna Urbis
 * rust/bridge/terrainGround.ts, terrainForest.ts and forestBrushAtlas.ts
 * (GPL-3.0), whose own Render material is ported from this repository.
 *
 * Differences from Magna Urbis: the forest field is baked on the GPU for each
 * physical tile from Render's own terms and the spatial seasonal climate,
 * instead of one CPU raster for a fixed-climate map. Riparian galleries and
 * applied land use are not yet connected to hydrology or settlement data.
 * Planet-scale coordinates are split per tile into exact integer cells plus
 * f32 fractions (computed here in float64), so lattices stay seamless and
 * sub-metre detail keeps its precision far from the origin. Display only.
 */
(root=>{
  'use strict';
  const MODES={render3d:-6,'render3d-brush':-7};
  const isStyle=mode=>Object.hasOwn(MODES,mode);
  // Magna Urbis ground bands: wavelength m, amplitude m, anisotropy, filter, rotation, slope variance.
  const BANDS=[
    {wavelength:8,amplitude:.17,aspect:.35,filterWidth:1,c:.8,s:.6,variance:.20},
    {wavelength:2.2,amplitude:.04,aspect:.24,filterWidth:.5,c:-.6,s:.8,variance:1.8},
    {wavelength:.32,amplitude:.012,aspect:.85,filterWidth:.5,c:.28,s:.96,variance:1.3},
    {wavelength:.055,amplitude:.002,aspect:.72,filterWidth:1,c:-.96,s:.28,variance:.40},
  ];
  // Forest-field noise scales (m): warp, stand mosaic, disturbance, conifer
  // admixture, stand tone, productivity, treeline, clearing bias and clearings.
  const BAKE_SCALES=[5200,2600,2400,1200,600,1100,550,1800,900,1300,650,9000,4500,520,260,400,200];
  // Split-entry indices shared by JS and WGSL.
  const SPLIT={crown:18,patch:25,ground:26,gravel:29,strata:30,entries:32};
  const BAND_OFFSET=[[17.137,3.731,-11.517],[9.173,-5.731,7.337]];
  // Crown lattice: per-level target d -> control tau (Monte-Carlo inverse of the
  // jittered-lattice union coverage) and the patch/crown threshold CDFs.
  const TAU=[0,.0211,.0446,.066,.0881,.1113,.1338,.1573,.182,.2051,.2303,.2538,.2796,.3045,.3293,.3546,
    .3816,.4091,.435,.4636,.4895,.5188,.5472,.5668,.586,.6063,.6289,.6524,.6805,.7112,.751,.8099,1];
  const PATCH_CDF=[0,0,.0001,.001,.0085,.0401,.1202,.2729,.5009,.7274,.8801,.9598,.9914,.999,.9999,1,1];
  const CROWN_CDF=[0,0,0,.0005,.0063,.0336,.1125,.2742,.4998,.7261,.8874,.9665,.9936,.9994,1,1,1];
  const f=v=>{const s=String(v);return /[.e]/.test(s)?s:s+'.';};
  const list=values=>values.map(f).join(',');
  const turnRotation=(k,seed)=>{
    let c=BANDS[k].c,s=BANDS[k].s;const turn=(seed>>>29)&3;
    if(turn===1)[c,s]=[-s,c];else if(turn===2)[c,s]=[-c,-s];else if(turn===3)[c,s]=[s,-c];
    return [c,s];
  };

  /** Per-tile uniform: cells (i32) + fractions for each lattice, from the
   * world position of the tile's first halo corner. */
  function tileUniform(origin,seed){
    const n=SPLIT.entries,buffer=new ArrayBuffer(n*32+16);
    const cells=new Int32Array(buffer,0,n*4),fracs=new Float32Array(buffer,n*16,n*4);
    const [ox,oy]=origin;
    const split=(entry,values,extra=[0,0])=>{
      values.forEach((v,i)=>{const c=Math.floor(v);cells[entry*4+i]=c;fracs[entry*4+i]=v-c;});
      if(values.length===2){fracs[entry*4+2]=extra[0];fracs[entry*4+3]=extra[1];}
    };
    BAKE_SCALES.forEach((s,i)=>split(i,[ox/s,oy/s],[s,1/s]));
    for(let level=-1;level<=6;level++){const s=2*2**(level/2);split(SPLIT.crown+level,[ox/s,oy/s],[s,1/s]);}
    split(SPLIT.patch,[ox/56,oy/56],[56,1/56]);
    [0,1,3].forEach((k,i)=>{
      const [c,s]=turnRotation(k,seed),{wavelength,aspect}=BANDS[k],x=ox,z=oy;
      split(SPLIT.ground+i,[
        ((.36*c-.48*s)*x+(.36*s+.48*c)*z)/wavelength+BAND_OFFSET[0][0]+k*BAND_OFFSET[1][0],
        ((.48*c-.64*s)*x+(.48*s+.64*c)*z)/(aspect*wavelength)+BAND_OFFSET[0][1]+k*BAND_OFFSET[1][1],
        ((-.8*c-.6*s)*x+(-.8*s+.6*c)*z)/wavelength+BAND_OFFSET[0][2]+k*BAND_OFFSET[1][2]]);
    });
    {const [c,s]=turnRotation(2,seed),{wavelength,aspect}=BANDS[2];split(SPLIT.gravel,[(c*ox+s*oy)/wavelength,(-s*ox+c*oy)/(wavelength*aspect)]);}
    const phase=((.13*ox-.09*oy)*9.66643893)%(2*Math.PI);fracs[SPLIT.strata*4]=phase;
    new Uint32Array(buffer,n*32,4)[0]=seed>>>0;
    return buffer;
  }

  const common=`
    struct R3Tile { cells: array<vec4<i32>,${SPLIT.entries}>, fracs: array<vec4<f32>,${SPLIT.entries}>, info: vec4<u32> }
    fn r3Seed()->u32{return r3.info.x;}
    fn groundSmooth(a:f32,b:f32,x:f32)->f32{let t=clamp((x-a)/(b-a),0.,1.);return t*t*(3.-2.*t);}
    fn groundHash(c:vec3<i32>,seed:u32)->u32{
      let v=bitcast<vec3<u32>>(c);var h=(v.x*0x9e3779b9u)^(v.y*0x85ebca6bu)^(v.z*0xc2b2ae35u)^seed;
      h=(h^(h>>16u))*0x7feb352du;h=(h^(h>>15u))*0x846ca68bu;return h^(h>>16u);
    }
    // Exact world lattice coordinate of a tile-local position (metres) in units of entry e.
    fn r3Point(e:i32,p:vec2<f32>)->vec2<f32>{return r3.fracs[e].xy+p*r3.fracs[e].w;}
    fn r3CellAt(e:i32,p:vec2<f32>)->vec2<i32>{return r3.cells[e].xy+vec2<i32>(floor(r3Point(e,p)));}
    fn r3CellLocal(e:i32,cell:vec2<i32>)->vec2<f32>{return (vec2<f32>(cell-r3.cells[e].xy)-r3.fracs[e].xy)*r3.fracs[e].z;}
  `;

  const captureState=`
    var<private> r3GuideOpen:vec3<f32>;var<private> r3GuideClosed:vec3<f32>;var<private> r3Woods:f32;
    var<private> r3Humid:f32;var<private> r3Valley:f32;var<private> r3Ridge:f32;var<private> r3THot:f32;
    var<private> r3TS:f32;var<private> r3TW:f32;var<private> r3Outcrop:f32;var<private> r3Perennial:f32;
    var<private> r3Grade:f32;var<private> r3WTrop:f32;var<private> r3RockTan:f32;var<private> r3M1:f32;
    var<private> r3Herbs:f32;var<private> r3Beach:f32;var<private> r3Flats:f32;var<private> r3Sediment:vec3<f32>;var<private> r3BaseOpen:vec3<f32>;
    var<private> r3BaseClosed:vec3<f32>;
  `;
  // Forest field, evaluated in the tile bake after surfaceMaterial has
  // captured its terms. Seeded 2D gradient noise in exact world lattices.
  const bake=`
    const R3_GRAD=array<vec2<f32>,16>(${Array.from({length:16},(_,i)=>`vec2<f32>(${f(Math.cos(i*Math.PI/8+.2))},${f(Math.sin(i*Math.PI/8+.2))})`).join(',')});
    ${captureState}
    fn r3GradientNoise(c:vec2<i32>,q:vec2<f32>,salt:u32)->f32{
      let u=q*q*q*(q*(q*6.-15.)+10.);
      let a=dot(R3_GRAD[groundHash(vec3<i32>(c,0),salt)&15u],q);
      let b=dot(R3_GRAD[groundHash(vec3<i32>(c+vec2<i32>(1,0),0),salt)&15u],q-vec2<f32>(1.,0.));
      let d=dot(R3_GRAD[groundHash(vec3<i32>(c+vec2<i32>(0,1),0),salt)&15u],q-vec2<f32>(0.,1.));
      let e=dot(R3_GRAD[groundHash(vec3<i32>(c+vec2<i32>(1,1),0),salt)&15u],q-vec2<f32>(1.,1.));
      return ((a+(b-a)*u.x)*(1.-u.y)+(d+(e-d)*u.x)*u.y)*4.639;
    }
    fn r3Noise(e:i32,p:vec2<f32>,offset:vec2<f32>,salt:u32)->f32{
      let q=r3Point(e,p)+offset;let fl=floor(q);return r3GradientNoise(r3.cells[e].xy+vec2<i32>(fl),q-fl,salt);
    }
    // Fixed octave set (never footprint- or resolution-dependent), unit variance.
    fn r3Fbm(e:i32,octaves:i32,p:vec2<f32>,salt:u32)->f32{
      var sum=0.;var norm=0.;var a=1.;
      for(var i=0;i<octaves;i++){sum+=a*r3Noise(e+i,p,vec2<f32>(31.7,-17.3)*f32(i),salt+u32(i)*0x9e37u);norm+=a*a;a*=.55;}
      return sum/sqrt(norm);
    }
    // RGBA: canopy cover, potential height/50 m, needleleaf fraction, stand tone.
    // Cover and height carry Render's tree eligibility (rock, slope, perennial
    // snow, treeline) from the same material terms, so rock-coloured ground
    // stays treeless. Climate replaces Magna Urbis's single preset.
    fn r3ForestField(height:f32,slope:f32,p:vec2<f32>,seed:u32)->vec4<f32>{
      let wx=900.*r3Fbm(0,2,p,seed+11u);let wy=900.*r3Fbm(0,2,p+vec2<f32>(7310.,-2210.),seed+13u);
      let stand=r3Fbm(2,3,p+vec2<f32>(wx,wy),seed+17u);
      let disturbed=sstep(1.05,1.5,r3Fbm(5,2,p+vec2<f32>(-wy,wx),seed+19u));
      let conifer=r3Fbm(7,2,p,seed+23u);let toneNoise=r3Fbm(9,2,p+vec2<f32>(wy,-wx),seed+29u);
      let productivity=r3Fbm(11,2,p,seed+31u);let ragged=r3Fbm(13,2,p,seed+37u);
      let valley=r3Valley;let ridge=r3Ridge;let trop=r3WTrop;
      // Summer temperature at altitude is the growing-season proxy.
      let grow=max(r3TS,r3TW);let humid=r3Humid*exp(.12*productivity);
      let treeline=grow+.9*ragged+.7*valley-.5*ridge;
      let thermal=sstep(5.,8.5,treeline);let moisture=sstep(.38,1.25,humid);
      let coverMax=.6+.38*sstep(.8,2.2,humid);
      let bog=valley*(1.-sstep(.02,.08,slope))*sstep(1.6,3.,humid)*(1.-sstep(11.,17.,grow))*(1.-trop);
      var cover=thermal*moisture*coverMax*(1.-.6*bog)*(1.-.2*ridge*(1.-moisture));
      // Wet tropical mosaic and disturbance lower and lighten the canopy instead of opening it.
      let mosaic=.22*(1.-.75*trop);
      cover=clamp(cover*(1.+mosaic*stand)+.05*stand*cover*(1.-cover),0.,1.)*(1.-.3*disturbed*(1.-.7*trop));
      let gentle=1.-sstep(.05,.25,slope);let rugged=sstep(.35,.8,slope);
      let bias=.45*(ridge-valley)+.3*(gentle-.5)-.6*rugged+.35*stand+.45*productivity+r3Noise(15,p,vec2<f32>(0.),seed+41u)/1.077;
      let thermalHeight=.5*sstep(5.5,9.5,grow)+.5*sstep(9.,22.,grow);
      var h=3.+31.*pow(sstep(.4,2.8,humid),.8)*thermalHeight;
      h*=(1.+(.16+.08*trop)*stand)*(1.-.55*disturbed)*(1.+.15*valley-.12*ridge)*(.35+.65*sstep(5.,9.,treeline));
      let needle=clamp(1.-sstep(10.,19.,grow)+.3*conifer+.35*(1.-sstep(.7,1.4,humid))*(1.-trop)-trop,0.,1.);
      let tone=clamp(.5+(.16+.1*trop)*toneNoise+.22*disturbed-.05*valley,0.,1.);
      // Clearings: logistic normal quantile of the target opened share.
      let cleared=.4*params.render2.w*(1.-.85*trop);
      if(cleared>0.){
        let c=max(cleared,1e-4);let threshold=1.15*log((1.-c)/c)/1.702;
        cover*=1.-sstep(threshold-.25,threshold+.25,.4*r3Noise(16,p,vec2<f32>(31.7,-17.3),seed+41u+0x9e37u)/1.077+bias);
      }
      let density=params.render1.w*params.render0.z;
      if(density!=1.){cover=1.-pow(1.-min(cover,.999),density);}
      let rock=sstep(.3,.85,1.-r3Outcrop)*(1.-sstep(r3RockTan*.9,r3RockTan*1.6,r3Grade));
      let eligible=rock*(1.-r3Perennial)*sstep(5.5,13.5,r3THot+.8*valley);
      cover*=eligible;
      if(height<=0.||params.render1.z<.5){cover=0.;}
      return vec4<f32>(cover,h*sstep(0.,.4,eligible)/50.,needle,tone);
    }
  `;
  // A distant proxy of the same population. All lattice residuals have mean
  // zero; clearings are integrated as an opened fraction, not sampled as a
  // point of the threshold profile. Actual fine fields remain authoritative.
  const forestExpected=`
    fn r3ForestExpected(height:f32,slope:f32)->vec4<f32>{
      let valley=r3Valley;let ridge=r3Ridge;let trop=r3WTrop;
      let grow=max(r3TS,r3TW);let humid=r3Humid;
      let treeline=grow+.7*valley-.5*ridge;
      let moisture=sstep(.38,1.25,humid);let thermal=sstep(5.,8.5,treeline);
      let coverMax=.6+.38*sstep(.8,2.2,humid);
      let bog=valley*(1.-sstep(.02,.08,slope))*sstep(1.6,3.,humid)*(1.-sstep(11.,17.,grow))*(1.-trop);
      var cover=clamp(thermal*moisture*coverMax*(1.-.6*bog)*(1.-.2*ridge*(1.-moisture)),0.,1.);
      let gentle=1.-sstep(.05,.25,slope);let rugged=sstep(.35,.8,slope);
      let bias=.45*(ridge-valley)+.3*(gentle-.5)-.6*rugged;
      let cleared=.4*params.render2.w*(1.-.85*trop);
      if(cleared>0.){
        let c=clamp(cleared,1e-4,.999);
        // Expected thinning, not a thresholded clearing. Centered symmetric
        // scores preserve c before the cap; fine residual covariance is absent.
        let opened=min(.55,c);
        cover*=1.-opened;
      }
      let density=params.render1.w*params.render0.z;
      if(density!=1.){cover=1.-pow(1.-min(cover,.999),density);}
      let thermalHeight=.5*sstep(5.5,9.5,grow)+.5*sstep(9.,22.,grow);
      var h=3.+31.*pow(sstep(.4,2.8,humid),.8)*thermalHeight;
      h*=(1.+.15*valley-.12*ridge)*(.35+.65*sstep(5.,9.,treeline));
      let needle=clamp(1.-sstep(10.,19.,grow)+.35*(1.-sstep(.7,1.4,humid))*(1.-trop)-trop,0.,1.);
      let rock=sstep(.3,.85,1.-r3Outcrop)*(1.-sstep(r3RockTan*.9,r3RockTan*1.6,r3Grade));
      let eligible=rock*(1.-r3Perennial)*sstep(5.5,13.5,r3THot+.8*valley);
      cover*=eligible;if(height<=0.||params.render1.z<.5){cover=0.;}
      return vec4<f32>(cover,h*sstep(0.,.4,eligible)/50.,needle,clamp(.5-.05*valley,0.,1.));
    }
  `;
  // Captures inserted into surfaceMaterial for the Render guide and the field.
  const captures={
    '/*R3:base-canopy*/':'r3BaseClosed=canopy;',
    '/*R3:canopy-detail*/':'r3BaseClosed=r3BaseClosed*tint*(1.-.18*highland)*(1.+variation*canopyDetail);',
    '/*R3:open*/':'r3GuideOpen=color;r3BaseOpen=color;',
    '/*R3:herbs*/':'r3Herbs=herbs*(1.-outcrop);',
    '/*R3:beach*/':'r3GuideOpen=surfaceMix(surfaceMix(r3GuideOpen,sediment*.9,flats*(1.-sstep(240.,960.,fp))*.8),sediment,beach*.8);r3Beach=beach*.8;r3Flats=flats;r3Sediment=sediment;',
    '/*R3:end*/':`let r3Snow=surfaceCover(snow,crag,.35*variation);
      r3GuideOpen=clamp(surfaceMix(r3GuideOpen,params.render4.xyz,r3Snow)*(1.-.06*valley),vec3<f32>(0.),vec3<f32>(1.));
      r3GuideClosed=clamp(surfaceMix(canopy,params.render4.xyz,r3Snow)*(1.-.06*valley),vec3<f32>(0.),vec3<f32>(1.));
      r3Woods=woods;r3Humid=humid;r3Valley=valley;r3Ridge=ridge;r3THot=tHot;r3TS=ts;r3TW=tw;r3Outcrop=outcrop;
      r3Perennial=perennial;r3Grade=grade;r3WTrop=wTrop;r3RockTan=rockTan;r3M1=m1;`,
  };
  const withCaptures=material=>Object.entries(captures).reduce((code,[marker,value])=>{
    if(!code.includes(marker))throw Error('Render3D capture marker missing: '+marker);
    return code.replace(marker,value);
  },material);

  const glsl=key=>BANDS.map(b=>f(b[key])).join(',');
  const ground=`
    const GROUND_WAVELENGTH=array<f32,4>(${glsl('wavelength')});
    const GROUND_AMPLITUDE=array<f32,4>(${glsl('amplitude')});
    const GROUND_ASPECT=array<f32,4>(${glsl('aspect')});
    const GROUND_FILTER=array<f32,4>(${glsl('filterWidth')});
    const GROUND_C=array<f32,4>(${glsl('c')});
    const GROUND_S=array<f32,4>(${glsl('s')});
    const GROUND_VARIANCE=array<f32,4>(${glsl('variance')});
    fn groundToLinear(c:vec3<f32>)->vec3<f32>{return pow(max(c,vec3<f32>(0.)),vec3<f32>(2.2));}
    fn groundToDisplay(c:vec3<f32>)->vec3<f32>{return pow(clamp(c,vec3<f32>(0.),vec3<f32>(1.)),vec3<f32>(1./2.2));}
    fn groundNoise(cell:vec3<i32>,q:vec3<f32>,seed:u32)->vec4<f32>{
      let u=q*q*q*(q*(q*6.-15.)+10.);let du=30.*q*q*(q*(q-2.)+1.);var n=vec4<f32>(0.);
      for(var k=0;k<8;k++){
        let corner=vec3<i32>(k&1,(k>>1)&1,k>>2);let h=groundHash(cell+corner,seed);let cf=vec3<f32>(corner);
        let g=vec3<f32>(f32(h&1023u),f32((h>>10u)&1023u),f32((h>>20u)&1023u))/511.5-1.;
        let w=mix(1.-u,u,cf);let d=mix(-du,du,cf);let v=dot(g,q-cf);
        n+=vec4<f32>(v*w.x*w.y*w.z,g*w.x*w.y*w.z+v*vec3<f32>(d.x*w.y*w.z,w.x*d.y*w.z,w.x*w.y*d.z));
      }
      return n*1.7;
    }
    // Compact C1 elliptical clasts on a jittered lattice; saturating sum of caps.
    fn groundGravel(cell:vec2<i32>,q:vec2<f32>,seed:u32)->vec4<f32>{
      var sum=0.;var gradient=vec2<f32>(0.);
      for(var j=-1;j<=1;j++){for(var i=-1;i<=1;i++){
        let nb=cell+vec2<i32>(i,j);let h=groundHash(vec3<i32>(nb.x,0,nb.y),seed);
        if((h&7u)>4u){continue;}
        let d=q-(vec2<f32>(f32(i),f32(j))+.5+(vec2<f32>(f32(h&255u),f32((h>>8u)&255u))/255.-.5));
        let angle=(h>>24u)&3u;
        let c=select(select(select(-.7071067811865476,.7071067811865476,angle==1u),0.,angle==2u),1.,angle==0u);
        let s=select(select(.7071067811865476,1.,angle==2u),0.,angle==0u);
        let uv=vec2<f32>(c*d.x+s*d.y,-s*d.x+c*d.y);
        let radius=.10+.30*vec2<f32>(f32((h>>16u)&15u),f32((h>>20u)&15u))/15.;
        let inv=1./(radius*radius);let e=dot(uv*uv,inv);if(e>=1.){continue;}
        let t=1.-e;let weight=.6+1.8*f32(h>>28u)/15.;sum+=weight*t*t;
        gradient+=-4.*weight*t*vec2<f32>(c*uv.x*inv.x-s*uv.y*inv.y,s*uv.x*inv.x+c*uv.y*inv.y);
      }}
      let attenuation=exp(-sum);return vec4<f32>(1.-attenuation,gradient.x*attenuation,0.,gradient.y*attenuation);
    }
    fn groundVisibility(wavelength:f32,footprint:f32)->f32{
      let t=clamp((footprint/wavelength-.18)/.24,0.,1.);return 1.-t*t*t*(t*(t*6.-15.)+10.);
    }
    fn groundSlopeVariance(footprint:f32)->f32{
      var variance=0.;
      for(var k=0;k<4;k++){
        let a=GROUND_AMPLITUDE[k]/GROUND_WAVELENGTH[k];
        let keep=groundVisibility(GROUND_WAVELENGTH[k]*GROUND_ASPECT[k]*GROUND_FILTER[k],max(footprint,0.));
        variance+=a*a*(1.+1./(GROUND_ASPECT[k]*GROUND_ASPECT[k]))*GROUND_VARIANCE[k]*(1.-keep*keep);
      }
      return variance;
    }
    fn groundRotation(k:i32,seed:u32)->vec2<f32>{
      var c=GROUND_C[k];var s=GROUND_S[k];let turn=(seed>>29u)&3u;
      if(turn==1u){let o=c;c=-s;s=o;}else if(turn==2u){c=-c;s=-s;}else if(turn==3u){let o=c;c=s;s=-o;}
      return vec2<f32>(c,s);
    }
    // [height m, dH/dX, dH/dUp, dH/dZ]; bands hidden by the footprint are skipped, not renormalized.
    fn groundFieldWithClasts(p:vec2<f32>,seed:u32,footprint:f32,clasts:ptr<function,f32>)->vec4<f32>{
      var sum=vec4<f32>(0.);*clasts=0.;
      for(var k=0;k<4;k++){
        let wavelength=GROUND_WAVELENGTH[k];let aspect=GROUND_ASPECT[k];
        let keep=groundVisibility(wavelength*aspect*GROUND_FILTER[k],max(footprint,0.));if(keep<=0.){continue;}
        let cs=groundRotation(k,seed);let c=cs.x;let s=cs.y;let a=GROUND_AMPLITUDE[k]*keep;
        if(k==2){
          let scale=vec2<f32>(wavelength,wavelength*aspect);
          let q=r3.fracs[${SPLIT.gravel}].xy+vec2<f32>(c*p.x+s*p.y,-s*p.x+c*p.y)/scale;let fl=floor(q);
          let cap=groundGravel(r3.cells[${SPLIT.gravel}].xy+vec2<i32>(fl),q-fl,seed+u32(k+1)*0x6c8e9cf5u);
          let g=cap.yw/scale;
          sum+=vec4<f32>((cap.x-.1)*a,(c*g.x-s*g.y)*a,0.,(s*g.x+c*g.y)*a);*clasts=cap.x*keep;continue;
        }
        let e=${SPLIT.ground}+select(k,2,k==3);
        let q=r3.fracs[e].xyz+vec3<f32>((.36*c-.48*s)*p.x+(.36*s+.48*c)*p.y,((.48*c-.64*s)*p.x+(.48*s+.64*c)*p.y)/aspect,(-.8*c-.6*s)*p.x+(-.8*s+.6*c)*p.y)/wavelength;
        let fl=floor(q);let n=groundNoise(r3.cells[e].xyz+vec3<i32>(fl),q-fl,seed+u32(k+1)*0x6c8e9cf5u);
        let inverse=inverseSqrt(1.+n.x*n.x);let v=.72*n.x*inverse;var value=v;var derivative=1.;
        if(k==0){let rounded=sqrt(v*v+.02);value=.3-rounded;derivative=-v/rounded;}
        else if(k==1){let groove=exp2(-32.*v*v);value=.4-groove;derivative=44.36141956*v*groove;}
        let d=a*derivative*.72*inverse*inverse*inverse/wavelength;
        let g=vec3<f32>(.36*n.y+.48*n.z/aspect-.8*n.w,-.8*n.y+.6*n.z/aspect,.48*n.y+.64*n.z/aspect+.6*n.w);
        sum+=vec4<f32>(value*a,vec3<f32>(c*g.x-s*g.z,g.y,s*g.x+c*g.z)*d);
      }
      return sum;
    }
    // rock / soil / grass / snow; continuous and sums to one.
    fn groundCover(m:vec3<f32>,climate:vec2<f32>)->vec4<f32>{
      let grade=length(m.yz);let temperature=climate.x-.0065*max(m.x,0.);let wet=groundSmooth(250.,1400.,climate.y);
      let snow=(1.-groundSmooth(-5.,1.,temperature))*(1.-groundSmooth(.55,1.3,grade));
      let exposed=1.-snow;let rock=groundSmooth(.25,.9,grade)*exposed;
      let growth=groundSmooth(-8.,4.,temperature)*(1.-groundSmooth(28.,42.,temperature));
      let grass=wet*growth*(exposed-rock);let soil=exposed-rock-grass;
      return vec4<f32>(rock,soil,grass,snow);
    }
    fn groundGain(m:vec3<f32>,cover:vec4<f32>)->f32{
      return groundSmooth(.025,.18,length(m.yz))*groundSmooth(0.,2.,m.x)*dot(cover,vec4<f32>(1.,.16,.045,.018));
    }
    struct GroundMaterial {albedo:vec3<f32>,normal:vec3<f32>,roughness:f32,occlusion:f32}
    // Cover and detail shading of the last ground material, reused to recolour Render's guides.
    var<private> groundCoverLast:vec4<f32>;var<private> groundShadeLast:f32;
    fn groundRecolor(base:vec3<f32>)->vec3<f32>{
      let c=groundCoverLast;
      let color=mix(base,vec3<f32>(.43,.42,.39),.7)*c.x+mix(base,vec3<f32>(.40,.32,.22),.18)*c.y+base*vec3<f32>(.96,1.02,.93)*c.z+mix(base,vec3<f32>(.87,.91,.95),.35)*c.w;
      return clamp(color*groundShadeLast,vec3<f32>(.015),vec3<f32>(.96));
    }
    // m: (height, dH/dX, dH/dZ) physical; relief scales only the macro shading slope.
    fn terrainGroundMaterial(m:vec3<f32>,p:vec2<f32>,footprint:f32,macroColor:vec3<f32>,climate:vec2<f32>,seed:u32,relief:f32)->GroundMaterial{
      let fp=max(0.,footprint);let cover=groundCover(m,climate);let gain=groundGain(m,cover);
      var clasts=0.;let detail=groundFieldWithClasts(p,seed,fp,&clasts);let gradient=detail.yw*gain;
      var result:GroundMaterial;result.normal=normalize(vec3<f32>(-relief*m.y-gradient.x,1.,-relief*m.z-gradient.y));
      let form=clamp(detail.x*18.,-1.,1.);let cavity=clamp(-detail.x*20.,0.,1.)*cover.x;
      // The cap that displaced a stone lightens its mineral crest; no independent colour noise.
      let cap=groundSmooth(.05,.7,clasts);
      let grade=length(m.yz);let strataKeep=groundVisibility(.65/max(1.,grade),fp);
      let strata=sin(m.x*9.66643893+r3.fracs[${SPLIT.strata}].x+(.13*p.x-.09*p.y)*9.66643893+detail.x*3.);
      groundCoverLast=cover;
      groundShadeLast=(1.+form*dot(cover,vec4<f32>(.12,.055,.045,.012)))*(1.+cap*.12*cover.x)*(1.+cover.x*.022*strata*strataKeep)*(1.-cavity*.11);
      let wet=groundSmooth(500.,1800.,climate.y)*(1.-cover.w);var roughness=dot(cover,vec4<f32>(.52,.83,.91,.76));
      roughness-=wet*(.08*cover.y+.10*cavity);roughness+=(form*.025-cap*.055)*cover.x;
      // Unresolved bands widen the GGX slope distribution instead of shimmering.
      let alpha2=pow(clamp(roughness,.38,.98),4.)+gain*gain*groundSlopeVariance(fp);
      result.albedo=groundToLinear(groundRecolor(macroColor));result.roughness=clamp(pow(alpha2,.25),.38,.98);
      // Local cavity approximation, not traced visibility.
      result.occlusion=clamp(1.-.22*cavity-.04*max(-form,0.)*cover.y,.72,1.);
      return result;
    }
    fn groundBrdf(m:GroundMaterial,view:vec3<f32>,light:vec3<f32>)->vec3<f32>{
      let n=m.normal;let v=normalize(view);let l=normalize(light);let hv=normalize(v+l);
      let nv=max(dot(n,v),.001);let nl=max(dot(n,l),0.);let nh=max(dot(n,hv),0.);let vh=max(dot(v,hv),0.);
      let a=m.roughness*m.roughness;let a2=a*a;let den=nh*nh*(a2-1.)+1.;
      let distribution=a2/(3.14159265*den*den);
      let visibility=.5/max(nl*sqrt(nv*nv*(1.-a2)+a2)+nv*sqrt(nl*nl*(1.-a2)+a2),.001);
      let fresnel=.04+.96*pow(1.-vh,5.);
      return ((1.-fresnel)*m.albedo/3.14159265+vec3<f32>(distribution*visibility*fresnel))*nl;
    }
    fn shadeTerrainGround(m:GroundMaterial,view:vec3<f32>,light:vec3<f32>,light2:vec3<f32>,strength:f32)->vec3<f32>{
      let amount=clamp(strength,0.,1.);let ambient=m.albedo*(.34+.16*clamp(m.normal.y,0.,1.))*m.occlusion;
      let direct=groundBrdf(m,view,light)*2.2+groundBrdf(m,view,light2)*.58;
      return groundToDisplay(mix(m.albedo,ambient,amount)+direct*max(strength,0.));
    }
  `;

  // Share the exact BRDF with distant material means without pulling in
  // world lattices, texture bindings or the resolved ground-detail functions.
  const lighting=ground.slice(ground.indexOf('    fn groundToLinear'),ground.indexOf('    fn groundNoise'))+
    'struct GroundMaterial {albedo:vec3<f32>,normal:vec3<f32>,roughness:f32,occlusion:f32}\n'+
    ground.slice(ground.indexOf('    fn groundBrdf'));

  const forest=`
    var<private> forestResolvedCover:f32=0.;
    const FOREST_TAU=array<f32,33>(${list(TAU)});
    const FOREST_PATCH_CDF=array<f32,17>(${list(PATCH_CDF)});
    const FOREST_CROWN_CDF=array<f32,17>(${list(CROWN_CDF)});
    const FOREST_PATCH=56.;
    struct RenderGuide {color:vec3<f32>,open:vec3<f32>,closed:vec3<f32>,woods:f32}
    struct ForestClimate {trop:f32,dry:f32,cold:f32,annual:f32,humid:f32}
    fn forestClimate(c:vec4<f32>)->ForestClimate{
      var k:ForestClimate;k.annual=c.r*100.-50.;k.humid=c.g*4.;let coldest=c.b*100.-50.;
      k.trop=groundSmooth(13.,19.,coldest);k.dry=1.-groundSmooth(.8,2.,k.humid);k.cold=1.-groundSmooth(8.,16.,2.*k.annual-coldest);
      return k;
    }
    // Tile-local metres -> aux texture (full tile, halo included).
    fn r3Aux(p:vec2<f32>,layer:i32,lod:f32)->vec4<f32>{return textureSampleLevel(aux,filtering,p/(params.viewport.w*params.field.xy),layer,lod);}
    fn forestLookup17(x:f32,table:i32)->f32{
      let u=clamp(x,0.,1.)*16.;let k=min(i32(u),15);
      let a=select(FOREST_CROWN_CDF[k],FOREST_PATCH_CDF[k],table==0);let b=select(FOREST_CROWN_CDF[k+1],FOREST_PATCH_CDF[k+1],table==0);
      return mix(a,b,u-f32(k));
    }
    fn forestTau(d:f32)->f32{let u=clamp(d,0.,.985)*32.;let k=min(i32(u),31);return mix(FOREST_TAU[k],FOREST_TAU[k+1],u-f32(k));}
    fn forestRho(tau:f32)->f32{return .5+.4*max(0.,(tau-.55)/.45);}
    fn forestU8(h:u32,shift:u32)->f32{return (f32((h>>shift)&255u)+.5)/256.;}
    fn forestGradient(c:vec2<i32>)->vec2<f32>{let h=groundHash(vec3<i32>(c,991),r3Seed()^0x6a09e667u);return vec2<f32>(f32(h&255u),f32((h>>8u)&255u))*(1./128.)+(.5/128.-1.);}
    // 2D gradient noise from four corner gradients: (value, d/dx, d/dy) in lattice units.
    fn forestPerlin(q:vec2<f32>,g00:vec2<f32>,g10:vec2<f32>,g01:vec2<f32>,g11:vec2<f32>)->vec3<f32>{
      let n00=dot(g00,q);let n10=dot(g10,q-vec2<f32>(1.,0.));let n01=dot(g01,q-vec2<f32>(0.,1.));let n11=dot(g11,q-1.);let k=n00-n10-n01+n11;
      let u=q*q*q*(q*(q*6.-15.)+10.);let du=30.*q*q*(q*(q-2.)+1.);
      let g=g00+u.x*(g10-g00)+u.y*(g01-g00)+u.x*u.y*(g00-g10-g01+g11)+du*vec2<f32>(n10-n00+u.y*k,n01-n00+u.x*k);
      return vec3<f32>(n00+u.x*(n10-n00)+u.y*(n01-n00)+u.x*u.y*k,g);
    }
    fn forestLatticeNoise(e:i32,p:vec2<f32>,offset:vec2<f32>)->vec3<f32>{
      let g=r3Point(e,p)+offset;let fl=floor(g);let c=r3.cells[e].xy+vec2<i32>(fl);
      return forestPerlin(g-fl,forestGradient(c),forestGradient(c+vec2<i32>(1,0)),forestGradient(c+vec2<i32>(0,1)),forestGradient(c+vec2<i32>(1,1)));
    }
    // Uniform lattice threshold at p and its gradient magnitude (1/m).
    fn forestPatchThreshold(p:vec2<f32>)->vec2<f32>{
      let n=forestLatticeNoise(${SPLIT.patch},p,vec2<f32>(0.));
      let u=clamp(n.x*.6+.5,0.,1.);let k=min(i32(u*16.),15);
      return vec2<f32>(forestLookup17(u,0),(FOREST_PATCH_CDF[k+1]-FOREST_PATCH_CDF[k])*16.*.6*length(n.yz)/FOREST_PATCH);
    }
    struct ForestStand {p:vec2<f32>,cover:f32,height:f32,needle:f32,tone:f32,kappa:f32,trop:f32,dry:f32}
    fn forestDensity(c:f32,kappa:f32)->f32{return clamp(c+kappa*(.985-c),c,.985);}
    // Clumping: scattered savanna trees, solid closed stands, open conifer stands, continuous wet tropics.
    fn forestKappa(humid:f32,height:f32,needle:f32,trop:f32)->f32{
      return mix(.25,.78,groundSmooth(.9,1.8,humid))*mix(.55,1.,groundSmooth(3.,10.,height))*mix(1.,.6,needle)*(1.-.6*trop);
    }
    // 0 keeps Render's colour (>= 32 m/px), 1 the canonical forest (<= 2 m/px).
    fn forestReveal(fp:f32)->f32{return 1.-groundSmooth(1.,5.,log2(max(fp,.001)));}
    fn forestShrubShare(h:f32)->f32{return 1.-groundSmooth(3.,5.5,h);}
    // Shared corner gradients of the patch noise around one 3x3 crown neighbourhood.
    struct ForestLattice {base:vec2<i32>,g:array<vec2<f32>,9>}
    fn forestLatticeAround(e:i32,cell:vec2<i32>)->ForestLattice{
      var q:ForestLattice;q.base=r3CellAt(${SPLIT.patch},r3CellLocal(e,cell-vec2<i32>(1)));
      for(var j=0;j<3;j++){for(var i=0;i<3;i++){q.g[j*3+i]=forestGradient(q.base+vec2<i32>(i,j));}}
      return q;
    }
    fn forestLatticeValue(q:ForestLattice,p:vec2<f32>)->f32{
      let g=r3Point(${SPLIT.patch},p)-vec2<f32>(q.base-r3.cells[${SPLIT.patch}].xy);
      let c=clamp(vec2<i32>(floor(g)),vec2<i32>(0),vec2<i32>(1));let k=c.y*3+c.x;
      var corners=q.g;
      return clamp(forestPerlin(g-vec2<f32>(c),corners[k],corners[k+1],corners[k+3],corners[k+4]).x*.6+.5,0.,1.);
    }
    struct ForestCrown {c:vec2<f32>,r:f32,h:f32,depth:f32,needle:bool,alive:bool,edge:f32,id:u32,family:i32}
    // A candidate decides everything at its fixed world centre, including its
    // climate: fragment-varying inputs would slice a crown across an isoline.
    fn forestCrown(e:i32,cell:vec2<i32>,level:i32,lattice:ForestLattice)->ForestCrown{
      var k:ForestCrown;let seed=r3Seed()^0x2c1b3c6du;let s=r3.fracs[e].z;
      let h=groundHash(vec3<i32>(cell,level*2),seed);let h2=groundHash(vec3<i32>(cell,level*2+1),seed);k.id=h2;
      k.c=r3CellLocal(e,cell)+(.15+.7*vec2<f32>(forestU8(h,0u),forestU8(h,8u)))*s;
      let centre=r3Aux(k.c,2,0.);let cover=centre.r;let height=centre.g*50.;let needle=centre.b;
      let climate=forestClimate(r3Aux(k.c,3,0.));
      let density=forestDensity(cover,forestKappa(climate.humid,height,needle,climate.trop));let share=cover/max(density,1e-3);
      let radius=mix((.24*height+1.)*(1.+.45*climate.dry),.14*height+.9,needle);
      let centreLevel=clamp(2.*log2(radius/forestRho(forestTau(density))/2.),0.,5.999);
      let weight=max(0.,1.-abs(centreLevel-f32(level)));
      let tau=forestTau(1.-pow(1.-density,weight));let q=min(1.,tau/.55);let rho=forestRho(tau);
      let threshold=forestLookup17(.8*forestLatticeValue(lattice,k.c)+.2*forestU8(h2,0u),1);
      k.alive=forestU8(h,16u)<q&&threshold<share;
      // Harper et al. 2005: stand margins are denser and darker.
      k.edge=1.-groundSmooth(0.,.1,share-threshold);
      k.needle=false;k.r=0.;k.h=0.;k.depth=0.;k.family=1;
      if(k.alive){
        k.needle=forestU8(h2,8u)<needle;
        k.r=rho*s*(.8+.4*forestU8(h,24u))*(1.+.1*k.edge);
        let emergent=climate.trop*groundSmooth(.9,.97,forestU8(h2,24u));
        k.h=height*(.82+.36*forestU8(h2,16u))*(1.+.35*emergent);
        k.depth=k.r*select(.55,1.6,k.needle);
        // Brush family: pines, broadleaf, tropical, dry trees; shrubs follow physical height only.
        k.family=select(select(1,0,k.needle),2,climate.trop>.5);
        if(climate.dry>.35&&climate.annual>10.&&forestU8(k.id,16u)<.35){k.family=3;}
        let hs=k.id*0x85ebca6bu;
        if(forestU8(hs,8u)<forestShrubShare(k.h)){k.family=select(4,5,forestU8(hs,16u)<groundSmooth(.2,.7,climate.dry));}
      }
      return k;
    }
    // Outline radius along unit direction n: elliptic and trefoil harmonics, smaller for conifers.
    fn forestOutline(k:ForestCrown,n:vec2<f32>)->f32{
      let h=k.id*0x9e3779b9u;let a=vec2<f32>(forestU8(h,0u),forestU8(h,8u))*6.2831853;let amp=select(.13,.07,k.needle);
      let shape=vec4<f32>(cos(a.x),sin(a.x),.6*cos(a.y),.6*sin(a.y))*amp;
      let c2=vec2<f32>(n.x*n.x-n.y*n.y,2.*n.x*n.y);let c3=vec2<f32>(n.x*(4.*n.x*n.x-3.),n.y*(3.-4.*n.y*n.y));
      return k.r*(1.+dot(c2,shape.xy)+dot(c3,shape.zw));
    }
    struct ForestHit {cover:f32,top:f32,second:f32,slope:vec2<f32>,rel:f32,local:vec2<f32>,r:f32,needle:bool,edge:f32,id:u32,level:i32,near:f32,ink:f32}
    // Overhead painted crowns and shrubs: RG = coverage, centred luminance.
    fn forestBrush(k:ForestCrown,d:vec2<f32>,aa:f32,s:f32)->vec2<f32>{
      let extent=min(1.22*k.r,1.12*s);var uv=d/(2.*extent)+.5;
      if(any(uv<vec2<f32>(0.))||any(uv>vec2<f32>(1.))){return vec2<f32>(0.,.5);}
      // Mirror only: rotations would turn the painted light around.
      if((k.id&1u)==1u){uv.x=1.-uv.x;}
      let layer=k.family*8+i32((k.id>>4u)&7u);
      return textureSampleLevel(brushes,filtering,uv,layer,max(0.,log2(128.*aa/(2.*extent)))).rg;
    }
    fn forestCrowns(hit:ptr<function,ForestHit>,e:i32,level:i32,st:ForestStand,aa:f32,brush:bool){
      let cell=r3CellAt(e,st.p);let lattice=forestLatticeAround(e,cell);let s=r3.fracs[e].z;
      for(var j=-1;j<=1;j++){for(var i=-1;i<=1;i++){
        let k=forestCrown(e,cell+vec2<i32>(i,j),level,lattice);if(!k.alive){continue;}
        let d=st.p-k.c;let dist=length(d);
        if(dist>=2.*k.r+aa){continue;}
        var r=k.r;var a=0.;var ink=.5;
        if(brush){let t=forestBrush(k,d,aa,s);a=t.r;ink=t.g;(*hit).near=min((*hit).near,dist/r);if(a<.002){continue;}}
        else{r=forestOutline(k,d/max(dist,1e-4));(*hit).near=min((*hit).near,dist/r);if(dist>=r+aa){continue;}a=clamp((r-dist)/aa+.5,0.,1.);}
        let delta=min(dist/r,1.);
        let z=k.h-k.depth*select(delta*delta,sqrt(delta*delta+.015),k.needle);
        // Union of translucent layers (Boolean model behind the tau table).
        (*hit).cover=1.-(1.-(*hit).cover)*(1.-a);
        if(z>(*hit).top){
          (*hit).second=(*hit).top;(*hit).top=z;
          let g=select(1.1*delta,1.6*delta/sqrt(delta*delta+.015),k.needle);
          (*hit).slope=-g*d/max(dist,1e-4);(*hit).rel=delta;(*hit).local=d/r;(*hit).r=r;(*hit).needle=k.needle;
          (*hit).edge=k.edge;(*hit).id=k.id;(*hit).level=level;(*hit).ink=ink;
        } else {(*hit).second=max((*hit).second,z);}
      }}
    }
    // Soft cast shadow of the crowns around the sunward point onto height zp.
    fn forestShadow(hit:ForestHit,e:i32,level:i32,st:ForestStand,aa:f32,zp:f32,sun:vec2<f32>,cot:f32)->f32{
      let probe=st.p+sun*max(st.height*.75-zp,0.)*cot;
      let cell=r3CellAt(e,probe);let lattice=forestLatticeAround(e,cell);var shade=0.;
      for(var j=-1;j<=1;j++){for(var i=-1;i<=1;i++){
        let k=forestCrown(e,cell+vec2<i32>(i,j),level,lattice);
        if(!k.alive||(hit.level==level&&k.id==hit.id&&hit.cover>0.)){continue;}
        let caster=k.h-.5*k.depth;let o=(caster-zp)*cot;if(o<=0.){continue;}
        let penumbra=max(aa,.22*k.r+.055*o);
        let d=st.p-(k.c-sun*o);let sd=length(d);if(sd>=1.3*k.r+penumbra){continue;}
        shade=max(shade,clamp((.92*forestOutline(k,d/max(sd,1e-4))-sd)/penumbra+.5,0.,1.));
      }}
      return shade;
    }
    // Absolute climate palette of one crown type; only its ratios reach the display.
    fn forestPalette(st:ForestStand,needle:f32)->vec3<f32>{
      var broad=mix(vec3<f32>(.21,.31,.13),vec3<f32>(.16,.30,.11),st.trop);
      broad=mix(broad,vec3<f32>(.31,.32,.19),st.dry*(1.-.5*st.trop));
      return mix(broad,mix(vec3<f32>(.15,.25,.16),vec3<f32>(.20,.24,.15),st.dry),needle);
    }
    fn forestCanopy(st:ForestStand,needle:f32,jitter:vec2<f32>,special:f32)->vec3<f32>{
      var c=forestPalette(st,needle);
      // Young or disturbed stands lighter and yellower, old ones darker and bluer.
      let t=st.tone-.5;c*=1.+.34*t;
      c=mix(c,c*vec3<f32>(1.12,1.07,.78),max(t,0.)*.9);c=mix(c,c*vec3<f32>(.9,.97,1.08),max(-t,0.)*.7);
      let amp=mix(.09,.2,st.trop)*(1.-.45*needle);
      c*=1.+amp*jitter.x;
      c=mix(c,c*vec3<f32>(1.12,1.06,.72),max(jitter.y,0.)*amp*2.2);c=mix(c,c*vec3<f32>(.84,.97,1.12),max(-jitter.y,0.)*amp*1.6);
      // Tropical canopies: a few flushing, flowering or senescent crowns.
      let accent=select(select(vec3<f32>(.36,.38,.32),vec3<f32>(.40,.30,.22),special<.8),vec3<f32>(.40,.42,.19),special<.45);
      return mix(c,accent,st.trop*groundSmooth(.965,.99,fract(special*7.31))*.6);
    }
    // Crown colour as a residual of Render's own canopy colour (guide).
    fn forestCrownColor(st:ForestStand,guide:vec3<f32>,needle:f32,jitter:vec2<f32>,special:f32)->vec3<f32>{
      return guide*forestCanopy(st,needle,jitter,special)/max(forestPalette(st,st.needle),vec3<f32>(.02));
    }
    fn forestFloor(ground:vec3<f32>,st:ForestStand,cold:f32)->vec3<f32>{
      var litter=mix(vec3<f32>(.26,.21,.15),vec3<f32>(.24,.21,.15),st.needle);
      litter=mix(litter,vec3<f32>(.15,.24,.10),st.trop);
      // Boreal floors: moss in wet stands, pale lichen in dry ones.
      litter=mix(litter,mix(vec3<f32>(.27,.32,.18),vec3<f32>(.58,.58,.48),.7*st.dry),cold*st.needle);
      return mix(ground,litter,.65*(1.-.6*st.dry));
    }
    // Shadowed fraction s: remove the direct-light component of the normal.
    fn forestTilt(n:vec3<f32>,l:vec3<f32>,s:f32)->vec3<f32>{return normalize(mix(n,normalize(n-l*max(dot(n,l),0.)),clamp(s,0.,1.)));}
    // Crown-scale grain between the crown band and the mean, blended across two lattices.
    fn forestGrain(p:vec2<f32>,radius:f32)->f32{
      let level=clamp(2.*log2(1.6*radius/2.),-1.,5.999);let l0=floor(level);let w=level-l0;let e=${SPLIT.crown}+i32(l0);
      let a=forestLatticeNoise(e,p,vec2<f32>(17.3,-41.9)).x;let b=forestLatticeNoise(e+1,p,vec2<f32>(17.3,-41.9)).x;
      return mix(a,b,w);
    }
    fn applyForest(material:ptr<function,GroundMaterial>,p:vec2<f32>,footprint:f32,light:vec3<f32>,guide:RenderGuide,climate:ForestClimate,brush:bool){
      forestResolvedCover=0.;
      let fp=max(footprint,.02);var reveal=forestReveal(fp);
      // Partly revealed stands come from a coarser, mean-preserving mip: clearings start soft.
      let lod=max(log2(fp/params.viewport.w),0.);let standLod=lod+2.*(1.-reveal);let field=r3Aux(p,2,standLod);
      var st:ForestStand;st.p=p;st.cover=field.r;st.height=field.g*50.;st.needle=field.b;st.tone=field.a;
      st.trop=climate.trop;st.dry=climate.dry;st.kappa=forestKappa(climate.humid,st.height,st.needle,st.trop);
      // Render reference: its colour, macro woods included, while the forest is unresolved.
      // The open guide takes the same ground detail directly (Magna Urbis transfers
      // it as a colour ratio, which tints rock beside green woods).
      let openGround=groundRecolor(guide.open);
      let refAlbedo=(*material).albedo;
      let radius=mix((.24*st.height+1.)*(1.+.45*st.dry),.14*st.height+.9,st.needle);
      // Painted sprites need more pixels than an analytic crown before they are legible.
      let crownVis=select(groundSmooth(.9,2.,radius/fp),groundSmooth(2.,5.,radius/fp),brush);
      reveal*=mix(.06,1.,crownVis*crownVis);
      if(reveal<=0.){return;}
      let l=normalize(light);let sun=select(vec2<f32>(0.),normalize(l.xz),length(l.xz)>1e-4);let cot=clamp(length(l.xz)/max(l.y,.05),0.,4.);
      let q=p+sun*st.height*.6*cot;let fq=r3Aux(q,2,standLod);
      // No canonical tree here, in the next mip or in the sunward stand probe:
      // Render's macro woods give way to its open ground, without crown work.
      if(st.cover<.002&&fq.r<.002&&r3Aux(p,2,max(lod,1.)).r<.00001){(*material).albedo=mix(refAlbedo,groundToLinear(openGround),reveal);return;}
      let patchDetail=select(groundSmooth(.6,2.5,radius/fp),crownVis,brush);
      let patchVis=(1.-groundSmooth(6.,14.,fp))*mix(.18,1.,patchDetail);
      // Sparse arid cover resolves to a fine trace: taper its unresolved fringe.
      let fringe=(1.-groundSmooth(.38,.85,climate.humid))*(1.-crownVis);
      let cover=mix(st.cover,st.cover*st.cover,fringe);
      let d=forestDensity(cover,st.kappa);let share=cover/max(d,1e-3);
      let nGround=(*material).normal;
      let cq=mix(fq.r,fq.r*fq.r,fringe);let dq=forestDensity(cq,st.kappa);let sq=cq/max(dq,1e-3);
      var mask=share;var maskQ=sq;
      if(patchVis>0.){
        let pv=forestPatchThreshold(p);let pq=forestPatchThreshold(q);let feather=mix(.16,.015,crownVis);
        mask=mix(share,clamp((share-pv.x)/max(pv.y*fp,feather)+.5,0.,1.),patchVis);
        maskQ=mix(sq,clamp((sq-pq.x)/max(pq.y*fp,feather)+.5,0.,1.),patchVis);
      }
      let grainVis=(1.-crownVis)*groundSmooth(.35,1.2,radius/fp);var grain=0.;
      if(grainVis>0.){grain=forestGrain(p,radius)*grainVis;}
      let canopy=clamp(mask*d,0.,1.);let stand=mask;let groundShadow=clamp(maskQ*dq,0.,1.);
      let litter=forestFloor(openGround,st,climate.cold);
      let mean=forestCrownColor(st,guide.closed,st.needle,vec2<f32>(0.),0.);
      let selfShade=mix(.24,.42,st.needle);
      var albedo=mix(groundToLinear(mix(openGround,litter,stand)),groundToLinear(mean)*(1.+.5*grain),canopy);
      var normal=forestTilt(nGround,l,clamp(canopy*selfShade*(1.-.8*grain)+(1.-canopy)*groundShadow*.25,0.,1.));
      var occlusion=mix((*material).occlusion*mix(1.,.72,stand),.8,canopy);
      var roughness=mix((*material).roughness,.9,canopy);
      if(crownVis>0.){
        // Explicit crowns on two lattice levels whose union keeps cover d.
        let rho=forestRho(forestTau(d));
        let level=clamp(2.*log2(radius/rho/2.),0.,5.999);let l0=floor(level);let w=level-l0;let L=i32(l0);
        let d0=1.-pow(1.-d,1.-w);let d1=1.-pow(1.-d,w);let aa=fp*.75;
        var hit:ForestHit;hit.cover=0.;hit.top=-1e9;hit.second=-1e9;hit.slope=vec2<f32>(0.);hit.rel=1.;hit.local=vec2<f32>(0.);
        hit.r=1.;hit.needle=false;hit.edge=0.;hit.id=0u;hit.level=-1;hit.near=9.;hit.ink=.5;
        if(d0>.002){forestCrowns(&hit,${SPLIT.crown}+L,L,st,aa,brush);}
        if(d1>.002){forestCrowns(&hit,${SPLIT.crown}+L+1,L+1,st,aa,brush);}
        forestResolvedCover=hit.cover*crownVis*reveal;
        let zp=select(0.,hit.top,hit.cover>0.);var shadow=0.;
        // Brush motifs already contain their contact shade: no second projected copy.
        if(!brush){shadow=select(forestShadow(hit,${SPLIT.crown}+L+1,L+1,st,aa,zp,sun,cot),forestShadow(hit,${SPLIT.crown}+L,L,st,aa,zp,sun,cot),w<.5);}
        let hc=hit.id*0x9e3779b9u;
        let jitter=vec2<f32>(forestU8(hc,0u),forestU8(hc,8u))*2.-1.;let special=forestU8(hc,16u);let phase=forestU8(hc,24u)*6.2831853;
        let clumpVis=groundSmooth(2.5,7.,hit.r/fp);var clump=0.;var dClump=vec2<f32>(0.);
        if(brush){clump=(hit.ink-.5)*2.;}
        else if(hit.needle){
          // Straight radial branch ridges seen from above.
          let rel=length(hit.local);let angle=atan2(hit.local.y,hit.local.x)*7.+phase;clump=cos(angle)*rel;
          let radial=hit.local/max(rel,1e-3);let tangent=vec2<f32>(-radial.y,radial.x);
          dClump=(-sin(angle)*7.*tangent*rel/max(rel,.05)+cos(angle)*radial)/hit.r;
        } else {
          let a=hit.local*6.5;let ph=vec3<f32>(phase,phase*1.7,phase*2.3);
          let arg=vec3<f32>(a.x,.5*a.x+.866*a.y,-.5*a.x+.866*a.y)+ph;
          clump=(cos(arg.x)+cos(arg.y)+cos(arg.z))/3.;
          let sn=sin(arg);dClump=-6.5*vec2<f32>(sn.x+.5*sn.y-.5*sn.z,.866*(sn.y+sn.z))/(3.*hit.r);
          // Finer rotated leaf clusters once a crown spans many pixels.
          let fine=groundSmooth(8.,20.,hit.r/fp);
          if(fine>0.){
            let b=mat2x2<f32>(.62,-.78,.78,.62)*hit.local*15.;
            let arg2=vec3<f32>(b.x,.5*b.x+.866*b.y,-.5*b.x+.866*b.y)+ph.zxy*1.3;
            let sn2=sin(arg2);clump=mix(clump,.65*clump+.35*(cos(arg2.x)+cos(arg2.y)+cos(arg2.z))/3.,fine);
            dClump+=fine*.35*(-15./(3.*hit.r))*(mat2x2<f32>(.62,.78,-.78,.62)*vec2<f32>(sn2.x+.5*sn2.y-.5*sn2.z,.866*(sn2.y+sn2.z)));
          }
        }
        let slope=select(hit.slope+clumpVis*dClump*hit.r*select(.16,.14,hit.needle),.25*hit.slope,brush);
        let macroSlope=-nGround.xz/max(nGround.y,.2);
        // Shaded crowns keep their sky light.
        let nCrown=forestTilt(normalize(vec3<f32>(-(macroSlope.x+slope.x),1.,-(macroSlope.y+slope.y))),l,shadow*.3);
        let trough=select(1.,1.-.3*exp(-(hit.top-hit.second)/max(.25*hit.r,.1)),hit.second>-1e8);
        let occCrown=select(max((.6+.4*(1.-hit.rel*hit.rel))*trough*(1.-.25*clumpVis*max(-clump-.15,0.)),.5),.94,brush);
        let crown=forestCrownColor(st,guide.closed,select(0.,1.,hit.needle),jitter,special)*(1.+select(.12,.36,brush)*clumpVis*clump)*(1.-.12*hit.edge);
        let under=select(1.-groundSmooth(1.,1.8,hit.near),hit.cover,brush);
        let nearAlbedo=mix(groundToLinear(mix(openGround,litter,under)),groundToLinear(crown),hit.cover);
        let nearNormal=normalize(mix(forestTilt(nGround,l,shadow*.3),nCrown,hit.cover));
        let nearOcc=mix((*material).occlusion*mix(1.,.72,under),occCrown,hit.cover);
        let nearRough=mix((*material).roughness,.9,hit.cover);
        albedo=mix(albedo,nearAlbedo,crownVis);normal=normalize(mix(normal,nearNormal,crownVis));
        occlusion=mix(occlusion,nearOcc,crownVis);roughness=mix(roughness,nearRough,crownVis);
      }
      // The whole material response, not only its colour, leaves the Render reference.
      (*material).albedo=mix(refAlbedo,albedo,reveal);(*material).normal=normalize(mix((*material).normal,normal,reveal));
      (*material).occlusion=mix((*material).occlusion,occlusion,reveal);(*material).roughness=mix((*material).roughness,roughness,reveal);
    }
  `;

  // Pure population/material helpers also serve the coarse V2 expectation.
  // Reuse their exact definitions without world lattices or texture bindings.
  const forestMaterial=forest.slice(forest.indexOf('    struct ForestClimate'),forest.indexOf('    // Tile-local metres'))+
    forest.slice(forest.indexOf('    struct ForestStand'),forest.indexOf('    fn forestShrubShare'))+
    forest.slice(forest.indexOf('    fn forestPalette'),forest.indexOf('    // Shadowed fraction'));

  // Render3D quads: Render's baked albedo/light, then per-pixel ground and
  // forest from the exact world position as the footprint shrinks.
  const draw=`
    struct Params { rect:vec4<f32>, viewport:vec4<f32>, crop:vec4<f32>, field:vec4<f32>, lighting:vec4<f32>, sunlight:vec4<f32>, style:vec4<f32>, contours:vec4<f32>, render0:vec4<f32>, render1:vec4<f32>, render2:vec4<f32>, render3:vec4<f32>, render4:vec4<f32> }
    struct Vertex { @builtin(position) position:vec4<f32>, @location(0) uv:vec2<f32> }
    @group(0) @binding(0) var<uniform> params:Params;
    @group(0) @binding(1) var color:texture_2d<f32>;
    @group(0) @binding(2) var filtering:sampler;
    @group(0) @binding(3) var elevation:texture_2d<f32>;
    @group(0) @binding(4) var aux:texture_2d_array<f32>;
    @group(0) @binding(5) var<uniform> r3:R3Tile;
    @group(0) @binding(6) var brushes:texture_2d_array<f32>;
    ${common}
    ${ground}
    ${forest}
    fn r3Height(p:vec2<i32>)->f32{return textureLoad(elevation,clamp(p,vec2<i32>(0),vec2<i32>(textureDimensions(elevation))-1),0).r;}
    // Uniform cubic B-spline height and analytic gradient (metres, metres/metre).
    fn r3Surface(texel:vec2<f32>,r:f32)->vec3<f32>{
      let g=texel-.5;let fl=floor(g);let t=g-fl;let b=vec2<i32>(fl)-1;
      let t2=t*t;let t3=t2*t;let o=1.-t;
      let wx=vec4<f32>(o.x*o.x*o.x,3.*t3.x-6.*t2.x+4.,-3.*t3.x+3.*t2.x+3.*t.x+1.,t3.x)/6.;
      let wy=vec4<f32>(o.y*o.y*o.y,3.*t3.y-6.*t2.y+4.,-3.*t3.y+3.*t2.y+3.*t.y+1.,t3.y)/6.;
      let dx=vec4<f32>(-.5*o.x*o.x,1.5*t2.x-2.*t.x,-1.5*t2.x+t.x+.5,.5*t2.x);
      let dy=vec4<f32>(-.5*o.y*o.y,1.5*t2.y-2.*t.y,-1.5*t2.y+t.y+.5,.5*t2.y);
      var s=vec3<f32>(0.);
      for(var j=0;j<4;j++){
        var row=0.;var rowD=0.;
        for(var i=0;i<4;i++){let h=r3Height(b+vec2<i32>(i,j));row+=h*wx[i];rowD+=h*dx[i];}
        s+=vec3<f32>(row*wy[j],rowD*wy[j],row*dy[j]);
      }
      return vec3<f32>(s.x,s.y/r,s.z/r);
    }
    fn r3Water(depth:f32,ice:f32)->vec3<f32>{
      var water=mix(vec3<f32>(.10,.30,.36),vec3<f32>(.045,.16,.27),groundSmooth(5.,120.,depth));
      water=mix(water,vec3<f32>(.018,.065,.15),groundSmooth(150.,3000.,depth));
      return mix(water,vec3<f32>(.80,.84,.88),ice);
    }
    @vertex fn vertex(@builtin(vertex_index) id:u32)->Vertex{
      let corners=array<vec2<f32>,6>(vec2<f32>(0,0),vec2<f32>(1,0),vec2<f32>(0,1),vec2<f32>(0,1),vec2<f32>(1,0),vec2<f32>(1,1));
      let uv=corners[id];let xy=params.rect.xy+uv*params.rect.zw;
      var result:Vertex;result.position=vec4<f32>(xy.x/params.viewport.x*2.-1.,1.-xy.y/params.viewport.y*2.,0.,1.);result.uv=params.crop.xy+uv*params.crop.zw;return result;
    }
    @fragment fn fragment(input:Vertex)->@location(0) vec4<f32>{
      let size=params.field.xy;let halo=params.viewport.z;let r=params.viewport.w;
      let texel=input.uv*(size-2.*halo)+halo;let p=texel*r;
      let fp=max(length(dpdx(p)),length(dpdy(p)));
      let baked=textureSample(color,filtering,input.uv);
      let reveal3d=1.-groundSmooth(1.,5.,log2(max(fp,1e-3)));
      let s=r3Surface(texel,r);
      let auxUv=texel/size;let auxLod=max(log2(fp/r),0.);
      let closed=textureSampleLevel(aux,filtering,auxUv,1,auxLod);let cl=textureSampleLevel(aux,filtering,auxUv,3,auxLod);
      // Water and its antialiased shoreline come from the interpolated DEM at every scale.
      let water=clamp(r3Water(max(-s.x,0.),cl.a)*(closed.a*.4+.8),vec3<f32>(0.),vec3<f32>(1.));
      let landCover=clamp(s.x/max(length(s.yz)*fp,1e-3)+.5,0.,1.);
      let reference=baked.rgb*baked.a;
      if(reveal3d<=0.){return vec4<f32>(mix(water,reference,landCover),1.);}
      let open=textureSampleLevel(aux,filtering,auxUv,0,auxLod);
      let climate=forestClimate(cl);
      // Sea-level annual temperature and precipitation for the ground cover terms.
      let rain=climate.humid*max(20.*climate.annual+280.,120.);
      let ground=vec2<f32>(climate.annual+.0065*max(s.x,0.),rain);
      // Neural Earth's light vector shades (dH/dx, dH/dy, 1); this is the same light for true normals (X, up, Z).
      let l=normalize(vec3<f32>(-params.sunlight.x,params.sunlight.z,-params.sunlight.y));
      let angle=atan2(l.z,l.x)+2.35619449;let horizontal=length(l.xz)/.70710678;
      let fill=vec2<f32>(.2*cos(angle)+.75*sin(angle),.2*sin(angle)-.75*cos(angle))*horizontal;
      let l2=vec3<f32>(fill.x,.63*l.y/.70710678,fill.y);
      let relief=mix(6.*params.lighting.w,2.*params.lighting.w,reveal3d);
      var material=terrainGroundMaterial(vec3<f32>(max(s.x,1e-3),s.yz),p,fp,baked.rgb,ground,r3Seed(),relief);
      if(params.render1.z>.5){applyForest(&material,p,fp,l,RenderGuide(baked.rgb,open.rgb,closed.rgb,open.a),climate,params.field.z< -6.5);}
      let shaded=shadeTerrainGround(material,vec3<f32>(0.,1.,0.),l,l2,params.lighting.x);
      // Resolved crowns overhang the shore; distant means never hide water.
      return vec4<f32>(mix(water,mix(reference,shaded,reveal3d),max(landCover,forestResolvedCover)),1.);
    }
  `;

  // Mean-preserving box mips of all aux layers at once.
  const auxMip=`
    @group(0) @binding(0) var source:texture_2d_array<f32>;
    @group(0) @binding(1) var dest:texture_storage_2d_array<rgba8unorm,write>;
    @compute @workgroup_size(8,8,1) fn main(@builtin(global_invocation_id) id:vec3<u32>){
      if(any(id.xy>=textureDimensions(dest))){return;}
      let last=vec2<i32>(textureDimensions(source))-1;let p=vec2<i32>(id.xy)*2;let layer=i32(id.z);
      let sum=textureLoad(source,min(p,last),layer,0)+textureLoad(source,min(p+vec2<i32>(1,0),last),layer,0)
        +textureLoad(source,min(p+vec2<i32>(0,1),last),layer,0)+textureLoad(source,min(p+vec2<i32>(1,1),last),layer,0);
      textureStore(dest,vec2<i32>(id.xy),layer,sum*.25);
    }`;

  // Display-only controls, outside the generation and Render material settings.
  const defaults={enabled:true,density:1,clearings:.45};
  let settings={...defaults},changed=()=>{},sync=()=>{};
  function normalize(value){
    const s={...defaults};
    if(typeof value?.enabled==='boolean')s.enabled=value.enabled;
    if(Number.isFinite(value?.density))s.density=Math.max(0,Math.min(2,value.density));
    if(Number.isFinite(value?.clearings))s.clearings=Math.max(0,Math.min(1,value.clearings));
    return s;
  }
  function fromParams(params){
    const value={};
    if(params.has('forest'))value.enabled=params.get('forest')!=='0';
    for(const [key,name]of [['density','forestDensity'],['clearings','forestClearings']])if(params.has(name))value[key]=Number(params.get(name));
    return normalize(value);
  }
  // Links keep forest=0|1 while a Render3D layer is shown, and non-default values otherwise.
  function writeParams(params,shown=true){
    if(shown||!settings.enabled)params.set('forest',settings.enabled?'1':'0');else params.delete('forest');
    for(const [key,name]of [['density','forestDensity'],['clearings','forestClearings']]){if(settings[key]!==defaults[key])params.set(name,String(settings[key]));else params.delete(name);}
  }
  if(typeof location!=='undefined')try{settings=fromParams(new URLSearchParams(location.search));}catch(_){}
  function mount(container,onChange){
    changed=onChange;
    const field=document.createElement('fieldset');field.id='render3dSettings';field.hidden=true;
    field.innerHTML='<legend>Render3D · ground and forest</legend><label>Detailed canopy <input type="checkbox" id="forestEnabled"></label>'
      +'<label>Density (%) <span class="numericControl"><input type="range" id="forestDensity_range" min="0" max="200" step="5" aria-label="Forest density"><input type="number" id="forestDensity" min="0" max="200" step="5" aria-label="Forest density"></span></label>'
      +'<label>Clearings (%) <span class="numericControl"><input type="range" id="forestClearings_range" min="0" max="100" step="5" aria-label="Forest clearings"><input type="number" id="forestClearings" min="0" max="100" step="5" aria-label="Forest clearings"></span></label>'
      +'<small>Display only. Render stays the reference beyond about 32 m per pixel; ground grains, stands and crowns resolve progressively down to 1/128 m per pixel. Render3D Brush paints overhead crowns and shrubs.</small>';
    const anchor=container.querySelector('#renderMaterialSettings');if(anchor)anchor.after(field);else container.append(field);
    const enabled=field.querySelector('#forestEnabled');
    enabled.onchange=()=>{settings.enabled=enabled.checked;sync();changed();};
    for(const key of ['density','clearings'])for(const id of ['forest'+key[0].toUpperCase()+key.slice(1),'forest'+key[0].toUpperCase()+key.slice(1)+'_range']){
      const input=field.querySelector('#'+id);
      input.oninput=()=>{if(!Number.isFinite(input.valueAsNumber))return;settings=normalize({...settings,[key]:input.valueAsNumber/100});sync();changed();};
    }
    sync=()=>{
      enabled.checked=settings.enabled;
      for(const key of ['density','clearings'])for(const suffix of ['','_range'])field.querySelector('#forest'+key[0].toUpperCase()+key.slice(1)+suffix).value=Math.round(settings[key]*100);
    };
    sync();
    return {show:visible=>{field.hidden=!visible;}};
  }

  // Overhead canopy/shrub atlas from Magna Urbis, decoded once per page.
  let atlasPromise=null;
  function loadBrushes(base=''){
    atlasPromise??=(async()=>{
      const manifest=root.TerrainForestBrushes;if(!manifest)throw Error('Forest brush manifest unavailable');
      // A static asset beside the page, like the Fantasy sheets: never routed through the API transport.
      const response=await fetch(base+manifest.url);
      if(!response.ok)throw Error('Forest brushes unavailable ('+response.status+')');
      const stream=new Blob([await response.arrayBuffer()]).stream().pipeThrough(new DecompressionStream('deflate'));
      const pixels=new Uint8Array(await new Response(stream).arrayBuffer());
      if(pixels.length!==manifest.bytes)throw Error('Forest brush atlas has an unexpected size');
      // Box mips per layer, as WebGL generateMipmap: no neighbouring-sprite bleed.
      const levels=[pixels];
      for(let size=manifest.size;size>1;size>>=1){
        const source=levels.at(-1),half=size>>1,next=new Uint8Array(half*half*2*manifest.layers);
        for(let layer=0;layer<manifest.layers;layer++){
          const a=layer*size*size*2,b=layer*half*half*2;
          for(let y=0;y<half;y++)for(let x=0;x<half;x++)for(let c=0;c<2;c++){
            const i=a+((2*y)*size+2*x)*2+c;
            next[b+(y*half+x)*2+c]=(source[i]+source[i+2]+source[i+size*2]+source[i+size*2+2]+2)>>2;
          }
        }
        levels.push(next);
      }
      return {size:manifest.size,layers:manifest.layers,levels};
    })();
    return atlasPromise;
  }

  const api=Object.freeze({MODES,SPLIT,BAKE_SCALES,BANDS,isStyle,number:mode=>MODES[mode],tileUniform,withCaptures,captureState,
    shaders:{common,bake,ground,forest,draw,auxMip,lighting,forestMaterial,forestExpected},
    defaults:()=>({...defaults}),get:()=>({...settings}),set(value){settings=normalize(value);sync();},
    normalize,fromParams,writeParams,mount,loadBrushes,
    vectors:()=>[settings.enabled?1:0,settings.density,settings.clearings]});
  root.TerrainRender3D=api;
  if(typeof module!=='undefined'&&module.exports)module.exports=api;
})(typeof window==='undefined'?globalThis:window);

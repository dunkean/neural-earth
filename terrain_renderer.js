/* Physical terrain -> shaded tiles -> screen, entirely on WebGPU after one upload.
 * Canvas2D remains the independent fallback behind this transparent canvas.
 * No float32-filterable feature, CPU hillshade or per-frame GPU readback is used.
 */
(() => {
  'use strict';
  const gaussian = (sigma, radius) => {
    const values = Array.from({length: radius * 2 + 1}, (_, i) => Math.exp(-((i - radius) ** 2) / (2 * sigma * sigma)));
    const sum = values.reduce((a, b) => a + b, 0);
    return values.map(x => (x / sum).toFixed(12)).join(',');
  };
  const blurShader = `
    const LARGE = array<f32,49>(${gaussian(6, 24)});
    const SMALL = array<f32,11>(${gaussian(1.2, 5)});
    @group(0) @binding(0) var source: texture_2d<f32>;
    @group(0) @binding(1) var destination: texture_storage_2d<rg32float,write>;
    fn location(p: vec2<i32>) -> vec2<i32> {
      // Reflect outer halo as scipy.ndimage.gaussian_filter does.
      let size = vec2<i32>(textureDimensions(source));
      let reflected=select(p, -p-1, p<vec2<i32>(0));
      return clamp(select(reflected,2*size-reflected-1,reflected>=size),vec2<i32>(0),size-1);
    }
    @compute @workgroup_size(8,8) fn horizontal(@builtin(global_invocation_id) id: vec3<u32>) {
      if (any(id.xy>=textureDimensions(source))) { return; }
      let p=vec2<i32>(id.xy); var a=0.; var b=0.;
      for (var i=-24;i<=24;i++) { a+=textureLoad(source,location(p+vec2<i32>(i,0)),0).r*LARGE[u32(i+24)]; }
      for (var i=-5;i<=5;i++) { b+=textureLoad(source,location(p+vec2<i32>(i,0)),0).r*SMALL[u32(i+5)]; }
      textureStore(destination,p,vec4<f32>(a,b,0,0));
    }
    @compute @workgroup_size(8,8) fn vertical(@builtin(global_invocation_id) id: vec3<u32>) {
      if (any(id.xy>=textureDimensions(source))) { return; }
      let p=vec2<i32>(id.xy); var a=0.; var b=0.;
      for (var i=-24;i<=24;i++) { a+=textureLoad(source,location(p+vec2<i32>(0,i)),0).r*LARGE[u32(i+24)]; }
      for (var i=-5;i<=5;i++) { b+=textureLoad(source,location(p+vec2<i32>(0,i)),0).g*SMALL[u32(i+5)]; }
      textureStore(destination,p,vec4<f32>(a,b,0,0));
    }`;
  const city=window.TerrainStyleRendering;
  const modeNumber=mode=>mode==='render'?-3:mode==='soil'?-4:mode==='orogen-biomes'?-1:mode==='orogen-koppen'?-2:city?.number(mode)??({relief:0,biomes:1,temperature:2,precipitation:3}[mode]??0);
  // The compute and native coarse fragment paths compile this SAME material
  // function. It is independent of tile layouts and height sampling strategies.
  const materialShader=`
    fn surfaceMeta(i:i32)->f32{return textureLoad(climate,vec2<i32>(i,0),45,0).r;}
    fn surfacePoint(local:vec2<f32>,origin:vec2<f32>,spacing:f32)->vec3<f32>{
      let lo=vec2<f32>(surfaceMeta(0),surfaceMeta(1));let extent=vec2<f32>(surfaceMeta(2),surfaceMeta(3))-lo;
      if(surfaceMeta(4)<.5){return vec3<f32>(origin+(local+.5)*spacing,0.);}
      // Divide before adding small local offsets to retain sub-metre precision.
      let uv=(origin-lo)/extent+(local+.5)*(spacing/extent);
      let lon=uv.x*6.283185307-3.141592654;let lat=1.570796327-uv.y*3.141592654;
      var dir=vec3<f32>(cos(lat)*cos(lon),sin(lat),cos(lat)*sin(lon));
      if(surfaceMeta(5)>.5){dir=vec3<f32>(dir.x,-dir.z,dir.y);}
      return dir*(extent.x/6.283185307);
    }
    fn surfaceHash(c:vec3<i32>,salt:u32)->f32{
      let v=bitcast<vec3<u32>>(c);
      var h=(v.x*0x9e3779b9u)^(v.y*0x85ebca6bu)^(v.z*0xc2b2ae35u)^salt;
      h=(h^(h>>16u))*0x7feb352du;h=(h^(h>>15u))*0x846ca68bu;h=h^(h>>16u);
      return f32(h>>8u)/16777216.;
    }
    fn surfaceNorth(local:vec2<f32>,origin:vec2<f32>,spacing:f32)->vec2<f32>{
      if(surfaceMeta(4)<.5){return vec2<f32>(0.);}
      if(surfaceMeta(5)<.5){return vec2<f32>(0.,-1.);}
      let lo=vec2<f32>(surfaceMeta(0),surfaceMeta(1));let extent=vec2<f32>(surfaceMeta(2),surfaceMeta(3))-lo;
      let uv=(origin-lo)/extent+(local+.5)*(spacing/extent);
      let lon=uv.x*6.283185307-3.141592654;let lat=1.570796327-uv.y*3.141592654;
      // Geographic north of rotated (x,-z,y); analytic derivatives avoid four
      // extra trigonometric coordinate evaluations for every visible pixel.
      let d=vec2<f32>(-cos(lat)*cos(lon)*6.283185307/extent.x,-sin(lat)*sin(lon)*3.141592654/extent.y);
      return d/max(length(d),1e-12);
    }
    fn surfaceNoise(p:vec3<f32>,salt:u32)->f32{
      let c=vec3<i32>(floor(p));let q=fract(p);let f=q*q*(3.-2.*q);
      return mix(mix(mix(surfaceHash(c,salt),surfaceHash(c+vec3<i32>(1,0,0),salt),f.x),mix(surfaceHash(c+vec3<i32>(0,1,0),salt),surfaceHash(c+vec3<i32>(1,1,0),salt),f.x),f.y),
        mix(mix(surfaceHash(c+vec3<i32>(0,0,1),salt),surfaceHash(c+vec3<i32>(1,0,1),salt),f.x),mix(surfaceHash(c+vec3<i32>(0,1,1),salt),surfaceHash(c+vec3<i32>(1,1,1),salt),f.x),f.y),f.z);
    }
    fn surfaceOctave(p:vec3<f32>,scale:f32,offset:f32,footprint:f32)->f32{
      // An octave is skipped entirely above Nyquist, faded before it gets there.
      if(footprint>=scale*.5){return .5;}
      return .5+(surfaceNoise(p/scale+offset,u32(surfaceMeta(6))+u32(offset))-.5)*(1.-smoothstep(scale*.25,scale*.5,footprint));
    }
    const SURFACE_FAMILY=array<i32,31>(4,0,0,1,2,2,3,3,4,4,4,1,1,1,4,4,4,4,4,5,5,4,4,5,5,4,4,5,5,6,7);
    const SURFACE_PALETTE=array<vec4<f32>,8>(vec4<f32>(.12,.27,.13,.94),vec4<f32>(.35,.40,.19,.55),vec4<f32>(.65,.56,.37,.08),vec4<f32>(.47,.46,.26,.42),vec4<f32>(.20,.32,.16,.86),vec4<f32>(.16,.27,.19,.88),vec4<f32>(.40,.43,.28,.5),vec4<f32>(.39,.40,.37,0.));
    const SURFACE_DECIDUOUS=array<f32,8>(.08,.15,0.,.1,.72,.15,0.,0.);
    fn surfaceColor(id:i32,height:f32,slope:vec2<f32>,curvature:f32,point:vec3<f32>,north:vec2<f32>,footprint:f32,soil:vec3<f32>,substrateRock:vec3<f32>,humus:f32,ts:f32,tw:f32,rain:f32,alpine:f32,snowline:f32,water:vec2<f32>)->vec3<f32>{
      if(height<0.){return mix(vec3<f32>(.24,.40,.49),vec3<f32>(.035,.10,.22),smoothstep(0.,4500.,-height));}
      let broad=surfaceOctave(point,12000.,0.,footprint);let grove=surfaceOctave(point+(broad-.5)*vec3<f32>(900.,675.,-225.),1600.,11.,footprint);
      let holes=surfaceOctave(point,180.,23.,footprint);let fine=surfaceOctave(point,25.,37.,footprint);let micro=surfaceOctave(point,12.,53.,footprint);
      let family=SURFACE_FAMILY[clamp(id,0,30)];let fam=SURFACE_PALETTE[family];
      let phase=.5-.5*cos(6.283185307*params.render0.x);let temp=mix(tw,ts,phase);
      let moisture=clamp(rain/2200.,0.,1.);let grade=length(slope);let convex=clamp(curvature,-1.,1.);
      let wet=clamp(water.y,0.,1.);let bank=clamp(water.x,0.,1.);
      let herbs=smoothstep(-6.,8.,max(ts,tw))*moisture*(.35+.25*humus)*(1.-smoothstep(.45,1.4,grade));
      let ground=mix(soil*(1.-.2*wet-.08*bank),vec3<f32>(.40,.43,.25),herbs);var rock=substrateRock*params.render3.xyz;
      var vegetation=fam.xyz*params.render2.xyz;
      let season=fract(params.render0.x+select(0.,.5,ts<tw));
      let autumn=smoothstep(.58,.72,season)*(1.-smoothstep(.83,.97,season))*SURFACE_DECIDUOUS[family];
      vegetation=mix(vegetation,vec3<f32>(.47,.34,.15),autumn*.45);
      let dry=clamp((temp-12.)/22.,0.,1.)*(1.-moisture);
      vegetation=mix(vegetation,vec3<f32>(.51,.46,.25),dry*.35);
      let forestAlt=(1.-smoothstep(max(alpine*1000.-500.,0.),max(alpine*1000.+500.,1000.),height))*smoothstep(0.,12.,max(ts,tw));
      let density=clamp(fam.w*params.render0.z*(.72+.28*moisture)*forestAlt*(.65+.7*broad)+.1*bank,0.,1.);
      let clearing=smoothstep(.19,.74,.62*grove+.38*holes+.2*(broad-.5));
      let cover=clamp(density*smoothstep(.12,.72,clearing+density*.3-.16)*(1.-smoothstep(.45,1.4,grade)),0.,1.);
      var color=mix(ground,vegetation,cover);
      let exposed=smoothstep(params.render0.w*.65,params.render0.w*1.5,grade);
      let highland=smoothstep(max(alpine*1000.,0.),max(alpine*1000.+1600.,1600.),height)*.65;
      let rockCover=clamp(max(exposed,highland)+.12*convex+.12*(holes-.5),0.,1.);
      rock*=1.+params.render0.y*(.12*(broad-.5)+.08*(fine-.5));color=mix(color,rock,rockCover);
      let aspect=dot(slope,north)/max(grade,.00001)*sign(point.y);
      let cold=1.-smoothstep(-5.,3.,temp);
      let localSnowline=snowline*1000.+(holes-.5)*300.+(broad-.5)*400.;
      let altitude=smoothstep(max(localSnowline-250.,0.),max(localSnowline+700.,700.),height);
      let potential=clamp(max(cold*.85,altitude)*params.render1.x*(1.+.15*aspect-.12*convex+.25*(holes-.5)),0.,1.);
      let slip=.65+.55*altitude+.25*cold;let retention=1.-smoothstep(slip,slip+1.4,grade);
      let snow=clamp(potential*retention*(1.-.2*wet),0.,1.);color=mix(color,params.render4.xyz,snow);
      color*=1.+params.render0.y*(.15*(broad-.5)+.12*(grove-.5)+.08*(fine-.5)+.04*(micro-.5));
      return clamp(color,vec3<f32>(0.),vec3<f32>(1.));
    }
  `;
  const shadeShader = `
    struct Params { resolution:f32, halo:f32, width:f32, height:f32, mode:f32, pad0:f32, pad1:f32, pad2:f32, lighting:vec4<f32>, sunlight:vec4<f32>, style:vec4<f32>, render0:vec4<f32>, render1:vec4<f32>, render2:vec4<f32>, render3:vec4<f32>, render4:vec4<f32> }
    @group(0) @binding(0) var elevation: texture_2d<f32>;
    @group(0) @binding(1) var blurred: texture_2d<f32>;
    @group(0) @binding(2) var color: texture_storage_2d<rgba8unorm,write>;
    @group(0) @binding(3) var<uniform> params:Params;
    @group(0) @binding(4) var climate:texture_2d_array<f32>;
    fn clim(p:vec2<i32>,layer:i32)->f32 {
      let sourceSize=vec2<f32>(textureDimensions(elevation))-vec2<f32>(1.);
      let size=vec2<i32>(textureDimensions(climate));
      let q=vec2<f32>(p)/sourceSize*vec2<f32>(size-1);
      let lo=vec2<i32>(floor(q));let hi=min(lo+vec2<i32>(1),size-1);let f=fract(q);
      return mix(mix(textureLoad(climate,lo,layer,0).r,textureLoad(climate,vec2<i32>(hi.x,lo.y),layer,0).r,f.x),mix(textureLoad(climate,vec2<i32>(lo.x,hi.y),layer,0).r,textureLoad(climate,hi,layer,0).r,f.x),f.y);
    }
    fn surfaceHeight(q:vec2<f32>,level:f32)->f32{return textureLoad(elevation,vec2<i32>(q),0).r;}
    fn shade(d:vec2<f32>) -> f32 {
      // Same directional light/vertical exaggeration as upstream relief_map.py.
      let normal=normalize(vec3<f32>(d.x*params.lighting.w,d.y*params.lighting.w,1.));
      return clamp(dot(normal,params.sunlight.xyz),0.,1.);
    }
    fn terrain(t:f32) -> vec3<f32> {
      if (t<.5) { return mix(vec3<f32>(0.,.8,.4),vec3<f32>(1.,1.,.6),(t-.25)*4.); }
      if (t<.75) { return mix(vec3<f32>(1.,1.,.6),vec3<f32>(.5,.36,.33),(t-.5)*4.); }
      return mix(vec3<f32>(.5,.36,.33),vec3<f32>(1.),(t-.75)*4.);
    }
    // Lookup planes contain exact class palettes and the user's thresholds.
    // Only the continuous seasonal fields are spatially interpolated.
    fn koppenSetting(index:i32)->f32{return textureLoad(climate,vec2<i32>(index,0),35,0).r;}
    fn classField(id:i32,layer:i32)->f32{return textureLoad(climate,vec2<i32>(id,0),layer,0).r;}
    fn koppen(ts:f32,tw:f32,ps:f32,pw:f32)->i32{
      let hot=max(ts,tw);let cold=min(ts,tw);let annual=(ts+tw)*.5;
      if(hot<koppenSetting(0)){return 30;}
      if(hot<koppenSetting(1)){return 29;}
      let shoulder=hot-(hot-cold)*koppenSetting(20);
      let summer=select(pw,ps,ts>=tw);let winter=select(ps,pw,ts>=tw);let total=summer+winter;
      var fraction=.5;if(total>0.){fraction=summer/total;}
      var extra=koppenSetting(10);
      if(fraction>=koppenSetting(11)){extra=koppenSetting(9);}
      else if(fraction<=koppenSetting(12)){extra=0.;}
      let threshold=max(0.,koppenSetting(8)*annual+extra);
      if(total<threshold){return select(6,4,total<threshold*koppenSetting(13))+select(0,1,annual<koppenSetting(7));}
      if(cold>=koppenSetting(2)){
        let dry=min(summer,winter)/6.;
        if(dry>=koppenSetting(17)){return 1;}
        return select(3,2,total>=koppenSetting(18)*(koppenSetting(19)-dry));
      }
      var pattern=0;
      if(summer<winter&&summer/6.<koppenSetting(14)&&summer<winter/koppenSetting(15)){pattern=1;}
      else if(summer>=winter&&winter<summer/koppenSetting(16)){pattern=2;}
      var letter=3;
      if(hot>=koppenSetting(4)){letter=0;}else if(shoulder>=koppenSetting(5)){letter=1;}
      else if(cold>=koppenSetting(6)){letter=2;}
      if(cold>=koppenSetting(3)){return select(9,8+pattern*3+letter,letter<3);}
      return 17+pattern*4+letter;
    }
    ${city?.shader||''}
    ${materialShader}
    @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3<u32>) {
      if (any(id.xy>=textureDimensions(color))) { return; }
      let p=vec2<i32>(id.xy)+vec2<i32>(i32(params.halo));
      let height=textureLoad(elevation,p,0).r;
      let level=0.;
      let temp=clim(p,0)+clim(p,4)*max(height,0.);
      let rain=max(clim(p,2),0.);
      var rgb:vec3<f32>;
      var reliefLight=1.;
      if (height<0.) {
        let depth=pow(clamp(-height/10000.,0.,1.),.7);
        rgb=mix(vec3<f32>(.68,.88,1.),vec3<f32>(0.,.1,.45),depth);
      } else {
        let dx=(textureLoad(blurred,p+vec2<i32>(1,0),0).rg-textureLoad(blurred,p-vec2<i32>(1,0),0).rg)*3./params.resolution;
        let dy=(textureLoad(blurred,p+vec2<i32>(0,1),0).rg-textureLoad(blurred,p-vec2<i32>(0,1),0).rg)*3./params.resolution;
        let hs=pow(clamp(.75*shade(vec2<f32>(dx.r,dy.r))+.25*shade(vec2<f32>(dx.g,dy.g)),0.,1.),.85*params.lighting.z);
        reliefLight=mix(1.,params.lighting.y+(1.-params.lighting.y)*hs,params.lighting.x);
        rgb=terrain(.25+.75*pow(clamp(height/4500.,0.,1.),.7))*reliefLight;
      }
      var climateClass=0;
      if((params.mode == -1. || params.mode == -2. || params.mode == -3.) && textureNumLayers(climate)>=36u){
        // Render applies altitude continuously in its masks. Regional material
        // families use the baseline climate to avoid ET/EF snowline feedback.
        let classificationHeight=select(max(height,0.),0.,params.mode == -3.);
        climateClass=koppen(clamp(clim(p,21)-clim(p,23)*classificationHeight,-45.,45.),clamp(clim(p,24)-clim(p,26)*classificationHeight,-45.,45.),max(clim(p,22),0.),max(clim(p,25),0.));
        if(height<0.){climateClass=0;}
      }
      if((params.mode == -3. || params.mode == -4.) && textureNumLayers(climate)>=46u){
        let soil=vec3<f32>(clim(p,36),clim(p,37),clim(p,38));
        if(params.mode == -4.){rgb=select(soil,vec3<f32>(.06,.17,.27),height<0.);}
        else{
          let local=vec2<f32>(p);let point=surfacePoint(local,params.style.xy,params.resolution);
          let sx=surfaceHeight(local+vec2<f32>(1,0),level);let nx=surfaceHeight(local-vec2<f32>(1,0),level);
          let sy=surfaceHeight(local+vec2<f32>(0,1),level);let ny=surfaceHeight(local-vec2<f32>(0,1),level);
          let slopeX=(sx-nx)*.5/params.resolution;let slopeY=(sy-ny)*.5/params.resolution;
          let convex=(4.*height-sx-nx-sy-ny)/params.resolution;
          let north=surfaceNorth(local,params.style.xy,params.resolution);
          let materialFootprint=params.resolution;
          rgb=surfaceColor(climateClass,height,vec2<f32>(slopeX,slopeY),convex,point,north,materialFootprint,soil,vec3<f32>(clim(p,39),clim(p,40),clim(p,41)),clim(p,44),clamp(clim(p,21)-clim(p,23)*max(height,0.),-45.,45.),clamp(clim(p,24)-clim(p,26)*max(height,0.),-45.,45.),max(clim(p,22),0.)+max(clim(p,25),0.),classField(climateClass,30),classField(climateClass,31),vec2<f32>(0.));
          rgb*=select(reliefLight,1.,height<0.);
        }
      } else if(params.mode == -2. && textureNumLayers(climate)>=36u){
        rgb=vec3<f32>(classField(climateClass,32),classField(climateClass,33),classField(climateClass,34));
      } else if (params.mode == -1. && textureNumLayers(climate)>=21u) {
        let h=max(height,0.)/1000.;
        var alpine=clim(p,8);var snow=clim(p,9);
        let low=max(clim(p,10),.000001);
        let rock=vec3<f32>(clim(p,15),clim(p,16),clim(p,17));
        let snowColor=vec3<f32>(clim(p,18),clim(p,19),clim(p,20));
        var biome=vec3<f32>(clim(p,5),clim(p,6),clim(p,7));
        if(textureNumLayers(climate)>=36u){
          biome=vec3<f32>(classField(climateClass,27),classField(climateClass,28),classField(climateClass,29));
          alpine=classField(climateClass,30);snow=classField(climateClass,31);
        }
        if(h<low){biome*=1.-clim(p,11)+clim(p,11)*h/low;}
        if(alpine>0.&&h>low&&h<alpine){biome*=1.-(h-low)/max(alpine-low,.000001)*clim(p,12);}
        if(alpine>0.&&h>alpine){
          let span=select(clim(p,13),snow-alpine,snow>alpine);
          let t=clamp((h-alpine)/max(span,.000001),0.,1.);
          biome=mix(biome,rock,t*t);
        }
        if(snow>0.&&h>snow){let t=clamp((h-snow)/max(clim(p,14),.000001),0.,1.);biome=mix(biome,snowColor,t*t);}
        // Apply hillshade directly: deriving light from the white altitude
        // palette and clipping it would erase shadows on snowy mountains.
        rgb=biome*reliefLight;
        if(height<0.){
          let e=height/10000.;
          rgb=select(vec3<f32>(.11,.20,.48)+clamp((e+.1)/.1,0.,1.)*vec3<f32>(.19,.22,.12),vec3<f32>(.04,.06,.30)+clamp((e+.5)/.4,0.,1.)*vec3<f32>(.07,.14,.18),e<-.1);
        }
      } else if (params.mode==2.) {
        rgb=mix(vec3<f32>(.18,.38,.88),vec3<f32>(.95,.22,.08),clamp((temp+35.)/70.,0.,1.));
      } else if (params.mode==3.) {
        rgb=mix(vec3<f32>(.78,.61,.32),vec3<f32>(.08,.36,.72),clamp(log(1.+rain)/log(4001.),0.,1.));
      } else if (params.mode==1.) {
        let season=max(clim(p,1),0.)/100.;
        let effectiveRain=rain/(1.+max(clim(p,3),0.)/200.);
        let dry=clamp((450.+max(temp,0.)*25.+season*8.-effectiveRain)/650.,0.,1.);
        var biome=mix(vec3<f32>(.42,.57,.28),vec3<f32>(.08,.35,.20),clamp(effectiveRain/2500.,0.,1.));
        biome=mix(biome,vec3<f32>(.82,.69,.42),dry);
        biome=mix(biome,vec3<f32>(.52,.48,.43),clamp((height-2300.)/2200.,0.,1.));
        biome=mix(biome,vec3<f32>(.93,.97,.99),clamp((1.5-temp)/7.,0.,1.));
        let light=mix(1.,clamp(dot(rgb,vec3<f32>(.333333333))*1.8,.55,1.),params.lighting.x);
        rgb=biome*light;
        if(height<0.) {
          let ocean=vec3<f32>(.10,.30,.50)+clamp(1.+height/6000.,0.,1.)*vec3<f32>(.18,.24,.20);
          rgb=mix(ocean,vec3<f32>(.82,.93,.98),clamp((-2.-temp)/10.,0.,1.));
        }
      }
      ${city?`if(params.mode>=4.){
        let deltaX=(textureLoad(blurred,p+vec2<i32>(1,0),0).rg-textureLoad(blurred,p-vec2<i32>(1,0),0).rg)*.5/params.resolution;
        let deltaY=(textureLoad(blurred,p+vec2<i32>(0,1),0).rg-textureLoad(blurred,p-vec2<i32>(0,1),0).rg)*.5/params.resolution;
        let hill=.75*dot(normalize(vec3<f32>(-deltaX.r*params.lighting.w,-deltaY.r*params.lighting.w,1.)),params.sunlight.xyz)+.25*dot(normalize(vec3<f32>(-deltaX.g*params.lighting.w,-deltaY.g*params.lighting.w,1.)),params.sunlight.xyz);
        // Grain and hatching follow terrain samples, including experimental fine LODs.
        // Divide the world origin first so small local offsets retain f32 precision.
        rgb=cityColor(height,params.mode,hill,floor(params.style.xy/params.resolution+vec2<f32>(p)+.5));
      }`:''}
      textureStore(color,vec2<i32>(id.xy),vec4<f32>(rgb,1.));
    }`;
  const contourShader = `
    struct Params {rect:vec4<f32>,viewport:vec4<f32>,crop:vec4<f32>,field:vec4<f32>,lighting:vec4<f32>,sunlight:vec4<f32>,style:vec4<f32>,contours:vec4<f32>}
    struct Vertex {@builtin(position) position:vec4<f32>,@location(0) local:vec2<f32>,@location(1) @interpolate(flat) length:f32,@location(2) @interpolate(flat) index:f32,@location(3) uv:vec2<f32>}
    @group(0) @binding(0) var<uniform> params:Params;
    ${city?.contourInk||'fn contourInk(mode:f32)->vec4<f32>{return vec4<f32>(.72,.46,.23,.75);} fn contourWidth(mode:f32,index:f32)->f32{return select(.9,1.8,index>0.);}' }
    @vertex fn vertex(@builtin(vertex_index) id:u32,@location(0) ends:vec4<f32>,@location(1) index:f32)->Vertex{
      let a=params.rect.xy+(ends.xy-params.crop.xy)/params.crop.zw*params.rect.zw;
      let b=params.rect.xy+(ends.zw-params.crop.xy)/params.crop.zw*params.rect.zw;
      let len=length(b-a);let direction=(b-a)/max(len,.00001);let normal=vec2<f32>(-direction.y,direction.x);
      let halfWidth=contourWidth(params.field.z,index)*params.contours.z/.9*.5;let margin=halfWidth+1.;
      let corners=array<vec2<f32>,6>(vec2<f32>(0,-1),vec2<f32>(1,-1),vec2<f32>(0,1),vec2<f32>(0,1),vec2<f32>(1,-1),vec2<f32>(1,1));
      let corner=corners[id];let local=vec2<f32>(mix(0.,len,corner.x),corner.y*margin);
      let xy=a+direction*local.x+normal*local.y;
      var out:Vertex;out.position=vec4<f32>(xy.x/params.viewport.x*2.-1.,1.-xy.y/params.viewport.y*2.,0.,1.);out.local=local;out.length=len;out.index=index;out.uv=params.crop.xy+(xy-params.rect.xy)/params.rect.zw*params.crop.zw;return out;
    }
    @fragment fn fragment(input:Vertex)->@location(0) vec4<f32>{
      if(any(input.uv<max(params.crop.xy,vec2<f32>(0.)))||any(input.uv>min(params.crop.xy+params.crop.zw,vec2<f32>(1.)))){discard;}
      let distance=length(vec2<f32>(max(max(-input.local.x,input.local.x-input.length),0.),input.local.y));
      let halfWidth=contourWidth(params.field.z,input.index)*params.contours.z/.9*.5;
      let ink=contourInk(params.field.z);let alpha=(1.-smoothstep(max(0.,halfWidth-.5),halfWidth+.5,distance))*ink.w*select(.7,1.,input.index>0.);
      return vec4<f32>(ink.xyz,alpha);
    }`;
  const renderShader = `
    struct Params { rect:vec4<f32>, viewport:vec4<f32>, crop:vec4<f32>, field:vec4<f32>, lighting:vec4<f32>, sunlight:vec4<f32>, style:vec4<f32>, contours:vec4<f32> }
    struct Vertex { @builtin(position) position:vec4<f32>, @location(0) uv:vec2<f32> }
    @group(0) @binding(0) var<uniform> params:Params;
    @group(0) @binding(1) var color:texture_2d<f32>;
    @group(0) @binding(2) var filtering:sampler;
    @group(0) @binding(3) var elevation:texture_2d<f32>;
    ${city?.paperShader||''}
    @vertex fn vertex(@builtin(vertex_index) id:u32) -> Vertex {
      let corners=array<vec2<f32>,6>(vec2<f32>(0,0),vec2<f32>(1,0),vec2<f32>(0,1),vec2<f32>(0,1),vec2<f32>(1,0),vec2<f32>(1,1));
      let uv=corners[id]; let xy=params.rect.xy+uv*params.rect.zw;
      var result:Vertex; result.position=vec4<f32>(xy.x/params.viewport.x*2.-1.,1.-xy.y/params.viewport.y*2.,0.,1.); result.uv=params.crop.xy+uv*params.crop.zw; return result;
    }
    @fragment fn fragment(input:Vertex) -> @location(0) vec4<f32> {
      let sampled=textureSample(color,filtering,input.uv);
      ${city?`if(paperAmount(params.field.z)>0.){
        let p=vec2<i32>(input.uv*(params.field.xy-2.*params.viewport.z)+params.viewport.z);
        let height=textureLoad(elevation,clamp(p,vec2<i32>(0),vec2<i32>(textureDimensions(elevation))-1),0).r;
        return vec4<f32>(paperColor(sampled.rgb,params.field.z,height,input.position.xy),sampled.a);
      }`:''}
      return sampled;
    }`;

  // Coarse stays in its native signed-sqrt representation. Mips average
  // physical metres, then encode again; averaging signed roots is incorrect.
  const coarseMipShader=`
    @group(0) @binding(0) var source:texture_2d<f32>;
    @group(0) @binding(1) var dest:texture_storage_2d<r32float,write>;
    fn metres(v:f32)->f32{return sign(v)*v*v;}
    @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3<u32>){
      if(any(id.xy>=textureDimensions(dest))){return;}
      let p=vec2<i32>(id.xy)*2;let hi=vec2<i32>(textureDimensions(source))-1;
      let h=(metres(textureLoad(source,p,0).r)+metres(textureLoad(source,min(p+vec2<i32>(1,0),hi),0).r)+metres(textureLoad(source,min(p+vec2<i32>(0,1),hi),0).r)+metres(textureLoad(source,min(p+vec2<i32>(1,1),hi),0).r))*.25;
      textureStore(dest,vec2<i32>(id.xy),vec4<f32>(sign(h)*sqrt(abs(h)),0,0,0));
    }`;
  const coarsePalette=shadeShader.slice(shadeShader.indexOf('    fn shade('),shadeShader.indexOf('    @compute'));
  const coarseColor=shadeShader.slice(shadeShader.indexOf('      let temp='),shadeShader.indexOf('      textureStore(color'))
    .replaceAll('params.mode','params.field.z')
    .replace('let dx=(textureLoad(blurred,p+vec2<i32>(1,0),0).rg-textureLoad(blurred,p-vec2<i32>(1,0),0).rg)*3./params.resolution;',
      'let dx=(heightAt(p+vec2<f32>(1,0),level)-heightAt(p-vec2<f32>(1,0),level))*3./params.viewport.w;')
    .replace('let dy=(textureLoad(blurred,p+vec2<i32>(0,1),0).rg-textureLoad(blurred,p-vec2<i32>(0,1),0).rg)*3./params.resolution;',
      'let dy=(heightAt(p+vec2<f32>(0,1),level)-heightAt(p-vec2<f32>(0,1),level))*3./params.viewport.w;')
    .replaceAll('*params.resolution','*params.viewport.w')
    .replace('let deltaX=(textureLoad(blurred,p+vec2<i32>(1,0),0).rg-textureLoad(blurred,p-vec2<i32>(1,0),0).rg)*.5/params.resolution;','let deltaX=vec2<f32>((heightAt(p+vec2<f32>(1,0),level)-heightAt(p-vec2<f32>(1,0),level))*.5/params.viewport.w);')
    .replace('let deltaY=(textureLoad(blurred,p+vec2<i32>(0,1),0).rg-textureLoad(blurred,p-vec2<i32>(0,1),0).rg)*.5/params.resolution;','let deltaY=vec2<f32>((heightAt(p+vec2<f32>(0,1),level)-heightAt(p-vec2<f32>(0,1),level))*.5/params.viewport.w);')
    .replaceAll('/params.resolution','/params.viewport.w')
    .replace('vec2<f32>(p)','p')
    .replace('let materialFootprint=params.resolution;', 'let materialFootprint=max(footprint*params.viewport.w,.01);')
    .replaceAll('params.resolution','params.viewport.w')
    .replace('let hs=pow(clamp(.75*shade(vec2<f32>(dx.r,dy.r))+.25*shade(vec2<f32>(dx.g,dy.g)),0.,1.),.85*params.lighting.z);',
      'let hs=pow(shade(vec2<f32>(dx,dy)),.85*params.lighting.z);');
  const coarseShader=`
    struct Params {rect:vec4<f32>,viewport:vec4<f32>,crop:vec4<f32>,field:vec4<f32>,lighting:vec4<f32>,sunlight:vec4<f32>,style:vec4<f32>,contours:vec4<f32>,render0:vec4<f32>,render1:vec4<f32>,render2:vec4<f32>,render3:vec4<f32>,render4:vec4<f32>}
    struct Vertex {@builtin(position) position:vec4<f32>,@location(0) uv:vec2<f32>}
    @group(0) @binding(0) var<uniform> params:Params;
    @group(0) @binding(1) var elevation:texture_2d<f32>;
    @group(0) @binding(2) var climate:texture_2d_array<f32>;
    @vertex fn vertex(@builtin(vertex_index) id:u32)->Vertex{
      let corners=array<vec2<f32>,6>(vec2<f32>(0,0),vec2<f32>(1,0),vec2<f32>(0,1),vec2<f32>(0,1),vec2<f32>(1,0),vec2<f32>(1,1));
      let uv=corners[id];let xy=params.rect.xy+uv*params.rect.zw;
      var out:Vertex;out.position=vec4<f32>(xy.x/params.viewport.x*2.-1.,1.-xy.y/params.viewport.y*2.,0.,1.);out.uv=params.crop.xy+uv*params.crop.zw;return out;
    }
    fn load(p:vec2<i32>,level:i32)->f32{return textureLoad(elevation,clamp(p,vec2<i32>(0),vec2<i32>(textureDimensions(elevation,level))-1),level).r;}
    fn slope(a:f32,b:f32)->f32{if(a*b<=0.){return 0.;}return 2.*a*b/(a+b);}
    fn cubic(a:f32,b:f32,c:f32,d:f32,t:f32)->f32{
      let m=slope(b-a,c-b);let n=slope(c-b,d-c);let t2=t*t;let t3=t2*t;
      return clamp((2.*t3-3.*t2+1.)*b+(t3-2.*t2+t)*m+(-2.*t3+3.*t2)*c+(t3-t2)*n,min(b,c),max(b,c));
    }
    fn rootAt(p:vec2<f32>,level:i32)->f32{
      let q=(p+.5)/exp2(f32(level))-.5;let lo=vec2<i32>(floor(q));let f=fract(q);
      if(params.field.w==1.&&level==0){
        var rows:array<f32,4>;
        for(var j=0;j<4;j++){let y=lo.y+j-1;rows[j]=cubic(load(vec2<i32>(lo.x-1,y),level),load(vec2<i32>(lo.x,y),level),load(vec2<i32>(lo.x+1,y),level),load(vec2<i32>(lo.x+2,y),level),f.x);}
        return cubic(rows[0],rows[1],rows[2],rows[3],f.y);
      }
      return mix(mix(load(lo,level),load(lo+vec2<i32>(1,0),level),f.x),mix(load(lo+vec2<i32>(0,1),level),load(lo+vec2<i32>(1,1),level),f.x),f.y);
    }
    fn physicalAt(p:vec2<f32>,level:i32)->f32{let v=rootAt(p,level);return sign(v)*v*v;}
    fn heightAt(p:vec2<f32>,level:f32)->f32{let lo=i32(floor(level));return mix(physicalAt(p,lo),physicalAt(p,min(lo+1,3)),fract(level));}
    fn surfaceHeight(q:vec2<f32>,level:f32)->f32{return heightAt(q,level);}
    fn clim(p:vec2<f32>,layer:i32)->f32{
      let size=vec2<i32>(textureDimensions(climate));let q=clamp(p/params.field.xy,vec2<f32>(0.),vec2<f32>(1.))*vec2<f32>(size-1);
      let lo=vec2<i32>(floor(q));let hi=min(lo+1,size-1);let f=fract(q);
      return mix(mix(textureLoad(climate,lo,layer,0).r,textureLoad(climate,vec2<i32>(hi.x,lo.y),layer,0).r,f.x),mix(textureLoad(climate,vec2<i32>(lo.x,hi.y),layer,0).r,textureLoad(climate,hi,layer,0).r,f.x),f.y);
    }
    ${coarsePalette}
    ${city?.paperShader||''}
    @fragment fn fragment(input:Vertex)->@location(0) vec4<f32>{
      let p=input.uv*(params.field.xy-2.*params.viewport.z)+params.viewport.z-.5;
      let footprint=max(length(dpdx(p)),length(dpdy(p)));
      let level=clamp(log2(max(footprint,1.)),0.,3.);
      let height=heightAt(p,level);
      ${coarseColor}
      ${city?'rgb=paperColor(rgb,params.field.z,height,input.position.xy);':''}
      return vec4<f32>(rgb,1.);
    }`;

  // Opt-in timeline compatible with Chrome/Perfetto. Queue completion latency
  // includes preceding submissions; it is deliberately not a GPU kernel timer.
  const timingEnabled=new URLSearchParams(location.search).get('profiling')==='1';
  const timingEvents=new Array(4096);let timingCount=0,queueObservers=0,omittedQueueObservers=0;
  window.TerrainTiming={enabled:timingEnabled,
    record(name,start,args={}){if(!timingEnabled)return;const end=performance.now();timingEvents[timingCount++%4096]={name,ph:'X',pid:2,tid:'browser',ts:(performance.timeOrigin+start)*1000,dur:(end-start)*1000,args};},
    queue(queue,name){if(!timingEnabled)return;if(queueObservers>=3){omittedQueueObservers++;return;}const start=performance.now();queueObservers++;queue.onSubmittedWorkDone().then(()=>this.record(name+'.queue_completion',start,{includes_previous_submissions:true}),()=>{}).finally(()=>queueObservers--);},
    snapshot(){const first=Math.max(0,timingCount-4096);return{enabled:timingEnabled,dropped_events:first,omitted_queue_observers:omittedQueueObservers,capacity:4096,notes:'Queue completion is latency, not isolated GPU execution or display presentation.',traceEvents:Array.from({length:timingCount-first},(_,i)=>timingEvents[(first+i)%4096])};}};
  class TerrainGpuRenderer {
    constructor(canvas, options) {
      this.canvas=canvas; this.options=options; this.tiles=new Map(); this.bytes=0;
      this.maxBytes=options.maxBytes ?? 96*1024*1024; this.available=false;
      this.status='initializing'; this.frames=0; this.uploads=0; this.styleRevision=0; this.serial=0; this.capacity=0;
    }
    notify(status, detail='') { this.status=status; this.options.onStatus?.(status,detail); }
    async initialize() {
      if (!navigator.gpu) throw Error('WebGPU indisponible');
      this.adapter=await navigator.gpu.requestAdapter({powerPreference:'high-performance'});
      if (!this.adapter) throw Error('Aucun adaptateur WebGPU');
      this.device=await this.adapter.requestDevice(); const device=this.device;
      this.context=this.canvas.getContext('webgpu');
      if (!this.context) throw Error('Canvas WebGPU indisponible');
      this.format=navigator.gpu.getPreferredCanvasFormat();
      this.context.configure({device,format:this.format,alphaMode:'premultiplied'});
      device.addEventListener('uncapturederror',e=>this.fail(e.error.message));
      device.lost.then(info=>{if(this.status!=='disposed')this.fail(`GPU perdu : ${info.message || info.reason}`);});
      device.pushErrorScope('validation');
      const module=async(code,label)=>{
        const result=device.createShaderModule({code,label});
        const info=await result.getCompilationInfo();
        const errors=info.messages.filter(x=>x.type==='error');
        if(errors.length)throw Error(errors.map(x=>`${label}:${x.lineNum} ${x.message}`).join('\n'));
        return result;
      };
      const [blur,shade,render,coarse,mip,contour]=await Promise.all([module(blurShader,'terrain Gaussian'),module(shadeShader,'terrain relief'),module(renderShader,'terrain quads'),module(coarseShader,'native coarse relief'),module(coarseMipShader,'native coarse mips'),module(contourShader,'terrain vector contours')]);
      this.blurLayout=device.createBindGroupLayout({entries:[
        {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
        {binding:1,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:'write-only',format:'rg32float'}},
      ]});
      const blurLayout=device.createPipelineLayout({bindGroupLayouts:[this.blurLayout]});
      this.shadeLayout=device.createBindGroupLayout({entries:[
        {binding:0,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
        {binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
        {binding:2,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:'write-only',format:'rgba8unorm'}},
        {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
        {binding:4,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'2d-array'}},
      ]});
      this.renderLayout=device.createBindGroupLayout({entries:[
        {binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:208}},
        {binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{}},
        {binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:'filtering'}},
        {binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'unfilterable-float'}},
      ]});
      this.coarseLayout=device.createBindGroupLayout({entries:[
        {binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:208}},
        {binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'unfilterable-float'}},
        {binding:2,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'unfilterable-float',viewDimension:'2d-array'}},
      ]});
      this.coarsePipeline=await device.createRenderPipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[this.coarseLayout]}),vertex:{module:coarse,entryPoint:'vertex'},fragment:{module:coarse,entryPoint:'fragment',targets:[{format:this.format}]},primitive:{topology:'triangle-list'}});
      this.coarseMipPipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:mip,entryPoint:'main'}});
      [this.horizontal,this.vertical,this.shade,this.pipeline]=await Promise.all([
        device.createComputePipelineAsync({layout:blurLayout,compute:{module:blur,entryPoint:'horizontal'}}),
        device.createComputePipelineAsync({layout:blurLayout,compute:{module:blur,entryPoint:'vertical'}}),
        device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[this.shadeLayout]}),compute:{module:shade,entryPoint:'main'}}),
        device.createRenderPipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[this.renderLayout]}),vertex:{module:render,entryPoint:'vertex'},fragment:{module:render,entryPoint:'fragment',targets:[{format:this.format}]},primitive:{topology:'triangle-list'}}),
      ]);
      this.contourLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:208}}]});
      this.contourPipeline=await device.createRenderPipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[this.contourLayout]}),vertex:{module:contour,entryPoint:'vertex',buffers:[{arrayStride:20,stepMode:'instance',attributes:[{shaderLocation:0,offset:0,format:'float32x4'},{shaderLocation:1,offset:16,format:'float32'}]}]},fragment:{module:contour,entryPoint:'fragment',targets:[{format:this.format,blend:{color:{srcFactor:'src-alpha',dstFactor:'one-minus-src-alpha'},alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha'}}}]},primitive:{topology:'triangle-list'}});
      const validation=await device.popErrorScope(); if(validation)throw Error(validation.message);
      this.sampler=device.createSampler({magFilter:'linear',minFilter:'linear'});
      this.available=true; this.notify('webgpu'); return this;
    }
    fail(message) {
      if (!this.available && this.status==='fallback') return;
      this.available=false; this.clear(); this.canvas.style.visibility='hidden'; this.notify('fallback',message);
    }
    makeTexture(width,height,format,extra=0) {
      return this.device.createTexture({size:[width,height],format,usage:GPUTextureUsage.TEXTURE_BINDING|extra});
    }
    scratch(width,height) {
      if(this.scratchWidth===width&&this.scratchHeight===height)return;
      this.scratchA?.destroy(); this.scratchB?.destroy();
      this.scratchWidth=width; this.scratchHeight=height;
      this.scratchA=this.makeTexture(width,height,'rg32float',GPUTextureUsage.STORAGE_BINDING);
      this.scratchB=this.makeTexture(width,height,'rg32float',GPUTextureUsage.STORAGE_BINDING);
      this.verticalGroup=this.device.createBindGroup({layout:this.blurLayout,entries:[{binding:0,resource:this.scratchA.createView()},{binding:1,resource:this.scratchB.createView()}]});
    }
    uploadTile(key, heights, {width=304,height=304,halo=24,metresPerSample=30,climate=null,climateWidth=33,climateHeight=33,mode='relief',lod=0,origin=[0,0]}={}) {
      if(!this.available)return false;
      if(!(heights instanceof Float32Array)||heights.length!==width*height||![width,height,halo].every(Number.isInteger)||halo<1||width<=2*halo||height<=2*halo||!Number.isFinite(metresPerSample)||metresPerSample<=0)throw Error('Invalid physical tile');
      if(Math.max(width,height)>this.device.limits.maxTextureDimension2D)throw Error('Tile exceeds GPU limits');
      this.deleteTile(key); const started=performance.now(); this.scratch(width,height);
      const innerWidth=width-2*halo,innerHeight=height-2*halo;
      const elevation=this.makeTexture(width,height,'r32float',GPUTextureUsage.COPY_DST);
      const color=this.makeTexture(innerWidth,innerHeight,'rgba8unorm',GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC);
      if(climate!==null&&(!(climate instanceof Float32Array)||![5,21,36,46].includes(climate.length/(climateWidth*climateHeight))))throw Error('Invalid tile climate');
      const climateLayers=climate?climate.length/(climateWidth*climateHeight):5;
      const climateTexture=this.device.createTexture({size:[climateWidth,climateHeight,climateLayers],format:'r32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
      const climateValues=climate||new Float32Array(climateWidth*climateHeight*5);
      this.device.queue.writeTexture({texture:climateTexture},climateValues,{bytesPerRow:climateWidth*4,rowsPerImage:climateHeight},[climateWidth,climateHeight,climateLayers]);
      this.device.queue.writeTexture({texture:elevation},heights,{bytesPerRow:width*4,rowsPerImage:height},[width,height]);
      const selectedMode=modeNumber(this.mode??mode);
      const tile={elevation,color,climate:climateTexture,halo,resolution:metresPerSample,params:[width,height,selectedMode,0],origin,lod,heights};
      this.shadeTile(tile);
      const bytes=width*height*4+innerWidth*innerHeight*4+climateWidth*climateHeight*climateLayers*4;
      this.tiles.set(key,Object.assign(tile,{bytes,lastUsed:++this.serial}));this.bytes+=bytes;this.uploads++;this.lastUploadSubmitMs=performance.now()-started;
      window.TerrainTiming.record('render.upload_and_shade_submit',started,{key,bytes});
      window.TerrainTiming.queue(this.device.queue,'render.shade');
      this.evict();return this.hasTile(key);
    }
    shadeTile(tile) {
      const {elevation,color,climate:climateTexture,halo,resolution:metresPerSample,origin,lod}=tile;
      const [width,height]=tile.params,innerWidth=width-2*halo,innerHeight=height-2*halo;
      this.scratch(width,height);
      const params=this.device.createBuffer({size:160,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
      const selectedMode=tile.params[2];
      this.device.queue.writeBuffer(params,0,new Float32Array([metresPerSample,halo,innerWidth,innerHeight,selectedMode,0,0,0,...(window.TerrainLighting?.vectors(lod)||[1,.35,1,1,-.5,-.5,.70710678,0]),...origin,0,0,...this.renderVectors()]));
      const horizontalGroup=this.device.createBindGroup({layout:this.blurLayout,entries:[{binding:0,resource:elevation.createView()},{binding:1,resource:this.scratchA.createView()}]});
      const shadeGroup=this.device.createBindGroup({layout:this.shadeLayout,entries:[{binding:0,resource:elevation.createView()},{binding:1,resource:this.scratchB.createView()},{binding:2,resource:color.createView()},{binding:3,resource:{buffer:params}},{binding:4,resource:climateTexture.createView({dimension:'2d-array'})}]});
      const encoder=this.device.createCommandEncoder({label:'terrain shade once'});
      for(const [pipeline,group,w,h] of [[this.horizontal,horizontalGroup,width,height],[this.vertical,this.verticalGroup,width,height],[this.shade,shadeGroup,innerWidth,innerHeight]]) {
        const pass=encoder.beginComputePass(); pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(w/8),Math.ceil(h/8));pass.end();
      }
      this.device.queue.submit([encoder.finish()]); params.destroy();tile.shadedMode=selectedMode;
    }
    setBiomeRockSlope(angle) {
      // Compatibility for existing callers; Biomes no longer uses this value.
      if(!Number.isFinite(angle)||angle<1||angle>89)throw Error('Invalid biome rock slope');
      this.biomeRockSlope=angle;
    }
    renderVectors(){
      const s={season:.5,variation:1,forest:1,rock_slope:40,snow:1,vegetation_tint:[1,1,1],rock_tint:[1,1,1],snow_color:[.94,.96,.97],...this.renderSettings};
      return [s.season,s.variation,s.forest,Math.tan(s.rock_slope*Math.PI/180),s.snow,0,0,0,...s.vegetation_tint,0,...s.rock_tint,0,...s.snow_color,0];
    }
    setRenderSettings(value){
      const s=window.TerrainRender?.normalize(value)||value;
      const signature=JSON.stringify(s);if(signature===this.renderSignature)return;
      this.renderSignature=signature;this.renderSettings=s;this.styleRevision++;
      for(const tile of this.tiles.values())if(tile.params[2]===-3)tile.shadedMode=null;
    }
    setMode(mode) {
      if(this.mode===mode)return;this.mode=mode;this.styleRevision++;
      const selectedMode=modeNumber(mode);
      for(const tile of this.tiles.values())tile.params[2]=selectedMode;
    }
    uploadCoarse(key, roots, {width=160,height=160,halo=16,metresPerSample=7680,climate,climateWidth=41,climateHeight=41,mode='relief',interpolation='monotone',origin=[0,0]}={}) {
      if(!this.available)return false;
      if(!(roots instanceof Float32Array)||roots.length!==width*height||width%8||height%8||halo<8||!(climate instanceof Float32Array)||![5,21,36,46].includes(climate.length/(climateWidth*climateHeight)))throw Error('Invalid native coarse block');
      this.deleteTile(key);
      const elevation=this.device.createTexture({size:[width,height],format:'r32float',mipLevelCount:4,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.STORAGE_BINDING});
      const climateLayers=climate?climate.length/(climateWidth*climateHeight):5;
      const climateTexture=this.device.createTexture({size:[climateWidth,climateHeight,climateLayers],format:'r32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
      this.device.queue.writeTexture({texture:elevation},roots,{bytesPerRow:width*4,rowsPerImage:height},[width,height]);
      this.device.queue.writeTexture({texture:climateTexture},climate,{bytesPerRow:climateWidth*4,rowsPerImage:climateHeight},[climateWidth,climateHeight,climateLayers]);
      const encoder=this.device.createCommandEncoder({label:'native coarse physical mips'});
      let bytes=climate.byteLength;
      for(let level=0;level<4;level++){
        const w=width>>level,h=height>>level;bytes+=w*h*4;if(!level)continue;
        const group=this.device.createBindGroup({layout:this.coarseMipPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:elevation.createView({baseMipLevel:level-1,mipLevelCount:1})},{binding:1,resource:elevation.createView({baseMipLevel:level,mipLevelCount:1})}]});
        const pass=encoder.beginComputePass();pass.setPipeline(this.coarseMipPipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(w/8),Math.ceil(h/8));pass.end();
      }
      this.device.queue.submit([encoder.finish()]);
      this.tiles.set(key,{native:true,elevation,climate:climateTexture,bytes,lastUsed:++this.serial,params:[width,height,modeNumber(this.mode??mode),interpolation==='monotone'?1:0],halo,resolution:metresPerSample,origin,heights:roots,encoding:'signed-sqrt'});
      this.bytes+=bytes;this.uploads++;this.evict();return this.hasTile(key);
    }
    prepareContours(tile,interval,rect) {
      const footprint=(tile.params[0]-2*tile.halo)*(rect.uv?.[2]??1)/rect.width;
      const level=tile.native?Math.min(3,Math.max(0,Math.floor(Math.log2(Math.max(1,footprint))))):0;
      if(tile.contourInterval===interval&&tile.contourLevel===level)return;
      tile.contourMips??=[tile.heights];
      for(let mip=tile.contourMips.length;mip<=level;mip++){
        const previous=tile.contourMips[mip-1],w=tile.params[0]>>(mip-1),h=tile.params[1]>>(mip-1),next=new Float32Array((w>>1)*(h>>1));
        const metres=v=>tile.encoding==='signed-sqrt'?Math.sign(v)*v*v:v;
        for(let y=0;y<(h>>1);y++)for(let x=0;x<(w>>1);x++){
          const p=y*2*w+x*2,value=(metres(previous[p])+metres(previous[p+1])+metres(previous[p+w])+metres(previous[p+w+1]))*.25;
          next[y*(w>>1)+x]=tile.encoding==='signed-sqrt'?Math.sign(value)*Math.sqrt(Math.abs(value)):value;
        }
        tile.contourMips.push(next);
      }
      const values=city.buildContours(tile.contourMips[level],{width:tile.params[0]>>level,height:tile.params[1]>>level,halo:tile.halo/2**level,encoding:tile.encoding},interval);
      tile.contourBuffer?.destroy();this.bytes-=tile.contourBytes||0;tile.bytes-=tile.contourBytes||0;
      tile.contourBytes=values.byteLength;tile.contourCount=values.length/5;tile.contourInterval=interval;tile.contourLevel=level;
      if(values.length){tile.contourBuffer=this.device.createBuffer({size:values.byteLength,usage:GPUBufferUsage.VERTEX|GPUBufferUsage.COPY_DST});this.device.queue.writeBuffer(tile.contourBuffer,0,values);}
      this.bytes+=tile.contourBytes;tile.bytes+=tile.contourBytes;
    }
    hasTile(key) { return this.available&&this.tiles.has(key); }
    deleteTile(key) {
      const tile=this.tiles.get(key); if(!tile)return;
      tile.elevation.destroy();tile.color?.destroy();tile.climate.destroy();tile.contourBuffer?.destroy();this.bytes-=tile.bytes;this.tiles.delete(key);
    }
    evict() {
      const overhead=(this.scratchWidth||0)*(this.scratchHeight||0)*16+this.capacity*this.device.limits.minUniformBufferOffsetAlignment;
      while(this.bytes+overhead>this.maxBytes&&this.tiles.size) {
        let key,oldest=Infinity;for(const [candidate,tile]of this.tiles)if(tile.lastUsed<oldest){oldest=tile.lastUsed;key=candidate;}
        this.deleteTile(key);
      }
    }
    clear() { for(const key of this.tiles.keys())this.deleteTile(key); }
    draw(rects, {width,height,dpr=window.devicePixelRatio||1}) {
      this.lastDrawnRects=[];
      if(!this.available)return false;
      const started=timingEnabled?performance.now():0;
      if(!Number.isFinite(width)||!Number.isFinite(height)||width<=0||height<=0)return false;
      const max=this.device.limits.maxTextureDimension2D;
      dpr=Math.min(dpr,max/width,max/height);
      const w=Math.max(1,Math.round(width*dpr)),h=Math.max(1,Math.round(height*dpr));
      if(this.canvas.width!==w||this.canvas.height!==h){this.canvas.width=w;this.canvas.height=h;}
      let visible=rects.filter(r=>this.tiles.has(r.key)&&r.width>0&&r.height>0);
      const stride=this.device.limits.minUniformBufferOffsetAlignment;
      if(visible.length>this.capacity) {
        this.uniform?.destroy();this.capacity=Math.max(16,2**Math.ceil(Math.log2(visible.length)));
        this.uniform=this.device.createBuffer({size:this.capacity*stride,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
        this.uniformData=new Float32Array(this.capacity*stride/4);
        for(const tile of this.tiles.values()){tile.group=null;tile.contourGroup=null;}
        this.evict();visible=visible.filter(r=>this.tiles.has(r.key));
      }
      for(const r of visible){const tile=this.tiles.get(r.key);if(!tile.native&&tile.shadedMode!==tile.params[2])this.shadeTile(tile);}
      const contourSettings=city?.get();
      if(contourSettings?.enabled)for(const r of visible){const tile=this.tiles.get(r.key);this.prepareContours(tile,contourSettings.interval,r);}
      this.evict();visible=visible.filter(r=>this.tiles.has(r.key));
      visible.forEach((r,i)=>{const tile=this.tiles.get(r.key);this.uniformData.set([r.x,r.y,r.width,r.height,width,height,tile.halo||0,tile.resolution||0,...(r.uv||[0,0,1,1]),...(tile.params||[0,0,0,0]),...(window.TerrainLighting?.vectors(r.lod)||[1,.35,1,1,-.5,-.5,.70710678,0]),...(tile.origin||[0,0]),0,0,...(city?.vectors()||[0,100,0,0]),...this.renderVectors()],i*stride/4)});
      if(visible.length)this.device.queue.writeBuffer(this.uniform,0,this.uniformData,0,visible.length*stride/4);
      const encoder=this.device.createCommandEncoder();const pass=encoder.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:'clear',storeOp:'store'}]});
      pass.setPipeline(this.pipeline);
      visible.forEach((r,i)=>{
        const tile=this.tiles.get(r.key);tile.lastUsed=++this.serial;
        pass.setPipeline(tile.native?this.coarsePipeline:this.pipeline);
        tile.group??=this.device.createBindGroup(tile.native?{layout:this.coarseLayout,entries:[{binding:0,resource:{buffer:this.uniform,size:208}},{binding:1,resource:tile.elevation.createView()},{binding:2,resource:tile.climate.createView({dimension:'2d-array'})}]}:{layout:this.renderLayout,entries:[{binding:0,resource:{buffer:this.uniform,size:208}},{binding:1,resource:tile.color.createView()},{binding:2,resource:this.sampler},{binding:3,resource:tile.elevation.createView()}]});
        pass.setBindGroup(0,tile.group,[i*stride]);pass.draw(6);
        if(contourSettings?.enabled&&tile.contourCount){
          pass.setPipeline(this.contourPipeline);tile.contourGroup??=this.device.createBindGroup({layout:this.contourLayout,entries:[{binding:0,resource:{buffer:this.uniform,size:208}}]});
          pass.setBindGroup(0,tile.contourGroup,[i*stride]);pass.setVertexBuffer(0,tile.contourBuffer);pass.draw(6,tile.contourCount);
        }
      });
      pass.end();this.device.queue.submit([encoder.finish()]);this.lastDrawnRects=visible;this.frames++;this.canvas.style.visibility='visible';
      window.TerrainTiming.record('render.draw_submit',started,{quads:visible.length});
      window.TerrainTiming.queue(this.device.queue,'render.draw');return true;
    }
    getStats() { const scratchBytes=(this.scratchWidth||0)*(this.scratchHeight||0)*16,uniformBytes=this.capacity*(this.device?.limits.minUniformBufferOffsetAlignment||0);return {status:this.status,tiles:this.tiles.size,bytes:this.bytes,scratchBytes,uniformBytes,totalBytes:this.bytes+scratchBytes+uniformBytes,maxBytes:this.maxBytes,frames:this.frames,uploads:this.uploads,styleRevision:this.styleRevision,contourSegments:[...this.tiles.values()].reduce((n,t)=>n+(t.contourCount||0),0),lastUploadSubmitMs:this.lastUploadSubmitMs||0}; }
    dispose() {
      this.available=false;this.clear();this.scratchA?.destroy();this.scratchB?.destroy();this.uniform?.destroy();this.context?.unconfigure();this.notify('disposed');this.device?.destroy();
    }
  }
  window.createTerrainRenderer=async(canvas,options={})=>{
    const renderer=new TerrainGpuRenderer(canvas,options);
    try{return await renderer.initialize();}catch(error){renderer.dispose();options.onStatus?.('fallback',error.message);return null;}
  };
})();

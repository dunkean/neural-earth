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
  const city=window.TerrainStyleRendering,r3=window.TerrainRender3D,v2=window.TerrainRenderV2,rg=window.TerrainResourcesGPU;
  // Resource layers (terrain_resources_gpu.js) take numbers <= -100.
  const modeNumber=mode=>rg?.isMode(mode)?rg.number(mode):v2?.isStyle(mode)?v2.number(mode):r3?.isStyle(mode)?r3.number(mode):mode==='render'?-3:mode==='soil'?-4:mode==='pedology'?-5:mode==='orogen-biomes'?-1:mode==='orogen-koppen'?-2:city?.number(mode)??({relief:0,biomes:1,temperature:2,precipitation:3}[mode]??0);
  // The fine compute path and the coarse bake compile this SAME material,
  // mirrored operation for operation by terrain_render.py. It uses continuous
  // climate (no class IDs, hence no hard biome borders), terrain derivatives and
  // gradient noise in canonical geographic coordinates. Noise gently modulates
  // continuous coverage; unresolved detail converges to the base mixture.
  const materialShader=`
    var<private> surfaceDetailOverride:vec4<f32>;
    fn sstep(a:f32,b:f32,x:f32)->f32{let t=clamp((x-a)/(b-a),0.,1.);return t*t*(3.-2.*t);}
    fn surfaceMix(a:vec3<f32>,b:vec3<f32>,t:f32)->vec3<f32>{return a*(1.-t)+b*t;}
    fn surfacePedology(p:vec2<PTYPE>)->vec3<f32>{
      if(textureNumLayers(climate)>=50u){return vec3<f32>(clim(p,46),clim(p,47),clim(p,48));}
      return vec3<f32>(clim(p,36),clim(p,37),clim(p,38));
    }
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
    fn surfaceLandPoint(local:vec2<f32>,origin:vec2<f32>,spacing:f32)->vec3<f32>{
      let lo=vec2<f32>(surfaceMeta(0),surfaceMeta(1));let extent=vec2<f32>(surfaceMeta(2),surfaceMeta(3))-lo;
      if(surfaceMeta(4)<.5){return vec3<f32>(origin+(local+.5)*spacing,0.);}
      let uv=(origin-lo)/extent+(local+.5)*(spacing/extent);
      let lon=fract(uv.x)*6.283185307-3.141592654;let radius=extent.x/6.283185307;
      return vec3<f32>(radius*cos(lon),radius*sin(lon),uv.y*extent.y);
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
    fn surfaceHash(c:vec3<i32>,salt:u32)->u32{
      let v=bitcast<vec3<u32>>(c);
      var h=(v.x*0x9e3779b9u)^(v.y*0x85ebca6bu)^(v.z*0xc2b2ae35u)^salt;
      h=(h^(h>>16u))*0x7feb352du;h=(h^(h>>15u))*0x846ca68bu;return h^(h>>16u);
    }
    fn surfaceCorner(c:vec3<i32>,f:vec3<f32>,o:vec3<i32>,salt:u32)->f32{
      let h=surfaceHash(c+o,salt);let d=f-vec3<f32>(o);
      let g=vec3<f32>(f32(h&1023u),f32((h>>10u)&1023u),f32((h>>20u)&1023u))/511.5-1.;
      return g.x*d.x+g.y*d.y+g.z*d.z;
    }
    // 3D gradient noise, quintic fade, unit standard deviation.
    fn surfaceNoise(p:vec3<f32>,salt:u32)->f32{
      let fl=floor(p);let c=vec3<i32>(fl);let f=p-fl;let u=f*f*f*(f*(f*6.-15.)+10.);
      let a0=surfaceCorner(c,f,vec3<i32>(0,0,0),salt);let a1=surfaceCorner(c,f,vec3<i32>(1,0,0),salt);
      let a2=surfaceCorner(c,f,vec3<i32>(0,1,0),salt);let a3=surfaceCorner(c,f,vec3<i32>(1,1,0),salt);
      let b0=surfaceCorner(c,f,vec3<i32>(0,0,1),salt);let b1=surfaceCorner(c,f,vec3<i32>(1,0,1),salt);
      let b2=surfaceCorner(c,f,vec3<i32>(0,1,1),salt);let b3=surfaceCorner(c,f,vec3<i32>(1,1,1),salt);
      let x0=a0+(a1-a0)*u.x;let x1=a2+(a3-a2)*u.x;let x2=b0+(b1-b0)*u.x;let x3=b2+(b3-b2)*u.x;
      let y0=x0+(x1-x0)*u.y;let y1=x2+(x3-x2)*u.y;
      return (y0+(y1-y0)*u.z)*5.25;
    }
    // (unit-variance pattern of the resolved octaves, retained variance fraction).
    // Octaves under 8 samples per wavelength fade out, then are skipped: closer
    // to Nyquist they read as grain rather than detail.
    fn surfaceFbm(p:vec3<f32>,scale:f32,octaves:i32,footprint:f32,salt:u32)->vec2<f32>{
      var value=0.;var kept=0.;var total=0.;var s=scale;var a=1.;
      for(var i=0;i<octaves;i++){
        total+=a*a;
        if(footprint<s*.25){let fade=1.-sstep(s*.125,s*.25,footprint);value+=a*fade*surfaceNoise(p/s,salt+u32(i)*0x9e37u);kept+=a*fade*a*fade;}
        a*=.5;s*=.5;
      }
      if(kept>0.){value/=sqrt(kept);}
      return vec2<f32>(value,kept/total);
    }
    fn surfaceCover(fraction:f32,pattern:vec2<f32>,sharp:f32)->f32{
      let c=clamp(fraction,0.,1.);
      let v=pattern.x/sqrt(1.+pattern.x*pattern.x)*sqrt(pattern.y);
      return c+clamp(sharp,0.,1.)*c*(1.-c)*v;
    }
    fn surfaceSeason()->f32{return .5-.5*cos(6.283185307*params.render0.x);}
    fn surfaceSeaIce(point:vec3<f32>,ts0:f32,tw0:f32,footprint:f32)->f32{
      let regional=surfaceFbm(point,64000.,4,footprint,u32(surfaceMeta(6))+1u);let m1=regional.x*sqrt(regional.y);
      let ts=clamp(ts0,-45.,45.);let tw=clamp(tw0,-45.,45.);let now=tw+(ts-tw)*surfaceSeason();
      return clamp(sstep(-1.5,-6.,now+.4*m1)*params.render1.x,0.,1.)*.9;
    }
    fn surfaceWater(depth:f32,ice:f32)->vec3<f32>{
      var water=surfaceMix(vec3<f32>(.10,.30,.36),vec3<f32>(.045,.16,.27),sstep(5.,120.,depth));
      water=surfaceMix(water,vec3<f32>(.018,.065,.15),sstep(150.,3000.,depth));
      return surfaceMix(water,vec3<f32>(.80,.84,.88),ice);
    }
    fn surfaceOcean(height:f32,color:vec3<f32>,now:f32,m1:f32,variation:f32)->vec3<f32>{
      if(height>=0.){return clamp(color,vec3<f32>(0.),vec3<f32>(1.));}
      let ice=clamp(sstep(-1.5,-6.,now+.4*m1)*params.render1.x,0.,1.)*.9;
      return clamp(surfaceWater(max(-height,0.),ice)*(1.+.04*variation*m1),vec3<f32>(0.),vec3<f32>(1.));
    }
    // Climate: sea-level seasonal temperatures/lapse and seasonal precipitation.
    // bare=true returns the substrate shown by the Soil layer.
    fn surfaceMaterial(height:f32,gradient:vec2<f32>,tpi:f32,point:vec3<f32>,north:vec2<f32>,footprint:f32,soil:vec3<f32>,pedology:vec3<f32>,substrateRock:vec3<f32>,sandFraction:f32,ts0:f32,ps:f32,ls:f32,tw0:f32,pw:f32,lw:f32,bare:bool,shore:f32,openness:f32,erosion:vec4<f32>)->vec3<f32>{
      let season=params.render0.x;let variation=params.render0.y;let forestAmount=params.render0.z;let rockTan=params.render0.w;
      let snowAmount=params.render1.x;let moisture=params.render1.y;
      let salt=u32(surfaceMeta(6));let fp=footprint;let phase=surfaceSeason();let h=max(height,0.);
      let materialDetail=sstep(30.,60.,fp)*(1.-sstep(120.,240.,fp));
      let regional=surfaceFbm(point,64000.,4,fp,salt+1u);let regional2=surfaceFbm(point+vec3<f32>(5101.,-2297.,3313.),48000.,4,fp,salt+2u);
      let detailPoint=mix(point,surfaceDetailOverride.xyz,surfaceDetailOverride.w);
      let grove=surfaceFbm(detailPoint,4800.,4,fp,salt+3u);let crag=surfaceFbm(detailPoint,700.,4,fp,salt+4u);
      let fine=surfaceFbm(detailPoint,240.,4,fp,salt+5u);let province=surfaceFbm(point,900000.,3,fp,salt+6u);
      let m1=regional.x*sqrt(regional.y);let m2=regional2.x*sqrt(regional2.y);let pv=province.x*sqrt(province.y);
      let ts=clamp(ts0-ls*h,-45.,45.);let tw=clamp(tw0-lw*h,-45.,45.);
      let locality=1.-sstep(600.,6000.,fp);
      let terrain=sstep(.015,.12,abs(tpi));
      let jitter=.8*m2*variation*locality*(.25+.75*terrain);let tHot=max(ts,tw)+jitter;let tCold=min(ts,tw)+jitter;
      let now=tw+(ts-tw)*phase;
      // Koppen-like aridity threshold; valleys gather water, ridges drain.
      let pth=max(20.*(ts+tw)*.5+280.,120.);
      let wetShift=exp(.6*moisture+.10*variation*m1*locality);
      let valley=sstep(0.,-.12,tpi);let ridge=sstep(0.,.12,tpi);
      var humid=max(ps+pw,0.)/pth*wetShift;
      humid=humid*(1.+.7*valley*(1.-sstep(1.5,3.,humid))-.18*ridge);
      let humidNow=max(pw+(ps-pw)*phase,0.)*2./pth*wetShift*(1.+.7*valley-.18*ridge);
      // Coarser samples see gentler slopes than the 30 m terrain they average.
      let grade=length(gradient)*pow(max(fp,30.)/30.,.3);
      let arid=1.-sstep(.35,1.,humid);
      let wTrop=sstep(13.,19.,tCold);let wBoreal=1.-sstep(-14.,-3.,tCold);
      // Substrate: transported regional soil, zonal soils, geological provinces.
      var ground=select(pedology,soil,bare);
      let sand=surfaceMix(vec3<f32>(.80,.66,.46),vec3<f32>(.70,.50,.33),sstep(-.3,.5,pv));
      let desert=surfaceMix(vec3<f32>(.55,.46,.36),sand,sstep(-.4,.4,m2+1.5*sandFraction-.5-.8*grade/rockTan));
      ground=surfaceMix(ground,desert,arid*select(0.,.8,bare));
      ground=surfaceMix(ground,vec3<f32>(.50,.29,.17),wTrop*sstep(1.,2.,humid)*select(0.,.6,bare));
      let steppe=(1.-arid)*(1.-wTrop)*(1.-wBoreal)*sstep(.8,1.2,humid)*(1.-sstep(1.6,2.4,humid));
      ground=surfaceMix(ground,vec3<f32>(.25,.21,.16),steppe*select(0.,.5,bare));
      ground=ground*(1.-.12*valley+.08*ridge);
      ground=ground*(1.+variation*(.035*m1*locality+.025*grove.x*sqrt(grove.y)+.02*fine.x*sqrt(fine.y)));
      var rock=surfaceMix(vec3<f32>(.50,.48,.45),vec3<f32>(.30,.29,.28),sstep(.15,.6,pv));
      rock=surfaceMix(rock,vec3<f32>(.66,.62,.55),sstep(-.15,-.6,pv));
      // Arid outcrops: dark desert varnish, or red sandstone in some provinces.
      rock=surfaceMix(rock,surfaceMix(vec3<f32>(.36,.30,.25),vec3<f32>(.60,.38,.25),sstep(.1,.6,m2)),arid*.85);
      rock=surfaceMix(rock,substrateRock,.35);
      if(!bare){
        // Warm stone at every LOD; regional minerals only affect brightness.
        let neutral=dot(rock,vec3<f32>(.2126,.7152,.0722));
        rock=neutral*vec3<f32>(1.04,1.,.94);
      }
      rock=rock*params.render3.xyz;
      // Horizontal strata, resolved only on fine tiles; strongest in arid ranges.
      let strata=sin(6.283185307*(h+120.*grove.x)/90.)*(1.-sstep(25.,45.,fp));
      rock=rock*(1.+variation*.10*strata*(.15+.85*arid)+variation*.07*crag.x*sqrt(crag.y));
      let rockTanEff=rockTan*(.55+.45*sstep(.3,1.3,humid));
      let alpine=1.-sstep(2.,9.,tHot);
      let exposure=sstep(rockTanEff*.5,rockTanEff*1.3,grade)+.45*alpine*sstep(rockTanEff*.15,rockTanEff*.7,grade)
        +.35*ridge*sstep(.35,.9,grade/rockTan)-.3*valley;
      let outcrop=max(surfaceCover(clamp(exposure,0.,1.),vec2<f32>(.75*crag.x+.25*grove.x*sqrt(grove.y),crag.y),.35),erosion.w*.85);
      ground=surfaceMix(ground,surfaceMix(ground,vec3<f32>(.48,.40,.29),.45),erosion.y*.65);
      ground=surfaceMix(ground,surfaceMix(rock,vec3<f32>(.70,.67,.59),.3),erosion.z*.65);
      ground*=1.-.18*erosion.x;
      var color=surfaceMix(ground,rock,outcrop);
      if(bare){return surfaceOcean(height,color,now,m1,variation);}
      // Vegetation: grasses/shrubs over soil, canopy patches over grasses.
      let tint=params.render2.xyz;
      let green=sstep(.5,1.4,humidNow)*sstep(3.,12.,now);
      var grass=surfaceMix(vec3<f32>(.60,.53,.34),vec3<f32>(.29,.37,.16),green);
      grass=surfaceMix(grass,vec3<f32>(.40,.38,.27),1.-sstep(7.,14.,tHot))*tint;
      let herbs=sstep(.2,.85,humid)*sstep(0.,7.,tHot)*(1.-sstep(rockTan*.8,rockTan*1.4,grade));
      /*R3:herbs*/
      // Grasses and shrubs thin out diffusely; only canopy forms crisp stands.
      let sward=select(grove,fine,fine.y>0.);
      color=surfaceMix(color,grass,surfaceCover(herbs*(1.-outcrop)*(1.-.65*erosion.x)*(1.-.8*erosion.z),sward,.12*variation));
      let dryForest=1.-sstep(1.3,3.,humid);
      var canopy=surfaceMix(vec3<f32>(.20,.28,.14),vec3<f32>(.16,.255,.13),wTrop);
      canopy=surfaceMix(canopy,vec3<f32>(.15,.22,.16),wBoreal);
      canopy=surfaceMix(canopy,vec3<f32>(.27,.29,.17),dryForest*.75);
      /*R3:base-canopy*/
      let deciduous=(1.-wTrop)*(1.-.7*wBoreal)*(1.-.5*dryForest);
      let local=fract(season+select(.5,0.,ts>=tw));
      let autumn=sstep(.66,.76,local)*(1.-sstep(.84,.92,local));
      canopy=surfaceMix(canopy,vec3<f32>(.42,.27,.11),autumn*deciduous*.6);
      let leafless=sstep(8.,2.,now)*deciduous;
      canopy=surfaceMix(canopy,vec3<f32>(.36,.33,.28),leafless*.75);
      let highland=(1.-sstep(10.,17.,tHot))*materialDetail;
      // Reuse already filtered noise; no extra FBM evaluations or reads.
      let canopyDetail=clamp(.12*highland*crag.x*sqrt(crag.y)+.025*fine.x*sqrt(fine.y),-.18,.18);
      canopy=canopy*tint*(1.-.18*highland)*(1.+variation*canopyDetail);
      /*R3:canopy-detail*/
      var trees=sstep(.65,2.1,humid)*sstep(5.5,13.5,tHot+.8*valley)*forestAmount*(1.-sstep(rockTan*.9,rockTan*1.6,grade))*(1.-outcrop);
      // Valley stands come from relief; noise only adds gentle clearings.
      trees=trees*(1.-.22*ridge)+.14*valley*sstep(2.,10.,tHot)*sstep(.4,1.2,humid)*(1.-outcrop)*forestAmount;
      trees*=(1.-.8*erosion.x)*(1.-.85*erosion.z);
      let woods=surfaceCover(clamp(trees,0.,1.),grove,.65*variation);
      /*R3:open*/
      color=surfaceMix(color,canopy,woods);
      let flats=(1.-sstep(1.5,6.,height))*(1.-sstep(.05,.2,grade));
      // Beaches: an intermittent strip along open water (none beside ponds), up to
      // ~190 m wide with a ragged inner edge, as area coverage (visible at 240 m
      // samples); low flats behind it stay as wetter river/estuary bars.
      let reach=surfaceFbm(point,20000.,3,fp,salt+7u);
      let width=140.*sstep(-.15,.75,.65*reach.x*sqrt(reach.y)+.5*grove.x*sqrt(grove.y))*sstep(.15,.4,openness)*(1.+.35*fine.x*sqrt(fine.y));
      let beach=clamp((min(width,20.*fp)-shore+fp)/fp,0.,1.)*(1.-sstep(4.,12.,height))*(1.-sstep(.2,.5,grade));
      // Sediment follows the coast's geology: quartz or golden sand by province,
      // tropical coral sand, and local stone on cool or rugged coasts. Summer
      // warmth preserves sandy continental coasts despite their winter frost.
      // Reuse slope/TPI and resolved noise, with no additional samples or FBM.
      let shingle=max(1.-sstep(4.,14.,tHot),sstep(.04,.22,grade)*terrain);
      var sediment=surfaceMix(vec3<f32>(.80,.75,.61),vec3<f32>(.76,.63,.43),sstep(-.2,.9,pv));
      sediment=surfaceMix(sediment,vec3<f32>(.86,.84,.76),wTrop*sstep(0.,-.8,pv)*.7);
      sediment=surfaceMix(sediment,rock,shingle*.9);
      sediment=surfaceMix(sediment,surfaceMix(ground,rock,.4),sstep(-.9,-1.5,m2)*.85);
      sediment=surfaceMix(sediment,vec3<f32>(.27,.26,.25),sstep(1.25,1.75,m1+.35*shingle)*.9);
      color=surfaceMix(color,sediment*.9,flats*(1.-sstep(240.,960.,fp))*.8);
      color=surfaceMix(color,sediment,beach*.8);
      /*R3:beach*/
      // Snow: seasonal and perennial, exposure-aware, sheds from steep ridges.
      let hemisphere=point.y/max(sqrt(point.y*point.y+.0064*dot(point,point)),1.);
      let aspect=dot(gradient,north)/max(length(gradient),.00001)*hemisphere;
      let sun=aspect*min(length(gradient),1.);
      let tSnow=now+2.2*sun+.45*m1*variation*locality+.25*ridge-.35*valley;
      // Snow needs supply: cold deserts stay mostly bare, wet ranges glaciate.
      let supply=sstep(60.,400.,max(ps+pw,0.)*wetShift*(1.+.5*valley-.25*ridge));
      let perennial=sstep(2.5,-2.5,max(ts,tw)+1.5*sun+.35*m1*variation*locality)*supply;
      let seasonal=sstep(1.,-4.,tSnow)*sstep(.15,.5,humid)*supply;
      var snow=clamp(max(seasonal,perennial)*snowAmount,0.,1.);
      // Physical slope: LOD compensation must not turn a ridge into a cliff.
      let slip=sstep(1.25+.65*perennial,3.5+perennial,length(gradient));
      snow=snow*(1.-(.88-.18*perennial)*slip)*(1.-.3*woods*(1.-perennial)*(1.-.7*leafless));
      color=surfaceMix(color,params.render4.xyz,surfaceCover(snow,crag,.35*variation));
      color=color*(1.-.06*valley);
      // Render3D captures its guide (same surface without and with full canopy)
      // and forest-field terms here; other layers compile this as a comment.
      /*R3:end*/
      return surfaceOcean(height,color,now,m1,variation);
    }
  `;
  const materialI32=(materialShader+(v2?.environment||'')).replaceAll('PTYPE','i32');
  // Horizontal half of shoreDistance's square search: distance in columns to
  // the nearest sea sample of the same (clamped) row within its reach, 99 when none.
  const shoreRowShader=`
    @group(0) @binding(0) var elevation:texture_2d<f32>;
    @group(0) @binding(1) var rows:texture_storage_2d<rg32float,write>;
    @group(0) @binding(2) var<uniform> resolution:vec4<f32>;
    @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3<u32>){
      let size=vec2<i32>(textureDimensions(elevation));let p=vec2<i32>(id.xy);
      if(any(p>=size)){return;}
      let r=resolution.x;let reach=min(21,i32(ceil((190.+r)/r)));var best=99.;
      for(var k=0;k<=reach;k++){
        let a=vec2<i32>(clamp(p.x-k,0,size.x-1),p.y);let b=vec2<i32>(clamp(p.x+k,0,size.x-1),p.y);
        if(hydroGround(textureLoad(elevation,a,0).r,vec2<f32>(a))<0.||hydroGround(textureLoad(elevation,b,0).r,vec2<f32>(b))<0.){best=f32(k);break;}
      }
      textureStore(rows,p,vec4<f32>(best,0.,0.,0.));
    }`;
  // Physical water: surface level, wet coverage, class, validity. A zero
  // fallback texture preserves the exact appearance of unconditioned tiles.
  const hydrologyShader = `
    @group(0) @binding(8) var hydrology:texture_2d<f32>;
    fn hydroAtLevel(p:vec2<f32>,level:i32)->vec4<f32>{
      let size=vec2<i32>(textureDimensions(hydrology,level));let q=clamp((p+.5)/exp2(f32(level))-.5,vec2<f32>(0.),vec2<f32>(size-1));
      let lo=vec2<i32>(floor(q));let hi=min(lo+1,size-1);let f=fract(q);
      let samples=array<vec4<f32>,4>(textureLoad(hydrology,lo,level),textureLoad(hydrology,vec2<i32>(hi.x,lo.y),level),textureLoad(hydrology,vec2<i32>(lo.x,hi.y),level),textureLoad(hydrology,hi,level));
      let weights=array<f32,4>((1.-f.x)*(1.-f.y),f.x*(1.-f.y),(1.-f.x)*f.y,f.x*f.y);
      var coverage=0.;var weightedLevel=0.;var valid=0.;var strongest=0.;var kind=0.;
      for(var i=0;i<4;i++){
        let s=samples[i];let a=weights[i]*clamp(s.y,0.,1.)*select(0.,1.,s.z>=1.5&&s.w>.5);
        coverage+=a;weightedLevel+=a*s.x;valid+=weights[i]*s.w;
        if(a>strongest){strongest=a;kind=s.z;}
      }
      if(coverage>1e-6){return vec4<f32>(weightedLevel/coverage,coverage,kind,valid);}
      let nearest=textureLoad(hydrology,vec2<i32>(round(q)),level);
      return vec4<f32>(nearest.xyz,valid);
    }
    fn hydroAt(p:vec2<f32>)->vec4<f32>{return hydroAtLevel(p,0);}
    fn hydroFiltered(p:vec2<f32>,footprint:f32)->vec4<f32>{
      let l=clamp(log2(max(footprint,1.)),0.,f32(textureNumLevels(hydrology)-1u));
      let lo=i32(floor(l));let hi=min(lo+1,i32(textureNumLevels(hydrology))-1);let t=fract(l);
      let a=hydroAtLevel(p,lo);let b=hydroAtLevel(p,hi);
      let wa=a.y*(1.-t)*select(0.,1.,a.z>=1.5);let wb=b.y*t*select(0.,1.,b.z>=1.5);let c=wa+wb;
      if(c<=1e-6){return a;}
      return vec4<f32>((a.x*wa+b.x*wb)/c,c,select(a.z,b.z,wb>wa),mix(a.w,b.w,t));
    }
    fn hydroGround(h:f32,p:vec2<f32>)->f32{
      let w=hydroAt(p);return select(h,max(h,.5),w.w>.5&&w.z<.5);
    }
    fn hydroBlend(rgb:vec3<f32>,height:f32,w:vec4<f32>)->vec3<f32>{
      if(w.w<.5||w.z<1.5||w.y<=0.){return rgb;}
      let depth=max(w.x-height,0.);let t=smoothstep(.15,12.,depth);
      let water=mix(vec3<f32>(.1019608,.3019608,.3607843),vec3<f32>(.0431373,.1607843,.2705882),t);
      return mix(rgb,water,clamp(w.y,0.,1.));
    }
  `;
  // Rasterization spatializes the same regional vector ribbons as the overlay.
  // No fragment searches all segments. Signed levels are split into two f16
  // components, retaining millimetre-scale precision in the 8-byte mask.
  const riverDisplayShader=`
    struct DisplayParams { size:vec4<f32> }
    struct Segment { ends:vec4<f32>, water:vec4<f32> }
    @group(0) @binding(0) var<uniform> display:DisplayParams;
    @group(0) @binding(1) var<storage,read> segments:array<Segment>;
    struct RiverVertex { @builtin(position) position:vec4<f32>, @location(0) @interpolate(flat) index:u32 }
    @vertex fn vertex(@builtin(vertex_index) id:u32,@builtin(instance_index) instance:u32)->RiverVertex{
      let s=segments[instance];let a=s.ends.xy;let b=s.ends.zw;
      let tangent=normalize(b-a+vec2<f32>(1e-12,0.));let normal=vec2<f32>(-tangent.y,tangent.x);
      let corners=array<vec2<f32>,6>(vec2<f32>(0,-1),vec2<f32>(1,-1),vec2<f32>(0,1),vec2<f32>(0,1),vec2<f32>(1,-1),vec2<f32>(1,1));
      let c=corners[id];let pad=s.water.x+1.5/min(display.size.z,display.size.w);
      let p=mix(a,b,c.x)+(2.*c.x-1.)*pad*tangent+c.y*pad*normal;
      var v:RiverVertex;v.position=vec4<f32>(p.x/display.size.x*2.-1.,1.-p.y/display.size.y*2.,0.,1.);v.index=instance;return v;
    }
    struct RiverFragment { @location(0) value:vec4<f32>, @builtin(frag_depth) depth:f32 }
    @fragment fn fragment(v:RiverVertex)->RiverFragment{
      let s=segments[v.index];let p=v.position.xy/display.size.zw;let a=s.ends.xy;let b=s.ends.zw;
      let delta=b-a;let t=clamp(dot(p-a,delta)/max(dot(delta,delta),1e-20),0.,1.);
      let d=length(p-mix(a,b,t));let half=s.water.x;
      let radial=p-mix(a,b,t);let axis=normalize(vec2<f32>(-delta.y,delta.x)+vec2<f32>(1e-12,0.));
      let normal=select(normalize(radial+vec2<f32>(1e-12,0.)),axis,d<1e-5);
      let pixel=max(dot(abs(normal),1./display.size.zw),1e-5);
      let cover=clamp((min(d+pixel*.5,half)-max(d-pixel*.5,-half))/pixel,0.,1.);
      if(cover<=0.){discard;}
      let level=mix(s.water.y,s.water.z,t);let hi=unpack2x16float(pack2x16float(vec2<f32>(level,0.))).x;
      var f:RiverFragment;f.value=vec4<f32>(cover,hi,level-hi,2.);f.depth=cover;return f;
    }`;
  const riverDisplaySampleShader=`
    @group(0) @binding(11) var riverDisplay:texture_2d<f32>;
    fn hydroDisplayWater(p:vec2<f32>,physical:vec4<f32>,height:f32)->vec4<f32>{
      let uv=(p/params.viewport.w-params.viewport.z)/(params.field.xy-2.*params.viewport.z);
      let screen=params.rect.xy+(uv-params.crop.xy)/params.crop.zw*params.rect.zw;
      if(any(screen<vec2<f32>(0.))||any(screen>=params.viewport.xy)){return physical;}
      let size=vec2<i32>(textureDimensions(riverDisplay));
      let m=textureLoad(riverDisplay,clamp(vec2<i32>(screen/params.viewport.xy*vec2<f32>(size)),vec2<i32>(0),size-1),0);
      if(m.a<.5){return physical;}
      var w=physical;
      // The old source ribbon is no longer the regional display authority.
      if(w.z>2.5){w=vec4<f32>(0.,0.,0.,w.w);}
      let still=physical.z>=.5&&physical.z<2.5;
      let dryBelowSea=physical.w>.5&&physical.z<.5&&height<0.;
      if(m.a>1.5&&m.r>0.&&!still&&!dryBelowSea){w=vec4<f32>(m.g+m.b,m.r,3.,1.);}
      return w;
    }`;
  const hydrologyMipShader=`
    @group(0) @binding(0) var source:texture_2d<f32>;
    @group(0) @binding(1) var destination:texture_storage_2d<rgba32float,write>;
    @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3<u32>){
      if(any(id.xy>=textureDimensions(destination))){return;}
      let p=vec2<i32>(id.xy)*2;let last=vec2<i32>(textureDimensions(source))-1;
      var cover=0.;var level=0.;var valid=0.;var lake=0.;var river=0.;
      for(var y=0;y<2;y++){for(var x=0;x<2;x++){
        let s=textureLoad(source,min(p+vec2<i32>(x,y),last),0);valid+=s.w*.25;
        if(s.z>=1.5&&s.w>.5){let w=clamp(s.y,0.,1.)*.25;cover+=w;level+=s.x*w;if(s.z<2.5){lake+=w;}else{river+=w;}}
      }}
      let kind=select(2.,3.,river>lake);
      textureStore(destination,vec2<i32>(id.xy),vec4<f32>(level/max(cover,1e-12),cover,select(0.,kind,cover>0.),valid));
    }`;
  const shadeShader = `
    ${hydrologyShader}
    struct Params { resolution:f32, halo:f32, width:f32, height:f32, mode:f32, pad0:f32, pad1:f32, pad2:f32, lighting:vec4<f32>, sunlight:vec4<f32>, style:vec4<f32>, render0:vec4<f32>, render1:vec4<f32>, render2:vec4<f32>, render3:vec4<f32>, render4:vec4<f32> }
    @group(0) @binding(0) var elevation: texture_2d<f32>;
    @group(0) @binding(1) var blurred: texture_2d<f32>;
    @group(0) @binding(2) var color: texture_storage_2d<rgba8unorm,write>;
    @group(0) @binding(3) var<uniform> params:Params;
    @group(0) @binding(4) var climate:texture_2d_array<f32>;
    @group(0) @binding(5) var rows:texture_2d<f32>;
    @group(0) @binding(10) var fineGradient:texture_2d<f32>;
    @group(0) @binding(9) var erosionSurface:texture_2d<f32>;
    fn erosionAt(p:vec2<i32>)->vec4<f32>{return textureLoad(erosionSurface,clamp(p,vec2<i32>(0),vec2<i32>(textureDimensions(erosionSurface))-1),0);}
    fn clim(p:vec2<i32>,layer:i32)->f32 {
      let sourceSize=vec2<f32>(textureDimensions(elevation))-vec2<f32>(1.);
      let size=vec2<i32>(textureDimensions(climate));
      let q=vec2<f32>(p)/sourceSize*vec2<f32>(size-1);
      let lo=vec2<i32>(floor(q));let hi=min(lo+vec2<i32>(1),size-1);let f=fract(q);
      return mix(mix(textureLoad(climate,lo,layer,0).r,textureLoad(climate,vec2<i32>(hi.x,lo.y),layer,0).r,f.x),mix(textureLoad(climate,vec2<i32>(lo.x,hi.y),layer,0).r,textureLoad(climate,hi,layer,0).r,f.x),f.y);
    }
    ${materialI32}
    // Metres to the nearest sea sample (terrain_render.shore_distance) in the
    // square of half-width reach; only low land needs it. Beaches reach 189 m
    // (and at most 20 samples): farther sea is irrelevant. Exact separable
    // form: rows holds each sample's nearest sea column (shoreRowShader), so
    // the square costs one load per row instead of every ring sample.
    fn shoreDistance(p:vec2<i32>,height:f32,r:f32)->f32{
      var best=1e9;
      if(height<0.||height>=12.){return best;}
      let last=vec2<i32>(textureDimensions(elevation))-1;let reach=min(21,i32(ceil((190.+r)/r)));
      for(var k=0;k<=reach;k++){
        if(f32(k)*r>best){break;}
        for(var side=-1;side<=1;side+=2){
          let column=textureLoad(rows,vec2<i32>(p.x,clamp(p.y+side*k,0,last.y)),0).r;
          if(column<=f32(reach)){best=min(best,length(vec2<f32>(column,f32(k)))*r);}
        }
      }
      return best;
    }
    // Sea share of terrain_render.openness_probes around a shore sample.
    fn shoreOpenness(p:vec2<i32>,shore:f32,r:f32)->f32{
      if(shore>=190.+r){return 0.;}
      let last=vec2<i32>(textureDimensions(elevation))-1;let radius=clamp(300./r,2.,20.);var sea=0.;
      for(var k=0;k<24;k++){
        let a=6.2831853/12.*(f32(k%12)+.25+.5*f32(k/12));let o=vec2<i32>(round(vec2<f32>(cos(a),sin(a))*radius*select(.5,1.,k>=12)));
        let at=clamp(p+o,vec2<i32>(0),last);
        if(hydroGround(textureLoad(elevation,at,0).r,vec2<f32>(at))<0.){sea+=1.;}
      }
      return sea/24.;
    }
    fn renderSurface(p:vec2<i32>,height:f32)->vec3<f32>{
      // Derivatives match terrain_render.terrain_derivatives: central gradient and
      // the difference of the two Gaussian scales already used for hillshade.
      let local=vec2<f32>(p);let r=params.resolution;let blur=textureLoad(blurred,p,0).rg;
      var gradient=vec2<f32>(textureLoad(elevation,p+vec2<i32>(1,0),0).r-textureLoad(elevation,p-vec2<i32>(1,0),0).r,
        textureLoad(elevation,p+vec2<i32>(0,1),0).r-textureLoad(elevation,p-vec2<i32>(0,1),0).r)*.5/r;
      if(params.pad1>0.){gradient=textureLoad(fineGradient,p,0).rg;}
      let shore=shoreDistance(p,height,r);
      return surfaceMaterial(height,gradient,(blur.g-blur.r)/(6.*r),surfacePoint(local,params.style.xy,r),surfaceNorth(local,params.style.xy,r),r,
        vec3<f32>(clim(p,36),clim(p,37),clim(p,38)),surfacePedology(p),vec3<f32>(clim(p,39),clim(p,40),clim(p,41)),clim(p,42),
        clim(p,21),clim(p,22),clim(p,23),clim(p,24),clim(p,25),clim(p,26),params.mode == -4.,shore,shoreOpenness(p,shore,r),erosionAt(p));
    }
    ${rg?.wgsl||''}
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
    @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3<u32>) {
      if (any(id.xy>=textureDimensions(color))) { return; }
      let p=vec2<i32>(id.xy)+vec2<i32>(i32(params.halo));
      let height=hydroGround(textureLoad(elevation,p,0).r,vec2<f32>(p));
      let level=0.;
      let temp=clim(p,0)+clim(p,4)*max(height,0.);
      let rain=max(clim(p,2),0.);
      var rgb:vec3<f32>;
      var reliefLight=1.;
      if (height<0.) {
        let depth=pow(clamp(-height/10000.,0.,1.),.7);
        rgb=mix(vec3<f32>(.68,.88,1.),vec3<f32>(0.,.1,.45),depth);
      } else {
        var dx=(textureLoad(blurred,p+vec2<i32>(1,0),0).rg-textureLoad(blurred,p-vec2<i32>(1,0),0).rg)*3./params.resolution;
        var dy=(textureLoad(blurred,p+vec2<i32>(0,1),0).rg-textureLoad(blurred,p-vec2<i32>(0,1),0).rg)*3./params.resolution;
        /*fine-gradient*/if(params.pad1>0.){dx=vec2<f32>(textureLoad(fineGradient,p,0).r*6.);dy=vec2<f32>(textureLoad(fineGradient,p,0).g*6.);}/*end-fine-gradient*/
      let hs=pow(clamp(.75*shade(vec2<f32>(dx.r,dy.r))+.25*shade(vec2<f32>(dx.g,dy.g)),0.,1.),.85*params.lighting.z);
        reliefLight=mix(1.,params.lighting.y+(1.-params.lighting.y)*hs,params.lighting.x);
        rgb=terrain(.25+.75*pow(clamp(height/4500.,0.,1.),.7))*reliefLight;
      }
      var climateClass=0;
      if((params.mode == -1. || params.mode == -2.) && textureNumLayers(climate)>=36u){
        climateClass=koppen(clamp(clim(p,21)-clim(p,23)*max(height,0.),-45.,45.),clamp(clim(p,24)-clim(p,26)*max(height,0.),-45.,45.),max(clim(p,22),0.),max(clim(p,25),0.));
        if(height<0.){climateClass=0;}
      }
      ${rg?`/*resources*/if(params.mode<=-100.){if(res.zones2.z>.5&&(height>=0.||res.zones2.z>1.5)){rgb=resourceColor(p,height,reliefLight);}} else /*end-resources*/`:''}${v2?`if(params.mode<=-8. && textureNumLayers(climate)>=46u){
        var g=vec2<f32>(textureLoad(elevation,p+vec2<i32>(1,0),0).r-textureLoad(elevation,p-vec2<i32>(1,0),0).r,textureLoad(elevation,p+vec2<i32>(0,1),0).r-textureLoad(elevation,p-vec2<i32>(0,1),0).r)*.5/params.resolution;
        /*fine-gradient*/if(params.pad1>0.){g=textureLoad(fineGradient,p,0).rg;}/*end-fine-gradient*/
        let b=textureLoad(blurred,p,0).rg;
        let e=v2Evaluate(surfacePoint(vec2<f32>(p),params.style.xy,params.resolution),height,g,(b.g-b.r)/(6.*params.resolution),clim(p,21)-clim(p,23)*max(height,0.),clim(p,24)-clim(p,26)*max(height,0.),clim(p,22),clim(p,25),clim(p,42),0.,params.render3.w,params.resolution,u32(surfaceMeta(6)));
        rgb=v2Color(v2Appearance(e,params.resolution),max(0.,-params.mode-8.))*select(1.,reliefLight,params.mode == -8.);if(height<0.){rgb=v2Water(-height,1.,1.,e,surfacePoint(vec2<f32>(p),params.style.xy,params.resolution),params.resolution,u32(surfaceMeta(6)));}
      } else `:""}if(params.mode == -5. && textureNumLayers(climate)>=50u){
        rgb=select(vec3<f32>(.10,.22,.30),surfacePedology(p),clim(p,49)>=.5);
      } else if((params.mode == -3. || params.mode == -4.) && textureNumLayers(climate)>=46u){
        // Render and Soil: one material; the coarse path samples its baked albedo.
        rgb=renderSurface(p,height)*select(reliefLight,1.,height<0.);
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
  // Render3D bake (terrain_render3d.js): Render's land material with its guide
  // captured, plus forest field and climate on the whole tile, halo included,
  // so crown candidates beside a tile edge read real neighbouring values.
  // rgb: land albedo, even under water (the draw places the shoreline); a: Render's relief light.
  const v2ForestTarget=source=>{
    const legacy='let cleared=.4*params.render2.w*(1.-.85*trop);';
    if(!source.includes(legacy))throw Error('Render v2 clearing target hook missing');
    return source.replace(legacy,'let cleared=v2ClearingTarget(params.render2.w,humid,trop);');
  };
  const v2ForestBake=()=>{
    const legacy='cover*=1.-sstep(threshold-.25,threshold+.25,.4*r3Noise(16,p,vec2<f32>(31.7,-17.3),seed+41u+0x9e37u)/1.077+bias);';
    const coverage='cover=clamp(cover*(1.+mosaic*stand)+.05*stand*cover*(1.-cover),0.,1.)*(1.-.3*disturbed*(1.-.7*trop));';
    if(!r3.shaders.bake.includes(legacy))throw Error('Render v2 clearing profile hook missing');
    if(!r3.shaders.bake.includes(coverage))throw Error('Render v2 optical coverage hook missing');
    return v2ForestTarget(r3.shaders.bake).replace(coverage,'cover=v2StandCoverage(cover,stand,disturbed,trop);')
      .replace(legacy,'cover*=1.-v2ClearingFraction(.4*r3Noise(16,p,vec2<f32>(31.7,-17.3),seed+41u+0x9e37u)/1.077+bias,threshold);');
  };
  const v2ForestMaterial=()=>{
    const legacy='return mix(ground,litter,.65*(1.-.6*st.dry));';
    if(!r3.shaders.forestMaterial.includes(legacy))throw Error('Render v2 coarse snow/litter hook missing');
    return r3.shaders.forestMaterial.replace(legacy,'if(params.field.z<=-8.){litter=mix(litter,ground,v2GroundSnow);}return mix(ground,litter,.65*(1.-.6*st.dry));');
  };
  const render3dBakeShader=(auxFormat='rgba8unorm')=>shadeShader.replace(materialI32,()=>r3.withCaptures(materialI32))+`
    @group(0) @binding(6) var aux:texture_storage_2d_array<${auxFormat},write>;
    @group(0) @binding(7) var<uniform> r3:R3Tile;
    ${r3.shaders.common}
    ${auxFormat==='rgba16float'&&v2?v2ForestBake():r3.shaders.bake}
    fn r3Load(p:vec2<i32>)->f32{return textureLoad(elevation,clamp(p,vec2<i32>(0),vec2<i32>(textureDimensions(elevation))-1),0).r;}
    fn r3Blur(p:vec2<i32>)->vec2<f32>{return textureLoad(blurred,clamp(p,vec2<i32>(0),vec2<i32>(textureDimensions(blurred))-1),0).rg;}
    @compute @workgroup_size(8,8) fn bake(@builtin(global_invocation_id) id:vec3<u32>){
      let p=vec2<i32>(id.xy);if(any(p>=vec2<i32>(textureDimensions(elevation)))){return;}
      let r=params.resolution;let height=hydroGround(textureLoad(elevation,p,0).r,vec2<f32>(p));let blur=textureLoad(blurred,p,0).rg;let local=vec2<f32>(p);
      var gradient=vec2<f32>(r3Load(p+vec2<i32>(1,0))-r3Load(p-vec2<i32>(1,0)),r3Load(p+vec2<i32>(0,1))-r3Load(p-vec2<i32>(0,1)))*.5/r;
      if(params.pad1>0.){gradient=textureLoad(fineGradient,p,0).rg;}
      // Submerged samples are evaluated as shore, so filtering never mixes inland colour into beaches.
      let shore=select(shoreDistance(p,height,r),0.,height<0.);let point=surfacePoint(local,params.style.xy,r);
      if(params.mode<=-8.){surfaceDetailOverride=vec4<f32>(surfaceLandPoint(local,params.style.xy,r),1.);}
      let land=surfaceMaterial(max(height,.5),gradient,(blur.g-blur.r)/(6.*r),point,surfaceNorth(local,params.style.xy,r),r,
        vec3<f32>(clim(p,36),clim(p,37),clim(p,38)),surfacePedology(p),vec3<f32>(clim(p,39),clim(p,40),clim(p,41)),clim(p,42),
        clim(p,21),clim(p,22),clim(p,23),clim(p,24),clim(p,25),clim(p,26),false,shore,shoreOpenness(p,shore,r),erosionAt(p));
      var dx=(r3Blur(p+vec2<i32>(1,0))-r3Blur(p-vec2<i32>(1,0)))*3./r;var dy=(r3Blur(p+vec2<i32>(0,1))-r3Blur(p-vec2<i32>(0,1)))*3./r;
      if(params.pad1>0.){dx=vec2<f32>(textureLoad(fineGradient,p,0).r*6.);dy=vec2<f32>(textureLoad(fineGradient,p,0).g*6.);}
      let hs=pow(clamp(.75*shade(vec2<f32>(dx.r,dy.r))+.25*shade(vec2<f32>(dx.g,dy.g)),0.,1.),.85*params.lighting.z);
      let light=mix(1.,params.lighting.y+(1.-params.lighting.y)*hs,params.lighting.x);
      let inner=p-vec2<i32>(i32(params.halo));
      if(all(inner>=vec2<i32>(0))&&all(inner<vec2<i32>(textureDimensions(color)))){textureStore(color,inner,vec4<f32>(land,light));}
      ${v2?`if(params.mode<=-8.){
        if(all(inner>=vec2<i32>(0))&&all(inner<vec2<i32>(textureDimensions(color)))){textureStore(color,inner,vec4<f32>(r3BaseOpen,light));}
        var bank=0.;
        if(params.pad0<.5&&r<=60.){for(var k=0;k<8;k++){let a=f32(k)*.785398163;let o=vec2<f32>(cos(a),sin(a));for(var j=1;j<=3;j++){let w=hydroAt(local+o*f32(j));if(w.z>2.5&&w.w>.5){bank=max(bank,w.y*(1.-sstep(0.,65.,f32(j)*r)));}}}}
        let e=v2Evaluate(point,height,gradient,(blur.g-blur.r)/(6.*r),clim(p,21)-clim(p,23)*max(height,0.),clim(p,24)-clim(p,26)*max(height,0.),clim(p,22),clim(p,25),clim(p,42),bank,.5,0.,r3Seed());
        let water=hydroAt(local);let eligible=select(1.,0.,height<0.||(water.w>.5&&water.z>=1.5&&(water.z<2.5||params.pad0<.5)&&water.y>.05));
        var glacier=0.;if(textureNumLayers(climate)>=51u){glacier=clim(p,50);}
        // Flat coastal bars are a separate inherited sediment contract, not
        // a wider tree-exclusion beach. Keep its raw coverage for pixel-scale
        // composition; blue separately carries the filtered macro Y slope.
        let macroGradient=.75*vec2<f32>(dx.r,dy.r)+.25*vec2<f32>(dx.g,dy.g);
        textureStore(aux,p,0,vec4<f32>(glacier,r3Flats,macroGradient.y,clamp(.5+(blur.g-blur.r)/(6.*r),0.,1.)));
        textureStore(aux,p,5,vec4<f32>(r3BaseClosed,macroGradient.x));
        textureStore(aux,p,1,vec4<f32>(r3Sediment,clim(p,22)/max(1.,clim(p,22)+clim(p,25))));
        var forest=r3ForestField(height,length(gradient),(local+.5)*r,r3Seed());
        let erosion=erosionAt(p);forest.x*=(1.-.8*erosion.x)*(1.-.85*erosion.z);forest.y*=1.-.6*max(erosion.z,erosion.w);
        let bankTrees=bank*.5*sstep(5.5,9.,r3THot)*(1.-r3Perennial)*sstep(.025,.08,forest.g);
        textureStore(aux,p,2,vec4<f32>(clamp(forest.r+bankTrees,0.,.985)*eligible,forest.gb,bank));
        let ts=clim(p,21)-clim(p,23)*max(height,0.);let tw=clim(p,24)-clim(p,26)*max(height,0.);
        // Use the same physical seasonal lapse and denominator on encode and
        // decode. The legacy fixed .0065 silently changed rainfall with altitude.
        textureStore(aux,p,3,vec4<f32>((ts+50.)/100.,(clim(p,22)+clim(p,25))/max(120.,20.*(ts+tw)*.5+280.)/5.,(tw+50.)/100.,clim(p,42)));
        textureStore(aux,p,4,vec4<f32>(clamp(r3Woods+bankTrees,0.,.985)*eligible,r3Herbs*eligible,r3Outcrop,r3Beach));
        return;
      }`:""}
      var field=r3ForestField(height,length(gradient),(local+.5)*r,r3Seed());
      let erosion=erosionAt(p);field.x*=(1.-.8*erosion.x)*(1.-.85*erosion.z);field.y*=1.-.6*max(erosion.z,erosion.w);
      textureStore(aux,p,0,vec4<f32>(r3GuideOpen,r3Woods));
      textureStore(aux,p,1,vec4<f32>(r3GuideClosed,clamp((.2+.04*params.render0.y*r3M1)/.4,0.,1.)));
      textureStore(aux,p,2,field);
      textureStore(aux,p,3,vec4<f32>(clamp(((r3TS+r3TW)*.5+50.)/100.,0.,1.),clamp(r3Humid/4.,0.,1.),clamp((min(r3TS,r3TW)+50.)/100.,0.,1.),
        surfaceSeaIce(point,clim(p,21),clim(p,24),r)));
      textureStore(aux,p,4,vec4<f32>(0.));
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
    ${hydrologyShader}
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
      let hydroPixel=input.uv*(params.field.xy-2.*params.viewport.z)+params.viewport.z-.5;
      let hp=clamp(hydroPixel,vec2<f32>(0.),vec2<f32>(textureDimensions(elevation))-1.);let h0=vec2<i32>(floor(hp));let h1=min(h0+1,vec2<i32>(textureDimensions(elevation))-1);let hf=fract(hp);
      let hydroHeight=mix(mix(textureLoad(elevation,h0,0).r,textureLoad(elevation,vec2<i32>(h1.x,h0.y),0).r,hf.x),mix(textureLoad(elevation,vec2<i32>(h0.x,h1.y),0).r,textureLoad(elevation,h1,0).r,hf.x),hf.y);
      let hydroFootprint=max(length(dpdx(hydroPixel)),length(dpdy(hydroPixel)));
      let hydroRgb=hydroBlend(sampled.rgb,hydroHeight,hydroFiltered(hydroPixel,hydroFootprint));
      ${city?`if(paperAmount(params.field.z)>0.){
        let p=vec2<i32>(input.uv*(params.field.xy-2.*params.viewport.z)+params.viewport.z);
        let height=textureLoad(elevation,clamp(p,vec2<i32>(0),vec2<i32>(textureDimensions(elevation))-1),0).r;
        return vec4<f32>(paperColor(hydroRgb,params.field.z,height,input.position.xy),sampled.a);
      }`:''}
      return vec4<f32>(hydroRgb,sampled.a);
    }`;

  // Shaded colours of a refined tile, box-averaged per mip: a finer cached
  // tile shown below one sample per pixel would otherwise alias its metre
  // relief into speckle.
  const colorMipShader=`
    @group(0) @binding(0) var source:texture_2d<f32>;
    @group(0) @binding(1) var dest:texture_storage_2d<rgba8unorm,write>;
    @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3<u32>){
      if(any(id.xy>=textureDimensions(dest))){return;}
      let last=vec2<i32>(textureDimensions(source))-1;let p=vec2<i32>(id.xy)*2;
      let sum=textureLoad(source,min(p,last),0)+textureLoad(source,min(p+vec2<i32>(1,0),last),0)+textureLoad(source,min(p+vec2<i32>(0,1),last),0)+textureLoad(source,min(p+vec2<i32>(1,1),last),0);
      textureStore(dest,vec2<i32>(id.xy),sum*.25);
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
  const coarseColor=shadeShader.slice(shadeShader.indexOf('      let temp='),shadeShader.indexOf('      textureStore(color')).replace(/\/\*resources\*\/[\s\S]*?\/\*end-resources\*\//g,rg?'if(params.mode<=-100.){if(res.zones2.z>.5&&(height>=0.||res.zones2.z>1.5)){rgb=resourceCoarseColor(p,height,reliefLight,footprint);}} else ':'').replace(/\/\*fine-gradient\*\/[\s\S]*?\/\*end-fine-gradient\*\//g,'')
    .replaceAll('params.mode','params.field.z')
    .replace('var g=vec2<f32>(textureLoad(elevation,p+vec2<i32>(1,0),0).r-textureLoad(elevation,p-vec2<i32>(1,0),0).r,textureLoad(elevation,p+vec2<i32>(0,1),0).r-textureLoad(elevation,p-vec2<i32>(0,1),0).r)*.5/params.resolution;', 'let g=vec2<f32>(heightAt(p+vec2<f32>(1,0),0.)-heightAt(p-vec2<f32>(1,0),0.),heightAt(p+vec2<f32>(0,1),0.)-heightAt(p-vec2<f32>(0,1),0.))*.5/params.viewport.w;')
    .replace('let b=textureLoad(blurred,p,0).rg;', 'let b=vec2<f32>(heightAt(p,3.),height);')
    .replaceAll('vec2<f32>(p)','p')
    .replace('var dx=(textureLoad(blurred,p+vec2<i32>(1,0),0).rg-textureLoad(blurred,p-vec2<i32>(1,0),0).rg)*3./params.resolution;',
      'let dx=(heightAt(p+vec2<f32>(1,0),level)-heightAt(p-vec2<f32>(1,0),level))*3./params.viewport.w;')
    .replace('var dy=(textureLoad(blurred,p+vec2<i32>(0,1),0).rg-textureLoad(blurred,p-vec2<i32>(0,1),0).rg)*3./params.resolution;',
      'let dy=(heightAt(p+vec2<f32>(0,1),level)-heightAt(p-vec2<f32>(0,1),level))*3./params.viewport.w;')
    .replaceAll('*params.resolution','*params.viewport.w')
    .replace('let deltaX=(textureLoad(blurred,p+vec2<i32>(1,0),0).rg-textureLoad(blurred,p-vec2<i32>(1,0),0).rg)*.5/params.resolution;','let deltaX=vec2<f32>((heightAt(p+vec2<f32>(1,0),level)-heightAt(p-vec2<f32>(1,0),level))*.5/params.viewport.w);')
    .replace('let deltaY=(textureLoad(blurred,p+vec2<i32>(0,1),0).rg-textureLoad(blurred,p-vec2<i32>(0,1),0).rg)*.5/params.resolution;','let deltaY=vec2<f32>((heightAt(p+vec2<f32>(0,1),level)-heightAt(p-vec2<f32>(0,1),level))*.5/params.viewport.w);')
    .replaceAll('/params.resolution','/params.viewport.w')
    .replace('vec2<f32>(p)','p')
    .replace('let hs=pow(clamp(.75*shade(vec2<f32>(dx.r,dy.r))+.25*shade(vec2<f32>(dx.g,dy.g)),0.,1.),.85*params.lighting.z);',
      'let hs=pow(shade(vec2<f32>(dx,dy)),.85*params.lighting.z);')
    .replaceAll('params.resolution','params.viewport.w')
    .replace('rgb=v2Color(v2Appearance(e,params.viewport.w),max(0.,-params.field.z-8.))*select(1.,reliefLight,params.field.z == -8.);',
      `if(params.field.z == -8.){
        let mean=textureSampleLevel(albedo,filtering,(p+.5)/params.field.xy,0.);
        let flatHS=pow(clamp(params.sunlight.z,0.,1.),.85*params.lighting.z);
        let flatLight=mix(1.,params.lighting.y+(1.-params.lighting.y)*flatHS,params.lighting.x);
        rgb=v2DisplayShoulder(mean.rgb*reliefLight/max(flatLight,.001));
      }else{rgb=v2Color(v2Appearance(e,params.viewport.w),max(0.,-params.field.z-8.));}`);
  const coarseParams=`
    struct Params {rect:vec4<f32>,viewport:vec4<f32>,crop:vec4<f32>,field:vec4<f32>,lighting:vec4<f32>,sunlight:vec4<f32>,style:vec4<f32>,contours:vec4<f32>,render0:vec4<f32>,render1:vec4<f32>,render2:vec4<f32>,render3:vec4<f32>,render4:vec4<f32>}
`;
  // Monotone bicubic physical heights and bilinear climate in block samples.
  const coarseSampling=`
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
    fn clim(p:vec2<f32>,layer:i32)->f32{
      let size=vec2<i32>(textureDimensions(climate));let q=clamp(p/params.field.xy,vec2<f32>(0.),vec2<f32>(1.))*vec2<f32>(size-1);
      let lo=vec2<i32>(floor(q));let hi=min(lo+1,size-1);let f=fract(q);
      return mix(mix(textureLoad(climate,lo,layer,0).r,textureLoad(climate,vec2<i32>(hi.x,lo.y),layer,0).r,f.x),mix(textureLoad(climate,vec2<i32>(lo.x,hi.y),layer,0).r,textureLoad(climate,hi,layer,0).r,f.x),f.y);
    }
`;
  const coarseShader=`
    ${coarseParams}
    struct Vertex {@builtin(position) position:vec4<f32>,@location(0) uv:vec2<f32>}
    @group(0) @binding(0) var<uniform> params:Params;
    @group(0) @binding(1) var elevation:texture_2d<f32>;
    @group(0) @binding(2) var climate:texture_2d_array<f32>;
    @group(0) @binding(3) var albedo:texture_2d<f32>;
    @group(0) @binding(4) var filtering:sampler;
    @vertex fn vertex(@builtin(vertex_index) id:u32)->Vertex{
      let corners=array<vec2<f32>,6>(vec2<f32>(0,0),vec2<f32>(1,0),vec2<f32>(0,1),vec2<f32>(0,1),vec2<f32>(1,0),vec2<f32>(1,1));
      let uv=corners[id];let xy=params.rect.xy+uv*params.rect.zw;
      var out:Vertex;out.position=vec4<f32>(xy.x/params.viewport.x*2.-1.,1.-xy.y/params.viewport.y*2.,0.,1.);out.uv=params.crop.xy+uv*params.crop.zw;return out;
    }
    ${coarseSampling}
    ${materialShader.replaceAll('PTYPE','f32')}
    ${v2?.environment||''}
    ${r3?.shaders.lighting.replace('pow(clamp(c,vec3<f32>(0.),vec3<f32>(1.)),vec3<f32>(1./2.2))','pow(max(c,vec3<f32>(0.)),vec3<f32>(1./2.2))')||''}
    fn v2MeanShade(color:vec3<f32>,roughness:f32)->vec3<f32>{
      var m:GroundMaterial;m.albedo=groundToLinear(color);m.normal=vec3<f32>(0.,1.,0.);m.roughness=roughness;m.occlusion=1.;
      let l=normalize(vec3<f32>(-params.sunlight.x,params.sunlight.z,-params.sunlight.y));
      let angle=atan2(l.z,l.x)+2.35619449;let horizontal=length(l.xz)/.70710678;
      let fill=vec2<f32>(.2*cos(angle)+.75*sin(angle),.2*sin(angle)-.75*cos(angle))*horizontal;
      let l2=vec3<f32>(fill.x,.63*l.y/.70710678,fill.y);
      return shadeTerrainGround(m,vec3<f32>(0.,1.,0.),l,l2,params.lighting.x);
    }
    fn renderSurface(p:vec2<f32>,height:f32)->vec3<f32>{
      // Materials are baked once per block and setting (rgb: land, a: sea ice);
      // the coastline stays at the per-pixel interpolated DEM.
      let baked=textureSampleLevel(albedo,filtering,(p+.5)/params.field.xy,0.);
      if(height>=0.){return baked.rgb;}
      return surfaceWater(-height,baked.a);
    }
    ${coarsePalette}
    ${rg?.coarse||''}
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
  // Render3D modes (-6 analytic crowns, -7 brushes) on fine tiles; native
  // coarse blocks are far beyond their reveal and keep Render (-3).
  const isV2=mode=>mode<=-8&&mode>=-12;
  const isRender3d=mode=>mode===-6||mode===-7||isV2(mode);
  const coarseMode=mode=>isV2(mode)?mode:isRender3d(mode)?-3:mode;
  // Render/Soil materials of a native coarse block at twice its sample density.
  // The footprint stays the block resolution, like the CPU reference.
  const coarseBakeShader=`
    ${coarseParams}
    @group(0) @binding(0) var<uniform> params:Params;
    @group(0) @binding(1) var elevation:texture_2d<f32>;
    @group(0) @binding(2) var climate:texture_2d_array<f32>;
    @group(0) @binding(3) var albedo:texture_storage_2d<rgba8unorm,write>;
    ${coarseSampling}
    ${r3?.withCaptures(materialShader.replaceAll('PTYPE','f32'))||materialShader.replaceAll('PTYPE','f32')}
    ${r3?.captureState||''}
    ${v2?.environment||''}
    ${r3?.shaders.lighting.replace('pow(clamp(c,vec3<f32>(0.),vec3<f32>(1.)),vec3<f32>(1./2.2))','pow(max(c,vec3<f32>(0.)),vec3<f32>(1./2.2))')||''}
    fn groundSmooth(a:f32,b:f32,x:f32)->f32{let t=clamp((x-a)/(b-a),0.,1.);return t*t*(3.-2.*t);}
    var<private> v2GroundSnow:f32;
    ${r3?v2ForestMaterial():''}
    ${v2&&r3?v2ForestTarget(r3.shaders.forestExpected):r3?.shaders.forestExpected||''}
    ${v2?.canopyMeanShader||''}
    fn v2CoarseResponse(rgb:vec3<f32>,roughness:f32,occlusion:f32,l:vec3<f32>,l2:vec3<f32>)->vec3<f32>{
      var m:GroundMaterial;m.albedo=groundToLinear(rgb);m.normal=vec3<f32>(0.,1.,0.);m.roughness=roughness;m.occlusion=occlusion;
      return groundToLinear(shadeTerrainGround(m,vec3<f32>(0.,1.,0.),l,l2,params.lighting.x));
    }
    @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3<u32>){
      if(any(id.xy>=textureDimensions(albedo))){return;}
      let p=(vec2<f32>(id.xy)+.5)*.5-.5;let r=params.viewport.w;
      let height=heightAt(p,0.);
      let gradient=vec2<f32>(heightAt(p+vec2<f32>(.5,0.),0.)-heightAt(p-vec2<f32>(.5,0.),0.),heightAt(p+vec2<f32>(0.,.5),0.)-heightAt(p-vec2<f32>(0.,.5),0.))/r;
      // Box mips approximate the Gaussian relief position of fine tiles.
      let tpi=(height-heightAt(p,3.))/(6.*r);
      let point=surfacePoint(p,params.style.xy,r);
      if(params.field.z<=-8.){surfaceDetailOverride=vec4<f32>(surfaceLandPoint(p,params.style.xy,r),1.);}
      // Same sea distance and openness as the fine tiles (shoreDistance and
      // shoreOpenness), on the coarse grid.
      var shore=1e9;var openness=0.;
      if(height>=0.&&height<12.){
        let reach=min(21,i32(ceil((190.+r)/r)));
        for(var j=-reach;j<=reach;j++){for(var i=-reach;i<=reach;i++){
          let o=vec2<f32>(f32(i),f32(j));
          if(length(o)*r<shore&&heightAt(p+o,0.)<0.){shore=length(o)*r;}
        }}
      }
      if(shore<190.+r){
        let radius=clamp(300./r,2.,20.);
        for(var k=0;k<24;k++){
          let a=6.2831853/12.*(f32(k%12)+.25+.5*f32(k/12));
          if(heightAt(p+round(vec2<f32>(cos(a),sin(a))*radius*select(.5,1.,k>=12)),0.)<0.){openness+=1./24.;}
        }
      }
      let land=surfaceMaterial(max(height,.5),gradient,tpi,point,surfaceNorth(p,params.style.xy,r),r,
        vec3<f32>(clim(p,36),clim(p,37),clim(p,38)),surfacePedology(p),vec3<f32>(clim(p,39),clim(p,40),clim(p,41)),clim(p,42),
        clim(p,21),clim(p,22),clim(p,23),clim(p,24),clim(p,25),clim(p,26),params.field.z == -4.,shore,openness,vec4<f32>(0.));
      ${v2?`if(params.field.z<=-8.){
        var e=v2Appearance(v2EvaluateDomains(point,surfaceDetailOverride.xyz,height,gradient,tpi,
          clim(p,21)-clim(p,23)*max(height,0.),clim(p,24)-clim(p,26)*max(height,0.),clim(p,22),clim(p,25),clim(p,42),0.,params.render3.w,r,u32(surfaceMeta(6))),r);
        let beach=clamp(r3Beach/.8,0.,1.);let eligible=select(0.,1.,height>=0.);
        let population=r3ForestExpected(height,length(gradient));
        e.woody=population.r*eligible*(1.-beach);e.needle=population.b;
        e.grass=r3Herbs*eligible*(1.-e.snow)*(1.-beach);e.rock=r3Outcrop*(1.-beach);
        let flats=r3Flats*(1.-sstep(240.,960.,r))*.8;
        let opened=mix(v2GroundTint(r3BaseOpen,e),vec3<f32>(.88,.91,.91),e.snow);
        var leafOn=e;leafOn.leaf=1.;leafOn.snow=0.;var snowFree=e;snowFree.snow=0.;
        var closed=r3BaseClosed*v2Canopy(snowFree)/max(v2Canopy(leafOn),vec3<f32>(.02));
        closed=mix(closed,vec3<f32>(.72,.79,.77),e.snow*.62);
        let l=normalize(vec3<f32>(-params.sunlight.x,params.sunlight.z,-params.sunlight.y));
        let angle=atan2(l.z,l.x)+2.35619449;let horizontal=length(l.xz)/.70710678;
        let fill=vec2<f32>(.2*cos(angle)+.75*sin(angle),.2*sin(angle)-.75*cos(angle))*horizontal;
        let l2=vec3<f32>(fill.x,.63*l.y/.70710678,fill.y);
        // Reuse the exact pure forest climate/material functions. Their
        // decoder takes annual/coldest temperature and potential/4.
        let annual=(r3TS+r3TW)*.5;let potential=(clim(p,22)+clim(p,25))/max(120.,20.*annual+280.);
        let fc=forestClimate(vec4<f32>((annual+50.)/100.,potential/4.,(min(r3TS,r3TW)+50.)/100.,0.));
        var st:ForestStand;st.p=vec2<f32>(0.);st.cover=population.r;st.height=population.g*50.;st.needle=population.b;st.tone=e.tone;
        st.trop=fc.trop;st.dry=fc.dry;st.kappa=forestKappa(fc.humid,st.height,st.needle,st.trop);
        let density=forestDensity(st.cover,st.kappa);let available=eligible*(1.-beach);
        let share=st.cover/max(density,.001)*available;let canopy=st.cover*e.leaf*available;
        let weights=vec3<f32>(1.-share,max(share-canopy,0.),canopy);
        let gain=mix(mix(.02,.45,e.rock)+.07*(1.-e.grass),.008,e.snow);
        // Every ground band is unresolved at a native coarse block footprint.
        let variance=${(r3?.BANDS||[]).reduce((sum,b)=>sum+(b.amplitude/b.wavelength)**2*(1+1/(b.aspect*b.aspect))*b.variance,0).toPrecision(12)};
        let roughness=clamp(pow(pow(mix(.88,.54,e.rock),4.)+gain*gain*variance,.25),.4,.98);
        v2GroundSnow=e.snow;
        let litter=forestFloor(opened,st,fc.cold);
        let crownBase=forestCrownColor(st,closed,st.needle,vec2<f32>(0.),0.);
        let crown=crownBase*mix(vec3<f32>(1.),v2CanopyMeanRatio(crownBase,st.needle,l,l2,params.lighting.x),1.-e.snow);
        var response=weights.x*v2CoarseResponse(opened,roughness,1.,l,l2);
        if(weights.y>0.){response+=weights.y*v2CoarseResponse(litter,roughness,.72,l,l2);}
        if(weights.z>0.){response+=weights.z*v2CoarseResponse(crown,.9,1.,l,l2);}
        let flatWeight=flats*(1.-e.snow);let beachWeight=r3Beach*(1.-e.snow);
        if(flatWeight>0.){response=mix(response,v2CoarseResponse(r3Sediment*.9,roughness,1.,l,l2),flatWeight);}
        if(beachWeight>0.){response=mix(response,v2CoarseResponse(r3Sediment,roughness,1.,l,l2),beachWeight);}
        // Light-dependent flat response, still HDR. The display fragment adds
        // normalized macro relief and the shoulder exactly once.
        textureStore(albedo,vec2<i32>(id.xy),vec4<f32>(groundToDisplay(response),canopy));return;
      }`:''}
      textureStore(albedo,vec2<i32>(id.xy),vec4<f32>(land,surfaceSeaIce(point,clim(p,21),clim(p,24),r)));
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
      this.noGradient=this.makeTexture(1,1,'rg32float',GPUTextureUsage.COPY_DST);device.queue.writeTexture({texture:this.noGradient},new Float32Array(2),{bytesPerRow:8},[1,1]);
      this.noRiverDisplay=this.device.createTexture({size:[1,1],format:'rgba16float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
      device.queue.writeTexture({texture:this.noRiverDisplay},new Uint16Array(4),{bytesPerRow:8},[1,1]);
      this.noHydrology=this.makeTexture(1,1,'rgba32float',GPUTextureUsage.COPY_DST);
      if(rg)this.resetResources();
      device.queue.writeTexture({texture:this.noHydrology},new Float32Array(4),{bytesPerRow:16},[1,1]);
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
      const [blur,shade,render,coarse,mip,contour,bake,bakeV2,colorMip,shoreRows,waterMip]=await Promise.all([module(blurShader,'terrain Gaussian'),module(shadeShader,'terrain relief'),module(renderShader,'terrain quads'),module(coarseShader,'native coarse relief'),module(coarseMipShader,'native coarse mips'),module(contourShader,'terrain vector contours'),module(coarseBakeShader,'native coarse materials'),module(coarseBakeShader.replace('texture_storage_2d<rgba8unorm,write>','texture_storage_2d<rgba16float,write>'),'native coarse V2 responses'),module(colorMipShader,'terrain colour mips'),module(hydrologyShader+shoreRowShader,'terrain shore rows'),module(hydrologyMipShader,'terrain water mips')]);
      this.compileShader=module;
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
        {binding:10,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
        {binding:9,visibility:GPUShaderStage.COMPUTE,texture:{}},
        {binding:5,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
        {binding:8,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
        ...(rg?[{binding:11,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'2d-array'}},
          {binding:12,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'2d-array'}},
          {binding:13,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
          {binding:14,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
          {binding:15,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}}]:[]),
      ]});
      this.renderLayout=device.createBindGroupLayout({entries:[
        {binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:208}},
        {binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{}},
        {binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:'filtering'}},
        {binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'unfilterable-float'}},
        {binding:8,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'unfilterable-float'}},
      ]});
      this.coarseLayout=device.createBindGroupLayout({entries:[
        {binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:208}},
        {binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'unfilterable-float'}},
        {binding:2,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'unfilterable-float',viewDimension:'2d-array'}},
        {binding:3,visibility:GPUShaderStage.FRAGMENT,texture:{}},
        {binding:4,visibility:GPUShaderStage.FRAGMENT,sampler:{type:'filtering'}},
        ...(rg?[{binding:11,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'unfilterable-float',viewDimension:'2d-array'}},
          {binding:12,visibility:GPUShaderStage.FRAGMENT,texture:{sampleType:'unfilterable-float',viewDimension:'2d-array'}},
          {binding:13,visibility:GPUShaderStage.FRAGMENT,buffer:{type:'uniform'}},
          {binding:14,visibility:GPUShaderStage.FRAGMENT,buffer:{type:'read-only-storage'}},
          {binding:15,visibility:GPUShaderStage.FRAGMENT,buffer:{type:'read-only-storage'}}]:[]),
      ]});
      this.bakeLayout=device.createBindGroupLayout({entries:[
        {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
        {binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
        {binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'2d-array'}},
        {binding:3,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:'write-only',format:'rgba8unorm'}},
      ]});
      this.bakePipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[this.bakeLayout]}),compute:{module:bake,entryPoint:'main'}});
      this.v2CoarseBakeLayout=device.createBindGroupLayout({entries:[
        {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
        {binding:1,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float'}},
        {binding:2,visibility:GPUShaderStage.COMPUTE,texture:{sampleType:'unfilterable-float',viewDimension:'2d-array'}},
        {binding:3,visibility:GPUShaderStage.COMPUTE,storageTexture:{access:'write-only',format:'rgba16float'}},
      ]});
      this.v2CoarseBakePipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[this.v2CoarseBakeLayout]}),compute:{module:bakeV2,entryPoint:'main'}});
      this.noAlbedo=this.makeTexture(1,1,'rgba8unorm');
      this.noSurface=this.makeTexture(1,1,'rgba8unorm',GPUTextureUsage.COPY_DST);
      device.queue.writeTexture({texture:this.noSurface},new Uint8Array(4),{bytesPerRow:4},[1,1]);
      this.coarsePipeline=await device.createRenderPipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[this.coarseLayout]}),vertex:{module:coarse,entryPoint:'vertex'},fragment:{module:coarse,entryPoint:'fragment',targets:[{format:this.format}]},primitive:{topology:'triangle-list'}});
      this.coarseMipPipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:mip,entryPoint:'main'}});
      this.colorMipPipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:colorMip,entryPoint:'main'}});
      this.shoreRowPipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:shoreRows,entryPoint:'main'}});
      this.waterMipPipeline=await device.createComputePipelineAsync({layout:'auto',compute:{module:waterMip,entryPoint:'main'}});
      [this.horizontal,this.vertical,this.shade,this.pipeline]=await Promise.all([
        device.createComputePipelineAsync({layout:blurLayout,compute:{module:blur,entryPoint:'horizontal'}}),
        device.createComputePipelineAsync({layout:blurLayout,compute:{module:blur,entryPoint:'vertical'}}),
        device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[this.shadeLayout]}),compute:{module:shade,entryPoint:'main'}}),
        device.createRenderPipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[this.renderLayout]}),vertex:{module:render,entryPoint:'vertex'},fragment:{module:render,entryPoint:'fragment',targets:[{format:this.format}]},primitive:{topology:'triangle-list'}}),
      ]);
      this.contourLayout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.VERTEX|GPUShaderStage.FRAGMENT,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:208}}]});
      this.contourPipeline=await device.createRenderPipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[this.contourLayout]}),vertex:{module:contour,entryPoint:'vertex',buffers:[{arrayStride:20,stepMode:'instance',attributes:[{shaderLocation:0,offset:0,format:'float32x4'},{shaderLocation:1,offset:16,format:'float32'}]}]},fragment:{module:contour,entryPoint:'fragment',targets:[{format:this.format,blend:{color:{srcFactor:'src-alpha',dstFactor:'one-minus-src-alpha'},alpha:{srcFactor:'one',dstFactor:'one-minus-src-alpha'}}}]},primitive:{topology:'triangle-list'}});
      const validation=await device.popErrorScope(); if(validation)throw Error(validation.message);
      this.sampler=device.createSampler({magFilter:'linear',minFilter:'linear',mipmapFilter:'linear'});
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
      this.scratchGeneration=(this.scratchGeneration||0)+1;
    }
    uploadTile(key, heights, {width=304,height=304,halo=24,metresPerSample=30,climate=null,climateWidth=33,climateHeight=33,hydrology=null,surface=null,gradient=null,mode='relief',lod=0,origin=[0,0]}={}) {
      if(!this.available)return false;
      if(!(heights instanceof Float32Array)||heights.length!==width*height||![width,height,halo].every(Number.isInteger)||halo<1||width<=2*halo||height<=2*halo||!Number.isFinite(metresPerSample)||metresPerSample<=0)throw Error('Invalid physical tile');
      if(Math.max(width,height)>this.device.limits.maxTextureDimension2D)throw Error('Tile exceeds GPU limits');
      if(hydrology!==null&&(!(hydrology instanceof Float32Array)||hydrology.length!==width*height*4))throw Error('Invalid water tile');
      if(surface!==null&&(!(surface instanceof Uint8Array)||surface.length!==width*height*4))throw Error('Invalid erosion surface tile');
      this.deleteTile(key); const started=performance.now(); this.scratch(width,height);
      const innerWidth=width-2*halo,innerHeight=height-2*halo;
      const elevation=this.makeTexture(width,height,'r32float',GPUTextureUsage.COPY_DST);
      const colorMips=Math.min(5,Math.floor(Math.log2(Math.min(innerWidth,innerHeight)))+1);
      const color=this.device.createTexture({size:[innerWidth,innerHeight],format:'rgba8unorm',mipLevelCount:colorMips,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC});
      if(climate!==null&&(!(climate instanceof Float32Array)||![5,21,36,46,50,51].includes(climate.length/(climateWidth*climateHeight))))throw Error('Invalid ticlimatee');
      const climateLayers=climate?climate.length/(climateWidth*climateHeight):5;
      const climateTexture=this.device.createTexture({size:[climateWidth,climateHeight,climateLayers],format:'r32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
      const climateValues=climate||new Float32Array(climateWidth*climateHeight*5);
      this.device.queue.writeTexture({texture:climateTexture},climateValues,{bytesPerRow:climateWidth*4,rowsPerImage:climateHeight},[climateWidth,climateHeight,climateLayers]);
      this.device.queue.writeTexture({texture:elevation},heights,{bytesPerRow:width*4,rowsPerImage:height},[width,height]);
      const selectedMode=modeNumber(this.mode??mode);
      // Render's own material seed (climate metadata), shared by Render3D lattices.
      const materialSeed=climateLayers>=46?climateValues[45*climateWidth*climateHeight+6]>>>0:0;
      const gradientTexture=gradient?this.makeTexture(width,height,'rg32float',GPUTextureUsage.COPY_DST):null;
      if(gradientTexture)this.device.queue.writeTexture({texture:gradientTexture},gradient,{bytesPerRow:width*8,rowsPerImage:height},[width,height]);
      const erosion=surface?this.makeTexture(width,height,'rgba8unorm',GPUTextureUsage.COPY_DST):null;
      if(erosion)this.device.queue.writeTexture({texture:erosion},surface,{bytesPerRow:width*4,rowsPerImage:height},[width,height]);
      const waterLevels=hydrology?Math.min(6,Math.floor(Math.log2(Math.min(width,height)))+1):0;
      const water=hydrology?this.device.createTexture({size:[width,height],format:'rgba32float',mipLevelCount:waterLevels,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST|GPUTextureUsage.STORAGE_BINDING}):null;
      if(water){
        this.device.queue.writeTexture({texture:water},hydrology,{bytesPerRow:width*16,rowsPerImage:height},[width,height]);
        const encoder=this.device.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(this.waterMipPipeline);
        for(let level=1;level<waterLevels;level++){
          const group=this.device.createBindGroup({layout:this.waterMipPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:water.createView({baseMipLevel:level-1,mipLevelCount:1})},{binding:1,resource:water.createView({baseMipLevel:level,mipLevelCount:1})}]});
          pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil((width>>level)/8),Math.ceil((height>>level)/8));
        }
        pass.end();this.device.queue.submit([encoder.finish()]);
      }
      const tile={elevation,color,climate:climateTexture,hydrology:water,surface:erosion,gradient:gradientTexture,halo,resolution:metresPerSample,params:[width,height,selectedMode,0],origin,lod,heights,materialSeed,materialMeta:climateLayers>=46?Array.from(climateValues.subarray(45*climateWidth*climateHeight,45*climateWidth*climateHeight+8)):null};
      this.shadeTile(tile);
      const bytes=width*height*4+(gradient?gradient.byteLength:0)+(surface?surface.byteLength:0)+(hydrology?Math.ceil(width*height*16*4/3):0)+Math.ceil(innerWidth*innerHeight*16/3)+climateWidth*climateHeight*climateLayers*4;
      // shadeTile already counted any Render3D layers in tile.r3Bytes.
      this.tiles.set(key,Object.assign(tile,{bytes:bytes+(tile.r3Bytes||0),lastUsed:++this.serial}));this.bytes+=bytes;this.uploads++;this.lastUploadSubmitMs=performance.now()-started;
      window.TerrainTiming.record('render.upload_and_shade_submit',started,{key,bytes});
      window.TerrainTiming.queue(this.device.queue,'render.shade');
      this.evict();return this.hasTile(key);
    }
    // Render3D compiles on first use; until then its tiles shade as Render.
    effectiveMode(mode) {
      if(!isRender3d(mode))return mode;
      this.ensureRender3d();
      return this.r3Ready?mode:-3;
    }
    shadeTile(tile) {
      const {elevation,color,climate:climateTexture,halo,resolution:metresPerSample,origin,lod}=tile;
      const [width,height]=tile.params,innerWidth=width-2*halo,innerHeight=height-2*halo;
      const selectedMode=this.effectiveMode(tile.params[2]),render3d=isRender3d(selectedMode);
      this.scratch(width,height);
      const auxFormat=isV2(selectedMode)?'rgba16float':'rgba8unorm';
      const auxLayers=isV2(selectedMode)?6:5;
      if(tile.aux&&(!render3d||tile.auxFormat!==auxFormat||tile.auxLayers!==auxLayers)){
        tile.aux.destroy();tile.r3Buffer.destroy();tile.aux=tile.r3Buffer=null;tile.shadeKey=null;
        tile.group=null;if(tile.bytes!==undefined)tile.bytes-=tile.r3Bytes;this.bytes-=tile.r3Bytes;tile.r3Bytes=0;
      }
      if(render3d&&!tile.aux){
        // V2 climate/cover must not become 0.392 C isotherm steps under zoom.
        // Keep legacy Render3D byte textures; V2 retains half-float climate,
        // rain and cover through its mean-preserving mip chain.
        const levels=Math.min(5,Math.floor(Math.log2(Math.min(width,height)))+1);
        tile.aux=this.device.createTexture({size:[width,height,auxLayers],format:auxFormat,mipLevelCount:levels,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC});tile.auxFormat=auxFormat;tile.auxLayers=auxLayers;
        const uniform=r3.tileUniform(origin,tile.materialSeed);
        if(tile.materialMeta){const m=tile.materialMeta;new Float32Array(uniform,32*16+31*16,4).set([m[0],m[1],m[2]-m[0],m[3]-m[1]]);new Uint32Array(uniform,32*32,4)[1]=(m[4]>.5?1:0)+(m[5]>.5?2:0);}
        tile.r3Buffer=this.device.createBuffer({size:uniform.byteLength,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
        this.device.queue.writeBuffer(tile.r3Buffer,0,uniform);tile.shadeKey=null;
        const bytes=Math.ceil(width*height*auxLayers*(auxFormat==='rgba16float'?8:4)*4/3)+uniform.byteLength;tile.r3Bytes=bytes;if(tile.bytes!==undefined)tile.bytes+=bytes;this.bytes+=bytes;
      }
      // One parameter buffer: each shade writes it, then submits before the next write.
      const params=this.shadeParams??=this.device.createBuffer({size:160,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
      const values=new Float32Array([metresPerSample,halo,innerWidth,innerHeight,selectedMode,Number(!!window.TerrainHydrology?.regionalRiverDisplayEnabled?.()),tile.gradient?1:0,0,...(window.TerrainLighting?.vectors(lod)||[1,.35,1,1,-.5,-.5,.70710678,0]),...origin,0,0,...this.renderVectors(selectedMode)]);
      // Global index of the first sample (halo included), as int bits: f32 loses it below LOD -2.
      new Int32Array(values.buffer,18*4,2).set([Math.round(origin[0]/metresPerSample),Math.round(origin[1]/metresPerSample)]);
      this.device.queue.writeBuffer(params,0,values);
      // Only Render, Soil and Render3D read shore distances.
      const shore=selectedMode===-3||selectedMode===-4||render3d;
      // Bind groups and views persist per tile: reshading after a settings change only records dispatches.
      const v2Aux=render3d&&auxFormat==='rgba16float',bakeLayout=v2Aux?this.v2BakeLayout:this.r3BakeLayout,mipPipeline=v2Aux?this.v2MipPipeline:this.r3MipPipeline;
      const key=this.scratchGeneration+'|'+(render3d?'render3d-'+auxFormat:'shade|'+(this.resGeneration||0));
      if(tile.shadeKey!==key){
        const view=(texture,level,dimension)=>texture.createView({baseMipLevel:level,mipLevelCount:1,...(dimension?{dimension}:{})});
        const entries=[{binding:0,resource:elevation.createView()},{binding:1,resource:this.scratchB.createView()},{binding:2,resource:view(color,0)},{binding:3,resource:{buffer:params}},{binding:4,resource:climateTexture.createView({dimension:'2d-array'})},{binding:5,resource:this.scratchA.createView()},{binding:10,resource:(tile.gradient||this.noGradient).createView()},{binding:9,resource:(tile.surface||this.noSurface).createView()},{binding:8,resource:(tile.hydrology||this.noHydrology).createView()}];
        tile.shadeGroups={
          horizontal:this.device.createBindGroup({layout:this.blurLayout,entries:[{binding:0,resource:elevation.createView()},{binding:1,resource:this.scratchA.createView()}]}),
          // After the vertical blur has consumed scratch A, it holds the shore rows.
          rows:this.device.createBindGroup({layout:this.shoreRowPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:elevation.createView()},{binding:1,resource:this.scratchA.createView()},{binding:2,resource:{buffer:params,size:16}},{binding:8,resource:(tile.hydrology||this.noHydrology).createView()}]}),
          shade:render3d?this.device.createBindGroup({layout:bakeLayout,entries:[...entries,{binding:6,resource:view(tile.aux,0,'2d-array')},{binding:7,resource:{buffer:tile.r3Buffer}}]})
            :this.device.createBindGroup({layout:this.shadeLayout,entries:[...entries,...(rg?this.resourceEntries():[])]}),
          colorMips:Array.from({length:color.mipLevelCount-1},(_,i)=>this.device.createBindGroup({layout:this.colorMipPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:view(color,i)},{binding:1,resource:view(color,i+1)}]})),
          auxMips:render3d?Array.from({length:tile.aux.mipLevelCount-1},(_,i)=>this.device.createBindGroup({layout:mipPipeline.getBindGroupLayout(0),entries:[{binding:0,resource:view(tile.aux,i,'2d-array')},{binding:1,resource:view(tile.aux,i+1,'2d-array')}]})):[],
        };
        tile.shadeKey=key;
      }
      const groups=tile.shadeGroups,encoder=this.device.createCommandEncoder({label:render3d?'terrain Render3D bake once':'terrain shade once'});
      // One compute pass: every dispatch is its own usage scope, so scratch and mip
      // levels may switch between storage and sampling from one dispatch to the next.
      const pass=encoder.beginComputePass();
      const dispatch=(pipeline,group,w,h,layers=1)=>{pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(Math.max(1,w)/8),Math.ceil(Math.max(1,h)/8),layers);};
      dispatch(this.horizontal,groups.horizontal,width,height);
      dispatch(this.vertical,this.verticalGroup,width,height);
      if(shore)dispatch(this.shoreRowPipeline,groups.rows,width,height);
      if(render3d)dispatch(v2Aux?this.v2BakePipeline:this.r3BakePipeline,groups.shade,width,height);else dispatch(this.shade,groups.shade,innerWidth,innerHeight);
      groups.colorMips.forEach((group,i)=>dispatch(this.colorMipPipeline,group,innerWidth>>(i+1),innerHeight>>(i+1)));
      groups.auxMips.forEach((group,i)=>dispatch(mipPipeline,group,width>>(i+1),height>>(i+1),auxLayers));
      pass.end();this.device.queue.submit([encoder.finish()]);tile.shadedMode=selectedMode;tile.v2BakeSeason=isV2(selectedMode)?v2?.season():null;tile.v2RegionalRivers=!!window.TerrainHydrology?.regionalRiverDisplayEnabled?.();
    }
    ensureRender3d() {
      if(this.r3Promise||!r3||!this.available)return;
      const device=this.device,started=performance.now();
      this.r3Promise=(async()=>{
        device.pushErrorScope('validation');
        try{
          const waterDraw=hydrologyShader+riverDisplaySampleShader+(v2?v2.drawShader(r3.shaders.draw,materialShader):r3.shaders.draw)
            .replace('let landCover=clamp(s.x/', 'let landCover=clamp(hydroGround(s.x,texel-.5)/')
            .replace('return vec4<f32>(mix(water,reference,landCover),1.);','return vec4<f32>(hydroBlend(mix(water,reference,landCover),s.x,hydroFiltered(texel-.5,fp/r)),1.);')
            .replace('return vec4<f32>(mix(water,mix(reference,shaded,reveal3d),max(landCover,forestResolvedCover)),1.);','return vec4<f32>(hydroBlend(mix(water,mix(reference,shaded,reveal3d),max(landCover,forestResolvedCover)),s.x,hydroFiltered(texel-.5,fp/r)),1.);');
          const [bake,draw,mip,v2Bake,v2Mip]=await Promise.all([this.compileShader(render3dBakeShader(),'Render3D bake'),this.compileShader(waterDraw,'Render3D quads'),this.compileShader(r3.shaders.auxMip,'Render3D aux mips'),this.compileShader(render3dBakeShader('rgba16float'),'Render v2 precise climate bake'),this.compileShader(r3.shaders.auxMip.replaceAll('rgba8unorm','rgba16float'),'Render v2 precise climate mips')]);
          const compute=GPUShaderStage.COMPUTE,fragment=GPUShaderStage.FRAGMENT;
          const bakeEntries=auxFormat=>[
            {binding:0,visibility:compute,texture:{sampleType:'unfilterable-float'}},
            {binding:1,visibility:compute,texture:{sampleType:'unfilterable-float'}},
            {binding:2,visibility:compute,storageTexture:{access:'write-only',format:'rgba8unorm'}},
            {binding:3,visibility:compute,buffer:{type:'uniform'}},
            {binding:4,visibility:compute,texture:{sampleType:'unfilterable-float',viewDimension:'2d-array'}},
            {binding:10,visibility:compute,texture:{sampleType:'unfilterable-float'}},
            {binding:9,visibility:compute,texture:{}},
            {binding:5,visibility:compute,texture:{sampleType:'unfilterable-float'}},
            {binding:6,visibility:compute,storageTexture:{access:'write-only',format:auxFormat,viewDimension:'2d-array'}},
            {binding:7,visibility:compute,buffer:{type:'uniform'}},
            {binding:8,visibility:compute,texture:{sampleType:'unfilterable-float'}},
          ];
          this.r3BakeLayout=device.createBindGroupLayout({entries:bakeEntries('rgba8unorm')});
          this.v2BakeLayout=device.createBindGroupLayout({entries:bakeEntries('rgba16float')});
          this.r3DrawLayout=device.createBindGroupLayout({entries:[
            {binding:0,visibility:GPUShaderStage.VERTEX|fragment,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:208}},
            {binding:1,visibility:fragment,texture:{}},
            {binding:2,visibility:fragment,sampler:{type:'filtering'}},
            {binding:3,visibility:fragment,texture:{sampleType:'unfilterable-float'}},
            {binding:4,visibility:fragment,texture:{viewDimension:'2d-array'}},
            {binding:5,visibility:fragment,buffer:{type:'uniform'}},
            {binding:9,visibility:fragment,texture:{}},
            {binding:6,visibility:fragment,texture:{viewDimension:'2d-array'}},
            {binding:8,visibility:fragment,texture:{sampleType:'unfilterable-float'}},
            {binding:10,visibility:fragment,texture:{sampleType:'unfilterable-float'}},
            {binding:11,visibility:fragment,texture:{sampleType:'unfilterable-float'}},
          ]});
          [this.r3BakePipeline,this.r3Pipeline,this.r3MipPipeline,this.v2BakePipeline,this.v2MipPipeline]=await Promise.all([
            device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[this.r3BakeLayout]}),compute:{module:bake,entryPoint:'bake'}}),
            device.createRenderPipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[this.r3DrawLayout]}),vertex:{module:draw,entryPoint:'vertex'},fragment:{module:draw,entryPoint:'fragment',targets:[{format:this.format}]},primitive:{topology:'triangle-list'}}),
            device.createComputePipelineAsync({layout:'auto',compute:{module:mip,entryPoint:'main'}}),
            device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[this.v2BakeLayout]}),compute:{module:v2Bake,entryPoint:'bake'}}),
            device.createComputePipelineAsync({layout:'auto',compute:{module:v2Mip,entryPoint:'main'}}),
          ]);
          this.noBrushes=device.createTexture({size:[1,1,1],format:'rg8unorm',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
          device.queue.writeTexture({texture:this.noBrushes},new Uint8Array([0,128]),{bytesPerRow:2},[1,1,1]);
          this.brushes??=this.noBrushes;
        }finally{
          const error=await device.popErrorScope();if(error)throw Error(error.message);
        }
        if(!this.available)return;
        this.r3Ready=true;this.r3CompileMs=performance.now()-started;this.styleRevision++;
        window.TerrainTiming.record('render.render3d_compile',started);
        this.options.onStatus?.(this.status,'render3d');
      })().catch(error=>{this.r3Error=error.message;console.warn('Render3D unavailable; Render remains displayed.',error);});
    }
    // Overhead canopy atlas, fetched only when Render3D Brush is displayed.
    ensureBrushes() {
      if(this.brushPromise||!r3)return;
      this.brushPromise=r3.loadBrushes().then(atlas=>{
        if(!this.available)return;
        const texture=this.device.createTexture({size:[atlas.size,atlas.size,atlas.layers],format:'rg8unorm',mipLevelCount:atlas.levels.length,usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
        atlas.levels.forEach((pixels,level)=>{const size=Math.max(1,atlas.size>>level);this.device.queue.writeTexture({texture,mipLevel:level},pixels,{bytesPerRow:size*2,rowsPerImage:size},[size,size,atlas.layers]);});
        this.brushes=texture;this.brushesReady=true;this.styleRevision++;
        for(const tile of this.tiles.values())if(tile.groupKind==='render3d')tile.group=null;
        this.options.onStatus?.(this.status,'render3d-brushes');
      }).catch(error=>{this.brushError=error.message;console.warn('Render3D brushes unavailable; analytic crowns remain displayed.',error);});
    }
    setBiomeRockSlope(angle) {
      // Compatibility for existing callers; Biomes no longer uses this value.
      if(!Number.isFinite(angle)||angle<1||angle>89)throw Error('Invalid biome rock slope');
      this.biomeRockSlope=angle;
    }
    renderVectors(mode=modeNumber(this.mode)){
      const s={season:.5,variation:1,forest:1,moisture:0,rock_slope:40,snow:1,vegetation_tint:[1,1,1],rock_tint:[1,1,1],snow_color:[.94,.96,.97],...this.renderSettings};
      // V2 has one calendar, including its distant inherited grass/soil guide.
      // This changes display material only; terrain and hydrology cache identity
      // stay independent of the colour bake.
      if(isV2(mode))s.season=v2?.season()??s.season;
      // Render3D forest controls ride in otherwise unused slots: render1.zw, render2.w.
      const [enabled,density,clearings]=r3?.vectors()||[1,1,.45];
      return [s.season,s.variation,s.forest,Math.tan(s.rock_slope*Math.PI/180),s.snow,s.moisture,enabled,density,...s.vegetation_tint,clearings,...s.rock_tint,v2?.season()??.5,...s.snow_color,0];
    }
    setRenderSettings(value){
      const s=window.TerrainRender?.normalize(value)||value;
      const signature=JSON.stringify(s);if(signature===this.renderSignature)return;
      this.renderSignature=signature;this.renderSettings=s;this.styleRevision++;
      for(const tile of this.tiles.values())if(tile.params[2]===-3||tile.params[2]===-4||isRender3d(tile.params[2]))tile.shadedMode=null;
    }
    // Forest density/clearings are baked into Render3D tiles only.
    syncForest(){
      const signature=JSON.stringify(r3?.get()??null);if(signature===this.forestSignature)return;
      this.forestSignature=signature;this.styleRevision++;
      for(const tile of this.tiles.values())if(isRender3d(tile.shadedMode))tile.shadedMode=null;
    }
    // Resource layers: one package per world and mode (fields, zone thresholds, deposits).
    resetResources() {
      const d=this.device,f16=new Uint16Array(4);
      const tex=()=>{const t=d.createTexture({size:[1,1,1],format:'r16float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});d.queue.writeTexture({texture:t},f16,{bytesPerRow:2,rowsPerImage:1},[1,1,1]);return t;};
      const buffer=(size,usage)=>d.createBuffer({size,usage:usage|GPUBufferUsage.COPY_DST});
      this.res={field:tex(),tau:tex(),params:buffer(rg.PARAM_BYTES,GPUBufferUsage.UNIFORM),deposits:buffer(48,GPUBufferUsage.STORAGE),buckets:buffer(16,GPUBufferUsage.STORAGE),header:null,season:.5};
      this.resGeneration=(this.resGeneration||0)+1;
      if(this.tiles)this.markResourceTiles();
    }
    markResourceTiles() {
      // Physical tiles re-shade; native coarse blocks rebind the new package.
      for(const tile of this.tiles.values()){if(tile.native)tile.group=null;else if(rg?.isNumber(tile.params[2]))tile.shadedMode=null;}
    }
    resourceEntries() {
      const r=this.res;
      return [{binding:11,resource:r.field.createView({dimension:'2d-array'})},{binding:12,resource:r.tau.createView({dimension:'2d-array'})},
        {binding:13,resource:{buffer:r.params}},{binding:14,resource:{buffer:r.deposits}},{binding:15,resource:{buffer:r.buckets}}];
    }
    setResources(header,data,{season=.5}={}) {
      if(!rg||!this.available)return false;
      const d=this.device,old=this.res,started=performance.now();
      const upload=(values,[w,h,k])=>{const t=d.createTexture({size:[w,h,k],format:'r16float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});d.queue.writeTexture({texture:t},values,{bytesPerRow:w*2,rowsPerImage:h},[w,h,k]);return t;};
      const storage=(values,min)=>{const b=d.createBuffer({size:Math.max(min,Math.ceil(values.byteLength/4)*4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});d.queue.writeBuffer(b,0,values);return b;};
      const [W,H,K]=data.fieldShape;
      this.res={field:upload(data.field,[W,H,K]),tau:data.tau?upload(data.tau,[W,H,header.thresholds]):old.tau,
        params:old.params,deposits:data.deposits?storage(data.deposits,48):old.deposits,buckets:data.buckets?storage(data.buckets,16):old.buckets,header,season};
      d.queue.writeBuffer(this.res.params,0,rg.pack(header,{season,active:true,marine:header.mode.includes('fish-marine')||header.mode.includes('season-fish')}));
      for(const key of ['field','tau','deposits','buckets'])if(old[key]!==this.res[key]&&old[key]!==this.res.params)setTimeout(()=>old[key].destroy(),2000);
      this.resGeneration++;this.lastResourceUploadMs=performance.now()-started;
      this.markResourceTiles();
      return true;
    }
    setResourceSeason(season) {
      if(!rg||!this.res?.header)return;
      this.res.season=season;
      const h=this.res.header;this.device.queue.writeBuffer(this.res.params,0,rg.pack(h,{season,active:true,marine:h.mode.includes('fish-marine')||h.mode.includes('season-fish')}));
      for(const tile of this.tiles.values())if(!tile.native&&rg.isNumber(tile.params[2]))tile.shadedMode=null;
    }
    setMode(mode) {
      if(this.mode===mode)return;this.mode=mode;this.styleRevision++;
      const selectedMode=modeNumber(mode);
      if(isRender3d(selectedMode))this.ensureRender3d();
      if(selectedMode===-7)this.ensureBrushes();
      for(const tile of this.tiles.values())tile.params[2]=tile.native?coarseMode(selectedMode):selectedMode;
    }
    uploadCoarse(key, roots, {width=160,height=160,halo=16,metresPerSample=7680,climate,climateWidth=41,climateHeight=41,mode='relief',interpolation='monotone',origin=[0,0]}={}) {
      if(!this.available)return false;
      if(!(roots instanceof Float32Array)||roots.length!==width*height||width%8||height%8||halo<8||!(climate instanceof Float32Array)||![5,21,36,46,50,51].includes(climate.length/(climateWidth*climateHeight)))throw Error('Invalid native coarse block');
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
      this.tiles.set(key,{native:true,elevation,climate:climateTexture,climateLayers,albedo:null,bytes,lastUsed:++this.serial,params:[width,height,coarseMode(modeNumber(this.mode??mode)),interpolation==='monotone'?1:0],halo,resolution:metresPerSample,origin,heights:roots,encoding:'signed-sqrt'});
      this.bytes+=bytes;this.uploads++;this.evict();return this.hasTile(key);
    }
    coarseMaterialKey(mode,lod=7) {
      // The canopy mean integrates the current light. Its display bake must
      // change with that light; neural terrain/cache identity stays unchanged.
      return mode+'|'+this.renderSignature+(isV2(mode)?'|'+v2.season()+'|'+this.forestSignature+'|'+JSON.stringify(window.TerrainLighting?.vectors(lod)||[1,.35,1,1,-.5,-.5,.70710678,0]):'');
    }
    bakeCoarse(tile,lod=7) {
      // Once per block, mode and setting; frames then sample it with filtering.
      const [width,height,mode,interpolation]=tile.params;
      const precise=isV2(mode),format=precise?'rgba16float':'rgba8unorm';
      if(!tile.albedo||tile.albedoFormat!==format){
        tile.albedo?.destroy();const previous=tile.albedoBytes||0;
        tile.albedo=this.makeTexture(width*2,height*2,format,GPUTextureUsage.STORAGE_BINDING);
        tile.albedoFormat=format;tile.albedoBytes=width*height*(precise?32:16);
        tile.bytes+=tile.albedoBytes-previous;this.bytes+=tile.albedoBytes-previous;tile.group=null;
      }
      const params=this.device.createBuffer({size:208,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
      const meanLight=precise?window.TerrainLighting?.vectors(lod)||[1,.35,1,1,-.5,-.5,.70710678,0]:[0,0,0,0,0,0,0,0];
      this.device.queue.writeBuffer(params,0,new Float32Array([0,0,0,0,0,0,tile.halo,tile.resolution,0,0,1,1,width,height,mode,interpolation,...meanLight,...tile.origin,0,0,0,0,0,0,...this.renderVectors(mode)]));
      const group=this.device.createBindGroup({layout:precise?this.v2CoarseBakeLayout:this.bakeLayout,entries:[{binding:0,resource:{buffer:params}},{binding:1,resource:tile.elevation.createView()},{binding:2,resource:tile.climate.createView({dimension:'2d-array'})},{binding:3,resource:tile.albedo.createView()}]});
      const encoder=this.device.createCommandEncoder({label:'native coarse materials'}),pass=encoder.beginComputePass();
      pass.setPipeline(precise?this.v2CoarseBakePipeline:this.bakePipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(width/4),Math.ceil(height/4));pass.end();
      this.device.queue.submit([encoder.finish()]);params.destroy();tile.bakedKey=this.coarseMaterialKey(mode,lod);
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
      tile.gradient?.destroy();tile.elevation.destroy();tile.surface?.destroy();tile.hydrology?.destroy();tile.color?.destroy();tile.albedo?.destroy();tile.aux?.destroy();tile.r3Buffer?.destroy();tile.climate.destroy();tile.contourBuffer?.destroy();this.bytes-=tile.bytes;this.tiles.delete(key);
    }
    evict() {
      const overhead=(this.scratchWidth||0)*(this.scratchHeight||0)*16+this.capacity*this.device.limits.minUniformBufferOffsetAlignment;
      while(this.bytes+overhead>this.maxBytes&&this.tiles.size) {
        let key,oldest=Infinity;for(const [candidate,tile]of this.tiles)if(tile.lastUsed<oldest){oldest=tile.lastUsed;key=candidate;}
        this.deleteTile(key);
      }
    }
    invalidateRiverBindings(){for(const tile of this.tiles.values())if(tile.groupKind==='render3d')tile.group=null;}
    clearRiverDisplay(){
      this.riverMask?.destroy();this.riverDepth?.destroy();this.riverSegments?.destroy();this.riverDisplayUniform?.destroy();
      this.riverMask=this.riverDepth=this.riverSegments=this.riverDisplayUniform=null;this.riverSegmentCapacity=0;
      this.riverMaskWidth=this.riverMaskHeight=0;this.riverDisplayKey=null;this.riverPendingKey=null;
      this.riverDisplayStats={ready:false,mode:'inactive',segments:0,bytes:0};
      this.invalidateRiverBindings();window.TerrainHydrology?.acknowledgeRiverDisplay?.(null,[]);
    }
    ensureRiverDisplayPipeline(){
      if(this.riverDisplayPipeline||this.riverDisplayPromise)return;
      this.riverDisplayPromise=(async()=>{
        const module=await this.compileShader(riverDisplayShader,'regional river ribbons');
        this.riverDisplayPipeline=await this.device.createRenderPipelineAsync({layout:'auto',
          vertex:{module,entryPoint:'vertex'},fragment:{module,entryPoint:'fragment',targets:[{format:'rgba16float'}]},
          primitive:{topology:'triangle-list'},depthStencil:{format:'depth24plus',depthWriteEnabled:true,depthCompare:'greater'}});
        if(this.available)this.options.onStatus?.(this.status,'river-display-ready');
      })().catch(error=>{this.riverDisplayError=error.message;this.options.onStatus?.(this.status,'river-display-fallback');});
    }
    prepareRiverDisplay(encoder,width,height,w,h,visible){
      const frame=window.TerrainHydrology?.riverDisplay?.();
      if(!frame){if(this.riverMask)this.clearRiverDisplay();return;}
      const fineRects=visible.filter(r=>{const tile=this.tiles.get(r.key);return !tile.native&&isV2(tile.shadedMode);})
        .map(r=>({x:r.x,y:r.y,width:r.width,height:r.height}));
      if(!fineRects.length){window.TerrainHydrology?.acknowledgeRiverDisplay?.(null,[]);return;}
      if(frame.ready&&frame.segments.length)this.ensureRiverDisplayPipeline();
      const active=frame.ready&&(!frame.segments.length||!!this.riverDisplayPipeline);
      const count=active?frame.segments.length/8:0;
      const mw=count?w:1,mh=count?h:1,mode=active?'regional-mask':'Render3D-overlay-fallback';
      const key=frame.key+'|'+w+'x'+h+'|'+mode;
      window.TerrainHydrology?.acknowledgeRiverDisplay?.(active?frame.key:null,active?fineRects:[],frame.key);
      if(key===this.riverDisplayKey&&this.riverMask)return;
      if(!this.riverMask||mw!==this.riverMaskWidth||mh!==this.riverMaskHeight){
        this.riverMask?.destroy();this.riverDepth?.destroy();this.riverDepth=null;
        this.riverMask=this.device.createTexture({size:[mw,mh],format:'rgba16float',
          usage:GPUTextureUsage.RENDER_ATTACHMENT|GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_SRC});
        if(count)this.riverDepth=this.device.createTexture({size:[mw,mh],format:'depth24plus',usage:GPUTextureUsage.RENDER_ATTACHMENT});
        this.riverMaskWidth=mw;this.riverMaskHeight=mh;this.invalidateRiverBindings();
      }
      if(count&&!this.riverDepth)this.riverDepth=this.device.createTexture({size:[mw,mh],format:'depth24plus',usage:GPUTextureUsage.RENDER_ATTACHMENT});
      let group;
      if(count){
        const bytes=frame.segments.byteLength;
        if(bytes>(this.riverSegmentCapacity||0)){this.riverSegments?.destroy();this.riverSegmentCapacity=Math.max(256,2**Math.ceil(Math.log2(bytes)));
          this.riverSegments=this.device.createBuffer({size:this.riverSegmentCapacity,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST});}
        this.device.queue.writeBuffer(this.riverSegments,0,frame.segments);
        this.riverDisplayUniform??=this.device.createBuffer({size:16,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
        this.device.queue.writeBuffer(this.riverDisplayUniform,0,new Float32Array([width,height,w/width,h/height]));
        group=this.device.createBindGroup({layout:this.riverDisplayPipeline.getBindGroupLayout(0),entries:[
          {binding:0,resource:{buffer:this.riverDisplayUniform}},{binding:1,resource:{buffer:this.riverSegments}}]});
      }
      const pass=encoder.beginRenderPass({colorAttachments:[{view:this.riverMask.createView(),loadOp:'clear',storeOp:'store',clearValue:{r:0,g:0,b:0,a:active?2:1}}],
        ...(count?{depthStencilAttachment:{view:this.riverDepth.createView(),depthLoadOp:'clear',depthStoreOp:'discard',depthClearValue:0}}:{})});
      if(count){pass.setPipeline(this.riverDisplayPipeline);pass.setBindGroup(0,group);pass.draw(6,count);}pass.end();
      this.riverDisplayKey=key;this.riverPendingKey=key;
      this.riverDisplayStats={ready:false,phase:'submitted',mode,frameKey:frame.key,networkRevision:frame.networkRevision,
        hydroId:frame.hydroId,month:frame.month,segments:count,dimensions:[mw,mh],logicalViewport:[width,height],format:'rgba16float+depth24plus',
        levelEncoding:'f16-high+f16-residual; signed metres',bytes:mw*mh*(count?12:8)+this.riverSegmentCapacity+(this.riverDisplayUniform?16:0),
        empty:active&&count===0,error:this.riverDisplayError||null,authority:frame.authority};
    }
    finishRiverDisplay(){
      const key=this.riverPendingKey;if(!key)return;this.riverPendingKey=null;
      this.device.queue.onSubmittedWorkDone().then(()=>{
        if(this.riverDisplayKey===key){this.riverDisplayStats.ready=true;this.riverDisplayStats.phase='ready';this.options.onStatus?.(this.status,'river-display-uploaded');}
      },error=>{if(this.riverDisplayKey===key){this.riverDisplayStats.ready=false;this.riverDisplayStats.error=error.message;}});
    }
    clear() { for(const key of this.tiles.keys())this.deleteTile(key);this.clearRiverDisplay(); }
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
      this.syncForest();
      const regionalRivers=!!window.TerrainHydrology?.regionalRiverDisplayEnabled?.();
      for(const tile of this.tiles.values())if(!tile.native&&isV2(tile.params[2])&&tile.v2RegionalRivers!==regionalRivers)tile.shadedMode=null;
      const v2date=v2?.season();if(this.v2date!==v2date){
        this.v2date=v2date;this.styleRevision++;
        for(const tile of this.tiles.values())if(!tile.native&&isV2(tile.params[2])&&tile.v2BakeSeason!==v2date)tile.shadedMode=null;
      }
      for(const r of visible){
        const tile=this.tiles.get(r.key);
        if(!tile.native){if(tile.shadedMode!==this.effectiveMode(tile.params[2]))this.shadeTile(tile);}
        else if((tile.params[2]===-3||tile.params[2]===-4||tile.params[2]===-8)&&tile.climateLayers>=46&&tile.bakedKey!==this.coarseMaterialKey(tile.params[2],r.lod))this.bakeCoarse(tile,r.lod);
      }
      const contourSettings=city?.get();
      if(contourSettings?.enabled)for(const r of visible){const tile=this.tiles.get(r.key);if(tile.params[2]!==-5)this.prepareContours(tile,contourSettings.interval,r);}
      this.evict();visible=visible.filter(r=>this.tiles.has(r.key));
      // Brush tiles draw analytic crowns until the atlas arrives.
      const fieldMode=tile=>tile.shadedMode===-7&&!this.brushesReady?-6:tile.params[2];
      visible.forEach((r,i)=>{const tile=this.tiles.get(r.key);this.uniformData.set([r.x,r.y,r.width,r.height,width,height,tile.halo||0,tile.resolution||0,...(r.uv||[0,0,1,1]),tile.params[0],tile.params[1],fieldMode(tile),tile.params[3],...(window.TerrainLighting?.vectors(r.lod)||[1,.35,1,1,-.5,-.5,.70710678,0]),...(tile.origin||[0,0]),tile.gradient?1:0,0,...(city?.vectors()||[0,100,0,0]),...this.renderVectors(tile.params[2])],i*stride/4)});
      if(visible.length)this.device.queue.writeBuffer(this.uniform,0,this.uniformData,0,visible.length*stride/4);
      const encoder=this.device.createCommandEncoder();
      this.prepareRiverDisplay(encoder,width,height,w,h,visible);
      const pass=encoder.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:'clear',storeOp:'store'}]});
      let current=null;
      visible.forEach((r,i)=>{
        const tile=this.tiles.get(r.key);tile.lastUsed=++this.serial;
        const kind=tile.native?'coarse':isRender3d(tile.shadedMode)?'render3d':'quad';
        if(tile.groupKind!==kind){tile.group=null;tile.groupKind=kind;}
        const pipeline=kind==='coarse'?this.coarsePipeline:kind==='render3d'?this.r3Pipeline:this.pipeline;
        if(pipeline!==current){pass.setPipeline(pipeline);current=pipeline;}
        tile.group??=this.device.createBindGroup(kind==='coarse'?{layout:this.coarseLayout,entries:[{binding:0,resource:{buffer:this.uniform,size:208}},{binding:1,resource:tile.elevation.createView()},{binding:2,resource:tile.climate.createView({dimension:'2d-array'})},{binding:3,resource:(tile.albedo||this.noAlbedo).createView()},{binding:4,resource:this.sampler},...(rg?this.resourceEntries():[])]}
          :kind==='render3d'?{layout:this.r3DrawLayout,entries:[{binding:0,resource:{buffer:this.uniform,size:208}},{binding:1,resource:tile.color.createView()},{binding:2,resource:this.sampler},{binding:3,resource:tile.elevation.createView()},{binding:4,resource:tile.aux.createView({dimension:'2d-array'})},{binding:5,resource:{buffer:tile.r3Buffer}},{binding:6,resource:this.brushes.createView({dimension:'2d-array'})},{binding:8,resource:(tile.hydrology||this.noHydrology).createView()},{binding:9,resource:(tile.surface||this.noSurface).createView()},{binding:10,resource:(tile.gradient||this.noGradient).createView()},{binding:11,resource:(this.riverMask||this.noRiverDisplay).createView()}]}
          :{layout:this.renderLayout,entries:[{binding:0,resource:{buffer:this.uniform,size:208}},{binding:1,resource:tile.color.createView()},{binding:2,resource:this.sampler},{binding:3,resource:tile.elevation.createView()},{binding:8,resource:(tile.hydrology||this.noHydrology).createView()}]});
        pass.setBindGroup(0,tile.group,[i*stride]);pass.draw(6);
        if(contourSettings?.enabled&&tile.contourCount&&tile.params[2]!==-5){
          pass.setPipeline(this.contourPipeline);current=this.contourPipeline;tile.contourGroup??=this.device.createBindGroup({layout:this.contourLayout,entries:[{binding:0,resource:{buffer:this.uniform,size:208}}]});
          pass.setBindGroup(0,tile.contourGroup,[i*stride]);pass.setVertexBuffer(0,tile.contourBuffer);pass.draw(6,tile.contourCount);
        }
      });
      pass.end();this.device.queue.submit([encoder.finish()]);this.finishRiverDisplay();this.lastDrawnRects=visible;this.frames++;this.canvas.style.visibility='visible';
      window.TerrainTiming.record('render.draw_submit',started,{quads:visible.length});
      window.TerrainTiming.queue(this.device.queue,'render.draw');return true;
    }
    getStats() { const scratchBytes=(this.scratchWidth||0)*(this.scratchHeight||0)*16,uniformBytes=this.capacity*(this.device?.limits.minUniformBufferOffsetAlignment||0);return {status:this.status,tiles:this.tiles.size,bytes:this.bytes,scratchBytes,uniformBytes,riverDisplay:this.riverDisplayStats||{ready:false,mode:'inactive'},totalBytes:this.bytes+scratchBytes+uniformBytes+(this.riverDisplayStats?.bytes||0),maxBytes:this.maxBytes,frames:this.frames,uploads:this.uploads,styleRevision:this.styleRevision,contourSegments:[...this.tiles.values()].reduce((n,t)=>n+(t.contourCount||0),0),lastUploadSubmitMs:this.lastUploadSubmitMs||0,render3d:{ready:!!this.r3Ready,error:this.r3Error||null,compileMs:this.r3CompileMs||0,brushes:!!this.brushesReady,brushError:this.brushError||null,tiles:[...this.tiles.values()].filter(t=>isRender3d(t.shadedMode)).length}}; }
    dispose() {
      this.available=false;this.clear();this.noGradient?.destroy();this.noHydrology?.destroy();this.noRiverDisplay?.destroy();this.noSurface?.destroy();this.noAlbedo?.destroy();this.shadeParams?.destroy();if(this.brushes!==this.noBrushes)this.brushes?.destroy();this.noBrushes?.destroy();this.scratchA?.destroy();this.scratchB?.destroy();this.uniform?.destroy();this.context?.unconfigure();this.notify('disposed');this.device?.destroy();
    }
  }
  window.createTerrainRenderer=async(canvas,options={})=>{
    const renderer=new TerrainGpuRenderer(canvas,options);
    try{return await renderer.initialize();}catch(error){renderer.dispose();options.onStatus?.('fallback',error.message);return null;}
  };
})();

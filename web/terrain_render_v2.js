/* Render v2. Appearance/microhabitat model, not new terrain or weather.
 * The environmental equations are mirrored in backend/terrain_render_v2.py
 * and Magna Urbis terrainRenderV2.ts. World metres; temperatures Celsius;
 * annual rain mm; cover fractions. See docs/design/render-v2-implementation.fr.md. */
(root=>{
  'use strict';
  const MODES={'render-v2':-8,'render-v2-substrate':-9,'render-v2-moisture':-10,'render-v2-vegetation':-11,'render-v2-snow':-12};
  const isStyle=s=>Object.hasOwn(MODES,s)||s==='Render_v2';
  let season=.5,changed=()=>{},sync=()=>{};
  const normalize=x=>Number.isFinite(Number(x))?((Number(x)%1)+1)%1:.5;
  if(typeof location!=='undefined') {const x=new URLSearchParams(location.search).get('v2season');if(x!==null)season=normalize(x);}
  function mount(container,onChange,onDiagnostic=()=>{}){
    changed=onChange;const field=document.createElement('fieldset');field.id='renderV2Settings';field.hidden=true;
    field.innerHTML='<legend>Render v2 · satellite landscape</legend><label>Saison <input id="renderV2Season" type="range" min="0" max="1" step="0.0025" aria-label="Render v2 season"><output id="renderV2Date"></output></label><label>Diagnostic <select id="renderV2Diagnostic" aria-label="Render v2 diagnostic"><option value="render-v2">None · satellite landscape</option><option value="render-v2-substrate">Substrate</option><option value="render-v2-moisture">Local moisture</option><option value="render-v2-vegetation">Vegetation</option><option value="render-v2-snow">Snow and ice</option></select></label><small>One annual calendar for foliage, ground, snow and ice. Appearance model; no terrain regeneration. <a href="/docs/design/render-v2-implementation.fr.md" target="_blank">Method and limitations</a></small>';
    container.prepend(field);const input=field.querySelector('input'),output=field.querySelector('output'),diagnostic=field.querySelector('select');
    sync=()=>{input.value=season;output.textContent=new Intl.DateTimeFormat('en',{month:'long'}).format(new Date(Date.UTC(2024,0,1+Math.floor(season*365))));};
    input.oninput=()=>{season=normalize(input.value);sync();changed();};diagnostic.onchange=()=>onDiagnostic(diagnostic.value);sync();return {show:(yes,mode='render-v2')=>{field.hidden=!yes;diagnostic.value=isStyle(mode)?mode:'render-v2';}};
  }
  // Fixed-amplitude bands. Filtering removes variance; it never amplifies a
  // last remaining octave, and ecological identity is independent of the view.
  const environment=`
    struct V2Env { temperature:f32,potential:f32,moisture:f32,exposure:f32,mineral:f32,sand:f32,clay:f32,organic:f32,rock:f32,grass:f32,woody:f32,height:f32,needle:f32,leaf:f32,snow:f32,ice:f32,tone:f32,glacier:f32 }
    // Appearance policy for natural woodland: large disturbance openings are
    // rarer in humid forest. Drought, bog, rocks and treeline retain their own
    // physical openings; this is not an empirical universal gap-rate model.
    fn v2ClearingTarget(control:f32,humid:f32,trop:f32)->f32{
      return control*mix(.4,.08,sstep(.8,1.8,humid))*(1.-.85*trop);
    }
    // A 100--400 m disturbance thins a stand, never clears its whole cell.
    // Decode the inherited quantile's target; a symmetric zero-bias score
    // keeps that mean before the .55 cap. Small gaps remain in crown candidates.
    // Physical bog/rock/treeline eligibility is evaluated independently.
    fn v2ClearingFraction(score:f32,threshold:f32)->f32{
      let meanShare=1./(1.+exp(clamp(threshold*1.702/1.15,-20.,20.)));
      return min(.55,2.*meanShare/(1.+exp(clamp(-score/1.15,-20.,20.))));
    }
    // Stand age/productivity changes effective optical density, rather than
    // multiplying an already saturated cover fraction. This is the same gap
    // law as the existing forest-density control, not a measured LAI model.
    // Physical eligibility, height, tone and candidate positions stay separate.
    fn v2StandCoverage(cover:f32,stand:f32,disturbed:f32,trop:f32)->f32{
      let mosaic=.22*(1.-.75*trop);
      let density=exp(mosaic*stand+.05*stand*(1.-cover))*(1.-.3*disturbed*(1.-.7*trop));
      return 1.-pow(max(1.-cover,.001),density);
    }
    fn v2Pattern(point:vec3<f32>,scale:f32,fp:f32,salt:u32)->f32{
      return surfaceNoise(point/scale,salt)*(1.-sstep(scale*.12,scale*.3,fp));
    }
    // Climate/slope snow fraction, shared with the distant material. Stand
    // noise and river-bank moisture must not draw patches in a snow mantle.
    fn v2Snow(temperature:f32,potential:f32,slope:f32,valley:f32)->f32{
      return clamp(sstep(2.,-5.,temperature-.8*valley)*sstep(.12,.7,potential)*(1.-sstep(.45,1.25,slope)),0.,1.);
    }
    // An annually cold site retains an existing snowpack through its warmest
    // season. This is the inherited perennial temperature interval, with the
    // same supply/slope retention as V2 seasonal snow. No stand noise, biome
    // label or coastal glacier channel can create this annual floor.
    fn v2PerennialSnow(warmest:f32,potential:f32,slope:f32)->f32{
      return clamp(sstep(2.5,-2.5,warmest)*sstep(.12,.7,potential)*(1.-sstep(.45,1.25,slope)),0.,1.);
    }
    fn v2EvaluateDomains(point:vec3<f32>,landPoint:vec3<f32>,h:f32,gradient:vec2<f32>,tpi:f32,ts:f32,tw:f32,ps:f32,pw:f32,sand:f32,bank:f32,date:f32,footprint:f32,salt:u32)->V2Env{
      var e:V2Env;let provinceOffset=vec3<f32>(f32(salt&255u),f32((salt>>8u)&255u),f32((salt>>16u)&255u))*.37;let phase=.5-.5*cos(6.283185307*date);let slope=length(gradient);
      let province=.72*surfaceNoise(point/180000.+provinceOffset,salt+601u)*(1.-sstep(21600.,54000.,footprint))+.28*surfaceNoise(point/690000.+provinceOffset,salt+607u)*(1.-sstep(82800.,207000.,footprint));
      let mosaic=.72*surfaceNoise(landPoint/2400.,salt+613u)*(1.-sstep(288.,720.,footprint))+.28*surfaceNoise(landPoint/670.,salt+617u)*(1.-sstep(80.4,201.,footprint));
      let valley=sstep(0.,-.10,tpi);let ridge=sstep(0.,.10,tpi);
      e.exposure=clamp(.5+.28*gradient.y/max(.2,slope)+.22*ridge-.22*valley,0.,1.);
      let thermal=.65*(e.exposure-.5)+.35*mosaic;
      e.temperature=mix(tw,ts,phase)+thermal;
      let annual=(ts+tw)*.5;let threshold=max(120.,20.*annual+280.);
      e.potential=clamp((ps+pw)/threshold*exp(.12*mosaic)*(1.+.45*valley-.14*ridge)+bank*.8,0.,5.);
      let rain=mix(pw,ps,phase)*2./threshold;
      e.moisture=clamp((.62*e.potential+.38*rain)*(1.-.18*sstep(16.,34.,e.temperature))+.18*valley+bank*.6,0.,4.);
      e.mineral=clamp(.5+.42*province,0.,1.);
      e.sand=clamp(sand*(1.-.65*sstep(.16,.7,slope))+.13*valley*(1.-sstep(.5,1.4,e.potential)),0.,1.);
      e.clay=clamp(.18+.25*valley+.16*(1.-e.sand)-.12*slope,0.,1.-e.sand);
      e.organic=min(1.-e.sand-e.clay,clamp(.18*sstep(.6,2.,e.potential)*(1.-sstep(.12,.7,slope))+.13*valley,0.,.6));
      e.rock=clamp(sstep(.20,.85,slope)*(.65+.35*sstep(-.5,.9,mosaic))+.12*ridge*(1.-sstep(.5,1.5,e.potential)),0.,1.);
      let grow=max(ts,tw);let wetTropic=sstep(16.,23.,min(ts,tw))*sstep(1.5,3.,e.potential);
      let thermalGate=sstep(5.,9.,grow+.7*mosaic+.6*valley);
      let stands=clamp(1.+(.20-.15*wetTropic)*mosaic-.17*sstep(.7,1.4,mosaic)*(1.-wetTropic),.55,1.15);
      e.woody=clamp(sstep(.38,1.65,e.potential)*(.72+.25*wetTropic)*thermalGate*stands*(1.-e.rock)+bank*.5*thermalGate,0.,.985);
      e.height=(2.+30.*sstep(.5,3.,e.potential)*sstep(5.,22.,grow))*(1.+.15*mosaic)*thermalGate;
      e.needle=clamp(1.-sstep(9.,19.,grow)+.25*sstep(.2,.8,province)-wetTropic,0.,1.);
      let deciduous=sstep(-3.,14.,e.temperature);let dryLeaf=sstep(.28,.9,e.moisture);
      e.leaf=mix(.12+.88*deciduous,dryLeaf,sstep(16.,23.,min(ts,tw)));
      e.leaf=mix(e.leaf,1.,e.needle);
      // Seasonal snow cannot melt the perennial annual floor. Both use only
      // climate and physical retention, never the view-filtered stand mosaic.
      let snowSupply=clamp((ps+pw)/threshold,0.,5.);let snowExposure=.65*(e.exposure-.5);
      e.snow=max(v2Snow(mix(tw,ts,phase)+snowExposure,snowSupply,slope,valley),v2PerennialSnow(grow+snowExposure,snowSupply,slope));
      e.ice=sstep(-1.8,-8.,e.temperature);
      e.grass=clamp(sstep(.12,.8,e.potential)*sstep(0.,9.,grow)*(1.-e.rock)*(1.-e.snow),0.,1.);
      e.tone=clamp(.5+.13*mosaic-.06*valley,0.,1.);e.glacier=0.;return e;
    }
    fn v2Evaluate(point:vec3<f32>,h:f32,gradient:vec2<f32>,tpi:f32,ts:f32,tw:f32,ps:f32,pw:f32,sand:f32,bank:f32,date:f32,footprint:f32,salt:u32)->V2Env{
      return v2EvaluateDomains(point,point,h,gradient,tpi,ts,tw,ps,pw,sand,bank,date,footprint,salt);
    }
    // Artistic contrast reveal is separate from ecological world properties.
    // The province keeps its position/identity; only its displayed contrast fades.
    fn v2Appearance(e:V2Env,fp:f32)->V2Env{
      var shown:V2Env=e;
      shown.mineral=mix(.5,e.mineral,1.-sstep(600.,2400.,fp));
      shown.tone=mix(.5,e.tone,1.-sstep(120.,500.,fp));
      return shown;
    }
    fn v2Mineral(e:V2Env)->vec3<f32>{
      let dark=vec3<f32>(.19,.19,.175);let brown=vec3<f32>(.39,.335,.27);let cream=vec3<f32>(.68,.655,.57);let red=vec3<f32>(.49,.37,.285);
      return mix(mix(dark,brown,sstep(.08,.35,e.mineral)),mix(cream,red,sstep(.68,.94,e.mineral)),sstep(.34,.66,e.mineral));
    }
    // Enrich the retained material mean without replacing its palette on zoom.
    // Applied before sediment/snow and below canopy: at most 14% on bare rock,
    // less than 5% on mineral soil, none on fully herbaceous/organic ground.
    fn v2GroundTint(base:vec3<f32>,e:V2Env)->vec3<f32>{
      var neutral:V2Env=e;neutral.mineral=.5;
      let mineralRatio=clamp(v2Mineral(e)/max(v2Mineral(neutral),vec3<f32>(.08)),vec3<f32>(.6),vec3<f32>(1.4));
      let mineralWeight=.35*(e.rock+(1.-e.rock)*(1.-e.grass)*(1.-e.organic)*.35)*(1.-e.snow);
      return base*mix(vec3<f32>(1.),mineralRatio,mineralWeight);
    }
    // Smooth display shoulder for the hybrid legacy/BRDF lighting path.
    // Keep dark/mid colours unchanged and preserve RGB ratios in highlights.
    // This is a small display approximation, not the full ACES transform.
    fn v2DisplayShoulder(rgb:vec3<f32>)->vec3<f32>{
      let peak=max(rgb.r,max(rgb.g,rgb.b));
      if(peak<=.8){return rgb;}
      let mapped=.8+.2*(1.-exp(-(peak-.8)/.2));
      return rgb*(mapped/peak);
    }
    fn v2Open(e:V2Env)->vec3<f32>{
      let mineral=v2Mineral(e);let sand=mix(vec3<f32>(.73,.64,.46),mineral,.24);
      var soil=mix(mineral*.8,vec3<f32>(.24,.21,.15),e.organic*1.2);
      soil=mix(soil,sand,e.sand);soil*=1.-.20*sstep(.5,2.,e.moisture);
      let green=sstep(.25,1.3,e.moisture)*sstep(1.,13.,e.temperature);
      let grass=mix(vec3<f32>(.53,.485,.335),vec3<f32>(.265,.355,.175),green);
      var color=mix(soil,grass,e.grass*.88);color=mix(color,mineral,e.rock);
      return mix(color,vec3<f32>(.88,.91,.91),e.snow);
    }
    fn v2Canopy(e:V2Env)->vec3<f32>{
      let evergreen=mix(vec3<f32>(.16,.235,.14),vec3<f32>(.12,.205,.15),e.needle);
      let senescent=mix(vec3<f32>(.34,.29,.215),vec3<f32>(.43,.335,.205),sstep(4.,12.,e.temperature));
      let c=mix(senescent,evergreen,sstep(.3,.9,e.leaf))*(.85+.3*e.tone);
      return mix(c,vec3<f32>(.72,.79,.77),e.snow*.62);
    }
    fn v2Color(e:V2Env,channel:f32)->vec3<f32>{
      if(channel==1.){return mix(mix(v2Mineral(e),vec3<f32>(.80,.70,.49),e.sand),vec3<f32>(.22,.18,.13),e.organic);}
      if(channel==2.){return mix(vec3<f32>(.66,.37,.14),vec3<f32>(.08,.40,.52),sstep(0.,2.4,e.moisture));}
      if(channel==3.){return vec3<f32>(e.grass*.55,e.woody*e.leaf,.16+e.needle*.6);}
      if(channel==4.){return mix(vec3<f32>(.13,.20,.22),vec3<f32>(.90,.96,1.),e.snow);}
      return mix(v2Open(e),v2Canopy(e),e.woody*e.leaf);
    }
    fn v2Freeze(kind:f32,e:V2Env)->f32{return select(select(e.ice,sstep(-5.,-13.,e.temperature),kind>2.5),sstep(-2.,-9.,e.temperature),kind<1.5);}
    // Area integration of the resolved two-scale pack, rather than a uniform
    // opacity on every plate. Dense refrozen pack has very little open water.
    fn v2PackMean(concentration:f32)->f32{
      return mix(concentration*(.55+.50*concentration),.993,sstep(.68,.90,concentration));
    }
    // A jittered spherical 3D cellular section: no latitude lattice or tile seed.
    // edge = F2-F1; radius/id/light identify the nearest irregular fragment.
    fn v2CellCore(point:vec3<f32>,scale:f32,salt:u32,bisector:bool)->vec4<f32>{
      let q=point/scale;let cell=vec3<i32>(floor(q));let local=q-floor(q);
      var first=10.;var second=10.;var identity=0.;var face=0.;
      var firstDelta=vec3<f32>(0.);var secondDelta=vec3<f32>(0.);
      for(var z:i32=-1;z<=1;z++){for(var y:i32=-1;y<=1;y++){for(var x:i32=-1;x<=1;x++){
        let offset=vec3<i32>(x,y,z);let hash=surfaceHash(cell+offset,salt);
        let jitter=vec3<f32>(f32(hash&1023u),f32((hash>>10u)&1023u),f32((hash>>20u)&1023u))/1023.;
        let delta=vec3<f32>(offset)+.12+.76*jitter-local;let distance=dot(delta,delta);
        if(distance<first){if(bisector){secondDelta=firstDelta;firstDelta=delta;}second=first;first=distance;identity=f32(hash&65535u)/65535.;face=dot(delta,vec3<f32>(.65,-.30,.70));}
        else if(distance<second){second=distance;if(bisector){secondDelta=delta;}}
      }}}
      var edge=sqrt(second)-sqrt(first);
      if(bisector){edge=(second-first)/max(2.*length(secondDelta-firstDelta),.00001);}
      return vec4<f32>(edge,sqrt(first),identity,face);
    }
    fn v2Cell(point:vec3<f32>,scale:f32,salt:u32)->vec4<f32>{
      return v2CellCore(point,scale,salt,false);
    }
    // Coverage, ice brightness, cyan submerged edge, cast shadow on water.
    // Unresolved fragments converge to area coverage rather than vanish.
    fn v2Ice(depth:f32,kind:f32,e:V2Env,point:vec3<f32>,fp:f32,salt:u32)->vec4<f32>{
      let freeze=v2Freeze(kind,e);let shore=sstep(900.,15.,depth);
      let province=v2Pattern(point,7200.,fp,salt+631u)*(1.-sstep(100.,500.,fp));
      let concentration=clamp(freeze+.12*shore*freeze*(1.-freeze)+.045*province*freeze*(1.-freeze)*4.,0.,.995);
      var pack=v2PackMean(concentration);var brightness=.88;var rim=0.;var shadow=0.;
      if(concentration>.002&&fp<450.){
        let warpX=v2Pattern(point,1800.,fp,salt+633u);
        let warpY=v2Pattern(point,2100.,fp,salt+634u);
        let warpZ=v2Pattern(point,1600.,fp,salt+635u);
        let displacement=vec3<f32>(warpX,warpY,warpZ);
        let packKeep=1.-sstep(140.,450.,fp);
        var resolved=0.;var localLight=.88;var localRim=0.;
        if(concentration<.90){
          let warped=point+displacement*215.;
          let plate=v2Cell(warped,2600.,salt+637u);
          let shard=v2Cell(warped,470.,salt+638u);
          let edge=plate.x+.012*v2Pattern(point,105.,fp,salt+639u);
          let aa=max(.0008,fp/2600.*.75);
          let lead=.0025+.020*(1.-concentration);
          let large=sstep(lead-aa,lead+aa,edge);
          let split=select(0.,1.,fract(plate.z*7.31)>.31);
          let smallLead=.007+.09*(1.-concentration);
          let smallAA=max(.002,fp/470.*.8);
          let shards=sstep(smallLead-smallAA,smallLead+smallAA,shard.x);
          let smallKeep=1.-sstep(45.,135.,fp);
          let subdivided=large*(1.-split*(1.-shards)*smallKeep);
          let identity=mix(plate.z,fract(shard.z*3.17),split*.78);
          let selection=select(0.,1.,identity>=1.-concentration);
          resolved=subdivided*selection;
          let ridge=(1.-sstep(lead+.006+aa,lead+.025+aa,edge))*large;
          localLight=clamp(.84+.08*plate.z+.07*ridge-.065*plate.w+.02*v2Pattern(point,38.,fp,salt+643u),.65,.99);
          localRim=(1.-sstep(lead+.004+aa,lead+.013+aa,edge))*resolved*.3;
        }
        // Compact pack retains rigid adjoining plates and thin leads. The
        // marginal field never shreds the dense plates into liquid swirls.
        if(concentration>.68){
          // Bend the shared cell domain, not its edge distance: both sides of
          // a lead move together, so irregular contours retain thin channels.
          let jitter=vec3<f32>(v2Pattern(point,240.,fp,salt+639u),v2Pattern(point,240.,fp,salt+640u),v2Pattern(point,240.,fp,salt+641u))*40.+vec3<f32>(v2Pattern(point,70.,fp,salt+669u),v2Pattern(point,70.,fp,salt+670u),v2Pattern(point,70.,fp,salt+671u))*7.;
          let compact=v2CellCore(point+displacement*65.+jitter,1200.,salt+637u,kind<1.5);
          let edge=compact.x;
          let aa=max(.001,fp/1200.*.8);
          var lead=(.004+.012*(1.-concentration))*(.72+.44*sstep(-.6,.6,warpY));
          var floe=sstep(lead-aa,lead+aa,edge);
          if(kind<1.5){
            // Metric distance keeps a narrow physical lead at vertices. Area
            // filtering retains its width instead of painting an opaque wire.
            lead=(.0015+.004*(1.-concentration))*(.72+.44*sstep(-.6,.6,warpY));
            let pixelWidth=max(fp/1200.,.0001);
            let gap=clamp((lead+pixelWidth*.5-edge)/pixelWidth,0.,1.)-clamp((-lead+pixelWidth*.5-edge)/pixelWidth,0.,1.);
            floe=1.-gap;
          }
          // Some cracks have closed and refrozen; they are gray young ice,
          // not a dark diagram connecting every floe. No extra cellular pass.
          let opening=sstep(-.35+.50*freeze,.05+.50*freeze,warpX+.28*warpZ);
          let openLead=opening*select(.38,1.,kind<1.5);
          let compactCover=1.-(1.-floe)*openLead;
          let ridge=(1.-sstep(lead+.009+aa,lead+.035+aa,edge))*floe;
          let dense=sstep(.68,.90,concentration);
          resolved=mix(resolved,compactCover,dense);
          let youngIce=(1.-floe)*(1.-openLead);
          var compactLight=.88;var compactRim=0.;
          if(kind<1.5){
            let snowGrain=.005*v2Pattern(point,350.,fp,salt+665u)+.014*v2Pattern(point,90.,fp,salt+667u);
            let ridgeKeep=min(1.,(lead+.010)*1200./max(fp,1.));
            let pressure=(1.-sstep(lead+aa,lead+.010+aa,edge))*floe*ridgeKeep;
            compactLight=clamp(.88+.002*(compact.z-.5)-.002*compact.w+.028*pressure*clamp(-compact.w*2.,-1.,1.)+snowGrain-.035*youngIce,.80,.93);
            compactRim=(1.-sstep(lead+.002+aa,lead+.008+aa,edge))*floe*openLead*.06*ridgeKeep;
          }else{
            let snowGrain=.012*v2Pattern(point,350.,fp,salt+665u)+.006*v2Pattern(point,110.,fp,salt+667u);
            compactLight=clamp(.88+.008*(compact.z-.5)+.010*ridge-.006*compact.w+snowGrain-.12*youngIce,.74,.92);
            compactRim=(1.-sstep(lead+.006+aa,lead+.024+aa,edge))*floe*openLead*.25;
          }
          localLight=mix(localLight,compactLight,dense);
          localRim=mix(localRim,compactRim,dense);
        }
        pack=mix(pack,resolved,packKeep);
        brightness=mix(.88,localLight,packKeep);
        rim=localRim*packKeep;
      }
      let source=e.glacier*(.30+.70*sstep(1800.,30.,depth))*select(0.,1.,kind<1.5);
      var berg=0.;var bergLight=.95;var bergRim=0.;
      if(source>.001){
        let grouping=clamp(.65+.23*v2Pattern(point,2800.,fp,salt+647u),.15,1.);
        let density=source*grouping;
        let meanBerg=.040*density*(1.-pack*.65);
        berg=meanBerg;
        if(fp<290.){
          let bend=v2Pattern(point,380.,fp,salt+651u);
          let piece=v2Cell(point+vec3<f32>(bend*70.,bend*35.,-bend*55.),850.,salt+653u);
          let selected=sstep(.94-.42*density,.99-.42*density,piece.z)*sstep(.001,.04,source);
          let lobe=.022*v2Pattern(point,68.,fp,salt+659u)+.009*v2Pattern(point,29.,fp,salt+661u);
          let cut=.065+.28*fract(piece.z*13.17);
          let field=piece.x-cut+lobe;
          let aa=max(.0015,fp/850.*.8);
          let cap=sstep(-aa,aa,field);
          let submerged=sstep(-.035-aa,-.012+aa,field);
          let castShade=sstep(-aa,aa,field+.12*piece.w-.006)*(1.-cap);
          let keep=1.-sstep(115.,290.,fp);
          berg=mix(meanBerg,cap*selected*(1.-pack*.65),keep);
          shadow=castShade*selected*.42*keep*(1.-pack);
          bergRim=max(0.,submerged-cap)*selected*.75*keep;
          let plateau=sstep(.006,.065,field);
          bergLight=mix(.95,clamp(.68+.29*plateau+.26*piece.w+.035*v2Pattern(point,75.,fp,salt+663u),.48,1.),keep);
        }
      }
      let coverage=pack+(1.-pack)*berg;
      brightness=mix(brightness,bergLight,berg/max(.001,coverage));
      return vec4<f32>(coverage,brightness,max(rim,bergRim),shadow);
    }
    fn v2Water(depth:f32,kind:f32,cover:f32,e:V2Env,point:vec3<f32>,fp:f32,salt:u32)->vec3<f32>{
      let sediment=sstep(.1,.9,e.sand)*sstep(.1,1.,e.moisture);
      let bed=mix(v2Mineral(e)*.65,vec3<f32>(.20,.33,.24),sediment);
      let shallow=mix(bed,vec3<f32>(.12,.31,.32),.60);
      // Retain Render's shelf-to-abyss depth cue beyond the optical shallows.
      // This is bathymetric appearance, not visibility of the deep seafloor.
      var deep=vec3<f32>(.025,.12,.21);
      if(kind<1.5){deep=mix(vec3<f32>(.045,.16,.27),vec3<f32>(.018,.065,.15),sstep(150.,3000.,depth));}
      let attenuation=select(.15,.035,kind<1.5);
      var water=mix(shallow,deep,1.-exp(-max(depth,0.)*attenuation*max(.08,cover*cover)));
      let frozen=v2Ice(depth,kind,e,point,fp,salt);
      water=mix(water,vec3<f32>(.12,.44,.48),frozen.z*.6)*(1.-frozen.w);
      let ice=mix(vec3<f32>(.38,.60,.66),vec3<f32>(.98,.99,.97),frozen.y);
      return mix(water,ice,frozen.x);
    }

  `;
  // Projected-disc quadrature of the existing analytic crowns, not a colour
  // gain: 64 equal-area radial x128 azimuth samples, roughness=.9. Entries
  // contain diffuse/specular BRDF coefficients for broadleaf and needle.
  // Intrinsic AO follows occCrown without neighbour overlap or leaf clumps.
  const canopyMeanShader=`
    const V2_CROWN_AMBIENT=vec2<f32>(.376629411,.341894852);
    const V2_CROWN_RESPONSE=array<vec4<f32>,19>(
      vec4<f32>(.05495809,.00137647778,.0813788117,.00203904631),
      vec4<f32>(.0660553961,.00154106708,.08850629,.0020669753),
      vec4<f32>(.0778368112,.00171198182,.0952016323,.00209813878),
      vec4<f32>(.0902080708,.00188738668,.101430834,.00212977456),
      vec4<f32>(.103111575,.00206622947,.107197465,.00216111102),
      vec4<f32>(.116499989,.00224785511,.112457862,.00219035487),
      vec4<f32>(.13033146,.00243201407,.117222519,.0022179789),
      vec4<f32>(.144564069,.00261878503,.121541253,.00224581331),
      vec4<f32>(.159132827,.00280803662,.125481928,.00227606726),
      vec4<f32>(.17388634,.00299793399,.12920948,.00231334651),
      vec4<f32>(.188276248,.00317723792,.132935776,.00236294396),
      vec4<f32>(.201329002,.0033263682,.137267752,.00244061725),
      vec4<f32>(.212849447,.0034477316,.14394931,.0025913043),
      vec4<f32>(.222749954,.00354514959,.150644987,.00272811941),
      vec4<f32>(.230955192,.00362148641,.15619416,.00282641325),
      vec4<f32>(.237402721,.00367886873,.1605546,.00289547597),
      vec4<f32>(.242043472,.00371883048,.163693123,.00294123078),
      vec4<f32>(.244842125,.00374240836,.165585843,.00296734911),
      vec4<f32>(.245777382,.00375020219,.166218354,.0029758416));
    fn v2CrownResponse(light:vec3<f32>,needle:f32)->vec2<f32>{
      let level=asin(clamp(normalize(light).y,0.,1.))*11.4591559;
      let i=min(i32(floor(level)),17);let t=clamp(level-f32(i),0.,1.);
      let q=mix(V2_CROWN_RESPONSE[i],V2_CROWN_RESPONSE[i+1],t);
      return mix(q.xy,q.zw,needle);
    }
    fn v2CanopyMeanRatio(rgb:vec3<f32>,needle:f32,light:vec3<f32>,light2:vec3<f32>,strength:f32)->vec3<f32>{
      let amount=clamp(strength,0.,1.);let albedo=groundToLinear(rgb);
      let response=2.2*v2CrownResponse(light,needle)+.58*v2CrownResponse(light2,needle);
      let ambient=mix(V2_CROWN_AMBIENT.x,V2_CROWN_AMBIENT.y,needle);
      let mean=albedo*(1.-amount+amount*ambient)+max(strength,0.)*(albedo*response.x+vec3<f32>(response.y));
      var flat:GroundMaterial;flat.albedo=albedo;flat.normal=vec3<f32>(0.,1.,0.);flat.roughness=.9;flat.occlusion=1.;
      let base=shadeTerrainGround(flat,vec3<f32>(0.,1.,0.),light,light2,strength);
      return pow(max(mean,vec3<f32>(0.)),vec3<f32>(1./2.2))/max(base,vec3<f32>(.0001));
    }
  `;
  function drawShader(source,material){
    // Reuse the precision-safe ground BRDF and deterministic crown candidates,
    // with a separate environment, ground albedo, calendar and water compositor.
    const noise=material.slice(material.indexOf('    fn surfaceHash('),material.indexOf('    // (unit-variance pattern'));
    const helpers=`
      fn sstep(a:f32,b:f32,x:f32)->f32{return groundSmooth(a,b,x);}
      ${noise}
      ${environment}
      ${canopyMeanShader}
      // Floor state survives the forest compositor; crowns never inherit its
      // albedo modulation, micro-normal or unresolved slope variance.
      var<private> v2FloorModulation:f32;
      var<private> v2MacroNormal:vec3<f32>;
      var<private> v2ReferenceRoughness:f32;
      var<private> v2BeachCoverage:f32;
      var<private> v2CanopyTone:f32;
      var<private> v2GroundSnow:f32;
      var<private> v2MeanResponseEnabled:bool;
      var<private> v2ForestNearReveal:f32;
      fn v2FloorAlbedo(color:vec3<f32>)->vec3<f32>{
        return groundToLinear(color*select(1.,v2FloorModulation,params.field.z<=-8.));
      }
      // Keep the same DEM normalization for each area component. Averaging
      // their linear light responses preserves albedo/AO covariance instead
      // of giving uncovered soil the occlusion of a neighbouring crown.
      fn v2ReliefResponse(m:GroundMaterial,referenceRoughness:f32,macroNormal:vec3<f32>,directionalMacro:f32,l:vec3<f32>,l2:vec3<f32>)->vec3<f32>{
        var reference=m;reference.normal=macroNormal;reference.roughness=referenceRoughness;reference.occlusion=1.;
        let referenceShade=shadeTerrainGround(reference,vec3<f32>(0.,1.,0.),l,l2,params.lighting.x);
        reference.normal=vec3<f32>(0.,1.,0.);
        let flatShade=shadeTerrainGround(reference,vec3<f32>(0.,1.,0.),l,l2,params.lighting.x);
        let shaded=shadeTerrainGround(m,vec3<f32>(0.,1.,0.),l,l2,params.lighting.x);
        let luma=vec3<f32>(.2126,.7152,.0722);let baseLuma=dot(flatShade,luma);
        return shaded*directionalMacro*baseLuma/max(dot(referenceShade,luma),max(.001,baseLuma*.05));
      }
      fn v2Point(p:vec2<f32>)->vec3<f32>{
        let m=r3.fracs[31];let world=params.style.xy+p;
        if((r3.info.y&1u)==0u){return vec3<f32>(world,0.);}
        let uv=(params.style.xy-m.xy)/m.zw+p/m.zw;
        let lon=uv.x*6.283185307-3.141592654;let lat=1.570796327-uv.y*3.141592654;
        var dir=vec3<f32>(cos(lat)*cos(lon),sin(lat),cos(lat)*sin(lon));
        if((r3.info.y&2u)!=0u){dir=vec3<f32>(dir.x,-dir.z,dir.y);}
        return dir*(m.z/6.283185307);
      }
      // Active-chart metric for terrestrial detail, periodic in longitude.
      // Marine appearance and large provinces retain canonical sphere points.
      fn v2LandPoint(p:vec2<f32>)->vec3<f32>{
        let world=params.style.xy+p;
        if((r3.info.y&1u)==0u){return vec3<f32>(world,0.);}
        let m=r3.fracs[31];let lon=fract((world.x-m.x)/m.z)*6.283185307-3.141592654;
        let radius=m.z/6.283185307;
        return vec3<f32>(radius*cos(lon),radius*sin(lon),world.y-m.y);
      }
      fn v2EnvironmentAt(p:vec2<f32>,s:vec3<f32>,footprint:f32,landPoint:vec3<f32>)->V2Env{
        let c=r3Aux(p,3,0.);let field=r3Aux(p,2,0.);let open=r3Aux(p,0,0.);let closed=r3Aux(p,1,0.);
        let ts=c.r*100.-50.;let tw=c.b*100.-50.;let threshold=max(120.,20.*(ts+tw)*.5+280.);
        let total=c.g*5.*threshold;
        var e=v2EvaluateDomains(v2Point(p),landPoint,s.x,s.yz,open.a-.5,ts,tw,total*closed.a,total*(1.-closed.a),c.a,field.a,params.render3.w,footprint,r3Seed());e.glacier=open.r;return e;
      }
      fn v2At(p:vec2<f32>,s:vec3<f32>,footprint:f32)->V2Env{
        var e=v2EnvironmentAt(p,s,footprint,v2LandPoint(p));
        let lod=max(log2(max(footprint,params.viewport.w)/params.viewport.w),0.);
        let ecology=r3Aux(p,4,lod);let field=r3Aux(p,2,lod);
        // Diagnostics and the statistical mean refer to the same population
        // that generates crowns, filtered at the physical pixel footprint.
        // Patch thresholds remain confined to the resolved compositor.
        e.woody=field.r;
        e.height=field.g*50.;e.needle=field.b;
        e.rock=ecology.b;e.grass=ecology.g*(1.-e.snow);e.sand=max(e.sand,ecology.a);
        return e;
      }
      // A lake is a horizontal plane intersecting the displayed DEM. At close
      // range, antialias that intersection in screen space instead of stretching
      // a bilinear 30 m wet/dry mask into a translucent shoreline. The existing
      // lake raster still bounds the support; distant area coverage is retained.
      fn v2LakeCoverage(coverage:f32,level:f32,s:vec3<f32>,fp:f32)->f32{
        let width=max((abs(s.y)+abs(s.z))*fp,1e-4);
        let geometric=clamp(.5+(level-s.x)/width,0.,1.);
        return mix(coverage,geometric,1.-sstep(15.,60.,fp));
      }
      fn v2Final(rgb:vec3<f32>,s:vec3<f32>,p:vec2<f32>,fp:f32,e:V2Env)->vec3<f32>{
        let pixel=p/params.viewport.w-.5;let w=hydroDisplayWater(p,hydroFiltered(pixel,fp/params.viewport.w),s.x);
        var cover=1.-clamp(hydroGround(s.x,pixel)/max(length(s.yz)*fp,1e-3)+.5,0.,1.);
        var depth=max(-s.x,0.);var kind=1.;
        if(w.w>.5){
          // Authority also retains dry below-sea-level inland depressions.
          if(w.z<.5){cover=0.;}else if(w.z>=1.5){
            cover=clamp(w.y,0.,1.);depth=max(w.x-s.x,0.);kind=w.z;
            if(kind<2.5&&cover>0.){cover=v2LakeCoverage(cover,w.x,s,fp);}
          }
        }
        if(cover<=0.){return rgb;}
        let marine=v2EnvironmentAt(p,s,fp,v2Point(p));
        var water=v2Water(depth,kind,cover,marine,v2Point(p),fp,r3Seed());
        if(params.field.z== -12.){water=mix(vec3<f32>(.08,.17,.24),vec3<f32>(.85,.94,1.),v2Ice(depth,kind,marine,v2Point(p),fp,r3Seed()).x);}
        if(params.field.z== -11.){water=vec3<f32>(0.,0.,.13);}
        return mix(rgb,water,cover);
      }
      fn v2Fragment(s:vec3<f32>,p:vec2<f32>,fp:f32,macroColor:vec3<f32>,macroLight:f32)->vec4<f32>{
        var e=v2Appearance(v2At(p,s,fp),fp);let channel=max(0.,-params.field.z-8.);
        let coast=r3Aux(p,4,0.);let sediment=r3Aux(p,1,0.).rgb;
        let meanLod=max(log2(max(fp,params.viewport.w)/params.viewport.w),0.);
        let inherited=r3Aux(p,5,meanLod);let retained=r3Aux(p,0,meanLod);
        let flats=retained.g*(1.-groundSmooth(240.,960.,fp));
        // The retained legacy beach contract stores 0.8 * physical coverage.
        // Snow may hide the sand but must not grow trees on the same beach.
        // Low flats retain their sediment too, without forbidding estuary trees.
        v2BeachCoverage=clamp(coast.a/.8,0.,1.);
        e.woody*=1.-v2BeachCoverage;e.grass*=1.-v2BeachCoverage;e.rock*=1.-v2BeachCoverage;
        if(channel>0.){return vec4<f32>(v2Final(v2Color(e,channel),s,p,fp,e),1.);}
        // One mean ground colour across zooms. V2 adds a physical detail
        // response below; replacing this guide with another palette at 6 m/px
        // changed the same ground's hue as the camera crossed that threshold.
        let groundMean=v2GroundTint(macroColor,e);
        var open=mix(groundMean,sediment*.9,flats*.8);
        open=mix(open,sediment,coast.a);
        open=mix(open,vec3<f32>(.88,.91,.91),e.snow);
        // The leaf-on guide fixes the canopy's mean material. V2 contributes
        // its own annual phenology/snow response, without a zoom palette swap.
        var leafOn=e;leafOn.leaf=1.;leafOn.snow=0.;
        var snowFree=e;snowFree.snow=0.;
        var closed=inherited.rgb*v2Canopy(snowFree)/max(v2Canopy(leafOn),vec3<f32>(.02));
        closed=mix(closed,vec3<f32>(.72,.79,.77),e.snow*.62);
        // Render paints low-flat sediment after its unresolved woods. Keep
        // that mean composition without painting sediment onto real crowns.
        let l=normalize(vec3<f32>(-params.sunlight.x,params.sunlight.z,-params.sunlight.y));
        let angle=atan2(l.z,l.x)+2.35619449;let horizontal=length(l.xz)/.70710678;
        let fill=vec2<f32>(.2*cos(angle)+.75*sin(angle),.2*sin(angle)-.75*cos(angle))*horizontal;
        let l2=vec3<f32>(fill.x,.63*l.y/.70710678,fill.y);
        var meanClosed=mix(closed,sediment*.9,flats*.8*(1.-e.snow));
        meanClosed=mix(meanClosed,sediment,coast.a*(1.-e.snow));
        let distantColor=mix(open,meanClosed,e.woody*e.leaf);
        let rain=e.potential*max(120.,20.*e.temperature+280.);
        var m:GroundMaterial;
        // Cover changes amplitudes and roughness, not grass-shaped mini crowns.
        let gain=mix(mix(.02,.45,e.rock)+.07*(1.-e.grass),.008,e.snow);
        var clasts=0.;let detail=groundFieldWithClasts(p,r3Seed(),fp,&clasts);
        // The bake stores six times its filtered physical slope. Divide by
        // three to match the deliberately gentler x2 BRDF slope response.
        let macroGradient=vec2<f32>(inherited.a,retained.b)/3.;
        let macroNormal=normalize(vec3<f32>(-macroGradient.x*params.lighting.w,1.,-macroGradient.y*params.lighting.w));
        v2MacroNormal=normalize(vec3<f32>(-s.y*2.*params.lighting.w,1.,-s.z*2.*params.lighting.w));
        m.normal=normalize(vec3<f32>(-s.y*2.*params.lighting.w-detail.y*gain,1.,-s.z*2.*params.lighting.w-detail.w*gain));
        let swardKeep=groundVisibility(.32,fp);let herb=e.grass*(1.-e.rock)*(1.-e.snow);
        let swardTone=clamp(detail.x*4.,-1.,1.)*.025*herb*swardKeep;
        // Bound the shape before its material amplitude. Clamping directly
        // to +/-amplitude made the 8 m band into two flat colour levels.
        let formAmplitude=mix(mix(.04,.06,e.rock),.006,e.snow);
        let form=clamp(detail.x*10.,-1.,1.)*formAmplitude;
        v2FloorModulation=1.+form+swardTone;
        // The inherited guide is a display-RGB area mean, like surfaceMix.
        // Keep that mean instead of brightening it through a second mixture
        // of decoded linear endpoints. Detail belongs only to exposed floor.
        m.albedo=groundToLinear(distantColor+open*(v2FloorModulation-1.)*(1.-e.woody*e.leaf));
        v2ReferenceRoughness=mix(.88,.54,e.rock);
        m.roughness=clamp(pow(pow(v2ReferenceRoughness,4.)+gain*gain*groundSlopeVariance(fp),.25),.4,.98);
        m.occlusion=clamp(1.-.10*max(-detail.x*12.,0.)*e.rock,.8,1.);
        // The canonical baked hillshade carries the user's ambient, contrast
        // and full relief exaggeration. Divide out the macro BRDF response,
        // retaining only local material/crown relief instead of shading twice.
        groundCoverLast=vec4<f32>(e.rock,(1.-e.rock)*(1.-e.grass),e.grass*(1.-e.rock),0.);groundShadeLast=1.;
        // V2 aux2.a carries bank moisture, not the legacy stand tone.
        v2CanopyTone=e.tone;v2GroundSnow=e.snow;v2MeanResponseEnabled=true;v2ForestNearReveal=0.;
        let floorMaterial=m;let floorReferenceRoughness=v2ReferenceRoughness;
        let c=r3Aux(p,3,0.);let climate=forestClimate(c);
        applyForest(&m,p,fp,l,RenderGuide(distantColor,open,closed,e.woody*e.leaf),climate,false);
        // Retained for material probes; production mixes area responses below.
        let shaded=shadeTerrainGround(m,vec3<f32>(0.,1.,0.),l,l2,params.lighting.x);
        let flatHillshade=mix(1.,params.lighting.y+(1.-params.lighting.y)*pow(clamp(params.sunlight.z,0.,1.),.85*params.lighting.z),params.lighting.x);
        let directionalMacro=macroLight/max(flatHillshade,.0001);
        var color:vec3<f32>;
        if(v2ForestNearReveal>=.9999){
          // Fully resolved photons stay exactly on the existing crown path.
          color=v2ReliefResponse(m,v2ReferenceRoughness,macroNormal,directionalMacro,l,l2);
        }else{
          // A statistical stand has three areas; no patch threshold is sampled
          // here. The same population later generates the explicit crowns.
          let population=r3Aux(p,2,meanLod);
          var st:ForestStand;st.p=p;st.cover=population.r;st.height=population.g*50.;st.needle=population.b;st.tone=e.tone;
          st.trop=climate.trop;st.dry=climate.dry;st.kappa=forestKappa(climate.humid,st.height,st.needle,st.trop);
          let density=forestDensity(st.cover,st.kappa);let available=1.-v2BeachCoverage;
          let share=st.cover/max(density,.001)*available;let canopy=st.cover*e.leaf*available;
          let weights=vec3<f32>(1.-share,max(share-canopy,0.),canopy);
          let bareOpen=mix(groundMean,vec3<f32>(.88,.91,.91),e.snow);
          var meanOpen=floorMaterial;meanOpen.albedo=v2FloorAlbedo(bareOpen);
          var meanLitter=meanOpen;meanLitter.albedo=v2FloorAlbedo(forestFloor(bareOpen,st,climate.cold));meanLitter.occlusion*=.72;
          let crownBase=forestCrownColor(st,closed,st.needle,vec2<f32>(0.),0.);
          let crownResponse=crownBase*mix(vec3<f32>(1.),v2CanopyMeanRatio(crownBase,st.needle,l,l2,params.lighting.x),1.-e.snow);
          var meanCrown=meanOpen;meanCrown.albedo=groundToLinear(crownResponse);meanCrown.normal=v2MacroNormal;meanCrown.roughness=.9;meanCrown.occlusion=1.;
          var meanResponse=weights.x*groundToLinear(v2ReliefResponse(meanOpen,floorReferenceRoughness,macroNormal,directionalMacro,l,l2));
          if(weights.y>0.){meanResponse+=weights.y*groundToLinear(v2ReliefResponse(meanLitter,floorReferenceRoughness,macroNormal,directionalMacro,l,l2));}
          if(weights.z>0.){meanResponse+=weights.z*groundToLinear(v2ReliefResponse(meanCrown,.9,macroNormal,directionalMacro,l,l2));}
          // Unresolved flats follow the legacy order AFTER canopy averaging.
          // Resolved estuary crowns retain their own material and eligibility.
          let flatWeight=flats*.8*(1.-e.snow);let beachWeight=coast.a*(1.-e.snow);
          if(flatWeight>0.){var sedimentMaterial=meanOpen;sedimentMaterial.albedo=v2FloorAlbedo(sediment*.9);meanResponse=mix(meanResponse,groundToLinear(v2ReliefResponse(sedimentMaterial,floorReferenceRoughness,macroNormal,directionalMacro,l,l2)),flatWeight);}
          if(beachWeight>0.){var sedimentMaterial=meanOpen;sedimentMaterial.albedo=v2FloorAlbedo(sediment);meanResponse=mix(meanResponse,groundToLinear(v2ReliefResponse(sedimentMaterial,floorReferenceRoughness,macroNormal,directionalMacro,l,l2)),beachWeight);}
          if(v2ForestNearReveal>0.){let nearResponse=groundToLinear(v2ReliefResponse(m,v2ReferenceRoughness,macroNormal,directionalMacro,l,l2));meanResponse=mix(meanResponse,nearResponse,v2ForestNearReveal);}
          color=groundToDisplay(meanResponse);
        }
        return vec4<f32>(v2Final(v2DisplayShoulder(color),s,p,fp,e),1.);
      }
    `;
    let code=source.replace('    @vertex fn vertex(',helpers+'\n    @vertex fn vertex(');
    code=code.replace('      // Water and its antialiased shoreline', '      if(params.field.z<=-8.){return v2Fragment(s,p,fp,baked.rgb,baked.a);}\n      // Water and its antialiased shoreline');
    code=code.replace('var k:ForestClimate;k.annual=c.r*100.-50.;k.humid=c.g*4.;let coldest=c.b*100.-50.;', 'var k:ForestClimate;k.annual=c.r*100.-50.;k.humid=c.g*4.;var coldest=c.b*100.-50.;if(params.field.z<=-8.){let ts=c.r*100.-50.;let tw=c.b*100.-50.;k.annual=(ts+tw)*.5;k.humid=c.g*5.;coldest=min(ts,tw);}');
    code=code.replace('      k.alive=forestU8(h,16u)<q&&threshold<share;', '      k.alive=forestU8(h,16u)<q&&threshold<share;\n      if(params.field.z<=-8.){let beach=clamp(r3Aux(k.c,4,0.).a/.8,0.,1.);let beachChoice=forestU8(groundHash(vec3<i32>(cell,level*2+37),seed),0u);let w=hydroDisplayWater(k.c,hydroAt(k.c/params.viewport.w-.5),r3Height(vec2<i32>(k.c/params.viewport.w)));if(beachChoice<beach||(w.w>.5&&w.z>=1.5&&w.y>.05)||hydroGround(r3Height(vec2<i32>(k.c/params.viewport.w)),k.c/params.viewport.w-.5)<0.){k.alive=false;}}');
    code=code.replace('      let hc=hit.id*0x9e3779b9u;', '      if(params.field.z<=-8.){let e=v2At(p,vec3<f32>(0.),0.);hit.cover*=e.leaf;shadow*=e.leaf;forestResolvedCover*=e.leaf;}\n      let hc=hit.id*0x9e3779b9u;');
    code=code.replace('      let openGround=groundRecolor(guide.open);','      let openGround=select(groundRecolor(guide.open),guide.open,params.field.z<=-8.);');
    // These hooks are deliberately checked: a changed forest template must
    // not silently fall back to a smooth floor or put soil detail on crowns.
    const hook=(before,after)=>{if(!code.includes(before))throw Error('Render v2 floor hook missing: '+before);code=code.replace(before,after);};
    hook('st.tone=field.a;', 'st.tone=select(field.a,v2CanopyTone,params.field.z<=-8.);');
    hook('return mix(ground,litter,.65*(1.-.6*st.dry));',
      'if(params.field.z<=-8.){litter=mix(litter,ground,v2GroundSnow);}return mix(ground,litter,.65*(1.-.6*st.dry));');
    // Evaluate the existing near material without displaying its patch mask
    // prematurely. The original reveal later blends LIGHT responses only.
    hook('      if(reveal<=0.){return;}',
      '      if(params.field.z<=-8.&&v2MeanResponseEnabled){v2ForestNearReveal=reveal;if(reveal>0.){reveal=1.;}}\n      if(reveal<=0.){return;}');
    hook('forestResolvedCover=hit.cover*crownVis*reveal;',
      'forestResolvedCover=hit.cover*crownVis*select(reveal,v2ForestNearReveal,params.field.z<=-8.&&v2MeanResponseEnabled);');
    // Preserve highlights until the final shoulder; legacy styles keep their
    // original display clamp. Reference BRDFs must remain uncompressed.
    hook('fn groundToDisplay(c:vec3<f32>)->vec3<f32>{return pow(clamp(c,vec3<f32>(0.),vec3<f32>(1.)),vec3<f32>(1./2.2));}',
      'fn groundToDisplay(c:vec3<f32>)->vec3<f32>{if(params.field.z<=-8.){return pow(max(c,vec3<f32>(0.)),vec3<f32>(1./2.2));}return pow(clamp(c,vec3<f32>(0.),vec3<f32>(1.)),vec3<f32>(1./2.2));}');
    // Beach eligibility belongs to each candidate's fixed root. Scaling the
    // final reveal by the fragment's beach fraction made whole crowns ghost
    // through sand, including crowns rooted on the vegetated side. Only the
    // unresolved stand response uses area availability at the shaded point.
    hook('      let grainVis=(1.-crownVis)',
      '      if(params.field.z<=-8.){mask*=1.-v2BeachCoverage;maskQ*=1.-clamp(r3Aux(q,4,0.).a/.8,0.,1.);}\n      let grainVis=(1.-crownVis)');
    hook('groundToLinear(openGround),reveal);return;', 'v2FloorAlbedo(openGround),reveal);return;');
    hook('groundToLinear(mix(openGround,litter,stand))', 'v2FloorAlbedo(mix(openGround,litter,stand))');
    hook('groundToLinear(mix(openGround,litter,under))', 'v2FloorAlbedo(mix(openGround,litter,under))');
    hook('      let macroSlope=-nGround.xz/max(nGround.y,.2);',
      '      let macroSlope=select(-nGround.xz/max(nGround.y,.2),-v2MacroNormal.xz/max(v2MacroNormal.y,.2),params.field.z<=-8.);');
    hook('      var roughness=mix((*material).roughness,.9,canopy);',
      '      var roughness=mix((*material).roughness,.9,canopy);var referenceRoughness=mix(v2ReferenceRoughness,.9,canopy);');
    hook('        let nearRough=mix((*material).roughness,.9,hit.cover);',
      '        let nearRough=mix((*material).roughness,.9,hit.cover);let nearReferenceRough=mix(v2ReferenceRoughness,.9,hit.cover);');
    hook('roughness=mix(roughness,nearRough,crownVis);',
      'roughness=mix(roughness,nearRough,crownVis);referenceRoughness=mix(referenceRoughness,nearReferenceRough,crownVis);');
    hook('(*material).roughness=mix((*material).roughness,roughness,reveal);',
      '(*material).roughness=mix((*material).roughness,roughness,reveal);if(params.field.z<=-8.){v2ReferenceRoughness=mix(v2ReferenceRoughness,referenceRoughness,reveal);}');
    code=code.replace('      let canopy=clamp(mask*d,0.,1.);', '      var canopy=clamp(mask*d,0.,1.);if(params.field.z<=-8.){canopy*=v2At(p,vec3<f32>(0.),0.).leaf;}');
    return code;
  }
  root.TerrainRenderV2={MODES,isStyle,number:s=>MODES[s]??-8,environment,canopyMeanShader,drawShader,mount,get:()=>({season}),set:v=>{season=normalize(v?.season);sync();},season:()=>season,writeParams:p=>p.set('v2season',String(season)),diagnostic:n=>Math.max(0,-n-8)};
})(typeof window==='undefined'?globalThis:window);



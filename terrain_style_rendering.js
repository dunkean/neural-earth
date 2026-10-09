/* Display tokens adapted from city_generator/web/src/render/styles.ts;
 * the MNE palette comes from rust/bridge/terrainRender.ts. GPL-3.0.
 * terrain_styles.json is shared with the PNG renderer.
 */
window.TerrainStyleRendering=(()=>{
  const styles=window.TerrainStyles;
  const number=mode=>{const basic={'orogen-biomes':-1,relief:0,biomes:1,temperature:2,precipitation:3}[mode];const index=Object.keys(styles).indexOf(mode);return basic??(index<0?0:index+4);};
  const f=x=>Number(x).toFixed(8);
  const rgb=hex=>'vec3<f32>('+[1,3,5].map(i=>f(parseInt(hex.slice(i,i+2),16)/255)).join(',')+')';
  const paletteWGSL=Object.entries(styles).map(([mode,p],i)=>{
    const stops=p.hypso;
    return `if(mode==${i+4}.){
      var c=${rgb(stops.at(-1)[1])};
      ${stops.slice(1).map(([t,color],j)=>`if(t<=${f(t)}){c=mix(${rgb(stops[j][1])},${rgb(color)},clamp((t-${f(stops[j][0])})/${f(t-stops[j][0])},0.,1.));}`).reverse().join('\n')}
      let dd=hill-params.sunlight.z;
      let base=clamp(1.+${f(p.shade)}*select(.8,1.,dd>0.)*dd*1.15,.58,1.25);
      let factor=pow(max(.01,1.+params.lighting.x*(base-1.)*(1.-params.lighting.y)/.65),params.lighting.z);
      ${p.hatch?`let hatch=select(0.,clamp((1.-factor)/.2,0.,1.)*${f(p.hatch)}*.65*params.lighting.x,fract((pos.x+pos.y)/6.)*6.<1.5);c=mix(c*(1.+.3*(factor-1.)),${rgb(p.ink)},hatch);`:'c*=factor;'}
      if(height<0.){c=${mode==='copernicus'?`mix(vec3<f32>(35.,125.,175.)/255.,vec3<f32>(8.,40.,96.)/255.,clamp(-height/10000.,0.,1.))`:rgb(p.seaFill)};}
      return c;
    }`;
  }).join('\n');
  const paperShader=`
    // Integer mixing avoids f32 sine/dot precision bands at planet-scale positions.
    // Keep this hash and its 24-bit output identical to terrain_styles._paper_noise.
    fn paperNoise(pos:vec2<f32>)->f32{
      let cell=bitcast<vec2<u32>>(vec2<i32>(pos));
      var h=(cell.x*0x9e3779b9u)^(cell.y*0x85ebca6bu);
      h=(h^(h>>16u))*0x7feb352du;
      h=(h^(h>>15u))*0x846ca68bu;
      h=h^(h>>16u);
      return f32(h>>8u)/16777216.;
    }
    fn paperAmount(mode:f32)->f32{
      ${Object.entries(styles).filter(([,p])=>p.grain).map(([mode,p])=>`if(mode==${number(mode)}.){return ${f(p.grain)};}`).join('\n')}
      return 0.;
    }
    fn paperColor(color:vec3<f32>,mode:f32,height:f32,pixel:vec2<f32>)->vec3<f32>{
      let amount=select(0.,paperAmount(mode),height>=0.);
      return color*(1.+(paperNoise(floor(pixel))-.5)*2.*amount);
    }`;
  // Grain belongs to the displayed pixels, never to the physical tile texture.
  const shader=`
    fn cityColor(height:f32,mode:f32,hill:f32,pos:vec2<f32>)->vec3<f32>{
      let t=pow(clamp(height/4500.,0.,1.),select(.85,1.,mode==${number('copernicus')}.));
      ${paletteWGSL} return vec3<f32>(0.);
    }`;
  const contourInk=`
    fn contourInk(mode:f32)->vec4<f32>{
      ${Object.entries(styles).map(([mode,p])=>`if(mode==${number(mode)}.){return vec4<f32>(${rgb(p.contour)},${f(p.contourOpacity)});}`).join('\n')}
      return vec4<f32>(.72156863,.45882353,.22745098,.75);
    }
    fn contourWidth(mode:f32,index:f32)->f32{
      var ratio=2.;
      ${Object.entries(styles).map(([mode,p])=>`if(mode==${number(mode)}.){ratio=${f(p.contourIndexW)};}`).join('\n')}
      return .9*select(1.,ratio,index>0.);
    }`;
  // Marching squares: one continuous segment per pair of crossed cell edges.
  // Coordinates are sample centres, including the halo, in the tile's UV space.
  // Saddle cells use the bilinear asymptotic decider rather than arbitrary joins.
  function buildContours(values,{width,height=width,halo=0,encoding},interval){
    if(!Number.isFinite(interval)||interval<=0)throw Error('Invalid contour interval');
    const segments=[],innerW=width-2*halo,innerH=height-2*halo;
    const metres=v=>encoding==='signed-sqrt'?Math.sign(v)*v*v:v;
    const pairs={1:[[3,0]],2:[[0,1]],3:[[3,1]],4:[[1,2]],6:[[0,2]],7:[[3,2]],8:[[3,2]],9:[[0,2]],11:[[1,2]],12:[[3,1]],13:[[0,1]],14:[[3,0]]};
    for(let y=Math.max(0,halo-1);y<Math.min(height-1,height-halo);y++)for(let x=Math.max(0,halo-1);x<Math.min(width-1,width-halo);x++){
      const a=values[y*width+x],b=values[y*width+x+1],c=values[(y+1)*width+x+1],d=values[(y+1)*width+x];
      const minimum=Math.min(metres(a),metres(b),metres(c),metres(d)),maximum=Math.max(metres(a),metres(b),metres(c),metres(d));
      if(!Number.isFinite(minimum)||!Number.isFinite(maximum)||minimum===maximum)continue;
      for(let k=Math.max(1,Math.ceil(minimum/interval));k*interval<=maximum;k++){
        const level=encoding==='signed-sqrt'?Math.sqrt(k*interval):k*interval;
        const code=(a>level?1:0)|(b>level?2:0)|(c>level?4:0)|(d>level?8:0);
        if(code===0||code===15)continue;
        let edges=pairs[code];
        if(code===5||code===10){
          const q=(a-level)*(c-level)-(b-level)*(d-level);
          edges=q>=0?[[0,1],[2,3]]:[[3,0],[1,2]];
        }
        const edge=e=>{
          if(e===0)return[x+(level-a)/(b-a),y];
          if(e===1)return[x+1,y+(level-b)/(c-b)];
          if(e===2)return[x+(level-d)/(c-d),y+1];
          return[x,y+(level-a)/(d-a)];
        };
        for(const [e0,e1]of edges){const p=edge(e0),q=edge(e1);if(Math.hypot(p[0]-q[0],p[1]-q[1])<1e-8)continue;
          segments.push((p[0]+.5-halo)/innerW,(p[1]+.5-halo)/innerH,(q[0]+.5-halo)/innerW,(q[1]+.5-halo)/innerH,Number(k%5===0));}
      }
    }
    return new Float32Array(segments);
  }
  // Each slider unit doubles density; the previous minimum (25%) is at its centre.
  const densityFromSlider=value=>25*2**(value-3);
  const densityToSlider=density=>3+Math.log2(density/25);
  const autoInterval=(mpp,density=25)=>Math.max(1,Math.min(10000,Math.round((mpp>=15000?500:mpp>=3000?200:mpp>=600?100:mpp>=100?50:20)*100/density)));
  let settings={enabled:false,interval:200,automatic:true,width:.9,density:25};
  return {number,shader,paperShader,contourInk,buildContours,autoInterval,densityFromSlider,densityToSlider,get:()=>({...settings}),set:value=>{settings={...settings,...value};},vectors:()=>[Number(settings.enabled),settings.interval,settings.width,settings.density]};
})();

/* Perspective globe with the map's progressive neural tiles; no external runtime. */
window.TerrainGlobe=(()=>{
  const vertex='attribute vec2 position; varying vec2 screen; void main(){screen=position;gl_Position=vec4(position,0.0,1.0);}';
  const fragment=`precision highp float;
    varying vec2 screen; uniform sampler2D atlas; uniform vec2 viewport;
    uniform vec2 orbit; uniform float altitude; uniform bool ready;
    uniform sampler2D detail; uniform sampler2D gpuDetail; uniform vec2 detailSpan; uniform vec2 detailOffset; uniform bool detailReady; uniform bool gpuDetailReady;
    uniform vec2 detailOrbit; uniform vec2 detailRoll;
    uniform vec4 lighting; uniform vec3 sunlight;
    const float PI=3.141592653589793;
    void main(){
      vec3 ray=normalize(vec3(screen.x*viewport.x/viewport.y,screen.y,-2.41421356));
      float b=(1.0+altitude)*ray.z,c=altitude*(2.0+altitude),disc=b*b-c;
      if(disc<0.0){gl_FragColor=vec4(0.035,0.065,0.10,1.0);return;}
      // The rationalized near root retains surface detail even near the ground.
      float t=c/(-b+sqrt(disc));
      // Longitude increases to screen right, as in the equirectangular map.
      vec3 p=vec3(-ray.x*t,ray.y*t,1.0+altitude+ray.z*t);
      float cp=cos(orbit.y),sp=sin(orbit.y),cy=cos(orbit.x),sy=sin(orbit.x);
      vec3 q=vec3(p.x,p.y*cp+p.z*sp,-p.y*sp+p.z*cp);
      vec3 n=normalize(vec3(q.x*cy+q.z*sy,q.y,-q.x*sy+q.z*cy));
      // asin(y) loses polar distance when y rounds to +/-1 in float32.
      // Measure it from the horizontal components, which retain that distance.
      float polarAngle=atan(length(n.xz),abs(n.y))/PI;
      vec2 uv=vec2(fract(atan(n.z,n.x)/(2.0*PI)+0.5),n.y>=0.0?polarAngle:1.0-polarAngle);
      vec3 rgb=ready?texture2D(atlas,uv).rgb:vec3(0.12,0.29,0.43);
      // The polar NN chart rotates the sphere. Rotate only local tangents
      // here, retaining sub-metre precision rather than subtracting world UVs.
      vec3 dp=vec3(detailRoll.x*p.x+detailRoll.y*p.y,-detailRoll.y*p.x+detailRoll.x*p.y,p.z);
      cp=cos(detailOrbit.y);sp=sin(detailOrbit.y);
      q=vec3(dp.x,dp.y*cp+dp.z*sp,-dp.y*sp+dp.z*cp);
      // Compute local angles before adding world coordinates: sub-metre camera
      // movement must not disappear in the precision of a global texture UV.
      float horizontal=length(q.xz);
      // Rationalize only on the near side; across the pole h-z is already
      // stable. Subtracting two global latitudes loses fine detail there.
      float correction=q.z>0.0?q.x*q.x/max(horizontal+q.z,1.0e-30):horizontal-q.z;
      float latitudeDelta=atan(dp.y-correction*sp,dp.z+correction*cp);
      vec2 delta=vec2(-atan(q.x,q.z)/(2.0*PI),-latitudeDelta/PI);
      vec2 local=(detailOffset+delta)/detailSpan;
      if(detailSpan.x>=0.99999)local.x=fract(local.x);
      if(detailReady&&local.x>=0.0&&local.x<=1.0&&local.y>=0.0&&local.y<=1.0){vec4 pixel=texture2D(detail,local);rgb=mix(rgb,pixel.rgb,pixel.a);if(gpuDetailReady){pixel=texture2D(gpuDetail,local);rgb=mix(rgb,pixel.rgb,pixel.a);}}
      float shade=pow(max(dot(n,sunlight),0.0),lighting.z);
      float intensity=mix(1.0,lighting.y+(1.0-lighting.y)*shade,lighting.x);
      gl_FragColor=vec4(rgb*intensity,1.0);
    }`;
  function create(canvas,{onChange=()=>{},onError=()=>{},isActive=()=>false}={}){
    const gl=canvas.getContext('webgl',{alpha:false,antialias:true});
    if(!gl){onError('WebGL indisponible pour le globe.');return null;}
    let yaw=Math.PI/2,pitch=0,altitude=2.1,ready=false,drag=null,lost=false,lastImage=null;
    let program,texture,detailTexture,gpuDetailTexture,locations,detailReady=false,gpuDetailReady=false,detailBounds=[0,0,1,1],detailOffset=[0,0],detailOrbit=[yaw,pitch],detailRoll=[1,0],surfaceSignature=null,surfaceTiles=[],surfacePresented=[];
    const surface=document.createElement('canvas'),surfaceContext=surface.getContext('2d');
    function initialize(){
      function compile(type,source){const shader=gl.createShader(type);gl.shaderSource(shader,source);gl.compileShader(shader);if(!gl.getShaderParameter(shader,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(shader));return shader;}
      const vs=compile(gl.VERTEX_SHADER,vertex),fs=compile(gl.FRAGMENT_SHADER,fragment);program=gl.createProgram();gl.attachShader(program,vs);gl.attachShader(program,fs);gl.linkProgram(program);gl.deleteShader(vs);gl.deleteShader(fs);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(program));gl.useProgram(program);
      const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);const pos=gl.getAttribLocation(program,'position');gl.enableVertexAttribArray(pos);gl.vertexAttribPointer(pos,2,gl.FLOAT,false,0,0);
      texture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,texture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([30,70,110,255]));
      detailTexture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,detailTexture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([0,0,0,0]));
      gpuDetailTexture=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,gpuDetailTexture);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([0,0,0,0]));
      locations=Object.fromEntries(['atlas','detail','gpuDetail','detailSpan','detailOffset','detailOrbit','detailRoll','detailReady','gpuDetailReady','viewport','orbit','altitude','ready','lighting','sunlight'].map(key=>[key,gl.getUniformLocation(program,key)]));ready=false;detailReady=false;gpuDetailReady=false;surfaceSignature=null;surfaceTiles=[];surfacePresented=[];lost=false;lastImage=null;
    }
    try{initialize();}catch(error){onError(error.message);return null;}
    canvas.addEventListener('webglcontextlost',event=>{event.preventDefault();lost=true;onError('Contexte du globe perdu.');});
    canvas.addEventListener('webglcontextrestored',()=>{try{initialize();onChange();}catch(error){onError(error.message);}});
    function orbitBy(dx,dy){const scale=altitude/2.1;yaw=(yaw+dx*scale)%(2*Math.PI);pitch=Math.max(-Math.PI*.499999,Math.min(Math.PI*.499999,pitch+dy*scale));onChange();}
    function projection(polar){
      if(!polar)return{orbit:[yaw,pitch],roll:[1,0]};
      const cp=Math.cos(pitch),sp=Math.sin(pitch),cy=Math.cos(yaw),sy=Math.sin(yaw);
      const chartYaw=Math.atan2(cp*sy,-sp),chartPitch=Math.atan2(cp*cy,Math.hypot(cp*sy,sp));
      return{orbit:[chartYaw,chartPitch],roll:[cy*Math.cos(chartYaw),-sp*sy*Math.cos(chartYaw)+cp*Math.sin(chartYaw)]};
    }
    function surfacePoint(x,y,width,height,polar=false){
      const rx=x*width/height,ry=y,rz=-2.41421356,length=Math.hypot(rx,ry,rz),z=rz/length;
      const b=(1+altitude)*z,c=altitude*(2+altitude),disc=b*b-c;if(disc<0)return null;
      const t=c/(-b+Math.sqrt(disc)),px=-rx/length*t,py=ry/length*t,pz=1+altitude+z*t;
      const qy=py*Math.cos(pitch)+pz*Math.sin(pitch),qz=-py*Math.sin(pitch)+pz*Math.cos(pitch);
      const nx=px*Math.cos(yaw)+qz*Math.sin(yaw),nz=-px*Math.sin(yaw)+qz*Math.cos(yaw);
      const ny=polar?nz:qy,zz=polar?-qy:nz;
      let u=Math.atan2(zz,nx)/(2*Math.PI)+.5,center=.75-projection(polar).orbit[0]/(2*Math.PI);center=((center%1)+1)%1;
      const polarAngle=Math.atan2(Math.hypot(nx,zz),Math.abs(ny))/Math.PI;
      u+=Math.round(center-u);return[u,ny>=0?polarAngle:1-polarAngle];
    }
    function camera(width,height,wb,{polar=true}={}){
      if(!width||!height||!wb)return null;
      const polarChart=polar&&Math.abs(pitch)>Math.PI/3&&altitude<.35;
      const center=surfacePoint(0,0,width,height,polarChart),points=[center];
      // Sample the viewport and the visible limb; include poles analytically.
      for(let i=0;i<=32;i++)for(const [x,y]of [[-1+2*i/32,-1],[-1+2*i/32,1],[-1,-1+2*i/32],[1,-1+2*i/32]]){const p=surfacePoint(x,y,width,height,polarChart);if(p)points.push(p);}
      const limb=2.41421356/Math.sqrt(altitude*(altitude+2));
      for(let i=0;i<128;i++){const angle=i*2*Math.PI/128,x=Math.cos(angle)*limb*height/width*(1-1e-10),y=Math.sin(angle)*limb*(1-1e-10);if(Math.abs(x)<=1&&Math.abs(y)<=1){const p=surfacePoint(x,y,width,height,polarChart);if(p)points.push(p);}}
      let u0=Math.min(...points.map(p=>p[0])),u1=Math.max(...points.map(p=>p[0])),v0=Math.min(...points.map(p=>p[1])),v1=Math.max(...points.map(p=>p[1]));
      if(!polarChart)for(const sign of [-1,1]){const z=sign*Math.sin(pitch),y=sign*Math.cos(pitch),den=1+altitude-z,screenY=2.41421356*y/den;if(z>=1/(1+altitude)&&Math.abs(screenY)<=1){u0=center[0]-.5;u1=center[0]+.5;if(sign>0)v0=0;else v1=1;}}
      const radius=(wb[2]-wb[0])/(2*Math.PI),resolution=2*altitude*radius/(2.41421356*height),pad=2*resolution/(wb[2]-wb[0]);
      u0-=pad;u1+=pad;v0=Math.max(0,v0-2*pad);v1=Math.min(1,v1+2*pad);
      if(u1-u0>=1){u0=0;u1=1;}
      const uvBounds=[u0,v0,u1,v1],regions=[];
      for(let shift=Math.floor(u0);shift<Math.ceil(u1);shift++){const a=Math.max(0,u0-shift),b=Math.min(1,u1-shift);if(a<b)regions.push([wb[0]+a*(wb[2]-wb[0]),wb[1]+v0*(wb[3]-wb[1]),wb[0]+b*(wb[2]-wb[0]),wb[1]+v1*(wb[3]-wb[1])]);}
      const geographic=surfacePoint(0,0,width,height),detailProjection=projection(polarChart);
      return{cx:wb[0]+center[0]*(wb[2]-wb[0]),cy:wb[1]+center[1]*(wb[3]-wb[1]),geographicCenter:[wb[0]+geographic[0]*(wb[2]-wb[0]),wb[1]+geographic[1]*(wb[3]-wb[1])],neuralChart:polarChart?'polar':null,detailProjection,mpp:resolution,regions,uvBounds,bounds:[Math.min(...regions.map(b=>b[0])),Math.min(...regions.map(b=>b[1])),Math.max(...regions.map(b=>b[2])),Math.max(...regions.map(b=>b[3]))]};
    }
    canvas.addEventListener('pointerdown',e=>{if(e.button!==0||!isActive())return;e.stopPropagation();drag={id:e.pointerId,x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);canvas.parentElement.focus();});
    canvas.addEventListener('pointermove',e=>{if(!drag||drag.id!==e.pointerId)return;e.stopPropagation();const scale=3/Math.max(100,canvas.clientHeight);orbitBy((e.clientX-drag.x)*scale,(e.clientY-drag.y)*scale);drag.x=e.clientX;drag.y=e.clientY;});
    for(const name of ['pointerup','pointercancel','lostpointercapture'])canvas.addEventListener(name,e=>{if(drag?.id!==e.pointerId)return;e.stopPropagation();drag=null;});
    canvas.addEventListener('wheel',e=>{if(!isActive())return;e.preventDefault();e.stopPropagation();api.zoom(Math.exp(-e.deltaY*.0015));},{passive:false});
    canvas.addEventListener('dblclick',e=>{e.stopPropagation();api.zoom(2);});
    const api={
      orbit:orbitBy,zoom(f){if(!Number.isFinite(f)||f<=0)return;altitude=Math.max(1e-9,Math.min(11,altitude/f));onChange();},fit(){altitude=2.1;onChange();},camera,
      setImage(image){if(lost||image===lastImage)return;lastImage=image;gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,texture);if(image){gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,image);ready=true;}else ready=false;},
      setTiles(plan,state,wb,renderer=null){
        if(lost)return[];if(!state){detailReady=false;gpuDetailReady=false;surfaceSignature=null;surfaceTiles=[];surfacePresented=[];return[];}
        const signature=JSON.stringify([state.uvBounds,state.mpp,state.detailProjection,!!renderer,renderer?.uploads,window.TerrainLighting?.serialize(),plan.map(p=>[p.tile.key,p.bounds,p.uv])]);
        if(signature===surfaceSignature&&plan.every((p,i)=>p.tile===surfaceTiles[i]))return surfacePresented;
        surfaceSignature=signature;surfaceTiles=plan.map(p=>p.tile);surfacePresented=[];gpuDetailReady=false;detailBounds=state.uvBounds;detailOrbit=state.detailProjection?.orbit||[yaw,pitch];detailRoll=state.detailProjection?.roll||[1,0];detailOffset=[(state.cx-wb[0])/(wb[2]-wb[0])-detailBounds[0],(state.cy-wb[1])/(wb[3]-wb[1])-detailBounds[1]];detailReady=plan.length>0;if(!detailReady)return[];
        const [u0,v0,u1,v1]=detailBounds,worldWidth=wb[2]-wb[0],worldHeight=wb[3]-wb[1],limit=Math.min(4096,gl.getParameter(gl.MAX_TEXTURE_SIZE));
        surface.width=Math.max(1,Math.min(limit,Math.ceil((u1-u0)*worldWidth/state.mpp)));surface.height=Math.max(1,Math.min(limit,Math.ceil((v1-v0)*worldHeight/state.mpp)));
        surfaceContext.clearRect(0,0,surface.width,surface.height);
        const gpuRects=[];
        for(const p of plan){const tile=p.tile,image=tile.image,b=p.bounds,uv=p.uv,a=(b[0]-wb[0])/worldWidth,z=(b[2]-wb[0])/worldWidth;
          if(renderer&&tile.gpu&&!renderer.hasTile(tile.key)&&tile.heights)(tile.native_coarse?renderer.uploadCoarse.bind(renderer):renderer.uploadTile.bind(renderer))(tile.key,tile.heights,tile.heightOptions);
          for(let shift=Math.floor(u0)-1;shift<=Math.ceil(u1);shift++){if(z+shift<=u0||a+shift>=u1)continue;
            const rect={key:tile.key,lod:tile.lod,uv,x:(a+shift-u0)/(u1-u0)*surface.width,y:((b[1]-wb[1])/worldHeight-v0)/(v1-v0)*surface.height,width:(z-a)/(u1-u0)*surface.width,height:(b[3]-b[1])/worldHeight/(v1-v0)*surface.height};
            if(image){surfaceContext.drawImage(image,uv[0]*image.width,uv[1]*image.height,uv[2]*image.width,uv[3]*image.height,rect.x,rect.y,rect.width,rect.height);surfacePresented.push(tile.key);}
            if(renderer&&tile.gpu&&renderer.hasTile(tile.key))gpuRects.push(rect);
          }
        }
        if(gpuRects.length&&renderer.draw(gpuRects,{width:surface.width,height:surface.height,dpr:1})){
          // Import the shaded GPU canvas once when coverage changes. The map
          // and globe reuse the same height data and shaded tile textures.
          gl.activeTexture(gl.TEXTURE2);gl.bindTexture(gl.TEXTURE_2D,gpuDetailTexture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,renderer.canvas);gpuDetailReady=true;
          surfacePresented.push(...(renderer.lastDrawnRects??gpuRects).map(r=>r.key));
        }
        gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,detailTexture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,surface);gl.activeTexture(gl.TEXTURE0);
        detailReady=surfacePresented.length>0;return surfacePresented;
      },
      draw(width,height){if(lost||!width||!height)return;const max=gl.getParameter(gl.MAX_RENDERBUFFER_SIZE),dpr=Math.min(window.devicePixelRatio||1,max/width,max/height);const w=Math.round(width*dpr),h=Math.round(height*dpr);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;}gl.viewport(0,0,w,h);gl.useProgram(program);const light=window.TerrainLighting?.vectors('global')||[1,.35,1,1,-.5,-.5,.707,0];gl.uniform1i(locations.atlas,0);gl.uniform2f(locations.viewport,width,height);gl.uniform2f(locations.orbit,yaw,pitch);gl.uniform2f(locations.detailOrbit,...detailOrbit);gl.uniform2f(locations.detailRoll,...detailRoll);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,texture);gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,detailTexture);gl.activeTexture(gl.TEXTURE2);gl.bindTexture(gl.TEXTURE_2D,gpuDetailTexture);gl.uniform1i(locations.gpuDetail,2);gl.uniform1i(locations.gpuDetailReady,gpuDetailReady?1:0);gl.uniform1i(locations.detail,1);gl.uniform2f(locations.detailSpan,detailBounds[2]-detailBounds[0],detailBounds[3]-detailBounds[1]);gl.uniform2f(locations.detailOffset,...detailOffset);gl.uniform1i(locations.detailReady,detailReady?1:0);gl.uniform1f(locations.altitude,altitude);gl.uniform1i(locations.ready,ready?1:0);gl.uniform4fv(locations.lighting,light.slice(0,4));gl.uniform3fv(locations.sunlight,light.slice(4,7));gl.drawArrays(gl.TRIANGLES,0,6);},
      snapshot:()=>({yaw,pitch,distance:1+altitude,altitude,ready,detailReady,patches:surfaceTiles.length})
    };
    return api;
  }
  return {create};
})();

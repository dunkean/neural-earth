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
  const shadeShader = `
    struct Params { resolution:f32, halo:f32, width:f32, height:f32, mode:f32, pad0:f32, pad1:f32, pad2:f32 }
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
    fn shade(d:vec2<f32>) -> f32 {
      // Same directional light/vertical exaggeration as upstream relief_map.py.
      let normal=normalize(vec3<f32>(d.x,d.y,1.));
      return clamp(dot(normal,vec3<f32>(-.5,-.5,.7071067812)),0.,1.);
    }
    fn terrain(t:f32) -> vec3<f32> {
      if (t<.5) { return mix(vec3<f32>(0.,.8,.4),vec3<f32>(1.,1.,.6),(t-.25)*4.); }
      if (t<.75) { return mix(vec3<f32>(1.,1.,.6),vec3<f32>(.5,.36,.33),(t-.5)*4.); }
      return mix(vec3<f32>(.5,.36,.33),vec3<f32>(1.),(t-.75)*4.);
    }
    @compute @workgroup_size(8,8) fn main(@builtin(global_invocation_id) id:vec3<u32>) {
      if (any(id.xy>=textureDimensions(color))) { return; }
      let p=vec2<i32>(id.xy)+vec2<i32>(i32(params.halo));
      let height=textureLoad(elevation,p,0).r;
      let temp=clim(p,0)+clim(p,4)*max(height,0.);
      let rain=max(clim(p,2),0.);
      var rgb:vec3<f32>;
      if (height<0.) {
        let depth=pow(clamp(-height/10000.,0.,1.),.7);
        rgb=mix(vec3<f32>(.68,.88,1.),vec3<f32>(0.,.1,.45),depth);
      } else {
        let dx=(textureLoad(blurred,p+vec2<i32>(1,0),0).rg-textureLoad(blurred,p-vec2<i32>(1,0),0).rg)*3./params.resolution;
        let dy=(textureLoad(blurred,p+vec2<i32>(0,1),0).rg-textureLoad(blurred,p-vec2<i32>(0,1),0).rg)*3./params.resolution;
        let hs=pow(clamp(.75*shade(vec2<f32>(dx.r,dy.r))+.25*shade(vec2<f32>(dx.g,dy.g)),0.,1.),.85);
        rgb=terrain(.25+.75*pow(clamp(height/4500.,0.,1.),.7))*(.35+.65*hs);
      }
      if (params.mode==2.) {
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
        let light=clamp(dot(rgb,vec3<f32>(.333333333))*1.8,.55,1.);
        rgb=biome*light;
        if(height<0.) {
          let ocean=vec3<f32>(.10,.30,.50)+clamp(1.+height/6000.,0.,1.)*vec3<f32>(.18,.24,.20);
          rgb=mix(ocean,vec3<f32>(.82,.93,.98),clamp((-2.-temp)/10.,0.,1.));
        }
      }
      textureStore(color,vec2<i32>(id.xy),vec4<f32>(rgb,1.));
    }`;
  const renderShader = `
    struct Params { rect:vec4<f32>, viewport:vec4<f32>, crop:vec4<f32> }
    struct Vertex { @builtin(position) position:vec4<f32>, @location(0) uv:vec2<f32> }
    @group(0) @binding(0) var<uniform> params:Params;
    @group(0) @binding(1) var color:texture_2d<f32>;
    @group(0) @binding(2) var filtering:sampler;
    @vertex fn vertex(@builtin(vertex_index) id:u32) -> Vertex {
      let corners=array<vec2<f32>,6>(vec2<f32>(0,0),vec2<f32>(1,0),vec2<f32>(0,1),vec2<f32>(0,1),vec2<f32>(1,0),vec2<f32>(1,1));
      let uv=corners[id]; let xy=params.rect.xy+uv*params.rect.zw;
      var result:Vertex; result.position=vec4<f32>(xy.x/params.viewport.x*2.-1.,1.-xy.y/params.viewport.y*2.,0.,1.); result.uv=params.crop.xy+uv*params.crop.zw; return result;
    }
    @fragment fn fragment(input:Vertex) -> @location(0) vec4<f32> { return textureSample(color,filtering,input.uv); }`;

  class TerrainGpuRenderer {
    constructor(canvas, options) {
      this.canvas=canvas; this.options=options; this.tiles=new Map(); this.bytes=0;
      this.maxBytes=options.maxBytes ?? 96*1024*1024; this.available=false;
      this.status='initializing'; this.frames=0; this.uploads=0; this.serial=0; this.capacity=0;
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
      const [blur,shade,render]=await Promise.all([module(blurShader,'terrain Gaussian'),module(shadeShader,'terrain relief'),module(renderShader,'terrain quads')]);
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
        {binding:0,visibility:GPUShaderStage.VERTEX,buffer:{type:'uniform',hasDynamicOffset:true,minBindingSize:48}},
        {binding:1,visibility:GPUShaderStage.FRAGMENT,texture:{}},
        {binding:2,visibility:GPUShaderStage.FRAGMENT,sampler:{type:'filtering'}},
      ]});
      [this.horizontal,this.vertical,this.shade,this.pipeline]=await Promise.all([
        device.createComputePipelineAsync({layout:blurLayout,compute:{module:blur,entryPoint:'horizontal'}}),
        device.createComputePipelineAsync({layout:blurLayout,compute:{module:blur,entryPoint:'vertical'}}),
        device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[this.shadeLayout]}),compute:{module:shade,entryPoint:'main'}}),
        device.createRenderPipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[this.renderLayout]}),vertex:{module:render,entryPoint:'vertex'},fragment:{module:render,entryPoint:'fragment',targets:[{format:this.format}]},primitive:{topology:'triangle-list'}}),
      ]);
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
    uploadTile(key, heights, {width=304,height=304,halo=24,metresPerSample=30,climate=null,climateWidth=33,climateHeight=33,mode='relief'}={}) {
      if(!this.available)return false;
      if(!(heights instanceof Float32Array)||heights.length!==width*height||![width,height,halo].every(Number.isInteger)||halo<1||width<=2*halo||height<=2*halo||!Number.isFinite(metresPerSample)||metresPerSample<=0)throw Error('Invalid physical tile');
      if(Math.max(width,height)>this.device.limits.maxTextureDimension2D)throw Error('Tile exceeds GPU limits');
      this.deleteTile(key); const started=performance.now(); this.scratch(width,height);
      const innerWidth=width-2*halo,innerHeight=height-2*halo;
      const elevation=this.makeTexture(width,height,'r32float',GPUTextureUsage.COPY_DST);
      const color=this.makeTexture(innerWidth,innerHeight,'rgba8unorm',GPUTextureUsage.STORAGE_BINDING|GPUTextureUsage.COPY_SRC);
      if(climate!==null&&(!(climate instanceof Float32Array)||climate.length!==5*climateWidth*climateHeight))throw Error('Invalid tile climate');
      const climateTexture=this.device.createTexture({size:[climateWidth,climateHeight,5],format:'r32float',usage:GPUTextureUsage.TEXTURE_BINDING|GPUTextureUsage.COPY_DST});
      const climateValues=climate||new Float32Array(climateWidth*climateHeight*5);
      this.device.queue.writeTexture({texture:climateTexture},climateValues,{bytesPerRow:climateWidth*4,rowsPerImage:climateHeight},[climateWidth,climateHeight,5]);
      const params=this.device.createBuffer({size:32,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
      const modeNumber=climate?({relief:0,biomes:1,temperature:2,precipitation:3}[mode]??0):0;
      this.device.queue.writeBuffer(params,0,new Float32Array([metresPerSample,halo,innerWidth,innerHeight,modeNumber,0,0,0]));
      this.device.queue.writeTexture({texture:elevation},heights,{bytesPerRow:width*4,rowsPerImage:height},[width,height]);
      const horizontalGroup=this.device.createBindGroup({layout:this.blurLayout,entries:[{binding:0,resource:elevation.createView()},{binding:1,resource:this.scratchA.createView()}]});
      const shadeGroup=this.device.createBindGroup({layout:this.shadeLayout,entries:[{binding:0,resource:elevation.createView()},{binding:1,resource:this.scratchB.createView()},{binding:2,resource:color.createView()},{binding:3,resource:{buffer:params}},{binding:4,resource:climateTexture.createView({dimension:'2d-array'})}]});
      const encoder=this.device.createCommandEncoder({label:'terrain shade once'});
      for(const [pipeline,group,w,h] of [[this.horizontal,horizontalGroup,width,height],[this.vertical,this.verticalGroup,width,height],[this.shade,shadeGroup,innerWidth,innerHeight]]) {
        const pass=encoder.beginComputePass(); pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(Math.ceil(w/8),Math.ceil(h/8));pass.end();
      }
      this.device.queue.submit([encoder.finish()]); params.destroy();
      const bytes=width*height*4+innerWidth*innerHeight*4+climateWidth*climateHeight*5*4;
      this.tiles.set(key,{elevation,color,climate:climateTexture,bytes,lastUsed:++this.serial});this.bytes+=bytes;this.uploads++;this.lastUploadSubmitMs=performance.now()-started;
      this.evict();return this.hasTile(key);
    }
    hasTile(key) { return this.available&&this.tiles.has(key); }
    deleteTile(key) {
      const tile=this.tiles.get(key); if(!tile)return;
      tile.elevation.destroy();tile.color.destroy();tile.climate.destroy();this.bytes-=tile.bytes;this.tiles.delete(key);
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
      if(!this.available)return false;
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
        for(const tile of this.tiles.values())tile.group=null;
        this.evict();visible=visible.filter(r=>this.tiles.has(r.key));
      }
      visible.forEach((r,i)=>this.uniformData.set([r.x,r.y,r.width,r.height,width,height,0,0,...(r.uv||[0,0,1,1])],i*stride/4));
      if(visible.length)this.device.queue.writeBuffer(this.uniform,0,this.uniformData,0,visible.length*stride/4);
      const encoder=this.device.createCommandEncoder();const pass=encoder.beginRenderPass({colorAttachments:[{view:this.context.getCurrentTexture().createView(),clearValue:{r:0,g:0,b:0,a:0},loadOp:'clear',storeOp:'store'}]});
      pass.setPipeline(this.pipeline);
      visible.forEach((r,i)=>{
        const tile=this.tiles.get(r.key);tile.lastUsed=++this.serial;
        tile.group??=this.device.createBindGroup({layout:this.renderLayout,entries:[{binding:0,resource:{buffer:this.uniform,size:48}},{binding:1,resource:tile.color.createView()},{binding:2,resource:this.sampler}]});
        pass.setBindGroup(0,tile.group,[i*stride]);pass.draw(6);
      });
      pass.end();this.device.queue.submit([encoder.finish()]);this.frames++;this.canvas.style.visibility='visible';return true;
    }
    getStats() { const scratchBytes=(this.scratchWidth||0)*(this.scratchHeight||0)*16,uniformBytes=this.capacity*(this.device?.limits.minUniformBufferOffsetAlignment||0);return {status:this.status,tiles:this.tiles.size,bytes:this.bytes,scratchBytes,uniformBytes,totalBytes:this.bytes+scratchBytes+uniformBytes,maxBytes:this.maxBytes,frames:this.frames,uploads:this.uploads,lastUploadSubmitMs:this.lastUploadSubmitMs||0}; }
    dispose() {
      this.available=false;this.clear();this.scratchA?.destroy();this.scratchB?.destroy();this.uniform?.destroy();this.context?.unconfigure();this.notify('disposed');this.device?.destroy();
    }
  }
  window.createTerrainRenderer=async(canvas,options={})=>{
    const renderer=new TerrainGpuRenderer(canvas,options);
    try{return await renderer.initialize();}catch(error){renderer.dispose();options.onStatus?.('fallback',error.message);return null;}
  };
})();

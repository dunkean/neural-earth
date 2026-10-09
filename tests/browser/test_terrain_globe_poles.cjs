const NEURAL_EARTH_ROOT = require('node:path').resolve(__dirname, '../..');
process.chdir(NEURAL_EARTH_ROOT);
// Compare the real WebGL projection with float64 spherical coordinates.
const fs=require('node:fs'),assert=require('node:assert/strict');
const {chromium}=require(NEURAL_EARTH_ROOT + '/webgpu/node_modules/playwright');

(async()=>{
  const browser=await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
  try{
    const page=await browser.newPage();
    await page.setContent('<canvas id="globe" width="512" height="512"></canvas>');
    // Instrument only the output color: the production ray, UV and local
    // projection calculations still run on the actual float32 GPU shader.
    await page.evaluate(()=>{
      const original=WebGLRenderingContext.prototype.shaderSource;
      WebGLRenderingContext.prototype.shaderSource=function(shader,source){
        source=source.replace('gl_FragColor=vec4(rgb*intensity,1.0);',
          'gl_FragColor=vec4((n.y>=0.0?uv.y:1.0-uv.y)*100000.0,local.x,local.y,1.0);');
        return original.call(this,shader,source);
      };
    });
    await page.addScriptTag({content:fs.readFileSync('terrain_globe.js','utf8')});
    const cases=await page.evaluate(()=>{
      const canvas=document.getElementById('globe'),gl=canvas.getContext('webgl');
      const wb=[-20e6,-10e6,20e6,10e6],results=[];
      for(const sign of [-1,1])for(const altitude of [.001,.000001]){
        const globe=TerrainGlobe.create(canvas);
        globe.orbit(.4,sign*Math.PI);globe.zoom(2.1/altitude);
        const state=globe.camera(512,512,wb),image=document.createElement('canvas');
        image.width=image.height=1;
        globe.setTiles([{tile:{key:'polar',image},bounds:wb,uv:[0,0,1,1]}],state,wb);
        globe.draw(512,512);
        const {pitch,yaw}=globe.snapshot(),centerU=(state.cx-wb[0])/(wb[2]-wb[0]),centerV=(state.cy-wb[1])/(wb[3]-wb[1]);
        for(const [x,y]of [[256,256],[100,100],[400,100],[100,400],[400,400],[256,400]]){
          const sx=(x+.5)/256-1,sy=(y+.5)/256-1,len=Math.hypot(sx,sy,2.41421356),rx=sx/len,ry=sy/len,rz=-2.41421356/len;
          const b=(1+altitude)*rz,c=altitude*(2+altitude),t=c/(-b+Math.sqrt(b*b-c));
          const px=-rx*t,py=ry*t,pz=1+altitude+rz*t;
          const qy=py*Math.cos(pitch)+pz*Math.sin(pitch),qz=-py*Math.sin(pitch)+pz*Math.cos(pitch);
          const nx=px*Math.cos(yaw)+qz*Math.sin(yaw),nz=-px*Math.sin(yaw)+qz*Math.cos(yaw);
          const polar=Math.atan2(Math.hypot(px,qz),Math.abs(qy))/Math.PI;
          const ny=state.neuralChart?nz:qy,zz=state.neuralChart?-qy:nz;
          let u=Math.atan2(zz,nx)/(2*Math.PI)+.5;u+=Math.round(centerU-u);
          const chartPolar=Math.atan2(Math.hypot(nx,zz),Math.abs(ny))/Math.PI;
          const v=ny>=0?chartPolar:1-chartPolar,[u0,v0,u1,v1]=state.uvBounds;
          let localX=(u-u0)/(u1-u0);if(u1-u0>=.99999)localX=((localX%1)+1)%1;
          const expected=[polar*100000,localX,(v-v0)/(v1-v0)].map(v=>Math.round(Math.min(1,Math.max(0,v))*255));
          const actual=new Uint8Array(4);gl.readPixels(x,y,1,1,gl.RGBA,gl.UNSIGNED_BYTE,actual);
          results.push({sign,altitude,x,y,centerV,expected,actual:[...actual],error:gl.getError()});
        }
      }
      return results;
    });
    for(const result of cases){
      assert.equal(result.error,0);
      for(let c=0;c<3;c++)assert(Math.abs(result.actual[c]-result.expected[c])<=2,JSON.stringify(result));
    }
    console.log(`Polar globe projection: ${cases.length} GPU samples match spherical coordinates (north/south, across poles, fine zoom).`);
  }finally{await browser.close();}
})().catch(error=>{console.error(error);process.exitCode=1});

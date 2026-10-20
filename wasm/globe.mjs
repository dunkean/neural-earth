// The build inserts the maintained camera and input handlers from terrain_globe.js.
// Only rendering changes: its perspective raster is implemented in Rust/WASM.
import init,{GlobeRaster} from './render-pkg/neural_earth_render.js';
await init();
export function create(canvas,{onChange=()=>{},onError=()=>{},isActive=()=>false}={}) {
  const ctx=canvas.getContext('2d'),raster=new GlobeRaster();
  let yaw=Math.PI/2,pitch=0,altitude=2.1,ready=false,drag=null,lastImage=null,detailBounds=[0,0,1,1],detailReady=false,surfaceTiles=[],surfacePresented=[],signature=null;
  const surface=document.createElement('canvas'),surfaceContext=surface.getContext('2d');
  /* SHARED_CAMERA */
  /* SHARED_INPUT */
  const pixels=(image,callback)=>{
    surface.width=image?.width||0;surface.height=image?.height||0;
    if(image){surfaceContext.clearRect(0,0,surface.width,surface.height);surfaceContext.drawImage(image,0,0);}
    callback(image?surfaceContext.getImageData(0,0,surface.width,surface.height).data:new Uint8Array(),surface.width,surface.height);
  };
  const api={
    orbit:orbitBy,zoom(f){if(Number.isFinite(f)&&f>0){altitude=Math.max(1e-9,Math.min(11,altitude/f));onChange();}},fit(){altitude=2.1;onChange();},camera,
    restore(state){if(![state.yaw,state.pitch,state.altitude].every(Number.isFinite))throw Error('Invalid globe camera');yaw=state.yaw;pitch=state.pitch;altitude=Math.max(1e-9,Math.min(11,state.altitude));onChange();},
    /* SHARED_COORDINATES */
    setImage(image){if(image===lastImage)return;lastImage=image;pixels(image,(...args)=>raster.set_atlas(...args));ready=!!image;signature=null;},
    setTiles(plan,state,wb){
      if(!state){detailReady=false;surfaceTiles=[];raster.set_detail(new Uint8Array(),0,0);return[];}
      const key=JSON.stringify([state.uvBounds,state.mpp,plan.map(p=>[p.tile.key,p.bounds,p.uv])]);
      if(key===signature&&plan.every((p,i)=>p.tile===surfaceTiles[i]))return surfacePresented;
      signature=key;surfaceTiles=plan.map(p=>p.tile);surfacePresented=[];detailBounds=state.uvBounds;
      const [u0,v0,u1,v1]=detailBounds,ww=wb[2]-wb[0],wh=wb[3]-wb[1];
      surface.width=Math.max(1,Math.min(2048,Math.ceil((u1-u0)*ww/state.mpp)));surface.height=Math.max(1,Math.min(2048,Math.ceil((v1-v0)*wh/state.mpp)));
      surfaceContext.clearRect(0,0,surface.width,surface.height);
      for(const p of plan){const image=p.tile.image;if(!image)continue;const [x0,y0,x1,y1]=p.bounds,[a,b,c,d]=p.uv;
        for(let shift=Math.floor(u0)-1;shift<=Math.ceil(u1);shift++){
          const left=(x0-wb[0])/ww+shift,right=(x1-wb[0])/ww+shift;if(right<=u0||left>=u1)continue;
          surfaceContext.drawImage(image,a*image.width,b*image.height,c*image.width,d*image.height,(left-u0)/(u1-u0)*surface.width,((y0-wb[1])/wh-v0)/(v1-v0)*surface.height,(right-left)/(u1-u0)*surface.width,(y1-y0)/wh/(v1-v0)*surface.height);surfacePresented.push(p.tile.key);
        }
      }
      raster.set_detail(surfaceContext.getImageData(0,0,surface.width,surface.height).data,surface.width,surface.height);detailReady=surfacePresented.length>0;return surfacePresented;
    },
    draw(width,height){if(!width||!height)return;const scale=Math.min(1,720/width,512/height),w=Math.max(1,Math.round(width*scale)),h=Math.max(1,Math.round(height*scale));canvas.width=w;canvas.height=h;
      const rgba=raster.frame(w,h,width/height,yaw,pitch,altitude,detailBounds,window.TerrainLighting?.vectors('global')||[1,.35,1,1,-.5,-.5,.707,0]);ctx.putImageData(new ImageData(new Uint8ClampedArray(rgba),w,h),0,0);},
    snapshot:()=>({yaw,pitch,distance:1+altitude,altitude,ready,detailReady,patches:surfaceTiles.length})
  };return api;
}
window.TerrainGlobe={create};

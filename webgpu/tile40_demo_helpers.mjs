// Physical input presets and a relief view with fixed metre scaling.
export const presets = ['Plaine', 'Collines', 'Massif', 'Crête', 'Vallée', 'Plateau'];
export function sketchPreset(kind) {
  return Array.from({length:16}, (_, y) => Array.from({length:16}, (_, x) => {
    const u=x/15-.5, v=y/15-.5;
    const gaussian = (a,b,s) => Math.exp(-((u-a)**2+(v-b)**2)/s);
    switch (kind) {
      case 'Plaine': return 80+15*Math.cos(u*4)*Math.cos(v*4);
      case 'Collines': return 150+500*(gaussian(-.2,.1,.025)+.7*gaussian(.2,-.15,.015));
      case 'Crête': return 300+2800*Math.exp(-((v+.25*u)**2)/.007-u*u/.16);
      case 'Vallée': return 200+2500*Math.exp(-v*v/.25)*(1-Math.exp(-u*u/.018));
      case 'Plateau': return 200+1800/(1+Math.exp((Math.hypot(u,v)-.3)*40));
      default: return 300+3300*gaussian(-.1,0,.045)+1500*gaussian(.25,.15,.015);
    }
  }));
}

export function heightSummary(height) {
  if (!height.length) throw new Error('Empty height field.');
  let min=Infinity, max=-Infinity, sum=0;
  for (const value of height) {
    if (!Number.isFinite(value)) throw new Error('Nonfinite generated terrain.');
    min=Math.min(min,value);max=Math.max(max,value);sum+=value;
  }
  const sorted=Float32Array.from(height).sort();
  const at=q => sorted[Math.floor(q*(sorted.length-1))];
  return {minM:min,maxM:max,meanM:sum/height.length,p10M:at(.1),p90M:at(.9),reliefM:at(.9)-at(.1)};
}

export function reliefPixels(height, side, spacing) {
  if (height.length !== side*side || !(spacing > 0)) throw new Error('Physical DEM geometry required.');
  const pixels = new Uint8ClampedArray(side*side*4);
  for (let y=0; y<side; y++) for (let x=0; x<side; x++) {
    const i=y*side+x;
    const dx=(height[y*side+Math.min(side-1,x+1)]-height[y*side+Math.max(0,x-1)])/(x===0 || x===side-1 ? spacing : 2*spacing);
    const dy=(height[Math.min(side-1,y+1)*side+x]-height[Math.max(0,y-1)*side+x])/(y===0 || y===side-1 ? spacing : 2*spacing);
    const slope=Math.atan(Math.hypot(dx,dy)*1.5), aspect=Math.atan2(-dx,dy);
    const light=Math.max(0,Math.min(1,Math.SQRT1_2*Math.cos(slope)+Math.SQRT1_2*Math.sin(slope)*Math.cos(315*Math.PI/180-aspect)));
    const colour=height[i]>0 ? [200,185,150] : [120,150,190];
    for (let c=0;c<3;c++) pixels[4*i+c]=colour[c]*(.25+.75*light);
    pixels[4*i+3]=255;
  }
  return pixels;
}

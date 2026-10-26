/* Read the same visible conditioning fields as the renderer; no extra terrain inference. */
(function(root){
  function summarize(plan){
    const samples=[];
    for(const entry of plan||[]){
      const tile=entry.tile,options=tile?.heightOptions,fields=options?.climate;
      const width=options?.climateWidth,height=options?.climateHeight;
      if(!fields||!width||!height)continue;
      const count=width*height,center=Math.floor(height/2)*width+Math.floor(width/2);
      const elevation=Number.isFinite(tile.elevationMin)&&Number.isFinite(tile.elevationMax)?Math.max(0,(tile.elevationMin+tile.elevationMax)/2):0;
      const temperature=fields[center]+fields[4*count+center]*elevation,rain=Math.max(0,fields[2*count+center]);
      if(Number.isFinite(temperature)&&Number.isFinite(rain))samples.push({temperature,rain});
    }
    if(!samples.length)return{};
    const average=key=>samples.reduce((sum,s)=>sum+s[key],0)/samples.length;
    const temperature=average('temperature'),rain=average('rain');
    const biome=temperature< -8?'polar ice and snow':temperature<2?'tundra and alpine vegetation':rain<180?'desert':rain<500?'dry grassland and scrub':temperature>22&&rain>1800?'tropical rainforest':temperature>18?'tropical woodland and savanna':temperature<8?'boreal forest':'temperate woodland and grassland';
    return {biome,biomeEvidence:'Approximate descriptive biome inferred from visible renderer climate fields; not a replacement for the Koppen classification.',
      meanTemperatureC:Math.round(temperature*10)/10,annualRainMm:Math.round(rain),sampledTiles:samples.length};
  }
  root.TerrainIllustrationContext={summarize};
  if(typeof module!=='undefined'&&module.exports)module.exports={summarize};
})(typeof window==='undefined'?globalThis:window);

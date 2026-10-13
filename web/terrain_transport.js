// Both viewers use this boundary. The Python viewer retains ordinary fetch.
window.TerrainTransport={fetch:(...args)=>fetch(...args),release:(path,data={})=>navigator.sendBeacon(path,new Blob([JSON.stringify(data)],{type:'application/json'}))};

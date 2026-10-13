// Static file hosting only. There are deliberately no application/API routes.
import {createServer} from 'node:http';
import {createReadStream} from 'node:fs';
import {stat} from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
import {resolve,extname,sep} from 'node:path';
const root=resolve(fileURLToPath(new URL('../../wasm/dist/',import.meta.url)));
const mime={'.html':'text/html','.js':'text/javascript','.mjs':'text/javascript','.wasm':'application/wasm','.json':'application/json','.webp':'image/webp','.png':'image/png','.svg':'image/svg+xml','.css':'text/css','.safetensors':'application/octet-stream'};
const port=Number(process.env.PORT||8780);
createServer(async(request,response)=>{
  try {
    const url=new URL(request.url,'http://localhost'),path=resolve(root,'.'+decodeURIComponent(url.pathname==='/'?'/index.html':url.pathname));
    if(!path.startsWith(root+sep)){response.writeHead(403).end();return;}
    if(!['GET','HEAD'].includes(request.method)){response.writeHead(405).end();return;}
    const info=await stat(path);if(!info.isFile()){response.writeHead(404).end();return;}
    response.writeHead(200,{'Content-Type':mime[extname(path)]||'application/octet-stream','Content-Length':info.size,'Cache-Control':'no-cache'});
    if(request.method==='HEAD')response.end();else createReadStream(path).pipe(response);
  }catch {response.writeHead(404).end('Static asset not found');}
}).listen(port,'127.0.0.1',()=>console.log(`Neural Earth WASM: http://127.0.0.1:${port}`));

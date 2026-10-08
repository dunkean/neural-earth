// Synchronous framed pipe protocol; Python owns the CUDA context. No network.
import fs from 'node:fs';
import {createHash} from 'node:crypto';
const arrayIds=new WeakMap(), hashes=new Map();let nextId=1;
const stats={kernels:0,labels:{},unsupported:{}};
class Unsupported extends Error {}
const types={Float32Array:'float',Float64Array:'double',Int32Array:'int',Uint32Array:'unsigned int',Int8Array:'signed char',Uint8Array:'unsigned char',Int16Array:'short',Uint16Array:'unsigned short'};
const dtype={Float32Array:'<f4',Float64Array:'<f8',Int32Array:'<i4',Uint32Array:'<u4',Int8Array:'i1',Uint8Array:'u1',Int16Array:'<i2',Uint16Array:'<u2'};
function read(n){const b=Buffer.allocUnsafe(n);let at=0;while(at<n){const k=fs.readSync(0,b,at,n-at,null);if(!k)throw Error('CUDA bridge closed');at+=k;}return b;}
function write(b){let at=0;while(at<b.length)at+=fs.writeSync(1,b,at,b.length-at);}
function rpc(header,blocks=[]){
 const b=Buffer.from(JSON.stringify(header)),len=Buffer.alloc(4);len.writeUInt32LE(b.length);write(len);write(b);blocks.forEach(write);
 const response=JSON.parse(read(read(4).readUInt32LE()).toString());
 if(response.error)throw Error(response.error);
 for(const out of response.outputs||[]){const bytes=read(out.bytes),array=out.array;
  const target=header._targets?.get(array);if(!target)throw Error('Unknown CUDA output buffer');
  Buffer.from(target.buffer,target.byteOffset,target.byteLength).set(bytes);
  hashes.set(array,createHash('sha256').update(bytes).digest('hex'));
 }
 return response;
}
// Map and non-JSON target references stay on the caller, not on the wire.
function request(header,blocks,targets){
 const targetMap=targets;const b=Buffer.from(JSON.stringify(header)),len=Buffer.alloc(4);len.writeUInt32LE(b.length);write(len);write(b);blocks.forEach(write);
 const response=JSON.parse(read(read(4).readUInt32LE()).toString());if(response.error)throw Error(response.error);
 for(const out of response.outputs||[]){const bytes=read(out.bytes),target=targetMap.get(out.array);if(!target)throw Error('Unknown GPU output');Buffer.from(target.buffer,target.byteOffset,target.byteLength).set(bytes);hashes.set(out.array,createHash('sha256').update(bytes).digest('hex'));}
 return response;
}
function transportArray(array,transfer,targets,output=false){
 if(!arrayIds.has(array))arrayIds.set(array,nextId++);const id=arrayIds.get(array);
 const bytes=Buffer.from(array.buffer,array.byteOffset,array.byteLength),hash=createHash('sha256').update(bytes).digest('hex');
 const changed=hashes.get(id)!==hash;hashes.set(id,hash);
 transfer.push({id,dtype:dtype[array.constructor.name],length:array.length,bytes:changed?bytes.length:0,data:changed?bytes:null});
 if(output)targets.set(id,array);return id;
}
const HEADER=String.raw`
#define INFINITY ((double)__int_as_float(0x7f800000))
#define NAN ((double)__int_as_float(0x7fc00000))
template<class T> struct Ref {
 T* p; __device__ operator double() const {return (double)*p;}
 __device__ Ref& operator=(double x){*p=(T)x;return *this;}
 __device__ Ref& operator=(const Ref& x){*p=(T)(double)x;return *this;}
 __device__ Ref& operator+=(double x){return *this=(double)*p+x;}
 __device__ Ref& operator-=(double x){return *this=(double)*p-x;}
 __device__ Ref& operator*=(double x){return *this=(double)*p*x;}
 __device__ Ref& operator/=(double x){return *this=(double)*p/x;}
};
template<class T> struct View {
 T* p; int length;
 __device__ Ref<T> operator[](double i) const {return Ref<T>{p+(long long)i};}
 __device__ explicit operator bool() const {return p!=nullptr;}
};
template<class T,int N> struct Fixed {
 T p[N]; int length=N;
 __device__ T operator[](double i) const {return p[(int)i];}
};
template<class T> struct Ragged {
 View<T> data; View<int> offsets; int length;
 __device__ View<T> operator[](double i) const {
  int j=(int)i,a=(int)(double)offsets[j],b=(int)(double)offsets[j+1];return View<T>{data.p+a,b-a};
 }
};
struct NumericMap {View<double> items;
 __device__ double operator[](double i)const{return i>=0&&i<items.length?(double)items[i]:0.;}
};
__device__ double jmin(double a,double b){return fmin(a,b);}
__device__ double jmax(double a,double b){return fmax(a,b);}
__device__ double jround(double x){return floor(x+.5);}
__device__ unsigned int ju32(double x){if(!isfinite(x))return 0;double v=fmod(trunc(x),4294967296.);if(v<0)v+=4294967296.;return (unsigned int)v;}
__device__ int ji32(double x){return (int)ju32(x);}
__device__ double jor(double a,double b){return a&&!isnan(a)?a:b;}
__device__ double elevToHeightKm(double e){if(e<=0)return e*10.;double t=fmin(e,1.),s=1.-t;return 6.*t*t*t*t*(5.-4.*t)+t*s*s*s*s;}
__device__ double smoothstep(double a,double b,double x){if(a==b)return x>=b?1.:0.;double t=fmax(0.,fmin(1.,(x-a)/(b-a)));return t*t*(3.-2.*t);}
struct Itcz {View<float> lats;double start,step;
 __device__ double operator()(double lon) const {double n=lats.length,fi=fmod(fmod((lon-start)/step,n)+n,n);int i=(int)floor(fi),j=(i+1)%lats.length;double f=fi-i;return (double)lats[i]*(1.-f)+(double)lats[j]*f;}
};
struct Noise {View<unsigned char> perm;
 __device__ explicit operator bool()const{return perm.p!=nullptr;}
 __device__ double noise3D(double x,double y,double z) const {
  const int g[12][3]={{1,1,0},{-1,1,0},{1,-1,0},{-1,-1,0},{1,0,1},{-1,0,1},{1,0,-1},{-1,0,-1},{0,1,1},{0,-1,1},{0,1,-1},{0,-1,-1}};
  double s=(x+y+z)/3.;int i=(int)floor(x+s),j=(int)floor(y+s),k=(int)floor(z+s);double t=(i+j+k)/6.;double x0=x-i+t,y0=y-j+t,z0=z-k+t;
  int i1,j1,k1,i2,j2,k2;
  if(x0>=y0){if(y0>=z0){i1=1;j1=0;k1=0;i2=1;j2=1;k2=0;}else if(x0>=z0){i1=1;j1=0;k1=0;i2=1;j2=0;k2=1;}else{i1=0;j1=0;k1=1;i2=1;j2=0;k2=1;}}
  else{if(y0<z0){i1=0;j1=0;k1=1;i2=0;j2=1;k2=1;}else if(x0<z0){i1=0;j1=1;k1=0;i2=0;j2=1;k2=1;}else{i1=0;j1=1;k1=0;i2=1;j2=1;k2=0;}}
  double xx[4]={x0,x0-i1+1./6.,x0-i2+1./3.,x0-.5},yy[4]={y0,y0-j1+1./6.,y0-j2+1./3.,y0-.5},zz[4]={z0,z0-k1+1./6.,z0-k2+1./3.,z0-.5};
  int ii=i&255,jj=j&255,kk=k&255;int a[4]={0,i1,i2,1},b[4]={0,j1,j2,1},c[4]={0,k1,k2,1};double sum=0;
  for(int h=0;h<4;h++){double q=.6-xx[h]*xx[h]-yy[h]*yy[h]-zz[h]*zz[h];if(q>0){q*=q;int v=(int)(double)perm[ii+a[h]+(int)(double)perm[jj+b[h]+(int)(double)perm[kk+c[h]]]]%12;sum+=q*q*(g[v][0]*xx[h]+g[v][1]*yy[h]+g[v][2]*zz[h]);}}
  return 32.*sum;
 }
 __device__ double fbm(double x,double y,double z,double octaves=5,double persistence=2./3.) const {double sum=0,m=0,amp=1;for(int o=0;o<(int)octaves;o++){double f=1<<o;sum+=amp*noise3D(x*f,y*f,z*f);m+=amp;amp*=persistence;}return sum/m;}
 __device__ double ridgedFbm(double x,double y,double z,double octaves=6,double lacunarity=2,double gain=.5,double offset=1) const {double sum=0,f=1,a=1,prev=1,m=0;for(int o=0;o<(int)octaves;o++){double n=offset-fabs(noise3D(x*f,y*f,z*f));n*=n;sum+=n*a*prev;m+=a;prev=fmin(n,1.);f*=lacunarity;a*=gain;}return sum/m;}
};
`;
class Compiler {
 constructor(env,outputs){this.env=env;this.outputs=new Set(outputs);this.args=[];this.values=[];this.defs=[];this.locals=[];this.bindings=new Map();this.counter=0;this.synthetic=[];}
 argument(type,value){const name='arg'+this.args.length;this.args.push(type+' '+name);this.values.push(value);return name;}
 value(v,output=false){
  if(v===null||v===undefined)return {code:'View<float>{nullptr,0}',type:'View<float>'};
  if(typeof v==='number'||typeof v==='boolean'){return {code:this.argument('double',{scalar:Number(v)}),type:'double'};}
  if(typeof v==='string'){let code=0;for(let i=0;i<v.length;i++)code=(code*31+v.charCodeAt(i))>>>0;return {code:String(code)+'.',type:'double'};}
  if(typeof v==='function'&&v.gpuItcz){const a=this.value(v.gpuItcz.lats),start=this.value(v.gpuItcz.start),step=this.value(v.gpuItcz.step);return {code:`Itcz{${a.code},${start.code},${step.code}}`,type:'Itcz'};}
  if(v.perm&&typeof v.noise3D==='function'){const a=this.value(v.perm);return {code:`Noise{${a.code}}`,type:'Noise'};}
  if(v instanceof Set){const a=Int32Array.from(v);this.synthetic.push(a);const view=this.value(a);const type='Set'+this.counter++;this.defs.push(`struct ${type}{${view.type} items; __device__ bool has(double n)const{for(int i=0;i<items.length;i++)if((double)items[i]==n)return true;return false;}};`);return {code:`${type}{${view.code}}`,type};}
  if(ArrayBuffer.isView(v)){
   if(!types[v.constructor.name])throw new Unsupported('Array type '+v.constructor.name);
   const type=`View<${types[v.constructor.name]}>`,pointer=this.argument(types[v.constructor.name]+'*',{array:v,output}),length=this.argument('int',{scalar:v.length,integer:true});
   return {code:`${type}{${pointer},${length}}`,type};
  }
  if(Array.isArray(v)){
   if(v.every(x=>typeof x==='number'||typeof x==='boolean')){const a=Float64Array.from(v);this.synthetic.push(a);return this.value(a);}
   if(v.every(Array.isArray)&&v.every(x=>x.every(y=>typeof y==='number'))){const a=Float64Array.from(v.flat()),o=Int32Array.from([0,...v.map(x=>x.length)]);for(let i=1;i<o.length;i++)o[i]+=o[i-1];this.synthetic.push(a,o);const da=this.value(a),off=this.value(o);return {code:`Ragged<double>{${da.code},${off.code},${v.length}}`,type:'Ragged<double>'};}
   if(v.length&&v.every(x=>x&&typeof x==='object'))return this.records(v);
   throw new Unsupported('Empty/object array');
  }
  if(typeof v==='object'){
   const keys=Object.keys(v);
   if(!keys.length)return {code:'NumericMap{View<double>{nullptr,0}}',type:'NumericMap'};
   if(keys.length&&keys.every(k=>/^\d+$/.test(k))){const size=Math.max(...keys.map(Number))+1;if(size>1000000)throw new Unsupported('Sparse dictionary too large');
    if(keys.every(k=>typeof v[k]==='number')){const a=new Float64Array(size);for(const k of keys)a[k]=v[k];this.synthetic.push(a);const x=this.value(a);return {code:`NumericMap{${x.code}}`,type:'NumericMap'};}
    return this.records(Array.from({length:size},(_,i)=>v[i]||null));
   }
   const fields=keys.filter(k=>/^[A-Za-z_]\w*$/.test(k)&&typeof v[k]!=='function');
   if(fields.length>64)throw new Unsupported('Large control object');
   const vals=fields.map(k=>[k,this.value(v[k])]),type='Record'+this.counter++;
   this.defs.push(`struct ${type}{${vals.map(([k,x])=>x.type+' '+k+';').join('')} __device__ explicit operator bool()const{return true;}};`);
   return {code:`${type}{${vals.map(x=>x[1].code).join(',')}}`,type};
  }
  throw new Unsupported('Non-numeric captured value');
 }
 records(v){
  const sample=v.find(Boolean);if(!sample)throw new Unsupported('Empty records');
  const fields=Object.keys(sample).filter(k=>/^[A-Za-z_]\w*$/.test(k)&&typeof sample[k]!=='function');
  const columns=fields.map(k=>{let a=v.map(x=>x?.[k]??(Array.isArray(sample[k])?[]:0));return [k,this.value(a)];});
  const present=Uint8Array.from(v,x=>x?1:0);this.synthetic.push(present);const pv=this.value(present);
  const record='Item'+this.counter++,array='Records'+this.counter++;
  const fieldType=x=>x.type==='Ragged<double>'?'View<double>':'double';
  this.defs.push(`struct ${record}{${columns.map(([k,x])=>fieldType(x)+' '+k+';').join('')} double present;__device__ explicit operator bool()const{return present!=0;}};`);
  this.defs.push(`struct ${array}{${columns.map(([k,x])=>x.type+' '+k+';').join('')}${pv.type} present;int length;__device__ ${record} operator[](double i)const{return ${record}{${columns.map(([k])=>k+'[i]').join(',')},(double)present[i]};}};`);
  return {code:`${array}{${columns.map(x=>x[1].code).join(',')},${pv.code},${v.length}}`,type:array};
 }
 bind(name){if(this.bindings.has(name))return this.bindings.get(name);if(!(name in this.env))throw new Unsupported('Uncaptured '+name);
  let value=this.env[name];if(name==='climate'){value=Object.fromEntries([...this.climateFields].map(k=>[k,value[k]]));}
  const x=this.value(value,this.outputs.has(name));const local='env_'+name;this.locals.push(`auto ${local}=${x.code};`);this.bindings.set(name,local);return local;
 }
 expr(n,scope=new Set()){
  if(!n)throw new Unsupported('Missing expression');
  const e=x=>this.expr(x,scope);
  switch(n.type){
   case 'Identifier':if(scope.has(n.name))return n.name;if(n.name==='Infinity')return 'INFINITY';if(n.name==='NaN'||n.name==='undefined')return 'NAN';return this.bind(n.name);
   case 'Literal':if(typeof n.value==='string'){let v=0;for(const c of n.value)v=(v*31+c.charCodeAt(0))>>>0;return v+'.';}return n.value===null?'nullptr':typeof n.value==='boolean'?(n.value?'true':'false'):Number.isInteger(n.value)?n.value+'.':String(n.value);
   case 'MemberExpression':if(n.object.name==='Math'&&['PI','E'].includes(n.property.name))return String(Math[n.property.name]);return n.computed?`(${e(n.object)})[${e(n.property)}]`:`(${e(n.object)}).${n.property.name}`;
   case 'BinaryExpression':case 'LogicalExpression':{
    const a=e(n.left),b=e(n.right),op=n.operator;
    if(op==='**')return `pow((double)(${a}),(double)(${b}))`;
    if(op==='%')return `fmod((double)(${a}),(double)(${b}))`;
    if(['&','|','^'].includes(op))return `(double)(ji32(${a}) ${op} ji32(${b}))`;
    if(op==='>>>')return `(double)(ju32(${a}) >> (ju32(${b})&31))`;
    if(op==='>>')return `(double)(ji32(${a}) >> (ju32(${b})&31))`;
    if(op==='<<')return `(double)(int)(ju32(${a}) << (ju32(${b})&31))`;
    if(['===','!==','==','!='].includes(op)&&(n.left.name==='undefined'||n.right.name==='undefined'))return `${['!==','!='].includes(op)?'!':''}isnan((double)(${n.left.name==='undefined'?b:a}))`;
    if(op==='||'||op==='??')return `jor(${a},${b})`;
    return `(${a} ${op==='==='?'==':op==='!=='?'!=':op} ${b})`;
   }
   case 'UnaryExpression':return `(${n.operator}${e(n.argument)})`;
   case 'ConditionalExpression':return `(${e(n.test)}?${e(n.consequent)}:${e(n.alternate)})`;
   case 'AssignmentExpression':return `(${e(n.left)}${n.operator}${e(n.right)})`;
   case 'UpdateExpression':return n.prefix?`(${n.operator}${e(n.argument)})`:`(${e(n.argument)}${n.operator})`;
   case 'SequenceExpression':return '('+n.expressions.map(e).join(',')+')';
   case 'ArrayExpression':return `Fixed<double,${n.elements.length}>{{${n.elements.map(e).join(',')}}}`;
   case 'ObjectExpression':{
    const fields=n.properties.map(p=>[p.key.name,e(p.value)]),type='Return'+this.counter++;
    this.defs.push(`struct ${type}{${fields.map(([k])=>'double '+k+';').join('')}};`);return `${type}{${fields.map(x=>x[1]).join(',')}}`;
   }
   case 'CallExpression':{
    const args=n.arguments.map(e);
    if(n.callee.type==='MemberExpression'&&n.callee.object.name==='Math'){
     const name={max:'jmax',min:'jmin',abs:'fabs',round:'jround'}[n.callee.property.name]||n.callee.property.name;
     if(!['jmax','jmin','fabs','jround','pow','exp','sqrt','sin','cos','tan','tanh','asin','acos','atan2','floor','ceil','log','sign'].includes(name))throw new Unsupported('Math.'+name);
     if(name==='sign')return `(${args[0]}<0?-1.:(${args[0]}>0?1.:0.))`;
     return `${name}(${args.map(a=>'(double)('+a+')').join(',')})`;
    }
    if(n.callee.type==='Identifier'&&this.helpers[n.callee.name]){
     const f=this.helpers[n.callee.name],inner=new Set(scope),params=[];
     f.params.forEach((p,i)=>{const id=p.type==='AssignmentPattern'?p.left.name:p.name;inner.add(id);params.push('auto '+id);if(i>=args.length){if(p.type!=='AssignmentPattern')throw new Unsupported('Missing helper argument');args.push(e(p.right));}});
     const body=this.stmt(f.body,inner,1);return `([&](${params.join(',')}) ${body})(${args.join(',')})`;
    }
    if(n.callee.type==='Identifier'&&['elevToHeightKm','smoothstep'].includes(n.callee.name))return `${n.callee.name}(${args.join(',')})`;
    return `${e(n.callee)}(${args.join(',')})`;
   }
   default:throw new Unsupported(n.type);
  }
 }
 stmt(n,scope=new Set(),depth=0){
  const e=x=>this.expr(x,scope),s=x=>this.stmt(x,scope,depth);
  switch(n.type){
   case 'BlockStatement':{const local=new Set(scope);return '{\n'+n.body.map(x=>this.stmt(x,local,depth)).join('\n')+'\n}';}
   case 'ExpressionStatement':return e(n.expression)+';';
   case 'VariableDeclaration':return n.declarations.map(d=>{
    if(d.id.type==='ObjectPattern'){const tmp='tmp'+this.counter++,code=e(d.init);const fields=d.id.properties.map(p=>{scope.add(p.value.name);return `auto ${p.value.name}=${tmp}.${p.key.name};`;});return `auto ${tmp}=${code};`+fields.join('');}
    if(d.id.type!=='Identifier')throw new Unsupported('Destructuring');
    const name=d.id.name,init=d.init?e(d.init):'0.';scope.add(name);
    // Primitive locals have JS double semantics. Array/record helper results are auto.
    const obj=d.init && (d.init.type==='ArrayExpression'||d.init.type==='ObjectExpression'||
      (d.init.type==='CallExpression'&&['plateVelocityAt','heuristicWind'].includes(d.init.callee.name))||
      (d.init.type==='MemberExpression'&&d.init.computed&&['plateVec','domes'].includes(d.init.object.name)));
    return `${obj?'auto':'double'} ${name}=${init};`;
   }).join('\n');
   case 'IfStatement':return `if(${e(n.test)})${s(n.consequent)}${n.alternate?'else '+s(n.alternate):''}`;
   case 'ForStatement':{const local=new Set(scope),init=n.init?this.stmt(n.init,local,depth).replace(/;\s*double /g,',').replace(/;\s*$/,''):'',test=n.test?this.expr(n.test,local):'',update=n.update?this.expr(n.update,local):'';return `for(${init};${test};${update})${this.stmt(n.body,local,depth+1)}`;}
   case 'WhileStatement':return `while(${e(n.test)})${this.stmt(n.body,scope,depth+1)}`;
   case 'BreakStatement':return 'break;';
   case 'ContinueStatement':return depth?'continue;':'return;';
   case 'ReturnStatement':return `return ${n.argument?e(n.argument):''};`;
   case 'EmptyStatement':return ';';
   default:throw new Unsupported(n.type);
  }
 }
 compile(plan){this.helpers=plan.helpers;this.climateFields=new Set();
  const visit=n=>{if(!n?.type)return;if(n.type==='MemberExpression'&&n.object.name==='climate'&&!n.computed)this.climateFields.add(n.property.name);for(const v of Object.values(n))if(Array.isArray(v))v.forEach(visit);else if(v?.type)visit(v);};visit(plan.body);Object.values(plan.helpers).forEach(visit);
  const index=plan.index||'r',scope=new Set([index]);const bound=this.expr(plan.bound,scope),body=this.stmt(plan.body,scope);
  for(const name of plan.batchNames||[])this.bind(name);
  return HEADER+'\n'+this.defs.join('\n')+`\nextern "C" __global__ void run(${this.args.join(',')}) {\n${this.locals.join('\n')}\nint ${index}=blockIdx.x*blockDim.x+threadIdx.x;if(${index}>=${bound})return;\n${body}\n}`;
 }
}
export function install(){
 console.log=(...args)=>console.error(...args);
 globalThis.OROGEN_GPU_LOOP=(plan,env,batch=null)=>{
  const bound=plan.bound.type==='Identifier'?env[plan.bound.name]:null;
  if(bound===0)return true;
  // Zero plate motion creates no hotspot domes. The original loop then does
  // nothing; avoid compiling an empty record list as a numeric CUDA view.
  if(Array.isArray(env.domes)&&env.domes.length===0)return true;
  if(batch?.iterations===0)return true;
  if(batch && (!Number.isInteger(batch.iterations)||batch.iterations<0||batch.iterations>4096))throw Error('Invalid GPU iteration count');
  if(batch)plan.outputs=[...new Set([...plan.outputs,...plan.batchNames])];
  let compiler,source;try{compiler=new Compiler(env,plan.outputs);source=compiler.compile(plan);}catch(error){if(error instanceof Unsupported){stats.unsupported[plan.label]=error.message;return false;}throw error;}
  const transfer=[],targets=new Map(),args=compiler.values.map(x=>x.array?{array:transportArray(x.array,transfer,targets,x.output)}:x);
  const trace=process.env.OROGEN_GPU_TRACE==='1';
  const inputDigest=trace?createHash('sha256').update(compiler.values.map(x=>x.array?hashes.get(arrayIds.get(x.array)):JSON.stringify(x)).join('|')).digest('hex'):null;
  const batchWire=batch?{iterations:batch.iterations,before:batch.before.map(pair=>pair.map(k=>arrayIds.get(env[k]))),after:batch.after.map(pair=>pair.map(k=>arrayIds.get(env[k]))),swaps:batch.swaps.map(pair=>pair.map(k=>arrayIds.get(env[k])))}:null;
  const response=request({cmd:'kernel',label:plan.label,group:plan.group,source,args,arrays:transfer.map(({data,...x})=>x),outputs:[...targets.keys()],bound:bound??env.numRegions??env.N??env.n,batch:batchWire},transfer.filter(x=>x.data).map(x=>x.data),targets);
  const count=batch?.iterations||1;stats.kernels+=count;stats.labels[plan.label]=(stats.labels[plan.label]||0)+count;
  if(trace){stats.trace??=[];stats.trace.push({label:plan.label,input:inputDigest,output:createHash('sha256').update([...targets.keys()].map(k=>hashes.get(k)).join('|')).digest('hex')});}
  return true;
 };
 globalThis.OROGEN_GPU_SPECIAL=(kind,env)=>{
  const transfer=[],targets=new Map(),fields={};
  for(const [k,v]of Object.entries(env)){
   if(v instanceof Set){const mask=new Uint8Array(env.adjOffset.length-1);for(const i of v)mask[i]=1;fields[k]={array:transportArray(mask,transfer,targets)};}
   else if(ArrayBuffer.isView(v))fields[k]={array:transportArray(v,transfer,targets,['result','r_stress','r_subductFactor','r_elevation'].includes(k))};
   else fields[k]={scalar:v};
  }
  request({cmd:'special',kind,fields,arrays:transfer.map(({data,...x})=>x),outputs:[...targets.keys()]},transfer.filter(x=>x.data).map(x=>x.data),targets);
 };
 globalThis.OROGEN_GPU_EXTERNAL=()=>rpc({cmd:'externalErosion'});
 globalThis.OROGEN_GPU_FINISH=()=>rpc({cmd:'finish',stats});
}

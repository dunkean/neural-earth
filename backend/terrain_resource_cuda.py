"""CUDA geology, mirroring terrain_resource_gpu's float32/uint32 recipe.

Seed-independent graph replay is prepared at startup for the 4,000 km authority;
other shapes use eager kernels. No torch.compile or world-dependent warm-up.
Ambiguous argmax decisions use the independent NumPy evaluator.
"""
import math
import threading
import numpy as np
import torch
import terrain_resource_gpu as G

_stream=torch.cuda.Stream(priority=-1)
_graph=None
_graph_lock=threading.Lock()


def _f(value):
    return float(np.float32(value))


def _hash(x):
    x=x & 0xffffffff
    x=((x ^ (x >> 16))*0x7feb352d) & 0xffffffff
    x=((x ^ (x >> 15))*0x846ca68b) & 0xffffffff
    return x ^ (x >> 16)


def _salt(seed,layer,octave,component=0):
    if not isinstance(seed,torch.Tensor):return G.salt_of(seed,layer,octave,component)
    return _hash(seed+layer*0x632be5ab+(octave+64)*0x27d4eb2d+component*0x165667b1)


def _constants(values,device,dtype=torch.float32):
    # Scalar fills are capture-safe; no host-to-device copies inside a graph.
    return torch.stack([torch.full((),v,dtype=dtype,device=device) for v in values])


def _quad(ix,iy,ux,uy,salt):
    a0=(ix*0x9e3779b1) & 0xffffffff
    a1=((ix+1)*0x9e3779b1) & 0xffffffff
    b0=_hash(iy+0x85ebca77);b1=_hash(iy+1+0x85ebca77)
    def unit(x): return ((_hash(x) >> 8).to(torch.float32)*_f(1/16777216.))*2-1
    n00,n10=unit(a0 ^ b0 ^ salt),unit(a1 ^ b0 ^ salt)
    n01,n11=unit(a0 ^ b1 ^ salt),unit(a1 ^ b1 ^ salt)
    a=n00*(1-ux)+n10*ux;b=n01*(1-ux)+n11*ux
    return a*(1-uy)+b*uy


def _lattice(g,lod,octave):
    k=octave-lod
    if k <= 0: return g,torch.full_like(g,.5,dtype=torch.float32)
    return g >> k,((g & ((1 << k)-1)).to(torch.float32)+.5)*_f(1/(1 << k))


def _fade(t): return t*t*t*(t*(t*6-15)+10)


def _octave(gi,gj,lod,octave,salt,wx,wy,stretch=0):
    ix,fx=_lattice(gi,lod,octave+stretch);iy,fy=_lattice(gj,lod,octave)
    qx=fx+wx/_f(30.*2**(octave+stretch));qy=fy+wy/_f(30.*2**octave)
    flx,fly=torch.floor(qx),torch.floor(qy)
    return _quad(ix+flx.to(torch.int64),iy+fly.to(torch.int64),_fade(qx-flx),_fade(qy-fly),salt)


def _noise_batch(gi,gj,lod,seed,specs,wx,wy,theta,aligned):
    """Batch all rock/group channels and all orientations in each launch."""
    if aligned:
        directions=G.DIRECTIONS
        a=torch.stack([c0*gi+c1*gj for c0,c1,_,_,_ in directions])
        b=torch.stack([c2*gi+c3*gj+c4 for _,_,c2,c3,c4 in directions])
        wa=torch.stack([_f(c0)*wx+_f(c1)*wy for c0,c1,_,_,_ in directions])
        wb=torch.stack([_f(c2)*wx+_f(c3)*wy for _,_,c2,c3,_ in directions])
        weights=torch.stack([torch.clamp(torch.cos(2*(theta-_f(phi))),min=0) for phi in G.PHI])
    else:
        a,b,wa,wb=gi[None],gj[None],wx[None],wy[None]
        weights=torch.ones_like(wa)
    total=torch.zeros((len(specs),len(a),len(gi)),device=gi.device,dtype=torch.float32)
    for o in range(lod+2,16):
        amplitudes=[_f(2.**(o-hi)) if lo<=o<=hi else 0. for _,lo,hi in specs]
        if not any(amplitudes): continue
        if isinstance(seed,torch.Tensor):
            salts=torch.stack([torch.stack([_salt(seed,salt,o,3+d if aligned else 0) for d in range(len(a))])
                               for salt,_,_ in specs])[:,:,None]
        else:
            salts=torch.tensor([[_salt(seed,salt,o,3+d if aligned else 0) for d in range(len(a))]
                                for salt,_,_ in specs],dtype=torch.int64,device=gi.device)[:,:,None]
        values=_octave(a[None],b[None],lod,o,salts,wa[None],wb[None],1 if aligned else 0)
        amp=(_constants(amplitudes,gi.device) if isinstance(seed,torch.Tensor) else
             torch.tensor(amplitudes,dtype=torch.float32,device=gi.device))[:,None,None]
        total=total+amp*values
    norm=[_f(max(math.sqrt(sum(_f(2.**(o-hi))**2 for o in range(max(lo,lod+2),hi+1))),1e-6)*.45)
          for _,lo,hi in specs]
    normalizer=(_constants(norm,gi.device) if isinstance(seed,torch.Tensor) else
                torch.tensor(norm,dtype=torch.float32,device=gi.device))
    total=total/normalizer[:,None,None]
    if not aligned: return total[:,0]
    # Same four-direction addition order as NumPy, with no parallel reduction.
    out=torch.zeros_like(total[:,0]);weight2=torch.zeros_like(weights[0])
    for d in range(4):
        out=out+weights[d]*total[:,d]
        weight2=weight2+weights[d]*weights[d]
    return out/torch.sqrt(torch.clamp(weight2,min=_f(1e-6)))


def scores(pkg,gi,gj,lod,*,data=None,bounds=None,seed=None):
    """Return all class scores, not just argmax, to expose numerical margins."""
    seed=pkg['seed32'] if seed is None else seed
    layer,field=pkg['layer'],pkg['field']
    device='cuda'
    if not isinstance(gi,torch.Tensor):gi=torch.as_tensor(np.ascontiguousarray(gi),dtype=torch.int64,device=device)
    if not isinstance(gj,torch.Tensor):gj=torch.as_tensor(np.ascontiguousarray(gj),dtype=torch.int64,device=device)
    gi,gj=torch.broadcast_tensors(gi,gj)
    shape=gi.shape
    gi,gj=gi.reshape(-1),gj.reshape(-1)
    if data is None:data=torch.as_tensor(field.data,device=device)
    if bounds is None and field.bounds is not None:bounds=torch.as_tensor(field.bounds,device=device)
    def seam():
        if bounds is None: return torch.ones_like(gi,dtype=torch.float32)
        x=(gi.to(torch.float32)+.5)*_f(2.**(lod-10))
        width=bounds[2]-bounds[0]
        x=(x-bounds[0]) % width
        t=torch.clamp(torch.minimum(x,width-x)/2,0,1)
        return t*t*(3-2*t)
    edge=seam()
    zero=torch.zeros_like(gi,dtype=torch.float32)
    wx,wy=zero,zero
    for o in range(max(lod+1,lod+2),12):
        # zone_warp has no stretched component; zero warp for its own noise.
        amplitude=_f(.15)*_f(30.*2**o)
        wx=wx+amplitude*_octave(gi,gj,lod,o,_salt(seed,layer*4+1,o,1),zero,zero)
        wy=wy+amplitude*_octave(gi,gj,lod,o,_salt(seed,layer*4+1,o,2),zero,zero)
    wx,wy=wx*edge,wy*edge
    def axis(g,warp,axis):
        k=10-lod
        if k >= 0:
            base=g >> k
            frac=((g & ((1 << k)-1)).to(torch.float32)+.5)*_f(1/(1 << k))-.5
        else:
            pos=(g.to(torch.float32)+.5)*_f(2.**(-k))-.5
            base=torch.floor(pos).to(torch.int64);frac=pos-torch.floor(pos)
        q=frac+warp*_f(1/G.AUTH_SIZE)
        fl=torch.floor(q);c=base+fl.to(torch.int64);fraction=q-fl
        if bounds is None: return c,fraction
        lo,hi=bounds[axis],bounds[axis+2]
        first,last=(field.i0,field.i0+field.w-1) if axis == 0 else (field.j0,field.j0+field.h-1)
        a,b=(_f(first+1)+lo)*.5,(_f(last)+hi)*.5
        x=c.to(torch.float32)+fraction+.5
        x=(x-lo)%(hi-lo)+lo if axis == 0 else torch.clamp(x,min=a,max=b)
        c=torch.clamp(torch.floor(x-.5).to(torch.int64),first,last-1)
        left=torch.where(c==first,a,c.to(torch.float32)+.5)
        right=torch.where(c+1==last,b,c.to(torch.float32)+1.5)
        fraction=(x-left)/(right-left)
        if axis == 0:
            at=(x<a)|(x>=b)
            fraction=torch.where(at,(torch.where(x<a,x+hi-lo,x)-b)/(a+hi-lo-b),fraction)
            c=torch.where(at,last,c)
        return c,fraction
    ci,fx=axis(gi,wx,0);cj,fy=axis(gj,wy,1)
    c0,c1=(ci-field.i0)%field.w,(ci+1-field.i0)%field.w
    r0,r1=torch.clamp(cj-field.j0,0,field.h-1),torch.clamp(cj+1-field.j0,0,field.h-1)
    def bilinear(k):
        a=data[k,r0,c0]*(1-fx)+data[k,r0,c1]*fx
        b=data[k,r1,c0]*(1-fx)+data[k,r1,c1]*fx
        return a*(1-fy)+b*fy
    theta=.5*torch.atan2(bilinear(pkg['classes']+1),bilinear(pkg['classes']))
    specs=[(layer*64+48+g,12,15) for g in range(len(pkg['group_aligned']))]
    specs += [(layer*64+32+k,11,14) for k in range(pkg['classes'])]
    flags=list(pkg['group_aligned'])+list(pkg['aligned'])
    noises={}
    for aligned in (False,True):
        indices=[k for k,flag in enumerate(flags) if flag == aligned]
        batch=_noise_batch(gi,gj,lod,seed,[specs[k] for k in indices],wx,wy,theta,aligned)
        noises.update({k:batch[n]*edge for n,k in enumerate(indices)})
    output=[]
    b1,b2=(_f(x) for x in pkg.get('beta',G.REALIZED_BETA))
    for k in range(pkg['classes']):
        noise=noises[len(pkg['group_aligned'])+k]
        output.append(bilinear(k)+b1*noises[pkg['groups'][k]]+b2*noise)
    return torch.stack(output).reshape((pkg['classes'],)+shape)


def _graph_key(pkg,lod):
    f=pkg['field']
    return (f.data.shape,f.i0,f.j0,lod,tuple(f.bounds) if f.bounds is not None else None,pkg['layer'],
            tuple(pkg['aligned']),tuple(pkg['groups']),tuple(pkg['group_aligned']),tuple(pkg.get('beta',G.REALIZED_BETA)))


class _Replay:
    def __init__(self,pkg,lod,capacity=196608):
        self.key=_graph_key(pkg,lod);self.capacity=capacity
        with torch.cuda.stream(_stream),torch.inference_mode():
            self.gi=torch.zeros(capacity,dtype=torch.int64,device='cuda')
            self.gj=torch.zeros_like(self.gi)
            self.data=torch.as_tensor(pkg['field'].data,device='cuda').clone()
            self.bounds=torch.as_tensor(pkg['field'].bounds,device='cuda')
            self.seed=torch.zeros((),dtype=torch.int64,device='cuda')
            def evaluate():
                value=scores(pkg,self.gi,self.gj,lod,data=self.data,bounds=self.bounds,seed=self.seed)
                return torch.topk(value,2,dim=0)
            evaluate();evaluate()
            _stream.synchronize()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=_stream):
                self.top=evaluate()

    def evaluate(self,pkg,gi,gj):
        a,b=np.broadcast_arrays(gi,gj);n=a.size
        self.data.copy_(torch.from_numpy(pkg['field'].data))
        self.seed.fill_(int(pkg['seed32']))
        self.gi[:n].copy_(torch.from_numpy(np.ascontiguousarray(a.reshape(-1))))
        self.gj[:n].copy_(torch.from_numpy(np.ascontiguousarray(b.reshape(-1))))
        if n<self.capacity:
            self.gi[n:].fill_(0);self.gj[n:].fill_(0)
        self.graph.replay()
        out=self.top.indices[0,:n].cpu().numpy().astype(np.int32).reshape(a.shape)
        margin=(self.top.values[0,:n]-self.top.values[1,:n]).cpu().numpy().reshape(a.shape)
        return out,margin<=5e-4


def prepare_graph():
    global _graph
    import terrain_resources as R
    bounds=(-np.pi*2e6,-np.pi*1e6,np.pi*2e6,np.pi*1e6)
    grid=R.Lattice(10,bounds=bounds)
    pkg=dict(seed32=0,layer=R.ROCK_LAYER,classes=len(R.ROCKS),aligned=R.ROCK_ALIGNED,groups=R.ROCK_GROUPS,
             group_aligned=R.ROCK_GROUP_ALIGNED,field=G.Field(np.zeros((len(R.ROCKS)+2,)+grid.shape,np.float32),
                                                          grid.i[0],grid.j[0],bounds))
    _graph=_Replay(pkg,9)


def realized_class(pkg,gi,gj,lod):
    # Independent data and a priority stream keep queued tile kernels from
    # sitting ahead of this small authority evaluation on the default stream.
    with _graph_lock,torch.cuda.stream(_stream),torch.inference_mode():
        if _graph is not None and _graph.key==_graph_key(pkg,lod) and np.broadcast(gi,gj).size<=_graph.capacity:
            out,ambiguous=_graph.evaluate(pkg,gi,gj)
        else:
            values=scores(pkg,gi,gj,lod)
            top=torch.topk(values,2,dim=0)
            out=top.indices[0].cpu().numpy().astype(np.int32)
            ambiguous=(top.values[0]-top.values[1] <= 5e-4).cpu().numpy()
        if ambiguous.any():
            a,b=np.broadcast_arrays(gi,gj)
            out[ambiguous]=G.realized_class(pkg,a[ambiguous],b[ambiguous],lod)
        return out

"""Batched deposit authority. Arrays are canonical; dictionaries are an API view.

No interpreter loop over sites in compilation, placement, or stock summation.
Record conversion is lazy and happens only at a legacy JSON/inspection boundary.
"""
import hashlib
import json
import numpy as np
from scipy.special import ndtri


class DepositTable:
    def __init__(self, arrays, families, metals):
        self.arrays = arrays
        self.families, self.metals = tuple(families), tuple(metals)
        self._records = None

    def __len__(self):
        return len(self.arrays['owner'])

    def fingerprint(self):
        digest = hashlib.sha256(json.dumps([self.families,self.metals]).encode())
        for name,value in sorted(self.arrays.items()):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(value).tobytes())
        return digest.hexdigest()

    def __eq__(self, other):
        if isinstance(other,DepositTable):
            return self.families == other.families and self.metals == other.metals and self.fingerprint() == other.fingerprint()
        return self.records() == other

    def records(self):
        if self._records is None:
            a=self.arrays
            result=[]
            for n in range(len(self)):
                family=self.families[int(a['family'][n])]
                contained={m:int(a['contained'][n,k]) for k,m in enumerate(self.metals) if a['present'][int(a['family'][n]),k]}
                recoverable={m:int(a['recoverable'][n,k]) for k,m in enumerate(self.metals) if m in contained}
                d=dict(id=f"{family}:{int(a['source'][n]):016x}:{int(a['source_index'][n])}",family=family,
                    source_cell=int(a['source'][n]),owner_cell=int(a['owner'][n]),index=int(a['index'][n]),
                    position_cell=int(a['position'][n]),position_lod=int(a['lod'][n]),position_i=int(a['i'][n]),position_j=int(a['j'][n]),
                    ore_t=int(a['ore'][n]),grade_factor=float(a['grade'][n]),depth_top_m=float(a['depth'][n]),
                    contained=contained,contained_units={m:('kg' if m in ('gold','silver','star_metal') else 't') for m in contained},
                    medieval_accessible_share=float(a['access'][n]),medieval_recoverable=recoverable,
                    outcrop=bool(a['outcrop'][n]),gossan=bool(a['gossan'][n]),status='synthetic' if a['fantasy'][n] else 'derived')
                if a['district'][n]:
                    d.update(district_id=f"{family}:{int(a['district'][n]):016x}:0",
                             district_center_m=a['district_xy'][n].tolist())
                result.append(d)
            result.sort(key=lambda d:d['id'])
            self._records=result
        return self._records

    def __iter__(self):
        return iter(self.records())

    def __getitem__(self, index):
        return self.records()[index]


def _route_keys(R,seed,domain,cids,indices):
    """Original addressed FNV route, vectorized over the decimal address bytes."""
    h=np.full(len(cids),R.FNV_OFFSET ^ R.splitmix(int(seed) & R.MASK),np.uint64)
    with np.errstate(over='ignore'):
        for part in domain+('district-route',):
            for byte in part.encode(): h=(h ^ np.uint64(byte))*np.uint64(R.FNV_PRIME)
            h=(h ^ np.uint64(255))*np.uint64(R.FNV_PRIME)
        for values in (cids,indices):
            chars=values.astype('U20').view(np.uint32).reshape(-1,20)
            for col in range(20):
                byte=chars[:,col].astype(np.uint64)
                h=np.where(byte != 0,(h ^ byte)*np.uint64(R.FNV_PRIME),h)
            h=(h ^ np.uint64(255))*np.uint64(R.FNV_PRIME)
    return R.splitmix_np(h-np.uint64(0))


def display_addresses(seed,table):
    """Original record ordering and LOD-0 keyed offsets, without record objects."""
    import terrain_resources as R
    a=table.arrays
    indices=a['source_index'].astype(str)
    order=np.lexsort((indices,a['source'],a['family']))
    families=np.asarray(table.families)[a['family'][order]]
    ids=np.char.add(np.char.add(np.char.add(families,':'),np.char.mod('%016x',a['source'][order])),':')
    ids=np.char.add(ids,indices[order])
    chars=ids.view(np.uint32).reshape(len(ids),-1)
    h=np.full(len(ids),R.FNV_OFFSET ^ R.splitmix(seed & R.MASK),np.uint64)
    with np.errstate(over='ignore'):
        for part in ('resources','deposit-lod0'):
            for byte in part.encode():h=(h ^ np.uint64(byte))*np.uint64(R.FNV_PRIME)
            h=(h ^ np.uint64(255))*np.uint64(R.FNV_PRIME)
        for col in range(chars.shape[1]):
            byte=chars[:,col].astype(np.uint64)
            h=np.where(byte != 0,(h ^ byte)*np.uint64(R.FNV_PRIME),h)
        h=(h ^ np.uint64(255))*np.uint64(R.FNV_PRIME)
    return order,R.splitmix_np(h)


def _expanded(count):
    flat=count.ravel()
    cells=np.repeat(np.flatnonzero(flat),flat[flat>0])
    starts=np.cumsum(flat)-flat
    return cells,np.arange(len(cells),dtype=np.int64)-starts[cells]


def compile_deposits(grid,f,seed,fantasy=False):
    import terrain_resources as R
    specs={**R.FAMILIES,**R.PLACERS,**(R.FANTASY if fantasy else {})}
    names=tuple(sorted(specs))
    metals=tuple(sorted({m for spec in specs.values() for m in spec['metals']}))
    present=np.array([[m in specs[name]['metals'] for m in metals] for name in names],bool)
    prosp=R.family_prospectivities(grid,f,seed,fantasy)
    f.update({'prospect_'+name:p.astype(np.float32) for name,p in prosp.items()})
    xx,yy=np.meshgrid(grid.x,grid.y)
    parts=[]
    for family,name in enumerate(names):
        p=prosp[name];spec=specs[name]
        domain=('resources','fantasy' if name in R.FANTASY else 'deposit',name)
        lam=spec['density']*np.clip(p,0,1)*grid.area_m2/1e10
        live=lam>0
        count=np.zeros(grid.shape,np.int64)
        count[live]=R._poisson(lam[live],R.to_unit(R.cell_keys(seed,domain+('count',f'lod{grid.lod}'),grid.ids[live])))
        src,sk=_expanded(count)
        if not len(src): continue
        ids=grid.ids.ravel()[src]
        centres=None
        if name in R.DISTRICT_SHAPE and grid.lod == R.WORLD_LOD:
            centres=R.district_centres(seed,name,grid.lod,grid.bounds)
            live=p>=max(float(p.max())*.15,1e-10)
            intensity=np.zeros(grid.shape)
            intensity[live]=R.district_intensity(xx[live],yy[live],name,centres,f['structural_strike'],grid,
                                                widen=grid.size/np.sqrt(12))
            count=R.apportion(len(src),R.quantize((lam*intensity*live).ravel()),grid.ids.ravel()).reshape(grid.shape)
            order=np.argsort(_route_keys(R,seed,domain,ids,sk),kind='stable')
            ids,src,sk=ids[order],src[order],sk[order]
        target,index=_expanded(count)
        def draw(what):
            return R.to_unit(R.splitmix_np(R.splitmix_np(np.uint64(R.key(seed,domain+(what,f'lod{grid.lod}'))) ^ ids) ^ (sk.astype(np.uint64)+np.uint64(1))))
        median,sigma=spec['tonnage']
        ore=median*10.**(sigma*ndtri(np.clip(draw('tonnage'),1e-9,1-1e-9)))
        grade=np.minimum(10.**(.25*ndtri(np.clip(draw('grade'),1e-9,1-1e-9))),1./sum(spec['metals'].values()))
        depth=np.round(spec['depth']*10.**(.4*ndtri(np.clip(draw('depth'),1e-9,1-1e-9))),1)
        ore_int=np.rint(ore).astype(np.int64)
        contained=np.stack([np.rint(ore*spec['metals'].get(m,0)*grade*(1000 if m in R.GRAM_METALS else 1)).astype(np.int64) for m in metals],axis=1)
        access=np.clip((40+.5*f['rugged_m'].ravel()[target]-depth)/np.maximum(20,(ore_int/2.7)**(1/3)),0,1)
        recoverable=(contained*access[:,None]*np.array([R.MEDIEVAL_RECOVERY.get(m,.5) for m in metals])).astype(np.int64)
        plod=R.placement_lod(grid.lod);n=1 << (grid.lod-plod);child=R.cell_size(plod)
        ci=grid.i[target % len(grid.i)];cj=grid.j[target // len(grid.i)]
        lo_i=np.maximum(ci*n,np.ceil(grid.bounds[0]/child-.5).astype(np.int64))
        lo_j=np.maximum(cj*n,np.ceil(grid.bounds[1]/child-.5).astype(np.int64))
        hi_i=np.minimum((ci+1)*n,np.ceil(grid.bounds[2]/child-.5).astype(np.int64))
        hi_j=np.minimum((cj+1)*n,np.ceil(grid.bounds[3]/child-.5).astype(np.int64))
        pi=lo_i+(draw('site-x')*(hi_i-lo_i)).astype(np.int64)
        pj=lo_j+(draw('site-y')*(hi_j-lo_j)).astype(np.int64)
        district=np.zeros(len(src),np.uint64);dxy=np.zeros((len(src),2))
        if centres is not None:
            cx,cy,bi0,bj0,active=centres
            bx=np.floor(xx.ravel()[target]/R.cell_size(R.DISTRICT_LOD)).astype(np.int64)
            by=np.floor(yy.ravel()[target]/R.cell_size(R.DISTRICT_LOD)).astype(np.int64)
            best=np.full(len(src),np.inf)
            width=grid.bounds[2]-grid.bounds[0]
            sa,sc=R.DISTRICT_SHAPE[name]
            for oy in (-1,0,1):
                for ox in (-1,0,1):
                    col=(bx+ox-bi0) % cx.shape[1];row=np.clip(by+oy-bj0,0,cx.shape[0]-1)
                    valid=(by+oy-bj0>=0)&(by+oy-bj0<cx.shape[0])&active[row,col,0]
                    px,py=cx[row,col,0],cy[row,col,0]
                    theta=R.strike_at(f['structural_strike'],grid,px,py)
                    dx=(px-(pi+.5)*child+width/2)%width-width/2;dy=py-(pj+.5)*child
                    q=((dx*np.cos(theta)+dy*np.sin(theta))/sa)**2+((-dx*np.sin(theta)+dy*np.cos(theta))/sc)**2
                    win=valid&(q<best)
                    best=np.where(win,q,best)
                    district=np.where(win,R.cell_id(R.DISTRICT_LOD,bi0+col,bj0+row),district)
                    dxy[:,0]=np.where(win,px,dxy[:,0]);dxy[:,1]=np.where(win,py,dxy[:,1])
            missing=np.flatnonzero(district == 0)
            if len(missing):
                rows,cols=np.nonzero(active[...,0])
                px,py=cx[rows,cols,0],cy[rows,cols,0]
                theta=R.strike_at(f['structural_strike'],grid,px,py)
                for start in range(0,len(missing),128):
                    part=missing[start:start+128]
                    dx=(px[None,:]-(pi[part,None]+.5)*child+width/2)%width-width/2
                    dy=py[None,:]-(pj[part,None]+.5)*child
                    q=((dx*np.cos(theta)+dy*np.sin(theta))/sa)**2+((-dx*np.sin(theta)+dy*np.cos(theta))/sc)**2
                    nearest=np.argmin(q,axis=1)
                    district[part]=R.cell_id(R.DISTRICT_LOD,bi0+cols[nearest],bj0+rows[nearest])
                    dxy[part,0]=px[nearest];dxy[part,1]=py[nearest]
        outcrop=depth<5
        parts.append(dict(family=np.full(len(src),family,np.int16),source=ids,source_index=sk,
            owner=grid.ids.ravel()[target],index=index,position=R.cell_id(plod,pi,pj),lod=np.full(len(src),plod,np.int8),
            i=pi,j=pj,ore=ore_int,grade=np.round(grade,4),depth=depth,contained=contained,recoverable=recoverable,
            access=np.round(access,4),outcrop=outcrop,gossan=outcrop&bool(spec.get('sulfide'))&(f['temperature_c'].max(axis=0).ravel()[src]>0),
            fantasy=np.full(len(src),name in R.FANTASY,bool),district=district,district_xy=dxy))
    if not parts:
        return []
    arrays={k:np.concatenate([part[k] for part in parts]) for k in parts[0]}
    arrays['present']=present
    return DepositTable(arrays,names,metals)

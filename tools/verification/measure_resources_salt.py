"""Measure coastal salt in a completed columnar resource cache (no recompilation)."""
from pathlib import Path
import argparse
import json
import os
import sys
import numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools._bootstrap import activate
activate()
import terrain_resource_layers as L
import terrain_resources as R


def measure(grid, fields, sites):
    a=sites.arrays
    kinds=np.asarray(sites.families)[a['family']]
    annual=a['annual_kg']
    live=annual>0
    def sphere(i,j):
        lon,lat=(i+.5)*30/grid.radius_m,(j+.5)*30/grid.radius_m
        return grid.radius_m*np.stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)),axis=1)
    xyz=sphere(a['i'][live],a['j'][live])
    near=cKDTree(xyz).query(xyz,k=2)[0][:,1]/1000
    candidates=sphere(fields['salt_coast_i'],fields['salt_coast_j'])
    distance=cKDTree(xyz).query(candidates)[0]/1000
    result=dict(types={},annual_production_t=float(annual.sum()/1000),
        native_low_coast_candidates=len(candidates),coverage_within_25km=float(np.mean(distance<=25)),
        nearest_site_km=dict(zip(('min','p10','median','p90','max'),map(float,np.quantile(near,[0,.1,.5,.9,1])))),
        budget_kg=int(fields['coastal_salt_kg_year'].sum()),sites_kg=int(annual.sum()))
    for family in ('evaporite_salt','salt_spring','sea_salt_solar','sea_salt_boiled'):
        m=kinds==family
        result['types'][family]=dict(sites=int(m.sum()),annual_production_t=float(annual[m].sum()/1000),
            recoverable_stock_t=int(a['recoverable'][m,sites.metals.index('salt')].sum()))
    assert result['sites_kg']==result['budget_kg']
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--cache',required=True,help='Directory above resource-layers-v3')
    parser.add_argument('--key',default='benchmark')
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    os.environ['TERRAIN_RESOURCES_CACHE']=args.cache
    bounds=(-np.pi*2e6,-np.pi*1e6,np.pi*2e6,np.pi*1e6)
    g,f,d,*_=L._from_disk(args.key,bounds)
    result=measure(g,f,d)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

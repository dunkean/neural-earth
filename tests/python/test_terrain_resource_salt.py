"""Coastal flow semantics, physical spacing, climate and exact integer budgets."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools._bootstrap import activate
activate()
import unittest
import numpy as np
from scipy.spatial import cKDTree
import terrain_resources as R
from terrain_resource_salt import append_sites, coast_candidates
from terrain_resource_deposits import DepositTable


class CoastalSaltTests(unittest.TestCase):
    def fixture(self, warm=True, fuel=True):
        g = R.Lattice(10,bounds=(-245760.,-122880.,245760.,122880.))
        shape = g.shape
        # A gentle N/S shore, with dense candidates across multiple owners.
        height = np.tile(np.linspace(-40,40,256,dtype=np.float32),(128,1))
        f = dict(temperature_c=np.full((12,)+shape,22. if warm else 6.,np.float32),
            precipitation_mm=np.full((12,)+shape,10. if warm else 120.,np.float32),
            pet_mm=np.full((12,)+shape,80.,np.float32),v2_ice_month=np.zeros((12,)+shape),
            harvest_month=np.full(shape,7,np.uint8),pedology_clayey=np.full(shape,.6),
            pedology_sandy=np.full(shape,.2),wood_stock_m3=g.area_m2*(.02 if fuel else 0),
            peat_stock_m3=g.area_m2*(.1 if fuel else 0))
        coast_candidates(g,f,height)
        arrays={k:np.zeros(0,dtype) for k,dtype in dict(family=np.int16,source=np.uint64,source_index=np.int64,
            owner=np.uint64,index=np.int64,position=np.uint64,lod=np.int8,i=np.int64,j=np.int64,ore=np.int64,
            grade=float,depth=float,access=float,outcrop=bool,gossan=bool,fantasy=bool,district=np.uint64).items()}
        arrays.update(contained=np.zeros((0,1),np.int64),recoverable=np.zeros((0,1),np.int64),
                      district_xy=np.zeros((0,2)),present=np.ones((1,1),bool))
        return g,f,DepositTable(arrays,('salt_spring',),('salt',))

    def test_warm_dry_solar_and_cold_wet_boiling(self):
        for warm,family in ((True,'sea_salt_solar'),(False,'sea_salt_boiled')):
            g,f,base=self.fixture(warm)
            result=append_sites(base,g,f,42)
            self.assertGreater(len(result),10)
            self.assertGreater(sum(d['family']==family for d in result),.8*len(result))
            for d in result:
                self.assertEqual(d['quantity_kind'],'annual_flow')
                self.assertGreaterEqual(d['annual_production_t'],1)
                self.assertEqual(d['contained'],{})
                self.assertEqual(d['medieval_recoverable'],{})
                self.assertTrue(d['production_months'])
                self.assertTrue(all(1<=m<=12 for m in d['production_months']))
                if not warm:self.assertNotIn(8,d['production_months'])
                lod,i,j=R.decode_id(np.uint64(d['owner_cell']))
                self.assertEqual((d['position_i'] >> int(lod),d['position_j'] >> int(lod)),(int(i),int(j)))

    def test_fuel_and_labour_limit_boiling_but_not_solar(self):
        g,f,base=self.fixture(False,False)
        self.assertEqual(len(append_sites(base,g,f,42)),0)
        f['peat_stock_m3']=g.area_m2*.1
        self.assertGreater(len(append_sites(base,g,f,42)),0)
        f['temperature_c'][:]=-10
        self.assertEqual(len(append_sites(base,g,f,42)),0)
        g,f,base=self.fixture(True,False)
        self.assertGreater(len(append_sites(base,g,f,42)),0)

    def test_order_independence_spacing_and_exact_budgets(self):
        g,f,base=self.fixture(False)
        first=append_sites(base,g,f,42)
        for name in ('i','j','flat'):f['salt_coast_'+name]=f['salt_coast_'+name][::-1]
        second=append_sites(base,g,f,42)
        # Canonical records, not caller order, determine identities and quantities.
        self.assertEqual(first.records(),second.records())
        self.assertEqual(base.arrays['family'].size,0)  # pure table input
        a=first.arrays
        lat,lon=(a['j']+.5)*30/g.radius_m,(a['i']+.5)*30/g.radius_m
        xyz=g.radius_m*np.stack((np.cos(lat)*np.cos(lon),np.cos(lat)*np.sin(lon),np.sin(lat)),axis=1)
        self.assertGreaterEqual(cKDTree(xyz).query(xyz,k=2)[0][:,1].min(),6000)
        self.assertEqual(int(f['coastal_salt_kg_year'].sum()),int(a['annual_kg'].sum()))
        for j,i in zip(*np.nonzero(f['coastal_salt_kg_year'])):
            _,amount=R.descend_budget(g,f,'coastal_salt_kg_year',j,i,lod=8,seed=42)
            self.assertEqual(int(amount.sum()),int(f['coastal_salt_kg_year'][j,i]))
        # Production is not part of finite stock or recovery arrays.
        R.stock_layers(g,f,first)
        self.assertEqual(int(f['stock_salt'].sum()),0)
        self.assertEqual(int(f['recoverable_salt'].sum()),0)

    def test_native_cliffs_and_diagnostic_menu(self):
        import terrain_resource_layers as L
        g,f,_=self.fixture()
        coast_candidates(g,f,np.tile(np.array([-3000.,1000.],np.float32),(8,1)))
        self.assertEqual(len(f['salt_coast_i']),0)
        for mode in ('resources-mat-sea-salt','resources-season-salt'):
            self.assertEqual(L.specs()[mode]['group'],'diagnostic')
        self.assertEqual(L.specs()['resources-dep-salt']['annual_equivalence'],100)


if __name__=='__main__':unittest.main()

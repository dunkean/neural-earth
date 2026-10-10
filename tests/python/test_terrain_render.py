"""Surface materials, generated substrate, seams and GPU reference fixtures."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path

import json
import sys
import unittest
from types import SimpleNamespace
import numpy as np
from terrain_soil import NAMES, generate_soil, coordinates, sample_soil_transport, seed_value
from terrain_render import (DEFAULTS, parse_settings, surface_material, colorize_surface,
                            gradient_noise, fbm, cover, terrain_derivatives)
from terrain_biomes import sample_biome_transport

SOIL = np.array([.42,.32,.22,.46,.44,.40,.3,.3,.4],np.float32)


def world():
    h,w=32,64
    layers={name:np.full((h,w),value,np.float32) for name,value in
            [('temperature_summer',65/90),('temperature_winter',45/90),
             ('precip_summer',.65),('precip_winter',.65),('koppen',9)]}
    atlas=SimpleNamespace(seed=42,bounds=(-20e6,-10e6,20e6,10e6),width=w,height=h,
                          height_m=np.full((h,w),1000,np.float32),layers=layers)
    layers.update(generate_soil(atlas.height_m,layers,42,atlas.bounds))
    return atlas


def material_case(height=1000,slope=0,resolution=30,settings=None,origin=(-5000,-4000),size=304):
    atlas=world()
    xs=origin[0]+(np.arange(size)+.5)*resolution
    ys=origin[1]+(np.arange(size)+.5)*resolution
    dem=np.broadcast_to(height+(np.arange(size)-size//2)*resolution*np.tan(np.deg2rad(slope)),(size,size)).astype(np.float32).copy()
    return atlas,xs,ys,dem,colorize_surface(atlas,xs,ys,dem,resolution,{},settings)


def direct(height,seasons,*,grade=0.,tpi=0.,footprint=30.,point=None,settings=None,bare=False,shape=(1,1)):
    """Material for uniform inputs, isolating one physical parameter."""
    h=np.broadcast_to(np.asarray(height,np.float32),shape).astype(np.float32)
    p=np.zeros((*shape,3),np.float32) if point is None else np.broadcast_to(point,(*shape,3)).astype(np.float32)
    g=np.zeros((*shape,2),np.float32);g[...,0]=grade
    seasons=np.asarray(seasons,np.float32)
    if seasons.size==6:seasons=np.broadcast_to(seasons.reshape(6,*([1]*len(shape))),(6,*shape))
    soil=np.broadcast_to(SOIL.reshape(9,*([1]*len(shape))),(9,*shape))
    north=np.broadcast_to(np.array([0,-1],np.float32),(*shape,2))
    return surface_material(h,g,np.broadcast_to(np.float32(tpi),shape),p,north,footprint,soil,seasons,
                            seed_value(42),settings,bare)


TEMPERATE=[22,500,.0065,2,500,.0065]


class SurfaceTests(unittest.TestCase):
    def test_generated_composition_is_bounded_deterministic_and_climate_driven(self):
        atlas=world();a=generate_soil(atlas.height_m,atlas.layers,42,atlas.bounds)
        b=generate_soil(atlas.height_m,atlas.layers,42,atlas.bounds)
        for name in NAMES:
            np.testing.assert_array_equal(a[name],b[name]);self.assertTrue(np.isfinite(a[name]).all())
            self.assertGreaterEqual(a[name].min(),0);self.assertLessEqual(a[name].max(),1)
        np.testing.assert_allclose(a['soil_sand']+a['soil_clay']+a['soil_humus'],1,atol=2e-7)
        dry=dict(atlas.layers,precip_summer=np.zeros_like(atlas.height_m),precip_winter=np.zeros_like(atlas.height_m))
        c=generate_soil(atlas.height_m,dry,42,atlas.bounds)
        self.assertGreater(a['soil_humus'].mean(),c['soil_humus'].mean())
        self.assertFalse(np.array_equal(a['rock_red'],generate_soil(atlas.height_m,atlas.layers,43,atlas.bounds)['rock_red']))

    def test_noise_has_unit_variance_and_fbm_filters_unresolved_octaves(self):
        p=np.random.default_rng(3).uniform(-1e4,1e4,(200000,3)).astype(np.float32)
        self.assertAlmostEqual(float(gradient_noise(p,11).std()),1,delta=.05)
        value,detail=fbm(p*100,2400.,6,5.,5)
        self.assertAlmostEqual(float(value.std()),1,delta=.1);self.assertAlmostEqual(float(detail),1,places=2)
        value,detail=fbm(p*100,2400.,6,5000.,5)
        self.assertEqual(float(detail),0);self.assertFalse(np.any(value))

    def test_filtered_cover_preserves_mean_without_binary_blotches(self):
        p=np.random.default_rng(4).uniform(-1e6,1e6,(200000,3)).astype(np.float32)
        pattern=fbm(p,2400.,6,30.,9)
        for fraction in (.1,.5,.85):
            crisp=cover(np.full(len(p),fraction,np.float32),pattern)
            self.assertAlmostEqual(float(crisp.mean()),fraction,delta=.04)
            self.assertLess(np.max(np.abs(crisp-fraction)),.09,'Noise is secondary to coverage')
            self.assertGreater(crisp.std(),.005)
            np.testing.assert_allclose(cover(np.full(4,fraction,np.float32),fbm(p[:4],2400.,6,5000.,9)),fraction,atol=1e-6)
        for strength in (0,1,2):
            result=cover(np.linspace(0,1,len(p),dtype=np.float32),pattern,strength)
            self.assertGreaterEqual(result.min(),0);self.assertLessEqual(result.max(),1)

    def test_mean_color_is_stable_across_footprints(self):
        # A forest mosaic seen at 30 m averages to the colour shown at 2 km.
        rng=np.random.default_rng(5);n=60000
        p=(rng.uniform(-1,1,(n,3))*np.array([2e5,2e5,0])+np.array([6.3e6,0,0])).astype(np.float32)
        mosaic=[22,330,.0065,2,330,.0065]
        fine=direct(500,mosaic,point=p,shape=(n,),footprint=30.,settings={'variation':0})
        broad=direct(500,mosaic,point=p,shape=(n,),footprint=8000.,settings={'variation':0})
        np.testing.assert_allclose(fine.mean(0),broad.mean(0),atol=.012)
        self.assertLess(fine.std(0).max(),.001,'Variation zero removes random forest patches')
        varied=direct(500,mosaic,point=p,shape=(n,),footprint=30.,settings={'variation':1})
        self.assertLess(varied.std(0).max(),.035,'Forest noise must not dominate the material')

    def test_biome_transitions_are_continuous(self):
        # Sweep sea-level temperature and precipitation: no class-like steps.
        n=2000;t=np.linspace(-15,30,n,dtype=np.float32);rain=np.linspace(50,1400,n,dtype=np.float32)
        for seasons in ([t+8,np.full(n,400),np.zeros(n),t-8,np.full(n,400),np.zeros(n)],
                        [np.full(n,24),rain,np.zeros(n),np.full(n,10),rain,np.zeros(n)]):
            rgb=direct(200,np.asarray(seasons,np.float32),shape=(n,),footprint=8000.,settings={'variation':0})
            self.assertLess(np.abs(np.diff(rgb,axis=0)).max(),.01)
            self.assertGreater(np.ptp(rgb,axis=0).max(),.15,'The sweep crosses visibly different biomes')

    def test_slopes_expose_rock_and_coarse_samples_compensate_their_gentler_slopes(self):
        flat=direct(800,TEMPERATE,settings={'variation':0},footprint=9000.)
        steep=direct(800,TEMPERATE,grade=1.4,settings={'variation':0},footprint=30.)
        rock=np.array([.46,.44,.40])*.35+np.array([.50,.48,.45])*.65
        self.assertLess(np.abs(steep[0,0]-rock).max(),np.abs(flat[0,0]-rock).max())
        fine=direct(800,TEMPERATE,grade=.35,settings={'variation':0},footprint=30.)
        coarse=direct(800,TEMPERATE,grade=.35,settings={'variation':0},footprint=7680.)
        self.assertLess(np.abs(coarse[0,0]-rock).max(),np.abs(fine[0,0]-rock).max()-.03)

    def coast(self,footprint,n,settings):
        # Sea for 1 km, then a 1 m lowland plain: beach strip, then river-bar flats.
        x=(np.arange(n)+.5)*footprint-1000
        h=np.where(x<0,-5.,1.).astype(np.float32)[None]
        p=np.zeros((1,n,3),np.float32);p[...,0]=6.3e6;p[...,2]=x+2e5
        shape=(1,n);seasons=np.broadcast_to(np.float32(TEMPERATE).reshape(6,1,1),(6,*shape))
        soil=np.broadcast_to(SOIL.reshape(9,1,1),(9,*shape))
        rgb=surface_material(h,np.zeros((*shape,2),np.float32),np.zeros(shape,np.float32),p,
                             np.broadcast_to(np.float32([0,-1]),(*shape,2)),footprint,soil,seasons,seed_value(42),settings)
        return x,rgb[0]

    def test_beaches_follow_the_sea_and_keep_their_area_at_lod3(self):
        on={'variation':0}
        x,fine=self.coast(30.,256,on)
        sand=np.array([.80,.74,.59])
        self.assertLess(np.abs(fine[np.searchsorted(x,0)]-sand).max(),.06,'shore sample is beach')
        inland=np.searchsorted(x,5000)
        self.assertGreater(np.abs(fine[inland]-sand).max(),.06,'no beach kilometres inland')
        self.assertLess(np.abs(fine[inland]-[.72,.67,.55]).max(),.1,'flats stay, as wetter bars')
        # The 240 m view shows the strip and averages to the 30 m view.
        xc,coarse=self.coast(240.,32,on)
        self.assertLess(np.abs(coarse[np.searchsorted(xc,0)]-sand).max(),.12,'beach visible at LOD 3')
        span=lambda x,rgb:rgb[(x>0)&(x<1920)].mean(0)
        np.testing.assert_allclose(span(x,fine),span(xc,coarse),atol=.02)
        # Away from the sea the coastal model only recolours the flats.
        self.assertLess(np.abs(coarse[-1]-fine[inland]).max(),.02)

    def test_legacy_coastal_toggle_is_accepted_and_ignored(self):
        self.assertEqual(parse_settings({'coasts':0}),parse_settings(None))

    def test_valleys_are_greener_than_ridges_in_semi_arid_climates(self):
        semi=[26,170,.0065,14,170,.0065]
        valley=direct(300,semi,tpi=-.1,footprint=8000.,settings={'variation':0})[0,0]
        ridge=direct(300,semi,tpi=.1,footprint=8000.,settings={'variation':0})[0,0]
        self.assertGreater(valley[1]-valley[0],ridge[1]-ridge[0]+.02)

    def test_snow_sheds_from_cliffs_but_perennial_snow_extends_retention(self):
        # Seasonal winter snow (warm summers) versus perennial snow (cold summers).
        cold=[8,650,0,-10,650,0];glacier=[-6,650,0,-14,650,0];winter={'variation':0,'season':0}
        flat=direct(1000,cold,settings=winter);steep=direct(1000,cold,grade=2.,settings=winter)
        cliff=direct(1000,cold,grade=6.,settings=winter)
        self.assertGreater(flat.mean(),steep.mean()+.04)
        self.assertGreater(steep.mean(),.75,'Snow remains on a 63 degree slope')
        self.assertGreater(direct(1000,glacier,grade=2.2,settings=winter).mean(),
                           direct(1000,cold,grade=2.2,settings=winter).mean()+.05)
        self.assertLess(cliff.max(),.7)

    def test_materials_vary_and_clearings_expose_ground(self):
        _,_,_,_,a=material_case(settings={'snow':0})
        _,_,_,_,bare=material_case(settings={'snow':0,'forest':0})
        self.assertGreater(np.std(a[...,1]),.003)
        self.assertGreater(np.max(np.abs(a-bare)),.08)
        change=np.max(np.abs(a-bare),axis=-1)
        self.assertGreater(np.ptp(change),.02,'Canopy density varies continuously')

    def test_equatorial_snow_aspect_has_no_hemisphere_step(self):
        p=np.array([[6.3e6,-1,0],[6.3e6,1,0]],np.float32)
        shape=(2,);g=np.broadcast_to([0,.8],(2,2)).astype(np.float32)
        rgb=surface_material(np.full(2,1000,np.float32),g,np.zeros(2,np.float32),p,
             np.broadcast_to([0,-1],(2,2)),30,np.broadcast_to(SOIL[:,None],(9,2)),
             np.broadcast_to(np.array([5,650,0,-2,650,0],np.float32)[:,None],(6,2)),
             42,{'season':0,'variation':0})
        self.assertLess(np.max(np.abs(rgb[0]-rgb[1])),1e-4)

    def test_coarse_snow_does_not_treat_slope_compensation_as_a_cliff(self):
        a=direct(1000,[-6,650,0,-14,650,0],grade=.45,footprint=30,settings={'variation':0})
        b=direct(1000,[-6,650,0,-14,650,0],grade=.45,footprint=7680,settings={'variation':0})
        np.testing.assert_allclose(a,b,atol=.01)

    def test_season_changes_snow_in_opposite_hemispheres(self):
        p=np.array([[0,6e6,0],[0,-6e6,0]],np.float32)
        seasons=np.array([[15,-8],[650,650],[0,0],[-8,15],[650,650],[0,0]],np.float32)
        def color(season):
            return surface_material(np.full(2,1000,np.float32),np.zeros((2,2),np.float32),np.zeros(2,np.float32),p,
                                    np.array([[0,-1],[0,-1]],np.float32),30,np.broadcast_to(SOIL[:,None],(9,2)),seasons,
                                    1,{'season':season,'variation':0})
        winter=color(0);summer=color(.5)
        self.assertGreater(winter[0].mean(),summer[0].mean()+.2)
        self.assertGreater(summer[1].mean(),winter[1].mean()+.2)

    def test_soil_layer_shows_substrate_without_vegetation_or_snow(self):
        cold=[-3,650,0,-3,650,0]
        self.assertLess(direct(1000,cold,bare=True).mean(),direct(1000,cold).mean()-.2)
        humid=direct(400,[24,1200,0,18,1200,0],bare=True,shape=(1,1),footprint=30.)
        arid=direct(400,[30,40,0,22,40,0],bare=True,shape=(1,1),footprint=30.)
        self.assertGreater(arid.mean(),humid.mean(),'Desert sands are brighter than humid soils')

    def test_adjacent_tiles_and_negative_coordinates_share_materials(self):
        atlas=world();r=15;xs=(-280+np.arange(560)+.5)*r;ys=(-152+np.arange(304)+.5)*r
        yy,xx=np.mgrid[:304,:560];dem=(3000+800*np.sin(xx*.03)+300*np.cos(yy*.07)).astype(np.float32)
        whole=colorize_surface(atlas,xs,ys,dem,r,{})[24:280,24:536]
        a=colorize_surface(atlas,xs[:304],ys,dem[:,:304],r,{})[24:280,24:280]
        b=colorize_surface(atlas,xs[256:],ys,dem[:,256:],r,{})[24:280,24:280]
        np.testing.assert_allclose(np.concatenate((a,b),axis=1),whole,atol=1e-5)

    def test_geographic_noise_matches_polar_chart_and_longitude_seam(self):
        from terrain_polar import to_chart
        atlas=world();x,y=3e6,-7e6;cx,cy=to_chart(x,y,{})
        a=coordinates([x],[y],atlas.bounds);b=coordinates([cx],[cy],atlas.bounds,polar=True)
        np.testing.assert_allclose(a,b,atol=.5)
        np.testing.assert_allclose(gradient_noise(a/180,23),gradient_noise(b/180,23),atol=2e-3)
        seam=coordinates([-20e6,20e6],[-3e6],atlas.bounds)/180
        np.testing.assert_allclose(gradient_noise(seam,23)[0,0],gradient_noise(seam,23)[0,1],atol=2e-3)

    def test_terrain_derivatives_match_gpu_conventions(self):
        dem=np.broadcast_to(np.arange(64,dtype=np.float32)*3,(64,64)).copy()
        gradient,tpi=terrain_derivatives(dem,30.)
        np.testing.assert_allclose(gradient[10:-10,10:-10,0],.1,atol=1e-6)
        np.testing.assert_allclose(tpi[30,30],0,atol=1e-5)
        bump=np.zeros((64,64),np.float32);bump[32,32]=600
        self.assertGreater(terrain_derivatives(bump,30.)[1][32,32],0)

    def test_settings_validate_colors_and_ranges(self):
        self.assertEqual(parse_settings(),DEFAULTS)
        for invalid in ({'season':2},{'snow_color':[1,2,3]},{'forest':float('nan')},{'moisture':3},{'unexpected':1}):
            with self.assertRaises(ValueError):parse_settings(invalid)

    def test_appearance_identity_changes_with_cpu_or_gpu_materials(self):
        from terrain_render import appearance_identity
        from pathlib import Path
        import tempfile
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('terrain_render.py','terrain_render_torch.py','terrain_renderer.js','terrain_render_controls.js','terrain_soil.py','terrain_pedology.py'):
                (source_path(name, root=root)).write_text('original',encoding='utf-8')
            a=appearance_identity(root)
            (source_path('terrain_render.py', root=root)).write_text('new CPU material',encoding='utf-8')
            b=appearance_identity(root);self.assertNotEqual(a,b)
            (source_path('terrain_renderer.js', root=root)).write_text('new GPU material',encoding='utf-8')
            self.assertNotEqual(b,appearance_identity(root))

    def test_snowline_has_no_jump_when_altitude_crosses_a_koppen_boundary(self):
        atlas=world();shape=(32,32)
        for name,value in [('temperature_summer',75/90),('temperature_winter',53/90)]:
            atlas.layers[name].fill(value)
        # Sea-level summer 30 C, default lapse: the ET boundary is around 3 km.
        from terrain_koppen import seasonal_fields
        xs=np.linspace(-500,500,32);ys=np.linspace(-500,500,32)
        seasons=seasonal_fields(atlas,xs,ys,{})
        boundary=(seasons[0,16,16]-10)/seasons[2,16,16]
        a=colorize_surface(atlas,xs,ys,np.full(shape,boundary-1,np.float32),30,{}, {'variation':0})
        b=colorize_surface(atlas,xs,ys,np.full(shape,boundary+1,np.float32),30,{}, {'variation':0})
        self.assertLess(np.max(np.abs(a-b)),.01)


def fixtures():
    atlas=world();xs=np.linspace(-5000,4120,33);ys=np.linspace(-4000,5120,33)
    transport=np.concatenate((np.zeros((5,33,33),np.float32),sample_biome_transport(atlas,xs,ys,{}),sample_soil_transport(atlas,xs,ys,{})))
    # Uniform sampled substrate and climate make the oracle independent of the
    # climate-grid interpolation. The atlas and its persistence are tested above.
    for i,value in enumerate(SOIL):transport[36+i].fill(value)
    ped_color=np.array([.62,.41,.25],np.float32)
    for i,value in enumerate(ped_color):transport[46+i].fill(value)
    transport[49].fill(1)
    climate=[20,650,.0065,0,650,.0065]
    for i,value in enumerate(climate):transport[21+i].fill(value)
    sea=lambda n:np.broadcast_to(np.array(climate,np.float32)[:,None,None],(6,n,n))
    soil=lambda n:np.broadcast_to(SOIL[:,None,None],(9,n,n))
    ped=lambda n:np.broadcast_to(ped_color[:,None,None],(3,n,n))
    cases=[]
    scenes=[(1000,0,.5,30,False,False),(1000,45,.5,30,False,False),(1000,0,0,30,False,False),(6000,0,.5,30,False,False),(6000,65,.5,15,False,False),(6000,82,.5,3.75,False,False),(1000,0,.5,15,True,False),(1000,0,.5,30,False,True),(-1000,0,.5,30,False,False),(-20,0,.5,30,False,False),(300,25,.5,120,False,False),(1000,0,.5,7680,False,False),(2500,0,0,7680,False,False),
            (500,8,.5,30,False,False),(1200,12,.5,30,False,False),(800,25,0,30,False,False),(1000,0,0,7680,False,False),
            (2800,50,.5,60,False,False),(2800,50,.5,120,False,False),(2000,8,.5,60,False,False),(2000,8,.5,120,False,False)]
    extra=[([26,160,.006,12,170,.006],{'variation':2,'forest':1.4}),
           ([30,900,.006,25,1000,.006],{'forest':.6}),
           ([20,500,.006,-4,500,.006],{'moisture':.5}),
           ([-10,25,0,-20,25,0],{}),
           ([38,60,.0055,24,80,.0055],{}),([38,60,.0055,24,80,.0055],{}),
           ([25,550,.006,8,550,.006],{}),([25,550,.006,8,550,.006],{})]
    for index,(height,slope,season,r,plane,polar) in enumerate(scenes):
        n=160 if r==7680 else 304;origin=(-5000,-4000)
        if polar:origin=(0,-10e6)
        x=origin[0]+(np.arange(n)+.5)*r;y=origin[1]+(np.arange(n)+.5)*r
        dem=np.broadcast_to(height+(np.arange(n)-n//2)*r*np.tan(np.deg2rad(slope)),(n,n)).astype(np.float32).copy()
        point=coordinates(x,y,atlas.bounds,not plane,polar)
        gradient,tpi=terrain_derivatives(dem,r)
        if plane:north=np.zeros_like(gradient)
        elif not polar:north=np.broadcast_to([0,-1],gradient.shape)
        else:
            ny,nx=np.gradient(point[...,1],r);north=np.stack((nx,ny),axis=-1);north/=np.maximum(np.linalg.norm(north,axis=-1,keepdims=True),1e-12)
        case_climate,options=extra[index-13] if index>=13 else (climate,{})
        seasons=np.broadcast_to(np.array(case_climate,np.float32)[:,None,None],(6,n,n))
        options=dict(options,season=season)
        result=surface_material(dem,gradient,tpi,point,north,r,soil(n),seasons,seed_value(42),options,pedology=ped(n))
        cases.append(dict(height=height,slope=slope,season=season,settings=options,seasons=case_climate,resolution=r,plane=plane,polar=polar,origin=origin,expected=(result[n//2,n//2]*255).tolist()))
    x=(np.arange(304)+.5)*30;dem=np.full((304,304),1000,np.float32);point=coordinates(x,x,atlas.bounds)
    gradient,tpi=terrain_derivatives(dem,30)
    bare=surface_material(dem,gradient,tpi,point,np.broadcast_to([0,-1],gradient.shape),30,soil(304),sea(304),seed_value(42),{},True)
    return dict(climate=transport.ravel().tolist(),seasons=climate,cases=cases,soil=(bare[152,152]*255).tolist(),pedology=(ped_color*255).tolist())


if __name__=='__main__':
    if '--fixtures' in sys.argv:print(json.dumps(fixtures()))
    else:unittest.main()

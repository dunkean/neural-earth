"""Surface physics, generated substrate, seams and GPU reference fixtures."""
import json
import sys
import unittest
from types import SimpleNamespace
import numpy as np
from terrain_soil import NAMES, generate_soil, coordinates, noise, sample_soil_transport, seed_value
from terrain_render import DEFAULTS, parse_settings, surface_material, colorize_surface
from terrain_biomes import sample_biome_transport


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

    def test_snow_sheds_from_cliffs_but_cold_altitude_extends_retention(self):
        # Fixed point/noise to isolate slope, independent of climate classification.
        h=np.array([[1000.,6000.]],np.float32);ids=np.array([[9,29]])
        p=np.zeros((1,2,3),np.float32);soil=np.broadcast_to(np.array([.4,.3,.2,.45,.44,.42,.3,.3,.4])[:,None,None],(9,1,2))
        seasons=np.broadcast_to(np.array([-3,650,0,-3,650,0])[:,None,None],(6,1,2))
        def color(grade):return surface_material(ids,h,np.broadcast_to([grade,0],(1,2,2)),np.zeros_like(h),p,30,soil,seasons,np.full_like(h,2),np.full_like(h,5),1,{'variation':0})
        flat=color(0);steep=color(2);cliff=color(6)
        self.assertGreater(flat[0,0].mean(),steep[0,0].mean()+.15)
        self.assertGreater(steep[0,1].mean(),steep[0,0].mean()+.06)
        self.assertLess(cliff.max(),.7)

    def test_materials_vary_and_clearings_expose_soil(self):
        _,_,_,_,a=material_case(settings={'snow':0})
        _,_,_,_,bare=material_case(settings={'snow':0,'forest':0})
        self.assertGreater(np.std(a[...,1]),.003)
        self.assertGreater(np.max(np.abs(a-bare)),.08)
        self.assertGreater(np.std(a[...,0]-a[...,1]),.01)
        self.assertLess(np.min(np.max(np.abs(a-bare),axis=-1)),.005,'Some clearings fully expose the ground')

    def test_season_changes_snow_in_opposite_hemispheres(self):
        h=np.full((1,2),1000,np.float32);p=np.array([[[0,6e6,0],[0,-6e6,0]]],np.float32)
        seasons=np.array([[[15,-8]],[[650,650]],[[0,0]],[[-8,15]],[[650,650]],[[0,0]]],np.float32)
        soil=np.broadcast_to(np.array([.4,.3,.2,.45,.44,.42,.3,.3,.4])[:,None,None],(9,1,2))
        args=(np.full_like(h,9,dtype=np.int32),h,np.zeros((1,2,2)),np.zeros_like(h),p,30,soil,seasons,np.full_like(h,2),np.full_like(h,5),1)
        winter=surface_material(*args,{'season':0,'variation':0});summer=surface_material(*args,{'season':.5,'variation':0})
        self.assertGreater(winter[0,0].mean(),summer[0,0].mean()+.2)
        self.assertGreater(summer[0,1].mean(),winter[0,1].mean()+.2)

    def test_adjacent_tiles_and_negative_coordinates_share_materials(self):
        atlas=world();r=15;xs=(-280+np.arange(560)+.5)*r;ys=(-152+np.arange(304)+.5)*r
        yy,xx=np.mgrid[:304,:560];dem=(3000+800*np.sin(xx*.03)+300*np.cos(yy*.07)).astype(np.float32)
        whole=colorize_surface(atlas,xs,ys,dem,r,{})[24:280,24:536]
        a=colorize_surface(atlas,xs[:304],ys,dem[:,:304],r,{})[24:280,24:280]
        b=colorize_surface(atlas,xs[256:],ys,dem[:,256:],r,{})[24:280,24:280]
        np.testing.assert_allclose(np.concatenate((a,b),axis=1),whole,atol=1e-6)

    def test_geographic_noise_matches_polar_chart_and_longitude_seam(self):
        from terrain_polar import to_chart
        atlas=world();x,y=3e6,-7e6;cx,cy=to_chart(x,y,{})
        a=coordinates([x],[y],atlas.bounds);b=coordinates([cx],[cy],atlas.bounds,polar=True)
        np.testing.assert_allclose(a,b,atol=.5)
        np.testing.assert_allclose(noise(a/180,23),noise(b/180,23),atol=1e-5)
        np.testing.assert_allclose(noise(coordinates([-20e6,20e6],[-3e6],atlas.bounds)/180,23)[0,0],noise(coordinates([-20e6,20e6],[-3e6],atlas.bounds)/180,23)[0,1],atol=1e-5)

    def test_future_water_inputs_are_neutral_until_available(self):
        a,xs,ys,dem,rgb=material_case()
        self.assertTrue(np.isfinite(rgb).all())
        # Current atlas contains no river input; shader and CPU take zero water.
        self.assertFalse(any('river' in key for key in a.layers))

    def test_settings_validate_colors_and_ranges(self):
        self.assertEqual(parse_settings(),DEFAULTS)
        for invalid in ({'season':2},{'snow_color':[1,2,3]},{'forest':float('nan')},{'unexpected':1}):
            with self.assertRaises(ValueError):parse_settings(invalid)

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
    # Uniform sampled substrate makes the oracle independent of climate-grid
    # interpolation. The physical atlas and its persistence are tested above.
    constants=[.42,.32,.22,.46,.44,.40,.3,.3,.4]
    for i,value in enumerate(constants):transport[36+i].fill(value)
    for i,value in enumerate([20,650,0,0,650,0]):transport[21+i].fill(value)
    cases=[]
    for height,slope,season,r,plane,polar in [(1000,0,.5,30,False,False),(1000,45,.5,30,False,False),(1000,0,0,30,False,False),(6000,0,.5,30,False,False),(6000,65,.5,15,False,False),(6000,82,.5,3.75,False,False),(1000,0,.5,15,True,False),(1000,0,.5,30,False,True),(-1000,0,.5,30,False,False),(1000,0,.5,7680,False,False)]:
        n=160 if r==7680 else 304;origin=(-5000,-4000)
        if polar:origin=(0,-10e6)
        # CPU material oracle fed the same uniform transport and exact lookup.
        x=origin[0]+(np.arange(n)+.5)*r;y=origin[1]+(np.arange(n)+.5)*r
        dem=np.broadcast_to(height+(np.arange(n)-n//2)*r*np.tan(np.deg2rad(slope)),(n,n)).astype(np.float32).copy()
        point=coordinates(x,y,atlas.bounds,not plane,polar)
        sea=np.broadcast_to(np.array([20,650,0,0,650,0],np.float32)[:,None,None],(6,n,n))
        from terrain_koppen import classify
        ids=classify(dem,sea,{})
        dy,dx=np.gradient(dem,r);gradient=np.stack((dx,dy),axis=-1)
        convex=(4*dem-np.roll(dem,1,0)-np.roll(dem,-1,0)-np.roll(dem,1,1)-np.roll(dem,-1,1))/r
        if plane:north=np.zeros_like(gradient)
        elif not polar:north=np.broadcast_to([0,-1],gradient.shape)
        else:
            ny,nx=np.gradient(point[...,1],r);north=np.stack((nx,ny),axis=-1);north/=np.maximum(np.linalg.norm(north,axis=-1,keepdims=True),1e-12)
        alpine=transport[30,0,ids];snow=transport[31,0,ids]
        result=surface_material(ids,dem,gradient,convex,point,r,np.broadcast_to(np.array(constants)[:,None,None],(9,n,n)),sea,alpine,snow,seed_value(42),{'season':season},north)
        cases.append(dict(height=height,slope=slope,season=season,resolution=r,plane=plane,polar=polar,origin=origin,expected=(result[n//2,n//2]*255).tolist()))
    return dict(climate=transport.ravel().tolist(),cases=cases)


if __name__=='__main__':
    if '--fixtures' in sys.argv:print(json.dumps(fixtures()))
    else:unittest.main()

"""DEM classification parity, continuous boundaries and GPU transport fixtures."""
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

import numpy as np

from terrain_biomes import sample_biome_fields, sample_biome_transport, colorize_biomes
from terrain_koppen import classify, sample_classes, COLORS


def atlas():
    shape=(2,2)
    return SimpleNamespace(bounds=(-1,-1,1,1),width=2,height=2,
        height_m=np.full(shape,1000,np.float32),layers={
            'koppen':np.full(shape,9,np.uint8),
            'climate_height_m':np.zeros(shape,np.float32),
            'temperature_summer':np.full(shape,65/90,np.float32),
            'temperature_winter':np.full(shape,50/90,np.float32),
            'precip_summer':np.full(shape,.5,np.float32),
            'precip_winter':np.full(shape,.5,np.float32)})


class KoppenTests(unittest.TestCase):
    def test_matches_native_classifier_with_default_and_custom_thresholds(self):
        rng=np.random.default_rng(2026)
        ts,tw=rng.uniform(-45,45,(2,32,32))
        ps,pw=rng.uniform(0,1800,(2,32,32))
        fields=np.array([ts,ps,np.zeros_like(ts),tw,pw,np.zeros_like(ts)],np.float32)
        script="""import {classifyKoppen} from './native/orogen/vendor/koppen.js';
import {climate} from './native/orogen/vendor/climate-config.js';
let input='';for await(const c of process.stdin)input+=c;const f=JSON.parse(input);
Object.assign(climate,f.settings);
const ids=classifyKoppen({numRegions:f.ts.length},Array(f.ts.length).fill(1),
{r_temperature_summer:f.ts.map(v=>(v+45)/90),r_temperature_winter:f.tw.map(v=>(v+45)/90)},
{r_precip_summer:f.ps.map(v=>v/1000),r_precip_winter:f.pw.map(v=>v/1000)});
console.log(JSON.stringify(Array.from(ids)));"""
        for settings in ({},{'orogen_koppen_tropical':14,'orogen_koppen_temperate':-3,
                              'orogen_koppen_desert_fraction':.7,'orogen_koppen_shoulder_fraction':.25}):
            payload=dict(ts=fields[0].ravel().tolist(),tw=fields[3].ravel().tolist(),
                         ps=fields[1].ravel().tolist(),pw=fields[4].ravel().tolist(),
                         settings={k[7:]:v for k,v in settings.items()})
            result=subprocess.run(['node','--input-type=module','-e',script],input=json.dumps(payload),
                                  text=True,capture_output=True,check=True,cwd=Path(__file__).parent)
            np.testing.assert_array_equal(classify(np.ones((32,32)),fields,settings).ravel(),json.loads(result.stdout))

    def test_classes_and_biome_palette_follow_dem_including_new_coasts(self):
        world=atlas();xs=np.linspace(-.5,.5,4);ys=[-.5,.5]
        dem=np.tile([-100,100,2000,5000],(2,1)).astype(np.float32)
        ids=sample_classes(world,xs,ys,{},dem)
        np.testing.assert_array_equal(ids[0],[0,9,29,30])
        settings={'orogen_biome_color_cfb':[.8,.1,.2],'orogen_biome_color_et':[.2,.3,.9]}
        fields=sample_biome_fields(world,xs,ys,settings,dem)
        np.testing.assert_allclose(fields[:3,0,1],[.8,.1,.2])
        np.testing.assert_allclose(fields[:3,0,2],[.2,.3,.9])

    def test_classification_happens_after_interpolating_source_climate(self):
        world=atlas()
        world.layers['temperature_summer'][:]=np.array([[50/90,70/90]]*2)
        world.layers['temperature_winter'][:]=np.array([[35/90,60/90]]*2)
        xs=np.linspace(-.5,.5,101)
        ids=sample_classes(world,xs,[-.5,.5],{},np.full((2,101),100,np.float32))
        transitions=np.flatnonzero(np.diff(ids[0]))
        self.assertGreater(len(transitions),1)
        self.assertTrue(any(abs(int(x)-50)>5 for x in transitions),'Boundaries are continuous climate thresholds, not nearest class cells')

    def test_sea_level_transport_uses_retained_climate_height(self):
        world=atlas();dem=np.full((2,2),100,np.float32)
        before=sample_classes(world,[-.5,.5],[-.5,.5],{},dem)
        world.height_m[:]=6000
        np.testing.assert_array_equal(sample_classes(world,[-.5,.5],[-.5,.5],{},dem),before)


def gpu_fixtures():
    world=atlas();xs=np.linspace(-.5,.5,33)
    settings={'orogen_biome_color_cfb':[.7,.2,.1],'orogen_koppen_shoulder_fraction':.25}
    climate=np.concatenate((np.zeros((5,33,33),np.float32),sample_biome_transport(world,xs,xs,settings)))
    cases=[]
    for height in (-100,100,1000,2000,5000):
        dem=np.full((33,33),height,np.float32)
        ids=sample_classes(world,xs,xs,settings,dem)
        fields=sample_biome_fields(world,xs,xs,settings,dem)
        cases.append(dict(height=height,koppen=(COLORS[ids[16,16]]*255).tolist(),
                          biome=(colorize_biomes(dem,fields,30)[16,16]*255).tolist()))
    rng=np.random.default_rng(123)
    ts,tw=rng.uniform(-45,45,(2,1,100000))
    ps,pw=rng.uniform(0,3000,(2,1,100000))
    seasons=np.asarray([ts,ps,np.zeros_like(ts),tw,pw,np.zeros_like(ts)],np.float32)
    ids=classify(np.full((1,100000),100,np.float32),seasons,settings)
    for code in range(1,31):
        indices=np.flatnonzero(ids[0]==code)
        if not len(indices):raise AssertionError('Missing GPU fixture class '+str(code))
        raw=seasons[:,0,indices[0]]
        # Feed these seasons through the normal CPU sampler too.
        source=atlas();source.layers['temperature_summer'][:]=(raw[0]+45)/90
        source.layers['temperature_winter'][:]=(raw[3]+45)/90
        source.layers['precip_summer'][:]=raw[1]/1000;source.layers['precip_winter'][:]=raw[4]/1000
        # Turn off the lapse for this reference set; altitude cases above test it.
        options={**settings,'orogen_temperature_wet_lapse':0,'orogen_temperature_dry_lapse_extra':0}
        dem=np.full((33,33),100,np.float32)
        fields=sample_biome_fields(source,xs,xs,options,dem)
        cases.append(dict(height=100,seasons=raw.tolist(),koppen=(COLORS[code]*255).tolist(),
                          biome=(colorize_biomes(dem,fields,30)[16,16]*255).tolist()))
    return dict(climate=climate.ravel().tolist(),cases=cases)


if __name__=='__main__':
    if '--fixtures' in sys.argv:print(json.dumps(gpu_fixtures()))
    else:unittest.main()

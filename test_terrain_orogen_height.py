"""Physical conversion contracts shared by Orogen relief, climate and imports."""
import json
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

import numpy as np

import terrain_orogen as core
import terrain_orogen_stages as stages
from terrain_orogen import _metres, _elevation, _atlas_to_elevation


class OrogenHeightCalibrationTests(unittest.TestCase):
    def test_sea_level_sign_order_and_bounded_mountain_change(self):
        elevations=np.linspace(-1,1,100001)
        heights=_metres(elevations)
        np.testing.assert_array_equal(heights<0,elevations<0)
        self.assertEqual(float(_metres(0)),0.)
        # Float32 rounds neighbouring samples together close to the flat summit.
        self.assertTrue(np.all(np.diff(heights)>=0))
        self.assertTrue(np.all(np.diff(heights[elevations<.9])>0))
        np.testing.assert_allclose(heights[elevations<0],10000*elevations[elevations<0],rtol=1e-7)
        t=np.linspace(0,1,10001)
        original=6000*t**4*(5-4*t)
        self.assertLessEqual(float(np.max(_metres(t)-original)),81.921)
        self.assertEqual(float(_metres(1)),6000.)
        self.assertGreater(float(_metres(.01)),9.)
        self.assertLess(float(_metres(.01)),10.)

    def test_imported_metres_roundtrip_including_submetre_shoreline(self):
        heights=np.array([-10000,-500,-.01,0,.001,.01,.1,1,10,50,500,2500,5999],np.float64)
        np.testing.assert_allclose(_metres(_elevation(heights)),heights,rtol=2e-6,atol=1e-6)

    def test_city_graph_resampling_uses_same_inverse_and_keeps_ocean(self):
        points=np.array([[1.,0,0],[0,0,1],[0,0,-1]],np.float32)
        raw=np.full((2,4),12.5,np.float32)
        original=np.array([.2,-.1,.01],np.float32)
        out=_atlas_to_elevation(raw,points,original)
        self.assertEqual(float(out[1]),float(original[1]))
        np.testing.assert_allclose(_metres(out[[0,2]]),12.5,atol=2e-6)

    def test_native_climate_conversion_matches_python(self):
        root=Path(__file__).resolve().parent
        samples=[-.5,-.001,0,.00001,.01,.03,.1,.2,.5,.9,1.,1.2]
        uri=(root/'native/orogen/vendor/color-map.js').as_uri()
        code=f"import {{elevToHeightKm}} from {json.dumps(uri)}; console.log(JSON.stringify({json.dumps(samples)}.map(x=>1000*elevToHeightKm(x))));"
        result=subprocess.run(['node','--input-type=module','-e',code],capture_output=True,text=True,check=True)
        np.testing.assert_allclose(_metres(samples),json.loads(result.stdout),rtol=1e-7,atol=1e-6)

    def test_historical_retained_state_keeps_physical_heights(self):
        root=Path(__file__).resolve().parent
        samples=[-.5,0,.001,.01,.05,.1,.5,1.]
        worker=(root/'native/orogen/vendor/planet-worker.js').as_uri()
        code=("globalThis.self={}; "
              f"const {{rebaseLegacyHeightConvention}}=await import({json.dumps(worker)}); "
              f"const original=Float32Array.from({json.dumps(samples)}); "
              "const state={prePostElev:original,r_elevation_final:original}; "
              "const result={prePostElev:original,r_elevation:original}; "
              "rebaseLegacyHeightConvention(state,result); "
              "console.log(JSON.stringify([state.prePostElev,state.r_elevation_final,result.prePostElev,result.r_elevation].map(a=>Array.from(a))));")
        result=subprocess.run(['node','--input-type=module','-e',code],capture_output=True,text=True,check=True)
        native=np.asarray(samples,dtype=np.float32).astype(np.float64)
        t=np.clip(native,0,1)
        old=np.where(native<=0,native*10000,6000*t**4*(5-4*t))
        for rebased in json.loads(result.stdout.strip().splitlines()[-1]):
            np.testing.assert_allclose(_metres(rebased),old,rtol=2e-6,atol=1e-6)

    def test_climate_resume_routes_legacy_and_calibrated_parents(self):
        config=core.style_config('earthlike')
        captured=[]
        def intercept(command,**kwargs):
            captured.append(json.loads(Path(command[2]).read_text()))
            raise RuntimeError('captured Node request')
        for annotation in (None,core.HYPSOMETRY_RULE):
            metadata={'namespace':{'settings':{'height_source':'orogen'}}}
            if annotation is not None:metadata['height_convention']=annotation
            parent={'metadata':metadata,'arrays':{'snapshot':np.zeros(1,np.uint8)}}
            with patch.object(core,'implementation_identity',return_value={'cuda_device':{'index':0}}), \
                 patch.object(stages.subprocess,'run',side_effect=intercept):
                with self.assertRaisesRegex(RuntimeError,'captured Node request'):
                    stages._execute('climate',1,'earthlike',config,4,2,parent,parent)
        self.assertEqual([r['legacyHeightConvention'] for r in captured],[True,False])
        self.assertEqual([r['resumeCommand'] for r in captured],['climate','climate'])

    def test_composition_reports_actual_active_height_convention(self):
        heights=np.array([[.2,-5],[12,50]],np.float32)
        for annotation in (None,core.HYPSOMETRY_RULE):
            metadata=dict(namespace={'parent':None,'settings':{}},projection_fallback_pixels=0,
                          original_pipeline=[],original_elevation_timing=[])
            if annotation is not None:metadata['height_convention']=annotation
            relief={'id':'fixture','metadata':metadata,'arrays':{'height_m':heights}}
            with patch.object(stages,'_settings',return_value={}), \
                 patch.object(stages,'STAGE_KEYS',{'relief':set()}):
                result=stages.compose(1,'earthlike',{},2,2,{'relief':relief},'relief')
            np.testing.assert_array_equal(result[0],heights)
            expected=annotation or 'original Orogen: ocean 10000*e; land 6000*t^4*(5-4*t)'
            self.assertEqual(result[2]['hypsometry']['rule'],expected)


if __name__=='__main__':
    unittest.main()

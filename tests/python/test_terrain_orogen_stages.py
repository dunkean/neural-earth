"""Physical integration: an individual command must not rerun other stages."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import terrain_orogen as core
import terrain_orogen_stages as stages
from terrain_generation import _defaults,register_generation,resolve_generation,OROGEN_GPU_PARAMETERS


@unittest.skipUnless(torch.cuda.is_available(),'CUDA atlas required')
class IndependentStages(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        root=Path(self.temp.name)
        for target,value in [('terrain_orogen.CACHE_ROOT',root/'atlas'),
            ('terrain_orogen.WIDTH',128),('terrain_orogen.HEIGHT',64),
            ('terrain_generation.REGISTRY_ROOT',root/'registry')]:
            p=patch(target,value);p.start();self.addCleanup(p.stop)
        self.settings=dict(_defaults('orogen'),orogen_detail=20000)

    def run_stage(self,settings,stage,source=None):
        result,executed=stages.run_generation(50,'orogen',settings,stage,source)
        profile=register_generation('orogen',result)
        descriptor=resolve_generation(profile)
        world=core.OrogenHeightmap(50,descriptor.bootstrap_style,options=descriptor.bootstrap_options)
        return profile,world,executed

    def test_full_cpu_chain_matches_original_reference(self):
        reference=core.generate_atlas(50,self.settings['continental_style'],width=128,height=64,options={'detail':20000},include_layers=True)
        _,world,execution=self.run_stage(self.settings,'all')
        self.assertEqual(set(execution),{'relief','erosion','climate'})
        np.testing.assert_array_equal(world.height_m,reference[0])
        for name,value in reference[4].items():np.testing.assert_array_equal(world.layers[name],value,err_msg=name)
        self.assertFalse(any(v.get('stale',False) for v in world.metadata['stage_state'].values() if isinstance(v,dict)))

    def test_individual_stages_retain_others_and_climate_uses_active_height(self):
        profile,initial,_=self.run_stage(self.settings,'all')
        options=dict(self.settings,orogen_temperature_equator=33.,orogen_hydraulic=.1)
        with patch.object(stages,'_execute',wraps=stages._execute) as execute, \
             patch.object(core,'initial_raster',side_effect=AssertionError('relief regenerated')):
            climate_profile,climate,execution=self.run_stage(options,'climate',profile)
        self.assertEqual([call.args[0] for call in execute.call_args_list],['climate'])
        self.assertEqual(set(execution),{'climate'})
        np.testing.assert_array_equal(climate.height_m,initial.height_m)
        np.testing.assert_array_equal(climate.layers['climate_height_m'],initial.height_m)
        self.assertFalse(np.array_equal(climate.layers['temperature_summer'],initial.layers['temperature_summer']))
        from terrain_soil import NAMES
        self.assertTrue(all(name in climate.layers for name in NAMES))
        self.assertEqual(climate.metadata['substrate']['composition'],['sand','clay','humus'])
        self.assertFalse(np.array_equal(climate.layers['soil_humus'],initial.layers['soil_humus']))
        from terrain_pedology import NAMES as PEDOLOGY_NAMES
        self.assertTrue(all(name in climate.layers for name in PEDOLOGY_NAMES))
        self.assertEqual(climate.metadata['pedology']['cell_metres'],200000)
        self.assertFalse(np.array_equal(climate.layers['pedology_red'],initial.layers['pedology_red']))
        self.assertEqual(climate.metadata['stage_state']['erosion']['id'],initial.metadata['stage_state']['erosion']['id'])
        self.assertEqual(resolve_generation(climate_profile).settings['orogen_hydraulic'],self.settings['orogen_hydraulic'])
        options=dict(resolve_generation(climate_profile).settings,orogen_hydraulic=0.,orogen_thermal=0.,orogen_glacial=0.)
        with patch.object(stages,'_execute',wraps=stages._execute) as execute:
            erosion_profile,eroded,_=self.run_stage(options,'erosion',climate_profile)
        self.assertEqual([call.args[0] for call in execute.call_args_list],['erosion'])
        np.testing.assert_array_equal(eroded.layers['climate_height_m'],climate.height_m)
        for name,value in climate.layers.items():
            if stages.is_climate_field(name):np.testing.assert_array_equal(eroded.layers[name],value)
        self.assertTrue(eroded.metadata['stage_state']['climate']['stale'])
        options=dict(resolve_generation(erosion_profile).settings,orogen_roughness=.9)
        with patch.object(stages,'_execute',wraps=stages._execute) as execute:
            relief_profile,relief,_=self.run_stage(options,'relief',erosion_profile)
        self.assertEqual([call.args[0] for call in execute.call_args_list],['relief'])
        self.assertEqual(relief.metadata['stage_state']['height_stage'],'relief')
        self.assertTrue(relief.metadata['stage_state']['erosion']['stale'])
        self.assertTrue(relief.metadata['stage_state']['climate']['stale'])
        self.assertEqual(relief.metadata['stage_state']['climate']['id'],eroded.metadata['stage_state']['climate']['id'])
        with patch.object(stages,'_execute',wraps=stages._execute) as execute:
            _,updated,_=self.run_stage(resolve_generation(relief_profile).settings,'climate',relief_profile)
        self.assertEqual([call.args[0] for call in execute.call_args_list],['climate'])
        self.assertFalse(updated.metadata['stage_state']['climate']['stale'])
        self.assertTrue(updated.metadata['stage_state']['erosion']['stale'])
        np.testing.assert_array_equal(updated.height_m,relief.height_m)

    def test_unchanged_stages_are_cached_and_tampered_snapshot_is_rejected(self):
        _,world,_=self.run_stage(self.settings,'all')
        with patch.object(stages,'_execute',side_effect=AssertionError('unchanged stage recalculated')):
            _,repeat,execution=self.run_stage(self.settings,'all')
        self.assertEqual(execution,dict(relief=False,erosion=False,climate=False))
        np.testing.assert_array_equal(repeat.height_m,world.height_m)
        key=world.metadata['stage_state']['relief']['id']
        artifact=stages.load_stage(key,'relief');arrays={k:v.copy() for k,v in artifact['arrays'].items()}
        arrays['snapshot'][0]^=1
        np.savez(stages._path(key),metadata=np.asarray(__import__('json').dumps(artifact['metadata'])),**arrays)
        with self.assertRaisesRegex(ValueError,'checksum'):stages.load_stage(key,'relief')

    def test_global_generation_does_not_load_previous_seed_or_stages(self):
        with patch.object(core,'get_heightmap',side_effect=AssertionError('previous world loaded')):
            self.run_stage(self.settings,'all','orogen')

    def test_gpu_and_city_stages_can_recompute_climate_independently(self):
        from terrain_orogen_gpu import implementation_identity
        if not implementation_identity()['available']:self.skipTest('Optional CUDA runtime unavailable')
        for engine in ('orogen','city-gpu'):
            with self.subTest(engine=engine):
                settings=dict(self.settings,relief_pipeline=engine,city_erosion_iterations=2,
                    **{key:True for key in OROGEN_GPU_PARAMETERS})
                profile,first,_=self.run_stage(settings,'all')
                changed=dict(resolve_generation(profile).settings,orogen_temperature_equator=32.)
                with patch.object(stages,'_execute',wraps=stages._execute) as execute:
                    _,second,_=self.run_stage(changed,'climate',profile)
                self.assertEqual([call.args[0] for call in execute.call_args_list],['climate'])
                np.testing.assert_array_equal(first.height_m,second.height_m)
                self.assertTrue(np.isfinite(second.layers['precip_summer']).all())


if __name__=='__main__':unittest.main()

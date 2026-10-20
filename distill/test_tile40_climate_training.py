import copy
import random
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from distill.student import StudentConfig, load_student
from distill.tile40_climate_run import export_inference, largest_width, terrain_summary, admissible, resume_checkpoint
from distill.tile40_conditioning import CLIMATES, RELIEFS
from distill.tile40_flow_network import TimeFlow
from distill.tile40_flow_train import climate_groups, select_climate_pairs, make_batch, loss_for


def references():
    rows = []
    yy,xx = torch.meshgrid(torch.arange(320),torch.arange(320),indexing='ij')
    for relief in RELIEFS:
        for climate_index,climate in enumerate(CLIMATES):
            coarse = np.zeros((6,24,24),np.float32)
            coarse[2:] = climate_index+1
            rows.append(dict(case=dict(split='train',seed=8100000+RELIEFS.index(relief),relief=relief,climate=climate),
                metadata=dict(coarse_origin_cells=[-8,-8],world_config=dict(parameters=dict(histogram_raw=[0.]*8))),
                coarse=coarse,latents=(yy+xx).float()[None].repeat(5,1,1)/320))
    return rows


def evaluation():
    rows = []
    for seed in (9100000,9100010):
        for relief in RELIEFS:
            for index,climate in enumerate(CLIMATES):
                rms = 10.+index
                rows.append(dict(case=dict(split='val',seed=seed+RELIEFS.index(relief),relief=relief,climate=climate),
                    metrics=dict(bands=[dict(amplitude_ratio=.8,prediction_rms_m=rms*.8,reference_rms_m=rms)],
                        directional_coherence=dict(prediction=.5,reference=.6),
                        prediction_stats=dict(slope_p90_degrees=30.),reference_stats=dict(slope_p90_degrees=35.))))
    return dict(status='complete',rows=rows)


class ClimateTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_maximum_fp32_width_and_real_inference_roundtrip(self):
        size = largest_width()
        self.assertEqual(size['width'],112)
        self.assertEqual(size['parameters'],24772725)
        self.assertLess(size['example_inference_bytes'],100000000)
        self.assertGreater(size['next_width_tensor_bytes'],100000000)
        config = StudentConfig(stage='base',width=16,depth=4)
        model = TimeFlow(config).eval()
        with torch.no_grad():
            model.head.weight.normal_(std=.01)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'inference.pt'
            size = export_inference(path,config,dict(residual_altitude=True),12,model.state_dict(),'ema',100000000)
            retained,saved = load_student(path)
            fields = torch.randn(1,32,32,32)
            fields[:,5].fill_(.4)
            with torch.no_grad():
                torch.testing.assert_close(model(fields),retained(fields),atol=0,rtol=0)
            self.assertEqual(saved['selected_weights'],'ema')
            self.assertNotIn('optimizer',saved)
            with self.assertRaisesRegex(ValueError,'strictly below'):
                export_inference(path,config,dict(residual_altitude=True),12,model.state_dict(),'ema',size)
            self.assertEqual(path.stat().st_size,size)

    def test_pairs_balance_climates_and_retain_hills_and_plains(self):
        groups = climate_groups(references())
        rng = random.Random(711)
        counts = dict.fromkeys(RELIEFS,0)
        climates = dict.fromkeys(CLIMATES,0)
        for _ in range(3000):
            a,b = select_climate_pairs(groups,rng,2,.7)
            self.assertEqual(a['case']['seed'],b['case']['seed'])
            self.assertNotEqual(a['case']['climate'],b['case']['climate'])
            counts[a['case']['relief']] += 1
            for row in (a,b):
                climates[row['case']['climate']] += 1
        self.assertAlmostEqual(counts['mountains']/3000,.7,delta=.025)
        self.assertGreater(min(counts.values()),250)
        self.assertLess(max(climates.values())-min(climates.values()),150)
        with self.assertRaisesRegex(ValueError,'TRAIN'):
            invalid = references()
            invalid[0]['case']['split']='test'
            climate_groups(invalid)
        with self.assertRaisesRegex(ValueError,'five climates'):
            climate_groups(references()[1:])

    def test_resume_uses_furthest_complete_state_after_a_hard_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            self.assertIsNone(resume_checkpoint(root))
            torch.save(dict(step=20),root/'latest.pt')
            torch.save(dict(step=2000),root/'step-002000.pt')
            self.assertEqual(resume_checkpoint(root).name,'step-002000.pt')
            torch.save(dict(step=2025),root/'latest.pt')
            self.assertEqual(resume_checkpoint(root).name,'latest.pt')

    def test_paired_batch_shares_crop_noise_and_time_and_backpropagates(self):
        rows = references()[:2]
        inputs,target = make_batch(rows,random.Random(814),torch.zeros(1,5,1,1),torch.ones(1,5,1,1),
                                  'cpu',size=96,paired_climates=True,residual_altitude=True)
        # Identical endpoints plus the same coarse altitude imply identical
        # states/velocities;climate feature channels still differ.
        torch.testing.assert_close(inputs[0,:10],inputs[1,:10],atol=0,rtol=0)
        torch.testing.assert_close(target[0],target[1],atol=0,rtol=0)
        self.assertGreater(float((inputs[0,12:16]-inputs[1,12:16]).abs().sum()),0.)
        model = TimeFlow(StudentConfig(stage='base',width=16,depth=4))
        loss,_ = loss_for(model,inputs,target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.head.weight.grad.abs().sum()),0.)
        with self.assertRaisesRegex(ValueError,'same geometry'):
            make_batch([rows[0],references()[5]],random.Random(1),0.,1.,'cpu',size=96,paired_climates=True)

    def test_selection_exposes_climate_collapse_and_rejects_one_bad_climate(self):
        report = evaluation()
        baseline = terrain_summary(report)
        collapsed = copy.deepcopy(report)
        for row in collapsed['rows']:
            row['metrics']['bands'][0]['prediction_rms_m']=10.
        actual = terrain_summary(collapsed)
        self.assertGreater(actual['mountain_climates']['arid']['climate_spectral_response_error'],.1)
        improved = copy.deepcopy(baseline)
        improved['groups']['mountains'] *= .8
        self.assertTrue(admissible(improved,baseline))
        improved['mountain_climates']['cold']['morphology_distance'] *= 1.2
        self.assertFalse(admissible(improved,baseline))
        report['rows'][0]['case']['split']='test'
        with self.assertRaisesRegex(ValueError,'TEST'):
            terrain_summary(report)


if __name__=='__main__':
    unittest.main()

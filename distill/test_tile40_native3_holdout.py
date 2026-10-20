import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from distill.tile40_conditioning import CLIMATES,RELIEFS
from distill.tile40_climate_run import terrain_summary
from distill.tile40_native3_test_summary import physical_summary_test,terrain_summary_test


class HeldoutMetricsTests(unittest.TestCase):
    def report(self):
        rows=[]
        for relief_index,relief in enumerate(RELIEFS):
            for geometry in range(2):
                for climate_index,climate in enumerate(CLIMATES):
                    case=dict(split='test',seed=1000+relief_index*10+geometry,
                        relief=relief,climate=climate,file=f'{relief}-{geometry}-{climate}.npz')
                    bands=[dict(amplitude_ratio=1.,prediction_rms_m=1.+climate_index*.1,
                        reference_rms_m=1.+climate_index*.1,correlation=1.) for _ in range(4)]
                    metrics=dict(height_mae_m=0.,bands=bands,directional_coherence=dict(prediction=1.,reference=1.),
                        prediction_stats=dict(slope_p90_degrees=1.),reference_stats=dict(slope_p90_degrees=1.))
                    rows.append(dict(case=case,metrics=metrics,paired_spatial_band_correlation=1.))
        return dict(status='complete',rows=rows)

    def test_test_schema_is_explicit_and_numeric_formulas_match_validation(self):
        report=self.report()
        test=terrain_summary_test(report)
        validation=copy.deepcopy(report)
        for row in validation['rows']:row['case']['split']='val'
        self.assertEqual(test,terrain_summary(validation))
        with self.assertRaises(ValueError):terrain_summary_test(validation)
        with self.assertRaises(ValueError):terrain_summary(report)

    def test_perfect_heldout_climate_response_passes_without_relabelling(self):
        report=self.report()
        with tempfile.TemporaryDirectory() as directory:
            for row in report['rows']:
                climate_index=list(CLIMATES).index(row['case']['climate'])
                height=np.arange(64,dtype=np.float32).reshape(8,8)*(1+climate_index*.2)
                np.savez(Path(directory)/row['case']['file'],height_m=height,reference_height_m=height)
            summary=physical_summary_test(report,directory)
        self.assertTrue(summary['near_original_numeric'])
        self.assertEqual(summary['groups']['mountains'],0.)
        self.assertTrue(all(value['relative_rmse']==0. for value in summary['climate_height_response'].values()))
        self.assertTrue(all(row['case']['split']=='test' for row in report['rows']))


if __name__=='__main__':unittest.main()

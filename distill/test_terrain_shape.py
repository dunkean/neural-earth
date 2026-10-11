"""Morphology checks catch displaced valleys despite matching power and slopes."""
import unittest
import numpy as np

from distill.terrain_shape import compare_shape


class TerrainShapeTests(unittest.TestCase):
    def terrain(self):
        yy,xx=np.indices((128,128),dtype=np.float64)
        return 500.-80.*np.exp(-((xx-64.-6.*np.sin(yy/12.))/4.)**2)+10.*np.cos(yy/10.)

    def test_shifted_valleys_lose_position_agreement(self):
        reference=self.terrain()
        exact=compare_shape(reference,reference)
        shifted=compare_shape(np.roll(reference,16,axis=1),reference)
        for row in exact['topographic_position']:
            self.assertAlmostEqual(row['tpi_correlation'],1.)
            self.assertEqual(row['valley_iou'],1.)
        self.assertLess(shifted['topographic_position'][0]['tpi_correlation'],.8)
        self.assertLess(shifted['topographic_position'][0]['valley_retention'],.8)

    def test_constant_height_bias_preserves_the_same_local_shape(self):
        reference=self.terrain()
        result=compare_shape(reference+20.,reference)
        self.assertAlmostEqual(result['gradient_direction_cosine'],1.)
        self.assertLess(max(row['tpi_mae_m'] for row in result['topographic_position']),1e-10)

    def test_sea_and_flat_cases_do_not_claim_valley_evidence(self):
        result=compare_shape(np.zeros((128,128)),np.zeros((128,128)))
        self.assertIsNone(result['gradient_direction_cosine'])
        self.assertFalse(any(row['available'] for row in result['topographic_position']))


if __name__=='__main__':
    unittest.main()

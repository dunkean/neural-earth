from pathlib import Path
import os
import tempfile
import unittest

from tools.verification.compare_base_variants import shared_stage_paths,resolve_evaluation_paths


class MatchedReferenceTests(unittest.TestCase):
    def test_coarse_is_shared_by_reference_and_native_student(self):
        paths = shared_stage_paths(['reference','native_student'],conditioning_coarse=Path('coarse2.pt'))
        self.assertEqual(paths,dict(coarse=Path('coarse2.pt')))

    def test_app_stack_shares_both_stages_with_fp32_floor_control(self):
        paths = shared_stage_paths(['reference','fp32base','native_student'],
            conditioning_coarse=Path('coarse4.pt'),conditioning_decoder=Path('decoder.pt'))
        self.assertEqual(set(paths),{'coarse','decoder'})

    def test_legacy_original_reference_stays_original(self):
        self.assertEqual(shared_stage_paths(['reference','student_all'],coarse_student=Path('coarse4.pt')), {})

    def test_candidate_coarse_cannot_override_the_shared_reference(self):
        with self.assertRaisesRegex(ValueError,'Shared-stage comparison'):
            shared_stage_paths(['reference','native_student'],conditioning_coarse=Path('coarse2.pt'),
                               coarse_student=Path('coarse4.pt'))

    def test_native_oracles_use_the_same_shared_coarse_and_decoder(self):
        paths=shared_stage_paths(['reference','native_teacher_height','native_teacher_features','native_teacher_all'],
            conditioning_coarse=Path('coarse2.pt'),conditioning_decoder=Path('decoder.pt'))
        self.assertEqual(set(paths),{'coarse','decoder'})

    def test_relative_weights_survive_backend_geography_chdir(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'upstream').mkdir()
            try:
                os.chdir(root)
                paths = resolve_evaluation_paths(coarse=Path('weights/coarse4.pt'),native=None)
                os.chdir(root/'upstream')
                self.assertEqual(paths['coarse'],root/'weights/coarse4.pt')
                self.assertTrue(paths['coarse'].is_absolute())
                self.assertIsNone(paths['native'])
            finally:
                os.chdir(previous)


if __name__=='__main__':
    unittest.main()

import copy
import unittest

from distill.native_multilod_eval import validation_records


class MatchedValidationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = dict(original_manifest_sha256='original', scenes=[
            dict(metadata=dict(split='train', seed=1000)),
            dict(metadata=dict(split='val', seed=2000))])
        self.statistics = dict(split='train', dataset_manifest_sha256='original')

    def test_validation_variance_cannot_replace_train_variance(self):
        with self.assertRaises(ValueError):
            validation_records(self.manifest, dict(self.statistics, split='val'))
        with self.assertRaises(ValueError):
            validation_records(self.manifest, dict(self.statistics, dataset_manifest_sha256='other'))

    def test_coarse_branch_does_not_allow_a_world_in_both_splits(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['scenes'].append(dict(coarse_variant='coarse2', metadata=dict(split='val', seed=1000)))
        with self.assertRaises(ValueError):
            validation_records(manifest, self.statistics)

    def test_reserved_final_world_cannot_enter_validation(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['scenes'][1]['metadata']['seed'] = 42
        with self.assertRaises(ValueError):
            validation_records(manifest, self.statistics)


if __name__ == '__main__':
    unittest.main()

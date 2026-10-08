"""Reject benchmark fallbacks and heterogeneous paired provenance."""
import unittest
from benchmark_inference_engine import verify_child


class ReceiptTests(unittest.TestCase):
    def receipt(self, effective=True):
        return dict(status='complete', require_exact=True, reference={'sha256':'r'},
                    source_sha256={'kernel':'s'}, samples=[dict(key='lod3_case1',
                        after_inference={'pointwise_kernels':dict(requested=True,
                            effective_by_device={'0':effective}, runtime={'wrapper_sha256':'s'})})])

    def test_fallback_is_not_a_speedup(self):
        with self.assertRaisesRegex(ValueError, 'fell back'):
            verify_child(self.receipt(False), 'exact', {'sha256':'r'}, None)

    def test_changed_sources_or_reference_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'source files changed'):
            verify_child(self.receipt(), 'exact', {'sha256':'r'}, {'kernel':'changed'})
        with self.assertRaisesRegex(ValueError, 'Reference archive changed'):
            verify_child(self.receipt(), 'exact', {'sha256':'changed'}, None)

    def test_engine_evidence_is_required_on_each_sample(self):
        child=self.receipt()
        self.assertEqual(verify_child(child, 'exact', {'sha256':'r'}, {'kernel':'s'})[0]['key'], 'lod3_case1')
        with self.assertRaisesRegex(ValueError, 'Reference process enabled'):
            verify_child(child, 'reference', {'sha256':'r'}, None)
        child['samples'][0]['after_inference']['pointwise_kernels']['requested']=False
        verify_child(child, 'reference', {'sha256':'r'}, None)


if __name__=='__main__':
    unittest.main()

import unittest

from distill.native_multilod_next import verify_smoke


class SmokeGateTests(unittest.TestCase):
    def proof(self):
        return dict(status='complete',original_decoder_frozen=True,rows=[
            dict(coarse_variant=v,loss=1.,grad_norm=.1,window_gradients=[[.1]*25,[.2]*9])
            for v in ['original','coarse4','coarse2']])

    def test_full_matched_gradient_proof_is_accepted(self):
        verify_smoke(self.proof())

    def test_one_disconnected_initial_window_is_rejected(self):
        proof=self.proof();proof['rows'][1]['window_gradients'][0][0]=0.
        with self.assertRaisesRegex(ValueError,'all34'):
            verify_smoke(proof)

    def test_missing_coarse2_cannot_start_training(self):
        proof=self.proof();proof['rows'].pop()
        with self.assertRaisesRegex(ValueError,'each frozen coarse'):
            verify_smoke(proof)

    def test_trainable_decoder_is_rejected(self):
        proof=self.proof();proof['original_decoder_frozen']=False
        with self.assertRaisesRegex(ValueError,'did not complete'):
            verify_smoke(proof)


if __name__=='__main__':
    unittest.main()

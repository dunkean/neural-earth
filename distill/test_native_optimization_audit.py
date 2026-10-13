import unittest

from distill.native_optimization_audit import block, diagnostic_reasons, morphology_guard, validate_parent_contract


def report(drops):
    rows = []
    for seed, drop in enumerate(drops):
        value = dict(valley_retention=.6-drop, tpi_correlation=.9)
        rows.append(dict(file=f'case-{seed}', seed=seed, coarse_variant='coarse4',
            case_kind='temperate-plain', profile='test', morphology=dict(fine=value, native=value)))
    return dict(rows=rows)


class OptimizationGuardTests(unittest.TestCase):
    def test_consistent_raw_regression_names_the_affected_group(self):
        reasons = morphology_guard(report([.05]*4), report([0.]*4))
        self.assertTrue(reasons)
        self.assertTrue(all(r['group'] == 'coarse4:temperate-plain' for r in reasons))
        self.assertTrue(all(r['lower95_drop'] > r['tolerance'] for r in reasons))

    def test_uncertain_small_changes_do_not_stop_the_pilot(self):
        self.assertEqual(morphology_guard(report([.02, -.02, -.02, .02]), report([0.]*4)), [])

    def test_undefined_candidate_morphology_is_not_discarded(self):
        candidate = report([0.]*4); candidate['rows'][0]['morphology']['fine'] = None
        reasons = morphology_guard(candidate, report([0.]*4))
        self.assertTrue(reasons)
        self.assertIn('undefined', reasons[0]['reason'])

    def test_diagnostic_only_native_recall_becomes_warning(self):
        recall = dict(scale='native', metric='valley_retention')
        fine = dict(scale='fine', metric='valley_retention')
        position = dict(scale='native', metric='tpi_correlation')
        undefined = dict(scale='native', reason='defined source morphology became undefined')
        reasons = [recall, fine, position, undefined]
        self.assertEqual(diagnostic_reasons(reasons, {}, {}), [fine, position, undefined])
        self.assertEqual(reasons[0], recall)  # Original failure is preserved.

    def test_untargeted_warning_does_not_hide_catastrophic_change(self):
        reason = dict(reason='large untargeted channel regression', variant='coarse4', phase=0, channel=4)
        def controls(value):
            return dict(operator_control={'coarse4': dict(channel_mse=[[1., 1., 1., 1., value], [1.]*5])})
        self.assertEqual(diagnostic_reasons([reason], controls(1.3), controls(1.)), [])
        hard = diagnostic_reasons([reason], controls(1.6), controls(1.))
        self.assertEqual(len(hard), 1)
        self.assertEqual(hard[0]['ratio'], 1.6)

    def test_output_gain_is_a_separate_audit_group(self):
        self.assertEqual(block('net.out_gain'), 'output_gain')
        self.assertEqual(block('net.out_conv.weight'), 'output_conv')


class ResumeContractTests(unittest.TestCase):
    def parent(self):
        return dict(lr=.0001, weights={'channel0': 2.}, image_label='parent',
                    producer_sha256={'distill/native_optimization_train.py': 'old',
                                     'distill/native_optimization_audit.py': 'old',
                                     'distill/native_phase_loss.py': 'frozen'})

    def test_guard_and_audit_change_can_continue_exact_numeric_recipe(self):
        parent = self.parent()
        child = dict(parent, image_label='diagnostic', producer_sha256={
            'distill/native_optimization_train.py': 'new',
            'distill/native_optimization_audit.py': 'new', 'distill/native_phase_loss.py': 'frozen'})
        validate_parent_contract(parent, child)

    def test_learning_rate_change_rejects_optimizer_resume(self):
        parent = self.parent()
        with self.assertRaisesRegex(ValueError, 'lr'):
            validate_parent_contract(parent, dict(parent, lr=.001))

    def test_loss_producer_change_rejects_optimizer_resume(self):
        parent = self.parent()
        child = dict(parent, producer_sha256=dict(parent['producer_sha256'], **{'distill/native_phase_loss.py': 'changed'}))
        with self.assertRaisesRegex(ValueError, 'native_phase_loss'):
            validate_parent_contract(parent, child)


if __name__ == '__main__':
    unittest.main()

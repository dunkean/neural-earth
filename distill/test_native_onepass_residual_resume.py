"""Continuation preserves the physical recipe and starts at the inherited LR."""
import unittest

from distill.native_onepass_residual_resume import continuation_lr, validate_parent_contract


class ResumeTests(unittest.TestCase):
    def contract(self):
        return dict(source_sha256='parent', seed=10, solver_evaluations=1,
            teacher_solver_evaluations=2, noise_offsets=[5819, 5820], ema_half_life=500,
            loss='matched positions', sampler='TRAIN only', teacher_revision='pinned',
            code_sha256={'native_onepass_residual_train.py': 'old', 'model.py': 'same'})

    def test_explicit_extension_accepts_new_budget_and_trainer(self):
        old = self.contract()
        new = self.contract() | dict(steps=20000, scheme='nf4')
        new['code_sha256'] = new['code_sha256'] | {'native_onepass_residual_train.py': 'new'}
        validate_parent_contract(old, new)

    def test_extension_rejects_changed_targets_or_model(self):
        for key, value in [('loss', 'different'), ('teacher_revision', 'different')]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_parent_contract(self.contract(), self.contract() | {key: value})
        new = self.contract()
        new['code_sha256']['model.py'] = 'changed'
        with self.assertRaises(ValueError):
            validate_parent_contract(self.contract(), new)

    def test_extension_inherits_lr_without_warmup_and_reaches_floor(self):
        self.assertEqual(continuation_lr(750, 750, 20000, 9e-6), 9e-6)
        self.assertEqual(continuation_lr(20000, 750, 20000, 9e-6), 1e-6)
        values = [continuation_lr(i, 750, 20000, 9e-6) for i in range(750, 20001, 250)]
        self.assertTrue(all(a >= b for a, b in zip(values, values[1:])))

    def test_extension_rejects_invalid_schedule(self):
        for values in [(500, 750, 20000, 9e-6), (750, 750, 750, 9e-6),
                       (20001, 750, 20000, 9e-6), (750, 750, 20000, 0.)]:
            with self.subTest(values=values), self.assertRaises(ValueError):
                continuation_lr(*values)


if __name__ == '__main__':
    unittest.main()

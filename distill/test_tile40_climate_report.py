import copy
import unittest

from distill.tile40_climate_report import completed_steps, briefing, render


def entry(step,weights,score=.2,admissible=False):
    climates={name:dict(morphology_distance=score,climate_spectral_response_error=.05)
        for name in ('arid','cold','tropical','temperate','oceanic')}
    return dict(step=step,weights=weights,summary=dict(score=score,groups=dict(mountains=score),
        mountain_climates=climates),admissible_vs_baseline=admissible,report=f'/eval/{step}/{weights}/report.json')


class ScoreReportTests(unittest.TestCase):
    def test_waits_for_both_versions_and_reports_only_new_checkpoints(self):
        history=[entry(2000,'raw'),entry(2000,'ema'),entry(4000,'raw')]
        self.assertEqual(completed_steps(history),[2000])
        baseline=entry(0,'raw',.1)['summary']
        with self.assertRaisesRegex(ValueError,'both raw and EMA'):
            briefing(history,baseline,4000)
        history.append(entry(4000,'ema',.08,True))
        report=briefing(history,baseline,4000,2000)
        self.assertEqual([row['step'] for row in report['rows']],[4000,4000])
        self.assertEqual(report['latest_selection']['weights'],'ema')
        self.assertAlmostEqual(report['rows'][0]['mountain_gain_percent'],20.)
        text=render(report)
        self.assertIn('plus bas = mieux',text)
        self.assertIn('froid',text)
        self.assertIn('+20.0 %',text)

    def test_catchup_report_excludes_future_winners(self):
        history=[entry(step,weight,score,True) for step,score in ((2000,.08),(4000,.07),(6000,.01))
            for weight in ('raw','ema')]
        baseline=entry(0,'raw',.1)['summary']
        report=briefing(history,baseline,4000)
        self.assertEqual(report['best_admissible']['step'],4000)
        self.assertEqual(len(report['rows']),4)

    def test_failed_candidates_do_not_replace_the_reference(self):
        baseline=entry(0,'raw',.1)['summary']
        history=[entry(2000,weight,.3) for weight in ('raw','ema')]
        report=briefing(history,baseline,2000)
        self.assertIsNone(report['best_admissible'])
        self.assertLess(report['rows'][0]['mountain_gain_percent'],0.)
        self.assertIn('référence48/8k reste retenue',render(report))


if __name__=='__main__':
    unittest.main()

import threading
import time
import unittest
from terrain_background import CoarseBackground
from terrain_jobs import TerrainJobs


class BackgroundTests(unittest.TestCase):
    def test_disk_budget_exhaustion_stops_future_quanta(self):
        jobs=TerrainJobs()
        calls=[]
        def coarse(seed,profile):
            calls.append((seed,profile))
            return {'complete_windows':1,'total_windows':100,
                    'disk_budget_exhausted':True}
        background=CoarseBackground(jobs,coarse,None)
        try:
            background.start(42,'natural',max_windows=100)
            deadline=time.monotonic()+3
            while background.status()['state']=='running':
                if time.monotonic()>deadline:self.fail('background did not stop at disk budget')
                time.sleep(.01)
            self.assertEqual(background.status()['state'],'disk-budget-exhausted')
            self.assertEqual(background.status()['quanta'],1)
            self.assertEqual(calls,[(42,'natural')])
        finally:
            background.close()
            jobs.close()

    def test_visible_work_precedes_next_background_quantum(self):
        jobs=TerrainJobs()
        entered,release=threading.Event(),threading.Event()
        order=[]
        def hold():
            entered.set()
            self.assertTrue(release.wait(3))
        blocker=jobs.submit('running-visible',hold,priority=0)
        self.assertTrue(entered.wait(2))
        def coarse(seed,profile):
            order.append('coarse')
            return {'complete_windows':len(order),'total_windows':100}
        background=CoarseBackground(jobs,coarse,None)
        try:
            background.start(0,'natural',max_windows=2)
            deadline=time.monotonic()+2
            while not any(j['key'].startswith('coarse-preparation') for j in jobs.status()['jobs']):
                if time.monotonic()>deadline:self.fail('background did not enqueue')
                time.sleep(.01)
            urgent=jobs.submit('teleport-visible',lambda:order.append('visible'),priority=0)
            release.set()
            blocker.wait(3)
            urgent.wait(3)
            deadline=time.monotonic()+2
            while background.status()['state']=='running':
                if time.monotonic()>deadline:self.fail('queue did not drain')
                time.sleep(.01)
            self.assertEqual(order,['visible','coarse','coarse'])
            self.assertEqual(background.status()['state'],'budget-complete')
        finally:
            release.set()
            background.close()
            jobs.close()


if __name__=='__main__':unittest.main()

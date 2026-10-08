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
            urgent=jobs.submit('teleport-visible-sea',lambda:order.append('visible'),priority=4999)
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

    def test_full_world_runs_to_completion_and_repeated_start_is_idempotent(self):
        jobs=TerrainJobs()
        entered,release=threading.Event(),threading.Event()
        calls=[]
        def coarse(seed,profile):
            calls.append((seed,profile))
            if len(calls)==1:
                entered.set()
                self.assertTrue(release.wait(3))
            return {'complete_windows':len(calls),'total_windows':5}
        background=CoarseBackground(jobs,coarse,None)
        try:
            background.start(42,'natural')
            self.assertTrue(entered.wait(2))
            token=background.generation
            background.start(42,'natural')
            self.assertEqual(background.generation,token)
            background.set_focus(42,'natural',[1,2,3,4])
            self.assertEqual(background.focus(42,'natural'),(1,2,3,4))
            self.assertIsNone(background.focus(43,'natural'))
            release.set()
            deadline=time.monotonic()+3
            while background.status()['state']=='running':
                if time.monotonic()>deadline:self.fail('whole world never completed')
                time.sleep(.01)
            self.assertEqual(background.status()['state'],'complete')
            self.assertEqual(background.status()['quanta'],5)
            self.assertIsNone(background.status()['budget'])
        finally:
            release.set();background.close();jobs.close()

    def test_pause_stops_after_the_current_window(self):
        jobs=TerrainJobs()
        entered,release=threading.Event(),threading.Event()
        calls=[]
        def coarse(seed,profile):
            calls.append(1);entered.set();release.wait(3)
            return {'complete_windows':1,'total_windows':100}
        background=CoarseBackground(jobs,coarse,None)
        try:
            background.start(42,'natural')
            self.assertTrue(entered.wait(2))
            self.assertEqual(background.stop()['state'],'stopped')
            release.set()
            jobs.close()
            self.assertEqual(len(calls),1)
        finally:
            release.set();background.close()


if __name__=='__main__':unittest.main()

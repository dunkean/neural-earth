"""Independent CPU previews, deduplication and opt-in timeline integrity."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import threading
import unittest
from terrain_jobs import TerrainJobs
import terrain_profiling as profiling
import numpy as np
from terrain_delivery import PhysicalDelivery
from terrain_jobs import JobCancelled


class PerformanceTests(unittest.TestCase):
    def test_withdrawn_cpu_interest_is_checked_after_admission_backpressure(self):
        queue=TerrainJobs(cpu_workers=1)
        entered,called=threading.Event(),threading.Event()
        # Hold every encoding slot so the worker has admitted the job but
        # cannot compute. A camera withdrawal must still stop the CPU work.
        for _ in range(2):queue.preview_slots.acquire()
        try:
            queue.update_view('preview-session',1,{'preview':0})
            job=queue.submit('preview',lambda:called.set(),session='preview-session',epoch=1,lane='cpu')
            with queue.condition:
                self.assertTrue(queue.condition.wait_for(lambda:job.state!='queued',timeout=1))
            queue.release_view('preview-session')
            queue.preview_slots.release()
            with self.assertRaises(JobCancelled):job.wait(1)
            self.assertFalse(called.is_set())
        finally:
            queue.preview_slots.release();queue.close()

    def test_ram_delivery_is_immediate_bounded_and_disk_writes_keep_order(self):
        delivery=PhysicalDelivery(max_bytes=32,max_pending=2)
        entered,release=threading.Event(),threading.Event()
        order=[]
        values=(np.zeros(4,np.float32),)
        def first():
            entered.set();release.wait(3);order.append(1)
        try:
            delivery.publish('tile','path',{'version':1},values,first)
            self.assertTrue(entered.wait(1))
            delivery.publish('tile','path',{'version':2},values,lambda:order.append(2))
            delivery.publish('other','path',{},values,lambda:order.append(3))
            self.assertEqual(delivery.get('tile')[1]['version'],2)
            self.assertEqual(delivery.status()['pending'],2)
            self.assertEqual(delivery.status()['skipped'],1)
            delivery.publish('third','path',{},values,lambda:None)
            self.assertLessEqual(delivery.status()['bytes'],32)
            release.set();delivery.flush()
            self.assertEqual(order,[1,2])
            def fail():raise OSError('disk failure')
            delivery.publish('failure','path',{},values,fail)
            delivery.flush()
            self.assertEqual(delivery.status()['failed'],1)
            self.assertIn('disk failure',delivery.status()['errors'])
        finally:
            release.set();delivery.close()
    def test_cpu_preview_completes_while_gpu_compute_is_blocked(self):
        queue=TerrainJobs()
        entered,release=threading.Event(),threading.Event()
        def compute():
            entered.set()
            release.wait(3)
            return 'gpu'
        try:
            gpu=queue.submit('gpu',compute)
            self.assertTrue(entered.wait(1))
            cpu=queue.submit('preview',lambda:'cpu',lane='cpu')
            self.assertEqual(cpu.wait(1),'cpu')
            self.assertFalse(gpu.event.is_set())
            release.set()
            self.assertEqual(gpu.wait(1),'gpu')
        finally:
            release.set()
            queue.close()

    def test_preview_finalizer_backpressure_cannot_block_gpu_lane(self):
        queue=TerrainJobs(cpu_workers=1)
        entered,release=threading.Event(),threading.Event()
        def finish(value):
            entered.set(); release.wait(3); return value
        try:
            jobs=[queue.submit(f'cpu{i}',lambda:1,finish,lane='cpu') for i in range(3)]
            self.assertTrue(entered.wait(1))
            self.assertEqual(queue.submit('gpu',lambda:2).wait(1),2)
        finally:
            release.set(); queue.close()

    def test_timeline_is_bounded_disabled_by_default_and_exception_is_visible(self):
        previous=(profiling.ENABLED,profiling.CUDA)
        try:
            profiling.set_enabled(True)
            with profiling.trace('test'):
                with self.assertRaises(ValueError), profiling.span('failing'):
                    raise ValueError('expected')
            data=profiling.snapshot()
            event=next(e for e in data['traceEvents'] if e['name']=='failing')
            self.assertEqual(event['trace'],'test')
            self.assertTrue(event['args']['failed'])
            self.assertGreaterEqual(event['dur'],0)
            for i in range(9000):
                profiling.instant('bounded',i=i)
            self.assertEqual(len(profiling.snapshot()['traceEvents']),8192)
            self.assertGreater(profiling.snapshot()['dropped_events'],0)
            profiling.set_enabled(False)
            profiling.instant('absent')
            self.assertNotIn('absent',[e['name'] for e in profiling.snapshot()['traceEvents']])
        finally:
            profiling.set_enabled(*previous)


if __name__=='__main__':
    unittest.main()

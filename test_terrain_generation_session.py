"""Exercise cancellation and real Flask handlers without CUDA/checkpoint startup."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from flask import Flask, jsonify, request
from terrain_generation_session import (GenerationCoordinator, GenerationToken,
    GenerationCancelled, generation_scope, check_generation, run_process)
from terrain_jobs import TerrainJobs, JobCancelled


class GenerationSessionTests(unittest.TestCase):
    def setUp(self):
        self.jobs=TerrainJobs()
        self.addCleanup(self.jobs.close)
        self.background=Mock()
        self.background.start.return_value={'state':'running'}
        self.coordinator=GenerationCoordinator(self.jobs,self.background)
        self.app=Flask(__name__)
        self.app.testing=True
        self.ran=[]
        self.metadata=Mock(side_effect=lambda seed,profile:{'seed':str(seed),'world_profile':profile})
        namespace=dict(app=self.app,request=request,jsonify=jsonify,subprocess=subprocess,
            jobs=self.jobs,coarse_background=self.background,generation_coordinator=self.coordinator,
            GenerationCancelled=GenerationCancelled,generation_scope=generation_scope,
            gpu_lock=threading.RLock(),SELECTED_DEVICE=0,
            torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=Mock())),
            resolve_generation=lambda profile:SimpleNamespace(base_profile=profile),
            generation_profile=lambda profile=None:profile or request.args.get('world_profile','natural'),world_manifest=lambda *args:None,
            json=json,
            register_generation=lambda base,settings:base,metadata=self.metadata)
        names={'generation_run','world_info','coarse_prepare','finish_generation','suspend_terrain_admission'}
        tree=ast.parse(Path('terrain_server.py').read_text(encoding='utf-8'))
        exec(compile(ast.Module(body=[n for n in tree.body if getattr(n,'name',None) in names],
                              type_ignores=[]),'terrain_server.py','exec'),namespace)
        self.fake_stages=SimpleNamespace(run_generation=self.run_stage)
        self.patch=patch.dict(sys.modules,{'terrain_orogen_stages':self.fake_stages})
        self.patch.start();self.addCleanup(self.patch.stop)

    def run_stage(self, seed, *args):
        self.ran.append(seed)
        return {},{}

    def test_new_request_cancels_previous_stage_and_blocks_all_terrain(self):
        first_entered,second_entered,release=threading.Event(),threading.Event(),threading.Event()
        responses={}
        def stage(seed,*args):
            self.ran.append(seed)
            if seed==1:
                first_entered.set()
                while True:
                    check_generation();time.sleep(.005)
            second_entered.set()
            self.assertTrue(release.wait(3))
            return {},{}
        self.fake_stages.run_generation=stage
        def post(seed):
            with self.app.test_client() as client:
                responses[seed]=client.post('/api/generation/run',json={
                    'seed':seed,'session':'session-test','generation_epoch':seed})
        first=threading.Thread(target=post,args=(1,));second=threading.Thread(target=post,args=(2,))
        first.start()
        try:
            self.assertTrue(first_entered.wait(3));second.start()
            self.assertTrue(second_entered.wait(3));first.join(3)
            self.assertEqual(responses[1].status_code,409)
            self.assertTrue(self.jobs.paused,'Old cleanup must not resume the latest generation')
            with self.app.test_client() as client:
                self.assertEqual(client.post('/api/coarse/prepare',json={'seed':2}).status_code,409)
                self.assertEqual(client.get('/coarse/natural-v1/1/0/0.bin').status_code,409)
                self.assertEqual(client.post('/api/view',json={}).status_code,409)
            with self.assertRaises(JobCancelled):self.jobs.submit('old-world',lambda:None)
            release.set();second.join(3)
            self.assertEqual(responses[2].status_code,200)
            self.assertEqual(self.metadata.call_count,1)
            self.assertEqual(self.metadata.call_args.args[0],2)
            self.assertFalse(self.jobs.paused)
            with self.app.test_client() as client:
                self.assertEqual(client.post('/api/coarse/prepare',json={'seed':2}).status_code,202)
        finally:
            release.set();first.join(3)
            if second.ident:second.join(3)

    def test_out_of_order_requests_do_not_cancel_latest(self):
        token=self.coordinator.begin('session-test',5)
        with self.app.test_client() as client:
            response=client.post('/api/generation/run',json={'session':'session-test','generation_epoch':4})
        self.assertEqual(response.status_code,409)
        self.assertIs(self.coordinator.current,token)
        self.assertFalse(token.cancelled.is_set())
        self.assertEqual(self.ran,[])
        self.coordinator.finish(token)

    def test_generation_also_cancels_initial_world_bootstrap(self):
        entered=threading.Event();responses=[]
        def metadata(seed,profile):
            if seed==1:
                entered.set()
                while True:
                    check_generation();time.sleep(.005)
            return {'seed':str(seed)}
        self.metadata.side_effect=metadata
        def load():
            with self.app.test_client() as client:
                responses.append(client.get('/api/world?seed=1&session=session-test&generation_epoch=1'))
        thread=threading.Thread(target=load);thread.start()
        self.assertTrue(entered.wait(3))
        with self.app.test_client() as client:
            response=client.post('/api/generation/run',json={'seed':2,'session':'session-test','generation_epoch':2})
        thread.join(3)
        self.assertEqual(response.status_code,200)
        self.assertEqual(responses[0].status_code,409)
        self.assertFalse(self.jobs.paused)

    def test_pause_cancels_running_legacy_and_queued_coarse(self):
        entered,release=threading.Event(),threading.Event()
        def compute():
            entered.set();release.wait(3);self.jobs.check_current_interest()
        running=self.jobs.submit('legacy-window',compute)
        try:
            self.assertTrue(entered.wait(3))
            queued=self.jobs.submit('coarse-preparation/old',lambda:self.fail('Old coarse ran'),priority=5000)
            token=self.coordinator.begin()
            release.set()
            with self.assertRaises(JobCancelled):queued.wait(3)
            with self.assertRaises(JobCancelled):running.wait(3)
            self.coordinator.finish(token)
            self.assertEqual(self.jobs.submit('fresh',lambda:42).wait(3),42)
        finally:release.set()

    def test_cancel_terminates_its_subprocess_without_waiting_for_full_stage(self):
        token=GenerationToken();errors=[]
        def run():
            try:
                with generation_scope(token):
                    run_process([sys.executable,'-c','import time; time.sleep(30)'],capture_output=True)
            except GenerationCancelled as exc:errors.append(exc)
        worker=threading.Thread(target=run);worker.start()
        deadline=time.monotonic()+3
        while not token.processes and time.monotonic()<deadline:time.sleep(.005)
        self.assertTrue(token.processes)
        token.cancel();worker.join(3)
        self.assertFalse(worker.is_alive());self.assertEqual(len(errors),1)
        self.assertEqual(token.processes,set())

    def test_every_stage_is_exclusive_and_reference_worker_is_suspended(self):
        reference=Mock();self.app.extensions['terrain_reference_worker']=reference
        def stage(*args):
            self.assertTrue(self.jobs.paused)
            return {},{}
        self.fake_stages.run_generation=stage
        with self.app.test_client() as client:
            for stage in ('all','relief','erosion','climate','settings','restore'):
                self.assertEqual(client.post('/api/generation/run',json={'stage':stage,'settings':{}}).status_code,200)
                self.assertFalse(self.jobs.paused)
        self.assertEqual([call.args[0] for call in reference.suspend.call_args_list],[True,False]*6)

    def test_generation_failure_releases_admission(self):
        self.fake_stages.run_generation=Mock(side_effect=RuntimeError('stage failed'))
        with self.app.test_client() as client:
            self.assertEqual(client.post('/api/generation/run',json={}).status_code,503)
        self.assertFalse(self.jobs.paused)
        self.assertIsNone(self.coordinator.current)

    def test_parent_gpu_reservation_cancels_reference_bootstrap_and_stays_paused(self):
        token=self.coordinator.begin()
        self.coordinator.suspend(True)
        self.assertTrue(token.cancelled.is_set())
        self.coordinator.finish(token)
        self.assertTrue(self.jobs.paused)
        with self.assertRaises(GenerationCancelled):self.coordinator.begin()
        self.coordinator.suspend(False)
        self.assertFalse(self.jobs.paused)


if __name__=='__main__':unittest.main()

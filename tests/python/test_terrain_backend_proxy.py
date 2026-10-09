"""CPU-only proxy/lifecycle tests: only a sleeping Python child, never CUDA."""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

from concurrent.futures import ThreadPoolExecutor
from email.message import Message
import io
import os
import json
import ctypes
import subprocess
import sys
import socket
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock,patch
from urllib.error import HTTPError

from flask import Flask
import terrain_backend_proxy as proxy


class FakeProcess:
    pid=12345
    def __init__(self):
        self.code=None;self.terminated=0
    def poll(self): return self.code
    def terminate(self): self.terminated+=1;self.code=0
    def kill(self): self.code=-9
    def wait(self,timeout): return self.code


class Response:
    def __init__(self,data=b'ok',status=200,headers=None):
        self.code=status;self.data=data;self.headers=Message()
        for key,value in (headers or {'Content-Type':'application/octet-stream'}).items():
            self.headers[key]=value
    def read(self,*args): return self.data
    def __enter__(self): return self
    def __exit__(self,*args): pass


class ProxyTests(unittest.TestCase):
    def test_suspend_does_not_spawn_idle_worker_and_blocks_reference_compute(self):
        worker=proxy.ReferenceWorker();worker.ensure_ready=Mock()
        with patch.object(proxy,'opener') as transport:
            worker.suspend(True)
            self.assertEqual(worker.forward('coarse/natural-v1/42/0/0.bin','GET',b'',b'',{})[0],409)
            transport.assert_not_called();worker.ensure_ready.assert_not_called()
        worker.suspend(False)
        self.assertFalse(worker.suspended)

    def test_suspend_drains_existing_worker_and_resume_reopens_admission(self):
        worker=proxy.ReferenceWorker();worker.state='ready';worker.process=FakeProcess()
        transport=Mock();transport.open.return_value=Response()
        with patch.object(proxy,'opener',return_value=transport):
            worker.suspend(True);worker.suspend(False)
        self.assertEqual([json.loads(call.args[0].data)['paused'] for call in transport.open.call_args_list],[True,False])
        self.assertTrue(all(call.args[0].full_url.endswith('/api/terrain/suspend') for call in transport.open.call_args_list))

    def test_real_loopback_bind_detects_free_and_occupied_port(self):
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as occupied:
            if os.name=='nt':occupied.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
            occupied.bind(('127.0.0.1',0))
            port=occupied.getsockname()[1]
            self.assertFalse(proxy._loopback_port_available(port))
            occupied.listen(1)
            self.assertFalse(proxy._loopback_port_available(port))
        self.assertTrue(proxy._loopback_port_available(port))

    def test_allowlist_rejects_remote_paths_traversal_and_mutation_methods(self):
        for path in ('api/world','api/status','height/natural-v1/42/-3/-14/10.bin',
                     'tiles/natural-v1/0/4/1/-2.png','api/overview/natural-v1/42.png'):
            self.assertTrue(proxy.allowed_path(path,'GET'))
        self.assertTrue(proxy.allowed_path('api/view','POST'))
        for path in ('http://elsewhere/api/world','//elsewhere','../api/world','api/%2e%2e/world',
                     'generated/secrets.txt','api/world?url=http://elsewhere','reference/api/world',''):
            self.assertFalse(proxy.allowed_path(path,'GET'))
        self.assertFalse(proxy.allowed_path('api/world','POST'))
        self.assertFalse(proxy.allowed_path('api/coarse/prepare','GET'))
        self.assertFalse(proxy.allowed_path('api/view','DELETE'))

    def test_registration_and_status_do_not_start_worker(self):
        worker=Mock();worker.status.return_value=dict(state='idle')
        app=Flask(__name__)
        with patch.dict(os.environ,{},clear=True):
            self.assertIs(proxy.register_backend_proxy(app,worker),worker)
            self.assertIs(proxy.register_backend_proxy(app),worker)
        client=app.test_client()
        self.assertEqual(client.get('/api/inference-backends').json['reference']['state'],'idle')
        worker.forward.assert_not_called()
        self.assertEqual(client.get('/reference/generated/private.txt').status_code,404)
        worker.forward.assert_not_called()

    def test_child_registers_identity_but_no_recursive_proxy(self):
        app=Flask(__name__)
        with patch.dict(os.environ,{proxy.WORKER_ENV:'test-token'}):
            proxy.register_backend_proxy(app)
        client=app.test_client()
        self.assertEqual(client.get(proxy.STATUS_PATH).json['worker_token'],'test-token')
        self.assertEqual(client.get('/reference/api/world').status_code,404)

    def test_body_query_status_and_terrain_headers_pass_through(self):
        worker=Mock();worker.forward.return_value=(409,b'{"conflict":true}',
            [('Content-Type','application/json'),('Cache-Control','no-store'),('X-Terrain-Identity','world42')])
        app=Flask(__name__)
        with patch.dict(os.environ,{},clear=True):proxy.register_backend_proxy(app,worker)
        response=app.test_client().post('/reference/api/view?seed=42&x=-1',data=b'{"seed":"42"}',content_type='application/json')
        self.assertEqual(response.status_code,409)
        self.assertEqual(response.headers['X-Terrain-Identity'],'world42')
        self.assertEqual(response.headers['X-Terrain-Backend'],'reference')
        args=worker.forward.call_args.args
        self.assertEqual(args[:4],('api/view','POST',b'seed=42&x=-1',b'{"seed":"42"}'))

    def test_transport_filters_cookies_and_forwards_http_errors(self):
        worker=proxy.ReferenceWorker()
        worker.ensure_ready=Mock()
        headers=Message();headers['Content-Type']='application/json';headers['Set-Cookie']='secret=1'
        headers['X-Terrain-Test']='yes';headers['Connection']='close'
        failure=HTTPError(worker.base_url+'/api/view',429,'busy',headers,io.BytesIO(b'busy'))
        transport=Mock();transport.open.side_effect=failure
        with patch.object(proxy,'opener',return_value=transport):
            status,body,forwarded=worker.forward('api/view','POST',b'seed=42',b'{}',
                {'Cookie':'secret=1','Authorization':'private','Content-Type':'application/json'})
        self.assertEqual((status,body),(429,b'busy'))
        self.assertNotIn('Set-Cookie',dict(forwarded))
        self.assertNotIn('Connection',dict(forwarded))
        request=transport.open.call_args.args[0]
        self.assertNotIn('Cookie',request.headers)
        self.assertNotIn('Authorization',request.headers)
        self.assertEqual(request.full_url,'http://127.0.0.1:8766/api/view?seed=42')

    def test_single_lazy_spawn_under_concurrent_requests_and_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker=proxy.ReferenceWorker(log_path=Path(tmp)/'worker.log')
            process=FakeProcess()
            ready=threading.Event();probing=threading.Event()
            def probe():probing.set();ready.wait(2);return True
            with patch.object(worker,'_port_available',return_value=True),patch.object(worker,'_probe',side_effect=probe),\
                 patch.object(proxy.subprocess,'Popen',return_value=process) as spawn,patch.object(proxy,'_windows_job',return_value=None):
                with ThreadPoolExecutor(3) as pool:
                    futures=[pool.submit(worker.ensure_ready) for _ in range(3)]
                    self.assertTrue(probing.wait(2));ready.set()
                    for future in futures:future.result(timeout=3)
                self.assertEqual(spawn.call_count,1)
                kwargs=spawn.call_args.kwargs
                self.assertEqual(kwargs['env']['TERRAIN_EXACT_KERNELS'],'0')
                self.assertEqual(kwargs['env']['TERRAIN_ATTENTION_BACKEND'],'reference')
                self.assertEqual(kwargs['env']['TERRAIN_PROFILE'],'0')
                self.assertNotIn('--prewarm-base',spawn.call_args.args[0])
                if os.name=='nt':self.assertEqual(kwargs['creationflags'],proxy.subprocess.CREATE_NO_WINDOW)
                worker.close()
                self.assertEqual(process.terminated,1)
                self.assertEqual(worker.status()['state'],'closed')

    def test_busy_port_is_not_adopted_or_killed(self):
        worker=proxy.ReferenceWorker()
        with patch.object(worker,'_port_available',return_value=False),patch.object(proxy.subprocess,'Popen') as spawn:
            with self.assertRaisesRegex(proxy.BackendUnavailable,'occupied'):
                worker.ensure_ready()
            spawn.assert_not_called()

    def test_startup_timeout_terminates_owned_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            worker=proxy.ReferenceWorker(startup_timeout=.02,log_path=Path(tmp)/'worker.log')
            process=FakeProcess()
            with patch.object(worker,'_port_available',return_value=True),patch.object(worker,'_probe',return_value=False),\
                 patch.object(proxy.subprocess,'Popen',return_value=process),patch.object(proxy,'_windows_job',return_value=None):
                with self.assertRaises(proxy.BackendUnavailable):worker.ensure_ready()
                self.assertEqual(process.terminated,1)
                self.assertEqual(worker.status()['state'],'failed')

    def test_forwarding_is_concurrent_after_ready(self):
        worker=proxy.ReferenceWorker();worker.ensure_ready=Mock()
        barrier=threading.Barrier(2)
        transport=Mock()
        def call(*args,**kwargs):barrier.wait(timeout=2);return Response()
        transport.open.side_effect=call
        with patch.object(proxy,'opener',return_value=transport),ThreadPoolExecutor(2) as pool:
            jobs=[pool.submit(worker.forward,'api/status','GET',b'',b'',{}) for _ in range(2)]
            self.assertEqual([job.result(timeout=3)[0] for job in jobs],[200,200])

    def test_probe_checks_token_config_and_accepts_venv_redirector_pid(self):
        worker=proxy.ReferenceWorker();worker.process=FakeProcess();worker.token='owned'
        transport=Mock()
        for token,pid,exact,expected in [('owned',12345,False,True),('owned',19684,False,True),
                                       ('foreign',12345,False,False),('owned',12345,True,False)]:
            import json
            transport.open.return_value=Response(json.dumps(dict(worker_role='reference',worker_token=token,pid=pid,
                launch_parent_pid=os.getpid(),exact_kernels=exact,attention_backend='reference')).encode())
            with patch.object(proxy,'opener',return_value=transport):self.assertEqual(worker._probe(),expected)

    @unittest.skipUnless(os.name=='nt','Windows job lifetime test')
    def test_real_venv_redirector_child_is_owned_and_terminated(self):
        # Exercise actual Win32 structures/permissions and the venv redirector.
        # This child only reports its PID and sleeps; it imports no terrain code.
        worker=proxy.ReferenceWorker();worker.token='cpu-lifecycle-test'
        child=subprocess.Popen([sys.executable,'-u','-c',
            'import os,time;print(os.getpid(),flush=True);time.sleep(30)'],
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,
            creationflags=subprocess.CREATE_NO_WINDOW)
        handle=None
        try:
            worker.process=child;worker.job=proxy._windows_job(child)
            actual_pid=int(child.stdout.readline().strip())
            kernel,job=worker.job
            # Keep a synchronization handle across worker.close to prove exit.
            handle=kernel.OpenProcess(0x00100000,False,actual_pid)
            self.assertTrue(handle)
            transport=Mock();transport.open.return_value=Response(json.dumps(dict(
                worker_role='reference',worker_token=worker.token,pid=actual_pid,
                launch_parent_pid=os.getpid(),exact_kernels=False,attention_backend='reference')).encode())
            with patch.object(proxy,'opener',return_value=transport):self.assertTrue(worker._probe())
            worker.state='ready'
            self.assertEqual(worker.status()['pid'],actual_pid)
            self.assertEqual(worker.status()['launcher_pid'],child.pid)
            worker.close()
            kernel.WaitForSingleObject.argtypes=[ctypes.c_void_p,ctypes.c_ulong]
            kernel.WaitForSingleObject.restype=ctypes.c_ulong
            self.assertEqual(kernel.WaitForSingleObject(handle,5000),0)
        finally:
            worker.close()
            if handle:kernel.CloseHandle(handle)
            child.stdout.close();child.stderr.close()


if __name__=='__main__':unittest.main()

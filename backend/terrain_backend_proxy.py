"""Lazy, isolated BF16 reference worker and a loopback-only terrain API proxy.

Import/registration/status do not launch Python, initialize CUDA or contact the
network. Production kernels/config remain immutable in the parent process.
"""
from __future__ import annotations

from terrain_paths import REPO_ROOT, WEB_ROOT, source_path

import atexit
import ctypes
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import time
from urllib import error, request as http
import uuid

ROOT=REPO_ROOT
WORKER_ENV='TERRAIN_REFERENCE_WORKER_TOKEN'
PARENT_ENV='TERRAIN_REFERENCE_PARENT_PID'
STATUS_PATH='/api/inference-backends'
MAX_BODY=4*1024**2
READ_APIS={'api/status','api/world','api/generation/schema','api/profile','api/inference/streams','generated/terrain.png'}
WRITE_APIS={'api/view','api/view/release','api/coarse/prepare','api/terrain/suspend','api/inference/streams'}
TILE_PATH=re.compile(r'(?:height/natural-v1/[0-9]+/-?[0-9]+/-?[0-9]+/-?[0-9]+\.bin|'
                     r'coarse/natural-v1/[0-9]+/-?[0-9]+/-?[0-9]+\.bin|'
                     r'tiles/natural-v1/[0-9]+/-?[0-9]+/-?[0-9]+/-?[0-9]+\.png|'
                     r'api/overview/natural-v1/[0-9]+\.png)\Z')
RESPONSE_HEADERS={'content-type','content-length','content-encoding','cache-control',
                  'etag','last-modified','expires','vary','content-range','accept-ranges','retry-after'}
REQUEST_HEADERS={'content-type','accept','if-none-match','if-modified-since','range','if-range'}


class BackendUnavailable(RuntimeError):
    pass


class NoRedirect(http.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def opener():
    # Ignore system HTTP_PROXY variables, and never follow a worker redirect.
    return http.build_opener(http.ProxyHandler({}),NoRedirect())


def allowed_path(path,method):
    if not isinstance(path,str) or any(c in path for c in ('\\','?','#','%',':','\x00')):
        return False
    if method in ('GET','HEAD'):
        return path in READ_APIS or TILE_PATH.fullmatch(path) is not None
    return method=='POST' and path in WRITE_APIS


def child_environment(token,parent_pid):
    env=os.environ.copy()
    env.update(TERRAIN_EXACT_KERNELS='0',TERRAIN_ATTENTION_BACKEND='reference',
               TERRAIN_PROFILE='0',TERRAIN_PROFILE_CUDA='0',TERRAIN_PREWARM_BASE='0',
               TERRAIN_REFERENCE_WORKER_TOKEN=token,TERRAIN_REFERENCE_PARENT_PID=str(parent_pid))
    return env


def _loopback_port_available(port):
    # A short connect timeout is not evidence of a listener on Windows. Ask
    # the OS whether this exact bind is available; never listen or reuse it.
    try:
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as probe:
            if os.name=='nt':
                probe.setsockopt(socket.SOL_SOCKET,socket.SO_EXCLUSIVEADDRUSE,1)
            probe.bind(('127.0.0.1',port))
        return True
    except OSError:
        return False


def _windows_job(process):
    """Tie the child to parent lifetime, including forced parent termination."""
    if os.name!='nt':
        return None
    from ctypes import wintypes as w
    class Basic(ctypes.Structure):
        _fields_=[('process_time',ctypes.c_longlong),('job_time',ctypes.c_longlong),
            ('flags',w.DWORD),('minimum_working_set',ctypes.c_size_t),('maximum_working_set',ctypes.c_size_t),
            ('active_process_limit',w.DWORD),('affinity',ctypes.c_size_t),
            ('priority',w.DWORD),('scheduling',w.DWORD)]
    class IO(ctypes.Structure):
        _fields_=[(name,ctypes.c_ulonglong) for name in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
    class Extended(ctypes.Structure):
        _fields_=[('basic',Basic),('io',IO),('process_memory',ctypes.c_size_t),
            ('job_memory',ctypes.c_size_t),('peak_process_memory',ctypes.c_size_t),('peak_job_memory',ctypes.c_size_t)]
    kernel=ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.CreateJobObjectW.argtypes=[ctypes.c_void_p,w.LPCWSTR];kernel.CreateJobObjectW.restype=w.HANDLE
    kernel.SetInformationJobObject.argtypes=[w.HANDLE,ctypes.c_int,ctypes.c_void_p,w.DWORD];kernel.SetInformationJobObject.restype=w.BOOL
    kernel.AssignProcessToJobObject.argtypes=[w.HANDLE,w.HANDLE];kernel.AssignProcessToJobObject.restype=w.BOOL
    kernel.OpenProcess.argtypes=[w.DWORD,w.BOOL,w.DWORD];kernel.OpenProcess.restype=w.HANDLE
    kernel.IsProcessInJob.argtypes=[w.HANDLE,w.HANDLE,ctypes.POINTER(w.BOOL)];kernel.IsProcessInJob.restype=w.BOOL
    kernel.CloseHandle.argtypes=[w.HANDLE];kernel.CloseHandle.restype=w.BOOL
    job=kernel.CreateJobObjectW(None,None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    limits=Extended();limits.basic.flags=0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel.SetInformationJobObject(job,9,ctypes.byref(limits),ctypes.sizeof(limits)) or not kernel.AssignProcessToJobObject(job,int(process._handle)):
        code=ctypes.get_last_error();kernel.CloseHandle(job)
        raise ctypes.WinError(code)
    return kernel,job


class ReferenceWorker:
    def __init__(self,*,startup_timeout=90.,request_timeout=180.,port=8766,log_path=None):
        if port!=8766 or not 0<startup_timeout<=90 or not 0<request_timeout<=180:
            raise ValueError('Reference endpoint fixed at127.0.0.1:8766 with bounded timeouts')
        self.base_url='http://127.0.0.1:8766'
        self.startup_timeout,self.request_timeout=startup_timeout,request_timeout
        from terrain_paths import RUNTIME_ROOT
        self.log_path=Path(log_path or RUNTIME_ROOT / 'reference-worker.log')
        self.condition=threading.Condition()
        self.process=None;self.job=None;self.log=None
        self.actual_pid=None
        self.suspended=False
        self.token=None;self.state='idle';self.last_error=None;self.failed_at=0.;self.closed=False

    def status(self):
        with self.condition:
            alive=self.process is not None and self.process.poll() is None
            return dict(state=self.state if self.state!='ready' or alive else 'exited',
                        pid=self.actual_pid if alive else None,launcher_pid=self.process.pid if alive else None,exact_kernels=False,
                        attention_backend='reference',worker_isolated=True,last_error=self.last_error,
                        log_path=str(self.log_path),startup_timeout_seconds=self.startup_timeout,
                        notes='Same current BF16 checkpoints/solver/graphs; only new exact kernels disabled. No TensorRT or quantization.')

    def _port_available(self):
        return _loopback_port_available(8766)

    def _spawn(self):
        if not self._port_available():
            raise BackendUnavailable('Reference port8766 already occupied; refusing to attach to or stop another process')
        self.token=uuid.uuid4().hex
        self.log_path.parent.mkdir(parents=True,exist_ok=True)
        self.log=self.log_path.open('ab',buffering=0)
        try:
            self.process=subprocess.Popen([sys.executable,str(ROOT/'tools/benchmarks/run_terrain_benchmark_server.py'),'--port','8766'],
                cwd=str(ROOT),env=child_environment(self.token,os.getpid()),stdin=subprocess.DEVNULL,
                stdout=self.log,stderr=subprocess.STDOUT,close_fds=True,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0) if os.name=='nt' else 0)
            self.job=_windows_job(self.process)
        except BaseException:
            self._terminate()
            raise

    def _probe(self):
        req=http.Request(self.base_url+STATUS_PATH,headers={'Accept':'application/json'})
        with opener().open(req,timeout=1.) as response:
            data=json.loads(response.read(65537))
        actual_pid=data.get('pid')
        valid=(data.get('worker_role')=='reference' and data.get('worker_token')==self.token
               and type(actual_pid) is int and actual_pid>0 and data.get('launch_parent_pid')==os.getpid()
               and data.get('exact_kernels') is False and data.get('attention_backend')=='reference')
        if not valid:
            return False
        with self.condition:
            if self.closed or self.process is None:
                return False
            # Windows venv python.exe is a redirector: its Popen PID can differ
            # from the interpreter PID. The private launch token proves this
            # worker's identity, not equality to the redirector's PID.
            if self.job is not None and actual_pid!=self.process.pid:
                from ctypes import wintypes as w
                kernel,job=self.job
                handle=kernel.OpenProcess(0x0100|0x0001|0x0400,False,actual_pid)
                if not handle:
                    raise ctypes.WinError(ctypes.get_last_error())
                try:
                    inside=w.BOOL()
                    if not kernel.IsProcessInJob(handle,job,ctypes.byref(inside)):
                        raise ctypes.WinError(ctypes.get_last_error())
                    # Cover the rare race where redirector spawned Python
                    # before the redirector was attached to the lifetime job.
                    if not inside.value and not kernel.AssignProcessToJobObject(job,handle):
                        raise ctypes.WinError(ctypes.get_last_error())
                finally:
                    kernel.CloseHandle(handle)
            self.actual_pid=actual_pid
        return True

    def ensure_ready(self):
        deadline=time.monotonic()+self.startup_timeout
        with self.condition:
            while self.state=='starting' and not self.closed:
                remaining=deadline-time.monotonic()
                if remaining<=0:
                    raise BackendUnavailable('Reference startup timed out while waiting')
                self.condition.wait(remaining)
            if self.closed:
                raise BackendUnavailable('Reference worker is closed')
            if self.state=='ready' and self.process is not None and self.process.poll() is None:
                return
            if self.state=='failed' and time.monotonic()-self.failed_at<5:
                raise BackendUnavailable(self.last_error or 'Reference startup failed')
            self._terminate()
            self.state='starting';self.last_error=None
        try:
            # Only the elected startup caller spawns/polls. HTTP forwarding does
            # not acquire this lock once readiness is established.
            with self.condition:
                if self.closed:
                    raise BackendUnavailable('Reference worker is closed')
                self._spawn()
            while time.monotonic()<deadline:
                with self.condition:
                    if self.closed:
                        raise BackendUnavailable('Reference worker was closed during startup')
                    if self.process is None or self.process.poll() is not None:
                        raise BackendUnavailable('Reference worker exited during startup; see its log')
                try:
                    ready=self._probe()
                except (OSError,ValueError):
                    ready=False
                if ready:
                    with self.condition:
                        if self.closed:
                            raise BackendUnavailable('Reference worker is closed')
                        self.state='ready';self.condition.notify_all()
                    return
                with self.condition:
                    self.condition.wait(min(.25,max(0,deadline-time.monotonic())))
            raise BackendUnavailable('Reference model startup exceeded90seconds; see its log')
        except BaseException as exc:
            with self.condition:
                self._terminate()
                self.state='closed' if self.closed else 'failed'
                self.last_error=f'{type(exc).__name__}: {exc}'
                self.failed_at=time.monotonic();self.condition.notify_all()
            if isinstance(exc,BackendUnavailable):
                raise
            raise BackendUnavailable(str(exc)) from exc

    def _terminate(self):
        process,self.process=self.process,None
        self.actual_pid=None
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5.)
                except subprocess.TimeoutExpired:
                    process.kill();process.wait(timeout=5.)
        except (OSError,subprocess.TimeoutExpired):
            # The child can exit between poll and terminate. The job's close
            # below remains the Windows lifetime guarantee.
            pass
        finally:
            if self.job is not None:
                kernel,handle=self.job;self.job=None
                kernel.CloseHandle(handle)
            if self.log is not None:
                self.log.close();self.log=None

    def close(self):
        with self.condition:
            self.closed=True
            self._terminate()
            self.state='closed';self.condition.notify_all()

    def forward(self,path,method,query,body,headers):
        if not allowed_path(path,method):
            raise ValueError('Reference route not allowed')
        with self.condition:
            if self.suspended and path not in ('api/status','api/view/release','api/terrain/suspend'):
                return 409,b'{"cancelled":true}',[('Content-Type','application/json')]
        self.ensure_ready()
        with self.condition:
            if self.suspended and path not in ('api/status','api/view/release','api/terrain/suspend'):
                return 409,b'{"cancelled":true}',[('Content-Type','application/json')]
        url=self.base_url+'/'+path+('?' + query.decode('ascii') if query else '')
        selected={key:value for key,value in headers.items() if key.lower() in REQUEST_HEADERS}
        selected['Accept-Encoding']='identity'
        req=http.Request(url,data=body if body else None,headers=selected,method=method)
        try:
            response=opener().open(req,timeout=self.request_timeout)
        except error.HTTPError as exc:
            response=exc
        with response:
            data=response.read()
            headers=[(key,value) for key,value in response.headers.items()
                     if key.lower() in RESPONSE_HEADERS or key.lower().startswith('x-terrain-')]
            return response.code,data,headers

    def suspend(self, paused):
        """Do not start an unused worker just to pause it."""
        with self.condition:
            self.suspended=paused
            ready=self.state=='ready' and self.process is not None and self.process.poll() is None
        if ready:
            req=http.Request(self.base_url+'/api/terrain/suspend',
                data=json.dumps({'paused':paused}).encode(),
                headers={'Content-Type':'application/json'},method='POST')
            with opener().open(req,timeout=30) as response:
                response.read()


def register_backend_proxy(app,worker=None):
    """Call once from terrain_server. Child registers identity, never a subproxy."""
    from flask import Response,jsonify,request
    if 'terrain_reference_worker' in app.extensions:
        return app.extensions['terrain_reference_worker']
    token=os.environ.get(WORKER_ENV)
    if token:
        app.extensions['terrain_reference_worker']=None
        @app.get(STATUS_PATH)
        def terrain_reference_identity():
            profile=getattr(sys.modules.get('terrain_server'),'runtime_profile',None)
            return jsonify(worker_role='reference',worker_token=token,pid=os.getpid(),
                           launch_parent_pid=int(os.environ.get(PARENT_ENV,'0')),
                           exact_kernels=getattr(profile,'exact_kernels',os.environ.get('TERRAIN_EXACT_KERNELS','1')=='1'),
                           attention_backend=getattr(profile,'attention_backend',os.environ.get('TERRAIN_ATTENTION_BACKEND','reference')))
        return None
    worker=worker or ReferenceWorker()
    app.extensions['terrain_reference_worker']=worker
    atexit.register(worker.close)

    @app.get(STATUS_PATH)
    def terrain_inference_backends():
        return jsonify(selected_by_client=True,optimized=dict(path_prefix='',exact_kernels=os.environ.get('TERRAIN_EXACT_KERNELS','1')=='1'),
            reference=dict(path_prefix='/reference',**worker.status()),
            comparison_notes='Keep seed, generation settings, camera and LOD policy equal. Workers have independent model/window memory and cache identities; cold versus warm must be reported separately.')

    @app.route('/reference/<path:path>',methods=['GET','HEAD','POST'])
    def terrain_reference_proxy(path):
        if not allowed_path(path,request.method):
            return jsonify(error='Reference endpoint not allowed'),404
        if request.content_length is not None and request.content_length>MAX_BODY:
            return jsonify(error='Reference request body exceeds4MiB'),413
        body=request.stream.read(MAX_BODY+1)
        if len(body)>MAX_BODY:
            return jsonify(error='Reference request body exceeds4MiB'),413
        try:
            status,data,headers=worker.forward(path,request.method,request.query_string,body,dict(request.headers))
        except BackendUnavailable as exc:
            return jsonify(error=str(exc),backend='reference',state=worker.status()['state']),503
        except (OSError,ValueError,error.URLError) as exc:
            return jsonify(error=f'Reference transport failed: {exc}',backend='reference'),502
        response=Response(data,status=status,headers=headers)
        response.headers['X-Terrain-Backend']='reference'
        response.headers['X-Terrain-Exact-Kernels']='0'
        return response
    return worker

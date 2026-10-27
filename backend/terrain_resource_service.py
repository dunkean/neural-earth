"""Persistent compiler process with shared-array RPC and incremental publication.

Only small JSON descriptors cross the pipe. Arrays are written once to shared
memory; receiver views retain their mapping after the sender releases its map.
The process is warmed before HTTP starts; world requests never spawn Python/JIT.
"""
import atexit
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import zipfile
from multiprocessing import shared_memory
import numpy as np

_worker=None


class _Owner:
    def __init__(self,block):self.block=block
    def __del__(self):
        try:self.block.close()
        except (OSError,BufferError):pass


class _Array(np.ndarray):
    """Direct buffer subclass: np.asarray retains this owner in its base chain."""
    pass


def _pack(arrays,buffer=None):
    description={};offset=0
    for name,value in arrays.items():
        a=np.asarray(value);offset=(offset+63)//64*64
        description[name]=dict(offset=offset,shape=a.shape,dtype=a.dtype.str)
        offset+=a.nbytes
    if buffer is not None and buffer.size < offset:
        buffer.close();buffer.unlink();buffer=None
    block=buffer if buffer is not None else shared_memory.SharedMemory(create=True,size=max(1,offset))
    try:
        for name,spec in description.items():
            np.copyto(np.ndarray(spec['shape'],dtype=spec['dtype'],buffer=block.buf,offset=spec['offset']),arrays[name])
        return block,dict(name=block.name,arrays=description)
    except BaseException:
        block.close();block.unlink();raise


def _receive(description,copy=True):
    block=shared_memory.SharedMemory(name=description['name'])
    if not copy:
        owner=_Owner(block);arrays={}
        for name,spec in description['arrays'].items():
            value=np.ndarray.__new__(_Array,spec['shape'],dtype=spec['dtype'],buffer=block.buf,offset=spec['offset'])
            value._owner=owner
            value.flags.writeable=False
            arrays[name]=value
        if os.name != 'nt':
            from multiprocessing import resource_tracker
            resource_tracker.unregister(block._name,'shared_memory')
        return arrays
    try:
        return {name:np.ndarray(spec['shape'],dtype=spec['dtype'],buffer=block.buf,offset=spec['offset']).copy()
                for name,spec in description['arrays'].items()}
    finally:
        block.close()
        if os.name != 'nt':
            # The independent creator owns unlink; do not let a receiver's
            # resource tracker try to remove an already acknowledged segment.
            from multiprocessing import resource_tracker
            resource_tracker.unregister(block._name,'shared_memory')


def _view(description):
    block=shared_memory.SharedMemory(name=description['name'])
    arrays={name:np.ndarray(spec['shape'],dtype=spec['dtype'],buffer=block.buf,offset=spec['offset'])
            for name,spec in description['arrays'].items()}
    return block,arrays


def _map_atlas(path):
    """Read the immutable, already validated Orogen NPZ without array copies."""
    from terrain_resource_layers import ATLAS_LAYERS
    from terrain_resource_cache import mapped_arrays
    with zipfile.ZipFile(path) as archive:
        names=[name.removesuffix('.npy') for name in archive.namelist()]
    names=[name for name in names if name == 'height_m' or name.startswith('layer_') and
           (name[6:] in ATLAS_LAYERS or name[6:].startswith('pedology_'))]
    mapped=mapped_arrays(path,names)
    if mapped is not None:return mapped
    with np.load(path,allow_pickle=False) as data:return {name:data[name] for name in names}


def _send(stream,value):
    stream.write(json.dumps(value,separators=(',',':')).encode()+b'\n');stream.flush()


def _read(stream):
    line=stream.readline()
    if not line: raise RuntimeError('Resource worker disconnected')
    return json.loads(line)


class Worker:
    def __init__(self):
        self.lock=threading.Lock()
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        self.process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--worker'],
            cwd=Path(__file__).resolve().parents[1],env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE)
        status=_read(self.process.stdout)
        if status.get('state') != 'ready':
            self.close()
            raise RuntimeError(status.get('error','Resource worker failed to initialize'))
        atexit.register(self.close)

    def compile(self,layers,height,seed,bounds,fantasy=False,progress=None,v2_fields=None,atlas_path=None):
        import terrain_resources as R
        from terrain_resource_layers import ATLAS_LAYERS
        from terrain_resource_deposits import DepositTable
        from terrain_resource_inputs import prepare,CLIMATE
        arrays={}
        if atlas_path is None:
            arrays['height']=height
            prepared=prepare(layers,height.shape,R.Lattice(R.WORLD_LOD,bounds=bounds))
            arrays.update({'layer__'+k:v for k,v in layers.items() if k in ATLAS_LAYERS and k not in CLIMATE and k != 'soil_sand'})
            arrays.update({'layer__'+k:v for k,v in prepared.items()})
        if v2_fields is not None: arrays.update({'v2__'+k:v for k,v in v2_fields.items()})
        with self.lock:
            started=time.perf_counter()
            block,description=_pack(arrays)
            packed=time.perf_counter()-started
            last_fields={}
            try:
                _send(self.process.stdin,dict(command='compile',input=description,input_file=atlas_path,
                    seed=int(seed),bounds=bounds,fantasy=fantasy))
                while True:
                    message=_read(self.process.stdout)
                    if message['state'] == 'error': raise RuntimeError(message['error'])
                    if message['state'] == 'progress':
                        if progress is not None: progress(message['stage'],last_fields)
                        continue
                    loaded=_receive(message['output'],copy=False)
                    if message['state'] == 'stage':
                        last_fields=loaded
                        if progress is not None: progress(message['stage'],loaded)
                    else:
                        _send(self.process.stdin,dict(command='ack'))
                        self.last_timings=dict(compute_s=message.get('compute_s'),stages=message.get('stages'),
                                               input_pack_s=packed,rpc_s=time.perf_counter()-started)
                        columns={k[5:]:loaded.pop(k) for k in list(loaded) if k.startswith('dep__')}
                        deposits=DepositTable(columns,message['families'],message['metals']) if columns else []
                        return R.Lattice(R.WORLD_LOD,bounds=bounds),loaded,deposits
            finally:
                block.close();block.unlink()

    def close(self):
        process=getattr(self,'process',None)
        if process is not None and process.poll() is None:
            try:
                process.stdin.close()
                process.wait(timeout=3)
            except (OSError,subprocess.TimeoutExpired):
                process.terminate()
                try: process.wait(timeout=3)
                except subprocess.TimeoutExpired: process.kill()


def start():
    global _worker
    if _worker is None: _worker=Worker()
    return _worker


def current():
    return _worker if _worker is not None and _worker.process.poll() is None else None


def _main():
    if os.name == 'nt':
        try:
            import psutil
            psutil.Process().nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        except (ImportError,OSError):
            pass
    from terrain_resource_acceleration import prepare
    import terrain_resources as R
    from terrain_resource_layers import _slim
    from terrain_resource_deposits import DepositTable
    incoming,outgoing=sys.stdin.buffer,sys.stdout.buffer
    spare=[None]
    def refill():
        if spare[0] is None:
            block=shared_memory.SharedMemory(create=True,size=128*1024*1024)
            np.ndarray((block.size,),np.uint8,buffer=block.buf).fill(0)
            spare[0]=block
    def cleanup():
        if spare[0] is not None:
            spare[0].close();spare[0].unlink();spare[0]=None
    atexit.register(cleanup)
    try:
        prepare()
        refill()
        _send(outgoing,dict(state='ready'))
    except Exception as exc:
        _send(outgoing,dict(state='error',error=f'{type(exc).__name__}: {exc}'));return
    stage_blocks=[]
    def publish(state,arrays,**extra):
        buffer=None
        if state == 'ready':buffer,spare[0]=spare[0],None
        block,description=_pack(arrays,buffer=buffer)
        if state == 'stage':
            stage_blocks.append(block)
            _send(outgoing,dict(state=state,output=description,**extra))
            return
        try:
            _send(outgoing,dict(state=state,output=description,**extra))
            if _read(incoming).get('command') != 'ack': raise RuntimeError('Resource IPC acknowledgement missing')
        finally:
            block.close();block.unlink()
            for item in stage_blocks:
                item.close();item.unlink()
            stage_blocks.clear()
    while True:
        line=incoming.readline()
        if not line: return
        try:
            command=json.loads(line)
            input_block,data=_view(command['input'])
            layers={k[7:]:v for k,v in data.items() if k.startswith('layer__')}
            if command.get('input_file'):
                atlas=_map_atlas(command['input_file'])
                data['height']=atlas.pop('height_m')
                layers.update({k[6:]:v for k,v in atlas.items()})
                atlas.clear()
            v2={k[4:]:v for k,v in data.items() if k.startswith('v2__')} or None
            started=time.perf_counter();stages=[]
            def progress(stage,fields):
                stages.append([stage,time.perf_counter()-started])
                # Publish NPP immediately; subsequent labels retain that small
                # snapshot until the complete authority is published once.
                if stage == 'climate':
                    publish('stage',_slim(fields),stage=stage)
                else:
                    _send(outgoing,dict(state='progress',stage=stage))
            grid,fields,deposits=R.compile_world(layers,data['height'],command['seed'],bounds=command['bounds'],
                fantasy=command['fantasy'],v2_fields=v2,progress=progress)
            arrays=_slim(fields)
            if isinstance(deposits,DepositTable):
                arrays.update({'dep__'+k:v for k,v in deposits.arrays.items()})
            publish('ready',arrays,compute_s=time.perf_counter()-started,stages=stages,
                    families=deposits.families if isinstance(deposits,DepositTable) else [],
                    metals=deposits.metals if isinstance(deposits,DepositTable) else [])
            arrays.clear();fields.clear()
            if v2 is not None:v2.clear()
        except Exception as exc:
            import traceback
            traceback.print_exc(file=sys.stderr)
            _send(outgoing,dict(state='error',error=f'{type(exc).__name__}: {exc}'))
        finally:
            for item in stage_blocks:
                item.close();item.unlink()
            stage_blocks.clear()
            if 'input_block' in locals():
                data.clear()
                layers.clear()
                input_block.close()
                if os.name != 'nt':
                    from multiprocessing import resource_tracker
                    resource_tracker.unregister(input_block._name,'shared_memory')
            refill()


if __name__ == '__main__':
    sys.path.insert(0,str(Path(__file__).resolve().parent))
    _main()

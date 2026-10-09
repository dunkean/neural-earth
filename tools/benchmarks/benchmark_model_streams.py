"""Offline graph-replay benchmark on real prepared Base/Decoder inputs.

Measures ready work only, excluding dependency preparation, fusion and I/O.
Original batch boundaries are harvested from WorldPipeline, never repartitioned.
Mixed runs join model coordinator streams only after every family is submitted.
"""

from pathlib import Path as _BootstrapPath
import sys as _bootstrap_sys
_REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
_bootstrap_sys.path.insert(0, str(_REPO_ROOT))
from tools._bootstrap import activate as _activate_repository
_activate_repository()

import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

ROOT=_REPO_ROOT
sys.path.insert(0,str(ROOT/'terrain-diffusion'))
import torch
from terrain_app import load_pipeline, MODEL_REVISION
from terrain_cuda_graphs import CudaGraphModel
from terrain_coarse_streams import CoarseStreamPool
from terrain_inference import set_coarse_streams, _coarse_batch
from benchmark_coarse_streams import scheduler, gpu_snapshot, compare


class ModelPool:
    def __init__(self,model,count,items):
        self.slots=[(torch.cuda.Stream(),CudaGraphModel(model,max_buckets=4,max_batch=16,
                                                      max_bytes=2*1024**3)) for _ in range(count)]
        # Capture every real shape serially on every slot, before any overlap.
        shapes={}
        for item in items:
            shapes.setdefault(self.slots[0][1]._key(*item[:4]),item)
        for stream,graph in self.slots:
            stream.wait_stream(torch.cuda.current_stream())
            for x,labels,conditions,embeds,expected in shapes.values():
                with torch.cuda.stream(stream):
                    actual=graph(x,noise_labels=labels,conditional_inputs=conditions,
                                 precomputed_embeds=embeds)
                stream.synchronize()
                if graph.fallback_calls or not compare(actual,expected)['byte_exact']:
                    raise RuntimeError(f'Graph admission/equality failed: {graph.stats()}')

    def run(self,items):
        caller=torch.cuda.current_stream()
        outputs=[]
        for offset in range(0,len(items),len(self.slots)):
            pending=[]
            for item,(stream,graph) in zip(items[offset:offset+len(self.slots)],self.slots):
                x,labels,conditions,embeds,_=item
                key=graph._key(x,labels,conditions,embeds)
                if key not in graph._buckets:raise RuntimeError('Unwarmed shape')
                stream.wait_stream(caller)
                with torch.cuda.stream(stream):
                    output=graph(x,noise_labels=labels,conditional_inputs=conditions,
                                 precomputed_embeds=embeds)
                    event=torch.cuda.Event();event.record(stream)
                    for tensor in (x,labels,*conditions,*((embeds,) if embeds is not None else ())):
                        tensor.record_stream(stream)
                pending.append((output,event))
            for output,event in pending:
                caller.wait_event(event);output.record_stream(caller);outputs.append(output)
        return outputs

    def close(self):
        for stream,graph in self.slots:
            stream.synchronize();graph._clear_buckets()


def measure(families,parallel=False):
    torch.cuda.synchronize()
    caller=torch.cuda.current_stream()
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    outputs={};timings={};coordinators=[]
    wall=time.perf_counter();start.record(caller)
    for name,pool,items in families:
        stream=torch.cuda.Stream() if parallel else caller
        stream.wait_event(start)
        begin,finish=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(stream):
            begin.record(stream)
            value=pool.run(items)
            outputs[name]=value[0] if name=='coarse' else value
            finish.record(stream)
        timings[name]=(begin,finish);coordinators.append(stream)
    for name,(begin,finish) in timings.items():
        caller.wait_event(finish)
        for output in outputs[name]:output.record_stream(caller)
    end.record(caller);end.synchronize()
    return outputs,dict(wall_seconds=time.perf_counter()-wall,
        gpu_seconds=start.elapsed_time(end)/1000,
        family_gpu_seconds={name:a.elapsed_time(b)/1000 for name,(a,b) in timings.items()})


@torch.inference_mode()
def main(path):
    report=dict(model_revision=MODEL_REVISION,torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
                scope='Prepared real inputs; excludes production dependency scheduling, I/O and fusion',
                gpu_before=gpu_snapshot(),single={},mixed={})
    report['source_sha256']={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
        ('tools/benchmarks/benchmark_model_streams.py','terrain_inference.py','terrain_coarse_streams.py','terrain_cuda_graphs.py')}
    def save():path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    save()
    world=load_pipeline(0)
    indices=[(0,0,i) for i in range(8)]
    set_coarse_streams(world,1)
    reference=[value for index in indices for value in world.coarse._f([index])]
    coarse_inputs=[]
    original=_coarse_batch
    # Get the exact production preparation via the explicit prepare-only path.
    pool_size=world.kwargs['coarse_pooling']
    t_cond=torch.atan(torch.tensor(world.kwargs['cond_snr'])).to(world.device,dtype=world._dtype)
    conditions=[v.detach().view(-1) for v in torch.log(torch.tan(t_cond)/8)]
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight=linear_weight_window(64//pool_size,'cpu',torch.float32).to(world.device)
    coarse_inputs=[original(world,[index],scheduler(),weight,t_cond,conditions,pool_size,prepare_only=True)
                   for index in indices]
    for count in (4,16):
        set_coarse_streams(world,count)
        actual=world.coarse._f(indices)
        report.setdefault('integrated_coarse',{})[str(count)]={
            'byte_exact':all(compare(a,b)['byte_exact'] for a,b in zip(actual,reference)),
            'effective':world._terrain_coarse_streams_effective,
            'error':world._terrain_coarse_streams_error}
        if not report['integrated_coarse'][str(count)]['byte_exact']:raise RuntimeError('Integrated mismatch')
        save()
    set_coarse_streams(world,4)
    harvested={'base':[],'decoder':[]}
    def recorder(name,forward):
        def call(x,noise_labels,conditional_inputs,**kw):
            output=forward(x,noise_labels=noise_labels,conditional_inputs=conditional_inputs,**kw)
            if len(harvested[name])<40:
                harvested[name].append((x.clone(),noise_labels.clone(),[t.clone() for t in conditional_inputs],
                    kw.get('precomputed_embeds'),output.clone()))
            return output
        return call
    with patch.object(world.base_model,'forward',recorder('base',world.base_model.forward)), \
         patch.object(world.decoder_model,'forward',recorder('decoder',world.decoder_model.forward)):
        world.residual[:,0:768,0:768]
    torch.cuda.synchronize()
    report['harvest_shapes']={name:[list(item[0].shape) for item in items] for name,items in harvested.items()}
    save();print('Harvest complete',report['harvest_shapes'],flush=True)
    selected={}
    for name in harvested:
        candidates=[item for item in harvested[name] if item[0].shape[0]==(16 if name=='base' else 1)]
        if not candidates:raise RuntimeError(f'No full actual batch for {name}')
        selected[name]=[candidates[i%len(candidates)] for i in range(8)]
    chosen={}
    for name in selected:
        model=getattr(world,name+'_model').model
        pools={}
        for count in (1,2,4):
            pool=None
            try:
                pool=ModelPool(model,count,selected[name]);trials=[]
                # Warm replays before timing, then verify all prepared values.
                outputs,_=measure([(name,pool,selected[name])])
                if not all(compare(a,b[-1])['byte_exact'] for a,b in zip(outputs[name],selected[name])):
                    raise RuntimeError('Replayed outputs mismatch')
                pools[count]=pool
                report['single'].setdefault(name,{})[str(count)]=dict(trials=trials,byte_exact=True,
                    graph_bytes=sum(sum(g._bucket_bytes.values()) for _,g in pool.slots))
            except RuntimeError as error:
                report['single'].setdefault(name,{})[str(count)]=dict(error=str(error))
                print(name,count,'refused',str(error)[:300],flush=True)
            finally:
                if pool and count not in pools:pool.close()
                save()
        for repeat in range(7):
            for count in (list(pools) if repeat%2==0 else list(reversed(pools))):
                _,stats=measure([(name,pools[count],selected[name])])
                report['single'][name][str(count)]['trials'].append(stats)
        for count,pool in pools.items():
            result=report['single'][name][str(count)]
            result['median_wall_seconds']=statistics.median(t['wall_seconds'] for t in result['trials'])
            result['median_gpu_seconds']=statistics.median(t['gpu_seconds'] for t in result['trials'])
            print(name,count,result['median_wall_seconds'],flush=True);pool.close()
        pools.clear();del pool;gc.collect();torch.cuda.empty_cache();save()
        results=report['single'][name]
        chosen[name]=int(min((k for k,v in results.items() if 'error' not in v),
                            key=lambda k:results[k]['median_wall_seconds']))
    # Fair sequential vs simultaneous comparison uses identical pool counts.
    coarse=CoarseStreamPool(world.coarse_model,scheduler(),world.device,streams=4)
    first=coarse_inputs[0]
    from terrain_coarse_graph import run_coarse_solver
    expected=run_coarse_solver(world,scheduler(),first.sample,first.conditions[0],list(first.labels),
                              list(first.conditions[1:]),list(first.embeds) if first.embeds is not None else None)
    coarse.warm(first,expected)
    base=ModelPool(world.base_model.model,chosen['base'],selected['base'])
    decoder=ModelPool(world.decoder_model.model,chosen['decoder'],selected['decoder'])
    # CoarsePool accepts one slot group; repeat two ready groups via adapter.
    class CoarseGroups:
        def run(self,items):
            return ([out for offset in range(0,len(items),4) for out in coarse.run(items[offset:offset+4])[0]],[])
    families=[('coarse',CoarseGroups(),coarse_inputs),('base',base,selected['base']),('decoder',decoder,selected['decoder'])]
    report['mixed']['streams']={'coarse':4,**chosen}
    _,baseline=measure(families)
    reference_mixed,_=measure(families)
    for parallel in (False,True):
        report['mixed']['parallel' if parallel else 'sequential']=dict(trials=[],byte_exact=True)
    for repeat in range(7):
        for parallel in ((False,True) if repeat%2==0 else (True,False)):
            outputs,stats=measure(families,parallel)
            if not all(compare(out,ref)['byte_exact'] for name in outputs
                       for out,ref in zip(outputs[name],reference_mixed[name])):raise RuntimeError('Mixed mismatch')
            report['mixed']['parallel' if parallel else 'sequential']['trials'].append(stats)
    for parallel in (False,True):
        result=report['mixed']['parallel' if parallel else 'sequential'];trials=result['trials']
        result.update(median_wall_seconds=statistics.median(t['wall_seconds'] for t in trials),
                      median_gpu_seconds=statistics.median(t['gpu_seconds'] for t in trials))
        save();print('Mixed',parallel,result,flush=True)
    coarse.close();base.close();decoder.close()
    report['gpu_after']=gpu_snapshot();save()


if __name__=='__main__':
    destination=Path(sys.argv[1]).resolve();destination.parent.mkdir(parents=True,exist_ok=True)
    main(destination)

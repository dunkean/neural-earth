"""Retrain the residual finite-tile flow at the largest FP32 width below100MB.

Without --run, inspect a CPU-only plan. Run the controller in distill.jobs for
durability. Full decoded VAL tiles select snapshots; TEST stays untouched.
"""
import argparse
from dataclasses import asdict
import io
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time

import torch

from distill.common import DATA, REPO, atomic_json, atomic_write, external_path
from distill.native_weight_quant import file_sha
from distill.student import StudentConfig
from distill.tile40_conditioning import CLIMATES, RELIEFS
from distill.tile40_flow_network import TimeFlow, ARCHITECTURE
from distill.tile40_quality_measure import measure


def inference_payload(config, contract, step, weights, selection):
    buffer = io.BytesIO()
    torch.save(dict(config=asdict(config), contract=contract, step=step,
        flow_architecture=ARCHITECTURE, inference_only=True, selected_weights=selection,
        model={name:value.detach().cpu() for name,value in weights.items()}), buffer)
    return buffer.getvalue()


def export_inference(path, config, contract, step, weights, selection, limit):
    payload = inference_payload(config,contract,step,weights,selection)
    if len(payload) >= limit:
        raise ValueError(f'Inference artifact is {len(payload)} bytes;must be strictly below {limit}.')
    atomic_write(path,lambda handle:handle.write(payload))
    return len(payload)


def largest_width(limit=100000000):
    """Maximize width in the existing multiple-of16 FP32 architecture.

    Leave room for the actual training contract and serialization headers, then
    measure a real inference payload. Export enforces the limit again on trained
    weights. No randomly initialized checkpoint is persisted.
    """
    selected = None
    for width in range(16,1025,16):
        config = StudentConfig(stage='base',width=width,depth=4)
        with torch.device('meta'):
            model = TimeFlow(config)
        raw_bytes = sum(t.numel()*t.element_size() for t in model.state_dict().values())
        if raw_bytes+262144 >= limit:
            break
        selected = dict(width=width,parameters=sum(p.numel() for p in model.parameters()),
                        tensor_bytes_fp32=raw_bytes)
    if selected is None:
        raise ValueError('Byte budget is too small for this architecture.')
    with torch.random.fork_rng(devices=[]):
        model = TimeFlow(StudentConfig(stage='base',width=selected['width'],depth=4))
    payload = inference_payload(model.config,{},0,model.state_dict(),'raw')
    selected.update(example_inference_bytes=len(payload),contract_reserve_bytes=262144,
                    next_width=width,next_width_tensor_bytes=raw_bytes)
    return selected


def terrain_summary(report):
    """Retain mountain/climate failures even when the aggregate improves."""
    if len(report['rows']) != 30 or any(r['case']['split']!='val' for r in report['rows']):
        raise ValueError('Selection requires all30 VAL cases and excludes TEST.')
    seen = set()
    for row in report['rows']:
        case = row['case']
        key = (case['seed'],case['relief'],case['climate'])
        if key in seen:
            raise ValueError('Duplicate VAL case.')
        seen.add(key)
    if any(sum(r['case']['relief']==relief and r['case']['climate']==climate for r in report['rows'])!=2
           for relief in RELIEFS for climate in CLIMATES):
        raise ValueError('Two VAL geometries per relief/climate are required.')
    score,groups = measure(report)
    climates = {}
    for climate in CLIMATES:
        rows = [r for r in report['rows'] if r['case']['relief']=='mountains' and r['case']['climate']==climate]
        if len(rows)!=2:
            raise ValueError('Two mountain geometries per climate are required.')
        spectral, directional, slope = [],[],[]
        for row in rows:
            m = row['metrics']
            spectral.append(statistics.mean(abs(math.log(max(b['amplitude_ratio'],1e-6))) for b in m['bands']))
            c = m['directional_coherence']
            directional.append(abs(math.log(max(c['prediction'],1e-6)/max(c['reference'],1e-6))))
            slope.append(m['prediction_stats']['slope_p90_degrees']-m['reference_stats']['slope_p90_degrees'])
        responses = []
        for row in rows:
            temperate = next(r for r in report['rows'] if r['case']['seed']==row['case']['seed'] and
                r['case']['relief']=='mountains' and r['case']['climate']=='temperate')
            errors = []
            for band,base in zip(row['metrics']['bands'],temperate['metrics']['bands'],strict=True):
                predicted = math.log(max(band['prediction_rms_m'],1e-6)/max(base['prediction_rms_m'],1e-6))
                reference = math.log(max(band['reference_rms_m'],1e-6)/max(base['reference_rms_m'],1e-6))
                errors.append(abs(predicted-reference))
            responses.append(statistics.mean(errors))
        climates[climate] = dict(morphology_distance=statistics.mean(a+b for a,b in zip(spectral,directional)),
            spectral_distance=statistics.mean(spectral),directional_distance=statistics.mean(directional),
            slope_p90_delta_degrees=statistics.mean(slope),
            climate_spectral_response_error=statistics.mean(responses))
    return dict(score=score,groups=groups,mountain_climates=climates,
        scope='Spectral,directional and slope diagnostics;images and physical climate-response review remain required.')


def compatible_reports(candidate, baseline):
    for key in ('coarse_sha256','decoder_sha256','dataset_manifest_sha256','decoder_layout','sampling_config'):
        if candidate.get(key)!=baseline.get(key):
            raise ValueError(f'Unmatched baseline evaluation:{key}')
    case_set = lambda report:{(r['case']['seed'],r['case']['relief'],r['case']['climate']) for r in report['rows']}
    if case_set(candidate)!=case_set(baseline):
        raise ValueError('Candidate and baseline must cover the same VAL scenes.')


def admissible(summary, baseline):
    # This selects experimental snapshots;it does not grant visual acceptance.
    return (summary['groups']['mountains'] < baseline['groups']['mountains'] and
        all(summary['groups'][relief] <= baseline['groups'][relief]*1.1 for relief in RELIEFS) and
        all(summary['mountain_climates'][climate]['morphology_distance'] <=
            baseline['mountain_climates'][climate]['morphology_distance']*1.1 for climate in CLIMATES) and
        all(summary['mountain_climates'][climate]['climate_spectral_response_error'] <=
            baseline['mountain_climates'][climate]['climate_spectral_response_error']*1.1+.005 for climate in CLIMATES))


def resume_checkpoint(train):
    """Recover the furthest complete optimizer/RNG state,including hard stops."""
    candidates = list(train.glob('step-*.pt'))
    if (train/'latest.pt').exists():
        candidates.append(train/'latest.pt')
    if not candidates:
        return None
    # Storage goes to meta;only the counters are used,no GPU allocation.
    return max(candidates,key=lambda path:torch.load(path,map_location='meta',weights_only=True)['step'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DATA/'tile40-20261014/reference-coarse2-guided-expanded-v5')
    parser.add_argument('--output',type=Path,default=DATA/'tile40-20261020/residual-climate112-v1')
    parser.add_argument('--baseline',type=Path,default=DATA/'tile40-20261014/residual-altitude48-8k-raw-4interval-val30-v1/report.json')
    parser.add_argument('--decoder',type=Path,default=DATA/'tile40-20261014/decoder16-original-pilot-v1/step-010000.pt')
    parser.add_argument('--steps',type=int,default=20000)
    parser.add_argument('--interval',type=int,default=2000)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--evaluation-gpu',choices=('0','1'),help='Evaluate concurrently on this physical GPU while the training GPU runs continuously.')
    parser.add_argument('--watch',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--plan-output',type=Path)
    args = parser.parse_args()
    if min(args.steps,args.interval)<1 or args.steps%args.interval:
        raise ValueError('Positive steps divisible by the evaluation interval required.')
    if args.batch_size<2 or args.batch_size%2 or (args.watch and args.evaluation_gpu is None):
        raise ValueError('Even batch size and an explicit evaluation GPU for the watcher required.')
    torch.set_num_threads(8)
    output = external_path(args.output)
    size = largest_width()
    manifest = json.loads((args.dataset/'manifest.json').read_text())
    baseline_report = json.loads(args.baseline.read_text())
    baseline = terrain_summary(baseline_report)
    if (file_sha(args.dataset/'manifest.json') != baseline_report['dataset_manifest_sha256'] or
        file_sha(args.decoder) != baseline_report['decoder_sha256'] or
        file_sha(REPO/'distill/weights/coarse2.pt') != baseline_report['coarse_sha256']):
        raise ValueError('Keep the exact reference bank,coarse and decoder of the best residual baseline.')
    plan = dict(schema=1,architecture=ARCHITECTURE,model=size,limit_bytes=100000000,
        initialization='Fresh weights;same residual-altitude flow-matching recipe as residual48/raw8k.',
        sampling=dict(mountain_fraction=.7,climate_pairs=True,batch_size=args.batch_size,
            climates=list(CLIMATES),pairing='Same TRAIN geometry,crop,Gaussian start and time;two distinct climates.'),
        steps=args.steps,interval=args.interval,lr=.0002,evaluation_gpu=args.evaluation_gpu,dataset=str(args.dataset.resolve()),
        dataset_manifest_sha256=file_sha(args.dataset/'manifest.json'),
        cases={split:sum(c['split']==split for c in manifest['cases']) for split in ('train','val','test')},
        baseline_report=str(args.baseline.resolve()),baseline_report_sha256=file_sha(args.baseline),
        baseline=baseline,decoder=str(args.decoder.resolve()),decoder_sha256=file_sha(args.decoder),
        output=str(output),cuda_available=torch.cuda.is_available(),
        scope='Finite40km tiles;FP32 inference base under100000000 bytes;optimizer/EMA training state excluded.',
        source_sha256={name:file_sha(REPO/'distill'/name) for name in
            ('tile40_climate_run.py','tile40_flow_train.py','tile40_flow_network.py')})
    if args.plan_output:
        atomic_json(args.plan_output,dict(status='prepared' if plan['cuda_available'] else 'blocked_no_cuda',
            training_started=False,plan=plan))
    if not args.run:
        print(json.dumps(dict(training_started=False,plan=plan),indent=2))
        return
    if not plan['cuda_available']:
        raise RuntimeError('CUDA unavailable;training has not started. Run this command in a GPU-enabled session.')
    # Check writability before starting subprocesses;never redirect a long run
    # into temporary storage if the configured persistent data root is blocked.
    request = output/'request.json'
    if request.exists():
        if json.loads(request.read_text()) != plan:
            raise ValueError('Existing run contract differs;use a fresh experiment directory.')
    else:
        if output.exists():
            raise ValueError('Existing experiment has no matching request;inspect it first.')
        atomic_json(request,plan)
    train = output/'train'
    history = []
    best = None
    stopped, children = False, []
    status_path = output/('status-watch.json' if args.watch else 'status.json')
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
        for child in children:
            if child.poll() is None:
                child.terminate()
    for signum in (signal.SIGTERM,signal.SIGINT):
        signal.signal(signum,stop)
    def execute(command):
        if stopped:
            raise InterruptedError('Controller stopped.')
        child = subprocess.Popen([sys.executable,'-u','-m',*command],cwd=REPO)
        children.append(child)
        code = child.wait()
        children.remove(child)
        if stopped:
            raise InterruptedError('Controller stopped;inspect the child checkpoint before resuming.')
        if code:
            raise RuntimeError(f'{command[0]} exited with code{code};inspect before resuming.')
    def training_command(until):
        command = ['distill.tile40_flow_train','--dataset',str(args.dataset),'--output',str(train),
            '--width',str(size['width']),'--steps',str(until),'--interval',str(args.interval),
            '--residual-altitude','--climate-pairs','--mountain-fraction','.7','--batch-size',str(args.batch_size),
            '--max-inference-bytes','100000000']
        parent=resume_checkpoint(train)
        if parent is not None:
            command += ['--resume',str(parent)]
        return command
    def evaluate(mark):
        nonlocal best
        for selection in ('raw','ema'):
            inference = train/'inference'/f'step-{mark:06d}-{selection}.pt'
            while not inference.exists():
                if stopped:
                    raise InterruptedError('Watcher stopped while waiting for a snapshot.')
                if not args.watch:
                    raise RuntimeError(f'Missing audited inference artifact:{inference}')
                atomic_json(status_path,dict(status='waiting_snapshot',step=mark,weights=selection))
                time.sleep(5)
            evaluation = output/f'val-{mark:06d}-{selection}'
            report_path = evaluation/'report.json'
            if not report_path.exists():
                atomic_json(status_path,dict(status='evaluating',step=mark,weights=selection,
                    gpu=torch.cuda.get_device_name()))
                execute(['distill.tile40_evaluate','--dataset',str(args.dataset),'--encoder',str(inference),
                    '--encoder-weights',selection,'--decoder',str(args.decoder),'--output',str(evaluation),
                    '--limit','30','--device','cuda','--flow','--finite-flow','--flow-steps','4'])
            report = json.loads(report_path.read_text())
            compatible_reports(report,baseline_report)
            summary = terrain_summary(report)
            entry = dict(step=mark,weights=selection,summary=summary,report=str(report_path),
                inference=str(inference),inference_bytes=inference.stat().st_size,
                admissible_vs_baseline=admissible(summary,baseline))
            history.append(entry)
            if entry['admissible_vs_baseline'] and (best is None or
                    summary['groups']['mountains'] < best['summary']['groups']['mountains']):
                best = entry
            atomic_json(output/'selection.json',dict(history=history,best=best,quality_accepted=False,
                fallback='Keep residual48/raw8k when no larger snapshot improves mountains without regressions.'))
    try:
        if args.evaluation_gpu is not None and not args.watch:
            command = training_command(args.steps)
            watcher = ['distill.tile40_climate_run','--run','--watch','--evaluation-gpu',args.evaluation_gpu,
                '--dataset',str(args.dataset),'--output',str(output),'--baseline',str(args.baseline),
                '--decoder',str(args.decoder),'--steps',str(args.steps),'--interval',str(args.interval),
                '--batch-size',str(args.batch_size)]
            train_child = subprocess.Popen([sys.executable,'-u','-m',*command],cwd=REPO)
            children.append(train_child)
            evaluation_env = dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=args.evaluation_gpu)
            watch_child = subprocess.Popen([sys.executable,'-u','-m',*watcher],cwd=REPO,env=evaluation_env)
            children.append(watch_child)
            atomic_json(status_path,dict(status='running',target_steps=args.steps,
                training_gpu=torch.cuda.get_device_name(),evaluation_gpu=args.evaluation_gpu,
                training_pid=train_child.pid,watcher_pid=watch_child.pid))
            while any(child.poll() is None for child in children):
                if stopped:
                    raise InterruptedError('Dual-GPU controller stopped.')
                for child in children:
                    if child.poll() not in (None,0):
                        raise RuntimeError(f'GPU worker{child.pid} exited{child.returncode};inspect before resuming.')
                if train_child.poll()==0:
                    progress=json.loads((train/'status.json').read_text())
                    if progress['step']!=args.steps or progress['status']!='segment_complete':
                        raise RuntimeError('Trainer stopped early;inspect disk reserve and trainer status.')
                time.sleep(2)
            if any(child.returncode for child in children):
                raise RuntimeError('GPU worker failed at completion.')
            children.clear()
            best=json.loads((output/'selection.json').read_text())['best']
        else:
            for mark in range(args.interval,args.steps+1,args.interval):
                checkpoint = train/f'step-{mark:06d}.pt'
                if not args.watch and not checkpoint.exists():
                    atomic_json(status_path,dict(status='training',target=mark))
                    execute(training_command(mark))
                    if not checkpoint.exists():
                        raise RuntimeError('Training stopped before its milestone;inspect trainer status.')
                evaluate(mark)
        atomic_json(status_path,dict(status='complete',step=args.steps,best=best,quality_accepted=False))
    except InterruptedError as exc:
        atomic_json(status_path,dict(status='interrupted',error=str(exc)))
        raise SystemExit(130)
    except Exception as exc:
        atomic_json(status_path,dict(status='failed',error=str(exc)))
        raise
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=120)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__=='__main__':
    main()

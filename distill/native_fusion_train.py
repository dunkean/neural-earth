"""Train native regions through both stages and fusion, after the operator pilot.

All artifacts remain outside Git. A short fusion run is a candidate, never
terrain acceptance. Decoder morphology, coast stability and cold seams follow.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import signal
import time

import numpy as np
import torch

from distill.common import DATA, HOLDOUT_SEEDS, REVISION, append_json, atomic_json, atomic_write, external_path
from distill.native_base import NativeBase, teacher_source
from distill.native_fusion import Region, region_forward, region_loss

STOP=False


def stop(signum,frame):
    global STOP
    STOP=True


@torch.no_grad()
def validate(model,files,weight,count):
    rows=[]
    for index in np.linspace(0,len(files)-1,min(count,len(files))).round().astype(int):
        for offset in ((32,32),(32,64),(64,32),(64,64)):
            region=Region(files[index],'cuda',offset)
            result=region_forward(model,region,weight)
            mse=float((result['core']-region.target).square().mean())
            gradient=float(torch.stack([(result['core'].diff(dim=axis)-region.target.diff(dim=axis)).square().mean()
                                        for axis in (-1,-2)]).mean())
            curvature=float(torch.stack([(result['core'][4:].diff(n=2,dim=axis)-region.target[4:].diff(n=2,dim=axis)).square().mean()
                                         for axis in (-1,-2)]).mean())
            rows.append(dict(file=files[index].name,seed=region.metadata['seed'],profile=region.metadata['profile'],
                             core_offset=offset,mse=mse,gradient=gradient,height_curvature=curvature,
                             selection_score=mse+.1*gradient+.05*curvature))
    profiles={profile:float(np.mean([row['selection_score'] for row in rows if row['profile']==profile]))
              for profile in {row['profile'] for row in rows}}
    return dict(mse=float(np.mean([row['mse'] for row in rows])),
                gradient=float(np.mean([row['gradient'] for row in rows])),rows=rows,
                selection_score=max(profiles.values()),profile_scores=profiles,
                note='Blended latent region; decoded terrain and cold seams not yet accepted.')


@torch.no_grad()
def check_teacher(teacher,path):
    """Replay original calls; cross-GPU rounding must fit the measured FP32 floor."""
    rows=[]
    prior=torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark=False
    try:
        with np.load(path,allow_pickle=False) as sample:
            for label in np.unique(sample['labels']):
                call_id=sample['call_ids'][np.flatnonzero(sample['labels']==label)[0]]
                indices=np.flatnonzero(sample['call_ids']==call_id)
                inputs,labels,condition,target=[torch.from_numpy(sample[key][indices]).cuda()
                    for key in ('inputs','labels','condition','target')]
                output=teacher(inputs.to(torch.bfloat16),noise_labels=labels.to(torch.bfloat16),
                               conditional_inputs=[condition.to(torch.bfloat16)]).float()
                error=output-target
                rows.append(dict(label=float(label),max_abs=float(error.abs().max()),
                                 mse=float(error.square().mean()),batch=len(indices),call_id=int(call_id)))
            if any(row['max_abs'] != 0. for row in rows):
                from terrain_diffusion.models.edm_unet import EDMUnet2D
                from terrain_inference import prepare_models
                from types import SimpleNamespace
                fp32=EDMUnet2D.from_pretrained(str(teacher_source())).to('cuda',torch.float32).eval().requires_grad_(False)
                prepare_models(SimpleNamespace(coarse_model=fp32,base_model=fp32,decoder_model=fp32))
                for row in rows:
                    indices=np.flatnonzero(sample['call_ids']==row['call_id'])
                    inputs,labels,condition,target=[torch.from_numpy(sample[key][indices]).cuda()
                        for key in ('inputs','labels','condition','target')]
                    output=fp32(inputs.float(),noise_labels=labels.float(),conditional_inputs=[condition.float()])
                    error=output-target
                    row['fp32_vs_stored_mse']=float(error.square().mean())
                    row['fp32_vs_stored_max_abs']=float(error.abs().max())
                    row['passed']=row['mse'] <= 1.21*row['fp32_vs_stored_mse']+1e-12
                del fp32
                torch.cuda.empty_cache()
            else:
                for row in rows:
                    row['passed']=True
    finally:
        torch.backends.cudnn.benchmark=prior
    return dict(max_abs=max(row['max_abs'] for row in rows),original_call_batches=True,
                gpu=torch.cuda.get_device_name(),rows=rows,passed=all(row['passed'] for row in rows),
                note='Same native inputs/batches and pinned weights. Cross-GPU BF16 RMSE must be within110% of the same-input FP32/stored-BF16 floor; this is not decoded physical acceptance.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,default=DATA/'crops/native-scenes-v1')
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=2000)
    parser.add_argument('--lr',type=float,default=1e-5)
    parser.add_argument('--eval-every',type=int,default=250)
    parser.add_argument('--val-count',type=int,default=48)
    parser.add_argument('--seed',type=int,default=20261011)
    parser.add_argument('--fresh-noise-fraction',type=float,default=0.,
                        help='Fraction of training regions retargeted by the full teacher on new base noises; validation remains fixed.')
    args=parser.parse_args()
    if (min(args.steps,args.eval_every,args.val_count) < 1 or not np.isfinite(args.lr) or args.lr <= 0 or
            not np.isfinite(args.fresh_noise_fraction) or not 0 <= args.fresh_noise_fraction <= 1):
        parser.error('Positive finite native fusion settings required.')
    root=external_path(args.dataset)
    output=external_path(args.output)
    if json.loads((root/'status.json').read_text())['status'] != 'complete':
        raise ValueError('Finish native capture before fusion training.')
    train_files,val_files=sorted(root.glob('train-*.npz')),sorted(root.glob('val-*.npz'))
    if not train_files or not val_files:
        raise ValueError('Independent train and validation scenes are required.')
    manifest=json.loads((root/'manifest.json').read_text())
    train_seeds={row['metadata']['seed'] for row in manifest['scenes'] if row['metadata']['split']=='train'}
    val_seeds={row['metadata']['seed'] for row in manifest['scenes'] if row['metadata']['split']=='val'}
    if train_seeds & val_seeds or (train_seeds|val_seeds) & HOLDOUT_SEEDS:
        raise ValueError('Native fusion worlds overlap or contain reserved seeds.')
    source_payload=args.source.read_bytes()
    source=torch.load(args.source,map_location='cpu',weights_only=True)
    contract=dict(dataset_manifest_sha256=hashlib.sha256((root/'manifest.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(source_payload).hexdigest(),source_step=source['step'],
        config=source['config'],seed=args.seed,lr=args.lr,val_count=args.val_count,
        core=32,init_windows=16,refinement_windows=4,on_policy_teacher=True,
        fresh_noise_fraction=args.fresh_noise_fraction,
        validation_offsets=[[32,32],[32,64],[64,32],[64,64]],selection='worst-profile latent+gradient+curvature',
        producer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        region_producer_sha256=hashlib.sha256(Path(__file__).with_name('native_fusion.py').read_bytes()).hexdigest())
    contract_path=output/'contract.json'
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError('Native fusion resume contract changed; inspect before resuming.')
    atomic_json(contract_path,contract)
    saved=torch.load(output/'latest.pt',map_location='cpu',weights_only=True) if (output/'latest.pt').exists() else None
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=True
    from terrain_diffusion.models.edm_unet import EDMUnet2D
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    from terrain_inference import prepare_models
    from types import SimpleNamespace
    teacher=EDMUnet2D.from_pretrained(str(teacher_source())).to('cuda',torch.bfloat16).eval().requires_grad_(False)
    prepare_models(SimpleNamespace(coarse_model=teacher,base_model=teacher,decoder_model=teacher))
    control=check_teacher(teacher,train_files[0])
    atomic_json(output/'teacher-control.json',control)
    if not control['passed']:
        raise RuntimeError('On-policy teacher did not reconstruct original native calls; inspect before fusion training.')
    model=NativeBase(source['config']).to('cuda').eval()
    model.load_state_dict(saved['model'] if saved else source.get('ema',source['model']))
    ema=NativeBase(source['config']).to('cuda').eval().requires_grad_(False)
    ema.load_state_dict(saved['ema'] if saved else model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,betas=(.9,.99),weight_decay=1e-4,fused=True)
    step=saved['step'] if saved else 0
    best=saved['best_score'] if saved else float('inf')
    if step > args.steps:
        raise ValueError('Cannot resume fusion to an earlier step.')
    if saved:
        optimizer.load_state_dict(saved['optimizer'])
        torch.set_rng_state(saved['rng'])
        torch.cuda.set_rng_state_all(saved['cuda_rng'])
    del source,source_payload,saved
    weight=linear_weight_window(64,'cuda',torch.float32)
    for signum in (signal.SIGINT,signal.SIGTERM):
        signal.signal(signum,stop)
    started=time.monotonic()
    last_save=started

    def checkpoint(name):
        atomic_write(output/name,lambda handle:torch.save(dict(config=model.config,
            model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),step=step,
            best_score=best,contract=contract,teacher_revision=REVISION,
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),handle))

    atomic_json(output/'run.json',dict(contract=contract,target_steps=args.steps,start_step=step,
        train_scenes=len(train_files),validation_scenes=len(val_files),accepted=False))
    if step==0:
        details=validate(ema,val_files,weight,args.val_count)
        best=details['selection_score']
        checkpoint('initial.pt')
        checkpoint('best.pt')
        append_json(output/'validation.jsonl',dict(step=0,**details))
        print(json.dumps(dict(step=0,blended_latent_mse=details['mse'],selection_score=best)),flush=True)
    try:
        while step < args.steps and not STOP:
            # Stateless scene/core sampling makes resume independent of wall time and loader prefetch.
            generator=np.random.default_rng(np.random.SeedSequence([args.seed,step]))
            index=int(generator.integers(len(train_files)))
            offset=tuple((32,64)[int(generator.integers(2))] for axis in range(2))
            region=Region(train_files[index],'cuda',offset)
            if generator.random() < args.fresh_noise_fraction:
                noise_seed=int(generator.integers(1_000_000,2**31-1))
                region.refresh_noise(teacher,weight,noise_seed)
            optimizer.zero_grad(set_to_none=True)
            result=region_forward(model,region,weight,checkpointed=True)
            loss,terms=region_loss(result,region,teacher)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite native fusion loss.')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            optimizer.step()
            step+=1
            with torch.no_grad():
                decay=min(.999,(1+step)/(10+step))
                for average,parameter in zip(ema.parameters(),model.parameters()):
                    average.lerp_(parameter,1-decay)
            if step%25==0 or step==1:
                row=dict(step=step,loss=float(loss),terms={k:float(v) for k,v in terms.items()},
                         grad_norm=float(norm),elapsed_seconds=time.monotonic()-started,
                         cuda_peak_bytes=torch.cuda.max_memory_allocated(),source=train_files[index].name,
                         core_offset=offset)
                row['noise_seed']=getattr(region,'noise_seed',region.metadata['seed'])
                append_json(output/'train.jsonl',row)
                print(json.dumps(row),flush=True)
            del result,loss,terms,region
            if step%args.eval_every==0 or step==args.steps:
                details=validate(ema,val_files,weight,args.val_count)
                append_json(output/'validation.jsonl',dict(step=step,**details))
                if details['selection_score'] < best:
                    best=details['selection_score']
                    checkpoint('best.pt')
                checkpoint(f'step-{step:06d}.pt')
            if time.monotonic()-last_save >= 120:
                checkpoint('latest.pt')
                last_save=time.monotonic()
                atomic_json(output/'status.json',dict(status='running',step=step,target_steps=args.steps,best_score=best))
        checkpoint('latest.pt')
        atomic_json(output/'status.json',dict(status='interrupted' if STOP else 'complete',step=step,
            best_score=best,elapsed_seconds=time.monotonic()-started,accepted=False))
    except Exception as exc:
        checkpoint('latest.pt')
        atomic_json(output/'status.json',dict(status='failed',step=step,error=f'{type(exc).__name__}: {exc}'))
        raise


if __name__ == '__main__':
    main()

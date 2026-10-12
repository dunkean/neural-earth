"""Bounded 4k teacher-basis pilot; 5090 training, asynchronous 4090 validation.

Keep the teacher encoder, skips, attention, heads and conditioning intact.
Anneal only selected decoder 3x3 filters to quarter rank over 1000 steps.
No decoded loss, no change to the accepted application stack.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch

from distill.common import DATA, REPO, REVISION, atomic_json, atomic_write, append_json, external_path
from distill.native_base import teacher_source
from distill.native_basis_training import (WindowBank, anneal, compact, inventory,
    kd_step, selective_model)
from distill.wet_mountains import ROOT, digest
from terrain_diffusion.models.edm_unet import EDMUnet2D

STOP=False


def main():
    global STOP
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DATA/'ckpt/native-basis-decoder-quarter-pilot')
    parser.add_argument('--profile',type=Path,default=DATA/'eval/native-lowrank-quarter-cached-profile.json')
    parser.add_argument('--steps',type=int,default=4000)
    parser.add_argument('--seed',type=int,default=202610125)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-root',type=Path)
    args=parser.parse_args()
    output=external_path(args.output)
    profile=json.loads(args.profile.read_text())
    proof=next(r for r in profile['results'] if r['rank_fraction']==.25)
    if profile['status']!='complete' or proof['speedup_vs_teacher']<1.3 or proof['wrapper_equivalence_mse']!=0:
        raise ValueError('Quarter-rank numerical equivalence and cached speed potential >=1.3 required.')
    selected=profile['selected_layers']
    if not selected or any(not n.startswith('dec.') for n in selected):
        raise ValueError('This experiment changes only the profiled decoder convolutions.')
    records,root,manifest=inventory()
    stats_path=DATA/'eval/native-decoded-probe/channel-statistics.json'
    bands_path=DATA/'eval/native-phase-train-band-statistics.json'
    stats,bands=json.loads(stats_path.read_text()),json.loads(bands_path.read_text())
    if bands['split']!='train' or bands['dataset_manifest_sha256']!=digest(root/'manifest.json') or bands['statistics_sha256']!=digest(stats_path):
        raise ValueError('Use the frozen matching TRAIN band powers.')
    if digest(teacher_source()/'diffusion_pytorch_model.safetensors')!=manifest['teacher_weights_sha256']:
        raise ValueError('Pinned original teacher changed.')
    sources=('native_basis_train.py','native_basis_training.py','native_basis_evaluate.py',
             'native_lowrank.py','native_feature_kd.py','native_phase_loss.py','native_decoded.py',
             'wet_mountain_progress.py','native_progress_images.py')
    contract=dict(teacher_revision=REVISION,teacher_weights_sha256=manifest['teacher_weights_sha256'],
        profile_sha256=digest(args.profile),basis_sha256=digest(DATA/'eval/native-lowrank-fp64-basis.pt'),
        dataset_manifest_sha256=digest(root/'manifest.json'),statistics_sha256=digest(stats_path),
        band_statistics_sha256=digest(bands_path),
        mountain_manifest_sha256={str(p):digest(p/'manifest.json') for p in
          (ROOT,DATA/'crops/native-wet-mountain-train-extension-v1')},
        sources_sha256={p:digest(REPO/'distill'/p) for p in sources},
        selected_layers=selected,target_rank_fraction=.25,anneal_steps=1000,
        target_steps=args.steps,seed=args.seed,batch=8,phases_per_batch=[4,4],
        optimizer='Adam beta .9/.99, clip1',factor_lr=5e-5,ordinary_lr=1e-5,
        ema_half_life=500,warmup_steps=100,feature_calibration_batches=50,
        feature_calibration_rank=.25,feature_calibration_target_ratio=.5,
        sampler='8 distinct TRAIN worlds,25% wet mountain,15% old rare,60% general; all3 coarse branches per phase',
        objective='normalized teacher operator + .05 gradient + balanced x0/TRAIN band terms + all-block/skip KD; double mountain band-energy',
        decoded_training_loss=False,prune_policy='FP64 re-SVD at1000; reset Adam and EMA in the new basis,100step warmup',
        evaluation='4090 asynchronous every500; mandatory wet mountain and old4rare cases; no deployment acceptance')
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text())!=contract:
        raise ValueError('Contract changed; preserve this experiment and use a new directory.')
    if not args.smoke:
        if args.smoke_root is None: raise ValueError('Successful exact-contract GPU smoke required.')
        smoke=json.loads((args.smoke_root/'gpu-smoke.json').read_text())
        if smoke['status']!='complete' or smoke['contract']!=contract or smoke['gpu']!='NVIDIA GeForce RTX 5090':
            raise ValueError('Smoke must verify this exact training recipe on the5090.')
    atomic_json(output/'contract.json',contract)
    torch.set_num_threads(8);torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    teacher=EDMUnet2D.from_pretrained(str(teacher_source())).cuda().eval().requires_grad_(False)
    basis=torch.load(DATA/'eval/native-lowrank-fp64-basis.pt',map_location='cpu',weights_only=True)
    model=selective_model(teacher,basis,selected).cuda()
    target_ranks={n:max(16,math.ceil(basis['factor_ranks'][n]*.25/16)*16) for n in selected}
    from types import SimpleNamespace
    from terrain_inference import prepare_models
    teacher=teacher.bfloat16()
    prepare_models(SimpleNamespace(coarse_model=teacher,base_model=teacher,decoder_model=teacher))
    variance=torch.tensor(stats['variance'],device='cuda')
    calibration={k:torch.tensor(v,device='cuda') for k,v in dict(
        first_power=bands['phases']['first']['band_power'],second_power=bands['phases']['second']['band_power'],
        second_variance=bands['phases']['second']['spatial_variance']).items()}
    bank=WindowBank(records,preload=not args.smoke)
    print(json.dumps(dict(train_records=len(records),gpu=torch.cuda.get_device_name(),preloaded=not args.smoke)),flush=True)
    def optimizer_for(net):
        factors=[p for n,p in net.named_parameters() if n.endswith(('.spatial','.pointwise'))]
        ordinary=[p for n,p in net.named_parameters() if not n.endswith(('.spatial','.pointwise'))]
        return torch.optim.Adam([dict(params=factors,lr=5e-5,base_lr=5e-5),
                                 dict(params=ordinary,lr=1e-5,base_lr=1e-5)],betas=(.9,.99),fused=True)
    optimizer=optimizer_for(model)
    # Calibrate on the target-rank discrepancy, not the full-rank BF16 floor.
    if args.smoke:
        calibration_model=selective_model(teacher,basis,selected,.25).cuda().requires_grad_(False)
        ratios=[]
        with torch.no_grad():
            for index in range(50):
                batch,_=bank.batch(records,np.random.default_rng(190120+index),index)
                _,terms,_=kd_step(calibration_model,teacher,batch,variance,calibration,0.)
                ratios.append(float(terms['operator']/terms['representation'].clamp_min(1e-8)))
        feature_weight=.5*float(np.median(ratios))
        del calibration_model;torch.cuda.empty_cache()
    else:
        feature_weight=smoke['feature_weight'];ratios=smoke['calibration_ratios']
    if not np.isfinite(feature_weight) or feature_weight<=0:
        raise ValueError('A finite positive TRAIN-calibrated KD weight is required.')
    if args.smoke:
        rows=[]
        for step in range(3):
            batch,selected_rows=bank.batch(records,np.random.default_rng(args.seed+step),step)
            optimizer.zero_grad(set_to_none=True)
            loss,terms,blocks=kd_step(model,teacher,batch,variance,calibration,feature_weight)
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            if any(p.grad is not None for p in teacher.parameters()): raise ValueError('Teacher received a gradient.')
            optimizer.step()
            rows.append(dict(loss=float(loss.detach()),grad_norm=float(norm),blocks=len(blocks),
                worlds=[r['metadata']['seed'] for _,r in selected_rows],
                variants=[r['coarse_variant'] for _,r in selected_rows],
                cases=[r['case_kind'] for _,r in selected_rows],peak_bytes=torch.cuda.max_memory_allocated(),
                gradients=sum(p.grad is not None for p in model.parameters())))
        del optimizer,loss,terms,blocks
        anneal(model,target_ranks,0.,1.)
        with torch.no_grad():
            before=model(batch['inputs'],batch['labels'],[batch['condition']])
        model=compact(model,target_ranks)
        with torch.no_grad():
            after=model(batch['inputs'],batch['labels'],[batch['condition']])
            cut_mse=float((before-after).square().mean()/before.square().mean().clamp_min(1e-8))
        if cut_mse>1e-3:raise ValueError('Post-cut SVD numerical equivalence failed.')
        optimizer=optimizer_for(model)
        batch,selected_rows=bank.batch(records,np.random.default_rng(args.seed+77),77)
        loss,terms,blocks=kd_step(model,teacher,batch,variance,calibration,feature_weight)
        optimizer.zero_grad(set_to_none=True);loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        atomic_json(output/'gpu-smoke.json',dict(status='complete',contract=contract,rows=rows,
                    teacher_frozen=True,gpu=torch.cuda.get_device_name(),accepted=False,
                    feature_weight=feature_weight,calibration_ratios=ratios,
                    cut_mse=cut_mse,post_cut_loss=float(loss.detach()),post_cut_grad_norm=float(norm),
                    post_cut_parameters=sum(p.numel() for p in model.parameters())))
        return
    if (output/'latest.pt').exists():
        raise ValueError('Do not restart or silently reset a previous optimizer; audit its exact saved state first.')
    del basis;torch.cuda.empty_cache()
    atomic_json(output/'feature-calibration.json',dict(split='train',rank_fraction=.25,batches=50,
        feature_weight=feature_weight,ratio_samples=ratios,contract=contract))
    ema=copy.deepcopy(model).eval().requires_grad_(False)
    started=time.monotonic();step=0;warmup_start=0
    def save(name,full=False):
        state=dict(config=model.config,ema=ema.state_dict(),step=step,architecture='native-lowrank',
            factor_ranks=model.ranks,factor_mode='factorized',rank_fraction=.25 if step>=1000 else 1.,
            teacher_revision=REVISION,teacher_weights_sha256=manifest['teacher_weights_sha256'],
            contract=contract,feature_weight=feature_weight,evaluation_only=False,accepted=False)
        if full:state.update(model=model.state_dict(),optimizer=optimizer.state_dict(),
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),warmup_start=warmup_start)
        atomic_write(output/name,lambda f:torch.save(state,f))
    save('step000000.pt')
    def stop(signum,frame):
        global STOP
        STOP=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    atomic_json(output/'status.json',dict(status='running',step=0,target_steps=args.steps))
    try:
        while step<args.steps and not STOP:
            evaluation=output/'evaluation-status.json'
            if evaluation.exists():
                decision=json.loads(evaluation.read_text())
                if decision.get('hard_stop') or decision.get('status')=='failed':
                    STOP=True;break
            generator=np.random.default_rng(np.random.SeedSequence([args.seed,step]))
            batch,selected_rows=bank.batch(records,generator,step)
            # On-policy inputs are supplied by the independent4090 evaluator.
            policy_index=output/'rollouts-index.json';policy_count=0
            if step>=1500 and policy_index.exists():
                policy=json.loads(policy_index.read_text())
                for i,(phase,row) in enumerate(selected_rows):
                    if phase!=1 or generator.random()>=.5:continue
                    used={r['metadata']['seed'] for j,(_,r) in enumerate(selected_rows) if j!=i}
                    pool=[r for r in policy['rows'] if r['case_kind']==row['case_kind'] and
                          r['coarse_variant']==row['coarse_variant'] and r['seed'] not in used]
                    if pool:
                        choice=pool[int(generator.integers(len(pool)))]
                        with np.load(choice['file'],allow_pickle=False) as sample:
                            if json.loads(str(sample['metadata']))['split']!='train':raise ValueError('On-policy leakage.')
                            j=int(generator.integers(len(sample['inputs'])))
                            for k in ('inputs','labels','condition'):
                                batch[k][i:i+1].copy_(torch.from_numpy(sample[k][j:j+1].copy()).cuda())
                        selected_rows[i]=(phase,choice['record'])
                        policy_count+=1
            optimizer.zero_grad(set_to_none=True)
            for group in optimizer.param_groups:
                group['lr']=group['base_lr']*min(1.,(step-warmup_start+1)/100)
            loss,terms,blocks=kd_step(model,teacher,batch,variance,calibration,feature_weight)
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite teacher-basis loss.')
            loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();step+=1
            with torch.no_grad():
                for a,p in zip(ema.parameters(),model.parameters()):a.lerp_(p,1-math.exp(math.log(.5)/500))
                if step<=1000:
                    anneal(model,target_ranks,(step-1)/1000,step/1000)
                    anneal(ema,target_ranks,(step-1)/1000,step/1000)
            if step==1000:
                save('anneal-end.pt',full=True)
                del optimizer
                model=compact(model,target_ranks)
                ema=copy.deepcopy(model).eval().requires_grad_(False)
                optimizer=optimizer_for(model);warmup_start=step
                atomic_json(output/'cut-lineage.json',dict(step=step,parent_sha256=digest(output/'anneal-end.pt'),
                    policy=contract['prune_policy'],new_ranks=target_ranks))
            if step==1 or step%25==0:
                row=dict(step=step,loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},
                    grad_norm=float(norm),on_policy_windows=policy_count,elapsed_seconds=time.monotonic()-started,
                    peak_bytes=torch.cuda.max_memory_allocated())
                append_json(output/'train.jsonl',row)
                atomic_json(output/'status.json',dict(status='running',step=step,target_steps=args.steps))
                print(json.dumps(row),flush=True)
            if step%500==0 or step==args.steps:
                save(f'step{step:06d}.pt');save('latest.pt',full=True)
        save('latest.pt',full=True)
        atomic_json(output/'status.json',dict(status='guarded' if STOP and evaluation.exists() and
            json.loads(evaluation.read_text()).get('hard_stop') else 'interrupted' if STOP else 'complete',
            step=step,target_steps=args.steps,elapsed_seconds=time.monotonic()-started,accepted=False))
    except Exception as exc:
        save('latest.pt',full=True)
        atomic_json(output/'status.json',dict(status='failed',step=step,error=str(exc),accepted=False))
        raise


if __name__=='__main__':main()

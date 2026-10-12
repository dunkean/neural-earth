"""Bounded unblended-window supervision after the guarded phase-coordinate run."""
import argparse
import copy
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch

from distill.common import DATA,REPO,atomic_json,atomic_write,append_json,external_path
from distill.field_phase import FieldPhaseBase
from distill.field_heads import FieldHeadsBase,is_head,recorded_heads,head_loss
from distill.field_scene import FieldScene,field_scene_forward
from distill.field_loss import field_loss,teacher_features,central_kd,shared_crop_loss
from distill.field_oracles import teacher_initial256
from distill.field_train import select
from distill.field_phase_train import ADAPTERS
from distill.native_basis_training import inventory
from distill.native_base import teacher_source
from distill.wet_mountains import KIND,ROOT,digest
from distill.wet_mountain_progress import selected_validation
from terrain_diffusion.models.edm_unet import EDMUnet2D

STOP=False


def resume_optimizer(model,parent):
    named=list(model.named_parameters())
    groups=[[p for n,p in named if n not in ADAPTERS and not is_head(n)],
            [p for n,p in named if n in ADAPTERS],
            [p for n,p in named if is_head(n)]]
    optimizer=torch.optim.Adam([dict(params=g,lr=1e-4 if i==2 else 3e-5) for i,g in enumerate(groups)],
                               betas=(.9,.99),fused=True)
    merged=optimizer.state_dict()
    if len(parent['param_groups'])!=2:raise ValueError('Two original phase parameter groups required.')
    for i in (0,1):
        if merged['param_groups'][i]['params']!=parent['param_groups'][i]['params']:
            raise ValueError('Do not shift inherited Adam IDs when inserting head parameters.')
        merged['param_groups'][i]=copy.deepcopy(parent['param_groups'][i])
        merged['param_groups'][i]['lr']=3e-5
    merged['state']=parent['state'];optimizer.load_state_dict(merged)
    for group,prior in zip(groups[:2],parent['param_groups']):
        for parameter,index in zip(group,prior['params']):
            for key,value in parent['state'].get(index,{}).items():
                actual=optimizer.state[parameter][key]
                if torch.is_tensor(value):
                    if not torch.equal(actual.cpu(),value.cpu()):raise ValueError('Parent Adam moment changed.')
                elif actual!=value:raise ValueError('Parent Adam state changed.')
            if index not in parent['state'] and optimizer.state.get(parameter):raise ValueError('Missing parent state changed.')
    if any(optimizer.state.get(p) for p in groups[2]):raise ValueError('New heads need fresh Adam state.')
    return optimizer


def main():
    global STOP
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=DATA/'ckpt/field-phase-pilot/latest.pt')
    parser.add_argument('--output',type=Path,default=DATA/'ckpt/field-heads-pilot')
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-root',type=Path)
    args=parser.parse_args();output=external_path(args.output)
    if (output/'latest.pt').exists():raise ValueError('No implicit head-run restart.')
    saved=torch.load(args.parent,map_location='cpu',weights_only=True)
    if saved['architecture']!='native-field-phase' or saved['step']!=2037:raise ValueError('Guarded B2037 parent required.')
    audit_path=DATA/'eval/field-first-features-final.json'
    audit=json.loads(audit_path.read_text())['rows']
    old={r['seed']:r['channels'][0]['mse_over_train_variance'] for r in audit
         if r['architecture']=='native-field' and r['coarse']=='coarse4'}
    current=[r for r in audit if r['architecture']=='native-field-phase' and r['coarse']=='coarse4']
    if len(current)!=3 or any(1-r['channels'][0]['mse_over_train_variance']/old[r['seed']]>=.3 for r in current):
        raise ValueError('Opus conditional gate: continue B if it improved channel0 by30%.')
    records,main_root,_=inventory()
    stats_path=DATA/'eval/native-decoded-probe/channel-statistics.json'
    bands_path=DATA/'eval/native-phase-train-band-statistics.json'
    stats,bands=json.loads(stats_path.read_text()),json.loads(bands_path.read_text())
    if bands['split']!='train' or bands['dataset_manifest_sha256']!=digest(main_root/'manifest.json'):
        raise ValueError('Frozen TRAIN calibration required.')
    sources=('field_base.py','field_phase.py','field_heads.py','field_scene.py','field_loss.py','field_oracles.py',
             'field_train.py','field_phase_train.py','field_heads_train.py','field_evaluate.py','field_heads_watch.py','field_attention_diagnostic.py')
    seed=saved['contract']['parent_contract']['seed'];global_start=saved['parent_step']+saved['step']
    contract=dict(parent_sha256=digest(args.parent),parent_phase_step=saved['step'],global_start=global_start,
        parent_contract=saved['contract'],channel0_condition_sha256=digest(audit_path),
        source_sha256={p:digest(REPO/'distill'/p) for p in sources},
        statistics_sha256=digest(stats_path),band_statistics_sha256=digest(bands_path),
        main_manifest_sha256=digest(main_root/'manifest.json'),
        mountain_manifests={str(root):digest(root/'manifest.json') for root in
                           (ROOT,DATA/'crops/native-wet-mountain-train-extension-v1')},
        steps=args.steps,seed=seed,heads=4,head_classes='native window-origin parity(y//32 mod2,x//32 mod2)',
        blend='native linear_weight_window64, world-anchored per-head period64, fixed normalized weights',
        initialization='parent raw+EMA, exact inherited Adam and RNG; copied output MPConv and gains, fresh head Adam',
        first_target='blended initial192 plus masked atlas of49 unblended first64 window outputs, all5channels',
        loss='parent initial1.0/final.5 plus normalized per-head MSE1.0, other losses and KD unchanged',
        forcing=.25,sampler='parent25%wet/15%oldrare/60%general,3coarses, continue global counter',
        lr=3e-5,head_lr=1e-4,warmup=100,ema_half_life=500,seam='unforced both sides,every4global steps,x/y alternating',
        accepted=False)
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text())!=contract:
        raise ValueError('Use a fresh output root.')
    if not args.smoke:
        if args.smoke_root is None:raise ValueError('Exact head gradient smoke required.')
        proof=json.loads((args.smoke_root/'gpu-smoke.json').read_text())
        if proof['contract']!=contract or proof['status']!='complete':raise ValueError('Head smoke recipe changed.')
    atomic_json(output/'contract.json',contract)
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=FieldHeadsBase(saved['config']).cuda().eval();model.load_parent(saved['model'])
    ema=copy.deepcopy(model).eval().requires_grad_(False);ema.load_parent(saved['ema'])
    legacy=FieldPhaseBase(saved['config'])
    if [n for n,_ in legacy.named_parameters()]!=[n for n,_ in model.named_parameters() if not is_head(n)]:
        raise ValueError('Inherited parameter names changed order.')
    identities=[]
    if args.smoke:
        legacy=legacy.cuda().eval().requires_grad_(False)
        with torch.no_grad():
            for coarse in ('original','coarse4','coarse2'):
                record=selected_validation(ROOT,coarse)[0][0]
                scene=FieldScene(ROOT/record['file'],'cuda')
                recorded_heads(ROOT/record['file'],scene)
                for key,candidate in (('model',model),('ema',ema)):
                    legacy.load_state_dict(saved[key]);a=field_scene_forward(legacy,scene);b=field_scene_forward(candidate,scene)
                    errors={k:float((a[k]-b[k]).square().mean()) for k in ('initial','final')}
                    if max(errors.values())>1e-9:raise ValueError('Identical heads changed the parent trajectory beyond FP32 blend rounding.')
                    identities.append(dict(coarse=coarse,weights=key,mse=errors,
                        max_abs={k:float((a[k]-b[k]).abs().max()) for k in ('initial','final')}))
        del a,b,scene
    del legacy
    optimizer=resume_optimizer(model,saved['optimizer']);feature_weight=saved['feature_weight']
    rng,cuda_rng=saved['rng'],saved['cuda_rng'];teacher_revision=saved['teacher_revision'];teacher_sha=saved['teacher_weights_sha256']
    del saved
    teacher=EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    from types import SimpleNamespace
    from terrain_inference import prepare_models
    prepare_models(SimpleNamespace(coarse_model=teacher,base_model=teacher,decoder_model=teacher))
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight=linear_weight_window(64,'cuda',torch.float32)
    variance=torch.tensor(stats['variance'],device='cuda')
    calibration={k:torch.tensor(v,device='cuda') for k,v in dict(
        first_power=bands['phases']['first']['band_power'],second_power=bands['phases']['second']['band_power']).items()}
    torch.set_rng_state(rng);torch.cuda.set_rng_state_all(cuda_rng)
    if not torch.equal(torch.get_rng_state(),rng) or any(not torch.equal(a,b) for a,b in zip(torch.cuda.get_rng_state_all(),cuda_rng)):
        raise ValueError('Parent RNG restore failed.')
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    def training_step(step,force=None):
        for i,group in enumerate(optimizer.param_groups):group['lr']=(1e-4 if i==2 else 3e-5)*min(1.,(step+1)/100)
        global_step=global_start+step;generator=np.random.default_rng(np.random.SeedSequence([seed,global_step]))
        record=select(records,generator,global_step);path=Path(record['root'])/record['file'];scene=FieldScene(path,'cuda')
        reference=teacher_features(teacher,scene);targets,mask=recorded_heads(path,scene)
        forced=generator.random()<.25 if force is None else force
        initial,reconstruction=teacher_initial256(path,scene,weight) if forced else (None,None)
        optimizer.zero_grad(set_to_none=True)
        result=field_scene_forward(model,scene,checkpointed=True,features=True,initial_override=initial)
        loss,terms=field_loss(result,scene,variance,calibration,record['case_kind']==KIND)
        heads,individual=head_loss(result['initial_heads'],targets,mask,variance)
        feature,blocks=central_kd(result,reference);loss=loss+.5*terms['initial']+heads+feature_weight*feature
        seam=loss.new_zeros(())
        if global_step%4==0:
            axis='x' if (global_step//4)%2==0 else 'y'
            current=field_scene_forward(model,scene,checkpointed=True)['core'] if forced else result['core']
            moved=FieldScene(path,'cuda',shift=(0,32) if axis=='x' else (32,0))
            with torch.no_grad():other=field_scene_forward(model,moved)['core']
            seam=shared_crop_loss(current,other,axis,variance);loss=loss+.1*seam
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite head-field loss.')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        grads={n:float(p.grad.abs().sum()) for n,p in model.named_parameters() if is_head(n)}
        if not all(np.isfinite(v) and v>0 for v in grads.values()):raise ValueError('Missing per-head gradients.')
        if any(p.grad is not None for p in teacher.parameters()):raise ValueError('Teacher received gradients.')
        optimizer.step()
        return dict(loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},
            head_mse=float(heads.detach()),per_head_mse=individual.detach().cpu().tolist(),head_gradients=grads,
            representation=float(feature.detach()),seam=float(seam.detach()),grad_norm=float(norm),
            forced=forced,reconstruction=reconstruction,seed=record['metadata']['seed'],coarse=record['coarse_variant'],
            kind=record['case_kind'],peak_bytes=torch.cuda.max_memory_allocated())
    if args.smoke:
        rows=[training_step(step,force) for step,force in ((0,True),(1,False),(2,False),(6,True))]
        atomic_json(output/'gpu-smoke.json',dict(status='complete',contract=contract,rows=rows,
            copied_head_identity=identities,parent_optimizer_exact=True,parent_rng_exact=True,
            teacher_frozen=True,gpu=torch.cuda.get_device_name(),accepted=False))
        return
    step=0;started=time.monotonic()
    def save(name,full=False):
        state=dict(config=model.config,ema=ema.state_dict(),architecture='native-field-heads',step=step,
            parent_global_step=global_start,contract=contract,feature_weight=feature_weight,
            teacher_revision=teacher_revision,teacher_weights_sha256=teacher_sha,
            evaluation_only=False,accepted=False,field_size=256,core_size=128,halo=64,phases=2)
        if full:state.update(model=model.state_dict(),optimizer=optimizer.state_dict(),rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        atomic_write(output/name,lambda f:torch.save(state,f))
    save('step000000.pt')
    def stop(signum,frame):
        global STOP
        STOP=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    atomic_json(output/'status.json',dict(status='running',step=0,target_steps=args.steps,global_start=global_start))
    guarded=False
    try:
        while step<args.steps and not STOP:
            evaluation=output/'evaluation-status.json'
            if evaluation.exists() and json.loads(evaluation.read_text()).get('hard_stop'):guarded=True;break
            row=training_step(step);step+=1
            with torch.no_grad():
                for a,p in zip(ema.parameters(),model.parameters()):a.lerp_(p,1-math.exp(math.log(.5)/500))
            if step==1 or step%25==0:
                row.update(step=step,global_step=global_start+step,elapsed_seconds=time.monotonic()-started)
                append_json(output/'train.jsonl',row);print(json.dumps(row),flush=True)
                atomic_json(output/'status.json',dict(status='running',step=step,target_steps=args.steps,global_start=global_start))
            if step%500==0 or step==args.steps:save(f'step{step:06d}.pt');save('latest.pt',full=True)
        save('latest.pt',full=True)
        atomic_json(output/'status.json',dict(status='guarded' if guarded else 'interrupted' if STOP else 'complete',
            step=step,target_steps=args.steps,global_start=global_start,elapsed_seconds=time.monotonic()-started,accepted=False))
    except Exception as exc:
        save('latest.pt',full=True)
        atomic_json(output/'status.json',dict(status='failed',step=step,error=str(exc),accepted=False));raise


if __name__=='__main__':main()

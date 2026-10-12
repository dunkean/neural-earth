"""Bounded phase-coordinate experiment, preserving parent Adam/EMA state."""
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
from distill.field_base import FieldBase
from distill.field_phase import FieldPhaseBase
from distill.field_scene import FieldScene,field_scene_forward
from distill.field_loss import field_loss,teacher_features,central_kd,shared_crop_loss
from distill.field_oracles import teacher_initial256
from distill.field_train import select
from distill.native_basis_training import inventory
from distill.native_base import teacher_source
from distill.wet_mountains import KIND,ROOT,digest
from distill.wet_mountain_progress import selected_validation
from terrain_diffusion.models.edm_unet import EDMUnet2D

STOP=False
ADAPTERS={'net.field_phase_stem','net.field_phase_embedding'}


def resume_optimizer(model,saved):
    named=list(model.named_parameters())
    old=[p for n,p in named if n not in ADAPTERS]
    new=[p for n,p in named if n in ADAPTERS]
    optimizer=torch.optim.Adam([dict(params=old,lr=3e-5),dict(params=new,lr=1e-4)],
                               betas=(.9,.99),fused=True)
    merged=optimizer.state_dict();parent=saved['optimizer']
    if len(parent['param_groups'])!=1 or merged['param_groups'][0]['params']!=parent['param_groups'][0]['params']:
        raise ValueError('Existing parameter IDs must preserve parent order.')
    merged['state']=parent['state']
    merged['param_groups'][0]=copy.deepcopy(parent['param_groups'][0])
    optimizer.load_state_dict(merged)
    for parameter,index in zip(old,parent['param_groups'][0]['params']):
        if index not in parent['state']:
            if optimizer.state.get(parameter):raise ValueError('Unused parent parameter acquired Adam state.')
            continue
        for key,value in parent['state'][index].items():
            actual=optimizer.state[parameter][key]
            if torch.is_tensor(value):
                if not torch.equal(actual.cpu(),value.cpu()):raise ValueError('Parent Adam moment changed.')
            elif actual!=value:raise ValueError('Parent Adam state changed.')
    if any(optimizer.state.get(p) for p in new):raise ValueError('New adapters must have fresh Adam state.')
    return optimizer


def main():
    global STOP
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=DATA/'ckpt/field-full-width-pilot/latest.pt')
    parser.add_argument('--output',type=Path,default=DATA/'ckpt/field-phase-pilot')
    parser.add_argument('--steps',type=int,default=4000)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-root',type=Path)
    args=parser.parse_args();output=external_path(args.output)
    if (output/'latest.pt').exists():raise ValueError('No implicit restart of an existing phase optimizer.')
    saved=torch.load(args.parent,map_location='cpu',weights_only=True)
    if saved['architecture']!='native-field' or saved['step']!=2037:
        raise ValueError('This frozen experiment resumes the guarded full-field2037 parent.')
    records,main_root,_=inventory()
    stats_path=DATA/'eval/native-decoded-probe/channel-statistics.json'
    bands_path=DATA/'eval/native-phase-train-band-statistics.json'
    stats,bands=json.loads(stats_path.read_text()),json.loads(bands_path.read_text())
    if bands['split']!='train' or bands['dataset_manifest_sha256']!=digest(main_root/'manifest.json'):
        raise ValueError('Frozen TRAIN-only calibration required.')
    sources=('field_base.py','field_phase.py','field_scene.py','field_loss.py','field_oracles.py',
             'field_train.py','field_phase_train.py','field_evaluate.py','field_phase_watch.py','field_attention_diagnostic.py')
    contract=dict(parent_sha256=digest(args.parent),parent_step=saved['step'],parent_contract=saved['contract'],
        sources_sha256={p:digest(REPO/'distill'/p) for p in sources},
        statistics_sha256=digest(stats_path),band_statistics_sha256=digest(bands_path),
        mountain_manifests={str(root):digest(root/'manifest.json') for root in
                           (ROOT,DATA/'crops/native-wet-mountain-train-extension-v1')},
        main_manifest_sha256=digest(main_root/'manifest.json'),steps=args.steps,
        phase_features=dict(period=64,harmonics=[1,2,3,4],channels=16,projections='zero additive stem192 and embedding768'),
        initialization='raw2037, exact parent Adam moments and RNG, parent EMA, fresh adapter Adam',
        lr=3e-5,adapter_lr=1e-4,adapter_warmup=200,old_lr_warmup=False,ema_half_life=500,
        forcing='50% before new1500,25% afterwards; detached teacher initial256 from captured49 windows; outer32 partial blending',
        loss='first1.0/final.5; remaining parent terms and fixed feature calibration unchanged',
        sampling='parent seed and global2037+new step,25%wet/15%oldrare/60%general,3coarse branches',
        evaluation='unforced only; every500; original eligibility guard retained; .03 correlation stop and two consecutive phase1 amplitude regressions',
        accepted=False)
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text())!=contract:
        raise ValueError('Use a fresh experiment root.')
    if not args.smoke:
        if args.smoke_root is None:raise ValueError('Exact smoke proof required.')
        proof=json.loads((args.smoke_root/'gpu-smoke.json').read_text())
        if proof['contract']!=contract or proof['status']!='complete':raise ValueError('Smoke recipe changed.')
    atomic_json(output/'contract.json',contract)
    torch.set_num_threads(8);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=FieldPhaseBase(saved['config']).cuda().eval();model.load_parent(saved['model'])
    ema=copy.deepcopy(model).eval().requires_grad_(False);ema.load_parent(saved['ema'])
    # Filtering root adapters restores legacy parameter order before assigning
    # integer Adam IDs. Checking names also catches same-shaped reorderings.
    legacy=FieldBase(saved['config'])
    legacy_names=[n for n,_ in legacy.named_parameters()]
    if legacy_names!=[n for n,_ in model.named_parameters() if n not in ADAPTERS]:
        raise ValueError('Legacy parameter names were reordered.')
    identities=[]
    if args.smoke:
        legacy=legacy.cuda().eval().requires_grad_(False)
        with torch.no_grad():
            for branch in ('original','coarse4','coarse2'):
                sample=selected_validation(ROOT,branch)[0][0]
                scene=FieldScene(ROOT/sample['file'],'cuda')
                for key,candidate in (('model',model),('ema',ema)):
                    legacy.load_state_dict(saved[key])
                    a=field_scene_forward(legacy,scene);b=field_scene_forward(candidate,scene)
                    if not all(torch.equal(a[k],b[k]) for k in ('initial','final')):
                        raise ValueError('Zero adapters changed the resumed function.')
                    identities.append(dict(coarse=branch,weights=key,initial_bitwise=True,final_bitwise=True))
        del a,b,scene
    del legacy
    optimizer=resume_optimizer(model,saved)
    feature_weight=saved['feature_weight'];parent_step=saved['step'];seed=saved['contract']['seed']
    rng,cuda_rng=saved['rng'],saved['cuda_rng'];del saved
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
        optimizer.param_groups[1]['lr']=1e-4*min(1.,(step+1)/200)
        global_step=parent_step+step
        generator=np.random.default_rng(np.random.SeedSequence([seed,global_step]))
        record=select(records,generator,global_step);path=Path(record['root'])/record['file']
        scene=FieldScene(path,'cuda');reference=teacher_features(teacher,scene)
        forced=generator.random()<(0.5 if step<1500 else .25) if force is None else force
        initial,reconstruction=teacher_initial256(path,scene,weight) if forced else (None,None)
        optimizer.zero_grad(set_to_none=True)
        result=field_scene_forward(model,scene,checkpointed=True,features=True,initial_override=initial)
        loss,terms=field_loss(result,scene,variance,calibration,record['case_kind']==KIND)
        loss=loss+.5*terms['initial']
        feature,blocks=central_kd(result,reference);loss=loss+feature_weight*feature
        seam=loss.new_zeros(())
        if global_step%4==0:
            axis='x' if (global_step//4)%2==0 else 'y'
            # Seam supervision must compare the deployed, unforced chain on
            # both sides, rather than a forced crop with an unforced neighbor.
            current=field_scene_forward(model,scene,checkpointed=True)['core'] if forced else result['core']
            shifted=FieldScene(path,'cuda',shift=(0,32) if axis=='x' else (32,0))
            with torch.no_grad():other=field_scene_forward(model,shifted)['core']
            seam=shared_crop_loss(current,other,axis,variance);loss=loss+.1*seam
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite phase loss.')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        adapter_grads={n:float(p.grad.abs().sum()) for n,p in model.named_parameters() if n in ADAPTERS}
        if not all(np.isfinite(v) and v>0 for v in adapter_grads.values()):raise ValueError('Missing adapter gradients.')
        if any(p.grad is not None for p in teacher.parameters()):raise ValueError('Teacher received gradients.')
        optimizer.step()
        levels={}
        for name,value in blocks.items():
            phase,block=name.split(':');key=phase+':'+block.split('_')[0]
            levels.setdefault(key,[]).append(float(value.detach()))
        return dict(loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},
            representation=float(feature.detach()),kd_levels={k:float(np.mean(v)) for k,v in levels.items()},
            seam=float(seam.detach()),grad_norm=float(norm),adapter_gradients=adapter_grads,
            forced=forced,reconstruction=reconstruction,seed=record['metadata']['seed'],
            coarse=record['coarse_variant'],kind=record['case_kind'],peak_bytes=torch.cuda.max_memory_allocated())
    if args.smoke:
        rows=[training_step(step,force) for step,force in ((0,True),(1,False),(3,True))]
        atomic_json(output/'gpu-smoke.json',dict(status='complete',contract=contract,rows=rows,
            zero_adapter_identity=identities,parent_optimizer_exact=True,parent_rng_exact=True,
            teacher_frozen=True,gpu=torch.cuda.get_device_name(),accepted=False))
        return
    step=0;started=time.monotonic()
    def save(name,full=False):
        state=dict(config=model.config,ema=ema.state_dict(),architecture='native-field-phase',step=step,
            parent_step=parent_step,contract=contract,feature_weight=feature_weight,
            teacher_revision=contract['parent_contract']['teacher_revision'],
            teacher_weights_sha256=contract['parent_contract']['teacher_weights_sha256'],
            evaluation_only=False,accepted=False,field_size=256,core_size=128,halo=64,phases=2)
        if full:state.update(model=model.state_dict(),optimizer=optimizer.state_dict(),
                             rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        atomic_write(output/name,lambda f:torch.save(state,f))
    save('step000000.pt')
    def stop(signum,frame):
        global STOP
        STOP=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    atomic_json(output/'status.json',dict(status='running',step=0,target_steps=args.steps,parent_step=parent_step))
    guarded=False
    try:
        while step<args.steps and not STOP:
            evaluation=output/'evaluation-status.json'
            if evaluation.exists() and json.loads(evaluation.read_text()).get('hard_stop'):
                guarded=True;break
            row=training_step(step);step+=1
            with torch.no_grad():
                for a,p in zip(ema.parameters(),model.parameters()):a.lerp_(p,1-math.exp(math.log(.5)/500))
            if step==1 or step%25==0:
                row.update(step=step,parent_step=parent_step,elapsed_seconds=time.monotonic()-started)
                append_json(output/'train.jsonl',row);print(json.dumps(row),flush=True)
                atomic_json(output/'status.json',dict(status='running',step=step,target_steps=args.steps,parent_step=parent_step))
            if step%500==0 or step==args.steps:
                save(f'step{step:06d}.pt');save('latest.pt',full=True)
        save('latest.pt',full=True)
        atomic_json(output/'status.json',dict(status='guarded' if guarded else 'interrupted' if STOP else 'complete',
            step=step,target_steps=args.steps,parent_step=parent_step,elapsed_seconds=time.monotonic()-started,accepted=False))
    except Exception as exc:
        save('latest.pt',full=True)
        atomic_json(output/'status.json',dict(status='failed',step=step,error=str(exc),accepted=False));raise


if __name__=='__main__':main()

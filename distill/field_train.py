"""Train the full-width two-phase field pupil against original blended teacher fields."""
import argparse
import copy
import json
import math
from pathlib import Path
import signal
import time

import numpy as np
import torch

from distill.common import DATA,REPO,REVISION,atomic_json,atomic_write,append_json,external_path
from distill.field_base import FieldBase
from distill.field_scene import FieldScene,field_scene_forward
from distill.field_loss import field_loss,teacher_features,central_kd,shared_crop_loss
from distill.native_base import teacher_source
from distill.native_basis_training import inventory
from distill.native_mixed_bank import KINDS,VARIANTS
from distill.wet_mountains import KIND,ROOT,digest
from terrain_diffusion.models.edm_unet import EDMUnet2D

STOP=False


def select(records,generator,step):
    variant=VARIANTS[step%3]
    # 20-slot cycle:5 mountains,3old rare,12general; independent random worlds.
    slot=step%20
    kind=KIND if slot<5 else KINDS[(step//20+slot)%4] if slot<8 else None
    pool=[r for r in records if r['coarse_variant']==variant and r['case_kind']==kind]
    profiles=sorted({r['metadata']['profile'] for r in pool})
    profile=profiles[int(generator.integers(len(profiles)))]
    pool=[r for r in pool if r['metadata']['profile']==profile]
    return pool[int(generator.integers(len(pool)))]


def main():
    global STOP
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=DATA/'ckpt/field-full-width-pilot')
    parser.add_argument('--source',type=Path,default=DATA/'eval/field-base-initial-probe/initial.pt')
    parser.add_argument('--steps',type=int,default=6000)
    parser.add_argument('--seed',type=int,default=202610128)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-root',type=Path)
    args=parser.parse_args();output=external_path(args.output)
    records,main_root,manifest=inventory()
    bands_path=DATA/'eval/native-phase-train-band-statistics.json'
    stats_path=DATA/'eval/native-decoded-probe/channel-statistics.json'
    stats,bands=json.loads(stats_path.read_text()),json.loads(bands_path.read_text())
    if bands['split']!='train' or bands['dataset_manifest_sha256']!=digest(main_root/'manifest.json'):
        raise ValueError('Frozen TRAIN calibration required.')
    sources=('field_base.py','field_scene.py','field_loss.py','field_train.py','field_evaluate.py','field_watch.py')
    contract=dict(source_sha256=digest(args.source),teacher_revision=REVISION,
        teacher_weights_sha256=digest(teacher_source()/'diffusion_pytorch_model.safetensors'),
        sources_sha256={p:digest(REPO/'distill'/p) for p in sources},
        manifest_sha256=digest(main_root/'manifest.json'),statistics_sha256=digest(stats_path),
        band_statistics_sha256=digest(bands_path),
        mountain_manifests={str(root):digest(root/'manifest.json') for root in
                           (ROOT,DATA/'crops/native-wet-mountain-train-extension-v1')},
        input_field=256,core=128,halo=64,phases=2,teacher_width=192,
        condition_spacing=32,condition_centres='captured native64 window centres in absolute world coordinates',
        attention='nonoverlapping64-latent world-anchored local windows',
        halo_note='NOT a mathematical full-UNet RF guarantee; crop consistency is a measured acceptance requirement',
        first_target='original blended initial192',final_target='original blended final128',
        kd='both phases, central32 crop of matched64 teacher window, all37 encoder/decoder outputs',
        field_loss='normalized5-channel initial/final MSE + .05gradient + .1position bands + .05energy; mountain energy doubled',
        seam_weight=.1,seam_fraction=.25,seam_policy='alternate axes, shifted32 same-world fields; auxiliary crop detached',
        feature_calibration_batches=16,feature_target_ratio=.5,
        lr=3e-5,warmup=200,clip=1.,ema_half_life=500,
        sampler='25%wet mountains/15%oldrare/60%general; original/coarse4/coarse2 cycle',
        steps=args.steps,seed=args.seed,eval_every=500,decoded_training_loss=False)
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text())!=contract:
        raise ValueError('Preserve existing contracts; use a new output root.')
    if not args.smoke:
        if args.smoke_root is None:raise ValueError('Exact-recipe gradient smoke required.')
        proof=json.loads((args.smoke_root/'gpu-smoke.json').read_text())
        if proof['contract']!=contract or proof['status']!='complete':raise ValueError('GPU smoke recipe changed.')
    atomic_json(output/'contract.json',contract)
    torch.set_num_threads(8);torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    teacher=EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    from types import SimpleNamespace
    from terrain_inference import prepare_models
    prepare_models(SimpleNamespace(coarse_model=teacher,base_model=teacher,decoder_model=teacher))
    source=torch.load(args.source,map_location='cpu',weights_only=True)
    model=FieldBase(source['config']).cuda().eval();model.load_state_dict(source['ema']);del source
    variance=torch.tensor(stats['variance'],device='cuda')
    calibration={key:torch.tensor(value,device='cuda') for key,value in dict(
        first_power=bands['phases']['first']['band_power'],second_power=bands['phases']['second']['band_power']).items()}
    if args.smoke:
        ratios=[]
        with torch.no_grad():
            for index in range(16):
                r=select(records,np.random.default_rng(78400+index),index)
                scene=FieldScene(Path(r['root'])/r['file'],'cuda')
                result=field_scene_forward(model,scene,features=True)
                reference=teacher_features(teacher,scene);feature,_=central_kd(result,reference)
                residual=.5*sum(((pred-target[None]).square()/variance[None,:,None,None]).mean()
                    for pred,target in ((result['initial192'],scene.initial),(result['core'],scene.target)))
                ratios.append(float(residual/feature.clamp_min(1e-8)))
        feature_weight=.5*float(np.median(ratios))
        del result,reference;torch.cuda.empty_cache()
    else:feature_weight=proof['feature_weight'];ratios=proof['calibration_ratios']
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-5,betas=(.9,.99),fused=True)
    def training_step(step):
        generator=np.random.default_rng(np.random.SeedSequence([args.seed,step]))
        record=select(records,generator,step)
        scene=FieldScene(Path(record['root'])/record['file'],'cuda')
        reference=teacher_features(teacher,scene)
        optimizer.zero_grad(set_to_none=True)
        result=field_scene_forward(model,scene,checkpointed=True,features=True)
        loss,terms=field_loss(result,scene,variance,calibration,record['case_kind']==KIND)
        feature,blocks=central_kd(result,reference);loss=loss+feature_weight*feature
        seam=loss.new_zeros(())
        if step%4==0:
            axis='x' if (step//4)%2==0 else 'y'
            shifted=FieldScene(Path(record['root'])/record['file'],'cuda',shift=(0,32) if axis=='x' else (32,0))
            with torch.no_grad():other=field_scene_forward(model,shifted)['core']
            seam=shared_crop_loss(result['core'],other,axis,variance)
            loss=loss+.1*seam
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite full-width field loss.')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if any(p.grad is not None for p in teacher.parameters()):raise ValueError('Frozen teacher received gradients.')
        if not any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()):raise ValueError('No pupil gradients.')
        optimizer.step()
        return dict(loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},
            representation=float(feature.detach()),seam=float(seam.detach()),grad_norm=float(norm),
            blocks=len(blocks),seed=record['metadata']['seed'],coarse=record['coarse_variant'],kind=record['case_kind'],
            peak_bytes=torch.cuda.max_memory_allocated())
    if args.smoke:
        rows=[training_step(step) for step in (0,1,2)]
        atomic_json(output/'gpu-smoke.json',dict(status='complete',contract=contract,rows=rows,
            feature_weight=feature_weight,calibration_ratios=ratios,teacher_frozen=True,
            gpu=torch.cuda.get_device_name(),accepted=False))
        return
    if (output/'latest.pt').exists():raise ValueError('Existing optimizer requires an explicit resume audit, no implicit restart.')
    ema=copy.deepcopy(model).eval().requires_grad_(False);step=0;started=time.monotonic()
    def save(name,full=False):
        state=dict(config=model.config,ema=ema.state_dict(),architecture='native-field',step=step,
            teacher_revision=REVISION,teacher_weights_sha256=contract['teacher_weights_sha256'],
            contract=contract,feature_weight=feature_weight,evaluation_only=False,accepted=False,
            field_size=256,core_size=128,halo=64,phases=2)
        if full:state.update(model=model.state_dict(),optimizer=optimizer.state_dict(),
            rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        atomic_write(output/name,lambda f:torch.save(state,f))
    save('step000000.pt')
    def stop(signum,frame):
        global STOP
        STOP=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    atomic_json(output/'status.json',dict(status='running',step=0,target_steps=args.steps))
    guarded=False
    try:
        while step<args.steps and not STOP:
            evaluation=output/'evaluation-status.json'
            if evaluation.exists() and json.loads(evaluation.read_text()).get('hard_stop'):
                guarded=True;break
            for group in optimizer.param_groups:group['lr']=3e-5*min(1.,(step+1)/200)
            row=training_step(step);step+=1
            with torch.no_grad():
                for a,p in zip(ema.parameters(),model.parameters()):a.lerp_(p,1-math.exp(math.log(.5)/500))
            if step==1 or step%25==0:
                row.update(step=step,elapsed_seconds=time.monotonic()-started)
                append_json(output/'train.jsonl',row);print(json.dumps(row),flush=True)
                atomic_json(output/'status.json',dict(status='running',step=step,target_steps=args.steps))
            if step%500==0 or step==args.steps:
                save(f'step{step:06d}.pt');save('latest.pt',full=True)
        save('latest.pt',full=True)
        atomic_json(output/'status.json',dict(status='guarded' if guarded else 'interrupted' if STOP else 'complete',
            step=step,target_steps=args.steps,elapsed_seconds=time.monotonic()-started,accepted=False))
    except Exception as exc:
        save('latest.pt',full=True)
        atomic_json(output/'status.json',dict(status='failed',step=step,error=str(exc),accepted=False));raise


if __name__=='__main__':main()

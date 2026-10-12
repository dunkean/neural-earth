"""Fit packed weight-only NF4 while preserving the native two-stage function."""
import argparse
import copy
import json
import math
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from distill.common import DATA, REPO, atomic_json, atomic_write, append_json, external_path
from distill.native_base import teacher_source
from distill.native_basis_training import inventory, capture_blocks
from distill.native_feature_kd import representation_loss
from distill.native_fp8_training import QATBank
from distill.native_phase_loss import x0_from_velocity
from distill.native_weight_quant import load_weight_quant, file_sha
from terrain_diffusion.models.edm_unet import EDMUnet2D

STOP=False


def loss_terms(prediction,target,batch,variance):
    scale=variance.reshape(1,5,1,1).detach()
    residual=(prediction-target.detach())/scale.sqrt()
    first=batch['labels']>1
    operator=residual.square().mean()
    channel0=residual[first,0].square().mean()
    gradient=sum(residual.diff(dim=axis).square().mean() for axis in (-1,-2))*.5
    actual=x0_from_velocity(batch['inputs'],prediction,batch['labels'])
    reference=x0_from_velocity(batch['inputs'],target.detach(),batch['labels']).detach()
    x0=((actual-reference).square()/scale).mean()
    # Preserve positions and amplitudes by matching the target. No stand-alone
    # spectrum-energy objective that could reward uncorrelated grain.
    weighted=operator+.1*channel0
    return weighted+.05*gradient+.25*x0,dict(operator=operator,weighted_operator=weighted,
        first_channel0_operator=channel0,gradient=gradient,x0=x0)


def step_loss(model,teacher,batch,variance,feature_weight):
    with torch.no_grad(),capture_blocks(teacher) as reference:
        target=teacher(batch['inputs'].bfloat16(),noise_labels=batch['labels'].bfloat16(),
                       conditional_inputs=[batch['condition'].bfloat16()]).float()
    with capture_blocks(model.net) as actual:
        prediction=model(batch['inputs'],batch['labels'],[batch['condition']])
    feature,blocks=representation_loss(actual,reference)
    loss,terms=loss_terms(prediction,target,batch,variance)
    return loss+feature_weight*feature,terms|dict(representation=feature),blocks


def main():
    global STOP
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=6000)
    parser.add_argument('--lr',type=float,default=3e-5)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--smoke-root',type=Path)
    args=parser.parse_args();output=external_path(args.output)
    if (output/'latest.pt').exists():raise ValueError('No implicit restart; retain the existing pilot.')
    if args.steps<1000 or args.lr<=0:raise ValueError('Positive LR and a substantial pilot required.')
    initial=json.loads((args.source.parent/'status.json').read_text())
    if initial['status']!='complete' or len(initial['packed_forward_checks'])!=3:
        raise ValueError('Three exact full-native GPU packed-forward controls required.')
    source=torch.load(args.source,map_location='cpu',weights_only=True,mmap=True)
    records,root,manifest=inventory()
    stats_path=DATA/'eval/native-decoded-probe/channel-statistics.json'
    sources=('native_weight_quant.py','native_weight_quant_train.py','native_weight_quant_evaluate.py',
             'native_weight_quant_watch.py','native_fp8_training.py','native_basis_training.py','native_decoded.py',
             'native_fusion.py','native_phase_loss.py','native_feature_kd.py')
    contract=dict(source=str(args.source),source_sha256=file_sha(args.source),
        teacher_revision=source['teacher_revision'],teacher_weights_sha256=source['teacher_weights_sha256'],
        source_code_sha256={n:file_sha(REPO/'distill'/n) for n in sources},
        manifest_sha256=file_sha(root/'manifest.json'),statistics_sha256=file_sha(stats_path),
        scheme=source['scheme'],block=source['block'],steps=args.steps,lr=args.lr,warmup=200,
        scheduler='cosine to0.1 after warmup',seed=2026101217,batch=8,phases=[4,4],clip=1.,ema_half_life=500,
        sampler='25%wet mountains/15%oldrare/60%general,8distinctTRAINworlds,3coarses per phase',
        on_policy='50% batches, live detached pupil first-state, same-input resident teacher',
        loss='all5 normalized operator,firstch0x2,.05gradient,.25x0,.25operator-calibrated KD37; no energy-only objective',
        activations='native BF16, no activation quantization',stored_scales='FP32',
        budget='50-100Mo target,150Mo fallback; actual native NF4 artifact149.605Mo',
        accepted=False,accelerated_arithmetic=False)
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text())!=contract:
        raise ValueError('Changed recipe; use a separate run.')
    if not args.smoke:
        if args.smoke_root is None:raise ValueError('Exact backward smoke required.')
        proof=json.loads((args.smoke_root/'gpu-smoke.json').read_text())
        if proof['contract']!=contract or proof['status']!='complete':raise ValueError('Smoke contract changed.')
        if shutil.disk_usage(DATA).free<15*2**30:raise ValueError('Insufficient hot checkpoint/atomic-save space.')
    atomic_json(output/'contract.json',contract)
    torch.set_num_threads(8);torch.manual_seed(contract['seed']);torch.backends.cudnn.benchmark=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    model=load_weight_quant(source,'cuda').requires_grad_(True)
    teacher=EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    from types import SimpleNamespace
    from terrain_inference import prepare_models
    prepare_models(SimpleNamespace(coarse_model=teacher,base_model=teacher,decoder_model=teacher))
    variance=torch.tensor(json.loads(stats_path.read_text())['variance'],device='cuda')
    bank=QATBank()
    if args.smoke:
        ratios=[]
        with torch.no_grad(),model.cached_weights():
            for step in range(16):
                batch,_=bank.batch(records,np.random.default_rng(185310+step),step)
                _,terms,_=step_loss(model,teacher,batch,variance,0.)
                ratios.append(float(terms['weighted_operator']/terms['representation'].clamp_min(1e-8)))
        feature_weight=.25*float(np.median(ratios))
    else:feature_weight=proof['feature_weight'];ratios=proof['calibration_ratios']
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr,betas=(.9,.99),fused=True)
    def training_step(step):
        progress=max(0.,(step-200)/max(1,args.steps-200))
        lr=args.lr*min(1.,(step+1)/200)*(.1+.9*.5*(1+math.cos(math.pi*progress)))
        for group in optimizer.param_groups:group['lr']=lr
        generator=np.random.default_rng(np.random.SeedSequence([contract['seed'],step]))
        with torch.no_grad(),model.cached_weights():
            batch,selected=bank.batch(records,generator,step,model,on_policy=step%2==0)
        optimizer.zero_grad(set_to_none=True)
        loss,terms,blocks=step_loss(model,teacher,batch,variance,feature_weight)
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite weight-only QAT loss.')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if any(p.grad is not None for p in teacher.parameters()):raise ValueError('Teacher received gradients.')
        if not any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters()):raise ValueError('No gradients.')
        optimizer.step()
        return dict(loss=float(loss.detach()),terms={k:float(v.detach()) for k,v in terms.items()},
            grad_norm=float(norm),lr=lr,blocks=len(blocks),on_policy=batch['on_policy'],
            states=len(batch['state_rows']),worlds=[r['metadata']['seed'] for _,r in selected],
            coarses=[r['coarse_variant'] for _,r in selected],peak_bytes=torch.cuda.max_memory_allocated())
    if args.smoke:
        torch.cuda.reset_peak_memory_stats();rows=[training_step(step) for step in range(3)]
        # Deepcopy must retarget the non-module layer owner references.
        clone=copy.deepcopy(model)
        if any(clone.net.get_submodule(n)._weight_quant_owner is not clone for n in clone.quantized_layers):
            raise ValueError('EMA owner references point to the train model.')
        atomic_json(output/'gpu-smoke.json',dict(status='complete',contract=contract,rows=rows,
            feature_weight=feature_weight,calibration_ratios=ratios,teacher_frozen=True,
            ema_owner_valid=True,gpu=torch.cuda.get_device_name(),accepted=False))
        return
    ema=copy.deepcopy(model).eval().requires_grad_(False);step=0;started=time.monotonic();guarded=False
    def save(name,full=False):
        state=dict(config=model.config,architecture='native-weight-quant',scheme=model.scheme,block=model.block,
            quantized_layers=model.quantized_layers,step=step,contract=contract,feature_weight=feature_weight,
            teacher_revision=source['teacher_revision'],teacher_weights_sha256=source['teacher_weights_sha256'],
            evaluation_only=False,accepted=False)
        if full:
            state.update(ema=ema.state_dict(),model=model.state_dict(),optimizer=optimizer.state_dict(),
                         rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        else:
            # The native functional forward casts these exact tensors to BF16
            # before normalization. This preserves evaluated weights while
            # keeping all checkpoints small enough; full FP32 resume stays latest.
            state['ema']={n:v.detach().bfloat16().cpu() for n,v in ema.state_dict().items()}
        atomic_write(output/name,lambda f:torch.save(state,f))
    save('step000000.pt')
    def stop(signum,frame):
        global STOP
        STOP=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    atomic_json(output/'status.json',dict(status='running',step=0,target_steps=args.steps))
    try:
        while step<args.steps and not STOP:
            status=output/'evaluation-status.json'
            if status.exists() and json.loads(status.read_text()).get('hard_stop'):guarded=True;break
            row=training_step(step);step+=1
            with torch.no_grad():
                for a,p in zip(ema.parameters(),model.parameters()):a.lerp_(p,1-math.exp(math.log(.5)/500))
            if step==1 or step%25==0:
                row.update(step=step,elapsed_seconds=time.monotonic()-started)
                append_json(output/'train.jsonl',row);print(json.dumps(row),flush=True)
                atomic_json(output/'status.json',dict(status='running',step=step,target_steps=args.steps))
            if step%500==0 or step==args.steps:save(f'step{step:06d}.pt')
            if step%1000==0 or step==args.steps:save('latest.pt',full=True)
        save('latest.pt',full=True)
        if step%500:save(f'step{step:06d}.pt')
        atomic_json(output/'status.json',dict(status='guarded' if guarded else 'interrupted' if STOP else 'complete',
            step=step,target_steps=args.steps,elapsed_seconds=time.monotonic()-started,accepted=False))
    except Exception as exc:
        save('latest.pt',full=True);atomic_json(output/'status.json',dict(status='failed',step=step,error=str(exc),accepted=False));raise


if __name__=='__main__':main()

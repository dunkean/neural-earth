"""Paired NF4 arms with inherited state; optional capped second-stage objectives."""
import argparse
import json
import math
from pathlib import Path
import shutil
import signal
import time

import numpy as np
import torch

from distill.common import DATA,REPO,atomic_json,atomic_write,append_json,external_path
from distill.native_base import teacher_source
from distill.native_basis_training import inventory
from distill.native_auxiliary_training import (BoundedQATBank,auxiliary_record,auxiliary_terms,
    preconditioners,gradient_norm,combine_gradients)
from distill.native_weight_quant import file_sha
from distill.native_weight_quant_resume import restore_training,restore_rng
from distill.native_weight_quant_train import step_loss
from terrain_diffusion.models.edm_unet import EDMUnet2D


class Engine:
    def __init__(self,source,powers,calibration,arm,cadence=2,zero=False):
        self.source=source;self.arm=arm;self.cadence=cadence;self.zero=zero
        self.model,self.ema,self.optimizer=restore_training(source)
        self.teacher=EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
        from types import SimpleNamespace
        from terrain_inference import prepare_models
        prepare_models(SimpleNamespace(coarse_model=self.teacher,base_model=self.teacher,decoder_model=self.teacher))
        self.records,_,_=inventory();self.bank=BoundedQATBank()
        self.variance=torch.tensor(json.loads((DATA/'eval/native-decoded-probe/channel-statistics.json').read_text())['variance'],device='cuda')
        self.params=list(self.model.parameters());self.powers=powers;self.calibration=calibration
        self.references=source.get('auxiliary_references',dict(norm=calibration['reference_norm'],
            preconditioned=calibration['reference_preconditioned_norm']))
        self.step=int(source['step']);restore_rng(source)
        # Full states restored before any branch-specific forward.
        for name,value in self.model.state_dict().items():
            if not torch.equal(value,source['model'][name].to(value.device)):raise ValueError('Raw restoration differs.')
        for name,value in self.ema.state_dict().items():
            if not torch.equal(value,source['ema'][name].to(value.device)):raise ValueError('EMA restoration differs.')

    def update(self):
        source=self.source;step=self.step;seed=source['contract']['seed']
        generator=np.random.default_rng(np.random.SeedSequence([seed,step]))
        with torch.no_grad(),self.model.cached_weights():
            batch,selected=self.bank.batch(self.records,generator,step,self.model,on_policy=step%2==0)
        self.optimizer.zero_grad(set_to_none=True)
        loss,terms,_=step_loss(self.model,self.teacher,batch,self.variance,source['feature_weight'])
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite main loss.')
        loss.backward();params=self.params
        current=[p.grad for p in params];raw=gradient_norm(current)
        auxiliary=step%self.cadence==1 and self.arm=='b'
        row=dict(step=step+1,loss=float(loss.detach()),operator=float(terms['operator'].detach()),
            first_c0=float(terms['first_channel0_operator'].detach()),regular_grad_norm=raw,
            on_policy=batch['on_policy'],worlds=[r['metadata']['seed'] for _,r in selected],
            coarses=[r['coarse_variant'] for _,r in selected],auxiliary=auxiliary)
        del loss,terms
        pre=preconditioners(self.optimizer,params);scaled=gradient_norm(current,pre)
        decay=math.exp(math.log(.5)/50)
        self.references=dict(norm=decay*self.references['norm']+(1-decay)*raw,
            preconditioned=decay*self.references['preconditioned']+(1-decay)*scaled)
        if auxiliary:
            record,noise_seed=auxiliary_record(self.records,seed,step,self.cadence)
            aux=auxiliary_terms(self.model,self.teacher,record,noise_seed,self.powers)
            row.update(aux_kind=record['case_kind'],aux_coarse=record['coarse_variant'],aux_seed=record['metadata']['seed'],
                       coast_loss=float(aux['coast'].detach()),flat_loss=float(aux['flat_highpass'].detach()))
            if not self.zero:
                baseline=[g.detach().clone() if g is not None else None for g in current]
                self.optimizer.zero_grad(set_to_none=True)
                extra=self.calibration['coefficients']['coast']*aux['coast']+self.calibration['coefficients']['flat']*aux['flat_highpass']
                if not torch.isfinite(extra):raise FloatingPointError('Nonfinite auxiliary.')
                extra.backward()
                row.update(combine_gradients(params,baseline,pre,self.references['norm'],self.references['preconditioned']))
                del baseline,extra
            del aux
        del pre
        if any(p.grad is not None for p in self.teacher.parameters()):raise ValueError('Teacher received gradients.')
        norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
        if not any(p.grad is not None and p.grad.abs().sum()>0 for p in params):raise ValueError('No gradients.')
        self.optimizer.step()
        with torch.no_grad():
            for a,p in zip(self.ema.parameters(),params):a.lerp_(p,1-math.exp(math.log(.5)/500))
        self.step+=1;row.update(grad_norm=float(norm),clipped=float(norm)>1,
            lr=self.optimizer.param_groups[0]['lr'],peak_allocated=torch.cuda.max_memory_allocated(),
            peak_reserved=torch.cuda.max_memory_reserved(),cache_entries=len(self.bank.regions))
        return row

    def save(self,path,experiment,full=False,raw=False):
        s=self.source;state={k:s[k] for k in ('config','architecture','scheme','block','quantized_layers',
            'contract','feature_weight','teacher_revision','teacher_weights_sha256')}
        state.update(step=self.step,arm=self.arm,experiment=experiment,auxiliary_references=self.references,
                     accepted=False,evaluation_only=False)
        if full:state.update(model=self.model.state_dict(),ema=self.ema.state_dict(),optimizer=self.optimizer.state_dict(),
                             rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        else:
            model=self.model if raw else self.ema
            state['ema']={n:v.detach().bfloat16().cpu() for n,v in model.state_dict().items()}
            state['selected_weights']='raw' if raw else 'ema'
        atomic_write(path,lambda f:torch.save(state,f))


def configure():
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--base-source',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--powers',type=Path,required=True)
    parser.add_argument('--calibration',type=Path,required=True)
    parser.add_argument('--smoke',type=Path,required=True)
    parser.add_argument('--arm',choices=('a','b'),required=True)
    parser.add_argument('--end-step',type=int,required=True)
    parser.add_argument('--cadence',type=int,choices=(2,4),default=2)
    args=parser.parse_args();root=external_path(args.root);output=root/args.arm
    configure();powers=json.loads(args.powers.read_text());calibration=json.loads(args.calibration.read_text())
    source=torch.load(args.source,map_location='cpu',weights_only=True,mmap=True)
    base_sha=file_sha(args.base_source)
    if calibration['status']!='complete' or calibration['source_sha256']!=base_sha:
        raise ValueError('Calibration is not from the common source.')
    if calibration['powers_sha256']!=file_sha(args.powers):raise ValueError('TRAIN powers changed.')
    files=('native_auxiliary_training.py','native_auxiliary_train.py','native_field_auxiliary.py')
    experiment=dict(base_source=str(args.base_source),base_sha256=base_sha,powers_sha256=file_sha(args.powers),
        calibration_sha256=file_sha(args.calibration),code_sha256={n:file_sha(REPO/'distill'/n) for n in files},
        start=40000,target=50000,block=2500,cadence=args.cadence,flags='deterministic cuDNN,benchmarkFalse,TF32False',
        cap=.5,reference_half_life=50,cache_limit=128,source_restoration='raw/EMA/Adam/RNG',
        objective='stage2 fused,pupil first detached,newnoise,teacher sameinputs,rootKLmin16,pairedphysicalHP',accepted=False)
    proof=json.loads(args.smoke.read_text())
    if proof['status']!='complete' or proof['base_sha256']!=base_sha or proof['code_sha256']!=experiment['code_sha256']:
        raise ValueError('Exact smoke required for this implementation.')
    if source['step']==40000:
        if file_sha(args.source)!=base_sha:raise ValueError('Initial arm source differs.')
    elif source.get('arm')!=args.arm or source.get('experiment')!=experiment:
        raise ValueError('Changed arm or experiment; no implicit restart.')
    if args.end_step<=source['step'] or args.end_step>50000:raise ValueError('Invalid segment endpoint.')
    if (root/'experiment.json').exists() and json.loads((root/'experiment.json').read_text())!=experiment:
        raise ValueError('Experiment contract changed.')
    if shutil.disk_usage(DATA).free<16*2**30:raise ValueError('Insufficient atomic checkpoint reserve.')
    atomic_json(root/'experiment.json',experiment)
    engine=Engine(source,powers,calibration,args.arm,args.cadence);started=time.monotonic();stop=False
    def terminate(signum,frame):
        nonlocal stop
        stop=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,terminate)
    atomic_json(output/'segment-start.json',dict(step=engine.step,source=str(args.source),end_step=args.end_step,arm=args.arm))
    atomic_json(output/'status.json',dict(status='running',step=engine.step,end_step=args.end_step,target_steps=50000))
    try:
        while engine.step<args.end_step and not stop:
            row=engine.update()
            if engine.step%100==0 or engine.step==40001:
                row['elapsed_seconds']=time.monotonic()-started
                append_json(output/'train.jsonl',row);print(json.dumps(row),flush=True)
                atomic_json(output/'status.json',dict(status='running',step=engine.step,end_step=args.end_step,target_steps=50000))
            if engine.step%1000==0:engine.save(output/'latest.pt',experiment,full=True)
        engine.save(output/'latest.pt',experiment,full=True)
        engine.save(output/f'step{engine.step:06d}.pt',experiment)
        engine.save(output/'raw'/f'step{engine.step:06d}.pt',experiment,raw=True)
        atomic_json(output/'status.json',dict(status='interrupted' if stop else 'segment_complete',step=engine.step,
            end_step=args.end_step,target_steps=50000,elapsed_seconds=time.monotonic()-started,accepted=False))
    except Exception as exc:
        engine.save(output/'latest.pt',experiment,full=True)
        atomic_json(output/'status.json',dict(status='failed',step=engine.step,error=str(exc)));raise


if __name__=='__main__':main()

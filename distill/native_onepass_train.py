"""Pilot distillation of a native single evaluation toward fused teacherT2.

Runs bounded segments under a shared GPU lock; resumes FP32 masters/Adam/EMA/RNG.
The physical objectives compare matched positions, not just spectral energy.
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
from torch.utils.checkpoint import checkpoint

from distill.common import DATA, REPO, atomic_json, atomic_write, append_json, external_path
from distill.native_auxiliary_training import root_coast
from distill.native_base import teacher_source
from distill.native_basis_training import inventory
from distill.native_decoded import DecodedRegion, physical_terms
from distill.native_fusion import overlap_loss, window_outputs
from distill.native_onepass import ARCHITECTURE, load_onepass, noise_windows
from distill.native_progressive_prepare import configure
from distill.native_progressive_recovery_loss import multiscale_height_terms
from distill.native_progressive_recovery_train import RecoveryDecoder
from distill.native_weight_quant import file_sha
from terrain_diffusion.models.edm_unet import EDMUnet2D


CODE = ('native_onepass.py', 'native_onepass_train.py', 'native_onepass_runtime.py',
        'native_progressive_recovery_loss.py', 'native_fusion.py', 'native_decoded.py')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preparation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--until', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--smoke-root', type=Path)
    args = parser.parse_args()
    output = external_path(args.output)
    configure()
    proof = json.loads((args.preparation/'proof.json').read_text())
    initial_path = args.preparation/'initial.pt'
    if proof['status'] != 'complete' or proof['initial_sha256'] != file_sha(initial_path):
        raise ValueError('Verified one-pass preparation required.')
    if any(file_sha(REPO/'distill'/name) != digest for name,digest in proof['code_sha256'].items()):
        raise ValueError('One-pass implementation differs from preparation.')
    source = torch.load(initial_path, map_location='cpu', weights_only=True, mmap=True)
    contract = dict(source_sha256=proof['initial_sha256'], steps=args.steps, seed=2026101453,
        solver_evaluations=1, teacher_solver_evaluations=2, noise_offsets=[5819,5820],
        lr='warmup200 to3e-5;cosine to3e-6', ema_half_life=500,
        loss='normalized window/core latent+matched gradients+normalized physical LF/HF/full+coast+teacher-relative overlap+decoded shape',
        sampler='25%wetmountains;20%coast/lowsea;15%plains;40%general;original/coarse4/coarse2',
        teacher_revision=source['teacher_revision'], fresh_noise=True,
        code_sha256={name:file_sha(REPO/'distill'/name) for name in CODE}, accepted=False,
        target_bytes=100000000, target_speedup=3.)
    if args.steps < 1000 or args.until is not None and not 0 < args.until <= args.steps:
        raise ValueError('Substantial total budget and a valid bounded segment are required.')
    if (output/'contract.json').exists() and json.loads((output/'contract.json').read_text()) != contract:
        raise ValueError('Changed recipe; preserve the previous run.')
    if not args.smoke:
        if args.smoke_root is None:
            raise ValueError('A successful matching smoke is required.')
        smoke = json.loads((args.smoke_root/'gpu-smoke.json').read_text())
        if smoke['status'] != 'complete' or smoke['contract'] != contract:
            raise ValueError('Smoke contract differs.')
    atomic_json(output/'contract.json', contract)
    torch.manual_seed(contract['seed']); torch.cuda.manual_seed_all(contract['seed'])
    model = load_onepass(source, 'cuda', masters=True)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=0., betas=(.9,.99), fused=True)
    latest = output/'latest.pt'; step = 0
    if latest.exists():
        saved = torch.load(latest, map_location='cpu', weights_only=True, mmap=True)
        if saved['contract'] != contract:
            raise ValueError('Resume contract differs.')
        model.load_state_dict(saved['model']); ema.load_state_dict(saved['ema'])
        optimizer.load_state_dict(saved['optimizer']); step = saved['step']
        torch.set_rng_state(saved['rng']); torch.cuda.set_rng_state_all(saved['cuda_rng'])
    teacher = EDMUnet2D.from_pretrained(str(teacher_source())).cuda().bfloat16().eval().requires_grad_(False)
    from types import SimpleNamespace
    from terrain_inference import prepare_models
    prepare_models(SimpleNamespace(coarse_model=teacher,base_model=teacher,decoder_model=teacher))
    decoder = RecoveryDecoder('cuda')
    records, _, _ = inventory()
    variance = torch.tensor(json.loads((DATA/'eval/native-decoded-probe/channel-statistics.json').read_text())['variance'],device='cuda')
    from terrain_diffusion.inference.world_pipeline import linear_weight_window
    weight = linear_weight_window(64,'cuda',torch.float32)
    stopped = False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGINT,signal.SIGTERM):
        signal.signal(sig,stop)
    started = time.monotonic()
    def save(path,full=False,raw=False):
        saved={key:value for key,value in source.items() if key not in ('model','ema')}
        net=model if raw else ema
        saved.update(step=step,contract=contract,selected_weights='raw' if raw else 'ema',accepted=False,
                     ema={name:value.detach().cpu().to(torch.float32 if name=='net.out_gain' else torch.bfloat16)
                          for name,value in net.state_dict().items()})
        if full:
            saved.update(model=model.state_dict(),ema=ema.state_dict(),optimizer=optimizer.state_dict(),
                         rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all())
        atomic_write(path,lambda handle:torch.save(saved,handle))
    endpoint=args.until or args.steps
    initial_first=next(iter(model.net.enc.values())).weight.detach().clone() if args.smoke else None
    try:
        while step<endpoint and not stopped:
            generator=np.random.default_rng(np.random.SeedSequence([contract['seed'],step]))
            coarse=('coarse4','coarse2','original')[(step//5)%3]
            draw=float(generator.random())
            kind=('wet-high-mountain' if draw<.25 else
                  ('wet-complex-coast','low-plain-sea')[step%2] if draw<.45 else
                  ('desert-plain','temperate-plain')[step%2] if draw<.60 else None)
            # Smoke must exercise coasts, both noises, and decoded backward.
            if args.smoke:
                kind=('wet-complex-coast','desert-plain','wet-high-mountain','low-plain-sea')[step%4]
            pool=[r for r in records if r['case_kind']==kind and r['coarse_variant']==coarse]
            record=pool[int(generator.integers(len(pool)))]
            region=DecodedRegion(Path(record['root'])/record['file'],'cuda')
            noise_seed=int(generator.integers(1,2**31-5821))
            region.refresh_noise(teacher,weight,noise_seed)
            noises=noise_windows(noise_seed,region.ref_origins,'cuda')
            conditions=region.stages[1]['condition']
            values=[]
            for start in range(0,len(noises),4):
                inputs=noises[start:start+4].detach().requires_grad_(True)
                values.append(checkpoint(model.windows,inputs,conditions[start:start+4],use_reentrant=True))
            prediction=torch.cat(values)
            core=region.core_crop(prediction,weight)
            target=window_outputs(region.stages[1]['inputs'],region.stages[1]['target'],region.stages[1]['labels'],region.dtype).detach()
            scale=variance[None,:,None,None]
            residual=(prediction-target)/scale.sqrt()
            latent=residual.square().mean()
            core_loss=((core-region.target.detach()).square()/variance[:,None,None]).mean()
            gradient=sum(residual.diff(dim=axis).square().mean() for axis in (-1,-2))*.5
            overlap=overlap_loss(prediction/scale.sqrt(),target/scale.sqrt(),region.ref_origins)
            physical=multiscale_height_terms(core[None],region.target.detach()[None])
            coast,disagreement,_=root_coast(core[None],region.target.detach()[None])
            loss=latent+core_loss+.05*gradient+.05*overlap+.01*(physical['low']+physical['full'])+.003*physical['high']+.02*coast
            metrics=dict(latent=float(latent.detach()),core=float(core_loss.detach()),gradient=float(gradient.detach()),
                         overlap=float(overlap.detach()),height_high=float(physical['high'].detach()),
                         coast=float(coast.detach()),coast_disagreement=float(disagreement.detach()))
            if step%4==3:
                decoded=physical_terms(dict(core=core),region,decoder,variance)
                loss=loss+.02*(decoded['height_gradient']+decoded['height_tpi'])
                metrics.update(decoded_slope=float(decoded['height_gradient'].detach()),decoded_tpi=float(decoded['height_tpi'].detach()))
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite one-pass objective.')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True)
            if any(p.grad is not None for p in teacher.parameters()) or any(p.grad is not None for p in decoder.parameters()):
                raise ValueError('Frozen reference received gradients.')
            lr=min(1.,(step+1)/200)*(3e-6+27e-6*.5*(1+math.cos(math.pi*max(0.,(step-200)/(args.steps-200)))))
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step()
            with torch.no_grad():
                for a,p in zip(ema.parameters(),model.parameters()):a.lerp_(p,1-math.exp(math.log(.5)/500))
            step+=1
            row=dict(step=step,loss=float(loss.detach()),grad_norm=float(norm),lr=lr,kind=kind,coarse=coarse,
                     world_seed=record['metadata']['seed'],noise_seed=noise_seed,elapsed_seconds=time.monotonic()-started,
                     peak_bytes=torch.cuda.max_memory_allocated(),**metrics)
            append_json(output/'train.jsonl',row)
            if step==1 or step%25==0 or args.smoke:
                print(json.dumps(row),flush=True)
                atomic_json(output/'status.json',dict(status='running',step=step,target=args.steps,segment_until=endpoint))
            del values,prediction,core,target,residual,physical,coast,loss,region
            if step%4==0:del decoded
            if not args.smoke and step%250==0:
                save(latest,full=True);save(output/f'step{step:06d}.pt');save(output/'raw'/f'step{step:06d}.pt',raw=True)
            if args.smoke and step>=4:break
        save(latest,full=True);save(output/f'step{step:06d}.pt');save(output/'raw'/f'step{step:06d}.pt',raw=True)
        if args.smoke:
            changed=next(iter(model.net.enc.values())).weight.detach()-initial_first
            if step!=4 or stopped or float(changed[:,5:10].abs().max())==0:
                raise ValueError('Both-noise optimizer update smoke incomplete.')
            atomic_json(output/'gpu-smoke.json',dict(status='complete',contract=contract,steps=step,
                        second_noise_weight_update=float(changed[:,5:10].abs().max()),
                        peak_bytes=torch.cuda.max_memory_allocated(),gpu=torch.cuda.get_device_name()))
        atomic_json(output/'status.json',dict(status='smoke_complete' if args.smoke else 'interrupted' if stopped else
                    'complete' if step==args.steps else 'segment_complete',step=step,target=args.steps))
    except Exception as exc:
        atomic_json(output/'status.json',dict(status='failed',step=step,error=str(exc)))
        raise


if __name__=='__main__':
    main()

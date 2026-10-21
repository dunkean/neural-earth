"""Refine actual few-step flow samples against original decoded terrain patches.

Unlike paired regression, independent samples are judged distributionally.
The frozen decoder and training-only critics add no product weight or inference
passes. Sampled full-tile VAL morphology is still required before acceptance.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import random
import signal
import time

import torch
from torch import nn
from torch.nn import functional as F

from distill.common import atomic_json, atomic_write, append_json, external_path
from distill.native_weight_quant import file_sha
from distill.student import load_student
from distill.tile40_adversarial_pilot import TerrainCritic, critic_hinge, gradient_weight
from distill.tile40_conditioning import RELIEFS
from distill.tile40_decoder_pilot import References
from distill.tile40_encoder_pilot import decode, highpass_delta, fine_relief_terms
from distill.tile40_flow_network import ARCHITECTURE
from distill.tile40_flow_pilot import colored_prior, condition, midpoint,coarse_altitude_guide
from distill.tile40_flow_train import make_batch, loss_for


class MultiScaleCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.critics = nn.ModuleList(TerrainCritic() for _ in range(2))

    def forward(self, height, conditioning):
        # The existing terrain critic pools by4. A2x preceding resize provides
        # complementary60m and120m terrain fields, with separate learned weights.
        values = []
        for scale, critic in zip((1.,2.), self.critics):
            h, c = height, conditioning
            if scale != 1.:
                h = F.interpolate(h, scale_factor=scale, mode='bilinear',align_corners=False)
                c = F.interpolate(c, scale_factor=scale, mode='nearest')
            values.append(critic(h,c))
        return values


def sampled_patch(model,decoder,row,mean,std,rng,device,intervals=2,residual_altitude=False):
    fields = condition(row,-64,-64,320)[None].to(device)
    start = colored_prior(torch.randn(1,5,320,320,device=device))
    latent = midpoint(model,start,fields,steps=intervals)*std+mean
    if residual_altitude:
        latent=torch.cat((latent[:,:4],latent[:,4:5]+coarse_altitude_guide(fields)),dim=1)
    y,x = rng.choice((0,48,96)),rng.choice((0,48,96))
    target_inputs, target_latents, guided_root = References.window(row,y*8,x*8,device)
    with torch.no_grad():
        reference_root = decode(decoder,target_latents,target_inputs[:,:1])
    predicted_root = decode(decoder,latent[...,y+64:y+128,x+64:x+128],target_inputs[:,:1])
    root = guided_root+highpass_delta(predicted_root-reference_root)
    root, target = root[...,64:-64,64:-64],guided_root[...,64:-64,64:-64]
    conditioning = F.interpolate(fields[:,10:16,y+64:y+128,x+64:x+128],size=(512,512),mode='nearest')[...,128:384,128:384]
    return root.sign()*root.square(),target.sign()*target.square(),conditioning


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--initial',type=Path,required=True)
    parser.add_argument('--decoder',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--interval',type=int,default=500)
    parser.add_argument('--flow-steps',type=int,default=2)
    parser.add_argument('--fm-weight',type=float,default=.1)
    parser.add_argument('--gradient-ratio',type=float,default=2.)
    parser.add_argument('--climate-negatives',action='store_true',help='Reject real terrain paired with another climate from the same TRAIN geometry.')
    parser.add_argument('--power-weight',type=float,default=0.,help='Preserve decoded band powers without matching independent samples pixel by pixel.')
    parser.add_argument('--warm-critic',type=Path,help='Explicit matching training-only critic initialization.')
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args()
    output=external_path(args.output)
    if (output.exists() and not args.resume) or min(args.steps,args.interval,args.flow_steps)<1 or args.fm_weight<=0 or args.gradient_ratio<=0 or args.power_weight<0:
        raise ValueError('Fresh bounded experiment or explicit resume required.')
    torch.set_num_threads(8)
    torch.manual_seed(8219)
    rng,device=random.Random(8219),torch.device('cuda')
    refs=References(args.dataset)
    model,saved=load_student(args.initial,device,ema=False)
    if saved.get('flow_architecture')!=ARCHITECTURE or refs.manifest_sha!=saved['contract']['dataset_manifest_sha256']:
        raise ValueError('Matching immutable flow/TRAIN contract required.')
    model.requires_grad_(True)
    decoder,_=load_student(args.decoder,device)
    decoder.requires_grad_(False)
    mean=torch.tensor(saved['contract']['latent_mean'],device=device)[None,:,None,None]
    std=torch.tensor(saved['contract']['latent_std'],device=device)[None,:,None,None]
    critic=MultiScaleCritic().to(device)
    if args.warm_critic:
        prior=torch.load(args.warm_critic,map_location='cpu',weights_only=False)
        if prior['contract']['dataset_manifest_sha256']!=refs.manifest_sha or prior['contract']['sample_adversarial']['decoder_sha256']!=file_sha(args.decoder):
            raise ValueError('Warm critic requires the same TRAIN bank and decoded height interface.')
        critic.load_state_dict(prior['critic'])
        del prior
    optimizer=torch.optim.AdamW(model.parameters(),lr=.00002,weight_decay=.001)
    d_optimizer=torch.optim.Adam(critic.parameters(),lr=.0001,betas=(0.,.99))
    ema={k:v.detach().clone() for k,v in model.state_dict().items()}
    contract=saved['contract']|dict(sample_adversarial=dict(initial_sha256=file_sha(args.initial),
        decoder_sha256=file_sha(args.decoder),intervals=args.flow_steps,fm_weight=args.fm_weight,
        gradient_ratio=args.gradient_ratio,loss_version=1,
        target='Original guided decoded root;independent-noise samples;no paired height regression.',
        calibration='Per relief;all generator gradients;every100steps;TRAIN only.',
        selection='Full sampled VAL terrain;critic/FM losses do not select quality.'))
    if args.climate_negatives:
        contract['sample_adversarial'].update(climate_negatives=True,loss_version=2,
            negative_scope='Other climate,same TRAIN world seed and sketch;not VAL or TEST.')
    if args.power_weight or args.warm_critic:
        contract['sample_adversarial'].update(power_weight=args.power_weight,
            warm_critic_sha256=file_sha(args.warm_critic) if args.warm_critic else None,loss_version=3)
    step,stopped,calibration=0,False,{}
    if args.resume:
        previous=torch.load(args.resume,map_location='cpu',weights_only=False)
        if previous['contract']!=contract:raise ValueError('Adversarial resume contract differs.')
        model.load_state_dict(previous['model'])
        ema={k:v.to(device) for k,v in previous['ema'].items()}
        critic.load_state_dict(previous['critic'])
        optimizer.load_state_dict(previous['optimizer'])
        d_optimizer.load_state_dict(previous['critic_optimizer'])
        rng.setstate(previous['rng_state'])
        torch.set_rng_state(previous['torch_rng_state'])
        torch.cuda.set_rng_state(previous['cuda_rng_state'])
        step,calibration=previous['step'],previous['calibration']
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'contract.json',contract|dict(target_steps=args.steps,producer_sha256=file_sha(Path(__file__))))
    def stop(signum,frame):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    def save(name):
        atomic_write(output/name,lambda f:torch.save(dict(flow_architecture=ARCHITECTURE,
            config=asdict(model.config),contract=contract,step=step,
            model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            ema={k:v.detach().cpu() for k,v in ema.items()},optimizer=optimizer.state_dict(),
            critic=critic.state_dict(),critic_optimizer=d_optimizer.state_dict(),calibration=calibration,
            rng_state=rng.getstate(),torch_rng_state=torch.get_rng_state(),cuda_rng_state=torch.cuda.get_rng_state()),f))
    by_relief={kind:[r for r in refs.splits['train'] if r['case']['relief']==kind] for kind in RELIEFS}
    by_seed={}
    for row in refs.splits['train']:
        by_seed.setdefault(row['case']['seed'],[]).append(row)
    parameters=tuple(model.parameters())
    started=time.monotonic()
    while step<args.steps and not stopped:
        kind=rng.choices(RELIEFS,weights=(.6,.25,.15))[0]
        row=rng.choice(by_relief[kind])
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction,target,fields=sampled_patch(model,decoder,row,mean,std,rng,device,args.flow_steps,saved['contract'].get('residual_altitude',False))
        # Fixed per-class scales keep plain gradients useful. They are not fit
        # to VAL samples and do not rescale the product output.
        scale={'mountains':1.,'hills':3.,'plain':10.}[kind]
        critic.train().requires_grad_(True)
        d_optimizer.zero_grad(set_to_none=True)
        real=critic(target*scale,fields)
        fake=critic(prediction.detach()*scale,fields)
        d_loss=sum(critic_hinge(a,b,paired=True) for a,b in zip(real,fake))/len(real)
        if args.climate_negatives:
            other=rng.choice([r for r in by_seed[row['case']['seed']] if r['case']['climate']!=row['case']['climate']])
            # Global locations agree because every climate in a geometry shares
            # the sketch and world seed. Use the central training patch here.
            _,_,wrong_root=References.window(other,384,384,device)
            wrong_root=wrong_root[...,64:-64,64:-64]
            _,_,correct_root=References.window(row,384,384,device)
            correct_root=correct_root[...,64:-64,64:-64]
            central_condition=condition(row,64,64,32)[None].to(device)
            central_condition=F.interpolate(central_condition[:,10:16],size=(256,256),mode='nearest')
            correct_scores=critic(correct_root.sign()*correct_root.square()*scale,central_condition)
            wrong_scores=critic(wrong_root.sign()*wrong_root.square()*scale,central_condition)
            d_loss=d_loss+.25*sum(critic_hinge(a,b,paired=True) for a,b in zip(correct_scores,wrong_scores))/len(real)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(),2.)
        d_optimizer.step()
        critic.eval().requires_grad_(False)
        adversarial=-sum(v.mean() for v in critic(prediction*scale,fields))/2
        inputs,velocity_target=make_batch([row],rng,mean,std,device,residual_altitude=saved['contract'].get('residual_altitude',False))
        fm,_=loss_for(model,inputs,velocity_target)
        regularizer=args.fm_weight*fm
        power=prediction.new_zeros(())
        if args.power_weight:
            roots=[h.sign()*torch.sqrt(h.abs().clamp_min(1e-8)) for h in (prediction,target)]
            power,_=fine_relief_terms(*roots)
            regularizer=regularizer+args.power_weight*power
        if kind not in calibration or step-calibration[kind]['step']>=100:
            weight,values=gradient_weight(regularizer,adversarial,parameters,args.gradient_ratio,cap=100.)
            calibration[kind]=dict(step=step,weight=weight,**values)
        loss=regularizer+calibration[kind]['weight']*adversarial
        if not torch.isfinite(loss) or not torch.isfinite(d_loss):raise ValueError('Nonfinite sample-adversarial loss.')
        loss.backward()
        gradient=torch.nn.utils.clip_grad_norm_(parameters,2.)
        if not torch.isfinite(gradient):raise ValueError('Nonfinite sample-adversarial gradient.')
        optimizer.step()
        step+=1
        with torch.no_grad():
            for name,value in model.state_dict().items():ema[name].lerp_(value,.005)
        if step%10==0 or step==args.steps:
            record=dict(step=step,relief=kind,loss=float(loss.detach()),fm=float(fm.detach()),critic=float(d_loss.detach()),
                adversarial=float(adversarial.detach()),power=float(power.detach()),calibration=calibration,elapsed_seconds=time.monotonic()-started)
            append_json(output/'training.jsonl',record)
            atomic_json(output/'status.json',dict(status='running',target_steps=args.steps,**record))
        if step%args.interval==0 or step==args.steps:save(f'step-{step:06d}.pt')
    save('latest.pt')
    atomic_json(output/'status.json',dict(status='interrupted' if stopped else 'segment_complete',step=step,
        target_steps=args.steps,elapsed_seconds=time.monotonic()-started))


if __name__=='__main__':main()

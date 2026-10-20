"""Conditioned latent critic restores joint terrain features beyond averaged regression.

Keep the fast one-pass encoder interface and enforce decoded-height constraints.
Critic weights are training-only;neither critic loss nor latent realism is a
substitute for complete40km morphology,climate,CPU/GPU andTEST acceptance.
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
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from distill.common import DATA, REPO, append_json, atomic_json, atomic_write, external_path
from distill.native_weight_quant import file_sha
from distill.student import load_student
from distill.tile40_conditioning import RELIEFS
from distill.tile40_decoder_pilot import References
from distill.tile40_encoder_pilot import terms, validate, ridge_structure_terms


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(spectral_norm(nn.Conv2d(11, 32, 3, stride=2, padding=1)), nn.LeakyReLU(.2),
            spectral_norm(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.LeakyReLU(.2),
            spectral_norm(nn.Conv2d(64, 64, 3, stride=2, padding=1)), nn.LeakyReLU(.2),
            spectral_norm(nn.Conv2d(64, 1, 3, padding=1)))

    def forward(self, latent, conditioning, mean, std):
        return self.net(torch.cat(((latent-mean)/std, conditioning), dim=1))


class TerrainCritic(nn.Module):
    """Judge ridge/valley organization in decoded metres with climate context."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(spectral_norm(nn.Conv2d(9, 32, 3, stride=2, padding=1)), nn.LeakyReLU(.2),
            spectral_norm(nn.Conv2d(32, 64, 3, stride=2, padding=1)), nn.LeakyReLU(.2),
            spectral_norm(nn.Conv2d(64, 64, 3, stride=2, padding=1)), nn.LeakyReLU(.2),
            spectral_norm(nn.Conv2d(64, 1, 3, padding=1)))

    def forward(self, height, conditioning, mean=None, std=None):
        field = F.avg_pool2d(height, 4)
        smooth = F.avg_pool2d(F.pad(field, (6, 6, 6, 6), mode='reflect'), 13, stride=1)
        dy = F.pad(torch.diff(field, dim=-2), (0, 0, 0, 1))
        dx = F.pad(torch.diff(field, dim=-1), (0, 1, 0, 0))
        # Fixed metre scales;never normalize away a plain's low amplitude.
        return self.net(torch.cat(((field-smooth)/50., dy/20., dx/20.,
                                  F.avg_pool2d(conditioning, 4)), dim=1))


def critic_hinge(real, fake, paired=False):
    # Matched conditions cancel climate/altitude offsets in the paired score.
    return F.relu(1-real+fake).mean() if paired else .5*(F.relu(1-real).mean()+F.relu(1+fake).mean())


def gradient_weight(physical, adversarial, parameters, ratio, cap=20.):
    def magnitude(loss):
        values=torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        sums=[g.float().square().sum() for g in values if g is not None]
        return float(torch.sqrt(torch.stack(sums).sum())) if sums else 0.
    p, a=magnitude(physical), magnitude(adversarial)
    return (min(cap,ratio*p/a) if a > 1e-12 else 0.), dict(physical_gradient=p,critic_gradient=a)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial', type=Path, default=DATA/'tile40-20261014/encoder64-pilot-v1/step-020000.pt')
    parser.add_argument('--decoder', type=Path, default=DATA/'tile40-20261014/decoder16-original-pilot-v1/step-001000.pt')
    parser.add_argument('--dataset', type=Path, default=DATA/'tile40-20261014/reference-coarse2-guided-v4')
    parser.add_argument('--steps', type=int, default=2000)
    parser.add_argument('--interval', type=int, default=500)
    parser.add_argument('--adversarial-weight', type=float, default=.1)
    parser.add_argument('--critic-space', choices=('latent', 'terrain'), default='latent')
    parser.add_argument('--paired-weight', type=float, default=1.)
    parser.add_argument('--relativistic', action='store_true', help='Discriminate matched real/fake patches with the same climate context.')
    parser.add_argument('--gradient-ratio', type=float, default=0., help='TRAIN-only adversarial/physical gradient ratio,calibrated per relief every100steps.')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--structure-weight',type=float,default=0.,help='Decoded local ridge-orientation loss;training only.')
    args = parser.parse_args()
    output = external_path(args.output)
    if output.exists() or args.steps < 1 or args.interval < 1 or args.adversarial_weight < 0 or not 0 <= args.paired_weight <= 1 or not 0 <= args.gradient_ratio <= 1 or args.structure_weight < 0:
        raise ValueError('A fresh bounded measured pilot is required.')
    torch.set_num_threads(8)
    torch.manual_seed(681903)
    rng, device = random.Random(681903), torch.device(args.device)
    refs = References(args.dataset)
    plan = json.loads((args.dataset/'manifest.json').read_text())
    if plan['internal_coarse_checkpoint']['sha256'] != file_sha(REPO/'distill/weights/coarse2.pt'):
        raise ValueError('Coarse condition must be available at deployment.')
    generator, _ = load_student(args.initial, device)
    decoder, _ = load_student(args.decoder, device)
    decoder.requires_grad_(False)
    critic = (TerrainCritic() if args.critic_space == 'terrain' else Critic()).to(device)
    means = torch.stack([r['latents'].mean(dim=(-2, -1)) for r in refs.splits['train']]).mean(0)
    variance = torch.stack([r['latents'].var(dim=(-2, -1), unbiased=False) for r in refs.splits['train']]).mean(0).clamp_min(.01)
    mean, std = means[None, :, None, None].to(device), variance.sqrt()[None, :, None, None].to(device)
    variance = variance.to(device)
    optimizer = torch.optim.AdamW(generator.parameters(), lr=.00005, weight_decay=.001)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=.0001, betas=(0., .99))
    ema = {k:v.detach().clone() for k,v in generator.state_dict().items()}
    contract = dict(dataset_manifest_sha256=refs.manifest_sha, initial_sha256=file_sha(args.initial),
        decoder_sha256=file_sha(args.decoder), adversarial_weight=args.adversarial_weight,
        physical_fine_weight=1., auxiliary_latent_weight=.01, critic='Spectral-normalized conditioned latent PatchGAN;training only.',
        scope='Fast one-pass candidate;complete40km acceptance remains unproven.')
    if args.critic_space != 'latent' or args.paired_weight != 1.:
        contract.update(critic=args.critic_space, paired_weight=args.paired_weight,
            physical_critic_scales_m=[50., 20., 20.], loss_version=2)
    if args.relativistic or args.gradient_ratio:
        contract.update(relativistic=args.relativistic,gradient_ratio=args.gradient_ratio,
            gradient_calibration='All generator parameters,TRAIN only,per relief every100steps,weight cap20.',loss_version=3)
    if args.device != 'cuda':
        contract['training_device']=args.device
    contract.update(train_positions=[0,48,96],validation_positions='legacy peripheral plus separate center',loss_version=5)
    if args.structure_weight:
        contract.update(structure_weight=args.structure_weight,structure_loss='960m highpass,local11px gradient tensor,metre floors,no inference cost.',loss_version=6)
    output.mkdir(parents=True)
    atomic_json(output/'contract.json', contract | dict(producer_sha256=file_sha(Path(__file__))))
    step, best, stopped = 0, float('inf'), False
    def stop(signum, frame):
        nonlocal stopped
        stopped = True
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, stop)
    def save(name):
        checkpoint = dict(config=asdict(generator.config), step=step, best=best, contract=contract,
            model={k:v.detach().cpu() for k,v in generator.state_dict().items()},
            ema={k:v.detach().cpu() for k,v in ema.items()}, optimizer=optimizer.state_dict(),
            critic=critic.state_dict(), critic_optimizer=critic_optimizer.state_dict(),
            critic_mean=mean.cpu(), critic_std=std.cpu(), rng_state=rng.getstate(), torch_rng_state=torch.get_rng_state())
        if getattr(generator,'conditioning_architecture',None):
            checkpoint['conditioning_architecture']=generator.conditioning_architecture
        atomic_write(output/name, lambda handle:torch.save(checkpoint, handle))
    def evaluate():
        nonlocal best
        raw = validate(generator, decoder, decoder, refs, variance, device, 1., .01)
        center_raw = validate(generator, decoder, decoder, refs, variance, device, 1., .01, center=True)
        state = {k:v.detach().clone() for k,v in generator.state_dict().items()}
        generator.load_state_dict(ema)
        average = validate(generator, decoder, decoder, refs, variance, device, 1., .01)
        center_ema = validate(generator, decoder, decoder, refs, variance, device, 1., .01, center=True)
        generator.load_state_dict(state)
        atomic_json(output/f'validation-{step:06d}.json', dict(step=step, raw=raw, ema=average, center_raw=center_raw, center_ema=center_ema))
        raw_score, ema_score = (raw['score']+center_raw['score'])/2, (average['score']+center_ema['score'])/2
        score = min(raw_score, ema_score)
        if score < best:
            best = score
            save('best.pt')
            atomic_json(output/'best.json', dict(step=step, score=score,
                selection='raw' if raw_score <= ema_score else 'ema'))
        print(json.dumps(dict(step=step, raw=raw['groups'], ema=average['groups'])), flush=True)
    started = time.monotonic()
    evaluate()
    save('step-000000.pt')
    by_relief = {relief:[r for r in refs.splits['train'] if r['case']['relief']==relief] for relief in RELIEFS}
    calibrated={}
    generator_parameters=tuple(generator.parameters())
    while step < args.steps and not stopped:
        row = rng.choice(by_relief[rng.choices(RELIEFS, weights=(.6, .25, .15))[0]])
        generator.train()
        optimizer.zero_grad(set_to_none=True)
        capture = {}
        physical, metrics = terms(generator, decoder, decoder, row, rng.choice((0, 48, 96)), rng.choice((0, 48, 96)),
                                  variance, device, 1., .01, capture, args.paired_weight)
        if args.critic_space == 'terrain':
            p, target, condition = capture['height_prediction'], capture['height_target'], capture['terrain_conditioning']
        else:
            p, target, condition = capture['prediction'], capture['target'], capture['conditioning']
        structure=None
        if args.structure_weight:
            heights=capture['height_prediction'],capture['height_target']
            roots=[h.sign()*torch.sqrt(h.abs().clamp_min(1e-8)) for h in heights]
            structure,gap=ridge_structure_terms(*roots)
            physical=physical+args.structure_weight*structure
            metrics.update(ridge_structure_loss=structure,ridge_coherence_gap=gap)
        critic.train().requires_grad_(True)
        critic_optimizer.zero_grad(set_to_none=True)
        d_real = critic(target, condition, mean, std)
        d_fake = critic(p.detach(), condition, mean, std)
        d_loss = critic_hinge(d_real,d_fake,args.relativistic)
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 2.)
        critic_optimizer.step()
        critic.eval().requires_grad_(False)
        adversarial = -critic(p, condition, mean, std).mean()
        weight=args.adversarial_weight
        relief=row['case']['relief']
        if args.gradient_ratio:
            if relief not in calibrated or step-calibrated[relief]['step'] >= 100:
                weight,values=gradient_weight(physical,adversarial,generator_parameters,args.gradient_ratio)
                calibrated[relief]=dict(step=step,weight=weight,**values)
            weight=calibrated[relief]['weight']
        loss = physical+weight*adversarial
        if not torch.isfinite(loss) or not torch.isfinite(d_loss):
            raise ValueError(f'Nonfinite adversarial pilot at step{step}.')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(generator.parameters(), 2.)
        optimizer.step()
        step += 1
        with torch.no_grad():
            for key,value in generator.state_dict().items():
                ema[key].lerp_(value, .005)
        if step%25 == 0:
            record = dict(step=step, loss=float(loss), physical=float(physical), critic=float(d_loss),
                adversarial=float(adversarial),adversarial_weight=weight,calibration=calibrated,
                **{k:float(v) for k,v in metrics.items()}, elapsed_seconds=time.monotonic()-started)
            append_json(output/'training.jsonl', record)
            atomic_json(output/'status.json', dict(status='running', target_steps=args.steps, **record))
        if step%args.interval == 0 or step == args.steps:
            evaluate()
            save(f'step-{step:06d}.pt')
    save('latest.pt')
    atomic_json(output/'status.json', dict(status='interrupted' if stopped else 'segment_complete', step=step,
        target_steps=args.steps, best=best, elapsed_seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
